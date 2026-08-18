"""Preprocessing: train-only fitting, Inf handling, column rejection, shards.

The point under test is not that scaling works but that nothing from val or
test can influence a fitted statistic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing import apply_preprocessor, fit_preprocessor  # noqa: E402

CFG = {
    "max_nan_ratio": 0.5,
    "drop_zero_variance": True,
    "impute_strategy": "median",
    "scaler": "standard",
    "shard_target_mb": 1,
    "max_shards_per_split": 64,
    "assert_finite_after_transform": True,
}


def test_high_missing_column_is_rejected():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [4.0, 1.0]], dtype=np.float32)
    fitted = fit_preprocessor(X, ["good", "mostly_missing"], CFG)
    assert fitted["kept_names"] == ["good"]
    assert fitted["rejected"]["high_missing"][0]["feature"] == "mostly_missing"


def test_zero_variance_column_is_rejected():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    fitted = fit_preprocessor(X, ["varies", "constant"], CFG)
    assert fitted["kept_names"] == ["varies"]
    assert fitted["rejected"]["zero_variance"] == ["constant"]


def test_median_ignores_missing_values():
    X = np.array([[1.0], [np.nan], [3.0], [5.0], [7.0]], dtype=np.float32)
    fitted = fit_preprocessor(X, ["f"], CFG)
    assert fitted["medians"][0] == pytest.approx(4.0)   # median of 1,3,5,7


def test_all_nan_column_does_not_poison_the_fill_values():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    fitted = fit_preprocessor(X, ["good", "empty"], CFG)
    assert np.isfinite(fitted["medians"]).all()


def test_transform_uses_train_statistics_only():
    """A val block with a wildly different distribution must not shift anything."""
    train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    fitted = fit_preprocessor(train, ["f"], CFG)
    mean_before = float(fitted["scaler"].mean_[0])

    huge = np.full((1000, 1), 10_000.0, dtype=np.float32)
    apply_preprocessor(huge, fitted)
    assert float(fitted["scaler"].mean_[0]) == mean_before


def test_inf_is_imputed_not_propagated():
    train = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    fitted = fit_preprocessor(train, ["f"], CFG)
    block = np.array([[np.inf], [-np.inf], [np.nan]], dtype=np.float32)
    block[~np.isfinite(block)] = np.nan            # what _to_float32_matrix does
    out = apply_preprocessor(block, fitted)
    assert np.isfinite(out).all()
    assert np.allclose(out, out[0])                 # all became the median


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def _write_shard(path: Path, labels: list[str]) -> None:
    n = len(labels)
    rng = np.random.default_rng(7)
    flow_bytes = rng.normal(1000, 50, n)
    flow_bytes[:3] = np.inf                       # CICFlowMeter's Infinity
    flow_bytes[3:6] = np.nan
    minutes = (np.arange(n) * 60) // n
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "Flow ID": [f"{path.stem}-{i}" for i in range(n)],
            " Source IP": [f"10.0.0.{i % 4}" for i in range(n)],
            " Destination IP": [f"10.1.0.{i % 3}" for i in range(n)],
            " Flow Duration": pa.array(rng.integers(1, 10**6, n), type=pa.int64()),
            " Fwd Packet Length Max": pa.array(rng.normal(size=n)),
            " Max Packet Length": pa.array(rng.normal(size=n)),
            "Flow Bytes/s": pa.array(flow_bytes),
            " Down/Up Ratio": pa.array(np.ones(n), type=pa.float64()),   # constant
            " Timestamp": [f"2018-12-01 {9 + m // 60:02d}:{m % 60:02d}:00" for m in minutes],
            "__capture_day": ["01-12"] * n,
            "__source_file_id": [path.stem] * n,
            "__source_row_id": pa.array(np.arange(n), type=pa.int64()),
            " Label": labels,
        }),
        path, compression="snappy", row_group_size=400,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("prep")
    data = root / "input"
    labels = []
    for i in range(4000):
        labels.append("BENIGN" if i % 8 == 0 else "DrDoS_DNS")
    _write_shard(data / "01-12" / "DrDoS_DNS.parquet", labels)

    with (REPO_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment["split"]["min_benign_per_split"] = 20
    experiment["subsample"]["attack_per_benign"] = 4
    config_path = root / "experiment_test.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment, handle)

    split_dir = root / "split"
    split = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "split.py"),
         "--input-root", str(data), "--out-dir", str(split_dir),
         "--repo-root", str(REPO_ROOT), "--experiment-config", str(config_path)],
        capture_output=True, text=True,
    )
    assert (split_dir / "split_assignment.npy").exists(), split.stdout + split.stderr

    out = root / "work"
    prep = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "preprocessing.py"),
         "--input-root", str(data), "--split-dir", str(split_dir),
         "--out-dir", str(out), "--repo-root", str(REPO_ROOT),
         "--experiment-config", str(config_path)],
        capture_output=True, text=True,
    )
    report_path = out / "config" / "preprocessing.json"
    assert report_path.exists(), prep.stdout + prep.stderr
    return {
        "out": out,
        "report": json.loads(report_path.read_text(encoding="utf-8")),
        "stdout": prep.stdout,
    }


def test_identifier_and_metadata_columns_never_become_features(pipeline):
    report = pipeline["report"]
    for excluded in ("flow_id", "source_ip", "destination_ip", "timestamp",
                     "capture_day", "source_file_id", "source_row_id", "label"):
        assert excluded not in report["candidate_features"]
        assert excluded in report["removed_by_config"] or excluded == "label"


def test_constant_column_is_dropped_end_to_end(pipeline):
    report = pipeline["report"]
    assert "down_up_ratio" in report["rejected"]["zero_variance"]
    assert "down_up_ratio" not in report["kept_features"]


def test_report_records_that_fitting_used_train_only(pipeline):
    assert pipeline["report"]["fitted_on"] == "train_only"
    assert pipeline["report"]["n_train_rows_used_for_fitting"] > 0


def test_shards_are_finite_and_row_counts_match(pipeline):
    out = pipeline["out"]
    cache = out / "cache" / "preprocess"
    report = pipeline["report"]
    for name in ("train", "val", "test"):
        entry = report["shards"][name]
        y = np.load(cache / f"{name}_y.npy")
        assert len(y) == entry["rows"], name

        total = 0
        for shard in entry["shards"]:
            block = np.load(cache / shard["file"])
            assert np.isfinite(block).all(), f"{name}/{shard['file']} has non-finite values"
            assert block.shape[1] == report["n_kept"]
            assert block.dtype == np.float32
            total += len(block)
        assert total == entry["rows"], name


def test_scaler_artifact_reloads_and_matches_the_report(pipeline):
    bundle = joblib.load(pipeline["out"] / "cache" / "scaler.joblib")
    assert bundle["kept_names"] == pipeline["report"]["kept_features"]
    assert np.allclose(bundle["scaler"].mean_, pipeline["report"]["scaler_mean"])


def test_train_shards_are_standardised(pipeline):
    """Train was what the scaler saw, so its columns should be near zero-mean."""
    cache = pipeline["out"] / "cache" / "preprocess"
    entry = pipeline["report"]["shards"]["train"]
    block = np.concatenate([np.load(cache / s["file"]) for s in entry["shards"]])
    assert np.abs(block.mean(axis=0)).max() < 1e-3
    assert np.abs(block.std(axis=0) - 1.0).max() < 1e-2
