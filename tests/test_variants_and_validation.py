"""Variant resolution, the deliberate paperlike leak, and acceptance validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from preprocessing import rows_for_split  # noqa: E402
from variants import available_variants, deep_merge, resolve  # noqa: E402


# --------------------------------------------------------------------------
# Variant resolution
# --------------------------------------------------------------------------

def test_all_seven_variants_resolve():
    names = available_variants(REPO_ROOT)
    assert set(names) == {"main", "featall", "feat20", "paperlike",
                          "mlponly", "cnnonly", "nobo"}
    for name in names:
        resolved = resolve(REPO_ROOT, name)
        assert resolved["run"]["variant"] == name
        assert resolved["training"]["epochs"] == 100


def test_deep_merge_replaces_lists_rather_than_extending():
    """A variant narrowing subsample.apply_to means the shorter list."""
    merged = deep_merge({"a": {"b": [1, 2, 3], "c": 1}}, {"a": {"b": [9]}})
    assert merged == {"a": {"b": [9], "c": 1}}


def test_variants_change_only_what_they_declare():
    main = resolve(REPO_ROOT, "main")
    feat20 = resolve(REPO_ROOT, "feat20")
    assert feat20["shap"]["top_k"] == 20
    assert main["shap"]["top_k"] == 40
    # Everything outside the shap block is untouched.
    for section in ("training", "session", "split", "device"):
        assert feat20[section] == main[section]


def test_only_main_runs_a_bayesian_search():
    """The others reuse main's parameters, so each ablation isolates one factor."""
    searching = [name for name in available_variants(REPO_ROOT)
                 if resolve(REPO_ROOT, name)["_variant_metadata"]
                 .get("_runs_bayesian_search")]
    assert searching == ["main"]


def test_a_variant_cannot_relax_a_fixed_constraint(tmp_path):
    bad = REPO_ROOT / "configs" / "variants" / "_tmp_bad.yaml"
    bad.write_text("training:\n  epochs: 50\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="must stay 100"):
            resolve(REPO_ROOT, "_tmp_bad")
    finally:
        bad.unlink()


def test_only_paperlike_may_leak(tmp_path):
    bad = REPO_ROOT / "configs" / "variants" / "_tmp_leak.yaml"
    bad.write_text("leakage:\n  upsample_before_split: true\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="only paperlike may"):
            resolve(REPO_ROOT, "_tmp_leak")
    finally:
        bad.unlink()

    assert resolve(REPO_ROOT, "paperlike")["leakage"]["upsample_before_split"] is True


# --------------------------------------------------------------------------
# The deliberate leak
# --------------------------------------------------------------------------

def test_rows_for_split_includes_upsampled_copies():
    assignment = np.array([0, 1, 2, 0], dtype=np.int8)
    also_in = np.array([0b100, 0, 0, 0b010], dtype=np.uint8)   # row0 -> test too
    assert rows_for_split(assignment, also_in, 0).tolist() == [True, False, False, True]
    assert rows_for_split(assignment, also_in, 2).tolist() == [True, False, True, False]


def _write_shard(path: Path, labels: list[str]) -> None:
    n = len(labels)
    rng = np.random.default_rng(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "Flow ID": [f"f{i}" for i in range(n)],
            " Source IP": [f"10.0.0.{i % 5}" for i in range(n)],
            " Destination IP": [f"10.1.0.{i % 3}" for i in range(n)],
            " Flow Duration": pa.array(rng.integers(0, 5000, n), type=pa.int64()),
            " Max Packet Length": pa.array(rng.integers(0, 5000, n), type=pa.int64()),
            " Timestamp": [f"2018-12-01 10:{i % 60:02d}:00" for i in range(n)],
            " Label": labels,
        }),
        path, compression="snappy", row_group_size=1000,
    )


def _run_split(tmp_path: Path, variant: str) -> dict:
    data = tmp_path / "input" / "01-12"
    labels = []
    for i in range(6000):
        labels.append("BENIGN" if i % 20 == 0 else "DrDoS_DNS")
    _write_shard(data / "DrDoS_DNS.parquet", labels)

    resolved = resolve(REPO_ROOT, variant)
    resolved["split"]["min_benign_per_split"] = 10
    resolved["subsample"]["attack_per_benign"] = 5
    config = tmp_path / f"{variant}.yaml"
    with config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle)

    out = tmp_path / f"out_{variant}"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "split.py"),
         "--input-root", str(tmp_path / "input"), "--out-dir", str(out),
         "--repo-root", str(REPO_ROOT), "--experiment-config", str(config)],
        capture_output=True, text=True, check=False,
    )
    return {
        "manifest": json.loads((out / "split_manifest.json").read_text(encoding="utf-8")),
        "leakage": json.loads((out / "leakage_audit.json").read_text(encoding="utf-8")),
        "assignment": np.load(out / "split_assignment.npy"),
        "also_in": np.load(out / "split_also_in.npy"),
    }


def test_clean_variant_produces_no_duplicated_rows(tmp_path):
    result = _run_split(tmp_path, "main")
    assert result["also_in"].max() == 0, "the clean run must not duplicate any row"
    assert result["leakage"]["cross_split_duplicate_rows"] == 0
    assert result["leakage"]["upsample_before_split"] is False


def test_paperlike_puts_benign_rows_on_both_sides(tmp_path):
    """The leak the variant exists to measure, made explicit.

    Up-sampling before the split is what section 2.5 of the paper describes, and
    it is the most likely reason a 0.16%-BENIGN dataset yields 99.95%.
    """
    result = _run_split(tmp_path, "paperlike")
    upsampling = result["manifest"]["upsampling_before_split"]

    assert upsampling["applied"] is True
    assert upsampling["copies_per_benign_row"] > 1
    assert upsampling["benign_rows_present_in_more_than_one_split"] > 0
    assert result["also_in"].max() > 0

    # A row in both train and test is exactly the failure mode.
    in_train = rows_for_split(result["assignment"], result["also_in"], 0)
    in_test = rows_for_split(result["assignment"], result["also_in"], 2)
    assert int(np.count_nonzero(in_train & in_test)) > 0

    assert "Do not read this run's metrics as a clean result" in \
        result["leakage"]["deliberate_leak"]


# --------------------------------------------------------------------------
# Acceptance validation
# --------------------------------------------------------------------------

def _minimal_run(root: Path, **overrides) -> Path:
    """A run directory that passes everything, so tests can break one thing."""
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    (root / "explainability").mkdir(parents=True, exist_ok=True)

    history = overrides.get("history", [{"epoch": e} for e in range(1, 101)])
    payloads = {
        "checkpoints/history.json": history,
        "checkpoints/training_state.json": {
            "total_epochs": 100, "current_epoch": 100,
            "feature_schema_hash": "abc123",
        },
        "config/run_config.json": {"experiment": {"training": {
            "epochs": 100, "batch_size": 4096, "learning_rate": 0.001,
            "early_stopping": False,
        }}},
        "config/leakage_audit.json": {"cross_split_duplicate_rows": 0,
                                      "upsample_before_split": False},
        "config/preprocessing.json": {"fitted_on": "train_only"},
        "cache/selected_features.json": {"feature_schema_hash": "abc123"},
        "bayesopt/best_params.json": {"n_trials_completed": 20,
                                      "n_trials_requested": 20,
                                      "params": {"dropout": 0.2}},
        "metrics/test_metrics.json": dict(
            {name: 0.9 for name in
             ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
              "macro_f1", "mcc", "roc_auc", "pr_auc", "log_loss",
              "specificity", "fpr", "fnr")},
            checkpoint="model_epoch_100.pt"),
        "explainability/explainability_notes.json": {"fed_back_into_selection": False},
    }
    payloads.update(overrides.get("payloads", {}))

    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    (root / "checkpoints" / "model_epoch_100.pt").write_bytes(b"weights")
    (root / "metrics" / "comparison_with_paper.csv").write_text(
        "metric,paper_headline,paper_body,ours\naccuracy,0.9995,0.95,0.99\n",
        encoding="utf-8")
    return root


def _validate(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_artifacts.py"),
         "--run-dir", str(root)],
        capture_output=True, text=True,
    )


def test_a_complete_run_passes_every_criterion(tmp_path):
    result = _validate(_minimal_run(tmp_path / "good"))
    assert result.returncode == 0, result.stdout
    assert "0 failed" in result.stdout


def test_a_missing_epoch_is_caught(tmp_path):
    """The failure resume is most likely to produce, and the hardest to spot."""
    history = [{"epoch": e} for e in range(1, 101) if e != 47]
    result = _validate(_minimal_run(tmp_path / "gap", history=history))
    assert result.returncode == 2
    assert "missing epochs: [47]" in result.stdout


def test_a_repeated_epoch_is_caught(tmp_path):
    history = [{"epoch": e} for e in range(1, 101)] + [{"epoch": 100}]
    result = _validate(_minimal_run(tmp_path / "dup", history=history))
    assert result.returncode == 2
    assert "more than once" in result.stdout


def test_scoring_test_with_the_best_val_checkpoint_is_caught(tmp_path):
    root = _minimal_run(tmp_path / "bestval")
    metrics = json.loads((root / "metrics" / "test_metrics.json").read_text())
    metrics["checkpoint"] = "model_best_val.pt"
    (root / "metrics" / "test_metrics.json").write_text(json.dumps(metrics))

    result = _validate(root)
    assert result.returncode == 2
    assert "not the epoch-100 model" in result.stdout


def test_a_mismatched_feature_hash_is_caught(tmp_path):
    root = _minimal_run(tmp_path / "hash")
    (root / "cache" / "selected_features.json").write_text(
        json.dumps({"feature_schema_hash": "different"}))
    result = _validate(root)
    assert result.returncode == 2
    assert "mixes two feature sets" in result.stdout


def test_searching_a_fixed_hyperparameter_is_caught(tmp_path):
    root = _minimal_run(tmp_path / "search")
    (root / "bayesopt" / "best_params.json").write_text(json.dumps({
        "n_trials_completed": 20, "n_trials_requested": 20,
        "params": {"learning_rate": 0.01}}))
    result = _validate(root)
    assert result.returncode == 2
    assert "meant to be fixed" in result.stdout


def test_paperlike_is_not_failed_for_its_deliberate_leak(tmp_path):
    """It leaks by design; failing it here would be the wrong signal."""
    root = _minimal_run(tmp_path / "paperlike")
    (root / "config" / "leakage_audit.json").write_text(json.dumps({
        "cross_split_duplicate_rows": 12345, "upsample_before_split": True}))
    result = _validate(root)
    assert result.returncode == 0, result.stdout


def test_an_unfinished_run_is_skipped_not_failed(tmp_path):
    """A run still in its search has no test metrics, which is not a failure."""
    root = tmp_path / "partial"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "history.json").write_text(
        json.dumps([{"epoch": e} for e in range(1, 21)]))
    result = _validate(root)
    assert "SKIP" in result.stdout
