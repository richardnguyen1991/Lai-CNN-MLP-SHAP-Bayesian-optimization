"""Split, dedup and subsample behaviour.

Builds a miniature CIC-DDoS2019: two capture days, duplicate rows planted on
purpose, a rare attack family, and BENIGN spread through the capture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from split import first_occurrence_mask  # noqa: E402


def test_first_occurrence_mask_keeps_the_earliest_row():
    hashes = np.array([7, 3, 7, 9, 3, 7], dtype=np.uint64)
    keep = first_occurrence_mask(hashes)
    assert keep.tolist() == [True, True, False, True, False, False]


def test_first_occurrence_mask_is_deterministic():
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 50, size=5000, dtype=np.uint64)
    first = first_occurrence_mask(hashes)
    for _ in range(3):
        assert np.array_equal(first, first_occurrence_mask(hashes))
    # Exactly one survivor per distinct value.
    assert int(first.sum()) == len(np.unique(hashes))


def _write_shard(path: Path, day: str, labels: list[str], n_duplicate: int = 0) -> None:
    """BENIGN is spread evenly through the capture so a temporal split keeps it."""
    n = len(labels)
    rng = np.random.default_rng(abs(hash(path.name)) % 2**31)
    features = rng.integers(0, 10**6, size=(n, 3))
    if n_duplicate:
        features[-n_duplicate:] = features[:n_duplicate]   # exact repeats
    minutes = (np.arange(n) * 600) // n
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "Flow ID": [f"{path.stem}-{i}" for i in range(n)],
            " Source IP": [f"10.0.0.{i % 5}" for i in range(n)],
            " Destination IP": [f"10.1.0.{i % 3}" for i in range(n)],
            " Flow Duration": pa.array(features[:, 0], type=pa.int64()),
            " Fwd Packet Length Max": pa.array(features[:, 1], type=pa.int64()),
            " Max Packet Length": pa.array(features[:, 2], type=pa.int64()),
            " Timestamp": [f"2018-12-01 {9 + m // 60:02d}:{m % 60:02d}:00" for m in minutes],
            "CaptureDay": [day] * n,
            " Label": labels,
        }),
        path, compression="snappy", row_group_size=500,
    )


def _interleave(benign: int, attack_label: str, attack: int) -> list[str]:
    labels = [attack_label] * attack
    step = max(attack // benign, 1)
    for i in range(benign):
        labels.insert(min(i * step + i, len(labels)), "BENIGN")
    return labels


@pytest.fixture(scope="module")
def split_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("split")
    data = root / "input"

    # Two shapes of rare family, because they behave differently:
    #   Portmap is sprinkled through the capture -> reaches every split, and is
    #     small enough that proportional sampling would round it away.
    #   WebDDoS sits at the tail of its file -> a temporal split gives it all to
    #     test, which must raise a warning.
    dns = _interleave(300, "DrDoS_DNS", 6000)
    step = len(dns) // 15
    for k in range(15):
        dns.insert(min(k * step + k, len(dns)), "Portmap")
    _write_shard(data / "01-12" / "DrDoS_DNS.parquet", "01-12", dns, n_duplicate=120)
    _write_shard(data / "01-12" / "UDPLag.parquet", "01-12",
                 _interleave(200, "UDP-lag", 3000) + ["WebDDoS"] * 20)
    _write_shard(data / "03-11" / "Syn.parquet", "03-11",
                 _interleave(400, "Syn", 5000))

    # The production thresholds assume 113,828 BENIGN and a 1:618 prior; this
    # fixture is four orders of magnitude smaller, so it carries its own config.
    # Scaling them down here is also what keeps the guards themselves under test.
    import yaml
    with (REPO_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment["split"]["min_benign_per_split"] = 50
    experiment["subsample"]["attack_per_benign"] = 3
    config_path = root / "experiment_test.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment, handle)

    out = root / "out"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "split.py"),
         "--input-root", str(data), "--out-dir", str(out),
         "--repo-root", str(REPO_ROOT),
         "--experiment-config", str(config_path)],
        capture_output=True, text=True,
    )
    manifest_path = out / "split_manifest.json"
    assert manifest_path.exists(), result.stdout + result.stderr
    return {
        "code": result.returncode,
        "data_root": data,
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "leakage": json.loads((out / "leakage_audit.json").read_text(encoding="utf-8")),
        "assignment": np.load(out / "split_assignment.npy"),
        "offsets": json.loads((out / "file_offsets.json").read_text(encoding="utf-8")),
        "stdout": result.stdout,
    }


def test_split_succeeds_and_every_split_has_benign(split_run):
    assert split_run["code"] == 0, split_run["manifest"]["checks"]
    summary = split_run["manifest"]["summary"]
    for name in ("train", "val", "test"):
        assert summary[name]["rows"] > 0
        assert summary[name]["benign"] > 0, f"{name} lost BENIGN"


def test_duplicates_are_removed_before_splitting(split_run):
    manifest = split_run["manifest"]
    assert manifest["summary"]["dropped_duplicate"] >= 120
    assert manifest["diagnostics"]["n_hash_columns"] == 3


def test_no_duplicate_row_spans_two_splits(split_run):
    assert split_run["leakage"]["cross_split_duplicate_rows"] == 0


def test_identifier_columns_are_excluded_from_the_hash(split_run):
    """Hashing flow_id/IP/timestamp would make every row unique and hide duplicates."""
    hashed = split_run["manifest"]["diagnostics"]["hash_columns"]
    for excluded in ("flow_id", "source_ip", "destination_ip", "timestamp", "label"):
        assert excluded not in hashed


def test_test_split_keeps_the_natural_prior(split_run):
    """B2 thins train and val only; test must stay at the original ratio."""
    sampling = split_run["manifest"]["subsampling"]
    assert sampling["applied"] is True
    assert sampling["not_applied_to"] == ["test"]
    assert "test" not in sampling["splits"]

    summary = split_run["manifest"]["summary"]
    assert summary["train"]["attack_per_benign"] <= sampling["attack_per_benign"] + 1
    # Test is far more imbalanced than the subsampled train split.
    assert summary["test"]["attack_per_benign"] > summary["train"]["attack_per_benign"]


def test_rare_family_survives_subsampling(split_run):
    """A tiny family must not be rounded out of the training split.

    WebDDoS is 439 rows out of 70M in the real dataset, so proportional
    sampling would round it to zero without the per-stratum floor.
    """
    strata = split_run["manifest"]["subsampling"]["splits"]["train"]["per_stratum"]
    rare = {k: v for k, v in strata.items() if v["before"] < 30}
    assert rare, "fixture produced no rare stratum to check"
    assert all(v["kept"] >= 1 for v in rare.values())


def test_every_family_reaches_every_split(split_run):
    """Stratifying the temporal cut by label is what fixes end-clustering.

    WebDDoS sits entirely at the tail of its capture file in the fixture, which
    is exactly what happens to BENIGN in 03-11/Syn.parquet in the real dataset.
    Cutting each (file, label) stream separately keeps it present everywhere.
    """
    coverage = split_run["manifest"]["family_coverage"]
    for family in ("BENIGN", "DrDoS_DNS", "Syn", "UDP-lag", "WebDDoS", "Portmap"):
        for name in ("train", "val", "test"):
            assert coverage[family][name] > 0, f"{family} missing from {name}"
    assert split_run["manifest"]["warnings"] == []


def test_benign_is_split_proportionally_despite_clustering(split_run):
    """The fixture's BENIGN is interleaved, but the guarantee must be structural:
    each split's share of BENIGN should track the configured ratios."""
    coverage = split_run["manifest"]["family_coverage"]["BENIGN"]
    total = coverage["train"] + coverage["val"] + coverage["test"]
    assert coverage["train"] / total == pytest.approx(0.70, abs=0.02)
    assert coverage["val"] / total == pytest.approx(0.15, abs=0.02)
    assert coverage["test"] / total == pytest.approx(0.15, abs=0.02)


def test_identifier_column_in_the_hash_is_rejected(tmp_path):
    """A per-row id inside the row hash silently disables deduplication.

    __source_row_id in the real dataset is exactly this: unique per row, so
    every hash differs and dedup reports zero duplicates while appearing to
    succeed. The guard must fail loudly instead.
    """
    data = tmp_path / "input" / "01-12"
    data.mkdir(parents=True)
    n = 4000
    pq.write_table(
        pa.table({
            " Flow Duration": pa.array(np.zeros(n, dtype=np.int64)),
            "__source_row_id": pa.array(np.arange(n, dtype=np.int64)),
            " Timestamp": ["2018-12-01 10:00:00"] * n,
            " Label": ["BENIGN"] * (n // 2) + ["Syn"] * (n // 2),
        }),
        data / "Syn.parquet", compression="snappy", row_group_size=n,
    )

    import yaml
    with (REPO_ROOT / "configs" / "dataset_cicddos2019.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)
    dataset_cfg["metadata_columns"] = []          # re-introduce the bug

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data_audit import build_label_mapping
    from split import scan_dataset

    with pytest.raises(ValueError, match="row identifiers"):
        scan_dataset(
            tmp_path / "input", dataset_cfg,
            build_label_mapping(dataset_cfg)["decisions"],
        )


def test_a_unique_continuous_feature_is_not_treated_as_an_identifier(tmp_path):
    """Continuous features are routinely unique across a sample.

    Two flows really do differ in flow_iat_std, and a hash that separates them
    is correct. Failing on that would block the run on data that is fine -- it
    is recorded as a note instead.
    """
    data = tmp_path / "input" / "01-12"
    data.mkdir(parents=True)
    n = 4000
    rng = np.random.default_rng(0)
    pq.write_table(
        pa.table({
            " Flow IAT Std": pa.array(rng.normal(size=n)),      # unique, a float
            " Flow Duration": pa.array(rng.integers(1, 50, n)),  # repeats
            " Timestamp": ["2018-12-01 10:00:00"] * n,
            " Label": ["BENIGN"] * (n // 2) + ["Syn"] * (n // 2),
        }),
        data / "Syn.parquet", compression="snappy", row_group_size=n,
    )

    import yaml
    with (REPO_ROOT / "configs" / "dataset_cicddos2019.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data_audit import build_label_mapping
    from split import scan_dataset

    _, diagnostics = scan_dataset(
        tmp_path / "input", dataset_cfg,
        build_label_mapping(dataset_cfg)["decisions"],
    )
    assert "flow_iat_std" in diagnostics["unique_per_row_features"]


def test_assignment_covers_every_row_exactly_once(split_run):
    assignment = split_run["assignment"]
    offsets = split_run["offsets"]
    assert sum(end - start for start, end in offsets.values()) == len(assignment)
    assert set(np.unique(assignment)).issubset({-2, -1, 0, 1, 2})


def test_no_burst_straddles_a_split_boundary(split_run):
    """Time ordering is guaranteed per (file, label) stream, not per file.

    That is the deliberate trade of stratifying the cut: splits overlap in time
    across labels, so test is not strictly later than train dataset-wide. What
    still holds -- and what actually blocks burst-level leakage -- is that
    inside one label's stream in one capture file, every train row precedes
    every val row, which precedes every test row.
    """
    assignment = split_run["assignment"]
    offsets = split_run["offsets"]
    checked = 0

    for relative, (start, end) in offsets.items():
        table = pq.read_table(split_run["data_root"] / relative, columns=[" Label"])
        labels = np.array([str(v).strip() for v in table.column(" Label").to_pylist()])
        local = assignment[start:end]
        assert len(labels) == len(local)

        for label in np.unique(labels):
            stream = np.flatnonzero(labels == label)
            codes = local[stream]
            positions = {code: np.flatnonzero(codes == code) for code in (0, 1, 2)}
            if positions[0].size and positions[1].size:
                assert positions[0].max() < positions[1].min(), f"{relative}/{label}"
            if positions[1].size and positions[2].size:
                assert positions[1].max() < positions[2].min(), f"{relative}/{label}"
            checked += 1

    assert checked >= 6, "fixture did not exercise enough (file, label) streams"
