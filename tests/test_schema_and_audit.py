"""Schema normalisation and data_audit fail-fast behaviour.

Synthesises small Parquet shards in both header styles (InSDN abbreviated,
CIC-DDoS2019 long-with-leading-space) so the alias table is exercised without
needing the real datasets.

Run:  python -m pytest tests/ -q       (from the repo root)
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

from schema import (  # noqa: E402
    build_alias_index,
    build_column_mapping,
    find_label_column,
    normalize_column_name,
)


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (" Destination Port", "destination_port"),
        ("Dst Port", "dst_port"),
        ("DestinationPort", "destination_port"),
        ("Flow Bytes/s", "flow_bytes_s"),
        ("Flow Byts/s", "flow_byts_s"),
        ("Fwd IAT Total", "fwd_iat_total"),
        ("FwdIATTotal", "fwd_iat_total"),
        ("min_seg_size_forward", "min_seg_size_forward"),
        ("Unnamed: 0", "unnamed_0"),
        ("SimillarHTTP", "simillar_http"),
        ("﻿ Label ", "label"),
    ],
)
def test_normalize_column_name(raw, expected):
    assert normalize_column_name(raw) == expected


def test_spaced_acronym_is_not_split_per_letter():
    """"Fwd IAT Total" must not become fwd_i_a_t_total."""
    assert normalize_column_name("Fwd IAT Total") == "fwd_iat_total"
    assert normalize_column_name("Bwd IAT Max") == "bwd_iat_max"


def test_both_header_styles_collapse_onto_one_canonical_name():
    aliases = {
        "destination_port": ["dst_port", "destination_port"],
        "fwd_packet_length_mean": ["fwd_pkt_len_mean", "fwd_packet_length_mean"],
    }
    insdn, _ = build_column_mapping(["Dst Port", "Fwd Pkt Len Mean"], aliases)
    cic, _ = build_column_mapping([" Destination Port", " Fwd Packet Length Mean"], aliases)
    assert set(insdn.values()) == set(cic.values())
    assert set(insdn.values()) == {"destination_port", "fwd_packet_length_mean"}


def test_ambiguous_alias_is_rejected():
    with pytest.raises(ValueError, match="claimed by both"):
        build_alias_index({"a_col": ["shared"], "b_col": ["shared"]})


def test_duplicate_canonical_is_reported_not_raised():
    aliases = {"destination_port": ["dst_port", "destination_port"]}
    _, collisions = build_column_mapping(["Dst Port", "Destination Port"], aliases)
    assert collisions == ["destination_port"]


def test_find_label_column_tolerates_leading_space():
    assert find_label_column(["Flow Duration", " Label"], ["Label"]) == " Label"
    with pytest.raises(KeyError):
        find_label_column(["Flow Duration"], ["Label"])


# --------------------------------------------------------------------------
# data_audit.py
# --------------------------------------------------------------------------

def _write_shard(path: Path, labels: list[str], with_inf: bool = True) -> None:
    """One InSDN-style shard; Flow Byts/s carries the ±Inf that CICFlowMeter emits."""
    n = len(labels)
    rng = np.random.default_rng(0)
    flow_bytes = rng.normal(size=n).astype("float64")
    if with_inf and n >= 2:
        flow_bytes[0] = np.inf
        flow_bytes[1] = np.nan

    table = pa.table(
        {
            "Flow ID": [f"flow-{i}" for i in range(n)],
            "Src IP": [f"10.0.0.{i % 7}" for i in range(n)],
            "Dst IP": [f"10.0.1.{i % 5}" for i in range(n)],
            "Dst Port": pa.array(rng.integers(0, 65535, n), type=pa.int32()),
            "Flow Duration": pa.array(rng.integers(1, 10**6, n), type=pa.int64()),
            "Fwd Pkt Len Mean": pa.array(rng.normal(size=n), type=pa.float64()),
            "Flow Byts/s": pa.array(flow_bytes, type=pa.float64()),
            "Timestamp": [f"2020-06-{(i % 28) + 1:02d} 10:00:00" for i in range(n)],
            "Label": labels,
        }
    )
    pq.write_table(table, path, compression="snappy")


def _run_audit(tmp_path: Path, labels: list[str]) -> tuple[int, dict]:
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    _write_shard(input_root / "shard000.parquet", labels)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "data_audit.py"),
            "--dataset", "insdn",
            "--input-root", str(input_root),
            "--out-dir", str(out_dir),
            "--repo-root", str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    profile_path = out_dir / "data_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    return result.returncode, profile


def test_audit_happy_path(tmp_path):
    labels = ["Normal"] * 40 + ["DDoS"] * 35 + ["DoS"] * 25
    code, profile = _run_audit(tmp_path, labels)

    assert code == 0, profile.get("checks")
    assert profile["total_rows"] == 100
    assert profile["label_counts_binary"] == {"0_benign": 40, "1_attack": 60}
    assert profile["checks"]["passed"] is True

    # "Flow Byts/s" resolves through the alias table onto the canonical rate name.
    assert "flow_bytes_per_s" in profile["columns"]

    # Inf and NaN are counted apart: one of each was injected. Arrow reports no
    # null here, which is exactly why n_nan has to be tracked separately.
    flow_bytes = profile["columns"]["flow_bytes_per_s"]
    assert flow_bytes["n_inf"] == 1
    assert flow_bytes["n_nan"] == 1
    assert flow_bytes["n_null"] == 0
    assert flow_bytes["n_missing"] == 2
    assert flow_bytes["missing_ratio"] == pytest.approx(0.02)

    # Per-file label counts exist, since group-aware split depends on them.
    assert profile["files"][0]["label_counts"] == {"DDoS": 35, "DoS": 25, "Normal": 40}
    assert profile["n_unique_source_ip"] == 7


def test_audit_applies_ddos_dos_only_scope(tmp_path):
    labels = ["Normal"] * 30 + ["DDoS"] * 30 + ["Probe"] * 25 + ["BFA"] * 15
    code, profile = _run_audit(tmp_path, labels)

    assert code == 0
    assert profile["attack_scope"] == "ddos_dos_only"
    assert profile["attack_scope_dropped_rows"] == 40      # Probe + BFA
    assert profile["label_counts_binary"] == {"0_benign": 30, "1_attack": 30}


def test_audit_fails_without_benign(tmp_path):
    code, profile = _run_audit(tmp_path, ["DDoS"] * 50 + ["DoS"] * 50)
    assert code == 2
    assert any("BENIGN" in f for f in profile["checks"]["failures"])


def test_audit_fails_on_unknown_label(tmp_path):
    code, profile = _run_audit(tmp_path, ["Normal"] * 50 + ["Mirai"] * 50)
    assert code == 2
    assert any("outside the configured mapping" in f for f in profile["checks"]["failures"])
