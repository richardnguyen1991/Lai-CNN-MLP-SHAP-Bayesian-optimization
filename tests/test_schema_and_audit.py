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


# --------------------------------------------------------------------------
# Locating the input files
# --------------------------------------------------------------------------
# The first real Kaggle session died here, and the message named only what was
# absent. On Kaggle the overwhelmingly likely cause is a dataset that never
# attached to the kernel, which looks identical to a wrong glob until you can
# see what is actually on disk.

from data_audit import describe_input_root, discover_files  # noqa: E402


def test_the_error_lists_what_is_there_when_the_glob_matches_nothing(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "notes.csv").write_text("a,b", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_files(root, "*.parquet")

    message = str(excinfo.value)
    assert "*.parquet" in message
    assert "notes.csv" in message      # what is present, not only what is not
    assert ".csv" in message


def test_the_error_names_the_sibling_directories_when_the_root_is_absent(tmp_path):
    (tmp_path / "some-other-dataset").mkdir()
    missing = tmp_path / "cicddos2019-parquet"

    with pytest.raises(FileNotFoundError) as excinfo:
        discover_files(missing, "*.parquet")

    message = str(excinfo.value)
    # Naming the siblings is what tells a mis-typed slug from a dataset that
    # was never attached.
    assert "some-other-dataset" in message
    assert "not attached" in message


def test_a_nested_layout_is_found_rather_than_reported_missing(tmp_path):
    root = tmp_path / "input"
    (root / "01-12").mkdir(parents=True)
    target = root / "01-12" / "Syn.parquet"
    target.write_bytes(b"")

    assert discover_files(root, "*.parquet") == [target]


def test_describe_walks_up_to_the_nearest_directory_that_is_there(tmp_path):
    # Several levels can be missing at once, and the useful thing to report is
    # the deepest one that does exist rather than the immediate parent.
    message = describe_input_root(tmp_path / "no" / "such" / "place")
    assert str(tmp_path) in message
    assert "does not exist" in message


# --------------------------------------------------------------------------
# Resolving the mount point
# --------------------------------------------------------------------------
# Kaggle moved its dataset mounts out of /kaggle/input/<slug>. Nothing in the
# repository changed, and a correct configuration became a dead one: the real
# session reported "/kaggle/input contains ['datasets']".

from data_audit import first_existing_ancestor, resolve_input_root  # noqa: E402


def kaggle_layout(tmp_path):
    """The mount as the failing session actually found it."""
    root = tmp_path / "input"
    data = (root / "datasets" / "dungnguyen28101991" / "cicddos2019-parquet"
            / "versions" / "1" / "cicddos2019_parquet_preserved" / "data")
    for day, name in (("01-12", "DrDoS_DNS"), ("01-12", "DrDoS_NTP"), ("03-11", "Syn")):
        target = data / day / f"{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    (data.parent / "README.txt").write_text("notes", encoding="utf-8")
    return root, data


def test_the_configured_root_is_kept_when_it_holds_the_data(tmp_path):
    root = tmp_path / "input" / "cicddos2019-parquet"
    (root / "01-12").mkdir(parents=True)
    (root / "01-12" / "Syn.parquet").write_bytes(b"")

    assert resolve_input_root(root, "*.parquet") == root


def test_a_moved_mount_is_found_under_the_nearest_existing_ancestor(tmp_path):
    root, data = kaggle_layout(tmp_path)
    configured = root / "cicddos2019-parquet"          # does not exist
    assert resolve_input_root(configured, "*.parquet") == data


def test_the_resolved_root_keeps_the_capture_day_in_the_relative_path(tmp_path):
    # The relative path is the file identity. Resolving one level too deep
    # would drop the capture day and merge two files that share a name across
    # the two capture days.
    root, data = kaggle_layout(tmp_path)
    resolved = resolve_input_root(root / "cicddos2019-parquet", "*.parquet")

    relative = sorted(p.relative_to(resolved).as_posix()
                      for p in resolved.rglob("*.parquet"))
    assert relative == ["01-12/DrDoS_DNS.parquet",
                        "01-12/DrDoS_NTP.parquet",
                        "03-11/Syn.parquet"]


def test_a_non_matching_file_does_not_drag_the_root_upwards(tmp_path):
    # README.txt sits above the data directory; only the glob matches count.
    root, data = kaggle_layout(tmp_path)
    assert resolve_input_root(root / "cicddos2019-parquet", "*.parquet") == data


def test_resolution_fails_when_nothing_matches_anywhere(tmp_path):
    root = tmp_path / "input"
    (root / "datasets").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_input_root(root / "cicddos2019-parquet", "*.parquet")

    assert "datasets" in str(excinfo.value)


def test_resolution_does_not_search_outside_the_configured_ancestor(tmp_path):
    # Widening the search by more than the mount moved could pick up an
    # unrelated dataset and train on the wrong data without saying so.
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "other.parquet").write_bytes(b"")
    configured = tmp_path / "input" / "cicddos2019-parquet"
    configured.parent.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        resolve_input_root(configured, "*.parquet")


def test_first_existing_ancestor_walks_up_until_something_is_there(tmp_path):
    assert first_existing_ancestor(tmp_path / "a" / "b" / "c") == tmp_path
    assert first_existing_ancestor(tmp_path) == tmp_path
