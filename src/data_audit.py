"""Audit the raw Parquet shards and emit data_profile.json.

Runs before anything else in the pipeline. Its job is to describe what is
actually in the files and to refuse to continue when the data cannot support the
experiment, rather than letting a silent assumption propagate into the results.

Reads one row group at a time through PyArrow, so peak memory is bounded by the
largest row group and not by the dataset size (§2 of the prompt: never load the
whole dataset into RAM).

Emits, into --out-dir:
    data_profile.json    files, rows, dtypes, null/Inf ratios, label counts
    column_mapping.json  raw header -> canonical snake_case
    label_mapping.json   raw label -> binary target, plus dropped scope

Fail-fast conditions (exit code 2):
    - no label column
    - no BENIGN/Normal class
    - fewer than 2 classes after binary mapping
    - a label outside the configured mapping (never silently coerced)

Usage:
    python src/data_audit.py --dataset insdn \
        --input-root /kaggle/input/<slug> --out-dir artifacts/config
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from schema import build_column_mapping, find_label_column, normalize_column_name

# Distinct IP values are only counted to expose capture-setup narrowness. Past a
# couple of million the exact number stops being informative and the set starts
# costing real memory, so tracking is capped and the cap is reported.
MAX_TRACKED_UNIQUE = 2_000_000

EXIT_FAIL_FAST = 2


class ColumnStats:
    """Streaming null/Inf/dtype accumulator for one column."""

    def __init__(self, canonical: str, raw: str) -> None:
        self.canonical = canonical
        self.raw = raw
        self.dtype: Optional[str] = None
        self.n_rows = 0
        self.n_null = 0
        self.n_nan = 0
        self.n_inf = 0
        self.numeric = False
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None

    def update(self, array: pa.Array) -> None:
        if self.dtype is None:
            self.dtype = str(array.type)
            self.numeric = pa.types.is_floating(array.type) or pa.types.is_integer(
                array.type
            )
        self.n_rows += len(array)
        self.n_null += array.null_count

        if not pa.types.is_floating(array.type):
            return

        # CICFlowMeter writes literal "Infinity" into Flow Bytes/s and
        # Flow Packets/s whenever flow duration is zero. Those land here as ±inf
        # and must be counted separately from NaN, because preprocessing treats
        # them differently (±Inf -> NaN -> median impute).
        #
        # Arrow's null_count does NOT include NaN: a float NaN is a valid value
        # with its validity bit set. Counting it here is what keeps
        # missing_ratio honest, and max_nan_ratio drops columns off that ratio.
        values = array.to_numpy(zero_copy_only=False)
        is_nan = np.isnan(values)
        finite = np.isfinite(values)
        self.n_nan += int(np.count_nonzero(is_nan))
        self.n_inf += int(np.count_nonzero(~finite & ~is_nan))
        if finite.any():
            block_min = float(np.min(values[finite]))
            block_max = float(np.max(values[finite]))
            self.min_value = block_min if self.min_value is None else min(self.min_value, block_min)
            self.max_value = block_max if self.max_value is None else max(self.max_value, block_max)

    def to_dict(self) -> Dict[str, Any]:
        denom = max(self.n_rows, 1)
        # ±Inf becomes NaN in preprocessing, so it counts as missing here too.
        n_missing = self.n_null + self.n_nan + self.n_inf
        return {
            "raw_name": self.raw,
            "dtype": self.dtype,
            "numeric": self.numeric,
            "n_null": self.n_null,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "n_missing": n_missing,
            "missing_ratio": round(n_missing / denom, 6),
            "inf_ratio": round(self.n_inf / denom, 6),
            "min": self.min_value,
            "max": self.max_value,
        }


def load_dataset_config(repo_root: Path, dataset: str) -> Dict[str, Any]:
    path = repo_root / "configs" / f"dataset_{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no dataset config at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def describe_input_root(input_root: Path) -> str:
    """What is actually at the input path, for an error that can be acted on.

    On Kaggle a missing input directory almost always means the dataset was not
    attached to the kernel, and the bare "no files matching" message gives no
    way to tell that from a wrong glob or a nested layout.
    """
    if input_root.is_dir():
        entries = sorted(child.name for child in input_root.iterdir())
        suffixes = sorted({child.suffix for child in input_root.rglob("*")
                           if child.is_file() and child.suffix})
        return (f"{input_root} exists and contains {entries[:20]}"
                f"{' ...' if len(entries) > 20 else ''}; "
                f"file extensions present: {suffixes or 'none'}")

    parent = input_root.parent
    if parent.is_dir():
        return (f"{input_root} does not exist; {parent} contains "
                f"{sorted(child.name for child in parent.iterdir())}. "
                "A Kaggle dataset mounts under its own slug, so this usually "
                "means the dataset was not attached to the kernel.")

    return f"neither {input_root} nor {parent} exists"


def discover_files(input_root: Path, pattern: str) -> List[Path]:
    files = sorted(p for p in input_root.rglob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"no files matching {pattern!r} under {input_root}. "
            + describe_input_root(input_root))
    return files


def value_counts(array: pa.Array) -> Counter:
    """Counter of non-null values in an Arrow array."""
    counts: Counter = Counter()
    table = pc.value_counts(array)
    values = table.field("values").to_pylist()
    frequencies = table.field("counts").to_pylist()
    for value, frequency in zip(values, frequencies):
        if value is not None:
            counts[str(value).strip()] += frequency
    return counts


def build_label_mapping(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the raw-label -> {0,1,dropped} decision table from config."""
    mapping_cfg = config["binary_label_mapping"]
    benign = {str(label).strip() for label in mapping_cfg["benign_labels"]}
    attack = {str(label).strip() for label in mapping_cfg["attack_labels"]}

    scope = config.get("attack_scope", "all_attacks")
    definitions = config.get("attack_scope_definitions", {})
    dropped: Set[str] = set()
    if scope in definitions:
        dropped = {str(label).strip() for label in definitions[scope].get("drop", [])}

    decisions: Dict[str, Any] = {}
    for label in sorted(benign):
        decisions[label] = 0
    for label in sorted(attack):
        decisions[label] = "dropped" if label in dropped else 1
    return {
        "attack_scope": scope,
        "benign_labels": sorted(benign),
        "attack_labels": sorted(attack),
        "dropped_labels": sorted(dropped),
        "decisions": decisions,
        "on_unknown_label": mapping_cfg.get("on_unknown_label", "fail"),
    }


def audit(dataset: str, input_root: Path, out_dir: Path, repo_root: Path) -> int:
    config = load_dataset_config(repo_root, dataset)
    files = discover_files(input_root, config.get("input_glob", "*.parquet"))
    label_mapping = build_label_mapping(config)
    decisions = label_mapping["decisions"]

    stats: Dict[str, ColumnStats] = {}
    column_mapping: Dict[str, str] = {}
    collisions: List[str] = []
    raw_label_counts: Counter = Counter()
    file_records: List[Dict[str, Any]] = []
    unique_ips: Dict[str, Set[str]] = {"source_ip": set(), "destination_ip": set()}
    ip_tracking_capped: Dict[str, bool] = {"source_ip": False, "destination_ip": False}
    timestamp_min: Optional[str] = None
    timestamp_max: Optional[str] = None
    total_rows = 0
    label_column_raw: Optional[str] = None

    for path in files:
        parquet_file = pq.ParquetFile(path)
        raw_columns = list(parquet_file.schema_arrow.names)

        if not column_mapping:
            column_mapping, collisions = build_column_mapping(
                raw_columns, config.get("schema_aliases", {})
            )
            label_column_raw = find_label_column(
                raw_columns, config["label_column_candidates"]
            )
        else:
            # A shard with a different header would silently misalign features.
            file_mapping, _ = build_column_mapping(
                raw_columns, config.get("schema_aliases", {})
            )
            if set(file_mapping.values()) != set(column_mapping.values()):
                missing = set(column_mapping.values()) - set(file_mapping.values())
                extra = set(file_mapping.values()) - set(column_mapping.values())
                raise ValueError(
                    f"{path.name} has a different schema; missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )

        file_rows = 0
        file_label_counts: Counter = Counter()

        for group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group_index)
            file_rows += table.num_rows

            for raw_name in table.column_names:
                canonical = column_mapping[raw_name]
                array = table.column(raw_name).combine_chunks()

                if canonical not in stats:
                    stats[canonical] = ColumnStats(canonical, raw_name)
                stats[canonical].update(array)

                if raw_name == label_column_raw:
                    file_label_counts.update(value_counts(array))
                elif canonical in unique_ips and not ip_tracking_capped[canonical]:
                    bucket = unique_ips[canonical]
                    bucket.update(str(v) for v in array.to_pylist() if v is not None)
                    if len(bucket) > MAX_TRACKED_UNIQUE:
                        ip_tracking_capped[canonical] = True
                        bucket.clear()
                elif canonical == "timestamp":
                    present = [str(v) for v in array.to_pylist() if v is not None]
                    if present:
                        low, high = min(present), max(present)
                        timestamp_min = low if timestamp_min is None else min(timestamp_min, low)
                        timestamp_max = high if timestamp_max is None else max(timestamp_max, high)

            del table

        raw_label_counts.update(file_label_counts)
        total_rows += file_rows
        file_records.append(
            {
                # Relative, not path.name: CIC-DDoS2019 ships the same basenames
                # on both capture days (01-12/Syn.parquet and 03-11/Syn.parquet),
                # and collapsing them would merge two distinct capture groups into
                # one, silently corrupting the group-aware split.
                "path": path.relative_to(input_root).as_posix(),
                "rows": file_rows,
                "row_groups": parquet_file.num_row_groups,
                "size_bytes": path.stat().st_size,
                # Per-file label counts drive the group-aware split: a file with
                # no BENIGN cannot be assigned to a split on its own.
                "label_counts": dict(sorted(file_label_counts.items())),
            }
        )

    unknown = sorted(set(raw_label_counts) - set(decisions))
    binary_counts = Counter()
    dropped_rows = 0
    for label, count in raw_label_counts.items():
        decision = decisions.get(label)
        if decision == "dropped":
            dropped_rows += count
        elif decision in (0, 1):
            binary_counts[decision] += count

    if len({record["path"] for record in file_records}) != len(file_records):
        raise ValueError("file records are not uniquely keyed; relative paths collided")

    numeric_columns = [name for name, stat in stats.items() if stat.numeric]
    profile: Dict[str, Any] = {
        "dataset_name": dataset,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "n_files": len(files),
        "files": file_records,
        "total_rows": total_rows,
        "n_raw_columns": len(column_mapping),
        "label_column_raw": label_column_raw,
        "column_collisions": collisions,
        "columns": {name: stat.to_dict() for name, stat in sorted(stats.items())},
        "label_counts_raw": dict(sorted(raw_label_counts.items())),
        "label_counts_binary": {
            "0_benign": binary_counts.get(0, 0),
            "1_attack": binary_counts.get(1, 0),
        },
        "attack_scope": label_mapping["attack_scope"],
        "attack_scope_dropped_rows": dropped_rows,
        "unknown_labels": unknown,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "n_unique_source_ip": (
            "capped_at_%d" % MAX_TRACKED_UNIQUE
            if ip_tracking_capped["source_ip"]
            else len(unique_ips["source_ip"])
        ),
        "n_unique_destination_ip": (
            "capped_at_%d" % MAX_TRACKED_UNIQUE
            if ip_tracking_capped["destination_ip"]
            else len(unique_ips["destination_ip"])
        ),
        "n_numeric_columns": len(numeric_columns),
        "estimated_float32_bytes": len(numeric_columns) * total_rows * 4,
        "paper_claimed_rows": config.get("paper_claimed_rows"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "column_mapping.json", column_mapping)
    _write_json(out_dir / "label_mapping.json", label_mapping)

    failures = _collect_failures(profile, label_mapping, unknown, binary_counts)
    profile["checks"] = {
        "passed": not failures,
        "failures": failures,
    }
    _write_json(out_dir / "data_profile.json", profile)

    _report(profile, failures)
    return EXIT_FAIL_FAST if failures else 0


def _collect_failures(
    profile: Dict[str, Any],
    label_mapping: Dict[str, Any],
    unknown: List[str],
    binary_counts: Counter,
) -> List[str]:
    failures: List[str] = []
    if profile["label_column_raw"] is None:
        failures.append("no label column found")
    if unknown and label_mapping["on_unknown_label"] == "fail":
        failures.append(
            f"labels outside the configured mapping: {unknown}; "
            "add them to the dataset config rather than letting them be guessed"
        )
    if binary_counts.get(0, 0) == 0:
        failures.append("no BENIGN/Normal rows after label mapping")
    if sum(1 for count in binary_counts.values() if count > 0) < 2:
        failures.append("fewer than 2 classes after binary mapping")
    if profile["column_collisions"]:
        failures.append(
            f"raw columns collapse onto the same canonical name: "
            f"{profile['column_collisions']}"
        )
    return failures


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)


def _report(profile: Dict[str, Any], failures: List[str]) -> None:
    benign = profile["label_counts_binary"]["0_benign"]
    attack = profile["label_counts_binary"]["1_attack"]
    kept = benign + attack
    print(f"dataset            : {profile['dataset_name']}")
    print(f"files              : {profile['n_files']}")
    print(f"rows (raw)         : {profile['total_rows']:,}")
    print(f"rows (after scope) : {kept:,}  (dropped {profile['attack_scope_dropped_rows']:,})")
    print(f"columns            : {profile['n_raw_columns']}  ({profile['n_numeric_columns']} numeric)")
    if kept:
        print(f"BENIGN             : {benign:,}  ({benign / kept:.4%})")
        print(f"ATTACK             : {attack:,}  ({attack / kept:.4%})")
    print(f"est. float32 size  : {profile['estimated_float32_bytes'] / 2**30:.2f} GiB")

    worst = sorted(
        profile["columns"].items(), key=lambda kv: kv[1]["missing_ratio"], reverse=True
    )[:5]
    print("worst missing columns (null + NaN + Inf):")
    for name, stat in worst:
        print(
            f"  {name:<34} missing={stat['missing_ratio']:.4f} "
            f"(null={stat['n_null']} nan={stat['n_nan']} inf={stat['n_inf']})"
        )

    claimed = profile.get("paper_claimed_rows")
    if claimed:
        delta = profile["total_rows"] - claimed["total"]
        print(f"paper claims {claimed['total']:,} rows; observed delta {delta:+,}")

    if failures:
        print("\nFAIL-FAST:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=["cicddos2019", "insdn"])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="directory containing configs/",
    )
    args = parser.parse_args()
    return audit(args.dataset, args.input_root, args.out_dir, args.repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
