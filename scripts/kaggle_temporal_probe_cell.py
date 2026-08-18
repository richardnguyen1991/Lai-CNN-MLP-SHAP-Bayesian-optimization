# ==========================================================================
# Step 2c -- is a temporal split viable?
#
# All 18 capture files contain BENIGN, so a within-file temporal split
# (first 70% train / next 15% val / last 15% test, ordered by Timestamp)
# would keep every attack family AND both classes in all three splits, while
# still blocking the burst-level leakage that a random row shuffle creates.
#
# That only holds if BENIGN is spread over the capture rather than clumped at
# one end. This measures exactly that, without sorting 70M rows: it aggregates
# (benign, total) per wall-clock MINUTE, then walks the minutes in time order to
# find where the 70% and 85% row boundaries land and how much BENIGN falls in
# each slice. Memory cost is one small dict per file.
#
# Projects two columns out of 91, so it is far cheaper than the full audit.
# Paste into a Kaggle cell and run. No internet, no credentials.
# ==========================================================================

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

INPUT_ROOT = Path("/kaggle/input/cicddos2019-parquet")
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

# --- kept byte-identical to src/schema.py so the two cannot disagree ---------
_PUNCT_TO_UNDERSCORE = re.compile(r"[^0-9a-zA-Z]+")
_MULTI_UNDERSCORE = re.compile(r"_+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def normalize_column_name(raw: str) -> str:
    name = unicodedata.normalize("NFKC", raw).replace("﻿", "").strip()
    if not name:
        return ""
    if not re.search(r"\s", name):
        name = _CAMEL_BOUNDARY.sub(" ", name)
    name = _PUNCT_TO_UNDERSCORE.sub("_", name)
    return _MULTI_UNDERSCORE.sub("_", name).strip("_").lower()
# ----------------------------------------------------------------------------


if not INPUT_ROOT.exists():
    candidates = [
        p for p in sorted(Path("/kaggle/input").rglob("*"))
        if p.is_dir() and next(p.glob("*.parquet"), None) is not None
    ]
    if not candidates:
        raise SystemExit("no directory containing .parquet under /kaggle/input")
    # Deepest common ancestor of every directory that directly holds parquet.
    INPUT_ROOT = Path(*Path(candidates[0]).parts[: len(Path(candidates[0]).parts) - 1])
    print(f"note: falling back to INPUT_ROOT = {INPUT_ROOT}")

shards = sorted(INPUT_ROOT.rglob("*.parquet"))
print(f"INPUT_ROOT = {INPUT_ROOT}")
print(f"{len(shards)} shards\n")

schema_names = list(pq.ParquetFile(shards[0]).schema_arrow.names)
by_canonical = {normalize_column_name(n): n for n in schema_names}
label_col = by_canonical.get("label")
time_col = by_canonical.get("timestamp")
print("resolved columns:")
for key in ("label", "timestamp", "capture_day", "source_file_id"):
    print(f"  {key:<16} -> {by_canonical.get(key)!r}")
if label_col is None or time_col is None:
    raise SystemExit(f"need label and timestamp; schema was {schema_names}")

report = {}
for shard in shards:
    relative = shard.relative_to(INPUT_ROOT).as_posix()
    parquet = pq.ParquetFile(shard)

    per_minute = defaultdict(lambda: [0, 0])   # minute -> [total, benign]
    monotonic = True
    previous = None

    for group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(group_index, columns=[label_col, time_col])
        labels = table.column(label_col).to_pylist()
        stamps = table.column(time_col).to_pylist()
        for label, stamp in zip(labels, stamps):
            text = str(stamp)
            if previous is not None and text < previous:
                monotonic = False
            previous = text
            bucket = per_minute[text[:16]]        # "YYYY-MM-DD HH:MM"
            bucket[0] += 1
            if str(label).strip().upper() in ("BENIGN", "NORMAL"):
                bucket[1] += 1
        del table

    minutes = sorted(per_minute)
    total = sum(per_minute[m][0] for m in minutes)
    benign_total = sum(per_minute[m][1] for m in minutes)

    # Walk minutes in time order, cutting at the 70% and 85% row boundaries.
    slices = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    seen = 0
    for minute in minutes:
        rows, benign = per_minute[minute]
        position = seen / total if total else 0.0
        name = "train" if position < TRAIN_FRAC else ("val" if position < TRAIN_FRAC + VAL_FRAC else "test")
        slices[name][0] += rows
        slices[name][1] += benign
        seen += rows

    report[relative] = {
        "rows": total,
        "benign": benign_total,
        "n_minutes": len(minutes),
        "timestamp_min": minutes[0] if minutes else None,
        "timestamp_max": minutes[-1] if minutes else None,
        "timestamp_monotonic_in_row_order": monotonic,
        "temporal_split": {
            name: {"rows": v[0], "benign": v[1]} for name, v in slices.items()
        },
    }

    worst = min(slices["val"][1], slices["test"][1])
    flag = "  <-- SPLIT WOULD LOSE BENIGN" if worst == 0 else ""
    print(
        f"{relative:<30} rows={total:>10,} benign={benign_total:>6,} "
        f"minutes={len(minutes):>4}  mono={str(monotonic):<5} "
        f"benign tr/va/te={slices['train'][1]:>6,}/{slices['val'][1]:>5,}/{slices['test'][1]:>5,}{flag}"
    )

aggregate = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
for entry in report.values():
    for name, counts in entry["temporal_split"].items():
        aggregate[name][0] += counts["rows"]
        aggregate[name][1] += counts["benign"]

print("\nAGGREGATE over all files, within-file temporal 70/15/15:")
grand = sum(v[0] for v in aggregate.values())
for name, (rows, benign) in aggregate.items():
    print(
        f"  {name:<6} rows={rows:>12,} ({rows / grand:>6.2%})  "
        f"benign={benign:>7,} ({benign / rows if rows else 0:.4%} of slice)"
    )

print("\n===== PASTE THIS BACK =====")
print(json.dumps(
    {
        "input_root": str(INPUT_ROOT),
        "resolved_columns": {k: by_canonical.get(k) for k in
                             ("label", "timestamp", "capture_day", "source_file_id")},
        "all_schema_names": schema_names,
        "aggregate_temporal_split": {
            k: {"rows": v[0], "benign": v[1]} for k, v in aggregate.items()
        },
        "per_file": report,
    },
    indent=1, ensure_ascii=False,
))
