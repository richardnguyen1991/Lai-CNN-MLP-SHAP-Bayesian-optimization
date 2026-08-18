# ==========================================================================
# Step 2b -- where does BENIGN actually live?
#
# BENIGN is 0.16% of CIC-DDoS2019, so the group-aware split stands or falls on
# which capture files carry BENIGN at all. This maps that out.
#
# Cheap by construction: it projects three columns out of the Parquet instead of
# re-reading all 91, so it costs a fraction of the full audit.
#
# Paste into a Kaggle cell and run. No internet, no credentials.
# ==========================================================================

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

INPUT_ROOT = Path("/kaggle/input/cicddos2019-parquet")
PROFILE = Path("/kaggle/working/audit_out/data_profile.json")

if not INPUT_ROOT.exists():
    candidates = [
        p for p in sorted(Path("/kaggle/input").glob("*"))
        if p.is_dir() and next(p.rglob("*.parquet"), None) is not None
    ]
    if len(candidates) != 1:
        raise SystemExit(f"cannot pick an input root among {[p.name for p in candidates]}")
    INPUT_ROOT = candidates[0]

shards = sorted(INPUT_ROOT.rglob("*.parquet"))
print(f"{len(shards)} shards under {INPUT_ROOT}\n")


def pick(names, *wanted):
    """Find a column by loose name match, tolerating the leading-space headers."""
    lowered = {n.strip().lower().replace(" ", "_"): n for n in names}
    for candidate in wanted:
        hit = lowered.get(candidate)
        if hit is not None:
            return hit
    return None


schema_names = list(pq.ParquetFile(shards[0]).schema_arrow.names)
label_col = pick(schema_names, "label")
day_col = pick(schema_names, "capture_day")
file_col = pick(schema_names, "source_file_id")
print(f"label column       : {label_col!r}")
print(f"capture_day column : {day_col!r}")
print(f"source_file_id col : {file_col!r}\n")
if label_col is None:
    raise SystemExit(f"no label column in {schema_names}")

projection = [c for c in (label_col, day_col, file_col) if c]

per_file = {}
per_day = defaultdict(Counter)
per_group = defaultdict(Counter)
day_values, file_values = set(), set()

for shard in shards:
    relative = shard.relative_to(INPUT_ROOT).as_posix()
    counts = Counter()
    parquet = pq.ParquetFile(shard)
    for group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(group_index, columns=projection)
        labels = [str(v).strip() for v in table.column(label_col).to_pylist()]
        counts.update(labels)

        days = (
            [str(v) for v in table.column(day_col).to_pylist()]
            if day_col else [None] * len(labels)
        )
        groups = (
            [str(v) for v in table.column(file_col).to_pylist()]
            if file_col else [None] * len(labels)
        )
        for label, day, group in zip(labels, days, groups):
            binary = "BENIGN" if label.upper() in ("BENIGN", "NORMAL") else "ATTACK"
            if day is not None:
                per_day[day][binary] += 1
                day_values.add(day)
            if group is not None:
                per_group[group][binary] += 1
                file_values.add(group)
        del table

    benign = counts.get("BENIGN", 0)
    total = sum(counts.values())
    per_file[relative] = {"rows": total, "benign": benign, "labels": dict(counts)}
    share = benign / total if total else 0.0
    flag = "  <-- NO BENIGN" if benign == 0 else ""
    print(f"{relative:<34} rows={total:>10,}  benign={benign:>7,}  ({share:.4%}){flag}")

print(f"\ncapture_day distinct values   : {len(day_values)} -> {sorted(day_values)[:8]}")
print(f"source_file_id distinct values: {len(file_values)}")

files_without_benign = [k for k, v in per_file.items() if v["benign"] == 0]
print(f"\nfiles with zero BENIGN: {len(files_without_benign)} / {len(per_file)}")
for name in files_without_benign:
    print(f"  {name}")

digest = {
    "n_shards": len(shards),
    "per_file": per_file,
    "per_capture_day": {k: dict(v) for k, v in sorted(per_day.items())},
    "n_distinct_capture_day": len(day_values),
    "n_distinct_source_file_id": len(file_values),
    "per_source_file_id": {k: dict(v) for k, v in sorted(per_group.items())},
    "files_without_benign": files_without_benign,
}
print("\n===== PASTE THIS BACK =====")
print(json.dumps(digest, indent=1, ensure_ascii=False))
