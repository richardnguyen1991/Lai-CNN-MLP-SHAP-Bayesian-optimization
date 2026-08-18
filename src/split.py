"""Deduplicate, split, and subsample -- before any preprocessing is fitted.

Order matters and is fixed: deduplicate -> split -> subsample. Fitting a scaler,
a SHAP selector or a Bayesian search on data that has already been pooled is the
failure mode this whole module exists to prevent.

Produces a single int8 assignment array over the dataset in a canonical row
order, which preprocessing.py then consumes. Nothing here reads feature values
beyond hashing them, so it stays cheap on 70M rows.

    0 = train      1 = val      2 = test
   -1 = dropped as a duplicate
   -2 = dropped by subsampling

Artifacts:
    split_assignment.npy   int8, one entry per row in canonical order
    file_offsets.json      relative path -> [start, end) into that array
    split_manifest.json    strategy, ratios, per-split counts, sampling rates
    leakage_audit.json     duplicate counts, cross-split check, what was fitted

Usage:
    python src/split.py --input-root /kaggle/input/... --out-dir work/config
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from schema import build_column_mapping, find_label_column

TRAIN, VAL, TEST = 0, 1, 2
DROP_DUPLICATE, DROP_SUBSAMPLE = -1, -2
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


# --------------------------------------------------------------------------
# Pass 1 -- collect the little that splitting actually needs
# --------------------------------------------------------------------------

class RowIndex:
    """Per-row metadata for the whole dataset, in canonical (sorted path) order.

    Deliberately narrow: file id, sort key, binary label, raw label id and a row
    hash. At 70M rows this is about 1.3 GiB, versus 21.8 GiB for the features.
    """

    def __init__(self) -> None:
        self.file_id: List[np.ndarray] = []
        self.sort_key: List[np.ndarray] = []
        self.y: List[np.ndarray] = []
        self.raw_label_id: List[np.ndarray] = []
        self.row_hash: List[np.ndarray] = []
        self.files: List[str] = []
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self.raw_labels: List[str] = []
        self._label_ids: Dict[str, int] = {}
        self._cursor = 0

    def label_id(self, label: str) -> int:
        if label not in self._label_ids:
            self._label_ids[label] = len(self.raw_labels)
            self.raw_labels.append(label)
        return self._label_ids[label]

    def add_file(self, relative: str, arrays: Dict[str, np.ndarray]) -> None:
        n = len(arrays["y"])
        index = len(self.files)
        self.files.append(relative)
        self.offsets[relative] = (self._cursor, self._cursor + n)
        self._cursor += n
        self.file_id.append(np.full(n, index, dtype=np.uint8))
        for key in ("sort_key", "y", "raw_label_id", "row_hash"):
            getattr(self, key).append(arrays[key])

    def finalize(self) -> Dict[str, np.ndarray]:
        return {
            "file_id": np.concatenate(self.file_id),
            "sort_key": np.concatenate(self.sort_key),
            "y": np.concatenate(self.y),
            "raw_label_id": np.concatenate(self.raw_label_id),
            "row_hash": np.concatenate(self.row_hash),
        }


def _timestamp_to_int64(values: pd.Series) -> Tuple[np.ndarray, int]:
    """Parse timestamps to microseconds, reporting how many failed."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    unparsed = int(parsed.isna().sum())
    micros = parsed.view("int64") if hasattr(parsed, "view") else parsed.astype("int64")
    micros = np.asarray(micros, dtype=np.int64)
    return micros, unparsed


def _detect_identifier_columns(
    parquet: pq.ParquetFile,
    hash_raw_columns: List[str],
    column_mapping: Dict[str, str],
    sample_rows: int = 20000,
) -> Tuple[List[str], List[str]]:
    """Separate row identifiers from features that merely happen to be unique.

    A per-row identifier inside the row hash makes every hash distinct, so
    deduplication finds nothing while reporting success -- silent, and it passes
    every other check. __source_row_id is exactly that.

    But "unique in this sample" alone is the wrong test. Continuous features are
    routinely unique across 20,000 rows: two flows really do differ in
    flow_iat_std, and a hash that separates them is correct. Failing on those
    would block the run on data that is perfectly fine.

    So a hard failure needs the column to look like an index: a string, or an
    integer whose sorted values step by a constant. Everything else that is
    unique is returned as a note, since it does explain why deduplication finds
    little.
    """
    frame = parquet.read_row_group(0, columns=hash_raw_columns).to_pandas()
    if len(frame) < 1000:
        return [], []
    frame = frame.head(sample_rows)

    identifiers: List[str] = []
    unique_features: List[str] = []
    for raw in hash_raw_columns:
        column = frame[raw]
        if column.nunique(dropna=False) != len(frame):
            continue

        canonical = column_mapping[raw]
        if not pd.api.types.is_numeric_dtype(column):
            identifiers.append(canonical)
            continue
        if pd.api.types.is_integer_dtype(column):
            steps = np.diff(np.sort(column.to_numpy()))
            if steps.size and np.all(steps == steps[0]):
                identifiers.append(canonical)      # an arithmetic run: a row index
                continue
        unique_features.append(canonical)

    return sorted(identifiers), sorted(unique_features)


def scan_dataset(
    input_root: Path,
    dataset_cfg: Dict[str, Any],
    label_decisions: Dict[str, Any],
) -> Tuple[RowIndex, Dict[str, Any]]:
    shards = sorted(input_root.rglob(dataset_cfg.get("input_glob", "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no parquet under {input_root}")

    index = RowIndex()
    diagnostics: Dict[str, Any] = {
        "unparsed_timestamps": 0,
        "rows_with_unmapped_label": 0,
        "used_row_order_for": [],
    }

    first = pq.ParquetFile(shards[0])
    raw_columns = list(first.schema_arrow.names)
    column_mapping, _ = build_column_mapping(raw_columns, dataset_cfg.get("schema_aliases", {}))
    label_raw = find_label_column(raw_columns, dataset_cfg["label_column_candidates"])

    # Hash the behavioural features only. Including flow_id / IPs / timestamp /
    # source_row_id would make every row unique and report zero duplicates,
    # which is exactly the mistake that lets CIC-DDoS2019's repeated flows leak
    # across splits. Driven from config so a new metadata column added upstream
    # cannot quietly re-enable the failure.
    excluded = (
        set(dataset_cfg.get("drop_columns_canonical", []))
        | set(dataset_cfg.get("metadata_columns", []))
        | {"label"}
    )
    hash_raw_columns = [
        raw for raw in raw_columns if column_mapping[raw] not in excluded and raw != label_raw
    ]
    timestamp_raw = next(
        (raw for raw in raw_columns if column_mapping[raw] == "timestamp"), None
    )
    diagnostics["n_hash_columns"] = len(hash_raw_columns)
    diagnostics["hash_columns"] = sorted(column_mapping[r] for r in hash_raw_columns)

    identifiers, unique_features = _detect_identifier_columns(
        first, hash_raw_columns, column_mapping
    )
    diagnostics["unique_per_row_features"] = unique_features
    if identifiers:
        raise ValueError(
            "these columns are row identifiers and would disable deduplication "
            f"if hashed: {identifiers}. Add them to metadata_columns or "
            "drop_columns_canonical in the dataset config."
        )
    if unique_features:
        # Not an error: a genuinely unique feature means the rows differ. Worth
        # recording, because it caps how many duplicates dedup can ever find.
        print(f"  note: {len(unique_features)} feature(s) unique per row in the "
              f"sample, e.g. {unique_features[:3]}")

    for shard in shards:
        relative = shard.relative_to(input_root).as_posix()
        parquet = pq.ParquetFile(shard)
        chunks: Dict[str, List[np.ndarray]] = defaultdict(list)

        for group_index in range(parquet.num_row_groups):
            needed = hash_raw_columns + [label_raw]
            if timestamp_raw and timestamp_raw not in needed:
                needed = needed + [timestamp_raw]
            frame = parquet.read_row_group(group_index, columns=needed).to_pandas()

            labels = frame[label_raw].astype(str).str.strip()
            decisions = labels.map(label_decisions)
            unmapped = int(decisions.isna().sum())
            diagnostics["rows_with_unmapped_label"] += unmapped

            y = np.where(decisions.to_numpy() == 1, 1, 0).astype(np.uint8)
            dropped_by_scope = (decisions.to_numpy() == "dropped") | decisions.isna().to_numpy()
            y = np.where(dropped_by_scope, 255, y).astype(np.uint8)  # 255 = excluded

            chunks["y"].append(y)
            chunks["raw_label_id"].append(
                np.array([index.label_id(v) for v in labels], dtype=np.uint8)
            )

            if timestamp_raw is not None:
                micros, unparsed = _timestamp_to_int64(frame[timestamp_raw])
                diagnostics["unparsed_timestamps"] += unparsed
                chunks["sort_key"].append(micros)
            else:
                chunks["sort_key"].append(np.zeros(len(frame), dtype=np.int64))

            # Vectorised 64-bit row hash; per-row hashlib would not finish on 70M.
            hashed = pd.util.hash_pandas_object(frame[hash_raw_columns], index=False)
            chunks["row_hash"].append(hashed.to_numpy(dtype=np.uint64))
            del frame

        arrays = {key: np.concatenate(values) for key, values in chunks.items()}
        if timestamp_raw is None or not np.any(arrays["sort_key"]):
            arrays["sort_key"] = np.arange(len(arrays["y"]), dtype=np.int64)
            diagnostics["used_row_order_for"].append(relative)
        index.add_file(relative, arrays)

    diagnostics["n_files"] = len(index.files)
    return index, diagnostics


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------

def first_occurrence_mask(row_hash: np.ndarray) -> np.ndarray:
    """Keep the first occurrence of each hash in original order.

    np.unique's return_index is not guaranteed to give the first occurrence,
    because its sort is not stable by default. A stable argsort makes the choice
    deterministic, which matters: the split must be reproducible from the seed.
    """
    order = np.argsort(row_hash, kind="stable")
    ordered = row_hash[order]
    is_first = np.empty(len(order), dtype=bool)
    is_first[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=is_first[1:])
    keep = np.zeros(len(order), dtype=bool)
    keep[order[is_first]] = True
    return keep


# --------------------------------------------------------------------------
# Split assignment
# --------------------------------------------------------------------------

def assign_splits(
    data: Dict[str, np.ndarray],
    eligible: np.ndarray,
    index: RowIndex,
    split_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    strategy = split_cfg["strategy"]
    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["val_ratio"]
    assignment = np.full(len(eligible), DROP_DUPLICATE, dtype=np.int8)

    if strategy == "temporal_stratified_within_file":
        # Primary protocol. Within each (file, raw label) stream, order by time
        # and cut 70/15/15.
        #
        # Plain temporal_within_file cuts each file as a whole, which fails on
        # this dataset: BENIGN is clustered in time rather than spread. In
        # 03-11/Syn.parquet, 34,822 of 35,790 BENIGN rows sit in the final
        # stretch of the capture, and in 01-12/TFTP.parquet 20,866 of 25,247 do
        # the same, so validation is left almost BENIGN-free. Stratifying the
        # cut by label guarantees every family and both classes appear in all
        # three splits in the right proportion.
        #
        # The cost, which belongs in the thesis: test is no longer strictly
        # later in wall-clock time than train across the dataset. Within each
        # (file, label) stream it still is, so a burst cannot straddle a
        # boundary, but the splits do overlap in time across labels.
        for relative in index.files:
            start, end = index.offsets[relative]
            positions = np.flatnonzero(eligible[start:end]) + start
            if positions.size == 0:
                continue
            labels = data["raw_label_id"][positions]
            for label_id in np.unique(labels):
                stream = positions[labels == label_id]
                order = stream[np.argsort(data["sort_key"][stream], kind="stable")]
                _cut_proportionally(order, assignment, train_ratio, val_ratio)

    elif strategy == "temporal_within_file":
        for file_index, relative in enumerate(index.files):
            start, end = index.offsets[relative]
            local = eligible[start:end]
            positions = np.flatnonzero(local) + start
            if positions.size == 0:
                continue
            order = positions[np.argsort(data["sort_key"][positions], kind="stable")]
            n_train = int(round(order.size * train_ratio))
            n_val = int(round(order.size * val_ratio))
            assignment[order[:n_train]] = TRAIN
            assignment[order[n_train:n_train + n_val]] = VAL
            assignment[order[n_train + n_val:]] = TEST

    elif strategy == "by_capture_day":
        # Day 01-12 -> train/val, day 03-11 -> test. Attack families are almost
        # disjoint between days, so this measures generalisation, not the same
        # quantity as the primary protocol.
        for file_index, relative in enumerate(index.files):
            start, end = index.offsets[relative]
            positions = np.flatnonzero(eligible[start:end]) + start
            if positions.size == 0:
                continue
            if relative.split("/")[-2] == "03-11":
                assignment[positions] = TEST
            else:
                order = positions[np.argsort(data["sort_key"][positions], kind="stable")]
                cut = int(round(order.size * (train_ratio / (train_ratio + val_ratio))))
                assignment[order[:cut]] = TRAIN
                assignment[order[cut:]] = VAL

    elif strategy == "group_by_file":
        assignment = _assign_group_by_file(
            data, eligible, index, train_ratio, val_ratio, assignment
        )

    elif strategy == "random_stratified":
        # Control condition only. Never the primary protocol: shuffling rows
        # scatters near-identical flows from one burst across all three splits.
        positions = np.flatnonzero(eligible)
        shuffled = rng.permutation(positions)
        n_train = int(round(shuffled.size * train_ratio))
        n_val = int(round(shuffled.size * val_ratio))
        assignment[shuffled[:n_train]] = TRAIN
        assignment[shuffled[n_train:n_train + n_val]] = VAL
        assignment[shuffled[n_train + n_val:]] = TEST

    else:
        raise ValueError(f"unknown split strategy {strategy!r}")

    return assignment


def upsample_before_split(
    assignment: np.ndarray,
    data: Dict[str, np.ndarray],
    eligible: np.ndarray,
    split_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reproduce the paper's Fig 6 ordering: up-sample, THEN split.

    Section 2.5 of the paper puts up-sampling before feature selection and never
    mentions a split at all. Duplicating BENIGN rows to balance the classes and
    only then dividing the data puts copies of the same record on both sides of
    the boundary, which is the most likely explanation for 99.95% accuracy on a
    dataset that is 0.16% BENIGN.

    Deliberately wrong, and only reachable from the `paperlike` variant. The
    delta against the clean run is the measurement this experiment exists for.

    Rather than materialising duplicate rows, each row carries a bitmask of the
    splits it additionally appears in. preprocessing.py emits it once per split,
    which is the same thing with none of the memory.
    """
    benign = np.flatnonzero(eligible & (data["y"] == 0))
    attack = np.flatnonzero(eligible & (data["y"] == 1))
    if benign.size == 0:
        raise RuntimeError("cannot up-sample: no BENIGN rows")

    ratios = np.array([split_cfg["train_ratio"], split_cfg["val_ratio"],
                       split_cfg["test_ratio"]], dtype=float)
    ratios = ratios / ratios.sum()

    # Balance, as "make a balanced dataset" in section 2.5 describes.
    copies = max(int(round(attack.size / benign.size)), 1)
    also_in = np.zeros(len(assignment), dtype=np.uint8)

    for position in (attack,):
        draws = rng.choice(3, size=position.size, p=ratios)
        assignment[position] = draws.astype(np.int8)

    # Every copy of a BENIGN row is placed independently, so the same record can
    # land in train and in test at once. That is the leak.
    first = rng.choice(3, size=benign.size, p=ratios)
    assignment[benign] = first.astype(np.int8)
    for _ in range(copies - 1):
        extra = rng.choice(3, size=benign.size, p=ratios)
        also_in[benign] |= (1 << extra).astype(np.uint8)

    # A copy landing in the split the row already occupies adds nothing.
    also_in[benign] &= ~(1 << first).astype(np.uint8)

    duplicated = int(np.count_nonzero(also_in))
    return also_in, {
        "applied": True,
        "benign_rows": int(benign.size),
        "attack_rows": int(attack.size),
        "copies_per_benign_row": copies,
        "benign_rows_present_in_more_than_one_split": duplicated,
        "note": ("Reproduces Fig 6 of the paper: up-sample, then split. The "
                 "duplicated BENIGN rows straddle the split boundary on purpose; "
                 "this variant exists to measure how much that inflates the "
                 "reported numbers."),
    }


def _cut_proportionally(
    order: np.ndarray, assignment: np.ndarray, train_ratio: float, val_ratio: float
) -> None:
    """Cut one time-ordered stream into train/val/test in place.

    Rounding alone would erase small strata: at 70/15/15 a 3-row stream gives
    round(0.45) = 0 validation rows. Streams of at least three rows are
    therefore guaranteed one row in each split, which is what keeps rare
    families such as WebDDoS (439 rows dataset-wide) present everywhere.
    """
    size = order.size
    n_train = int(round(size * train_ratio))
    n_val = int(round(size * val_ratio))

    if size >= 3:
        n_train = min(max(n_train, 1), size - 2)
        n_val = min(max(n_val, 1), size - n_train - 1)
    else:
        n_train, n_val = size, 0        # too small to divide meaningfully

    assignment[order[:n_train]] = TRAIN
    assignment[order[n_train:n_train + n_val]] = VAL
    assignment[order[n_train + n_val:]] = TEST


def _assign_group_by_file(
    data: Dict[str, np.ndarray],
    eligible: np.ndarray,
    index: RowIndex,
    train_ratio: float,
    val_ratio: float,
    assignment: np.ndarray,
) -> np.ndarray:
    """Whole files to splits, largest first, filling whichever split is furthest
    below its target share. Greedy, but deterministic and it keeps every file
    intact so no burst can straddle a boundary."""
    sizes = []
    for relative in index.files:
        start, end = index.offsets[relative]
        sizes.append((int(eligible[start:end].sum()), relative))
    sizes.sort(reverse=True)
    total = sum(size for size, _ in sizes)

    targets = {TRAIN: train_ratio * total, VAL: val_ratio * total,
               TEST: (1 - train_ratio - val_ratio) * total}
    current = {TRAIN: 0, VAL: 0, TEST: 0}
    for size, relative in sizes:
        pick = max(targets, key=lambda s: targets[s] - current[s])
        start, end = index.offsets[relative]
        positions = np.flatnonzero(eligible[start:end]) + start
        assignment[positions] = pick
        current[pick] += size
    return assignment


# --------------------------------------------------------------------------
# Subsampling (B2: keep every BENIGN, thin the attacks)
# --------------------------------------------------------------------------

def subsample(
    assignment: np.ndarray,
    data: Dict[str, np.ndarray],
    index: RowIndex,
    subsample_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if subsample_cfg.get("policy") != "keep_all_benign":
        return {"applied": False}

    ratio = subsample_cfg["attack_per_benign"]
    apply_to = {name: code for code, name in SPLIT_NAMES.items()
                if name in subsample_cfg.get("apply_to", [])}
    rates: Dict[str, Any] = {}

    for name, code in apply_to.items():
        in_split = assignment == code
        benign = int(np.count_nonzero(in_split & (data["y"] == 0)))
        budget = benign * ratio

        attack_positions = np.flatnonzero(in_split & (data["y"] == 1))
        if attack_positions.size <= budget:
            # Already below budget. Same key set as the sampled branch, so
            # downstream readers never have to special-case this.
            rates[name] = {
                "subsampled": False,
                "benign_kept": benign,
                "attacks_before": int(attack_positions.size),
                "attacks_kept": int(attack_positions.size),
                "rate": 1.0,
                "per_stratum": {},
            }
            continue

        # Stratify by (file, raw label) so no attack family is wiped out and the
        # family mix of the split is preserved.
        strata = defaultdict(list)
        for position in attack_positions:
            strata[(int(data["file_id"][position]), int(data["raw_label_id"][position]))].append(position)

        keep_rate = budget / attack_positions.size
        kept_total = 0
        per_stratum = {}
        for key, positions in sorted(strata.items()):
            array = np.asarray(positions)
            # At least one row per stratum, so tiny families (WebDDoS has 439
            # rows in the whole dataset) are not rounded out of existence.
            n_keep = max(1, int(round(array.size * keep_rate)))
            chosen = rng.choice(array, size=min(n_keep, array.size), replace=False)
            dropped = np.setdiff1d(array, chosen, assume_unique=False)
            assignment[dropped] = DROP_SUBSAMPLE
            kept_total += len(chosen)
            per_stratum[f"{index.files[key[0]]}|{index.raw_labels[key[1]]}"] = {
                "before": int(array.size), "kept": int(len(chosen)),
            }

        rates[name] = {
            "subsampled": True,
            "benign_kept": benign,
            "attacks_before": int(attack_positions.size),
            "attacks_kept": int(kept_total),
            "rate": round(kept_total / attack_positions.size, 6),
            "per_stratum": per_stratum,
        }

    return {"applied": True, "attack_per_benign": ratio,
            "not_applied_to": [n for n in SPLIT_NAMES.values() if n not in apply_to],
            "splits": rates}


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def audit_leakage(
    assignment: np.ndarray,
    data: Dict[str, np.ndarray],
    duplicates_removed: int,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-check on the FINAL selection that no hash appears in two splits."""
    selected = assignment >= 0
    hashes = data["row_hash"][selected]
    splits = assignment[selected]

    order = np.argsort(hashes, kind="stable")
    ordered_hash = hashes[order]
    ordered_split = splits[order]
    boundary = np.flatnonzero(ordered_hash[1:] != ordered_hash[:-1])
    starts = np.concatenate(([0], boundary + 1))
    ends = np.concatenate((boundary + 1, [len(ordered_hash)]))

    cross = 0
    for start, end in zip(starts, ends):
        if end - start > 1 and len(np.unique(ordered_split[start:end])) > 1:
            cross += int(end - start)

    return {
        "row_hash_bits": 64,
        "hash_columns_used": diagnostics["n_hash_columns"],
        "hash_excludes": ["flow_id", "source_ip", "destination_ip", "timestamp", "label"],
        "expected_false_collisions": round(len(hashes) ** 2 / 2 ** 65, 6),
        "duplicates_removed_before_split": duplicates_removed,
        "cross_split_duplicate_rows": cross,
        "fitted_on_train_only": {
            "scaler": "pending preprocessing.py",
            "imputer": "pending preprocessing.py",
            "shap_selector": "pending shap_selection.py",
            "bayesian_optimization": "validation objective only",
        },
        "test_touched": "not yet",
    }


def summarise(assignment: np.ndarray, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    summary = {}
    for code, name in SPLIT_NAMES.items():
        mask = assignment == code
        rows = int(np.count_nonzero(mask))
        benign = int(np.count_nonzero(mask & (data["y"] == 0)))
        summary[name] = {
            "rows": rows,
            "benign": benign,
            "attack": rows - benign,
            "benign_ratio": round(benign / rows, 8) if rows else 0.0,
            "attack_per_benign": round((rows - benign) / benign, 2) if benign else None,
        }
    summary["dropped_duplicate"] = int(np.count_nonzero(assignment == DROP_DUPLICATE))
    summary["dropped_subsample"] = int(np.count_nonzero(assignment == DROP_SUBSAMPLE))
    return summary


def family_coverage(
    assignment: np.ndarray, data: Dict[str, np.ndarray], index: RowIndex
) -> Tuple[Dict[str, Any], List[str]]:
    """How each raw attack family lands across the splits.

    A temporal split cuts each capture by time, so a family that only appears
    late in its file ends up entirely in test and the model never trains on it.
    WebDDoS is the live risk here: 439 rows in the whole dataset, all inside
    01-12/UDPLag.parquet. Worth a warning rather than a failure, since
    by_capture_day makes unseen test families the point of the experiment.
    """
    coverage: Dict[str, Any] = {}
    warnings: List[str] = []
    for label_index, label in enumerate(index.raw_labels):
        mask = data["raw_label_id"] == label_index
        total = int(np.count_nonzero(mask))
        if total == 0:
            continue
        counts = {
            name: int(np.count_nonzero(mask & (assignment == code)))
            for code, name in SPLIT_NAMES.items()
        }
        counts["dropped"] = int(np.count_nonzero(mask & (assignment < 0)))
        counts["total"] = total
        coverage[label] = counts
        if counts["train"] == 0 and counts["test"] > 0:
            warnings.append(
                f"family {label!r} appears in test ({counts['test']:,} rows) but "
                f"never in train; it is time-clustered within its capture file"
            )
        elif counts["train"] == 0 and counts["val"] > 0:
            warnings.append(f"family {label!r} is absent from train but present in val")
    return coverage, warnings


def check_split_health(summary: Dict[str, Any], split_cfg: Dict[str, Any]) -> List[str]:
    failures = []
    if not split_cfg.get("require_both_classes_per_split", True):
        return failures
    minimum = split_cfg.get("min_benign_per_split", 1)
    for name in ("train", "val", "test"):
        stats = summary[name]
        if stats["rows"] == 0:
            failures.append(f"{name} split is empty")
        elif stats["benign"] == 0:
            failures.append(
                f"{name} split has no BENIGN; switch split.strategy "
                "(temporal_within_file -> group_by_file) rather than proceeding"
            )
        elif stats["benign"] < minimum:
            failures.append(
                f"{name} split has only {stats['benign']} BENIGN, below "
                f"min_benign_per_split={minimum}"
            )
    return failures


# --------------------------------------------------------------------------

def run(
    input_root: Path,
    out_dir: Path,
    repo_root: Path,
    experiment_config: Optional[Path] = None,
) -> int:
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    dataset_name = experiment["run"]["dataset"]
    with (repo_root / "configs" / f"dataset_{dataset_name}.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    from data_audit import build_label_mapping
    label_mapping = build_label_mapping(dataset_cfg)
    rng = np.random.default_rng(experiment["run"]["seed"])

    print("scanning...")
    index, diagnostics = scan_dataset(input_root, dataset_cfg, label_mapping["decisions"])
    data = index.finalize()
    total = len(data["y"])
    print(f"  {total:,} rows across {len(index.files)} files")

    in_scope = data["y"] != 255
    print(f"  {int(np.count_nonzero(~in_scope)):,} rows outside attack_scope")

    if experiment["split"].get("deduplicate_before_split", True):
        keep = first_occurrence_mask(data["row_hash"])
        duplicates_removed = int(np.count_nonzero(~keep))
        print(f"  {duplicates_removed:,} duplicate rows removed before split "
              f"({duplicates_removed / total:.2%})")
    else:
        keep = np.ones(total, dtype=bool)
        duplicates_removed = 0

    eligible = keep & in_scope
    paperlike = experiment.get("leakage", {}).get("upsample_before_split", False)

    if paperlike:
        assignment = np.full(len(eligible), DROP_DUPLICATE, dtype=np.int8)
        also_in, upsampling = upsample_before_split(
            assignment, data, eligible, experiment["split"], rng
        )
        assignment[~eligible] = DROP_DUPLICATE
        sampling = {"applied": False, "reason": "paperlike up-samples instead"}
    else:
        assignment = assign_splits(data, eligible, index, experiment["split"], rng)
        assignment[~eligible] = DROP_DUPLICATE
        also_in = np.zeros(len(assignment), dtype=np.uint8)
        upsampling = {"applied": False}
        sampling = subsample(assignment, data, index, experiment["subsample"], rng)
    summary = summarise(assignment, data)
    coverage, warnings = family_coverage(assignment, data, index)
    failures = check_split_health(summary, experiment["split"])
    leakage = audit_leakage(assignment, data, duplicates_removed, diagnostics)
    leakage["upsample_before_split"] = paperlike
    if paperlike:
        leakage["deliberate_leak"] = (
            f"{upsampling['benign_rows_present_in_more_than_one_split']:,} BENIGN "
            "rows appear in more than one split by design, reproducing Fig 6 of "
            "the paper. Do not read this run's metrics as a clean result."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "split_assignment.npy", assignment)
    # Bitmask of the EXTRA splits a row also belongs to. All zeros for every
    # variant except paperlike, so downstream code has one shape to handle.
    np.save(out_dir / "split_also_in.npy", also_in)
    _write(out_dir / "file_offsets.json",
           {name: list(span) for name, span in index.offsets.items()})
    _write(out_dir / "split_manifest.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config": str(config_path),
        "dataset": dataset_name,
        "seed": experiment["run"]["seed"],
        "strategy": experiment["split"]["strategy"],
        "ratios": {k: experiment["split"][k] for k in ("train_ratio", "val_ratio", "test_ratio")},
        "total_rows": total,
        "rows_outside_scope": int(np.count_nonzero(~in_scope)),
        "summary": summary,
        "subsampling": sampling,
        "upsampling_before_split": upsampling,
        "family_coverage": coverage,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "checks": {"passed": not failures, "failures": failures},
    })
    _write(out_dir / "leakage_audit.json", leakage)

    _report(summary, sampling, leakage, warnings, failures)
    return 2 if failures else 0


def _write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _report(summary, sampling, leakage, warnings, failures) -> None:
    print("\nsplit:")
    for name in ("train", "val", "test"):
        stats = summary[name]
        ratio = f"1:{stats['attack_per_benign']:,.0f}" if stats["attack_per_benign"] else "n/a"
        print(f"  {name:<6} rows={stats['rows']:>11,}  benign={stats['benign']:>7,}  "
              f"attack/benign={ratio}")
    print(f"  dropped: {summary['dropped_duplicate']:,} duplicate, "
          f"{summary['dropped_subsample']:,} subsampled")
    if sampling.get("applied"):
        print(f"  subsample kept test at the natural prior: "
              f"{sampling['not_applied_to']}")
    print(f"\ncross-split duplicate rows: {leakage['cross_split_duplicate_rows']}")
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  ! {warning}")
    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"  - {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--experiment-config", type=Path, default=None,
        help="alternative experiment.yaml, e.g. configs/variants/paperlike.yaml",
    )
    args = parser.parse_args()
    return run(args.input_root, args.out_dir, args.repo_root, args.experiment_config)


if __name__ == "__main__":
    raise SystemExit(main())
