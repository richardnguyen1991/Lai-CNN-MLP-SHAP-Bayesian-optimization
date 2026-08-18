"""Impute, reject and scale -- fitted on the training split only.

Consumes split_assignment.npy from split.py and produces the float32 shards the
training loop memory-maps. Every statistic (median, mean, scale) and every
column rejection is derived from TRAIN rows alone; val and test are only ever
transformed.

The training split is subsampled (decision B2) so it is held in memory to fit
exactly. Val and test are streamed shard by shard, because test keeps the
natural 1:618 prior and is far too large to materialise.

Artifacts, under --out-dir:
    cache/preprocess/{train,val,test}_X_shard*.npy
    cache/preprocess/{train,val,test}_y.npy
    cache/scaler.joblib
    config/preprocessing.json

Usage:
    python src/preprocessing.py --input-root ... --split-dir ... --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from schema import build_column_mapping, find_label_column

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def rows_for_split(assignment: np.ndarray, also_in: np.ndarray, code: int) -> np.ndarray:
    """Rows belonging to one split, including up-sampled copies.

    also_in is all zeros for every variant except paperlike, where a BENIGN row
    duplicated before the split legitimately belongs to more than one side. That
    is the leak the variant exists to measure, so it must survive into the cache
    rather than being silently de-duplicated here.
    """
    return (assignment == code) | ((also_in >> code) & 1).astype(bool)


class ShardWriter:
    """Buffers rows and flushes fixed-size .npy shards."""

    def __init__(self, directory: Path, prefix: str, n_features: int,
                 shard_rows: int, max_shards: int) -> None:
        self.directory = directory
        self.prefix = prefix
        self.n_features = n_features
        self.shard_rows = shard_rows
        self.max_shards = max_shards
        self.buffer: List[np.ndarray] = []
        self.buffered = 0
        self.shards: List[Dict[str, Any]] = []

    def add(self, block: np.ndarray) -> None:
        self.buffer.append(block)
        self.buffered += len(block)
        while self.buffered >= self.shard_rows:
            merged = np.concatenate(self.buffer)
            self._flush(merged[: self.shard_rows])
            remainder = merged[self.shard_rows:]
            self.buffer = [remainder] if len(remainder) else []
            self.buffered = len(remainder)

    def close(self) -> List[Dict[str, Any]]:
        if self.buffered:
            self._flush(np.concatenate(self.buffer))
            self.buffer, self.buffered = [], 0
        return self.shards

    def _flush(self, block: np.ndarray) -> None:
        if len(self.shards) >= self.max_shards:
            raise RuntimeError(
                f"{self.prefix} exceeded max_shards_per_split={self.max_shards}; "
                "raise shard_target_mb or max_shards_per_split"
            )
        name = f"{self.prefix}_X_shard{len(self.shards):03d}.npy"
        path = self.directory / name
        np.save(path, np.ascontiguousarray(block))
        self.shards.append({"file": name, "rows": int(len(block))})


def resolve_feature_columns(
    schema: "pa.Schema", dataset_cfg: Dict[str, Any], label_raw: str
) -> Tuple[List[str], Dict[str, str], List[str], List[str]]:
    """Raw column names that are candidate features, plus what was excluded.

    Non-numeric columns are filtered on the Arrow schema rather than discovered
    when a cast blows up mid-run: SimillarHTTP is a string in every shard, and a
    2 GiB read is a bad place to learn that.
    """
    raw_columns = list(schema.names)
    column_mapping, _ = build_column_mapping(raw_columns, dataset_cfg.get("schema_aliases", {}))
    excluded = (
        set(dataset_cfg.get("drop_columns_canonical", []))
        | set(dataset_cfg.get("metadata_columns", []))
        | {"label"}
    )

    numeric_raw = {
        name for name, arrow_type in zip(raw_columns, schema.types)
        if pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_boolean(arrow_type)
    }

    features, removed, non_numeric = [], [], []
    for raw in raw_columns:
        canonical = column_mapping[raw]
        if raw == label_raw or canonical in excluded:
            if raw != label_raw:
                removed.append(canonical)
            continue
        if raw in numeric_raw:
            features.append(raw)
        else:
            non_numeric.append(canonical)

    return features, column_mapping, sorted(removed), sorted(non_numeric)


def _to_float32_matrix(frame, columns: List[str]) -> np.ndarray:
    """Coerce to float32 and turn +/-Inf into NaN so imputation sees it."""
    matrix = frame[columns].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix.astype(np.float32, copy=False)


def collect_train(
    input_root: Path,
    dataset_cfg: Dict[str, Any],
    assignment: np.ndarray,
    offsets: Dict[str, List[int]],
    label_decisions: Dict[str, Any],
    also_in: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, str], List[str]]:
    shards = sorted(input_root.rglob(dataset_cfg.get("input_glob", "*.parquet")))
    schema = pq.ParquetFile(shards[0]).schema_arrow
    label_raw = find_label_column(list(schema.names), dataset_cfg["label_column_candidates"])
    features, column_mapping, removed, non_numeric = resolve_feature_columns(
        schema, dataset_cfg, label_raw
    )

    train_mask = rows_for_split(assignment, also_in, TRAIN)
    n_train = int(np.count_nonzero(train_mask))
    X = np.empty((n_train, len(features)), dtype=np.float32)
    y = np.empty(n_train, dtype=np.int8)
    cursor = 0

    for shard in shards:
        relative = shard.relative_to(input_root).as_posix()
        start, _ = offsets[relative]
        parquet = pq.ParquetFile(shard)
        position = start
        for group_index in range(parquet.num_row_groups):
            frame = parquet.read_row_group(
                group_index, columns=features + [label_raw]
            ).to_pandas()
            local = train_mask[position:position + len(frame)]
            position += len(frame)

            selected = local
            if not selected.any():
                continue
            block = _to_float32_matrix(frame.loc[selected], features)
            labels = frame.loc[selected, label_raw].astype(str).str.strip()
            X[cursor:cursor + len(block)] = block
            y[cursor:cursor + len(block)] = (
                labels.map(label_decisions).to_numpy() == 1
            ).astype(np.int8)
            cursor += len(block)
            del frame

    if cursor != n_train:
        raise RuntimeError(f"collected {cursor} train rows, expected {n_train}")
    return X, y, features, column_mapping, removed, non_numeric


def fit_preprocessor(
    X: np.ndarray, canonical: List[str], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Decide which columns survive and fit the imputer and scaler on train."""
    n_rows = len(X)
    missing = np.isnan(X).sum(axis=0)
    missing_ratio = missing / max(n_rows, 1)

    # An all-NaN column makes nanmedian warn and return NaN. It is rejected
    # below by max_nan_ratio anyway, but a NaN left among the fill values would
    # be written into every transformed row of a column we thought we handled.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)

    filled = np.where(np.isnan(X), medians, X)
    variance = filled.var(axis=0)

    reject_nan = missing_ratio > cfg["max_nan_ratio"]
    reject_var = (variance == 0) if cfg.get("drop_zero_variance", True) else np.zeros(len(canonical), bool)
    keep = ~(reject_nan | reject_var)

    if not keep.any():
        raise RuntimeError("every candidate feature was rejected")

    kept_names = [name for name, flag in zip(canonical, keep) if flag]
    if cfg.get("scaler") == "quantile":
        scaler = QuantileTransformer(
            output_distribution="normal", subsample=200_000, random_state=0
        )
    else:
        scaler = StandardScaler()
    scaler.fit(filled[:, keep])

    return {
        "keep_mask": keep,
        "kept_names": kept_names,
        "medians": medians,
        "scaler": scaler,
        "rejected": {
            "high_missing": [
                {"feature": name, "missing_ratio": round(float(ratio), 6)}
                for name, ratio, flag in zip(canonical, missing_ratio, reject_nan) if flag
            ],
            "zero_variance": [
                name for name, flag in zip(canonical, reject_var) if flag and not reject_nan[list(canonical).index(name)]
            ],
        },
        "missing_ratio": {
            name: round(float(ratio), 6)
            for name, ratio in zip(canonical, missing_ratio) if ratio > 0
        },
    }


def apply_preprocessor(block: np.ndarray, fitted: Dict[str, Any]) -> np.ndarray:
    filled = np.where(np.isnan(block), fitted["medians"], block)
    return fitted["scaler"].transform(filled[:, fitted["keep_mask"]]).astype(np.float32)


def transform_splits(
    input_root: Path,
    dataset_cfg: Dict[str, Any],
    assignment: np.ndarray,
    offsets: Dict[str, List[int]],
    label_decisions: Dict[str, Any],
    features: List[str],
    fitted: Dict[str, Any],
    cache_dir: Path,
    cfg: Dict[str, Any],
    train_X: np.ndarray,
    train_y: np.ndarray,
    also_in: np.ndarray,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_kept = int(fitted["keep_mask"].sum())
    shard_rows = max(1, int(cfg["shard_target_mb"] * 2 ** 20 / (n_kept * 4)))
    max_shards = cfg["max_shards_per_split"]

    manifest: Dict[str, Any] = {"shard_rows": shard_rows, "n_features": n_kept}

    # Train is already in memory and already fitted on; transform it in place.
    writer = ShardWriter(cache_dir, "train", n_kept, shard_rows, max_shards)
    for begin in range(0, len(train_X), shard_rows):
        writer.add(apply_preprocessor(train_X[begin:begin + shard_rows], fitted))
    manifest["train"] = {"shards": writer.close(), "rows": int(len(train_X))}
    np.save(cache_dir / "train_y.npy", train_y)

    shards = sorted(input_root.rglob(dataset_cfg.get("input_glob", "*.parquet")))
    raw_columns = list(pq.ParquetFile(shards[0]).schema_arrow.names)
    label_raw = find_label_column(raw_columns, dataset_cfg["label_column_candidates"])

    for code in (VAL, TEST):
        name = SPLIT_NAMES[code]
        split_mask = rows_for_split(assignment, also_in, code)
        writer = ShardWriter(cache_dir, name, n_kept, shard_rows, max_shards)
        labels_out: List[np.ndarray] = []

        for shard in shards:
            relative = shard.relative_to(input_root).as_posix()
            position = offsets[relative][0]
            parquet = pq.ParquetFile(shard)
            for group_index in range(parquet.num_row_groups):
                frame = parquet.read_row_group(
                    group_index, columns=features + [label_raw]
                ).to_pandas()
                local = split_mask[position:position + len(frame)]
                position += len(frame)

                selected = local
                if not selected.any():
                    continue
                block = _to_float32_matrix(frame.loc[selected], features)
                transformed = apply_preprocessor(block, fitted)
                if cfg.get("assert_finite_after_transform", True) and not np.isfinite(transformed).all():
                    raise RuntimeError(f"non-finite values survived transform in {relative}")
                writer.add(transformed)
                labels_out.append(
                    (frame.loc[selected, label_raw].astype(str).str.strip()
                     .map(label_decisions).to_numpy() == 1).astype(np.int8)
                )
                del frame

        y = np.concatenate(labels_out) if labels_out else np.empty(0, dtype=np.int8)
        np.save(cache_dir / f"{name}_y.npy", y)
        manifest[name] = {"shards": writer.close(), "rows": int(len(y))}

    return manifest


def run(input_root: Path, split_dir: Path, out_dir: Path, repo_root: Path,
        experiment_config: Optional[Path] = None) -> int:
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with (repo_root / "configs" / "preprocessing.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dataset_name = experiment["run"]["dataset"]
    with (repo_root / "configs" / f"dataset_{dataset_name}.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    from data_audit import build_label_mapping
    decisions = build_label_mapping(dataset_cfg)["decisions"]

    assignment = np.load(split_dir / "split_assignment.npy")
    also_in_path = split_dir / "split_also_in.npy"
    also_in = (np.load(also_in_path) if also_in_path.exists()
               else np.zeros(len(assignment), dtype=np.uint8))
    with (split_dir / "file_offsets.json").open(encoding="utf-8") as handle:
        offsets = json.load(handle)

    print("collecting train rows...")
    train_X, train_y, features, column_mapping, removed, non_numeric = collect_train(
        input_root, dataset_cfg, assignment, offsets, decisions, also_in
    )
    canonical = [column_mapping[raw] for raw in features]
    print(f"  train {train_X.shape[0]:,} rows x {train_X.shape[1]} candidate features "
          f"({train_X.nbytes / 2**20:.0f} MiB)")

    if non_numeric:
        print(f"  skipped {len(non_numeric)} non-numeric columns: {non_numeric}")
    fitted = fit_preprocessor(train_X, canonical, cfg)
    print(f"  kept {len(fitted['kept_names'])} of {len(canonical)} features")

    cache_dir = out_dir / "cache" / "preprocess"
    manifest = transform_splits(
        input_root, dataset_cfg, assignment, offsets, decisions,
        features, fitted, cache_dir, cfg, train_X, train_y, also_in,
    )

    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": fitted["scaler"], "medians": fitted["medians"],
         "kept_names": fitted["kept_names"]},
        out_dir / "cache" / "scaler.joblib",
    )

    config_dir = out_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    scaler = fitted["scaler"]
    with (config_dir / "preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "fitted_on": "train_only",
            "n_train_rows_used_for_fitting": int(len(train_X)),
            "candidate_features": canonical,
            "kept_features": fitted["kept_names"],
            "n_kept": len(fitted["kept_names"]),
            "removed_by_config": removed,
            "rejected": fitted["rejected"],
            "missing_ratio_nonzero": fitted["missing_ratio"],
            "impute_strategy": cfg["impute_strategy"],
            "medians": {n: float(v) for n, v in zip(canonical, fitted["medians"])},
            "scaler": cfg["scaler"],
            "scaler_mean": [float(v) for v in getattr(scaler, "mean_", [])],
            "scaler_scale": [float(v) for v in getattr(scaler, "scale_", [])],
            "shards": manifest,
            "skipped_non_numeric": non_numeric,
        }, handle, indent=2, ensure_ascii=False)

    _report(fitted, manifest)
    return 0


def _report(fitted: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    print(f"\nfeatures kept: {len(fitted['kept_names'])}")
    for reason, items in fitted["rejected"].items():
        if items:
            print(f"  rejected ({reason}): {items}")
    print("\ncache:")
    for name in ("train", "val", "test"):
        entry = manifest[name]
        print(f"  {name:<6} rows={entry['rows']:>11,}  shards={len(entry['shards'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--experiment-config", type=Path, default=None)
    args = parser.parse_args()
    return run(args.input_root, args.split_dir, args.out_dir,
               args.repo_root, args.experiment_config)


if __name__ == "__main__":
    raise SystemExit(main())
