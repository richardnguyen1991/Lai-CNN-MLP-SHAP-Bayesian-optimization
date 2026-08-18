"""Load the preprocessed cache, sliced to the SHAP-selected columns.

The cache holds every feature that survived preprocessing; the model only takes
the selected ones. Slicing happens here, by the column indices recorded in
selected_features.json, so no other module has to know the mapping.

Train and val are small enough to hold in RAM after subsampling (decision B2)
and are loaded whole. Test keeps the natural 1:618 prior and is iterated shard
by shard instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch


class SelectedFeatures:
    """The column selection plus the hash that pins its identity."""

    def __init__(self, cache_dir: Path) -> None:
        with (cache_dir / "selected_features.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.names: List[str] = payload["selected_features"]
        self.indices = np.asarray(payload["column_index_in_cache"], dtype=np.int64)
        self.schema_hash: str = payload["feature_schema_hash"]
        if len(self.names) != len(self.indices):
            raise RuntimeError("selected_features.json has mismatched names and indices")

    def __len__(self) -> int:
        return len(self.names)


def _shard_paths(cache_dir: Path, split: str) -> List[Path]:
    paths = sorted((cache_dir / "preprocess").glob(f"{split}_X_shard*.npy"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards under {cache_dir / 'preprocess'}")
    return paths


def load_split(
    cache_dir: Path, split: str, features: SelectedFeatures
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whole split as float32 tensors, sliced to the selected columns."""
    blocks = [np.load(path, mmap_mode="r")[:, features.indices] for path in
              _shard_paths(cache_dir, split)]
    X = np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)
    y = np.load(cache_dir / "preprocess" / f"{split}_y.npy")
    if len(X) != len(y):
        raise RuntimeError(f"{split}: {len(X)} rows but {len(y)} labels")
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))


def iter_split_batches(
    cache_dir: Path, split: str, features: SelectedFeatures, batch_size: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Stream a split shard by shard, for anything too large to materialise."""
    y_all = np.load(cache_dir / "preprocess" / f"{split}_y.npy")
    cursor = 0
    for path in _shard_paths(cache_dir, split):
        block = np.load(path, mmap_mode="r")
        for begin in range(0, len(block), batch_size):
            chunk = np.ascontiguousarray(
                block[begin:begin + batch_size][:, features.indices], dtype=np.float32
            )
            labels = y_all[cursor:cursor + len(chunk)].astype(np.float32)
            cursor += len(chunk)
            yield torch.from_numpy(chunk), torch.from_numpy(labels)
    if cursor != len(y_all):
        raise RuntimeError(f"{split}: streamed {cursor} rows but have {len(y_all)} labels")


def epoch_batches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    shuffle: bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Shuffled batches from an in-memory split.

    A plain permutation rather than a DataLoader: the whole split already lives
    in RAM, so worker processes would only add overhead, and one Generator is
    far easier to checkpoint and restore exactly than a sampler's internal state.
    """
    n = len(X)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for begin in range(0, n, batch_size):
        index = order[begin:begin + batch_size]
        yield X[index], y[index]
