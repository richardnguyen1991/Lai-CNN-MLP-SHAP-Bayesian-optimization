"""Checkpointing built for a session that can be cancelled at any moment.

A Kaggle CPU session lasts at most 12 hours and may end sooner without warning.
Everything here exists so that the next session continues from the epoch after
the last completed one, and so that a half-written file can never be mistaken
for a good checkpoint.

Two properties are worth stating because they are easy to get wrong:

  Writes are atomic. Every file goes to a temporary name, is read back and
  size-checked, and only then replaces the live one. A session killed mid-write
  leaves the previous checkpoint intact.

  Resume is refused on any hash mismatch. Continuing a run whose config,
  feature set, split or preprocessing has changed would silently blend two
  different experiments into one history.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PHASES = ("shap", "bo", "final_train", "evaluate", "done")

STATUS_RUNNING = "RUNNING"
STATUS_RESUME_REQUIRED = "RESUME_REQUIRED"
STATUS_DONE = "DONE"


def config_hash(payload: Any) -> str:
    """Stable hash of a config tree, used to detect a changed experiment."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class TrainingState:
    run_id: str
    session_id: str
    dataset_name: str
    phase: str = "final_train"
    current_epoch: int = 0
    total_epochs: int = 100
    bo_trials_done: int = 0
    bo_total_trials: int = 20
    config_hash: str = ""
    feature_schema_hash: str = ""
    preprocess_hash: str = ""
    split_hash: str = ""
    # Recorded to draw the learning curve, never to choose the final model.
    best_val_metric: Optional[float] = None
    best_val_epoch: Optional[int] = None
    last_update_utc: str = ""
    status: str = STATUS_RUNNING
    sessions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase!r}; expected one of {PHASES}")

    def touch(self) -> None:
        self.last_update_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        self.touch()
        return asdict(self)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write via a temporary file, verify the size, then swap it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    written = temporary.stat().st_size
    if written != len(payload):
        temporary.unlink(missing_ok=True)
        raise IOError(f"{path.name}: wrote {written} bytes, expected {len(payload)}")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    )


def atomic_save_torch(path: Path, payload: Any) -> None:
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def capture_rng_state(generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """Everything needed to make the next epoch identical to an uninterrupted run."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "batch_generator": generator.get_state() if generator is not None else None,
    }


def restore_rng_state(state: Dict[str, Any],
                      generator: Optional[torch.Generator] = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if generator is not None and state.get("batch_generator") is not None:
        generator.set_state(state["batch_generator"])


class CheckpointManager:
    """Owns the checkpoint directory and the resume contract."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.directory / "training_state.json"

    @property
    def history_path(self) -> Path:
        return self.directory / "history.json"

    def model_path(self, name: str = "model_last.pt") -> Path:
        return self.directory / name

    # -- write ------------------------------------------------------------
    def save(
        self,
        state: TrainingState,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        history: List[Dict[str, Any]],
        generator: Optional[torch.Generator] = None,
        extra_model_name: Optional[str] = None,
    ) -> None:
        atomic_save_torch(self.model_path(), model.state_dict())
        atomic_save_torch(self.directory / "optimizer_last.pt", optimizer.state_dict())
        atomic_save_torch(self.directory / "rng_state.pt", capture_rng_state(generator))
        if extra_model_name:
            atomic_save_torch(self.model_path(extra_model_name), model.state_dict())
        # History and state go last: a reader that sees the new epoch number is
        # then guaranteed the weights for it are already on disk.
        atomic_write_json(self.history_path, history)
        atomic_write_json(self.state_path, state.to_dict())

    # -- read -------------------------------------------------------------
    def load_state(self) -> Optional[TrainingState]:
        if not self.state_path.exists():
            return None
        with self.state_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return TrainingState(**payload)

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        with self.history_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        model.load_state_dict(torch.load(self.model_path(), weights_only=True))
        optimizer.load_state_dict(
            torch.load(self.directory / "optimizer_last.pt", weights_only=True)
        )
        rng_path = self.directory / "rng_state.pt"
        if rng_path.exists():
            restore_rng_state(torch.load(rng_path, weights_only=False), generator)

    # -- verify -----------------------------------------------------------
    def verify_resumable(self, state: TrainingState, expected: Dict[str, str]) -> None:
        """Refuse to resume a run that is not the same experiment.

        Silently starting over, or worse continuing with a different feature set,
        would produce a history that looks complete and means nothing.
        """
        mismatches = [
            f"{name}: checkpoint has {getattr(state, name)!r}, expected {value!r}"
            for name, value in expected.items()
            if getattr(state, name) and getattr(state, name) != value
        ]
        if mismatches:
            raise RuntimeError(
                "refusing to resume, the experiment has changed:\n  "
                + "\n  ".join(mismatches)
            )


def validate_history(history: List[Dict[str, Any]], total_epochs: int) -> List[str]:
    """Every epoch from 1 to total_epochs, exactly once, in order.

    Resuming from a checkpoint written mid-epoch, or restarting a session
    without truncating, both show up here as a duplicate or a gap.
    """
    problems: List[str] = []
    epochs = [entry["epoch"] for entry in history]

    duplicates = sorted({e for e in epochs if epochs.count(e) > 1})
    if duplicates:
        problems.append(f"epochs recorded more than once: {duplicates}")
    if epochs != sorted(epochs):
        problems.append("epochs are not in ascending order")

    missing = sorted(set(range(1, total_epochs + 1)) - set(epochs))
    if missing:
        problems.append(f"missing epochs: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    unexpected = sorted(e for e in set(epochs) if not 1 <= e <= total_epochs)
    if unexpected:
        problems.append(f"epochs outside 1..{total_epochs}: {unexpected}")
    return problems


class SessionBudget:
    """Stop and checkpoint before the platform pulls the session out from under us."""

    def __init__(self, budget_minutes: float, safety_margin_minutes: float) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + (budget_minutes - safety_margin_minutes) * 60
        self.budget_minutes = budget_minutes
        self.safety_margin_minutes = safety_margin_minutes

    @property
    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.started) / 60

    @property
    def remaining_minutes(self) -> float:
        return (self.deadline - time.monotonic()) / 60

    def exhausted(self, next_step_estimate_minutes: float = 0.0) -> bool:
        """True when the next epoch would not finish before the safety margin."""
        return time.monotonic() + next_step_estimate_minutes * 60 >= self.deadline
