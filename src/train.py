"""Final training: 100 epochs, Adam, batch 4096, CPU, resumable by epoch.

Three rules this module exists to enforce, all of them easy to violate by
accident and none of them recoverable after the fact:

  Always 100 epochs. No early stopping, no convergence check, no "it stopped
  improving". A cancelled session resumes at the next epoch; it never restarts
  and never skips.

  The final model is the one after epoch 100. model_best_val.pt is written for
  the learning curve only. Picking the best-validation checkpoint and then
  reporting test turns validation into a second training signal.

  history.json holds epochs 1..100, each exactly once. Resume is what makes
  that non-trivial, so validate_artifacts.py asserts it afterwards.

No matplotlib import here; viz.py owns plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, f1_score

from checkpoint import (
    STATUS_DONE,
    STATUS_RESUME_REQUIRED,
    CheckpointManager,
    SessionBudget,
    TrainingState,
    atomic_save_torch,
    config_hash,
    validate_history,
)
from dataset import SelectedFeatures, epoch_batches, load_split
from model import assert_cpu_only, build_model, model_config_from_yaml, save_model_config


def configure_cpu(threads: Any = "auto") -> int:
    """Pin thread counts before the first tensor op, or the setting is ignored."""
    count = os.cpu_count() or 1
    n_threads = count if threads == "auto" else int(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass    # already initialised; harmless on resume inside one process
    return n_threads


@torch.no_grad()
def evaluate_split(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor,
    batch_size: int, threshold: float = 0.5,
) -> Dict[str, float]:
    """Loss and the metrics the learning curve and the BO objective need.

    Macro-F1 rather than accuracy: at 1:30 in train and 1:618 in test, a model
    that predicts "attack" for everything scores over 96% accuracy.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    logits = torch.empty(len(X))
    for begin in range(0, len(X), batch_size):
        chunk = model(X[begin:begin + batch_size])
        logits[begin:begin + len(chunk)] = chunk
        total_loss += criterion(chunk, y[begin:begin + len(chunk)]).item()

    probability = torch.sigmoid(logits).numpy()
    truth = y.numpy().astype(int)
    prediction = (probability >= threshold).astype(int)

    return {
        "loss": total_loss / max(len(X), 1),
        "accuracy": float((prediction == truth).mean()),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(truth, probability))
        if len(np.unique(truth)) > 1 else float("nan"),
    }


def train_one_epoch(
    model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
    X: torch.Tensor, y: torch.Tensor, batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for features, labels in epoch_batches(X, y, batch_size, generator):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(features)
        seen += len(features)
    return total / max(seen, 1)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    if cfg["optimizer"] != "adam":
        raise ValueError(f"the paper specifies Adam, got {cfg['optimizer']!r}")
    return torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )


def run(
    cache_dir: Path,
    out_dir: Path,
    repo_root: Path,
    run_id: str,
    experiment_config: Optional[Path] = None,
    best_params: Optional[Dict[str, Any]] = None,
) -> int:
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with (repo_root / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)

    training_cfg = dict(experiment["training"])
    session_cfg = experiment["session"]
    seed = experiment["run"]["seed"]

    assert_cpu_only(experiment["device"].get("force_cpu", True))
    n_threads = configure_cpu(experiment["device"].get("cpu_threads", "auto"))
    device = torch.device("cpu")

    # Bayesian search supplies architecture and weight_decay; learning rate and
    # batch size are fixed and are not part of the search space.
    overrides = dict(best_params or {})
    training_cfg["weight_decay"] = overrides.pop("weight_decay", training_cfg["weight_decay"])

    features = SelectedFeatures(cache_dir)
    X_train, y_train = load_split(cache_dir, "train", features)
    X_val, y_val = load_split(cache_dir, "val", features)
    print(f"train {tuple(X_train.shape)}  val {tuple(X_val.shape)}  threads {n_threads}")

    torch.manual_seed(seed)
    model_cfg = model_config_from_yaml(model_yaml, len(features), overrides)
    model = build_model(model_cfg, seed=seed).to(device)
    optimizer = build_optimizer(model, training_cfg)
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(seed)
    manager = CheckpointManager(out_dir / "checkpoints")
    save_model_config(model, out_dir / "config" / "model_config.json")

    expected_hashes = {
        "config_hash": config_hash({"experiment": experiment, "model": model_yaml,
                                    "best_params": best_params}),
        "feature_schema_hash": features.schema_hash,
    }

    session_id = uuid.uuid4().hex[:12]
    state = manager.load_state()
    history = manager.load_history()

    if state is None:
        state = TrainingState(
            run_id=run_id, session_id=session_id,
            dataset_name=experiment["run"]["dataset"],
            phase="final_train", total_epochs=training_cfg["epochs"],
            **expected_hashes,
        )
    else:
        manager.verify_resumable(state, expected_hashes)
        if state.current_epoch >= training_cfg["epochs"]:
            print(f"already at epoch {state.current_epoch}; nothing to do")
            return 0
        manager.restore(model, optimizer, generator)
        # Anything after the last completed epoch was never checkpointed.
        history = [e for e in history if e["epoch"] <= state.current_epoch]
        state.session_id = session_id
        print(f"resuming at epoch {state.current_epoch + 1} of {training_cfg['epochs']}")

    budget = SessionBudget(
        session_cfg["session_time_budget_minutes"], session_cfg["safety_margin_minutes"]
    )
    batch_size = training_cfg["batch_size"]
    threshold = training_cfg["decision_threshold"]
    epoch_seconds: List[float] = []

    for epoch in range(state.current_epoch + 1, training_cfg["epochs"] + 1):
        started = time.monotonic()
        train_loss = train_one_epoch(
            model, optimizer, criterion, X_train, y_train, batch_size, generator
        )
        metrics = evaluate_split(model, X_val, y_val, batch_size, threshold)
        elapsed = time.monotonic() - started
        epoch_seconds.append(elapsed)

        history.append({
            "epoch": epoch,
            "session_id": session_id,       # the resume marker on the curves
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_pr_auc": metrics["pr_auc"],
            "epoch_seconds": elapsed,
        })
        state.current_epoch = epoch

        improved = state.best_val_metric is None or metrics["macro_f1"] > state.best_val_metric
        if improved:
            state.best_val_metric = metrics["macro_f1"]
            state.best_val_epoch = epoch

        print(f"epoch {epoch:>3}/{training_cfg['epochs']}  "
              f"train_loss={train_loss:.5f}  val_loss={metrics['loss']:.5f}  "
              f"macro_f1={metrics['macro_f1']:.5f}  {elapsed:.1f}s")

        interval = session_cfg["checkpoint_interval_epochs"]
        estimate = float(np.median(epoch_seconds)) / 60
        out_of_time = budget.exhausted(estimate)

        if epoch % interval == 0 or out_of_time or epoch == training_cfg["epochs"]:
            manager.save(
                state, model, optimizer, history, generator,
                # Observation only. Never loaded as the final model.
                extra_model_name="model_best_val.pt" if improved else None,
            )

        if out_of_time and epoch < training_cfg["epochs"]:
            state.status = STATUS_RESUME_REQUIRED
            state.sessions.append({
                "session_id": session_id, "last_epoch": epoch,
                "reason": "time_budget", "elapsed_minutes": round(budget.elapsed_minutes, 2),
            })
            manager.save(state, model, optimizer, history, generator)
            print(f"\nstopping at epoch {epoch} with a valid checkpoint; "
                  f"{budget.elapsed_minutes:.1f} min used. status={STATUS_RESUME_REQUIRED}")
            # Exit 0: cancellation by the time limit is expected, not a failure.
            return session_cfg["resume_required_exit_code"]

    # The final model is epoch 100, full stop.
    atomic_save_torch(manager.model_path("model_epoch_100.pt"), model.state_dict())
    state.status = STATUS_DONE
    state.phase = "evaluate"
    state.sessions.append({
        "session_id": session_id, "last_epoch": state.current_epoch,
        "reason": "completed", "elapsed_minutes": round(budget.elapsed_minutes, 2),
    })
    manager.save(state, model, optimizer, history, generator)

    problems = validate_history(history, training_cfg["epochs"])
    if problems:
        print("\nHISTORY INVALID:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    print(f"\ndone: {training_cfg['epochs']} epochs, final model = model_epoch_100.pt")
    print(f"best validation macro-F1 was {state.best_val_metric:.5f} at epoch "
          f"{state.best_val_epoch} (recorded only; not the final model)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--experiment-config", type=Path, default=None)
    parser.add_argument("--best-params", type=Path, default=None,
                        help="best_params.json from bayesopt.py")
    args = parser.parse_args()

    best_params = None
    if args.best_params and args.best_params.exists():
        with args.best_params.open(encoding="utf-8") as handle:
            best_params = json.load(handle)["params"]

    return run(args.cache_dir, args.out_dir, args.repo_root, args.run_id,
               args.experiment_config, best_params)


if __name__ == "__main__":
    raise SystemExit(main())
