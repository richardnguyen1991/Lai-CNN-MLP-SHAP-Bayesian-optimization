"""Bayesian hyperparameter search with Optuna TPE, resumable across sessions.

The study lives in a SQLite file that is synced to S3 after every trial, so a
cancelled Kaggle session continues the search rather than restarting it: trial
21 stays trial 21.

Two boundaries this module holds:

  The objective is validation Macro-F1. Test is never touched here, and the
  best parameters are then used to retrain from scratch for the full 100
  epochs, which is what the paper describes ("the model is retrained using
  them").

  Pruning is a search-budget device and lives only inside the search. It is not
  early stopping of the reported experiment, and the distinction is recorded in
  best_params.json so a reader cannot conflate them.

learning_rate and batch_size are fixed and absent from the search space.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import yaml

from dataset import SelectedFeatures, load_split
from model import assert_cpu_only, build_model, model_config_from_yaml
from train import build_optimizer, configure_cpu, evaluate_split, train_one_epoch


def suggest_params(trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
            )
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            raise ValueError(f"unsupported search space type {spec['type']!r} for {name}")
    return params


def make_objective(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    model_yaml: Dict[str, Any], bo_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any], n_features: int, seed: int,
):
    search_space = bo_cfg["search_space"]
    fixed = bo_cfg["fixed"]

    def objective(trial) -> float:
        params = suggest_params(trial, search_space)
        weight_decay = params.pop("weight_decay")

        torch.manual_seed(seed)
        model = build_model(model_config_from_yaml(model_yaml, n_features, params), seed=seed)
        optimizer = build_optimizer(model, {
            "optimizer": fixed["optimizer"],
            "learning_rate": fixed["learning_rate"],
            "weight_decay": weight_decay,
        })
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(seed)

        score = 0.0
        for epoch in range(1, bo_cfg["bo_epochs"] + 1):
            train_one_epoch(model, optimizer, criterion, X_train, y_train,
                            fixed["batch_size"], generator)
            metrics = evaluate_split(model, X_val, y_val, fixed["batch_size"],
                                     training_cfg["decision_threshold"])
            score = metrics["macro_f1"]
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        return score

    return objective


def build_study(bo_cfg: Dict[str, Any], storage_dir: Path, seed: int):
    import optuna

    storage_dir.mkdir(parents=True, exist_ok=True)
    database = storage_dir / bo_cfg["storage"]["filename"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=bo_cfg.get("pruner_warmup_steps", 3))
        if bo_cfg.get("pruner") == "median" else optuna.pruners.NopPruner()
    )
    return optuna.create_study(
        study_name=bo_cfg["storage"]["study_name"],
        storage=f"sqlite:///{database.as_posix()}",
        direction=bo_cfg["direction"],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bo_cfg["storage"].get("load_if_exists", True),
    )


def dispose_study(study) -> None:
    """Close the storage's connection pool.

    Optuna keeps SQLAlchemy connections open for the life of the process. Left
    open they leak across a long session and, more importantly, the database
    file on disk may not be fully flushed -- which matters because that file is
    what gets uploaded to S3 and read back to resume.
    """
    storage = getattr(study, "_storage", None)
    storage = getattr(storage, "_backend", storage)     # unwrap _CachedStorage
    engine = getattr(storage, "engine", None)
    if engine is not None:
        engine.dispose()


def snapshot_study_db(source: Path, destination: Path) -> Path:
    """A consistent copy of the study database, safe to upload while it is open.

    The trial callback syncs after every trial, so the live file may have writes
    in flight. sqlite3's backup API takes a transactionally consistent snapshot;
    copying the file bytes does not, and a torn copy would be discovered only on
    the next session's resume.
    """
    import sqlite3
    from contextlib import closing

    # closing(), not a bare `with`: sqlite3's context manager commits the
    # transaction and leaves the connection open. Over a 20-trial search that is
    # 40 leaked handles, and on Windows an open handle blocks the file.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as live, \
            closing(sqlite3.connect(destination)) as copy:
        live.backup(copy)
    return destination


def completed_trials(study) -> int:
    """Trials that produced a value. Pruned and failed ones are re-run."""
    import optuna

    return sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)


def export_trials(study, path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        }
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(cache_dir: Path, out_dir: Path, repo_root: Path,
        experiment_config: Optional[Path] = None,
        trial_callback: Optional[Callable[[Any, Any], None]] = None) -> int:
    """Run the search to completion.

    trial_callback fires after every trial. The Kaggle session uses it to push
    the SQLite study to S3, so a cancellation costs at most the trial in flight
    rather than the whole search.
    """
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with (repo_root / "configs" / "bayesopt.yaml").open(encoding="utf-8") as handle:
        bo_cfg = yaml.safe_load(handle)
    with (repo_root / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)

    for forbidden in ("learning_rate", "batch_size"):
        if forbidden in bo_cfg["search_space"]:
            raise ValueError(
                f"{forbidden} is fixed by constraint and must not be searched; "
                "remove it from bayesopt.yaml search_space"
            )

    seed = experiment["run"]["seed"]
    assert_cpu_only(experiment["device"].get("force_cpu", True))
    configure_cpu(experiment["device"].get("cpu_threads", "auto"))

    features = SelectedFeatures(cache_dir)
    X_train, y_train = load_split(cache_dir, "train", features)
    X_val, y_val = load_split(cache_dir, "val", features)

    study = build_study(bo_cfg, out_dir / "bayesopt", seed)
    try:
        return _search(study, bo_cfg, experiment, model_yaml, out_dir,
                       X_train, y_train, X_val, y_val, len(features), seed,
                       trial_callback)
    finally:
        dispose_study(study)


def _search(study, bo_cfg, experiment, model_yaml, out_dir,
            X_train, y_train, X_val, y_val, n_features, seed, trial_callback) -> int:
    already = completed_trials(study)
    remaining = bo_cfg["n_trials"] - already
    print(f"{already} of {bo_cfg['n_trials']} trials complete; running {max(remaining, 0)}")

    if remaining > 0:
        study.optimize(
            make_objective(X_train, y_train, X_val, y_val, model_yaml, bo_cfg,
                           experiment["training"], n_features, seed),
            n_trials=remaining,
            gc_after_trial=True,
            callbacks=[trial_callback] if trial_callback else None,
        )

    export_trials(study, out_dir / "bayesopt" / "optuna_trials.csv")

    best = study.best_trial
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": bo_cfg["storage"]["study_name"],
        "objective": bo_cfg["objective"],
        "best_value": best.value,
        "best_trial_number": best.number,
        "params": best.params,
        "n_trials_completed": completed_trials(study),
        "n_trials_requested": bo_cfg["n_trials"],
        "fixed_not_searched": bo_cfg["fixed"],
        "pruner": bo_cfg.get("pruner"),
        "pruning_note": (
            "MedianPruner applies to the search only. The final model runs all "
            f"{bo_cfg['fixed']['epochs_final']} epochs with no pruning and no "
            "early stopping."
        ),
        "search_note": (
            "Optimised on validation. Test is not read during the search, and "
            "the best parameters are used to retrain from scratch."
        ),
    }
    with (out_dir / "bayesopt" / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"\nbest trial {best.number}: {bo_cfg['objective']} = {best.value:.5f}")
    for name, value in sorted(best.params.items()):
        print(f"  {name} = {value}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--experiment-config", type=Path, default=None)
    args = parser.parse_args()
    return run(args.cache_dir, args.out_dir, args.repo_root, args.experiment_config)


if __name__ == "__main__":
    raise SystemExit(main())
