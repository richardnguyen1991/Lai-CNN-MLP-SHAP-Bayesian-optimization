"""Bayesian search: resumable study, fixed hyperparameters, validation-only.

Acceptance criterion 4 is the one under test: a study interrupted partway must
continue to exactly n_trials in total, with no trial repeated.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import bayesopt as bayesopt_module  # noqa: E402
from bayesopt import (  # noqa: E402
    build_study,
    completed_trials,
    dispose_study,
    snapshot_study_db,
    suggest_params,
)
from shap_selection import feature_schema_hash  # noqa: E402

N_FEATURES = 8


def _make_cache(root: Path) -> Path:
    cache = root / "cache"
    (cache / "preprocess").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for split, n in (("train", 1024), ("val", 384)):
        X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
        y = (rng.random(n) < 1 / (1 + np.exp(-(1.7 * X[:, 2] - X[:, 5])))).astype(np.int8)
        np.save(cache / "preprocess" / f"{split}_X_shard000.npy", X)
        np.save(cache / "preprocess" / f"{split}_y.npy", y)

    names = [f"f{i:02d}" for i in range(N_FEATURES)]
    with (cache / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "selected_features": names,
            "column_index_in_cache": list(range(N_FEATURES)),
            "feature_schema_hash": feature_schema_hash(names),
        }, handle)
    return cache


def _write_configs(root: Path, n_trials: int) -> Path:
    """A miniature copy of the real configs, kept in the repo layout so
    bayesopt.run() reads them exactly as it would in production."""
    configs = root / "configs"
    configs.mkdir(exist_ok=True)

    with (REPO_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment["training"]["batch_size"] = 256

    with (REPO_ROOT / "configs" / "bayesopt.yaml").open(encoding="utf-8") as handle:
        bo = yaml.safe_load(handle)
    bo["n_trials"] = n_trials
    bo["bo_epochs"] = 2
    bo["fixed"]["batch_size"] = 256
    bo["search_space"] = {
        "cnn_filters_1": {"type": "categorical", "choices": [8, 16]},
        "mlp_units_1": {"type": "categorical", "choices": [32, 64]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "weight_decay": {"type": "float", "low": 1.0e-6, "high": 1.0e-3, "log": True},
    }

    for name, payload in (("experiment.yaml", experiment), ("bayesopt.yaml", bo)):
        with (configs / name).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle)

    model_yaml = (REPO_ROOT / "configs" / "model.yaml").read_text(encoding="utf-8")
    (configs / "model.yaml").write_text(model_yaml, encoding="utf-8")
    return root


# --------------------------------------------------------------------------

def test_search_space_types_are_all_supported():
    import optuna

    space = {
        "a": {"type": "categorical", "choices": [1, 2]},
        "b": {"type": "float", "low": 0.0, "high": 1.0},
        "c": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "d": {"type": "int", "low": 1, "high": 4},
    }
    study = optuna.create_study()
    params = suggest_params(study.ask(), space)
    assert set(params) == {"a", "b", "c", "d"}
    assert params["a"] in (1, 2)
    assert 1e-6 <= params["c"] <= 1e-3

    with pytest.raises(ValueError, match="unsupported search space type"):
        suggest_params(study.ask(), {"e": {"type": "loguniform"}})


def test_learning_rate_and_batch_size_cannot_be_searched(tmp_path):
    """Both are fixed by constraint. Letting one into the search space would
    invalidate the comparison and nothing downstream would notice."""
    cache = _make_cache(tmp_path)
    repo = _write_configs(tmp_path, n_trials=1)

    for forbidden in ("learning_rate", "batch_size"):
        with (repo / "configs" / "bayesopt.yaml").open(encoding="utf-8") as handle:
            bo = yaml.safe_load(handle)
        bo["search_space"][forbidden] = {"type": "float", "low": 1e-4, "high": 1e-2}
        with (repo / "configs" / "bayesopt.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(bo, handle)

        with pytest.raises(ValueError, match=f"{forbidden} is fixed"):
            bayesopt_module.run(cache, tmp_path / "out", repo)

        bo["search_space"].pop(forbidden)
        with (repo / "configs" / "bayesopt.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(bo, handle)


def test_study_resumes_without_repeating_trials(tmp_path):
    """Acceptance criterion 4: total trials equals n_trials, none re-run."""
    cache = _make_cache(tmp_path)
    out = tmp_path / "out"

    repo = _write_configs(tmp_path, n_trials=2)
    assert bayesopt_module.run(cache, out, repo) == 0
    first = json.loads((out / "bayesopt" / "best_params.json").read_text(encoding="utf-8"))
    assert first["n_trials_completed"] == 2

    # Same study directory, a larger budget: the search continues.
    repo = _write_configs(tmp_path, n_trials=5)
    assert bayesopt_module.run(cache, out, repo) == 0
    second = json.loads((out / "bayesopt" / "best_params.json").read_text(encoding="utf-8"))
    assert second["n_trials_completed"] == 5

    with (out / "bayesopt" / "optuna_trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numbers = [int(row["number"]) for row in rows]
    assert numbers == sorted(numbers)
    assert len(numbers) == len(set(numbers)), "a trial number was reused"
    assert len(rows) == 5


def test_sqlite_study_survives_a_new_process_handle(tmp_path):
    """Resume works because the study lives on disk, not in memory."""
    with (REPO_ROOT / "configs" / "bayesopt.yaml").open(encoding="utf-8") as handle:
        bo = yaml.safe_load(handle)

    study = build_study(bo, tmp_path / "bayesopt", seed=0)
    try:
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)
        assert completed_trials(study) == 3

        reopened = build_study(bo, tmp_path / "bayesopt", seed=0)
        try:
            assert completed_trials(reopened) == 3
            assert reopened.best_value == study.best_value
        finally:
            dispose_study(reopened)
    finally:
        dispose_study(study)


def test_study_snapshot_is_readable_while_the_study_is_open(tmp_path):
    """The session uploads a snapshot after every trial, taken from a live study."""
    with (REPO_ROOT / "configs" / "bayesopt.yaml").open(encoding="utf-8") as handle:
        bo = yaml.safe_load(handle)

    study = build_study(bo, tmp_path / "bayesopt", seed=0)
    try:
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
        live = tmp_path / "bayesopt" / bo["storage"]["filename"]
        snapshot = snapshot_study_db(live, tmp_path / "snapshot.db")
        assert snapshot.stat().st_size > 0

        import optuna

        restored = optuna.load_study(
            study_name=bo["storage"]["study_name"],
            storage=f"sqlite:///{snapshot.as_posix()}",
        )
        try:
            assert completed_trials(restored) == 2
        finally:
            dispose_study(restored)
    finally:
        dispose_study(study)


def test_best_params_records_what_was_not_searched(tmp_path):
    cache = _make_cache(tmp_path)
    out = tmp_path / "out"
    repo = _write_configs(tmp_path, n_trials=2)
    bayesopt_module.run(cache, out, repo)

    payload = json.loads((out / "bayesopt" / "best_params.json").read_text(encoding="utf-8"))
    assert payload["objective"] == "val_macro_f1"
    assert payload["fixed_not_searched"]["learning_rate"] == 0.001
    assert payload["fixed_not_searched"]["epochs_final"] == 100
    assert "learning_rate" not in payload["params"]
    assert "batch_size" not in payload["params"]

    # The pruning distinction has to survive into the artifact, or a reader
    # cannot tell search pruning from early stopping of the reported run.
    assert "search only" in payload["pruning_note"]
    assert "no early stopping" in payload["pruning_note"]
    assert "Test is not read" in payload["search_note"]


def test_best_params_feed_straight_into_the_model(tmp_path):
    """The searched names must be the ones the model builder accepts."""
    from model import build_model, model_config_from_yaml

    cache = _make_cache(tmp_path)
    out = tmp_path / "out"
    repo = _write_configs(tmp_path, n_trials=2)
    bayesopt_module.run(cache, out, repo)

    params = json.loads(
        (out / "bayesopt" / "best_params.json").read_text(encoding="utf-8")
    )["params"]
    params.pop("weight_decay")          # goes to the optimiser, not the model

    with (REPO_ROOT / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)
    model = build_model(model_config_from_yaml(model_yaml, N_FEATURES, params), seed=0)
    assert model(__import__("torch").randn(4, N_FEATURES)).shape == (4,)
