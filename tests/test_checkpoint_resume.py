"""Checkpointing and resume, run locally before anything touches Kaggle.

The central test drives training in several short sessions, each cut off by the
time budget exactly as a Kaggle cancellation would, and then requires that the
result is indistinguishable from one uninterrupted run: same weights, same
history, every epoch exactly once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import train as train_module  # noqa: E402
from checkpoint import (  # noqa: E402
    STATUS_DONE,
    STATUS_RESUME_REQUIRED,
    CheckpointManager,
    SessionBudget,
    TrainingState,
    atomic_write_bytes,
    capture_rng_state,
    config_hash,
    restore_rng_state,
    validate_history,
)
from shap_selection import feature_schema_hash  # noqa: E402

TOTAL_EPOCHS = 12
N_FEATURES = 8


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "model.bin"
    atomic_write_bytes(target, b"first version")
    assert target.read_bytes() == b"first version"
    atomic_write_bytes(target, b"second, longer version")
    assert target.read_bytes() == b"second, longer version"
    assert not list(tmp_path.glob("*.tmp"))


def test_rng_state_round_trips_exactly():
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(7)
    np.random.seed(7)

    state = capture_rng_state(generator)
    expected = (torch.randn(3), np.random.rand(3), torch.randperm(5, generator=generator))

    restore_rng_state(state, generator)
    assert torch.equal(torch.randn(3), expected[0])
    assert np.allclose(np.random.rand(3), expected[1])
    assert torch.equal(torch.randperm(5, generator=generator), expected[2])


def test_history_validation_catches_gaps_and_repeats():
    good = [{"epoch": e} for e in range(1, 6)]
    assert validate_history(good, 5) == []

    assert any("more than once" in p for p in
               validate_history([{"epoch": e} for e in [1, 2, 2, 3, 4, 5]], 5))
    assert any("missing" in p for p in
               validate_history([{"epoch": e} for e in [1, 2, 4, 5]], 5))
    assert any("ascending" in p for p in
               validate_history([{"epoch": e} for e in [1, 3, 2, 4, 5]], 5))
    assert any("outside" in p for p in
               validate_history([{"epoch": e} for e in [1, 2, 3, 4, 5, 9]], 5))


def test_resume_is_refused_when_the_experiment_changed(tmp_path):
    manager = CheckpointManager(tmp_path)
    state = TrainingState(
        run_id="r", session_id="s", dataset_name="cicddos2019",
        config_hash="aaa", feature_schema_hash="bbb",
    )
    manager.verify_resumable(state, {"config_hash": "aaa", "feature_schema_hash": "bbb"})

    with pytest.raises(RuntimeError, match="feature_schema_hash"):
        manager.verify_resumable(state, {"feature_schema_hash": "different"})


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        TrainingState(run_id="r", session_id="s", dataset_name="d", phase="finetune")


def test_config_hash_reacts_to_any_change():
    base = {"training": {"epochs": 100}, "seed": 42}
    assert config_hash(base) == config_hash({"seed": 42, "training": {"epochs": 100}})
    assert config_hash(base) != config_hash({"training": {"epochs": 50}, "seed": 42})


def test_session_budget_reserves_the_safety_margin():
    budget = SessionBudget(budget_minutes=10, safety_margin_minutes=4)
    assert not budget.exhausted(next_step_estimate_minutes=1)
    # An epoch that would run past the margin must stop the session first.
    assert budget.exhausted(next_step_estimate_minutes=7)


# --------------------------------------------------------------------------
# End-to-end resume
# --------------------------------------------------------------------------

def _make_cache(root: Path) -> Path:
    cache = root / "cache"
    (cache / "preprocess").mkdir(parents=True)
    rng = np.random.default_rng(0)

    for split, n in (("train", 2048), ("val", 512)):
        X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
        logit = 2.0 * X[:, 1] - 1.5 * X[:, 3]
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
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


def _make_config(root: Path, budget_minutes: float, margin_minutes: float) -> Path:
    with (REPO_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment["training"].update({"epochs": TOTAL_EPOCHS, "batch_size": 256})
    experiment["session"].update({
        "checkpoint_interval_epochs": 2,
        "session_time_budget_minutes": budget_minutes,
        "safety_margin_minutes": margin_minutes,
    })
    path = root / f"experiment_{budget_minutes}_{margin_minutes}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment, handle)
    return path


def _weights(directory: Path, name: str) -> dict:
    return torch.load(directory / "checkpoints" / name, weights_only=True)


def test_uninterrupted_run_completes_and_records_every_epoch(tmp_path):
    cache = _make_cache(tmp_path)
    out = tmp_path / "uninterrupted"
    config = _make_config(tmp_path, budget_minutes=600, margin_minutes=1)

    assert train_module.run(cache, out, REPO_ROOT, "run-a", config) == 0

    history = json.loads((out / "checkpoints" / "history.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in history] == list(range(1, TOTAL_EPOCHS + 1))
    assert validate_history(history, TOTAL_EPOCHS) == []

    state = json.loads((out / "checkpoints" / "training_state.json").read_text(encoding="utf-8"))
    assert state["status"] == STATUS_DONE
    assert state["current_epoch"] == TOTAL_EPOCHS
    assert (out / "checkpoints" / "model_epoch_100.pt").exists()


def test_interrupted_run_matches_an_uninterrupted_one(tmp_path):
    """Sessions cut short by the time budget must reproduce the single-run result.

    Weights are compared exactly: a resume that restored the optimiser but not
    the RNG, or that replayed an epoch, would diverge here even though every
    artifact would still look well-formed.
    """
    cache = _make_cache(tmp_path)

    reference = tmp_path / "reference"
    train_module.run(cache, reference, REPO_ROOT, "run-ref",
                     _make_config(tmp_path, budget_minutes=600, margin_minutes=1))

    # A budget smaller than the safety margin makes exhausted() fire after the
    # first epoch of every session -- the same shape as a Kaggle cancellation.
    resumed = tmp_path / "resumed"
    chopped = _make_config(tmp_path, budget_minutes=0.001, margin_minutes=0.0005)

    sessions = 0
    while sessions < TOTAL_EPOCHS + 2:
        code = train_module.run(cache, resumed, REPO_ROOT, "run-res", chopped)
        assert code == 0
        sessions += 1
        state = json.loads(
            (resumed / "checkpoints" / "training_state.json").read_text(encoding="utf-8")
        )
        if state["status"] == STATUS_DONE:
            break

    assert state["current_epoch"] == TOTAL_EPOCHS
    assert sessions > 1, "the run was never actually interrupted"

    history = json.loads((resumed / "checkpoints" / "history.json").read_text(encoding="utf-8"))
    assert validate_history(history, TOTAL_EPOCHS) == []
    assert [e["epoch"] for e in history] == list(range(1, TOTAL_EPOCHS + 1))

    reference_history = json.loads(
        (reference / "checkpoints" / "history.json").read_text(encoding="utf-8")
    )
    for resumed_epoch, reference_epoch in zip(history, reference_history):
        assert resumed_epoch["train_loss"] == pytest.approx(
            reference_epoch["train_loss"], rel=1e-6
        ), f"epoch {resumed_epoch['epoch']} diverged"

    for name, tensor in _weights(resumed, "model_epoch_100.pt").items():
        assert torch.allclose(tensor, _weights(reference, "model_epoch_100.pt")[name],
                              atol=1e-6), name


def test_interrupted_run_reports_more_than_one_session(tmp_path):
    """The learning curve marks where sessions changed; that needs distinct ids."""
    cache = _make_cache(tmp_path)
    out = tmp_path / "sessions"
    chopped = _make_config(tmp_path, budget_minutes=0.001, margin_minutes=0.0005)

    for _ in range(TOTAL_EPOCHS + 2):
        train_module.run(cache, out, REPO_ROOT, "run-s", chopped)
        state = json.loads((out / "checkpoints" / "training_state.json").read_text(encoding="utf-8"))
        if state["status"] == STATUS_DONE:
            break

    history = json.loads((out / "checkpoints" / "history.json").read_text(encoding="utf-8"))
    assert len({e["session_id"] for e in history}) > 1
    assert len(state["sessions"]) > 1
    assert state["sessions"][0]["reason"] == "time_budget"
    assert state["sessions"][-1]["reason"] == "completed"


def test_resuming_a_finished_run_is_a_no_op(tmp_path):
    cache = _make_cache(tmp_path)
    out = tmp_path / "finished"
    config = _make_config(tmp_path, budget_minutes=600, margin_minutes=1)

    train_module.run(cache, out, REPO_ROOT, "run-f", config)
    before = (out / "checkpoints" / "history.json").read_text(encoding="utf-8")

    assert train_module.run(cache, out, REPO_ROOT, "run-f", config) == 0
    assert (out / "checkpoints" / "history.json").read_text(encoding="utf-8") == before


def test_resume_refuses_a_different_feature_set(tmp_path):
    """Continuing with different features would produce a complete-looking
    history describing two different models."""
    cache = _make_cache(tmp_path)
    out = tmp_path / "mismatch"
    chopped = _make_config(tmp_path, budget_minutes=0.001, margin_minutes=0.0005)
    train_module.run(cache, out, REPO_ROOT, "run-m", chopped)

    selected = json.loads((cache / "selected_features.json").read_text(encoding="utf-8"))
    selected["feature_schema_hash"] = "tampered"
    with (cache / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle)

    with pytest.raises(RuntimeError, match="refusing to resume"):
        train_module.run(cache, out, REPO_ROOT, "run-m", chopped)


def test_best_val_checkpoint_is_not_the_final_model(tmp_path):
    """model_best_val.pt exists for the curve. The reported model is epoch 100."""
    cache = _make_cache(tmp_path)
    out = tmp_path / "bestval"
    train_module.run(cache, out, REPO_ROOT, "run-b",
                     _make_config(tmp_path, budget_minutes=600, margin_minutes=1))

    state = json.loads((out / "checkpoints" / "training_state.json").read_text(encoding="utf-8"))
    assert state["best_val_epoch"] is not None
    assert (out / "checkpoints" / "model_best_val.pt").exists()

    final = _weights(out, "model_epoch_100.pt")
    last = _weights(out, "model_last.pt")
    for name, tensor in final.items():
        assert torch.allclose(tensor, last[name]), name

    # Only meaningful when validation peaked before the end, which is the case
    # this rule exists for.
    if state["best_val_epoch"] != TOTAL_EPOCHS:
        best = _weights(out, "model_best_val.pt")
        assert any(not torch.allclose(final[n], best[n]) for n in final)


# --------------------------------------------------------------------------
# A state that exists before training does
# --------------------------------------------------------------------------
# The session sequencer records the phase in training_state.json, so the file
# existing is no longer proof that there is a checkpoint behind it. On Kaggle
# this surfaced twice: once as a resume that looked for a model_last.pt nobody
# had written, and once as a config_hash mismatch that refused the run.

def _sequencer_state(out: Path) -> Path:
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    payload = TrainingState(
        run_id="run-seq", session_id="sequencer", dataset_name="cicddos2019",
        phase="final_train", total_epochs=TOTAL_EPOCHS,
    ).to_dict()
    path = checkpoints / "training_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_state_written_before_any_epoch_is_not_mistaken_for_a_checkpoint(tmp_path):
    cache = _make_cache(tmp_path)
    out = tmp_path / "sequenced"
    config = _make_config(tmp_path, budget_minutes=600, margin_minutes=1)
    _sequencer_state(out)
    assert not (out / "checkpoints" / "model_last.pt").exists()

    assert train_module.run(cache, out, REPO_ROOT, "run-seq", config) == 0

    history = json.loads((out / "checkpoints" / "history.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in history] == list(range(1, TOTAL_EPOCHS + 1))
    assert validate_history(history, TOTAL_EPOCHS) == []


def test_a_sequencer_state_carries_no_hashes_and_so_passes_the_guard(tmp_path):
    # publish_phase leaves the hashes empty on purpose. They belong to the
    # training run, which computes them over experiment, model and best_params
    # together; a value derived any other way reads to verify_resumable as a
    # changed experiment and the run is refused.
    cache = _make_cache(tmp_path)
    out = tmp_path / "nohash"
    config = _make_config(tmp_path, budget_minutes=600, margin_minutes=1)
    path = _sequencer_state(out)

    assert json.loads(path.read_text(encoding="utf-8"))["config_hash"] == ""
    assert train_module.run(cache, out, REPO_ROOT, "run-nohash", config) == 0

    # The training run fills them in, so a genuine later resume is still guarded.
    assert json.loads(path.read_text(encoding="utf-8"))["config_hash"]


def test_a_real_checkpoint_is_still_protected_from_a_changed_experiment(tmp_path):
    cache = _make_cache(tmp_path)
    out = tmp_path / "guarded"
    config = _make_config(tmp_path, budget_minutes=600, margin_minutes=1)
    train_module.run(cache, out, REPO_ROOT, "run-guard", config)

    path = out / "checkpoints" / "training_state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["current_epoch"] = 4
    payload["config_hash"] = "a-different-experiment-entirely"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="the experiment has changed"):
        train_module.run(cache, out, REPO_ROOT, "run-guard", config)
