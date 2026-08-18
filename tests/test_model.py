"""Model: shapes, gradients, determinism, parameter counts, CPU-only.

The parameter-count tests are the ones that matter for the thesis. 126,721 at
K=40 is derived by hand from the paper's stated widths in SPEC section E2, so
if the architecture ever drifts these fail rather than the drift going into a
results table unnoticed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import (  # noqa: E402
    ModelConfig,
    assert_cpu_only,
    build_model,
    count_parameters,
    describe,
    model_config_from_yaml,
    save_model_config,
)


def _config(**overrides) -> ModelConfig:
    base = dict(n_features=40, cnn_filters_1=32, cnn_filters_2=64,
                cnn_kernel_size=3, cnn_dense=128, mlp_units_1=128,
                mlp_units_2=64, fusion_dim=128, dropout=0.2)
    base.update(overrides)
    return ModelConfig(**base)


# --------------------------------------------------------------------------
# Parameter counts, derived by hand in SPEC E2
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_features,expected", [(20, 83201), (40, 126721)])
def test_parameter_count_matches_the_hand_derivation(n_features, expected):
    model = build_model(_config(n_features=n_features), seed=42)
    assert count_parameters(model) == expected


def test_parameter_counts_match_the_config_file():
    with (REPO_ROOT / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)
    expected = model_cfg["expected_param_count"]
    for key, n_features in (("cnn_mlp_k20", 20), ("cnn_mlp_k40", 40)):
        cfg = model_config_from_yaml(model_cfg, n_features=n_features)
        assert count_parameters(build_model(cfg, seed=0)) == expected[key], key


def test_block_parameter_counts_add_up():
    model = build_model(_config(n_features=40), seed=0)
    report = describe(model)
    assert sum(report["parameters_by_block"].values()) == report["total_parameters"]
    assert report["parameters_by_block"]["cnn"] == 88384    # 128 + 6208 + 82048
    assert report["parameters_by_block"]["mlp"] == 13504    # 5248 + 8256
    assert report["parameters_by_block"]["fusion"] == 24833


# --------------------------------------------------------------------------
# Shapes and gradients
# --------------------------------------------------------------------------

@pytest.mark.parametrize("architecture", ["cnn_mlp", "mlp_only", "cnn_only"])
@pytest.mark.parametrize("batch", [1, 7, 4096])
def test_forward_shape(architecture, batch):
    model = build_model(_config(architecture=architecture), seed=0).eval()
    out = model(torch.randn(batch, 40))
    assert out.shape == (batch,)
    assert torch.isfinite(out).all()


def test_backward_produces_finite_gradients_for_every_parameter():
    model = build_model(_config(), seed=0)
    loss = torch.nn.BCEWithLogitsLoss()(
        model(torch.randn(64, 40)), torch.randint(0, 2, (64,)).float()
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradient"
        assert parameter.grad.abs().sum() > 0, f"{name} is disconnected from the loss"


def test_both_branches_reach_the_logit():
    """Acceptance criterion 5: the CNN and the MLP must each contribute.

    A fusion layer that had learned to ignore one branch, or a wiring mistake
    that never connected it, would still train and still report good numbers.
    """
    model = build_model(_config(), seed=0).eval()
    x = torch.randn(16, 40)
    baseline = model(x)

    for branch in ("mlp", "cnn"):
        model_copy = build_model(_config(), seed=0).eval()
        with torch.no_grad():
            for parameter in getattr(model_copy, branch).parameters():
                parameter.zero_()
        assert not torch.allclose(model_copy(x), baseline, atol=1e-6), (
            f"zeroing the {branch} branch left the output unchanged"
        )


def test_wrong_input_width_is_rejected():
    model = build_model(_config(n_features=40), seed=0).eval()
    with pytest.raises(ValueError, match="expected 40 features"):
        model(torch.randn(8, 39))
    with pytest.raises(ValueError, match=r"expected \[B, K\]"):
        model(torch.randn(8, 1, 40))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_weights_and_outputs():
    x = torch.randn(32, 40)
    first = build_model(_config(), seed=1234).eval()
    second = build_model(_config(), seed=1234).eval()
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), name
    assert torch.equal(first(x), second(x))


def test_different_seeds_give_different_weights():
    first = build_model(_config(), seed=1)
    second = build_model(_config(), seed=2)
    assert not torch.equal(
        next(first.parameters()), next(second.parameters())
    )


def test_eval_mode_is_deterministic_despite_dropout():
    model = build_model(_config(dropout=0.5), seed=0).eval()
    x = torch.randn(16, 40)
    assert torch.equal(model(x), model(x))


def test_train_mode_dropout_actually_fires():
    model = build_model(_config(dropout=0.5), seed=0).train()
    x = torch.randn(64, 40)
    torch.manual_seed(0)
    first = model(x)
    torch.manual_seed(1)
    assert not torch.equal(first, model(x))


# --------------------------------------------------------------------------
# Small-K guard and the Conv2d ablation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_features", [1, 2, 3, 4, 5, 8, 20, 40, 83])
def test_small_feature_counts_do_not_crash(n_features):
    """MaxPool1d(2) on length 1 yields length 0 and the next conv explodes deep
    inside autograd. Layers are dropped instead, and the drop is recorded."""
    model = build_model(_config(n_features=n_features), seed=0).eval()
    out = model(torch.randn(4, n_features))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_dropped_pooling_is_recorded_not_silent():
    model = build_model(_config(n_features=3), seed=0)
    notes = describe(model)["notes"]
    assert any("MaxPool skipped" in note for note in notes)


def test_full_size_model_keeps_both_pools():
    model = build_model(_config(n_features=40), seed=0)
    assert describe(model)["notes"] == []
    assert model.cnn.flat_features == 640          # 64 filters x (40 // 4)


def test_conv2d_reshape_ablation_runs_and_flags_its_own_artefact():
    cfg = _config(conv_mode="conv2d_reshape")
    model = build_model(cfg, seed=0).eval()
    out = model(torch.randn(8, 40))
    assert out.shape == (8,)
    notes = describe(model)["notes"]
    assert any("grid adjacency is an artefact" in note for note in notes)


# --------------------------------------------------------------------------
# CPU-only and the config record
# --------------------------------------------------------------------------

def test_no_parameter_lives_on_a_gpu():
    model = build_model(_config(), seed=0)
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_assert_cpu_only_passes_on_this_machine():
    assert_cpu_only(force_cpu=True)     # no CUDA in the Kaggle CPU image either


def test_saved_config_records_the_deviations_from_the_paper(tmp_path):
    model = build_model(_config(), seed=0)
    payload = save_model_config(model, tmp_path / "model_config.json")
    written = json.loads((tmp_path / "model_config.json").read_text(encoding="utf-8"))
    assert written == payload
    assert written["total_parameters"] == 126721
    assert written["device"] == "cpu"
    deviations = " ".join(written["deviations_from_paper"])
    for expected in ("Conv1d", "concatenation", "Dropout"):
        assert expected in deviations


def test_invalid_configs_are_rejected():
    for bad in (dict(architecture="transformer"), dict(conv_mode="conv3d"),
                dict(activation="swish"), dict(n_features=0), dict(dropout=1.0)):
        with pytest.raises(ValueError):
            _config(**bad)
