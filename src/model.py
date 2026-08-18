"""The Optimizer-equipped CNN-MLP, on CPU.

Two branches over the SAME feature vector, joined by concatenation:

    x -> [B, K]
    mlp : Linear(K,128) ReLU -> Linear(128,64) ReLU              -> [B, 64]
    cnn : reshape [B,1,K]
          Conv1d(1,f1,k,'same')  ReLU -> MaxPool1d(2)
          Conv1d(f1,f2,k,'same') ReLU -> MaxPool1d(2)
          Flatten -> Linear(.,128) ReLU                          -> [B, 128]
    out : concat -> Dropout(p) -> Linear(192,fusion) ReLU
                 -> Linear(fusion,1)                             -> logit

Branch widths are the paper's (Fig 2 / Algorithm 1, Fig 4 / Algorithm 2). Two
things are ours and are marked as such wherever they surface:

  Conv1d rather than Conv2d. The paper says 3x3 kernels and 2x2 pooling, which
  describes an image, while the input is a 1-D flow record. conv2d_reshape is
  kept as an ablation that folds K features into a padded grid.

  The junction. The paper draws each branch and never the point where they meet
  (SPEC finding B15), so late concatenation is a decision, not a reproduction.

No CUDA call anywhere: the paper's own hardware was an AMD RX 550, which has no
CUDA either, so CPU-only is faithful rather than a compromise.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


@dataclass
class ModelConfig:
    n_features: int
    architecture: str = "cnn_mlp"          # cnn_mlp | mlp_only | cnn_only
    conv_mode: str = "conv1d"              # conv1d | conv2d_reshape
    cnn_filters_1: int = 32
    cnn_filters_2: int = 64
    cnn_kernel_size: int = 3
    cnn_dense: int = 128
    mlp_units_1: int = 128
    mlp_units_2: int = 64
    fusion_dim: int = 128
    dropout: float = 0.2
    activation: str = "relu"
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.architecture not in ("cnn_mlp", "mlp_only", "cnn_only"):
            raise ValueError(f"unknown architecture {self.architecture!r}")
        if self.conv_mode not in ("conv1d", "conv2d_reshape"):
            raise ValueError(f"unknown conv_mode {self.conv_mode!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _pool_plan(length: int) -> Tuple[bool, bool, int, List[str]]:
    """Decide which pooling layers survive for a K this small.

    MaxPool1d(2) on a length-1 sequence produces length 0 and the next
    convolution fails with a shape error thrown from deep inside autograd. With
    K = 40 both pools apply (40 -> 20 -> 10) so this never fires in the main
    experiment, but a top_k ablation below 8 would hit it, and dropping a layer
    silently would change the architecture without anything saying so.
    """
    notes: List[str] = []
    use_first = length >= 2
    if not use_first:
        notes.append(f"first MaxPool skipped: input length {length} < 2")
    after_first = length // 2 if use_first else length

    use_second = after_first >= 2
    if not use_second:
        notes.append(f"second MaxPool skipped: length {after_first} < 2")
    after_second = after_first // 2 if use_second else after_first

    return use_first, use_second, after_second, notes


class MlpBranch(nn.Module):
    """Dense(128, ReLU) -> Dense(64, ReLU), exactly as Fig 2."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        activation = ACTIVATIONS[cfg.activation]
        self.net = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.mlp_units_1),
            activation(),
            nn.Linear(cfg.mlp_units_1, cfg.mlp_units_2),
            activation(),
        )
        self.out_features = cfg.mlp_units_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Cnn1dBranch(nn.Module):
    """Conv(32) -> pool -> Conv(64) -> pool -> flatten -> Dense(128), as Fig 4."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        activation = ACTIVATIONS[cfg.activation]
        use_first, use_second, final_length, notes = _pool_plan(cfg.n_features)
        cfg.notes.extend(notes)

        layers: List[nn.Module] = [
            nn.Conv1d(1, cfg.cnn_filters_1, cfg.cnn_kernel_size, padding="same"),
            activation(),
        ]
        if use_first:
            layers.append(nn.MaxPool1d(2))
        layers += [
            nn.Conv1d(cfg.cnn_filters_1, cfg.cnn_filters_2,
                      cfg.cnn_kernel_size, padding="same"),
            activation(),
        ]
        if use_second:
            layers.append(nn.MaxPool1d(2))
        layers.append(nn.Flatten())

        self.features = nn.Sequential(*layers)
        self.flat_features = cfg.cnn_filters_2 * final_length
        self.head = nn.Sequential(
            nn.Linear(self.flat_features, cfg.cnn_dense), activation()
        )
        self.out_features = cfg.cnn_dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1)))


class Cnn2dBranch(nn.Module):
    """Ablation: fold K features into a zero-padded HxW grid and use Conv2d.

    Exists only so the paper's literal 3x3 / 2x2 reading can be measured rather
    than argued about. The grid order is arbitrary, which is precisely why this
    is not the default.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        activation = ACTIVATIONS[cfg.activation]
        self.height = int(math.ceil(math.sqrt(cfg.n_features)))
        self.width = int(math.ceil(cfg.n_features / self.height))
        self.padded = self.height * self.width
        cfg.notes.append(
            f"conv2d_reshape: {cfg.n_features} features -> {self.height}x{self.width} "
            f"grid, {self.padded - cfg.n_features} zero-padded cells; "
            "grid adjacency is an artefact of the reshape, not of the data"
        )

        _, _, out_h, notes_h = _pool_plan(self.height)
        _, _, out_w, _ = _pool_plan(self.width)
        cfg.notes.extend(notes_h)
        use_first = self.height >= 2 and self.width >= 2
        use_second = (self.height // 2 if use_first else self.height) >= 2 and \
                     (self.width // 2 if use_first else self.width) >= 2

        layers: List[nn.Module] = [
            nn.Conv2d(1, cfg.cnn_filters_1, cfg.cnn_kernel_size, padding="same"),
            activation(),
        ]
        if use_first:
            layers.append(nn.MaxPool2d(2))
        layers += [
            nn.Conv2d(cfg.cnn_filters_1, cfg.cnn_filters_2,
                      cfg.cnn_kernel_size, padding="same"),
            activation(),
        ]
        if use_second:
            layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())

        self.features = nn.Sequential(*layers)
        height = out_h if use_second else (self.height // 2 if use_first else self.height)
        width = out_w if use_second else (self.width // 2 if use_first else self.width)
        self.flat_features = cfg.cnn_filters_2 * height * width
        self.head = nn.Sequential(
            nn.Linear(self.flat_features, cfg.cnn_dense), activation()
        )
        self.out_features = cfg.cnn_dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.padded > x.shape[1]:
            x = torch.nn.functional.pad(x, (0, self.padded - x.shape[1]))
        grid = x.view(batch, 1, self.height, self.width)
        return self.head(self.features(grid))


class CnnMlp(nn.Module):
    """Both branches over the same input, concatenated into a single logit."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        activation = ACTIVATIONS[cfg.activation]

        self.mlp = MlpBranch(cfg) if cfg.architecture in ("cnn_mlp", "mlp_only") else None
        if cfg.architecture in ("cnn_mlp", "cnn_only"):
            branch = Cnn2dBranch if cfg.conv_mode == "conv2d_reshape" else Cnn1dBranch
            self.cnn = branch(cfg)
        else:
            self.cnn = None

        fused = sum(b.out_features for b in (self.mlp, self.cnn) if b is not None)
        self.fusion = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(fused, cfg.fusion_dim),
            activation(),
            nn.Linear(cfg.fusion_dim, 1),
        )
        self.fused_features = fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"expected [B, K], got {tuple(x.shape)}")
        if x.shape[1] != self.config.n_features:
            raise ValueError(
                f"expected {self.config.n_features} features, got {x.shape[1]}"
            )
        parts = [branch(x) for branch in (self.mlp, self.cnn) if branch is not None]
        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.fusion(fused).squeeze(1)


def build_model(cfg: ModelConfig, seed: Optional[int] = None) -> CnnMlp:
    """Construct on CPU, optionally with deterministic initialisation."""
    if seed is not None:
        torch.manual_seed(seed)
    model = CnnMlp(cfg).to(torch.device("cpu"))
    return model


def assert_cpu_only(force_cpu: bool = True) -> None:
    """Refuse to continue if the run could silently drift onto a GPU."""
    if force_cpu and torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is visible but the experiment is specified as CPU-only; "
            "set force_cpu: false in experiment.yaml to override deliberately"
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: CnnMlp) -> Dict[str, Any]:
    """The record written to model_config.json."""
    cfg = model.config
    per_block = {
        name: sum(p.numel() for p in block.parameters())
        for name, block in (("mlp", model.mlp), ("cnn", model.cnn),
                            ("fusion", model.fusion))
        if block is not None
    }
    return {
        "config": asdict(cfg),
        "total_parameters": count_parameters(model),
        "parameters_by_block": per_block,
        "fused_features": model.fused_features,
        "cnn_flat_features": getattr(model.cnn, "flat_features", None),
        "device": "cpu",
        "loss": "bce_with_logits",
        "output": "single logit; sigmoid applied at inference",
        "deviations_from_paper": [
            "Conv1d with kernel_size=3 replaces the paper's Conv2d 3x3, because "
            "the input is a 1-D flow record rather than an image",
            "Late concatenation of the two branches; the paper does not describe "
            "the junction at all",
            "Dropout is present; the paper never mentions dropout",
        ],
        "notes": list(dict.fromkeys(cfg.notes)),
    }


def save_model_config(model: CnnMlp, path: Path) -> Dict[str, Any]:
    payload = describe(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def model_config_from_yaml(
    model_cfg: Dict[str, Any], n_features: int, overrides: Optional[Dict[str, Any]] = None
) -> ModelConfig:
    """Merge configs/model.yaml with Bayesian-search overrides."""
    fields = {
        key: model_cfg[key] for key in (
            "architecture", "conv_mode", "cnn_filters_1", "cnn_filters_2",
            "cnn_kernel_size", "cnn_dense", "mlp_units_1", "mlp_units_2",
            "fusion_dim", "dropout", "activation",
        ) if key in model_cfg
    }
    fields.update(overrides or {})
    return ModelConfig(n_features=n_features, **fields)
