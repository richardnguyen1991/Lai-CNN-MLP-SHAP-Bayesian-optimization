"""Explain the trained CNN-MLP, after epoch 100.

SHAP is the paper's own contribution, so it gets three passes rather than one,
each answering a different question:

  GradientExplainer on the CNN-MLP itself. shap_selection.py used a tree
  surrogate because TreeExplainer is what runs on CPU at 70M rows; that ranking
  describes the surrogate. This one describes the model that is actually
  reported.

  A single-instance waterfall, laid out the way Table 1 and Table 2 are. Those
  tables are captioned "summary plot" but their values sum to f(x) - E[f(x)] to
  three decimals, which makes them local decompositions. A global mean|SHAP|
  ranking is not comparable to them; this is.

  Permutation importance on test. Model-agnostic, explainer-independent, and the
  only one of the three that measures a change in the metric we report.

Nothing computed here feeds back into feature selection. Re-selecting features
from an explanation of the model trained on those features closes a loop that
would quietly launder test information into the feature set.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from dataset import SelectedFeatures, load_split
from model import build_model, model_config_from_yaml
from shap_selection import build_waterfall, pick_representative


class _SingleOutput(torch.nn.Module):
    """Re-adds the output dimension the model squeezes away.

    CnnMlp returns [B] so BCEWithLogitsLoss can take it directly, but SHAP's
    GradientExplainer indexes outputs as [B, n_outputs] and fails with a bare
    IndexError on a 1-D tensor.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).unsqueeze(1)


def deep_shap_values(model: torch.nn.Module, background: torch.Tensor,
                     sample: torch.Tensor) -> tuple[np.ndarray, float]:
    """SHAP for the CNN-MLP via GradientExplainer.

    Gradient rather than Kernel: KernelExplainer is model-agnostic but needs
    thousands of forward passes per explained row, which is not feasible on CPU
    even at this sample size. The paper never says which explainer it used.
    """
    import shap

    model.eval()
    explainer = shap.GradientExplainer(_SingleOutput(model), background)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    with torch.no_grad():
        expected = float(model(background).mean().item())
    return values.astype(np.float64), expected


def permutation_importance(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                           names: List[str], repeats: int = 5,
                           batch_size: int = 4096, seed: int = 42,
                           threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Drop in Macro-F1 when one column is shuffled.

    Macro-F1 rather than accuracy: at this prior, shuffling a decisive feature
    barely moves accuracy while collapsing the minority class entirely.
    """
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def score(data: torch.Tensor) -> float:
        model.eval()
        outputs = []
        for begin in range(0, len(data), batch_size):
            outputs.append(torch.sigmoid(model(data[begin:begin + batch_size])).numpy())
        prediction = (np.concatenate(outputs) >= threshold).astype(int)
        return float(f1_score(y.numpy().astype(int), prediction,
                              average="macro", zero_division=0))

    baseline = score(X)
    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            shuffled = X.clone()
            shuffled[:, index] = shuffled[rng.permutation(len(X)), index]
            drops.append(baseline - score(shuffled))
        rows.append({
            "feature": name,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
            "baseline_macro_f1": baseline,
            "repeats": repeats,
        })
    return sorted(rows, key=lambda r: -r["importance_mean"])


def compare_rankings(deep_ranking: List[Dict[str, Any]],
                     surrogate_ranking: List[Dict[str, Any]],
                     permutation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Where the three methods agree, and where they do not.

    Disagreement is informative rather than a defect: it bounds how much weight
    any single ranking can carry in the discussion.
    """
    surrogate_rank = {r["feature"]: int(r["rank"]) for r in surrogate_ranking}
    permutation_rank = {r["feature"]: i + 1 for i, r in enumerate(permutation)}

    rows = []
    for entry in deep_ranking:
        feature = entry["feature"]
        rows.append({
            "feature": feature,
            "rank_cnnmlp_gradient": entry["rank"],
            "mean_abs_shap_cnnmlp": entry["mean_abs_shap"],
            "rank_surrogate_tree": surrogate_rank.get(feature),
            "rank_permutation": permutation_rank.get(feature),
            "rank_spread": _spread(entry["rank"], surrogate_rank.get(feature),
                                   permutation_rank.get(feature)),
        })
    return rows


def _spread(*ranks) -> Optional[int]:
    present = [r for r in ranks if r is not None]
    return max(present) - min(present) if len(present) > 1 else None


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(cache_dir: Path, out_dir: Path, repo_root: Path,
        experiment_config: Optional[Path] = None,
        checkpoint_name: str = "model_epoch_100.pt") -> int:
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with (repo_root / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)

    shap_cfg = experiment["shap"]
    seed = experiment["run"]["seed"]
    rng = np.random.default_rng(seed)
    features = SelectedFeatures(cache_dir)

    overrides: Dict[str, Any] = {}
    best_path = out_dir / "bayesopt" / "best_params.json"
    if best_path.exists():
        with best_path.open(encoding="utf-8") as handle:
            overrides = dict(json.load(handle)["params"])
        overrides.pop("weight_decay", None)

    model = build_model(model_config_from_yaml(model_yaml, len(features), overrides), seed=0)
    model.load_state_dict(
        torch.load(out_dir / "checkpoints" / checkpoint_name, weights_only=True)
    )

    # Validation, not test: this is a description of the model, and test is read
    # exactly once, by evaluate.py.
    X, y = load_split(cache_dir, "val", features)
    sample_rows = min(shap_cfg.get("deep_shap_sample_rows", 2000), len(X))
    background_rows = min(shap_cfg.get("deep_shap_background_rows", 200), len(X))

    order = rng.permutation(len(X))
    background = X[order[:background_rows]]
    sample = X[order[background_rows:background_rows + sample_rows]]

    print(f"GradientExplainer on the CNN-MLP: {len(sample)} rows, "
          f"{len(background)} background")
    values, expected = deep_shap_values(model, background, sample)

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    ordering = np.argsort(-mean_abs)
    deep_ranking = [{
        "rank": position + 1,
        "feature": features.names[index],
        "mean_abs_shap": float(mean_abs[index]),
        "mean_signed_shap": float(mean_signed[index]),
    } for position, index in enumerate(ordering)]

    waterfall = build_waterfall(
        values, features.names, expected,
        pick_representative(values, expected),
        shap_cfg["waterfall"]["max_display"],
    )

    print("permutation importance on the validation sample")
    permutation = permutation_importance(
        model, sample, y[order[background_rows:background_rows + sample_rows]],
        features.names, repeats=shap_cfg.get("permutation_repeats", 5),
        batch_size=experiment["training"]["batch_size"], seed=seed,
        threshold=experiment["training"]["decision_threshold"],
    )

    explain_dir = out_dir / "explainability"
    _write_csv(explain_dir / "shap_ranking_cnnmlp.csv", deep_ranking)
    _write_csv(explain_dir / "shap_waterfall_cnnmlp.csv", waterfall["rows"])
    _write_csv(explain_dir / "permutation_importance.csv", permutation)
    _write_csv(explain_dir / "ranking_agreement.csv", compare_rankings(
        deep_ranking, _read_csv(explain_dir / "shap_feature_ranking.csv"), permutation
    ))
    with (explain_dir / "shap_waterfall_cnnmlp.json").open("w", encoding="utf-8") as handle:
        json.dump(waterfall, handle, indent=2)

    with (explain_dir / "explainability_notes.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_name,
            "computed_on": "validation split; test is read only by evaluate.py",
            "explainer_cnnmlp": "shap.GradientExplainer",
            "explainer_selection": "shap.TreeExplainer on a LightGBM surrogate",
            "paper_explainer": "not stated in the paper",
            "fed_back_into_selection": False,
            "caveats": [
                "SHAP attributes a model's output, not a causal mechanism. A high "
                "value says the model leaned on that feature, not that the feature "
                "causes the attack.",
                "The paper's Table 1 and Table 2 are single-instance waterfalls "
                "despite being captioned summary plots: their values sum to "
                "f(x) - E[f(x)] to three decimals. Compare against "
                "shap_waterfall_cnnmlp.csv, not against the global ranking.",
                "Port and protocol features reflect the capture setup as much as "
                "attack behaviour, so their importance does not transfer to another "
                "network.",
            ],
        }, handle, indent=2, ensure_ascii=False)

    print(f"\ntop 10 by mean|SHAP| on the CNN-MLP:")
    for entry in deep_ranking[:10]:
        print(f"  {entry['rank']:>2}. {entry['feature']:<32} {entry['mean_abs_shap']:.5f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--experiment-config", type=Path, default=None)
    parser.add_argument("--checkpoint", default="model_epoch_100.pt")
    args = parser.parse_args()
    return run(args.cache_dir, args.out_dir, args.repo_root,
               args.experiment_config, args.checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
