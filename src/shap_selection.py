"""SHAP feature selection -- fitted on the training split only.

Reproduces the paper's SHAP stage, with two departures that are both recorded
in the artifacts rather than glossed over:

  top_k = 40, not 20. Table 1 and Table 2 each show 19 named features plus a
  row reading "21 other features", so the explained model had 40 inputs.

  Two outputs, not one. The paper calls its tables "summary plots" but their
  values sum to f(x) - E[f(x)] -- +8.960 against +8.961 for Table 2 -- which
  makes them single-instance waterfalls. A global mean|SHAP| ranking is a
  different quantity, so this writes both: the global ranking that drives
  selection, and one instance decomposed the same way the paper did, so the
  comparison is like for like.

Selection uses a LightGBM surrogate with TreeExplainer because KernelExplainer
over a CNN-MLP at this scale is not feasible on CPU. explainability.py later
explains the real model with GradientExplainer; that output is reported, never
fed back into selection, which would close a leakage loop.

Artifacts, under --out-dir:
    explainability/shap_feature_ranking.csv      every feature, not just top_k
    explainability/shap_waterfall_surrogate.csv  one instance, paper-style
    explainability/shap_beeswarm_sample.npz      values retained for plotting
    explainability/comparison_vs_paper_table2.csv
    cache/selected_features.json                 with feature_schema_hash

Usage:
    python src/shap_selection.py --cache-dir work/cache --out-dir work
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def load_train_matrix(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-map the train shards rather than concatenating copies."""
    preprocess = cache_dir / "preprocess"
    shards = sorted(preprocess.glob("train_X_shard*.npy"))
    if not shards:
        raise FileNotFoundError(f"no train shards under {preprocess}")
    blocks = [np.load(path, mmap_mode="r") for path in shards]
    X = np.concatenate([np.asarray(block) for block in blocks])
    y = np.load(preprocess / "train_y.npy")
    if len(X) != len(y):
        raise RuntimeError(f"{len(X)} train rows but {len(y)} labels")
    return X, y


def stratified_sample(
    y: np.ndarray, n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample preserving the class ratio, keeping every row of a class too small
    to sample from."""
    if n_rows >= len(y):
        return np.arange(len(y))

    chosen: List[np.ndarray] = []
    for label in np.unique(y):
        positions = np.flatnonzero(y == label)
        take = max(1, int(round(n_rows * len(positions) / len(y))))
        take = min(take, len(positions))
        chosen.append(rng.choice(positions, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def fit_surrogate(X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any], seed: int):
    import lightgbm as lgb

    surrogate_cfg = cfg["surrogate"]
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=surrogate_cfg["n_estimators"],
        num_leaves=surrogate_cfg["num_leaves"],
        learning_rate=surrogate_cfg["learning_rate"],
        min_child_samples=surrogate_cfg["min_child_samples"],
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def compute_shap(model, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """SHAP values for the positive class, plus the expected value.

    shap returns different shapes across versions and model types: (n, f) for a
    single output, (n, f, 2) or a 2-element list for binary classifiers. All
    three are normalised to (n, f) for the attack class here.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).ravel()
        expected = float(expected[-1] if expected.size > 1 else expected[0])
    return values.astype(np.float64), float(expected)


def rank_features(values: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    """Global ranking by mean|SHAP|, carrying the signed mean alongside.

    The signed mean is what the paper's tables report, so keeping it lets the
    comparison show direction as well as magnitude.
    """
    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    order = np.argsort(-mean_abs)

    total = mean_abs.sum()
    running = 0.0
    ranking = []
    for rank, index in enumerate(order, start=1):
        running += mean_abs[index]
        ranking.append({
            "rank": rank,
            "feature": names[index],
            "mean_abs_shap": float(mean_abs[index]),
            "mean_signed_shap": float(mean_signed[index]),
            "share": float(mean_abs[index] / total) if total else 0.0,
            "cumulative_share": float(running / total) if total else 0.0,
        })
    return ranking


def select_features(
    ranking: List[Dict[str, Any]], cfg: Dict[str, Any], n_available: int
) -> List[str]:
    feature_set = cfg.get("feature_set", "top40")
    if feature_set == "all":
        return [entry["feature"] for entry in ranking]

    if cfg["selection_mode"] == "cumulative_ratio":
        threshold = cfg["cumulative_ratio"]
        selected = [entry["feature"] for entry in ranking
                    if entry["cumulative_share"] <= threshold]
        return selected or [ranking[0]["feature"]]

    top_k = {"top40": 40, "top20": 20}.get(feature_set, cfg["top_k"])
    if top_k > n_available:
        raise ValueError(
            f"top_k={top_k} exceeds the {n_available} features that survived "
            "preprocessing; lower shap.top_k or relax the rejection thresholds"
        )
    return [entry["feature"] for entry in ranking[:top_k]]


def pick_representative(values: np.ndarray, expected: float) -> int:
    """The instance whose model output sits at the median of the sample.

    Deterministic on purpose: the paper never says which record its waterfall
    describes, so the choice has to be stated as a rule rather than picked by
    hand or by seed.
    """
    outputs = expected + values.sum(axis=1)
    return int(np.argmin(np.abs(outputs - np.median(outputs))))


def build_waterfall(
    values: np.ndarray, names: List[str], expected: float, index: int, max_display: int
) -> Dict[str, Any]:
    """One instance decomposed the way the paper's tables are laid out."""
    row = values[index]
    order = np.argsort(-np.abs(row))
    shown = order[: max_display - 1]
    rest = order[max_display - 1:]

    rows = [
        {"rank": rank, "feature": names[i], "shap": float(row[i])}
        for rank, i in enumerate(shown, start=1)
    ]
    # The aggregate row only exists when something was actually aggregated;
    # the paper's "21 other features" has no counterpart when max_display
    # already covers every feature.
    if len(rest):
        rows.append({
            "rank": len(shown) + 1,
            "feature": f"{len(rest)} other features",
            "shap": float(row[rest].sum()),
        })
    return {
        "instance_index": index,
        "expected_value": expected,
        "model_output": float(expected + row.sum()),
        "sum_of_shap": float(row.sum()),
        "rows": rows,
    }


def compare_with_paper(
    ranking: List[Dict[str, Any]],
    waterfall: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Line up our features against Table 2, on both bases.

    Two comparison bases, because they answer different questions and mixing
    them is the mistake the paper's own labelling invites:
      global -- our mean|SHAP| rank, which is what selection used
      local  -- our waterfall rank, the only basis actually comparable to the
                paper's numbers
    """
    global_rank = {entry["feature"]: entry for entry in ranking}
    local_rank = {
        row["feature"]: row["rank"] for row in waterfall["rows"]
        if not row["feature"].endswith("other features")
    }

    comparison = []
    for entry in dataset_cfg.get("paper_table2_features", []):
        canonical = entry["canonical"]
        ours = global_rank.get(canonical)
        comparison.append({
            "feature": canonical,
            "rank_paper_local": entry["rank"],
            "shap_paper_local": entry["shap"],
            "rank_ours_local": local_rank.get(canonical),
            "shap_ours_local": next(
                (row["shap"] for row in waterfall["rows"] if row["feature"] == canonical),
                None,
            ),
            "rank_ours_global": ours["rank"] if ours else None,
            "mean_abs_shap_ours_global": ours["mean_abs_shap"] if ours else None,
            "present_in_our_features": ours is not None,
            "note": entry.get("note", ""),
        })
    return comparison


def feature_schema_hash(features: List[str]) -> str:
    """Stable identity for a feature set, so a resumed session cannot silently
    continue with a different one."""
    payload = json.dumps({"features": features, "n": len(features)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cache_dir: Path, out_dir: Path, repo_root: Path,
        experiment_config: Optional[Path] = None) -> int:
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    cfg = experiment["shap"]
    seed = experiment["run"]["seed"]
    dataset_name = experiment["run"]["dataset"]
    with (repo_root / "configs" / f"dataset_{dataset_name}.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    with (out_dir / "config" / "preprocessing.json").open(encoding="utf-8") as handle:
        preprocessing = json.load(handle)
    names = preprocessing["kept_features"]

    rng = np.random.default_rng(seed)
    X, y = load_train_matrix(cache_dir)
    if X.shape[1] != len(names):
        raise RuntimeError(f"{X.shape[1]} columns but {len(names)} kept feature names")
    print(f"train {X.shape[0]:,} rows x {X.shape[1]} features")

    fit_index = stratified_sample(y, cfg["sample_rows"], rng)
    print(f"surrogate sample: {len(fit_index):,} rows "
          f"({int((y[fit_index] == 0).sum()):,} benign)")
    model = fit_surrogate(X[fit_index], y[fit_index], cfg, seed)

    explain_index = stratified_sample(y[fit_index], cfg["explain_rows"], rng)
    explain_rows = fit_index[explain_index]
    values, expected = compute_shap(model, X[explain_rows])
    print(f"SHAP computed over {values.shape[0]:,} rows, E[f(x)] = {expected:.4f}")

    ranking = rank_features(values, names)
    selected = select_features(ranking, cfg, n_available=len(names))
    schema_hash = feature_schema_hash(selected)

    waterfall = build_waterfall(
        values, names, expected,
        pick_representative(values, expected),
        cfg["waterfall"]["max_display"],
    )

    explain_dir = out_dir / "explainability"
    _write_csv(explain_dir / "shap_feature_ranking.csv", ranking)
    _write_csv(explain_dir / "shap_waterfall_surrogate.csv", waterfall["rows"])
    _write_csv(
        explain_dir / "comparison_vs_paper_table2.csv",
        compare_with_paper(ranking, waterfall, dataset_cfg),
    )
    with (explain_dir / "shap_waterfall_surrogate.json").open("w", encoding="utf-8") as handle:
        json.dump(waterfall, handle, indent=2)

    # Retained so viz.py can draw the beeswarm without recomputing SHAP.
    keep = rng.choice(len(values), size=min(cfg["beeswarm_rows"], len(values)), replace=False)
    np.savez_compressed(
        explain_dir / "shap_beeswarm_sample.npz",
        shap_values=values[keep].astype(np.float32),
        feature_values=X[explain_rows][keep].astype(np.float32),
        feature_names=np.array(names),
        expected_value=np.float64(expected),
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "fitted_on": "train_only",
            "dataset": dataset_name,
            "seed": seed,
            "feature_set": cfg["feature_set"],
            "selection_mode": cfg["selection_mode"],
            "n_selected": len(selected),
            "selected_features": selected,
            "feature_schema_hash": schema_hash,
            "column_index_in_cache": [names.index(f) for f in selected],
            "surrogate": {
                "library": cfg["surrogate"]["library"],
                "explainer": "TreeExplainer",
                "fit_rows": int(len(fit_index)),
                "explained_rows": int(len(explain_rows)),
                "expected_value": expected,
            },
        }, handle, indent=2, ensure_ascii=False)

    _report(ranking, selected, waterfall, schema_hash)
    return 0


def _report(ranking, selected, waterfall, schema_hash) -> None:
    print(f"\nselected {len(selected)} features, schema hash {schema_hash[:12]}")
    print("top 10 by mean|SHAP|:")
    for entry in ranking[:10]:
        print(f"  {entry['rank']:>2}. {entry['feature']:<32} "
              f"{entry['mean_abs_shap']:.5f}  (signed {entry['mean_signed_shap']:+.5f})")
    print(f"\nwaterfall instance {waterfall['instance_index']}: "
          f"E[f(x)]={waterfall['expected_value']:.4f}  "
          f"f(x)={waterfall['model_output']:.4f}  "
          f"sum={waterfall['sum_of_shap']:+.4f}")


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
