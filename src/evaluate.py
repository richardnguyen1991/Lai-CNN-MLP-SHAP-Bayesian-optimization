"""Evaluate the epoch-100 model on test, once.

Test is read here and nowhere else. It keeps the natural 1:618 prior (only train
and val were subsampled), so these numbers are measured at the real operating
point rather than on a rebalanced set.

Accuracy alone is meaningless at this prior -- predicting "attack" for every row
scores about 99.84% -- so the full set is reported: Balanced Accuracy, MCC,
PR-AUC and the FPR/FNR pair carry the information that accuracy hides.

y_true and y_prob are saved so every curve can be redrawn later without loading
a model or touching test again.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset import SelectedFeatures, iter_split_batches, load_split
from model import build_model, model_config_from_yaml


@torch.no_grad()
def predict_test(model: nn.Module, cache_dir: Path, features: SelectedFeatures,
                 batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream test through the model. It is far too large to materialise."""
    model.eval()
    probabilities: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    for X, y in iter_split_batches(cache_dir, "test", features, batch_size):
        probabilities.append(torch.sigmoid(model(X)).numpy())
        truths.append(y.numpy())
    return (np.concatenate(truths).astype(np.int8),
            np.concatenate(probabilities).astype(np.float32))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    both_classes = len(np.unique(y_true)) > 1
    return {
        "threshold": threshold,
        "n_test": int(len(y_true)),
        "n_benign": int((y_true == 0).sum()),
        "n_attack": int((y_true == 1).sum()),
        "attack_per_benign": round(float((y_true == 1).sum() / max((y_true == 0).sum(), 1)), 2),

        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else None,
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),

        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity_recall_ddos": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,

        # The number a reader would get from accuracy alone, for contrast.
        "accuracy_of_always_attack": float((y_true == 1).mean()),
    }


def benchmark_inference(model: nn.Module, X: torch.Tensor, batch_size: int,
                        warmup: int = 50, runs: int = 500) -> Dict[str, Any]:
    """CPU latency and throughput.

    Preprocessing is timed apart from the forward pass: the paper discusses
    inference cost at length without measuring it, and a deployment decision
    turns on which half dominates.
    """
    model.eval()
    sample = X[:batch_size]
    raw = sample.numpy().astype(np.float64)

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)

    forward: List[float] = []
    preprocess: List[float] = []
    with torch.no_grad():
        for _ in range(runs):
            started = time.perf_counter()
            tensor = torch.from_numpy(raw.astype(np.float32))
            middle = time.perf_counter()
            model(tensor)
            preprocess.append((middle - started) * 1000)
            forward.append((time.perf_counter() - middle) * 1000)

    total = np.asarray(forward) + np.asarray(preprocess)
    return {
        "batch_size": int(batch_size),
        "warmup_runs": warmup,
        "measured_runs": runs,
        "forward_ms_p50": float(np.percentile(forward, 50)),
        "forward_ms_p95": float(np.percentile(forward, 95)),
        "preprocess_ms_p50": float(np.percentile(preprocess, 50)),
        "preprocess_ms_p95": float(np.percentile(preprocess, 95)),
        "total_ms_p50": float(np.percentile(total, 50)),
        "total_ms_p95": float(np.percentile(total, 95)),
        "throughput_flows_per_second": float(batch_size / (np.percentile(total, 50) / 1000)),
        "peak_rss_mb": _peak_rss_mb(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS bytes.
        return round(usage / 1024, 1)
    except ImportError:
        try:
            import psutil

            return round(psutil.Process().memory_info().rss / 2 ** 20, 1)
        except ImportError:
            return None


def comparison_with_paper(metrics: Dict[str, Any],
                          paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ours against BOTH sets of numbers the paper reports.

    One column would force a choice between 99.95% and 95% for the same metric.
    The paper never reconciles them, so neither do we.
    """
    headline = paper.get("headline", {})
    body = paper.get("body", {})
    rows = []
    for name, ours_key in (("accuracy", "accuracy"), ("precision", "precision"),
                           ("recall", "recall"), ("f1", "f1")):
        ours = metrics.get(ours_key)
        paper_headline = headline.get(name)
        paper_body = body.get(name)
        rows.append({
            "metric": name,
            "paper_headline": paper_headline,
            "paper_body": paper_body,
            "ours": round(ours, 6) if ours is not None else None,
            "delta_vs_headline": round(ours - paper_headline, 6)
            if (ours is not None and paper_headline is not None) else None,
            "delta_vs_body": round(ours - paper_body, 6)
            if (ours is not None and paper_body is not None) else None,
            "note": "paper reports two different values for this metric"
            if (paper_headline is not None and paper_body is not None) else "",
        })

    # Metrics we add. The paper reports none of these, and at 1:618 they are the
    # ones that decide whether a detector is usable.
    for name in ("balanced_accuracy", "macro_f1", "mcc", "roc_auc", "pr_auc",
                 "specificity", "fpr", "fnr"):
        value = metrics.get(name)
        rows.append({
            "metric": name,
            "paper_headline": None,
            "paper_body": None,
            "ours": round(value, 6) if value is not None else None,
            "delta_vs_headline": None,
            "delta_vs_body": None,
            "note": "not reported by the paper",
        })
    return rows


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
        experiment_config: Optional[Path] = None,
        checkpoint_name: str = "model_epoch_100.pt") -> int:
    config_path = experiment_config or (repo_root / "configs" / "experiment.yaml")
    with config_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    with (repo_root / "configs" / "model.yaml").open(encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)
    dataset_name = experiment["run"]["dataset"]
    with (repo_root / "configs" / f"dataset_{dataset_name}.yaml").open(encoding="utf-8") as handle:
        dataset_cfg = yaml.safe_load(handle)

    training_cfg = experiment["training"]
    features = SelectedFeatures(cache_dir)

    overrides: Dict[str, Any] = {}
    best_path = out_dir / "bayesopt" / "best_params.json"
    if best_path.exists():
        with best_path.open(encoding="utf-8") as handle:
            overrides = dict(json.load(handle)["params"])
        overrides.pop("weight_decay", None)

    model = build_model(model_config_from_yaml(model_yaml, len(features), overrides), seed=0)
    checkpoint = out_dir / "checkpoints" / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} is missing. The reported model is the epoch-100 one; "
            "model_best_val.pt exists for the learning curve and is not a substitute."
        )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))

    print(f"evaluating {checkpoint_name} on test (natural prior, read once)")
    y_true, y_prob = predict_test(model, cache_dir, features, training_cfg["batch_size"])
    metrics = compute_metrics(y_true, y_prob, training_cfg["decision_threshold"])

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "y_true.npy", y_true)
    np.save(raw_dir / "y_prob.npy", y_prob)

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics["evaluated_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["checkpoint"] = checkpoint_name
    metrics["final_model_rule"] = "epoch 100; best-validation checkpoint not used"
    with (metrics_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    y_pred = (y_prob >= metrics["threshold"]).astype(np.int8)
    (metrics_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=["BENIGN", "DDoS"],
                              digits=6, zero_division=0),
        encoding="utf-8",
    )

    matrix = np.array([[metrics["confusion"]["tn"], metrics["confusion"]["fp"]],
                       [metrics["confusion"]["fn"], metrics["confusion"]["tp"]]])
    _write_csv(metrics_dir / "confusion_matrix_raw.csv", [
        {"true": "BENIGN", "pred_BENIGN": int(matrix[0, 0]), "pred_DDoS": int(matrix[0, 1])},
        {"true": "DDoS", "pred_BENIGN": int(matrix[1, 0]), "pred_DDoS": int(matrix[1, 1])},
    ])
    normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    _write_csv(metrics_dir / "confusion_matrix_norm.csv", [
        {"true": "BENIGN", "pred_BENIGN": float(normalised[0, 0]), "pred_DDoS": float(normalised[0, 1])},
        {"true": "DDoS", "pred_BENIGN": float(normalised[1, 0]), "pred_DDoS": float(normalised[1, 1])},
    ])

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    _write_csv(metrics_dir / "roc_curve.csv", [
        {"fpr": float(a), "tpr": float(b), "threshold": float(c)}
        for a, b, c in zip(fpr, tpr, roc_thresholds)
    ])
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    _write_csv(metrics_dir / "pr_curve.csv", [
        {"recall": float(a), "precision": float(b),
         "threshold": float(c) if i < len(pr_thresholds) else None}
        for i, (a, b, c) in enumerate(
            zip(recall, precision, list(pr_thresholds) + [None])
        )
    ])

    _write_csv(metrics_dir / "summary_metrics.csv", [
        {"metric": k, "value": v} for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ])
    _write_csv(metrics_dir / "comparison_with_paper.csv",
               comparison_with_paper(metrics, dataset_cfg.get("paper_reported_metrics", {})))

    X_val, _ = load_split(cache_dir, "val", features)
    benchmark = benchmark_inference(model, X_val, training_cfg["batch_size"])
    with (metrics_dir / "inference_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2)

    _report(metrics, benchmark)
    return 0


def _report(metrics: Dict[str, Any], benchmark: Dict[str, Any]) -> None:
    print(f"\ntest: {metrics['n_test']:,} rows, "
          f"{metrics['n_benign']:,} benign (1:{metrics['attack_per_benign']:,.0f})")
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                 "macro_f1", "mcc", "roc_auc", "pr_auc", "log_loss"):
        value = metrics.get(name)
        if value is not None:
            print(f"  {name:<20} {value:.6f}")
    print(f"  {'specificity':<20} {metrics['specificity']:.6f}")
    print(f"  {'FPR':<20} {metrics['fpr']:.6f}")
    print(f"  {'FNR':<20} {metrics['fnr']:.6f}")
    print(f"\nalways-predict-attack would score "
          f"{metrics['accuracy_of_always_attack']:.6f} accuracy on this split")
    print(f"\ninference: p50 {benchmark['total_ms_p50']:.2f} ms/batch of "
          f"{benchmark['batch_size']}, {benchmark['throughput_flows_per_second']:,.0f} flows/s")


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
