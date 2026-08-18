"""Rebuild every figure from artifacts, without training anything.

Acceptance criterion 11. It also protects the thesis writing loop: redrawing a
figure because an axis label was wrong must not cost a 100-epoch run, and the
artifacts are what the numbers in the text are quoted from.

Anything missing is skipped with a line saying so, rather than aborting: a run
still in its Bayesian search has no test metrics yet, and that is not an error.

Usage:
    python scripts/make_report.py --run-dir work --out-dir work/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import viz  # noqa: E402


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows or None


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Report:
    def __init__(self, run_dir: Path, out_dir: Path) -> None:
        self.run = run_dir
        self.out = out_dir
        self.drawn: List[str] = []
        self.skipped: List[str] = []

    def attempt(self, name: str, source: Path, draw: Callable[[], None]) -> None:
        if not source.exists():
            self.skipped.append(f"{name} (no {source.relative_to(self.run)})")
            return
        draw()
        self.drawn.append(name)

    # -- training ---------------------------------------------------------
    def training_figures(self) -> None:
        path = self.run / "checkpoints" / "history.json"
        self.attempt("learning_curves", path,
                     lambda: viz.learning_curves(read_json(path), self.out))
        self.attempt("epoch_time", path,
                     lambda: viz.epoch_time(read_json(path), self.out))

    def distribution_figure(self) -> None:
        path = self.run / "config" / "split_manifest.json"
        if not path.exists():
            self.skipped.append("class_distribution (no split_manifest.json)")
            return
        summary = read_json(path)["summary"]
        counts = {name: {"benign": summary[name]["benign"],
                         "attack": summary[name]["attack"]}
                  for name in ("train", "val", "test") if name in summary}
        viz.class_distribution(counts, self.out)
        self.drawn.append("class_distribution")

    # -- test -------------------------------------------------------------
    def test_figures(self) -> None:
        metrics_path = self.run / "metrics" / "test_metrics.json"
        if not metrics_path.exists():
            self.skipped.append("test figures (no test_metrics.json)")
            return
        metrics = read_json(metrics_path)
        confusion = metrics["confusion"]
        matrix = np.array([[confusion["tn"], confusion["fp"]],
                           [confusion["fn"], confusion["tp"]]], dtype=float)
        viz.confusion_matrix_figure(matrix, self.out, normalised=False)
        viz.confusion_matrix_figure(matrix, self.out, normalised=True)
        self.drawn += ["confusion_matrix_raw", "confusion_matrix_norm"]

        roc = read_csv(self.run / "metrics" / "roc_curve.csv")
        if roc:
            viz.roc_curve_figure([float(r["fpr"]) for r in roc],
                                 [float(r["tpr"]) for r in roc],
                                 metrics.get("roc_auc"), self.out)
            self.drawn.append("roc_curve")

        pr = read_csv(self.run / "metrics" / "pr_curve.csv")
        if pr:
            prevalence = metrics["n_attack"] / max(metrics["n_test"], 1)
            viz.pr_curve_figure([float(r["recall"]) for r in pr],
                                [float(r["precision"]) for r in pr],
                                metrics.get("pr_auc"), prevalence, self.out)
            self.drawn.append("pr_curve")

        viz.per_class_metrics(self._per_class(metrics), self.out)
        self.drawn.append("per_class_metrics")

        comparison = read_csv(self.run / "metrics" / "comparison_with_paper.csv")
        if comparison:
            viz.comparison_with_paper_bar([
                {"metric": r["metric"],
                 "paper_headline": as_float(r.get("paper_headline")),
                 "paper_body": as_float(r.get("paper_body")),
                 "ours": as_float(r.get("ours"))}
                for r in comparison
            ], self.out)
            self.drawn.append("comparison_with_paper_bar")

    @staticmethod
    def _per_class(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        confusion = metrics["confusion"]
        tn, fp, fn, tp = (confusion["tn"], confusion["fp"],
                          confusion["fn"], confusion["tp"])

        def triple(true_positive, false_positive, false_negative):
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            return precision, recall, f1

        benign = triple(tn, fn, fp)
        attack = triple(tp, fp, fn)
        return [
            {"class": "BENIGN", "precision": benign[0], "recall": benign[1], "f1": benign[2]},
            {"class": "DDoS", "precision": attack[0], "recall": attack[1], "f1": attack[2]},
        ]

    # -- explainability ---------------------------------------------------
    def explainability_figures(self) -> None:
        ranking = self.run / "explainability" / "shap_feature_ranking.csv"
        self.attempt("shap_feature_ranking_bar", ranking,
                     lambda: viz.shap_ranking_bar(read_csv(ranking), self.out))

        waterfall = self.run / "explainability" / "shap_waterfall_cnnmlp.json"
        if not waterfall.exists():
            waterfall = self.run / "explainability" / "shap_waterfall_surrogate.json"
        self.attempt("shap_waterfall", waterfall,
                     lambda: viz.shap_waterfall(read_json(waterfall), self.out))

        permutation = self.run / "explainability" / "permutation_importance.csv"
        self.attempt("permutation_importance", permutation,
                     lambda: viz.permutation_importance_bar(read_csv(permutation), self.out))

    # -- search -----------------------------------------------------------
    def search_figures(self) -> None:
        trials = self.run / "bayesopt" / "optuna_trials.csv"
        self.attempt("bo_optimization_history", trials,
                     lambda: viz.bo_optimization_history(read_csv(trials) or [], self.out))

        database = self.run / "bayesopt" / "optuna_study.db"
        if not database.exists():
            self.skipped.append("bo_param_importance (no optuna_study.db)")
            return
        study = None
        try:
            import optuna

            from bayesopt import dispose_study

            study = optuna.load_study(
                study_name=None, storage=f"sqlite:///{database.as_posix()}"
            )
            importances = optuna.importance.get_param_importances(study)
            viz.bo_param_importance(importances, self.out)
            self.drawn.append("bo_param_importance")
        except Exception as error:                     # noqa: BLE001
            # Importance needs several completed trials; too few is normal early on.
            self.skipped.append(f"bo_param_importance ({type(error).__name__})")
        finally:
            if study is not None:
                from bayesopt import dispose_study

                dispose_study(study)

    def run_all(self) -> None:
        self.training_figures()
        self.distribution_figure()
        self.test_figures()
        self.explainability_figures()
        self.search_figures()

    def summarise(self) -> None:
        print(f"\n{len(self.drawn)} figures written to {self.out}")
        for name in self.drawn:
            print(f"  + {name}.png / .pdf / .csv")
        if self.skipped:
            print(f"\n{len(self.skipped)} skipped:")
            for name in self.skipped:
                print(f"  - {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    report = Report(args.run_dir, args.out_dir or (args.run_dir / "figures"))
    report.run_all()
    report.summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
