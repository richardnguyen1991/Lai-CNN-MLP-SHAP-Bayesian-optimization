"""Evaluation, explainability and report regeneration.

Runs the tail of the pipeline end to end on a miniature dataset: train,
evaluate, explain, then rebuild every figure from the artifacts alone.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate as evaluate_module  # noqa: E402
import explainability as explain_module  # noqa: E402
import train as train_module  # noqa: E402
import viz  # noqa: E402
from evaluate import comparison_with_paper, compute_metrics  # noqa: E402
from shap_selection import feature_schema_hash  # noqa: E402

N_FEATURES = 6
TOTAL_EPOCHS = 4


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_metrics_expose_what_accuracy_hides():
    """A model that calls everything an attack scores high accuracy and nothing else.

    This is the concrete reason section 11 forbids concluding from accuracy: at
    the test prior, the useless model looks excellent on the one metric the
    paper leads with.
    """
    y_true = np.array([0] * 10 + [1] * 990, dtype=np.int8)
    y_prob = np.full(1000, 0.99, dtype=np.float32)      # always "attack"
    metrics = compute_metrics(y_true, y_prob)

    assert metrics["accuracy"] == pytest.approx(0.99)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["specificity"] == pytest.approx(0.0)
    assert metrics["fpr"] == pytest.approx(1.0)
    assert metrics["accuracy_of_always_attack"] == pytest.approx(0.99)


def test_confusion_counts_match_the_reported_rates():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.8, 0.7, 0.3, 0.95], dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)

    confusion = metrics["confusion"]
    assert (confusion["tn"], confusion["fp"], confusion["fn"], confusion["tp"]) == (3, 1, 1, 3)
    assert metrics["fpr"] == pytest.approx(1 / 4)
    assert metrics["fnr"] == pytest.approx(1 / 4)
    assert metrics["specificity"] == pytest.approx(3 / 4)


def test_paper_comparison_keeps_both_reported_values():
    """The paper gives 99.95% and 95% for accuracy and never reconciles them.

    Collapsing them to one column would mean choosing which one to be judged
    against, and the flattering choice is the tempting one.
    """
    metrics = compute_metrics(
        np.array([0, 1, 1, 1], dtype=np.int8),
        np.array([0.1, 0.9, 0.9, 0.2], dtype=np.float32),
    )
    paper = {
        "headline": {"accuracy": 0.9995, "recall": 0.9998, "f1": 0.994},
        "body": {"accuracy": 0.95, "precision": 0.99},
    }
    rows = {r["metric"]: r for r in comparison_with_paper(metrics, paper)}

    accuracy = rows["accuracy"]
    assert accuracy["paper_headline"] == 0.9995
    assert accuracy["paper_body"] == 0.95
    assert accuracy["delta_vs_headline"] is not None
    assert accuracy["delta_vs_body"] is not None
    assert "two different values" in accuracy["note"]

    # Metrics the paper never reports must be present and marked as such.
    for name in ("balanced_accuracy", "mcc", "pr_auc", "specificity", "fpr"):
        assert rows[name]["note"] == "not reported by the paper"
        assert rows[name]["paper_headline"] is None


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluate")
    cache = root / "cache" / "preprocess"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(5)

    for split, n in (("train", 1536), ("val", 512), ("test", 768)):
        X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
        logit = 2.2 * X[:, 0] - 1.6 * X[:, 2]
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
        np.save(cache / f"{split}_X_shard000.npy", X)
        np.save(cache / f"{split}_y.npy", y)

    names = [f"f{i:02d}" for i in range(N_FEATURES)]
    with (root / "cache" / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "selected_features": names,
            "column_index_in_cache": list(range(N_FEATURES)),
            "feature_schema_hash": feature_schema_hash(names),
        }, handle)

    with (REPO_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment["training"].update({"epochs": TOTAL_EPOCHS, "batch_size": 128})
    experiment["session"]["checkpoint_interval_epochs"] = 2
    experiment["shap"].update({"deep_shap_sample_rows": 128,
                               "deep_shap_background_rows": 32,
                               "permutation_repeats": 2})
    config = root / "experiment_test.yaml"
    with config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment, handle)

    assert train_module.run(root / "cache", root, REPO_ROOT, "run-e", config) == 0
    assert evaluate_module.run(root / "cache", root, REPO_ROOT, config) == 0
    assert explain_module.run(root / "cache", root, REPO_ROOT, config) == 0
    return {"root": root, "config": config}


def test_evaluation_writes_the_full_metric_set(pipeline):
    metrics = json.loads(
        (pipeline["root"] / "metrics" / "test_metrics.json").read_text(encoding="utf-8")
    )
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                 "macro_f1", "weighted_f1", "mcc", "roc_auc", "pr_auc", "log_loss",
                 "specificity", "fpr", "fnr"):
        assert metrics.get(name) is not None, name
    assert metrics["checkpoint"] == "model_epoch_100.pt"
    assert "best-validation checkpoint not used" in metrics["final_model_rule"]


def test_raw_predictions_are_saved_for_redrawing(pipeline):
    """Curves must be reproducible without a model and without touching test again."""
    root = pipeline["root"]
    y_true = np.load(root / "raw" / "y_true.npy")
    y_prob = np.load(root / "raw" / "y_prob.npy")
    assert y_true.dtype == np.int8
    assert y_prob.dtype == np.float32
    assert len(y_true) == len(y_prob) == 768

    recomputed = compute_metrics(y_true, y_prob)
    stored = json.loads((root / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
    assert recomputed["macro_f1"] == pytest.approx(stored["macro_f1"])
    assert recomputed["confusion"] == stored["confusion"]


def test_evaluate_refuses_a_missing_epoch_100_checkpoint(pipeline, tmp_path):
    """best_val is not a stand-in for the reported model."""
    root = pipeline["root"]
    with pytest.raises(FileNotFoundError, match="epoch-100"):
        evaluate_module.run(root / "cache", root, REPO_ROOT, pipeline["config"],
                            checkpoint_name="model_does_not_exist.pt")


def test_inference_benchmark_separates_preprocess_from_forward(pipeline):
    benchmark = json.loads(
        (pipeline["root"] / "metrics" / "inference_benchmark.json").read_text(encoding="utf-8")
    )
    assert benchmark["device"] == "cpu"
    assert benchmark["warmup_runs"] == 50
    assert benchmark["measured_runs"] == 500
    for key in ("forward_ms_p50", "forward_ms_p95", "preprocess_ms_p50",
                "total_ms_p50", "throughput_flows_per_second"):
        assert benchmark[key] > 0
    assert benchmark["total_ms_p95"] >= benchmark["total_ms_p50"]


def test_explainability_produces_three_independent_rankings(pipeline):
    explain = pipeline["root"] / "explainability"
    for name in ("shap_ranking_cnnmlp.csv", "shap_waterfall_cnnmlp.csv",
                 "permutation_importance.csv", "ranking_agreement.csv"):
        assert (explain / name).exists(), name

    notes = json.loads((explain / "explainability_notes.json").read_text(encoding="utf-8"))
    assert notes["fed_back_into_selection"] is False
    assert notes["computed_on"].startswith("validation")
    assert any("not a causal mechanism" in c for c in notes["caveats"])
    assert any("single-instance waterfalls" in c for c in notes["caveats"])


def test_cnnmlp_waterfall_is_locally_additive(pipeline):
    waterfall = json.loads(
        (pipeline["root"] / "explainability" / "shap_waterfall_cnnmlp.json").read_text(
            encoding="utf-8")
    )
    assert waterfall["sum_of_shap"] == pytest.approx(
        waterfall["model_output"] - waterfall["expected_value"], abs=1e-4
    )


def test_report_redraws_everything_without_training(pipeline, tmp_path):
    """Acceptance criterion 11."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import make_report

    figures = tmp_path / "figures"
    report = make_report.Report(pipeline["root"], figures)
    report.run_all()

    expected = {"learning_curves", "epoch_time", "confusion_matrix_raw",
                "confusion_matrix_norm", "roc_curve", "pr_curve",
                "per_class_metrics", "comparison_with_paper_bar",
                "shap_waterfall", "permutation_importance"}
    assert expected <= set(report.drawn), expected - set(report.drawn)

    for name in report.drawn:
        assert (figures / f"{name}.png").stat().st_size > 5000, name
        assert (figures / f"{name}.pdf").stat().st_size > 1000, name
        assert (figures / f"{name}.csv").exists(), name


# --------------------------------------------------------------------------
# Figure conventions
# --------------------------------------------------------------------------

def test_no_figure_uses_a_second_y_axis():
    """Two y-scales on one frame invent a correlation the data does not contain.

    Loss and Macro-F1 are the tempting pair here, so they are drawn as two
    stacked panels sharing the epoch axis instead.
    """
    source = (REPO_ROOT / "src" / "viz.py").read_text(encoding="utf-8")
    for forbidden in ("twinx", "twiny", "secondary_yaxis"):
        assert forbidden not in source, forbidden


def test_training_modules_never_import_matplotlib():
    """viz.py owns plotting. Checked by parsing imports rather than searching for
    the word, so a docstring saying "no matplotlib here" does not fail."""
    import ast

    for name in ("train.py", "evaluate.py", "bayesopt.py", "model.py",
                 "preprocessing.py", "split.py", "explainability.py"):
        tree = ast.parse((REPO_ROOT / "src" / name).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "matplotlib" not in imported, name
        assert "viz" not in imported, name


def test_viz_is_headless_and_closes_its_figures():
    source = (REPO_ROOT / "src" / "viz.py").read_text(encoding="utf-8")
    assert 'matplotlib.use("Agg")' in source
    assert "plt.show" not in source
    assert "plt.close(fig)" in source


def test_learning_curve_marks_where_sessions_changed(tmp_path):
    """The resume marker is the evidence that a cancelled session continued."""
    history = [
        {"epoch": e, "session_id": "s1" if e <= 3 else "s2",
         "train_loss": 1.0 / e, "val_loss": 1.1 / e,
         "val_macro_f1": 0.5 + e / 20, "val_accuracy": 0.6 + e / 25,
         "epoch_seconds": 1.0}
        for e in range(1, 7)
    ]
    assert viz._resume_boundaries(history) == [{"epoch": 4, "session_id": "s2"}]

    viz.learning_curves(history, tmp_path)
    assert (tmp_path / "learning_curves.png").stat().st_size > 5000
    assert (tmp_path / "learning_curves.csv").exists()

    with (tmp_path / "learning_curves.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6


def test_figures_ship_png_pdf_and_csv(tmp_path):
    viz.epoch_time(
        [{"epoch": e, "session_id": "s", "epoch_seconds": 2.0 + e} for e in range(1, 5)],
        tmp_path,
    )
    for suffix in ("png", "pdf", "csv"):
        assert (tmp_path / f"epoch_time.{suffix}").exists(), suffix
