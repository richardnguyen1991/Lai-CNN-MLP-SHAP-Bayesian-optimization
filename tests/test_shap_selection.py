"""SHAP selection: train-only fitting, ranking, and the paper comparison.

The waterfall test is the important one. It pins down the identity the paper's
tables satisfy, sum(shap) == f(x) - E[f(x)], which is the evidence that those
tables are single-instance decompositions rather than the global summaries the
captions claim.
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

from shap_selection import (  # noqa: E402
    build_waterfall,
    compare_with_paper,
    compute_shap,
    feature_schema_hash,
    fit_surrogate,
    pick_representative,
    rank_features,
    run,
    select_features,
    stratified_sample,
)

CFG = {
    "feature_set": "top40",
    "selection_mode": "top_k",
    "top_k": 40,
    "cumulative_ratio": 0.95,
    "surrogate": {
        "library": "lightgbm",
        "n_estimators": 40,
        "num_leaves": 15,
        "learning_rate": 0.2,
        "min_child_samples": 20,
    },
}


@pytest.fixture(scope="module")
def shap_fixture():
    """A dataset where the first three features carry all the signal."""
    rng = np.random.default_rng(0)
    n, n_features = 4000, 12
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    logit = 2.5 * X[:, 0] - 1.8 * X[:, 1] + 1.1 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    names = [f"feature_{i:02d}" for i in range(n_features)]

    model = fit_surrogate(X, y, CFG, seed=42)
    values, expected = compute_shap(model, X)
    return {"X": X, "y": y, "names": names, "values": values, "expected": expected}


def test_stratified_sample_preserves_class_ratio():
    y = np.array([0] * 100 + [1] * 900, dtype=np.int8)
    index = stratified_sample(y, 200, np.random.default_rng(0))
    benign_share = (y[index] == 0).mean()
    assert benign_share == pytest.approx(0.10, abs=0.02)
    assert len(index) == len(np.unique(index))


def test_stratified_sample_keeps_a_class_too_small_to_sample():
    """A class of one row must not be rounded away."""
    y = np.array([0] + [1] * 9999, dtype=np.int8)
    index = stratified_sample(y, 100, np.random.default_rng(0))
    assert (y[index] == 0).sum() >= 1


def test_ranking_is_ordered_and_finds_the_informative_features(shap_fixture):
    ranking = rank_features(shap_fixture["values"], shap_fixture["names"])
    scores = [entry["mean_abs_shap"] for entry in ranking]
    assert scores == sorted(scores, reverse=True)
    assert {entry["feature"] for entry in ranking[:3]} == {
        "feature_00", "feature_01", "feature_02"
    }
    assert ranking[-1]["cumulative_share"] == pytest.approx(1.0)


def test_ranking_keeps_the_sign_the_paper_reports(shap_fixture):
    """Table 2 gives signed values, so direction has to survive the ranking."""
    ranking = {e["feature"]: e for e in rank_features(shap_fixture["values"], shap_fixture["names"])}
    assert ranking["feature_00"]["mean_signed_shap"] != 0.0
    for entry in ranking.values():
        assert entry["mean_abs_shap"] >= abs(entry["mean_signed_shap"]) - 1e-9


def test_shap_values_reconstruct_the_model_margin(shap_fixture):
    """E[f(x)] + sum(shap) must equal the surrogate's raw margin.

    shap returns a bare ndarray for some versions and a two-element list for
    others, and picking the wrong element yields values that still look
    self-consistent while explaining the opposite class. Checking against the
    model's own output is the only thing that catches that.
    """
    model = fit_surrogate(shap_fixture["X"], shap_fixture["y"], CFG, seed=42)
    margin = model.predict(shap_fixture["X"], raw_score=True)
    reconstructed = shap_fixture["expected"] + shap_fixture["values"].sum(axis=1)
    assert np.abs(reconstructed - margin).max() < 1e-9


def test_waterfall_satisfies_local_additivity(shap_fixture):
    """sum(shap) == f(x) - E[f(x)] -- the identity that identifies the paper's
    Table 1 and Table 2 as waterfalls (+8.960 vs +8.961 for Table 2)."""
    waterfall = build_waterfall(
        shap_fixture["values"], shap_fixture["names"], shap_fixture["expected"],
        index=17, max_display=8,
    )
    assert waterfall["sum_of_shap"] == pytest.approx(
        waterfall["model_output"] - waterfall["expected_value"], abs=1e-6
    )
    # Every row is accounted for: the shown ones plus the aggregated remainder.
    assert sum(row["shap"] for row in waterfall["rows"]) == pytest.approx(
        waterfall["sum_of_shap"], abs=1e-6
    )


def test_waterfall_aggregates_the_remainder_like_the_paper(shap_fixture):
    """The paper shows 19 rows plus '21 other features'; same shape here."""
    waterfall = build_waterfall(
        shap_fixture["values"], shap_fixture["names"], shap_fixture["expected"],
        index=0, max_display=8,
    )
    assert len(waterfall["rows"]) == 8
    assert waterfall["rows"][-1]["feature"] == "5 other features"   # 12 - 7

    # No remainder means no aggregate row, rather than "0 other features".
    full = build_waterfall(
        shap_fixture["values"], shap_fixture["names"], shap_fixture["expected"],
        index=0, max_display=99,
    )
    assert len(full["rows"]) == len(shap_fixture["names"])
    assert not any("other features" in row["feature"] for row in full["rows"])
    assert sum(row["shap"] for row in full["rows"]) == pytest.approx(
        full["sum_of_shap"], abs=1e-6
    )


def test_representative_instance_is_deterministic(shap_fixture):
    first = pick_representative(shap_fixture["values"], shap_fixture["expected"])
    for _ in range(3):
        assert pick_representative(shap_fixture["values"], shap_fixture["expected"]) == first


def test_select_features_respects_the_variant_axis(shap_fixture):
    ranking = rank_features(shap_fixture["values"], shap_fixture["names"])
    assert len(select_features(ranking, {**CFG, "feature_set": "all"}, 12)) == 12
    with pytest.raises(ValueError, match="exceeds"):
        select_features(ranking, {**CFG, "feature_set": "top40"}, 12)


def test_selecting_more_features_than_survived_fails_loudly(shap_fixture):
    """Silently returning fewer than top_k would change the model's input width
    without anything recording that it happened."""
    ranking = rank_features(shap_fixture["values"], shap_fixture["names"])
    with pytest.raises(ValueError):
        select_features(ranking, {**CFG, "feature_set": "top40", "top_k": 40}, 12)


def test_feature_schema_hash_is_order_sensitive():
    assert feature_schema_hash(["a", "b"]) == feature_schema_hash(["a", "b"])
    assert feature_schema_hash(["a", "b"]) != feature_schema_hash(["b", "a"])
    assert feature_schema_hash(["a", "b"]) != feature_schema_hash(["a", "b", "c"])


def test_paper_comparison_separates_the_two_bases(shap_fixture):
    """Our global ranking and the paper's local values are different quantities;
    the comparison must not present them as one column."""
    ranking = rank_features(shap_fixture["values"], shap_fixture["names"])
    waterfall = build_waterfall(
        shap_fixture["values"], shap_fixture["names"], shap_fixture["expected"], 0, 8
    )
    dataset_cfg = {"paper_table2_features": [
        {"rank": 1, "canonical": "feature_00", "shap": 2.79},
        {"rank": 2, "canonical": "not_in_our_data", "shap": 1.32, "note": "missing"},
    ]}
    rows = compare_with_paper(ranking, waterfall, dataset_cfg)

    assert {"rank_ours_global", "rank_ours_local", "rank_paper_local"} <= set(rows[0])
    assert rows[0]["present_in_our_features"] is True
    assert rows[1]["present_in_our_features"] is False
    assert rows[1]["rank_ours_global"] is None


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def selection_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("shap")
    cache = root / "cache" / "preprocess"
    cache.mkdir(parents=True)

    rng = np.random.default_rng(1)
    n, n_features = 6000, 10
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    logit = 3.0 * X[:, 4] - 2.0 * X[:, 7]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)

    np.save(cache / "train_X_shard000.npy", X[:3000])
    np.save(cache / "train_X_shard001.npy", X[3000:])
    np.save(cache / "train_y.npy", y)

    names = [f"f{i:02d}" for i in range(n_features)]
    (root / "config").mkdir(parents=True)
    with (root / "config" / "preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump({"kept_features": names}, handle)

    with (REPO_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    experiment["shap"].update({
        "feature_set": "top_k", "selection_mode": "top_k", "top_k": 4,
        "sample_rows": 4000, "explain_rows": 1500, "beeswarm_rows": 500,
        "surrogate": CFG["surrogate"],
    })
    config_path = root / "experiment_test.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment, handle)

    code = run(root / "cache", root, REPO_ROOT, config_path)
    return {"root": root, "code": code, "informative": {"f04", "f07"}}


def test_end_to_end_writes_every_artifact(selection_run):
    root = selection_run["root"]
    assert selection_run["code"] == 0
    for relative in (
        "explainability/shap_feature_ranking.csv",
        "explainability/shap_waterfall_surrogate.csv",
        "explainability/shap_waterfall_surrogate.json",
        "explainability/comparison_vs_paper_table2.csv",
        "explainability/shap_beeswarm_sample.npz",
        "cache/selected_features.json",
    ):
        assert (root / relative).exists(), relative


def test_full_ranking_is_saved_not_only_the_selection(selection_run):
    """The paper reports only its top rows; keeping all of them is what makes
    the ranking checkable."""
    path = selection_run["root"] / "explainability" / "shap_feature_ranking.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10                       # every feature, not top_k
    selected = json.loads(
        (selection_run["root"] / "cache" / "selected_features.json").read_text(encoding="utf-8")
    )
    assert selected["n_selected"] == 4


def test_selection_finds_the_planted_signal(selection_run):
    selected = json.loads(
        (selection_run["root"] / "cache" / "selected_features.json").read_text(encoding="utf-8")
    )
    assert selection_run["informative"] <= set(selected["selected_features"])


def test_selected_features_carry_their_cache_column_index(selection_run):
    """Training reads columns out of the cache by index; a name-only record
    would silently mis-slice if the kept-feature order ever changed."""
    selected = json.loads(
        (selection_run["root"] / "cache" / "selected_features.json").read_text(encoding="utf-8")
    )
    assert len(selected["column_index_in_cache"]) == selected["n_selected"]
    names = json.loads(
        (selection_run["root"] / "config" / "preprocessing.json").read_text(encoding="utf-8")
    )["kept_features"]
    for feature, index in zip(selected["selected_features"], selected["column_index_in_cache"]):
        assert names[index] == feature


def test_schema_hash_is_recorded_for_resume(selection_run):
    selected = json.loads(
        (selection_run["root"] / "cache" / "selected_features.json").read_text(encoding="utf-8")
    )
    assert selected["feature_schema_hash"] == feature_schema_hash(selected["selected_features"])
    assert selected["fitted_on"] == "train_only"
