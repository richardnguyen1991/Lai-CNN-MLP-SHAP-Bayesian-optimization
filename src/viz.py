"""Every figure in the thesis, drawn from artifacts on disk.

No module outside this one imports matplotlib, and nothing here trains anything.
Each figure ships as PNG at 300 dpi, PDF vector, and the CSV it was drawn from,
so a value is always readable without the picture.

Design rules that shaped the code, rather than being applied afterwards:

  No dual-axis plot anywhere. Loss and Macro-F1 live on separate stacked panels
  sharing the epoch axis. Two y-scales on one frame invent a correlation that is
  not in the data, and the alignment between them is arbitrary.

  Categorical hues are assigned in fixed slot order and never cycled. The three
  slots used here were validated for colour-vision deficiency separation
  (worst all-pairs deuteranope dE 9.2, normal-vision 24.0) against a white page.

  Magnitude bars get ONE colour. Shading a nominal ranking darker-where-bigger
  double-encodes the bar length and spends the only free channel on information
  the chart already carries. Signed SHAP is different: sign is polarity, so it
  takes the diverging pair.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")           # headless: no display on Kaggle or a runner

import matplotlib.pyplot as plt
import numpy as np

# --- palette ---------------------------------------------------------------
# Validated categorical slots (light surface). Fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_NEGATIVE = "#e34948"
SEQUENTIAL_HUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

FIGSIZE = (7.2, 4.2)
DPI = 300


def _style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: hairline solid grid, no top or right spine."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def _legend(ax: plt.Axes, **kwargs) -> None:
    """A legend whenever there are two or more series; never for one."""
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) < 2:
        return
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, **kwargs)
    legend.set_zorder(5)


def save(fig: plt.Figure, out_dir: Path, name: str,
         data: Optional[List[Dict[str, Any]]] = None,
         tight: bool = True) -> None:
    """PNG + PDF + the CSV the figure was drawn from.

    tight=False for figures carrying a colorbar: tight_layout cannot lay those
    out and warns, while bbox_inches="tight" at save time handles them fine.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    if data:
        with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def _resume_boundaries(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Epochs where the Kaggle session changed.

    Marking them is what makes the resume mechanism visible: a curve with no
    discontinuity across a boundary is the evidence that a cancelled session
    continued rather than restarted.
    """
    boundaries = []
    for previous, current in zip(history, history[1:]):
        if previous.get("session_id") != current.get("session_id"):
            boundaries.append({"epoch": current["epoch"],
                               "session_id": current.get("session_id", "")})
    return boundaries


def _mark_resumes(ax: plt.Axes, boundaries: List[Dict[str, Any]],
                  label_once: bool = True) -> None:
    for index, boundary in enumerate(boundaries):
        ax.axvline(boundary["epoch"], color=INK_MUTED, linewidth=0.9,
                   linestyle="-", alpha=0.55, zorder=1,
                   label="session resume" if (index == 0 and label_once) else None)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def learning_curves(history: List[Dict[str, Any]], out_dir: Path) -> None:
    """Two stacked panels, one y-scale each.

    Loss and Macro-F1 have unrelated units and ranges. Putting them on twin axes
    would make their crossing point a drawing artefact rather than a fact.
    """
    epochs = [e["epoch"] for e in history]
    boundaries = _resume_boundaries(history)

    # Constrained layout rather than tight_layout: two shared-x panels with
    # per-axes titles are exactly the case tight_layout cannot solve, and it
    # warns instead of failing, which is easy to leave in place unnoticed.
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(FIGSIZE[0], 6.4), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1, 1]},
    )

    top.plot(epochs, [e["train_loss"] for e in history], color=SERIES[0],
             linewidth=2, label="train loss", zorder=3)
    top.plot(epochs, [e["val_loss"] for e in history], color=SERIES[1],
             linewidth=2, label="validation loss", zorder=3)
    _mark_resumes(top, boundaries)
    _style(top, "Loss", ylabel="BCE loss")
    _legend(top, loc="upper right")

    bottom.plot(epochs, [e["val_macro_f1"] for e in history], color=SERIES[2],
                linewidth=2, label="validation Macro-F1", zorder=3)
    bottom.plot(epochs, [e["val_accuracy"] for e in history], color=SERIES[0],
                linewidth=2, alpha=0.75, label="validation accuracy", zorder=3)
    _mark_resumes(bottom, boundaries, label_once=False)
    _style(bottom, "Validation metrics", xlabel="epoch", ylabel="score")
    _legend(bottom, loc="lower right")

    # The last epoch is the reported model, so say so on the figure.
    if epochs:
        bottom.annotate(
            f"final model: epoch {epochs[-1]}",
            xy=(epochs[-1], [e["val_macro_f1"] for e in history][-1]),
            xytext=(-6, 10), textcoords="offset points",
            fontsize=8, color=INK_SECONDARY, ha="right",
        )

    save(fig, out_dir, "learning_curves", history, tight=False)


def epoch_time(history: List[Dict[str, Any]], out_dir: Path) -> None:
    epochs = [e["epoch"] for e in history]
    seconds = [e.get("epoch_seconds", 0.0) for e in history]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(epochs, seconds, color=SERIES[0], linewidth=2, zorder=3)
    _mark_resumes(ax, _resume_boundaries(history))
    _style(ax, "Seconds per epoch", xlabel="epoch", ylabel="seconds")
    _legend(ax)
    save(fig, out_dir, "epoch_time",
         [{"epoch": e, "seconds": s} for e, s in zip(epochs, seconds)])


def class_distribution(counts: Dict[str, Dict[str, int]], out_dir: Path) -> None:
    """Benign vs attack per split, on a log scale.

    Linear would render the benign bar as a hairline against 10 million attacks,
    which hides the very imbalance the chart exists to show.
    """
    splits = list(counts)
    benign = [counts[s]["benign"] for s in splits]
    attack = [counts[s]["attack"] for s in splits]
    positions = np.arange(len(splits))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(positions - width / 2 - 0.01, benign, width, color=SERIES[0],
           label="BENIGN", zorder=3)
    ax.bar(positions + width / 2 + 0.01, attack, width, color=SERIES[1],
           label="DDoS", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(positions, splits)

    for position, value in zip(positions - width / 2 - 0.01, benign):
        ax.annotate(f"{value:,}", (position, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5, color=INK_SECONDARY)
    for position, value in zip(positions + width / 2 + 0.01, attack):
        ax.annotate(f"{value:,}", (position, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5, color=INK_SECONDARY)

    _style(ax, "Class distribution per split (log scale)", ylabel="rows")
    _legend(ax, loc="upper left")
    save(fig, out_dir, "class_distribution",
         [{"split": s, "benign": b, "attack": a} for s, b, a in zip(splits, benign, attack)])


# --------------------------------------------------------------------------
# Test results
# --------------------------------------------------------------------------

def confusion_matrix_figure(matrix: np.ndarray, out_dir: Path,
                            normalised: bool = False) -> None:
    """Single-hue sequential heatmap with the value printed in every cell.

    The number in the cell is what makes this readable without colour, which
    matters most for the off-diagonal cells a reviewer actually cares about.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_HUE)
    display = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1) if normalised else matrix

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(display, cmap=cmap, vmin=0,
                      vmax=display.max() if display.max() else 1)
    labels = ["BENIGN", "DDoS"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("actual", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalised else ""),
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    threshold = display.max() * 0.55 if display.max() else 0.5
    for row in range(2):
        for column in range(2):
            value = display[row, column]
            text = f"{value:.4f}" if normalised else f"{int(matrix[row, column]):,}"
            ax.text(column, row, text, ha="center", va="center", fontsize=10,
                    color=SURFACE if value > threshold else INK)

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=7, length=0)

    # One row per actual class, both predicted columns on it. A row per cell
    # would give each dict different keys, and DictWriter takes its field names
    # from the first one.
    name = "confusion_matrix_norm" if normalised else "confusion_matrix_raw"
    save(fig, out_dir, name, [
        {"true": labels[row],
         f"pred_{labels[0]}": float(display[row, 0]),
         f"pred_{labels[1]}": float(display[row, 1])}
        for row in range(2)
    ], tight=False)


def roc_curve_figure(fpr: Sequence[float], tpr: Sequence[float],
                     auc: Optional[float], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9, zorder=2,
            label="chance")
    ax.plot(fpr, tpr, color=SERIES[0], linewidth=2, zorder=3,
            label=f"CNN-MLP (AUC {auc:.4f})" if auc is not None else "CNN-MLP")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, "ROC curve", xlabel="false positive rate", ylabel="true positive rate")
    _legend(ax, loc="lower right")
    save(fig, out_dir, "roc_curve",
         [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)])


def pr_curve_figure(recall: Sequence[float], precision: Sequence[float],
                    ap: Optional[float], baseline: Optional[float],
                    out_dir: Path) -> None:
    """Precision-recall, with the prevalence baseline drawn in.

    Without that baseline a PR curve is unreadable: at a 0.16% positive rate the
    no-skill line sits far from the 0.5 a reader assumes.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    if baseline is not None:
        ax.axhline(baseline, color=INK_MUTED, linewidth=0.9, zorder=2,
                   label=f"prevalence ({baseline:.4f})")
    ax.plot(recall, precision, color=SERIES[0], linewidth=2, zorder=3,
            label=f"CNN-MLP (AP {ap:.4f})" if ap is not None else "CNN-MLP")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, "Precision-recall curve", xlabel="recall", ylabel="precision")
    _legend(ax, loc="lower left")
    save(fig, out_dir, "pr_curve",
         [{"recall": float(a), "precision": float(b)} for a, b in zip(recall, precision)])


def per_class_metrics(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    classes = [r["class"] for r in rows]
    names = ("precision", "recall", "f1")
    positions = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, name in enumerate(names):
        offset = (index - 1) * (width + 0.02)
        values = [r[name] for r in rows]
        ax.bar(positions + offset, values, width, color=SERIES[index],
               label=name, zorder=3)
        for position, value in zip(positions + offset, values):
            ax.annotate(f"{value:.3f}", (position, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=INK_SECONDARY)

    ax.set_xticks(positions, classes)
    ax.set_ylim(0, 1.08)
    _style(ax, "Per-class metrics on test", ylabel="score")
    _legend(ax, loc="lower right")
    save(fig, out_dir, "per_class_metrics", rows)


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------

def shap_ranking_bar(ranking: List[Dict[str, Any]], out_dir: Path,
                     top_n: int = 20) -> None:
    """Global mean|SHAP|, one colour for every bar.

    Shading these by magnitude would encode bar length twice and say nothing new.
    """
    rows = ranking[:top_n][::-1]
    names = [r["feature"] for r in rows]
    values = [float(r["mean_abs_shap"]) for r in rows]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.2, 0.28 * len(rows) + 1.2)))
    ax.barh(np.arange(len(rows)), values, height=0.62, color=SERIES[0], zorder=3)
    ax.set_yticks(np.arange(len(rows)), names)
    for index, value in enumerate(values):
        ax.annotate(f"{value:.4f}", (value, index), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.18 if values else 1)
    _style(ax, f"SHAP feature importance (top {len(rows)})",
           xlabel="mean |SHAP value|")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, out_dir, "shap_feature_ranking_bar", ranking)


def shap_waterfall(waterfall: Dict[str, Any], out_dir: Path) -> None:
    """One instance, laid out as the paper's Table 2 is.

    Sign is polarity here, so the diverging pair is the correct encoding: blue
    pushes the prediction up, red pushes it down.
    """
    rows = waterfall["rows"][::-1]
    names = [r["feature"] for r in rows]
    values = [float(r["shap"]) for r in rows]
    colors = [DIVERGING_POSITIVE if v >= 0 else DIVERGING_NEGATIVE for v in values]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.2, 0.30 * len(rows) + 1.6)))
    ax.barh(np.arange(len(rows)), values, height=0.62, color=colors, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=2)
    ax.set_yticks(np.arange(len(rows)), names)

    # Room on both sides for the value labels. Without it a long negative bar
    # pushes its label off the left edge and into the feature name.
    extent = max(abs(v) for v in values) if values else 1.0
    ax.set_xlim(-extent * 1.45, extent * 1.45)
    for index, value in enumerate(values):
        ax.annotate(f"{value:+.3f}", (value, index),
                    xytext=(5 if value >= 0 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=7.5, color=INK_SECONDARY)

    _style(ax, "SHAP contributions for one instance", xlabel="SHAP value")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(
        f"E[f(x)] = {waterfall['expected_value']:.3f}     "
        f"f(x) = {waterfall['model_output']:.3f}",
        fontsize=8.5, color=INK_MUTED, loc="right", pad=10,
    )
    save(fig, out_dir, "shap_waterfall", waterfall["rows"])


def permutation_importance_bar(rows: List[Dict[str, Any]], out_dir: Path,
                               top_n: int = 20) -> None:
    ordered = sorted(rows, key=lambda r: -abs(float(r["importance_mean"])))[:top_n][::-1]
    names = [r["feature"] for r in ordered]
    values = [float(r["importance_mean"]) for r in ordered]
    errors = [float(r.get("importance_std", 0.0)) for r in ordered]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(3.2, 0.28 * len(ordered) + 1.2)))
    ax.barh(np.arange(len(ordered)), values, height=0.62, color=SERIES[2],
            xerr=errors, error_kw={"ecolor": INK_MUTED, "elinewidth": 0.8,
                                   "capsize": 2}, zorder=3)
    ax.set_yticks(np.arange(len(ordered)), names)
    ax.axvline(0, color=AXIS, linewidth=0.9, zorder=2)
    _style(ax, f"Permutation importance (top {len(ordered)})",
           xlabel="drop in Macro-F1 when shuffled")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, out_dir, "permutation_importance", rows)


# --------------------------------------------------------------------------
# Bayesian search
# --------------------------------------------------------------------------

def bo_optimization_history(trials: List[Dict[str, Any]], out_dir: Path) -> None:
    completed = [t for t in trials if t.get("value") not in (None, "")]
    numbers = [int(t["number"]) for t in completed]
    values = [float(t["value"]) for t in completed]
    running_best = np.maximum.accumulate(values) if values else []

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(numbers, values, s=34, color=SERIES[0], zorder=3,
               edgecolors=SURFACE, linewidths=1.4, label="trial")
    ax.plot(numbers, running_best, color=SERIES[1], linewidth=2, zorder=4,
            label="best so far")
    if values:
        best_index = int(np.argmax(values))
        ax.annotate(f"best {values[best_index]:.4f}",
                    (numbers[best_index], values[best_index]),
                    xytext=(6, -12), textcoords="offset points",
                    fontsize=8, color=INK_SECONDARY)
    _style(ax, "Bayesian optimisation history", xlabel="trial",
           ylabel="validation Macro-F1")
    _legend(ax, loc="lower right")
    save(fig, out_dir, "bo_optimization_history", trials)


def bo_param_importance(importances: Dict[str, float], out_dir: Path) -> None:
    ordered = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], max(2.8, 0.34 * len(names) + 1.2)))
    ax.barh(np.arange(len(names)), values, height=0.6, color=SERIES[0], zorder=3)
    ax.set_yticks(np.arange(len(names)), names)
    for index, value in enumerate(values):
        ax.annotate(f"{value:.3f}", (value, index), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK_SECONDARY)
    _style(ax, "Hyperparameter importance", xlabel="relative importance")
    save(fig, out_dir, "bo_param_importance",
         [{"parameter": k, "importance": v} for k, v in ordered])


# --------------------------------------------------------------------------
# Comparison with the paper
# --------------------------------------------------------------------------

def comparison_with_paper_bar(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """Ours against both of the paper's own figures, as a dot plot.

    Dots rather than bars, deliberately. Every value here sits between about
    0.95 and 1.00, so bars drawn from zero are visually identical and the
    differences that matter live in the fourth decimal. A bar encodes magnitude
    by length from zero and cannot be truncated honestly; a dot encodes it by
    position, so a zoomed axis is legitimate and the spread becomes readable.

    Both of the paper's value sets appear. It reports 99.95% and 95% for the
    same metric and reconciles neither, so a single "paper" marker would mean
    choosing one -- and the flattering choice is the tempting one.
    """
    usable = [r for r in rows if r.get("ours") is not None
              and (r.get("paper_headline") is not None or r.get("paper_body") is not None)]
    if not usable:
        return

    names = [r["metric"] for r in usable]
    positions = np.arange(len(usable))
    series = (
        ("paper (headline)", "paper_headline", SERIES[1], "o"),
        ("paper (body text)", "paper_body", SERIES[2], "s"),
        ("this work", "ours", SERIES[0], "D"),
    )

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 0.75 * len(usable) + 2.0))

    # A connector spans each row's range, so the eye reads the spread rather
    # than hunting for three separate marks.
    for position, row in zip(positions, usable):
        values = [row[key] for _, key, _, _ in series if row.get(key) is not None]
        if len(values) > 1:
            ax.plot([min(values), max(values)], [position, position],
                    color=GRID, linewidth=2.5, solid_capstyle="round", zorder=2)

    for label, key, color, marker in series:
        xs = [row.get(key) for row in usable]
        ys = [p for p, v in zip(positions, xs) if v is not None]
        vs = [v for v in xs if v is not None]
        # A 2px surface ring keeps overlapping marks separable.
        ax.scatter(vs, ys, s=90, color=color, marker=marker, label=label,
                   edgecolors=SURFACE, linewidths=1.6, zorder=4)

    # Label our own value only: a number beside every mark is unreadable, and
    # the paper's figures are already in the caption and the CSV.
    for position, row in zip(positions, usable):
        ax.annotate(f"{row['ours']:.4f}", (row["ours"], position),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_SECONDARY)

    everything = [v for row in usable for _, key, _, _ in series
                  if (v := row.get(key)) is not None]
    low, high = min(everything), max(everything)
    margin = max((high - low) * 0.22, 0.004)
    ax.set_xlim(low - margin, min(high + margin, 1.0 + margin))
    ax.set_yticks(positions, names)
    ax.set_ylim(-0.7, len(usable) - 0.3)

    _style(ax, "This work against the values the paper reports", xlabel="score")
    ax.annotate("axis is truncated; dots encode position, not length",
                xy=(0, 0), xytext=(0, -34), xycoords="axes fraction",
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    save(fig, out_dir, "comparison_with_paper_bar", rows, tight=False)


def ablation_bar(results: List[Dict[str, Any]], out_dir: Path,
                 metric: str = "macro_f1") -> None:
    """Fig 11 in spirit: the branches on their own against the full model."""
    ordered = sorted(results, key=lambda r: r.get(metric, 0.0))
    names = [r["variant"] for r in ordered]
    values = [float(r.get(metric, 0.0)) for r in ordered]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(np.arange(len(names)), values, height=0.58, color=SERIES[0], zorder=3)
    ax.set_yticks(np.arange(len(names)), names)
    for index, value in enumerate(values):
        ax.annotate(f"{value:.4f}", (value, index), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    _style(ax, f"Architecture ablation ({metric})", xlabel=metric)
    save(fig, out_dir, "ablation_bar", results)
