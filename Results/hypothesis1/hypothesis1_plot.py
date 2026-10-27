"""Visualization generation for hypothesis 1.

This module imports statistical transformations and test results from
hypothesis1_stats.py and writes the corresponding figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from hypothesis1_stats import *

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

TXT_PATH = OUTPUT_DIR / "hypothesis1.txt"

FIGURES_DIR = REPO_ROOT / "Figures"

PNG_PATH = FIGURES_DIR / "hypothesis1.png"

ALPHA = 0.05

ANALYTICAL_MAX = 10.0

EMOTIONAL_MAX = 9.0

SUBPLOT_TITLE_FONTSIZE = 26

AXIS_LABEL_FONTSIZE = 26

TICK_LABEL_FONTSIZE = 26

ANNOTATION_FONTSIZE = 24

SIGNIFICANCE_FONTSIZE = 24

PREPOST_COLORS = {
    "Pre": "#377eb8",
    "Post": "#c26a26",
}

ERROR_COLOR = "#4a4a4a"

def p_to_stars(pvalue: float) -> str:
    if np.isnan(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"

def draw_subplot(ax: plt.Axes, title: str, stats: dict[str, float], show_y_axis: bool) -> None:
    x = np.array([0.0, 1.0], dtype=float)
    labels = ["Pre", "Post"]
    means = np.array([stats["pre_mean"], stats["post_mean"]], dtype=float)
    cis = np.array([stats["pre_ci"], stats["post_ci"]], dtype=float)
    cis_clean = np.nan_to_num(cis, nan=0.0)

    for idx, label in enumerate(labels):
        ax.bar(
            x[idx],
            means[idx],
            width=0.8,
            color=PREPOST_COLORS[label],
            edgecolor="none",
            zorder=2,
        )
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=cis_clean[idx],
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=2.8,
            capsize=8,
            capthick=2.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONTSIZE + 2)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE + 2, pad=14)
    if show_y_axis:
        ax.set_ylabel("Trust Score (0-1)", fontsize=AXIS_LABEL_FONTSIZE + 2)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    y_top = float(np.nanmax(means + cis_clean))
    y_bottom = float(np.nanmin(means - cis_clean))
    y_span = max(y_top - y_bottom, 0.25)

    y_bracket = y_top + 0.07 * y_span
    y_text = y_bracket + 0.02 * y_span
    y_low = y_bottom - 0.15 * y_span
    y_high = y_bracket + 0.20 * y_span

    ax.plot(
        [0, 0, 1, 1],
        [y_bracket - 0.015 * y_span, y_bracket, y_bracket, y_bracket - 0.015 * y_span],
        color="black",
        lw=2.0,
        zorder=4,
    )
    ax.text(0.4, y_text, p_to_stars(stats["p_adj"]), ha="center", va="bottom", fontsize=SIGNIFICANCE_FONTSIZE, zorder=5)

    interpretation = "Significant pre-post effect" if float(stats["p"]) < ALPHA else "No significant pre-post effect"
    summary = f"t = {stats['t']:.3f} $q {p_text(float(stats['p_adj']))}$"
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SIGNIFICANCE_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.set_ylim(bottom=0, top=0.45)

# Plot-only entrypoint: statistical calculations and report writing are in the stats module.
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # Plot the same normalized paired outcomes used by the stats module.
    analytical_pre = df["Total Analytical Trust"] / ANALYTICAL_MAX
    analytical_post = df["Total Analytical Trust Post"] / ANALYTICAL_MAX
    emotional_pre = df["Total Emotional Trust"] / EMOTIONAL_MAX
    emotional_post = df["Total Emotional Trust Post"] / EMOTIONAL_MAX
    results = {
        "Analytical Trust": analyze_metric(analytical_pre, analytical_post),
        "Emotional Trust": analyze_metric(emotional_pre, emotional_post),
        "Overall Trust": analyze_metric((analytical_pre + emotional_pre) / 2.0, (analytical_post + emotional_post) / 2.0),
    }
    # Annotate B-H adjusted p-values (q), adjusted across the three sub-hypothesis tests as in the stats report.
    for stats, q in zip(results.values(), bh_qvalues([r["p"] for r in results.values()])):
        stats["p_adj"] = q
    plt.style.use("ggplot")
    plt.rcParams.update({"xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black"})
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    for idx, (ax, (name, stats)) in enumerate(zip(axes, results.items())):
        draw_subplot(ax, name, stats, show_y_axis=idx == 0)
    fig.suptitle("Hypothesis 1: Pre vs Post Experiment Trust Questionnaire", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
