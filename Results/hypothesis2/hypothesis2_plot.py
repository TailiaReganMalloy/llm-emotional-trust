"""Visualization generation for hypothesis 2.

This module imports statistical transformations and test results from
hypothesis2_stats.py and writes the corresponding figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from hypothesis2_stats import *

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

FIGURES_DIR = REPO_ROOT / "Figures"

PLOT_OUTPUT_PATH = FIGURES_DIR / "hypothesis2.png"

REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis2.txt"

ALPHA = 0.05

SUBPLOT_TITLE_FONTSIZE = 26

AXIS_LABEL_FONTSIZE = 26

TICK_LABEL_FONTSIZE = 26

ANNOTATION_FONTSIZE = 24

SIGNIFICANCE_FONTSIZE = 24

CONDITION_ORDER = ["Interactive", "Text"]

CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static",
}

CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}

def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""

def add_pair_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    star_text: str,
    y_step: float,
    text_above: bool = True,
    point_up: bool = False,
) -> None:
    y_ends = [y + y_step, y, y, y + y_step] if point_up else [y - y_step, y, y, y - y_step]
    ax.plot([x1, x1, x2, x2], y_ends, color="black", lw=2.0, zorder=4)
    text_y = y + 0.6 * y_step if text_above else y - 0.6 * y_step
    va = "bottom" if text_above else "top"
    ax.text(
        (x1 + x2) / 2,
        text_y,
        star_text or "ns",
        ha="center",
        va=va,
        fontsize=SIGNIFICANCE_FONTSIZE,
        zorder=5,
    )

def draw_condition_subplot(
    ax: plt.Axes,
    test_result: dict[str, float],
    title: str,
    show_y_axis: bool,
) -> None:
    x = np.array([0.0, 1.0])
    means = np.array([test_result["mean_interactive"], test_result["mean_text"]], dtype=float)
    lows = np.array([test_result["ci_low_interactive"], test_result["ci_low_text"]], dtype=float)
    highs = np.array([test_result["ci_high_interactive"], test_result["ci_high_text"]], dtype=float)

    ax.bar(
        x,
        means,
        width=0.58,
        color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=3.0,
        capsize=7,
        capthick=3.0,
        zorder=3,
    )

    finite_lows = lows[np.isfinite(lows)]
    finite_highs = highs[np.isfinite(highs)]
    if finite_lows.size == 0 or finite_highs.size == 0:
        finite_means = means[np.isfinite(means)]
        if finite_means.size == 0:
            y_min, y_max = -0.1, 0.1
        else:
            y_min = float(np.nanmin(finite_means)) - 0.1
            y_max = float(np.nanmax(finite_means)) + 0.1
    else:
        y_min = float(np.nanmin(finite_lows))
        y_max = float(np.nanmax(finite_highs))

    y_span = max(y_max - y_min, 0.2)
    y_bracket = y_min - 0.10 * y_span

    add_pair_bracket(
        ax,
        x[0],
        x[1],
        y_bracket,
        significance_stars(float(test_result["p_adj"])),
        0.03 * y_span,
        text_above=False,
        point_up=True,
    )

    summary = f"t = {test_result['t']:.3f} $q {format_p(float(test_result['p_adj']))}$"
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=TICK_LABEL_FONTSIZE + 2)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE + 2, pad=14)
    if show_y_axis:
        ax.set_ylabel("Trust Score Change", fontsize=AXIS_LABEL_FONTSIZE + 2)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(-0.3, 0.05)
    ax.set_yticks([-0.3, -0.2, -0.1, 0.0])

# Plot-only entrypoint: statistical calculations and report writing are in the stats module.
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    # Reuse the statistical module's data preparation and test results for plot annotations.
    analysis = build_analysis_frame(data)
    overall_test = condition_welch_test(analysis, "overall_change")
    emotional_test = condition_welch_test(analysis, "emotional_change")
    analytical_test = condition_welch_test(analysis, "analytical_change")
    # Annotate B-H adjusted p-values (q), adjusted across the three sub-hypothesis tests as in the stats report.
    tests = [analytical_test, emotional_test, overall_test]
    for test, q in zip(tests, bh_qvalues([t["p"] for t in tests])):
        test["p_adj"] = q
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True, sharey=True)
    draw_condition_subplot(axes[0], analytical_test, title="Analytical Change", show_y_axis=True)
    draw_condition_subplot(axes[1], emotional_test, title="Emotional Change", show_y_axis=False)
    draw_condition_subplot(axes[2], overall_test, title="Overall Change", show_y_axis=False)
    fig.suptitle("Hypothesis 2: Main Condition Effects Across Trust Change Outcomes", fontsize=32)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print("Saved plot:", PLOT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
