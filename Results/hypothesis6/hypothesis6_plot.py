"""Visualization generation for hypothesis 6.

This module imports statistical transformations and test results from
hypothesis6_stats.py and writes the corresponding figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from hypothesis6_stats import *

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

TXT_PATH = OUTPUT_DIR / "hypothesis6.txt"

FIGURES_DIR = REPO_ROOT / "Figures"

PNG_PATH = FIGURES_DIR / "hypothesis6.png"

ALPHA = 0.05

ANALYTICAL_MAX = 10.0

EMOTIONAL_MAX = 9.0

SUBPLOT_TITLE_FONTSIZE = 26

AXIS_LABEL_FONTSIZE = 24

TICK_LABEL_FONTSIZE = 20

ANNOTATION_FONTSIZE = 20

SIGNIFICANCE_FONTSIZE = 24

GROUP_COLORS = {"WEIRD": "#377eb8", "NORMAL": "#c26a26"}

TYPE_COLORS = {
    "Analytical": "#377eb8",
    "Emotional": "#c26a26",
}

UPDATED_DESCRIPTION = (
    "Hypothesis 6 tested whether pre-trust and post-trust component contrasts differed by cultural group, and whether analytical and emotional trust changed by different amounts within each cultural group. "
    "The figure includes three analyses in this order: pre-trust component source contrast, post-trust component source contrast, and a single within-group analytical-versus-emotional change contrast."
)

WESTERN_COUNTRIES = {
    "United States", "United Kingdom", "Canada", "Australia", "New Zealand", "Ireland", "Germany", "France",
    "Netherlands", "Belgium", "Switzerland", "Austria", "Denmark", "Sweden", "Norway", "Finland", "Iceland",
    "Luxembourg", "Italy", "Spain", "Portugal",
}

HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}

WEIRD_EMPLOYMENT = {"Full-Time", "Not in paid work (e.g. homemaker', 'retired or disabled)"}

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

def add_pair_bracket(ax: plt.Axes, x1: float, x2: float, y: float, label: str, y_step: float) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=2.0, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, label, ha="center", va="bottom", fontsize=SIGNIFICANCE_FONTSIZE)

def shared_ylim(stats_list: list[dict[str, float]]) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for stats in stats_list:
        lows.extend([stats["an_low"], stats["em_low"]])
        highs.extend([stats["an_high"], stats["em_high"]])

    y_min = float(np.nanmin(np.array(lows, dtype=float)))
    y_max = float(np.nanmax(np.array(highs, dtype=float)))
    y_span = max(y_max - y_min, 0.25)
    return y_min - 0.20 * y_span, y_max + 0.36 * y_span

def draw_component_subplot(
    ax: plt.Axes,
    test_a: dict[str, float],
    test_b: dict[str, float],
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
) -> None:
    x = np.array([0.0, 0.82, 2.1, 2.92], dtype=float)
    means = np.array([test_a["mean_weird"], test_a["mean_normal"], test_b["mean_weird"], test_b["mean_normal"]], dtype=float)
    lows = np.array([test_a["ci_low_weird"], test_a["ci_low_normal"], test_b["ci_low_weird"], test_b["ci_low_normal"]], dtype=float)
    highs = np.array([test_a["ci_high_weird"], test_a["ci_high_normal"], test_b["ci_high_weird"], test_b["ci_high_normal"]], dtype=float)

    ax.bar(x, means, width=0.70, color=[GROUP_COLORS["WEIRD"], GROUP_COLORS["NORMAL"]] * 2, edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#4a4a4a", elinewidth=3.0, capsize=8, capthick=3.0, zorder=3)

    y_top = float(np.nanmax(highs))
    y_bottom = float(np.nanmin(lows))
    y_span = max(y_top - y_bottom, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_top + 0.10 * y_span, p_to_stars(test_a["p_adj"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_top + 0.23 * y_span, p_to_stars(test_b["p_adj"]), 0.03 * y_span)

    ax.text(
        0.01,
        0.99,
        f"Analytical: t = {test_a['t']:.3f} $q {p_text(test_a['p_adj'])}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )
    ax.text(
        0.99,
        0.01,
        f"Emotional: t = {test_b['t']:.3f} $q {p_text(test_b['p_adj'])}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["WEIRD", "Normal", "WEIRD", "Normal"], fontsize=TICK_LABEL_FONTSIZE)
    ax.text(0.41, -0.12, label_a, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE)
    ax.text(2.51, -0.12, label_b, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE, pad=10)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(y_bottom - 0.26 * y_span, y_top + 0.96 * y_span)

def draw_change_contrast_subplot(
    ax: plt.Axes,
    weird_stats: dict[str, float],
    normal_stats: dict[str, float],
    ylim: tuple[float, float],
) -> None:
    x = np.array([0.0, 1.1, 2.8, 3.9], dtype=float)
    means = np.array([weird_stats["an_mean"], weird_stats["em_mean"], normal_stats["an_mean"], normal_stats["em_mean"]], dtype=float)
    lows = np.array([weird_stats["an_low"], weird_stats["em_low"], normal_stats["an_low"], normal_stats["em_low"]], dtype=float)
    highs = np.array([weird_stats["an_high"], weird_stats["em_high"], normal_stats["an_high"], normal_stats["em_high"]], dtype=float)

    ax.bar(x, means, width=0.78, color=[TYPE_COLORS["Analytical"], TYPE_COLORS["Emotional"]] * 2, edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#4a4a4a", elinewidth=3.0, capsize=8, capthick=3.0, zorder=3)

    y_top = float(np.nanmax(highs))
    y_bottom = float(np.nanmin(lows))
    y_span = max(y_top - y_bottom, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_top + 0.08 * y_span, p_to_stars(weird_stats["p_adj"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_top + 0.20 * y_span, p_to_stars(normal_stats["p_adj"]), 0.03 * y_span)

    ax.text(
        0.03,
        0.97,
        f"WEIRD t = {weird_stats['t']:.3f} $q {p_text(weird_stats['p_adj'])}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE - 1,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )
    ax.text(
        0.97,
        0.03,
        f"Normal t = {normal_stats['t']:.3f} $q {p_text(normal_stats['p_adj'])}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE - 1,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Analytical", "Emotional", "Analytical", "Emotional"], fontsize=TICK_LABEL_FONTSIZE - 3)
    ax.text(0.55, -0.12, "WEIRD", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 1)
    ax.text(3.35, -0.12, "Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 1)
    ax.set_title("Analytical vs Emotional\nChange Contrast", fontsize=SUBPLOT_TITLE_FONTSIZE, pad=10)
    ax.set_ylabel("Trust Score Change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(ylim[0], ylim[1] + 0.08 * (ylim[1] - ylim[0]))

# Plot-only entrypoint: statistical calculations and report writing are in the stats module.
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    # Reuse the stats module's group assignment and tests for plotted contrasts and significance annotations.
    analysis = build_analysis_frame(df)
    analytical_pre_test = welch_group_test(analysis, "analytical_pre")
    emotional_pre_test = welch_group_test(analysis, "emotional_pre")
    analytical_post_test = welch_group_test(analysis, "analytical_post")
    emotional_post_test = welch_group_test(analysis, "emotional_post")
    weird = analysis[analysis["Group"] == "WEIRD"].copy()
    normal = analysis[analysis["Group"] == "NORMAL"].copy()
    weird_stats = compare_change_types(weird)
    normal_stats = compare_change_types(normal)
    # Annotate B-H adjusted p-values (q), adjusted across all six sub-hypothesis tests in the order 6.1-6.6.
    ordered = [analytical_pre_test, emotional_pre_test, analytical_post_test, emotional_post_test, weird_stats, normal_stats]
    for st, q in zip(ordered, bh_qvalues([x["p"] for x in ordered])):
        st["p_adj"] = q
    ylim = shared_ylim([weird_stats, normal_stats])
    plt.style.use("ggplot")
    plt.rcParams.update({"xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black"})
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), constrained_layout=True)
    draw_component_subplot(axes[0], analytical_pre_test, emotional_pre_test, label_a="Analytical pre", label_b="Emotional pre", ylabel="Pre trust (normalized)", title="Pre-Trust Component\nSource Contrast")
    draw_component_subplot(axes[1], analytical_post_test, emotional_post_test, label_a="Analytical post", label_b="Emotional post", ylabel="Post trust (normalized)", title="Post-Trust Component\nSource Contrast")
    draw_change_contrast_subplot(axes[2], weird_stats, normal_stats, ylim)
    fig.suptitle("Hypothesis 6: Component and Change Contrasts", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
