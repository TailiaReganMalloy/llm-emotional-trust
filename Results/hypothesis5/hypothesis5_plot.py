"""Visualization generation for hypothesis 5.

This module imports statistical transformations and test results from
hypothesis5_stats.py and writes the corresponding figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from hypothesis5_stats import *

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

TXT_PATH = OUTPUT_DIR / "hypothesis5.txt"

FIGURES_DIR = REPO_ROOT / "Figures"

PNG_PATH = FIGURES_DIR / "hypothesis5.png"

ALPHA = 0.05

ANALYTICAL_MAX = 10.0

EMOTIONAL_MAX = 9.0

SUBPLOT_TITLE_FONTSIZE = 26

AXIS_LABEL_FONTSIZE = 26

TICK_LABEL_FONTSIZE = 26

ANNOTATION_FONTSIZE = 24

SIGNIFICANCE_FONTSIZE = 24

GROUP_COLORS = {"WEIRD": "#377eb8", "NORMAL": "#c26a26"}

GROUP_DISPLAY = {"WEIRD": "WEIRD", "NORMAL": "Normal"}

TIMEPOINT_COLORS = {
    "pre_weird": "#8fb7dd",
    "pre_normal": "#2c6aa0",
    "post_weird": "#f2b36d",
    "post_normal": "#c26a26",
}

WESTERN_COUNTRIES = {
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "New Zealand",
    "Ireland",
    "Germany",
    "France",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Austria",
    "Denmark",
    "Sweden",
    "Norway",
    "Finland",
    "Iceland",
    "Luxembourg",
    "Italy",
    "Spain",
    "Portugal",
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

def shared_ylim(stats_list: list[dict[str, float]]) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for stats in stats_list:
        lows.extend(
            [
                stats["pre"]["ci_low_weird"],
                stats["pre"]["ci_low_normal"],
                stats["post"]["ci_low_weird"],
                stats["post"]["ci_low_normal"],
            ]
        )
        highs.extend(
            [
                stats["pre"]["ci_high_weird"],
                stats["pre"]["ci_high_normal"],
                stats["post"]["ci_high_weird"],
                stats["post"]["ci_high_normal"],
            ]
        )

    y_min = float(np.nanmin(np.array(lows, dtype=float)))
    y_max = float(np.nanmax(np.array(highs, dtype=float)))
    y_span = max(y_max - y_min, 0.25)
    return y_min - 0.22 * y_span, y_max + 0.34 * y_span

def add_pair_bracket(ax: plt.Axes, x1: float, x2: float, y: float, label: str, y_step: float) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=2.0, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, label, ha="center", va="bottom", fontsize=SIGNIFICANCE_FONTSIZE)

def draw_group_subplot(
    ax: plt.Axes,
    title: str,
    stats: dict[str, float],
    show_y_axis: bool,
    ylim: tuple[float, float],
) -> None:
    x = np.array([0.0, 0.82, 2.1, 2.92], dtype=float)
    means = np.array(
        [
            stats["pre"]["mean_weird"],
            stats["pre"]["mean_normal"],
            stats["post"]["mean_weird"],
            stats["post"]["mean_normal"],
        ],
        dtype=float,
    )
    lows = np.array(
        [
            stats["pre"]["ci_low_weird"],
            stats["pre"]["ci_low_normal"],
            stats["post"]["ci_low_weird"],
            stats["post"]["ci_low_normal"],
        ],
        dtype=float,
    )
    highs = np.array(
        [
            stats["pre"]["ci_high_weird"],
            stats["pre"]["ci_high_normal"],
            stats["post"]["ci_high_weird"],
            stats["post"]["ci_high_normal"],
        ],
        dtype=float,
    )

    ax.bar(
        x,
        means,
        width=0.70,
        color=[
            TIMEPOINT_COLORS["pre_weird"],
            TIMEPOINT_COLORS["pre_normal"],
            TIMEPOINT_COLORS["post_weird"],
            TIMEPOINT_COLORS["post_normal"],
        ],
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

    y_top = float(np.nanmax(highs))
    y_bottom = float(np.nanmin(lows))
    y_span = max(y_top - y_bottom, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_top + 0.10 * y_span, p_to_stars(stats["pre"]["p_adj"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_top + 0.23 * y_span, p_to_stars(stats["post"]["p_adj"]), 0.03 * y_span)

    ax.text(
        0.01,
        0.99,
        f"Pre: t = {stats['pre']['t']:.3f} $q {p_text(stats['pre']['p_adj'])}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )
    ax.text(
        0.99,
        0.0,
        f"Post: t = {stats['post']['t']:.3f} $q {p_text(stats['post']['p_adj'])}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks([0.41, 2.51])
    ax.set_xticklabels(["Pre", "Post"], fontsize=TICK_LABEL_FONTSIZE + 2)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE + 2, pad=14)
    ax.text(0.41, -0.12, "WEIRD   Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 3)
    ax.text(2.51, -0.12, "WEIRD   Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 3)
    if show_y_axis:
        ax.set_ylabel("Trust Score", fontsize=AXIS_LABEL_FONTSIZE + 2)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(-0.25, 0.75)

# Plot-only entrypoint: statistical calculations and report writing are in the stats module.
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    # Reuse the stats module's outcomes and Welch tests for plotted means, intervals, and annotations.
    analysis = build_analysis_frame(df)
    analytical_plot = build_plot_stats(analysis, "analytical_pre", "analytical_post")
    emotional_plot = build_plot_stats(analysis, "emotional_pre", "emotional_post")
    overall_plot = build_plot_stats(analysis, "overall_pre", "overall_post")
    # Annotate B-H adjusted p-values (q), adjusted across all six sub-hypothesis tests in the order 5.1-5.6.
    ordered = [
        analytical_plot["pre"], analytical_plot["post"],
        emotional_plot["pre"], emotional_plot["post"],
        overall_plot["pre"], overall_plot["post"],
    ]
    for st, q in zip(ordered, bh_qvalues([x["p"] for x in ordered])):
        st["p_adj"] = q
    ylim = shared_ylim([analytical_plot, emotional_plot, overall_plot])
    plt.style.use("ggplot")
    plt.rcParams.update({"xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black"})
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True, sharey=True)
    draw_group_subplot(axes[0], "Analytical", analytical_plot, show_y_axis=True, ylim=ylim)
    draw_group_subplot(axes[1], "Emotional", emotional_plot, show_y_axis=False, ylim=ylim)
    draw_group_subplot(axes[2], "Overall", overall_plot, show_y_axis=False, ylim=ylim)
    fig.suptitle("Hypothesis 5: Trust Scores by Group Across Pre and Post", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
