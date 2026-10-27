"""Visualization generation for hypothesis 3.

This module imports statistical transformations and test results from
hypothesis3_stats.py and writes the corresponding figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from hypothesis3_stats import *

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

TXT_PATH = OUTPUT_DIR / "hypothesis3.txt"

FIGURES_DIR = REPO_ROOT / "Figures"

PNG_PATH = FIGURES_DIR / "hypothesis3.png"

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

TYPE_COLORS = {
    "Emotional z": "#377eb8",
    "Analytical z": "#c26a26",
}

ERROR_COLOR = "#4a4a4a"

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

WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}

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

def add_stat_box(ax: plt.Axes, stats: dict[str, float]) -> None:
    pvalue = float(stats["p_adj"]) if pd.notna(stats.get("p_adj")) else np.nan
    tvalue = float(stats["t"]) if pd.notna(stats.get("t")) else np.nan
    summary = f"t = {tvalue:.3f} $q {p_text(pvalue)}$" if pd.notna(tvalue) else f"t = NA $q {p_text(pvalue)}$"
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

def pre_post_ylim(stats_list: list[dict[str, float]]) -> tuple[float, float]:
    tops: list[float] = []
    bottoms: list[float] = []
    for stats in stats_list:
        means = np.array([stats["pre_mean"], stats["post_mean"]], dtype=float)
        cis = np.nan_to_num(np.array([stats["pre_ci"], stats["post_ci"]], dtype=float), nan=0.0)
        tops.append(float(np.nanmax(means + cis)))
        bottoms.append(float(np.nanmin(means - cis)))

    y_top = max(tops)
    y_bottom = min(bottoms)
    y_span = max(y_top - y_bottom, 0.25)
    return y_bottom - 0.15 * y_span, y_top + 0.20 * y_span

def draw_pre_post_subplot(
    ax: plt.Axes,
    title: str,
    stats: dict[str, float],
    show_y_axis: bool,
    ylim: tuple[float, float],
) -> None:
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
    ax.text(0.5, y_text, p_to_stars(stats["p_adj"]), ha="center", va="bottom", fontsize=24, zorder=5)
    add_stat_box(ax, stats)
    ax.set_ylim(bottom=-0.3, top=0.5)

def draw_type_subplot(ax: plt.Axes, title: str, stats: dict[str, float]) -> None:
    labels = ["Emotional z", "Analytical z"]
    x = np.array([0.0, 1.0], dtype=float)
    means = np.array([stats["em_mean"], stats["an_mean"]], dtype=float)
    cis = np.array([stats["em_ci"], stats["an_ci"]], dtype=float)
    cis_clean = np.nan_to_num(cis, nan=0.0)

    for idx, label in enumerate(labels):
        ax.bar(
            x[idx],
            means[idx],
            width=0.8,
            color=TYPE_COLORS[label],
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
    ax.set_ylabel("Standardized Change", fontsize=AXIS_LABEL_FONTSIZE + 2)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    ax.axhline(0.0, color=ERROR_COLOR, linewidth=1.6, zorder=1)

    y_top = float(np.nanmax(means + cis_clean))
    y_bottom = float(np.nanmin(means - cis_clean))
    y_span = max(y_top - y_bottom, 0.25)

    y_bracket = y_top + 0.10 * y_span
    y_text = y_bracket + 0.03 * y_span
    y_low = y_bottom - 0.18 * y_span
    y_high = y_bracket + 0.22 * y_span

    ax.plot(
        [0, 0, 1, 1],
        [y_bracket - 0.02 * y_span, y_bracket, y_bracket, y_bracket - 0.02 * y_span],
        color="black",
        lw=2.0,
        zorder=4,
    )
    ax.text(0.5, y_text, p_to_stars(stats["p_adj"]), ha="center", va="bottom", fontsize=SIGNIFICANCE_FONTSIZE, zorder=5)
    add_stat_box(ax, stats)
    ax.set_ylim(bottom=-0.3, top=0.5)

# Plot-only entrypoint: statistical calculations and report writing are in the stats module.
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    # Reuse the stats module's cohort and paired-test helpers for consistent plot values and annotations.
    analysis = build_analysis_frame(df)
    weird = analysis[analysis["is_weird"]].copy()
    overall_stats = paired_pre_post(weird["overall_pre"], weird["overall_post"])
    analytical_stats = paired_pre_post(weird["analytical_pre"], weird["analytical_post"])
    emotional_stats = paired_pre_post(weird["emotional_pre"], weird["emotional_post"])
    # Annotate B-H adjusted p-values (q), adjusted across the three sub-hypothesis tests as in the stats report.
    all_stats = [analytical_stats, emotional_stats, overall_stats]
    for st, q in zip(all_stats, bh_qvalues([x["p"] for x in all_stats])):
        st["p_adj"] = q
    trust_ylim = pre_post_ylim([analytical_stats, emotional_stats, overall_stats])
    plt.style.use("ggplot")
    plt.rcParams.update({"xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black"})
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True, sharey=True)
    draw_pre_post_subplot(axes[0], "Analytical Trust", analytical_stats, show_y_axis=True, ylim=trust_ylim)
    draw_pre_post_subplot(axes[1], "Emotional Trust", emotional_stats, show_y_axis=False, ylim=trust_ylim)
    draw_pre_post_subplot(axes[2], "Overall Trust", overall_stats, show_y_axis=False, ylim=trust_ylim)
    fig.suptitle("Hypothesis 3: WEIRD Trust Change Pattern", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
