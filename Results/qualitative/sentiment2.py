from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from Results.qualitative.sentiment1 import (
        DATA_PATH,
        FIGURES_DIR,
        format_p_text,
        build_analysis_frame,
        get_sentiment_engine,
        mean_ci,
        significance_stars,
        welch_test,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from Results.qualitative.sentiment1 import (
        DATA_PATH,
        FIGURES_DIR,
        format_p_text,
        build_analysis_frame,
        get_sentiment_engine,
        mean_ci,
        significance_stars,
        welch_test,
    )


GROUP_ORDER = ["WEIRD", "NORMAL"]
GROUP_LABELS = {
    "WEIRD": "WEIRD",
    "NORMAL": "Normal",
}
OUTCOMES = ["overall_change", "analytical_change", "emotional_change"]
OUTCOME_LABELS = {
    "overall_change": "Overall",
    "analytical_change": "Analytical",
    "emotional_change": "Emotional",
}


def compute_tercile_cutoffs(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return np.nan, np.nan
    low_cut = float(clean.quantile(1.0 / 3.0))
    high_cut = float(clean.quantile(2.0 / 3.0))
    return low_cut, high_cut


def split_low_high_even(values: pd.Series) -> pd.Series:
    clean = values.dropna().astype(float)
    labels = pd.Series("missing", index=values.index, dtype=object)
    if clean.empty:
        return labels

    ordered = clean.sort_values(kind="mergesort")
    n = len(ordered)
    low_n = n // 2
    labels.loc[ordered.index[:low_n]] = "low"
    labels.loc[ordered.index[low_n:]] = "high"
    return labels


def add_full_sample_split(df: pd.DataFrame, low_cutoff: float, high_cutoff: float) -> pd.DataFrame:
    work = df.copy()
    work["sentiment_group_full"] = "missing"
    work.loc[work["sentiment_mean"] < low_cutoff, "sentiment_group_full"] = "low"
    work.loc[work["sentiment_mean"] > high_cutoff, "sentiment_group_full"] = "high"
    return work


def add_groupwise_split(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["sentiment_group_groupwise"] = "missing"
    for group_name in GROUP_ORDER:
        mask = work["Group"] == group_name
        work.loc[mask, "sentiment_group_groupwise"] = split_low_high_even(work.loc[mask, "sentiment_mean"])
    return work


def compare_outcome(df: pd.DataFrame, group_col: str, outcome: str) -> dict[str, float]:
    low_vals = df.loc[df[group_col] == "low", outcome]
    high_vals = df.loc[df[group_col] == "high", outcome]
    return welch_test(high_vals, low_vals)


def build_panel_summary_lines(df: pd.DataFrame, group_col: str) -> list[str]:
    n_low = int((df[group_col] == "low").sum())
    n_high = int((df[group_col] == "high").sum())

    lines = [f"n_low={n_low}, n_high={n_high}"]
    for outcome in OUTCOMES:
        result = compare_outcome(df, group_col, outcome)
        p_value = float(result["p"])
        lines.append(f"{OUTCOME_LABELS[outcome]}: H-L p {format_p_text(p_value)} ({significance_stars(p_value)})")
    return lines


def _panel_bars(ax: plt.Axes, df: pd.DataFrame, group_col: str, title: str, show_legend: bool = False) -> None:
    x = np.arange(len(OUTCOMES), dtype=float)
    bar_width = 0.34

    low_means: list[float] = []
    low_err_low: list[float] = []
    low_err_high: list[float] = []
    high_means: list[float] = []
    high_err_low: list[float] = []
    high_err_high: list[float] = []

    for outcome in OUTCOMES:
        low_vals = df.loc[df[group_col] == "low", outcome]
        high_vals = df.loc[df[group_col] == "high", outcome]

        l_mean, l_low, l_high = mean_ci(low_vals)
        h_mean, h_low, h_high = mean_ci(high_vals)

        low_means.append(l_mean)
        low_err_low.append(l_mean - l_low)
        low_err_high.append(l_high - l_mean)

        high_means.append(h_mean)
        high_err_low.append(h_mean - h_low)
        high_err_high.append(h_high - h_mean)

    ax.bar(
        x - (bar_width / 2),
        low_means,
        width=bar_width,
        color="#377eb8",
        label="Low sentiment",
        yerr=[low_err_low, low_err_high],
        capsize=4,
        ecolor="#2b5f8c",
    )
    ax.bar(
        x + (bar_width / 2),
        high_means,
        width=bar_width,
        color="#c26a26",
        label="High sentiment",
        yerr=[high_err_low, high_err_high],
        capsize=4,
        ecolor="#8f4f1d",
    )

    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES], fontsize=11)
    ax.set_title(title, fontsize=14)

    summary_lines = build_panel_summary_lines(df, group_col)
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    if show_legend:
        ax.legend(loc="lower left", fontsize=9)


def make_combined_plot(
    full_df: pd.DataFrame,
    weird_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True, constrained_layout=True)

    _panel_bars(axes[0], full_df, "sentiment_group_full", "Full sample", show_legend=True)
    _panel_bars(axes[1], weird_df, "sentiment_group_groupwise", "WEIRD")
    _panel_bars(axes[2], normal_df, "sentiment_group_groupwise", "Normal")

    axes[0].set_ylabel("Normalized trust change", fontsize=13)
    fig.suptitle("Trust-Change Comparison by Sentiment Group", fontsize=18)

    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    engine = get_sentiment_engine()
    analysis_df = build_analysis_frame(raw, engine)

    low_cutoff, high_cutoff = compute_tercile_cutoffs(analysis_df["sentiment_mean"])
    analysis_df = add_full_sample_split(analysis_df, low_cutoff, high_cutoff)
    analysis_df = add_groupwise_split(analysis_df)

    full_df = analysis_df.loc[analysis_df["sentiment_group_full"].isin(["low", "high"])].copy()
    weird_df = analysis_df.loc[(analysis_df["Group"] == "WEIRD")].copy()
    normal_df = analysis_df.loc[(analysis_df["Group"] == "NORMAL")].copy()

    output_path = FIGURES_DIR / "sentiment2.png"
    make_combined_plot(full_df, weird_df, normal_df, output_path)

    print("Sentiment2 combined trust-change comparison")
    print("==========================================")
    print(f"Total participants: {len(analysis_df)}")
    print(f"Full-sample tercile cutoffs: low < {low_cutoff:.6f}, high > {high_cutoff:.6f}")

    print(f"Full sample n_low={(full_df['sentiment_group_full'] == 'low').sum()}, n_high={(full_df['sentiment_group_full'] == 'high').sum()}")
    for outcome in OUTCOMES:
        result = compare_outcome(full_df, "sentiment_group_full", outcome)
        print(f"  Full {outcome}: H-L p={float(result['p']):.6f}")

    print(
        f"WEIRD n={len(weird_df)} "
        f"(low={(weird_df['sentiment_group_groupwise'] == 'low').sum()}, "
        f"high={(weird_df['sentiment_group_groupwise'] == 'high').sum()})"
    )
    for outcome in OUTCOMES:
        result = compare_outcome(weird_df, "sentiment_group_groupwise", outcome)
        print(f"  WEIRD {outcome}: H-L p={float(result['p']):.6f}")

    print(
        f"Normal n={len(normal_df)} "
        f"(low={(normal_df['sentiment_group_groupwise'] == 'low').sum()}, "
        f"high={(normal_df['sentiment_group_groupwise'] == 'high').sum()})"
    )
    for outcome in OUTCOMES:
        result = compare_outcome(normal_df, "sentiment_group_groupwise", outcome)
        print(f"  Normal {outcome}: H-L p={float(result['p']):.6f}")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
