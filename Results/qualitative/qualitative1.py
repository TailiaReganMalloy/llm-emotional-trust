from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from Results.qualitative.sentiment1 import (
        DATA_PATH,
        FIGURES_DIR,
        METRICS_TEXT_COLS,
        build_analysis_frame,
        format_p_text,
        get_sentiment_engine,
        normalize_text,
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
        METRICS_TEXT_COLS,
        build_analysis_frame,
        format_p_text,
        get_sentiment_engine,
        normalize_text,
        significance_stars,
        welch_test,
    )

ic_mod = import_module("Results.qualitative.1c3")

SUBPLOT_TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20
STAT_BOX_FONTSIZE = 16


def add_ic_orientation(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    ic_columns: list[str] = []

    for source_col, suffix in METRICS_TEXT_COLS.items():
        if source_col not in work.columns:
            continue
        score_col = f"{suffix}_ic"
        work[score_col] = work[source_col].map(normalize_text).map(ic_mod.score_ic)
        ic_columns.append(score_col)

    if not ic_columns:
        raise ValueError("No IC source text columns found in Metrics.csv")

    work["ic_mean"] = work[ic_columns].mean(axis=1, skipna=True)
    work["ic_group"] = ic_mod.assign_ic_groups_terciles(work["ic_mean"])
    return work


def add_sentiment_polarity(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["sentiment_polarity"] = "neutral"
    work.loc[work["sentiment_mean"] < 0.0, "sentiment_polarity"] = "negative"
    work.loc[work["sentiment_mean"] > 0.0, "sentiment_polarity"] = "positive"
    return work


def add_ame_change(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["ame_change"] = work["analytical_change"] - work["emotional_change"]
    return work


def _box_with_points(
    ax: plt.Axes,
    series_list: list[pd.Series],
    positions: list[float],
    colors: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
) -> None:
    bp = ax.boxplot(
        [s.dropna().astype(float).to_numpy() for s in series_list],
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 1.2},
        capprops={"color": "#444444", "linewidth": 1.2},
    )

    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, edgecolor="white", alpha=0.9)

    rng = np.random.default_rng(7)
    for pos, vals, c in zip(positions, series_list, colors):
        clean = vals.dropna().astype(float).to_numpy()
        if clean.size == 0:
            continue
        jitter = rng.uniform(-0.07, 0.07, size=clean.size)
        ax.scatter(
            np.full(clean.size, pos) + jitter,
            clean,
            s=11,
            alpha=0.45,
            color=c,
            edgecolors="none",
            zorder=3,
        )

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)


def _regression_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
) -> dict[str, float]:
    use = df[[x_col, "ame_change", "Group"]].dropna().copy()
    weird = use[use["Group"] == "WEIRD"]
    normal = use[use["Group"] == "NORMAL"]

    def _binned(group_df: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
        temp = group_df[[x_col, "ame_change"]].dropna().copy()
        if len(temp) < 4:
            return pd.DataFrame(columns=["x_mean", "y_mean", "y_std", "n"])
        bins = pd.qcut(temp[x_col], q=min(n_bins, max(2, len(temp) // 8)), duplicates="drop")
        grouped = temp.groupby(bins, observed=True).agg(
            x_mean=(x_col, "mean"),
            y_mean=("ame_change", "mean"),
            y_std=("ame_change", "std"),
            n=("ame_change", "size"),
        )
        grouped["y_std"] = grouped["y_std"].fillna(0.0)
        return grouped.reset_index(drop=True)

    weird_bins = _binned(weird)
    normal_bins = _binned(normal)

    if not weird_bins.empty:
        ax.errorbar(
            weird_bins["x_mean"],
            weird_bins["y_mean"],
            yerr=weird_bins["y_std"],
            fmt="o",
            color="#377eb8",
            ecolor="#377eb8",
            alpha=0.95,
            capsize=3,
            markersize=6,
            label="WEIRD (bin mean ± sd)",
        )
    if not normal_bins.empty:
        ax.errorbar(
            normal_bins["x_mean"],
            normal_bins["y_mean"],
            yerr=normal_bins["y_std"],
            fmt="o",
            color="#c26a26",
            ecolor="#c26a26",
            alpha=0.95,
            capsize=3,
            markersize=6,
            label="non-WEIRD (bin mean ± sd)",
        )

    weird_lr = stats.linregress(weird[x_col], weird["ame_change"]) if len(weird) >= 3 else None
    normal_lr = stats.linregress(normal[x_col], normal["ame_change"]) if len(normal) >= 3 else None

    if weird_lr is not None:
        x_line_w = np.linspace(float(weird[x_col].min()), float(weird[x_col].max()), 200)
        y_line_w = weird_lr.intercept + weird_lr.slope * x_line_w
        ax.plot(x_line_w, y_line_w, color="#2b5f8c", linewidth=1.8, label="WEIRD fit")

    if normal_lr is not None:
        x_line_n = np.linspace(float(normal[x_col].min()), float(normal[x_col].max()), 200)
        y_line_n = normal_lr.intercept + normal_lr.slope * x_line_n
        ax.plot(x_line_n, y_line_n, color="#8f4f1d", linewidth=1.8, linestyle="--", label="non-WEIRD fit")

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Analytical - Emotional change (z)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="lower left", fontsize=10)

    weird_slope_txt = f"{weird_lr.slope:.3f}" if weird_lr is not None else "NA"
    weird_p_txt = format_p_text(float(weird_lr.pvalue)) if weird_lr is not None else "NA"
    weird_sig_txt = significance_stars(float(weird_lr.pvalue)) if weird_lr is not None else "ns"
    normal_slope_txt = f"{normal_lr.slope:.3f}" if normal_lr is not None else "NA"
    normal_p_txt = format_p_text(float(normal_lr.pvalue)) if normal_lr is not None else "NA"
    normal_sig_txt = significance_stars(float(normal_lr.pvalue)) if normal_lr is not None else "ns"

    ax.text(
        0.02,
        0.98,
        (
            f"n={len(use)}\n"
            f"WEIRD slope={weird_slope_txt}\n"
            f"WEIRD p {weird_p_txt} ({weird_sig_txt})\n"
            f"non-WEIRD slope={normal_slope_txt}\n"
            f"non-WEIRD p {normal_p_txt} ({normal_sig_txt})"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=STAT_BOX_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
    )

    return {
        "n": float(len(use)),
        "weird_slope": float(weird_lr.slope) if weird_lr is not None else np.nan,
        "weird_p": float(weird_lr.pvalue) if weird_lr is not None else np.nan,
        "weird_r2": float(weird_lr.rvalue**2) if weird_lr is not None else np.nan,
        "normal_slope": float(normal_lr.slope) if normal_lr is not None else np.nan,
        "normal_p": float(normal_lr.pvalue) if normal_lr is not None else np.nan,
        "normal_r2": float(normal_lr.rvalue**2) if normal_lr is not None else np.nan,
    }


def make_qualitative1_plot(df: pd.DataFrame, output_path: Path) -> dict[str, dict[str, float]]:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 4, figsize=(26, 7.4), constrained_layout=True)

    # Panel 1: sentiment score distributions by WEIRD vs non-WEIRD.
    sent = df[df["sentiment_mean"].notna()].copy()
    sent_weird = sent[sent["Group"] == "WEIRD"]["sentiment_mean"]
    sent_normal = sent[sent["Group"] == "NORMAL"]["sentiment_mean"]

    _box_with_points(
        axes[0],
        [sent_weird, sent_normal],
        [0.0, 1.0],
        ["#377eb8", "#c26a26"],
        ["WEIRD", "non-WEIRD"],
        "Sentiment Score by Demographic Group",
        "Participant mean sentiment",
    )

    sent_group_p = float(welch_test(sent_weird, sent_normal)["p"])
    sent_neg = sent[sent["sentiment_polarity"] == "negative"]["sentiment_mean"]
    sent_pos = sent[sent["sentiment_polarity"] == "positive"]["sentiment_mean"]
    sent_polarity_p = float(welch_test(sent_neg, sent_pos)["p"])
    axes[0].text(
        0.02,
        0.98,
        (
            f"WEIRD vs non-WEIRD: p {format_p_text(sent_group_p)} ({significance_stars(sent_group_p)})\n"
            f"Negative vs Positive (all): p {format_p_text(sent_polarity_p)} ({significance_stars(sent_polarity_p)})"
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=STAT_BOX_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
    )

    # Panel 2: IC score distributions by WEIRD vs non-WEIRD.
    ic = df[df["ic_mean"].notna()].copy()
    ic_weird = ic[ic["Group"] == "WEIRD"]["ic_mean"]
    ic_normal = ic[ic["Group"] == "NORMAL"]["ic_mean"]

    _box_with_points(
        axes[1],
        [ic_weird, ic_normal],
        [0.0, 1.0],
        ["#377eb8", "#c26a26"],
        ["WEIRD", "non-WEIRD"],
        "IC Score by Demographic Group",
        "Collectivist - Individualistic score",
    )

    ic_group_p = float(welch_test(ic_weird, ic_normal)["p"])
    ic_ind = ic[ic["ic_mean"] < 0.0]["ic_mean"]
    ic_col = ic[ic["ic_mean"] > 0.0]["ic_mean"]
    ic_orientation_p = float(welch_test(ic_ind, ic_col)["p"])
    axes[1].text(
        0.02,
        0.98,
        (
            f"WEIRD vs non-WEIRD: p {format_p_text(ic_group_p)} ({significance_stars(ic_group_p)})\n"
            f"Individualistic vs Collectivist (all): p {format_p_text(ic_orientation_p)} ({significance_stars(ic_orientation_p)})"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=STAT_BOX_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
    )

    # Panels 3 and 4: regressions predicting analytical-emotional change.
    reg_sent = _regression_panel(
        axes[2],
        df,
        x_col="sentiment_mean",
        x_label="Mean sentiment score",
        title="Regression: A-E\nChange ~ Sentiment",
    )
    reg_ic = _regression_panel(
        axes[3],
        df,
        x_col="ic_mean",
        x_label="Collectivity score (IC)",
        title="Regression: A-E\nChange ~ Collectivity",
    )

    fig.suptitle("Qualitative: Distribution Tests and Regression Models", fontsize=26)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)

    return {
        "sentiment_tests": {
            "p_group": sent_group_p,
            "p_negative_vs_positive_all": sent_polarity_p,
            "n_weird": float(sent_weird.dropna().shape[0]),
            "n_normal": float(sent_normal.dropna().shape[0]),
        },
        "ic_tests": {
            "p_group": ic_group_p,
            "p_individualistic_vs_collectivist_all": ic_orientation_p,
            "n_weird": float(ic_weird.dropna().shape[0]),
            "n_normal": float(ic_normal.dropna().shape[0]),
        },
        "regression_sentiment": reg_sent,
        "regression_collectivity": reg_ic,
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    engine = get_sentiment_engine()
    analysis_df = build_analysis_frame(raw, engine)
    analysis_df = add_ic_orientation(analysis_df)
    analysis_df = add_sentiment_polarity(analysis_df)
    analysis_df = add_ame_change(analysis_df)

    output_path = FIGURES_DIR / "qualitative1.png"
    results = make_qualitative1_plot(analysis_df, output_path)

    print("Qualitative1 analysis")
    print("====================")
    print(f"Participants total: {len(analysis_df)}")
    print("Sentiment tests:", results["sentiment_tests"])
    print("IC tests:", results["ic_tests"])
    print("Regression sentiment:", results["regression_sentiment"])
    print("Regression collectivity:", results["regression_collectivity"])
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
