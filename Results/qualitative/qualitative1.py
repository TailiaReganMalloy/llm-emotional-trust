from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from Results.qualitative.legacy_metrics import load_legacy_qualitative_frame
    from Results.qualitative.sentiment1 import (
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
    from Results.qualitative.legacy_metrics import load_legacy_qualitative_frame
    from Results.qualitative.sentiment1 import (
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

SUBPLOT_TITLE_FONTSIZE = 26
AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 22
STAT_BOX_FONTSIZE = 20
LEGEND_FONTSIZE = 13


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
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#444444", "linewidth": 1.4},
        capprops={"color": "#444444", "linewidth": 1.4},
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
            s=18,
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


def _combined_distribution_panel(ax: plt.Axes, df: pd.DataFrame) -> tuple[float, float, float, float]:
    sent = df[df["sentiment_mean"].notna()].copy()
    ic = df[df["ic_mean"].notna()].copy()

    sent_weird = sent[sent["Group"] == "WEIRD"]["sentiment_mean"]
    sent_normal = sent[sent["Group"] == "NORMAL"]["sentiment_mean"]
    ic_weird = ic[ic["Group"] == "WEIRD"]["ic_mean"]
    ic_normal = ic[ic["Group"] == "NORMAL"]["ic_mean"]

    positions = [0.0, 0.78, 2.0, 2.78]
    bp = ax.boxplot(
        [
            sent_weird.dropna().astype(float).to_numpy(),
            sent_normal.dropna().astype(float).to_numpy(),
            ic_weird.dropna().astype(float).to_numpy(),
            ic_normal.dropna().astype(float).to_numpy(),
        ],
        positions=positions,
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#444444", "linewidth": 1.4},
        capprops={"color": "#444444", "linewidth": 1.4},
    )

    for box, color in zip(bp["boxes"], ["#377eb8", "#c26a26", "#377eb8", "#c26a26"]):
        box.set(facecolor=color, edgecolor="white", alpha=0.9)

    rng = np.random.default_rng(7)
    for pos, vals, color in zip(positions, [sent_weird, sent_normal, ic_weird, ic_normal], ["#377eb8", "#c26a26", "#377eb8", "#c26a26"]):
        clean = vals.dropna().astype(float).to_numpy()
        if clean.size == 0:
            continue
        jitter = rng.uniform(-0.07, 0.07, size=clean.size)
        ax.scatter(np.full(clean.size, pos) + jitter, clean, s=18, alpha=0.45, color=color, edgecolors="none", zorder=3)

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks([0.39, 2.39])
    ax.set_xticklabels(["Sentiment", "IC"], fontsize=TICK_LABEL_FONTSIZE)
    ax.text(0.39, -0.08, "WEIRD   Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 2)
    ax.text(2.39, -0.08, "WEIRD   Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 2)
    ax.set_title("Sentiment and IC by Demographic Group", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_ylabel("Participant score", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    sent_group_p = float(welch_test(sent_weird, sent_normal)["p"])
    sent_neg = sent[sent["sentiment_polarity"] == "negative"]["sentiment_mean"]
    sent_pos = sent[sent["sentiment_polarity"] == "positive"]["sentiment_mean"]
    sent_polarity_p = float(welch_test(sent_neg, sent_pos)["p"])
    ic_group_p = float(welch_test(ic_weird, ic_normal)["p"])
    ic_ind = ic[ic["ic_mean"] < 0.0]["ic_mean"]
    ic_col = ic[ic["ic_mean"] > 0.0]["ic_mean"]
    ic_orientation_p = float(welch_test(ic_ind, ic_col)["p"])

    ax.text(
        0.02,
        0.98,
        (
            f"Sentiment, WEIRD vs Normal: $p {format_p_text(_adj('sent_group', sent_group_p)).replace(' ', '')}$ ({significance_stars(_adj('sent_group', sent_group_p))})\n"
            f"IC, WEIRD vs Normal: $p {format_p_text(_adj('ic_group', ic_group_p)).replace(' ', '')}$ ({significance_stars(_adj('ic_group', ic_group_p))})"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=STAT_BOX_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
    )

    return sent_group_p, sent_polarity_p, ic_group_p, ic_orientation_p


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
            markersize=7,
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
            markersize=7,
            label="Normal (bin mean ± sd)",
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
        ax.plot(x_line_n, y_line_n, color="#8f4f1d", linewidth=1.8, linestyle="--", label="Normal fit")

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Analytical - Emotional change (z)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)

    weird_adj = _adj(f"{x_col}_weird", float(weird_lr.pvalue)) if weird_lr is not None else np.nan
    normal_adj = _adj(f"{x_col}_normal", float(normal_lr.pvalue)) if normal_lr is not None else np.nan
    weird_p_txt = format_p_text(weird_adj).replace(" ", "") if weird_lr is not None else "NA"
    weird_sig_txt = significance_stars(weird_adj) if weird_lr is not None else "ns"
    normal_p_txt = format_p_text(normal_adj).replace(" ", "") if normal_lr is not None else "NA"
    normal_sig_txt = significance_stars(normal_adj) if normal_lr is not None else "ns"

    ax.text(
        0.02,
        0.98,
        (
            f"WEIRD $p {weird_p_txt}$ ({weird_sig_txt})\n"
            f"Normal $p {normal_p_txt}$ ({normal_sig_txt})"
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


# Benjamini-Hochberg adjusted p-values (p_a) for the six tests annotated in this figure; empty means "show raw p".
P_ADJ: dict[str, float] = {}


def _adj(key: str, raw_p: float) -> float:
    return P_ADJ.get(key, raw_p)


def _bh_adjust(named: dict[str, float]) -> dict[str, float]:
    keys = list(named)
    p = np.array([np.inf if np.isnan(named[k]) else named[k] for k in keys], dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.minimum(adjusted, 1.0)
    return {k: (np.nan if np.isinf(p[i]) else float(q[i])) for i, k in enumerate(keys)}


def make_qualitative1_plot(df: pd.DataFrame, output_path: Path) -> dict[str, dict[str, float]]:
    # First pass draws to a scratch figure only to collect the raw p-values, which are then BH-adjusted together.
    P_ADJ.clear()
    scratch = plt.figure()
    scratch_axes = [scratch.add_subplot(1, 3, i + 1) for i in range(3)]
    sent_p, _, ic_p, _ = _combined_distribution_panel(scratch_axes[0], df)
    raw_sent = _regression_panel(scratch_axes[1], df, x_col="sentiment_mean", x_label="", title="")
    raw_ic = _regression_panel(scratch_axes[2], df, x_col="ic_mean", x_label="", title="")
    plt.close(scratch)
    P_ADJ.update(_bh_adjust({
        "sent_group": sent_p,
        "ic_group": ic_p,
        "sentiment_mean_weird": raw_sent["weird_p"],
        "sentiment_mean_normal": raw_sent["normal_p"],
        "ic_mean_weird": raw_ic["weird_p"],
        "ic_mean_normal": raw_ic["normal_p"],
    }))

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    sent_group_p, sent_polarity_p, ic_group_p, ic_orientation_p = _combined_distribution_panel(axes[0], df)

    sent = df[df["sentiment_mean"].notna()].copy()
    sent_weird = sent[sent["Group"] == "WEIRD"]["sentiment_mean"]
    sent_normal = sent[sent["Group"] == "NORMAL"]["sentiment_mean"]

    ic = df[df["ic_mean"].notna()].copy()
    ic_weird = ic[ic["Group"] == "WEIRD"]["ic_mean"]
    ic_normal = ic[ic["Group"] == "NORMAL"]["ic_mean"]

    reg_sent = _regression_panel(
        axes[1],
        df,
        x_col="sentiment_mean",
        x_label="Mean sentiment score",
        title="Regression: A-E\nChange ~ Sentiment",
    )
    reg_ic = _regression_panel(
        axes[2],
        df,
        x_col="ic_mean",
        x_label="Collectivity score (IC)",
        title="Regression: A-E\nChange ~ Collectivity",
    )

    fig.suptitle("Qualitative: Distribution Tests and Regression Models", fontsize=32)
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

    raw = load_legacy_qualitative_frame()
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
