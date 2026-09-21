from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind


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


def score_weird(row: pd.Series) -> int:
    score = 0
    if row.get("Country of residence") in WESTERN_COUNTRIES:
        score += 1
    if str(row.get("Language", "")).lower().startswith("english"):
        score += 1
    if row.get("Education") in HIGH_EDUCATION:
        score += 1
    if row.get("Employment status") in WEIRD_EMPLOYMENT:
        score += 1
    if str(row.get("Ethnicity simplified", "")).strip().lower() == "white":
        score += 1
    if row.get("Nationality") in WESTERN_COUNTRIES or row.get("Country of birth") in WESTERN_COUNTRIES:
        score += 1
    return score


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


def p_text(pvalue: float) -> str:
    if np.isnan(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "< .001"
    return f"= {pvalue:.3f}".replace("0.", ".")


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    n = len(values)
    if n < 2:
        return mean, mean, mean
    sem = float(values.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf(0.975, df=n - 1) * sem)
    return mean, mean - margin, mean + margin


def welch_df(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna().astype(float)
    b = sample_b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan

    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    n_a = len(a)
    n_b = len(b)
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    if denominator <= np.finfo(float).tiny:
        return np.nan
    return float(numerator / denominator)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Group"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")

    work["analytical_pre"] = pd.to_numeric(work["Total Analytical Trust"], errors="coerce") / ANALYTICAL_MAX
    work["analytical_post"] = pd.to_numeric(work["Total Analytical Trust Post"], errors="coerce") / ANALYTICAL_MAX
    work["emotional_pre"] = pd.to_numeric(work["Total Emotional Trust"], errors="coerce") / EMOTIONAL_MAX
    work["emotional_post"] = pd.to_numeric(work["Total Emotional Trust Post"], errors="coerce") / EMOTIONAL_MAX

    work["overall_pre"] = (work["analytical_pre"] + work["emotional_pre"]) / 2.0
    work["overall_post"] = (work["analytical_post"] + work["emotional_post"]) / 2.0

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]
    work["overall_change"] = work["overall_post"] - work["overall_pre"]

    work["analytical_change_z"] = zscore(work["analytical_change"].astype(float))
    work["emotional_change_z"] = zscore(work["emotional_change"].astype(float))
    work["overall_change_z"] = zscore(work["overall_change"].astype(float))
    return work


def welch_group_test(frame: pd.DataFrame, metric_col: str) -> dict[str, float]:
    weird = frame.loc[frame["Group"] == "WEIRD", metric_col].dropna().astype(float)
    normal = frame.loc[frame["Group"] == "NORMAL", metric_col].dropna().astype(float)
    weird_mean, weird_low, weird_high = mean_ci(weird)
    normal_mean, normal_low, normal_high = mean_ci(normal)

    if len(weird) < 2 or len(normal) < 2:
        t_stat, pvalue, dfree = np.nan, np.nan, np.nan
    else:
        t_stat, pvalue = ttest_ind(weird, normal, equal_var=False, nan_policy="omit")
        dfree = welch_df(weird, normal)

    return {
        "n_weird": float(len(weird)),
        "n_normal": float(len(normal)),
        "mean_weird": weird_mean,
        "mean_normal": normal_mean,
        "ci_low_weird": weird_low,
        "ci_high_weird": weird_high,
        "ci_low_normal": normal_low,
        "ci_high_normal": normal_high,
        "df": dfree,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(pvalue) if pd.notna(pvalue) else np.nan,
        "delta_weird_minus_normal": weird_mean - normal_mean,
    }


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
    x = np.array([0.0, 1.0, 3.0, 4.0], dtype=float)
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
        width=0.58,
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
    add_pair_bracket(ax, x[0], x[1], y_top + 0.10 * y_span, p_to_stars(stats["pre"]["p"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_top + 0.23 * y_span, p_to_stars(stats["post"]["p"]), 0.03 * y_span)

    ax.text(
        0.01,
        0.99,
        f"Pre: t = {stats['pre']['t']:.3f} p {p_text(stats['pre']['p'])}",
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
        f"Post: t = {stats['post']['t']:.3f} p {p_text(stats['post']['p'])}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Pre", "Post"], fontsize=TICK_LABEL_FONTSIZE + 2)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE + 2, pad=14)
    ax.text(0.5, -0.12, "WEIRD   Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 3)
    ax.text(3.5, -0.12, "WEIRD   Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 3)
    if show_y_axis:
        ax.set_ylabel("Trust Score", fontsize=AXIS_LABEL_FONTSIZE + 2)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(-0.25, 0.75)


def build_plot_stats(
    frame: pd.DataFrame,
    pre_col: str,
    post_col: str,
    ) -> dict[str, dict[str, float]]:
    return {
        "pre": welch_group_test(frame, pre_col),
        "post": welch_group_test(frame, post_col),
    }


def write_report(
    analytical_plot: dict[str, dict[str, float]],
    emotional_plot: dict[str, dict[str, float]],
    overall_plot: dict[str, dict[str, float]],
) -> None:
    def ci_text(low: float, high: float) -> str:
        if np.isnan(low) or np.isnan(high):
            return "95\\% CI [NA, NA]"
        return f"95\\% CI [{low:.3f}, {high:.3f}]"

    def significance_text(pvalue: float, threshold: float = ALPHA) -> str:
        if np.isnan(pvalue):
            return "was not testable"
        if pvalue < threshold:
            return "was statistically significant"
        return "was not statistically significant"

    def df_text(value: float) -> str:
        if np.isnan(value):
            return "NA"
        return f"{value:.2f}"

    def bh_significance_map(named_pvalues: list[tuple[str, float]]) -> dict[str, str]:
        pvalues = np.array([np.inf if np.isnan(pvalue) else pvalue for _, pvalue in named_pvalues], dtype=float)
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]
        thresholds = ALPHA * (np.arange(1, len(named_pvalues) + 1, dtype=float) / len(named_pvalues))
        rejected = np.zeros(len(named_pvalues), dtype=bool)
        passing = np.where(sorted_p <= thresholds)[0]
        if len(passing):
            rejected[order[: passing[-1] + 1]] = True

        results_map: dict[str, str] = {}
        for idx, (name, pvalue) in enumerate(named_pvalues):
            if np.isnan(pvalue):
                results_map[name] = "was not testable"
            elif rejected[idx]:
                results_map[name] = "was statistically significant"
            else:
                results_map[name] = "was not statistically significant"
        return results_map

    def timepoint_sentence(label: str, stats: dict[str, float], subhypothesis: str) -> str:
        return (
            f"{label} trust was $M_{{WEIRD}}={stats['mean_weird']:.3f}$ "
            f"({ci_text(stats['ci_low_weird'], stats['ci_high_weird'])}) for WEIRD participants and "
            f"$M_{{Normal}}={stats['mean_normal']:.3f}$ ({ci_text(stats['ci_low_normal'], stats['ci_high_normal'])}) for Normal participants, "
            f"Welch's $t({df_text(stats['df'])})={stats['t']:.3f}$, $p {p_text(stats['p'])}$, and $\\Delta(M_W-M_N)={stats['delta_weird_minus_normal']:.3f}$. "
            f"Sub-hypothesis {subhypothesis} {significance_text(stats['p'])}."
        )

    lines: list[str] = []
    bh_results = bh_significance_map([
        ("5.1", analytical_plot["pre"]["p"]),
        ("5.2", analytical_plot["post"]["p"]),
        ("5.3", emotional_plot["pre"]["p"]),
        ("5.4", emotional_plot["post"]["p"]),
        ("5.5", overall_plot["pre"]["p"]),
        ("5.6", overall_plot["post"]["p"]),
    ])
    n_weird = int(analytical_plot["pre"]["n_weird"]) if pd.notna(analytical_plot["pre"]["n_weird"]) else 0
    n_normal = int(analytical_plot["pre"]["n_normal"]) if pd.notna(analytical_plot["pre"]["n_normal"]) else 0

    lines.append("\\subsection{Hypothesis 5}")
    lines.append("\\begin{center}")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{Figures/hypothesis5.png}")
    lines.append("\\vspace{-20pt}")
    lines.append("\\captionof{figure}{Visualization for Hypothesis 5.}")
    lines.append("\\label{fig:hypothesis5}")
    lines.append("\\end{center}")
    lines.append("")
    lines.append(
        "Hypothesis 5 tested whether WEIRD and Normal participants differed in analytical, emotional, and overall trust at the pre- and post-experiment timepoints. Figure \\ref{fig:hypothesis5} shows the normalized trust scores for each group, with separate WEIRD-versus-Normal tests at pre and post within each trust domain."
    )
    lines.append("")
    lines.append(
        f"Across $n_{{WEIRD}}={n_weird}$ and $n_{{Normal}}={n_normal}$ participants, "
        f"{timepoint_sentence('Analytical pre', analytical_plot['pre'], '5.1')} "
        f"{timepoint_sentence('Analytical post', analytical_plot['post'], '5.2')}"
    )
    lines.append("")
    lines.append(
        f"{timepoint_sentence('Emotional pre', emotional_plot['pre'], '5.3')} "
        f"{timepoint_sentence('Emotional post', emotional_plot['post'], '5.4')}"
    )
    lines.append("")
    lines.append(
        f"{timepoint_sentence('Overall pre', overall_plot['pre'], '5.5')} "
        f"{timepoint_sentence('Overall post', overall_plot['post'], '5.6')}"
    )
    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the six primary sub-hypothesis tests, sub-hypothesis 5.1 {bh_results['5.1']}, sub-hypothesis 5.2 {bh_results['5.2']}, sub-hypothesis 5.3 {bh_results['5.3']}, sub-hypothesis 5.4 {bh_results['5.4']}, sub-hypothesis 5.5 {bh_results['5.5']}, and sub-hypothesis 5.6 {bh_results['5.6']}."
    )
    lines.append("")
    lines.append("")
    lines.append("\\textbf{Hypothesis 5.} WEIRD and Normal participants will differ in trust across three main domains, each evaluated at both the pre- and post-experiment timepoints.")
    lines.append("5.1 Analytical trust will differ between WEIRD and Normal participants at both the pre-experiment and post-experiment timepoints.")
    lines.append("5.2 Emotional trust will differ between WEIRD and Normal participants at both the pre-experiment and post-experiment timepoints.")
    lines.append("5.3 Overall trust will differ between WEIRD and Normal participants at both the pre-experiment and post-experiment timepoints.")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    analytical_plot = build_plot_stats(analysis, "analytical_pre", "analytical_post")
    emotional_plot = build_plot_stats(analysis, "emotional_pre", "emotional_post")
    overall_plot = build_plot_stats(analysis, "overall_pre", "overall_post")
    ylim = shared_ylim([analytical_plot, emotional_plot, overall_plot])

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True, sharey=True)

    draw_group_subplot(axes[0], "Analytical", analytical_plot, show_y_axis=True, ylim=ylim)
    draw_group_subplot(axes[1], "Emotional", emotional_plot, show_y_axis=False, ylim=ylim)
    draw_group_subplot(axes[2], "Overall", overall_plot, show_y_axis=False, ylim=ylim)

    fig.suptitle("Hypothesis 5: Trust Scores by Group Across Pre and Post", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(analytical_plot, emotional_plot, overall_plot)

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
