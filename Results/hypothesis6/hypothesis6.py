from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind, ttest_rel, wilcoxon


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


def cohens_d_paired(a: pd.Series, b: pd.Series) -> float:
    diff = a - b
    if len(diff) < 2:
        return np.nan
    sd = float(diff.std(ddof=1))
    if np.isclose(sd, 0.0):
        return np.nan
    return float(diff.mean() / sd)


def safe_wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 1:
        return np.nan, np.nan
    diff = a - b
    if np.allclose(diff.to_numpy(dtype=float), 0.0):
        return 0.0, 1.0
    stat, pvalue = wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(pvalue)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Group"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")

    work["analytical_pre"] = pd.to_numeric(work["Total Analytical Trust"], errors="coerce") / ANALYTICAL_MAX
    work["analytical_post"] = pd.to_numeric(work["Total Analytical Trust Post"], errors="coerce") / ANALYTICAL_MAX
    work["emotional_pre"] = pd.to_numeric(work["Total Emotional Trust"], errors="coerce") / EMOTIONAL_MAX
    work["emotional_post"] = pd.to_numeric(work["Total Emotional Trust Post"], errors="coerce") / EMOTIONAL_MAX

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]
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


def compare_change_types(frame: pd.DataFrame) -> dict[str, float]:
    pair = frame[["analytical_change", "emotional_change"]].dropna().copy()
    analytical = pair["analytical_change"].astype(float)
    emotional = pair["emotional_change"].astype(float)

    if len(pair) < 2:
        return {
            "n": float(len(pair)),
            "an_mean": np.nan,
            "an_low": np.nan,
            "an_high": np.nan,
            "em_mean": np.nan,
            "em_low": np.nan,
            "em_high": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "diff_mean": np.nan,
        }

    t_stat, pvalue = ttest_rel(analytical, emotional, nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(analytical, emotional)
    an_mean, an_low, an_high = mean_ci(analytical)
    em_mean, em_low, em_high = mean_ci(emotional)

    return {
        "n": float(len(pair)),
        "an_mean": an_mean,
        "an_low": an_low,
        "an_high": an_high,
        "em_mean": em_mean,
        "em_low": em_low,
        "em_high": em_high,
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(analytical, emotional),
        "diff_mean": float((analytical - emotional).mean()),
    }


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
    x = np.array([0.0, 1.0, 3.0, 4.0], dtype=float)
    means = np.array([test_a["mean_weird"], test_a["mean_normal"], test_b["mean_weird"], test_b["mean_normal"]], dtype=float)
    lows = np.array([test_a["ci_low_weird"], test_a["ci_low_normal"], test_b["ci_low_weird"], test_b["ci_low_normal"]], dtype=float)
    highs = np.array([test_a["ci_high_weird"], test_a["ci_high_normal"], test_b["ci_high_weird"], test_b["ci_high_normal"]], dtype=float)

    ax.bar(x, means, width=0.58, color=[GROUP_COLORS["WEIRD"], GROUP_COLORS["NORMAL"]] * 2, edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#4a4a4a", elinewidth=3.0, capsize=8, capthick=3.0, zorder=3)

    y_top = float(np.nanmax(highs))
    y_bottom = float(np.nanmin(lows))
    y_span = max(y_top - y_bottom, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_top + 0.10 * y_span, p_to_stars(test_a["p"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_top + 0.23 * y_span, p_to_stars(test_b["p"]), 0.03 * y_span)

    ax.text(
        0.01,
        0.99,
        f"Analytical: t = {test_a['t']:.3f} p {p_text(test_a['p'])}",
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
        f"Emotional: t = {test_b['t']:.3f} p {p_text(test_b['p'])}",
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
    ax.text(0.5, -0.12, label_a, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE)
    ax.text(3.5, -0.12, label_b, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE)
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
    x = np.array([0.0, 1.7, 4.1, 5.8], dtype=float)
    means = np.array([weird_stats["an_mean"], weird_stats["em_mean"], normal_stats["an_mean"], normal_stats["em_mean"]], dtype=float)
    lows = np.array([weird_stats["an_low"], weird_stats["em_low"], normal_stats["an_low"], normal_stats["em_low"]], dtype=float)
    highs = np.array([weird_stats["an_high"], weird_stats["em_high"], normal_stats["an_high"], normal_stats["em_high"]], dtype=float)

    ax.bar(x, means, width=0.66, color=[TYPE_COLORS["Analytical"], TYPE_COLORS["Emotional"]] * 2, edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#4a4a4a", elinewidth=3.0, capsize=8, capthick=3.0, zorder=3)

    y_top = float(np.nanmax(highs))
    y_bottom = float(np.nanmin(lows))
    y_span = max(y_top - y_bottom, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_top + 0.08 * y_span, p_to_stars(weird_stats["p"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_top + 0.20 * y_span, p_to_stars(normal_stats["p"]), 0.03 * y_span)

    ax.text(
        0.03,
        0.97,
        f"WEIRD t = {weird_stats['t']:.3f} p {p_text(weird_stats['p'])}",
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
        f"Normal t = {normal_stats['t']:.3f} p {p_text(normal_stats['p'])}",
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
    ax.text(0.85, -0.12, "WEIRD", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 1)
    ax.text(4.95, -0.12, "Normal", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK_LABEL_FONTSIZE - 1)
    ax.set_title("Analytical vs Emotional\nChange Contrast", fontsize=SUBPLOT_TITLE_FONTSIZE, pad=10)
    ax.set_ylabel("Trust Score Change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(-0.65, 6.45)
    ax.set_ylim(ylim[0], ylim[1] + 0.08 * (ylim[1] - ylim[0]))


def write_report(
    analytical_pre_test: dict[str, float],
    emotional_pre_test: dict[str, float],
    analytical_post_test: dict[str, float],
    emotional_post_test: dict[str, float],
    weird_stats: dict[str, float],
    normal_stats: dict[str, float],
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

    def group_sentence(group_label: str, subhypothesis: str, stats: dict[str, float]) -> str:
        dfree = int(stats["n"]) - 1 if pd.notna(stats["n"]) and stats["n"] >= 2 else np.nan
        return (
            f"For {group_label} participants ($n={int(stats['n'])}$), analytical trust change was $M_{{A}}={stats['an_mean']:.3f}$ "
            f"({ci_text(stats['an_low'], stats['an_high'])}) and emotional trust change was $M_{{E}}={stats['em_mean']:.3f}$ "
            f"({ci_text(stats['em_low'], stats['em_high'])}); the paired comparison yielded $t({df_text(dfree)})={stats['t']:.3f}$, "
            f"$p {p_text(stats['p'])}$, and $\Delta(M_A-M_E)={stats['diff_mean']:.3f}$. "
            f"Sub-hypothesis {subhypothesis} {significance_text(stats['p'])}."
        )

    def welch_sentence(label: str, stats: dict[str, float], subhypothesis: str) -> str:
        sentence_label = label[0].upper() + label[1:]
        return (
            f"{sentence_label} was $M_{{WEIRD}}={stats['mean_weird']:.3f}$ ({ci_text(stats['ci_low_weird'], stats['ci_high_weird'])}) for WEIRD participants and "
            f"$M_{{Normal}}={stats['mean_normal']:.3f}$ ({ci_text(stats['ci_low_normal'], stats['ci_high_normal'])}) for Normal participants, "
            f"Welch's $t({df_text(stats['df'])})={stats['t']:.3f}$, $p {p_text(stats['p'])}$, and $\Delta(M_W-M_N)={stats['delta_weird_minus_normal']:.3f}$. "
            f"Sub-hypothesis {subhypothesis} {significance_text(stats['p'])}."
        )

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

    lines: list[str] = []
    bh_results = bh_significance_map([
        ("6.1", analytical_pre_test["p"]),
        ("6.2", emotional_pre_test["p"]),
        ("6.3", analytical_post_test["p"]),
        ("6.4", emotional_post_test["p"]),
        ("6.5", weird_stats["p"]),
        ("6.6", normal_stats["p"]),
    ])
    n_weird = int(analytical_pre_test["n_weird"]) if pd.notna(analytical_pre_test["n_weird"]) else 0
    n_normal = int(analytical_pre_test["n_normal"]) if pd.notna(analytical_pre_test["n_normal"]) else 0

    lines.append("\\subsection{Hypothesis 6}")
    lines.append("\\begin{center}")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{Figures/hypothesis6.png}")
    lines.append("\\vspace{-20pt}")
    lines.append("\\captionof{figure}{Visualization for Hypothesis 6.}")
    lines.append("\\label{fig:hypothesis6}")
    lines.append("\\end{center}")
    lines.append("")
    lines.append(UPDATED_DESCRIPTION + " Figure \\ref{fig:hypothesis6} summarizes the pre-trust, post-trust, and within-group change contrasts.")
    lines.append("")

    lines.append(
        f"Across $n_{{WEIRD}}={n_weird}$ and $n_{{Normal}}={n_normal}$ participants, "
        f"{welch_sentence('analytical pre-trust', analytical_pre_test, '6.1')} "
        f"{welch_sentence('emotional pre-trust', emotional_pre_test, '6.2')} "
        f"{welch_sentence('analytical post-trust', analytical_post_test, '6.3')} "
        f"{welch_sentence('emotional post-trust', emotional_post_test, '6.4')}"
    )
    lines.append("")
    lines.append(f"{group_sentence('WEIRD', '6.5', weird_stats)} {group_sentence('Normal', '6.6', normal_stats)}")
    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the six primary sub-hypothesis tests, sub-hypothesis 6.1 {bh_results['6.1']}, sub-hypothesis 6.2 {bh_results['6.2']}, sub-hypothesis 6.3 {bh_results['6.3']}, sub-hypothesis 6.4 {bh_results['6.4']}, sub-hypothesis 6.5 {bh_results['6.5']}, and sub-hypothesis 6.6 {bh_results['6.6']}."
    )
    lines.append("")
    lines.append("")
    lines.append("\\textbf{Hypothesis 6.} WEIRD and Normal participants will differ across three planned analytical-emotional contrast families.")
    lines.append("6.1 A pre-experiment component contrast analysis of analytical and emotional trust will demonstrate significant differences between WEIRD and Normal participants.")
    lines.append("6.2 A post-experiment component contrast analysis of analytical and emotional trust will demonstrate significant differences between WEIRD and Normal participants.")
    lines.append("6.3 A change contrast analysis will test whether the analytical-versus-emotional trust-change pattern differs within the WEIRD and Normal groups.")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    analytical_pre_test = welch_group_test(analysis, "analytical_pre")
    emotional_pre_test = welch_group_test(analysis, "emotional_pre")
    analytical_post_test = welch_group_test(analysis, "analytical_post")
    emotional_post_test = welch_group_test(analysis, "emotional_post")

    weird = analysis[analysis["Group"] == "WEIRD"].copy()
    normal = analysis[analysis["Group"] == "NORMAL"].copy()

    weird_stats = compare_change_types(weird)
    normal_stats = compare_change_types(normal)
    ylim = shared_ylim([weird_stats, normal_stats])

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), constrained_layout=True)

    draw_component_subplot(
        axes[0],
        analytical_pre_test,
        emotional_pre_test,
        label_a="Analytical pre",
        label_b="Emotional pre",
        ylabel="Pre trust (normalized)",
        title="Pre-Trust Component\nSource Contrast",
    )
    draw_component_subplot(
        axes[1],
        analytical_post_test,
        emotional_post_test,
        label_a="Analytical post",
        label_b="Emotional post",
        ylabel="Post trust (normalized)",
        title="Post-Trust Component\nSource Contrast",
    )
    draw_change_contrast_subplot(axes[2], weird_stats, normal_stats, ylim)

    fig.suptitle("Hypothesis 6: Component and Change Contrasts", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(
        analytical_pre_test,
        emotional_pre_test,
        analytical_post_test,
        emotional_post_test,
        weird_stats,
        normal_stats,
    )

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
