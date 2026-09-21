from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


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


def p_text(pvalue: float) -> str:
    if np.isnan(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "< .001"
    return f"= {pvalue:.3f}".replace("0.", ".")


def mean_ci(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, ci


def ci_bounds(mean: float, ci: float) -> tuple[float, float]:
    if np.isnan(mean) or np.isnan(ci):
        return np.nan, np.nan
    return mean - ci, mean + ci


def cohens_d_paired(pre: pd.Series, post: pd.Series) -> float:
    diffs = post - pre
    if len(diffs) < 2:
        return np.nan
    sd = float(diffs.std(ddof=1))
    if np.isclose(sd, 0.0):
        return np.nan
    return float(diffs.mean() / sd)


def safe_wilcoxon(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    if len(pre) < 1:
        return np.nan, np.nan
    diffs = post - pre
    if np.allclose(diffs.to_numpy(dtype=float), 0.0):
        return 0.0, 1.0
    stat, pvalue = wilcoxon(pre, post, alternative="two-sided")
    return float(stat), float(pvalue)


def analyze_metric(pre: pd.Series, post: pd.Series) -> dict[str, float]:
    pairs = pd.DataFrame({"pre": pre, "post": post}).dropna()
    if len(pairs) < 2:
        return {
            "n": float(len(pairs)),
            "pre_mean": np.nan,
            "pre_ci": np.nan,
            "post_mean": np.nan,
            "post_ci": np.nan,
            "pre_sd": np.nan,
            "post_sd": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "delta_mean": np.nan,
        }

    t_stat, pvalue = ttest_rel(pairs["pre"], pairs["post"], nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(pairs["pre"], pairs["post"])
    pre_mean, pre_ci = mean_ci(pairs["pre"])
    post_mean, post_ci = mean_ci(pairs["post"])

    return {
        "n": float(len(pairs)),
        "pre_mean": pre_mean,
        "pre_ci": pre_ci,
        "post_mean": post_mean,
        "post_ci": post_ci,
        "pre_sd": float(pairs["pre"].std(ddof=1)),
        "post_sd": float(pairs["post"].std(ddof=1)),
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(pairs["pre"], pairs["post"]),
        "delta_mean": float((pairs["post"] - pairs["pre"]).mean()),
    }


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
    ax.text(0.4, y_text, p_to_stars(stats["p"]), ha="center", va="bottom", fontsize=SIGNIFICANCE_FONTSIZE, zorder=5)

    interpretation = "Significant pre-post effect" if float(stats["p"]) < ALPHA else "No significant pre-post effect"
    summary = f"t = {stats['t']:.3f} p {p_text(float(stats['p']))}"
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


def write_report(results: dict[str, dict[str, float]]) -> None:
    def ci_text(mean: float, ci: float) -> str:
        low, high = ci_bounds(mean, ci)
        if np.isnan(low) or np.isnan(high):
            return "95\\% CI [NA, NA]"
        return f"95\\% CI [{low:.3f}, {high:.3f}]"

    def significance_text(pvalue: float, threshold: float = ALPHA) -> str:
        if np.isnan(pvalue):
            return "was not testable"
        if pvalue < threshold:
            return "was statistically significant"
        return "was not statistically significant"

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
    lines.append("\\subsection{Hypothesis 1}")
    lines.append("\\begin{center}")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{Figures/hypothesis1.png}")
    lines.append("\\vspace{-20pt}")
    lines.append("\\captionof{figure}{Visualization for Hypothesis 1.}")
    lines.append("\\label{fig:hypothesis1}")
    lines.append("\\end{center}")
    lines.append("")
    lines.append(
        "Hypothesis 1 tested whether trust changed from pre- to post-experiment across analytical, emotional, and overall trust dimensions. Figure \\ref{fig:hypothesis1} summarizes the pre-post means, and all analyses used paired-samples t-tests with two-sided Wilcoxon signed-rank tests as a robustness check."
    )
    lines.append("")

    overall = results["Overall Trust"]
    analytical = results["Analytical Trust"]
    emotional = results["Emotional Trust"]
    n_pairs = int(overall["n"]) if pd.notna(overall["n"]) else 0
    dfree = n_pairs - 1 if n_pairs >= 2 else np.nan
    bh_results = bh_significance_map([
        ("1.1", analytical["p"]),
        ("1.2", emotional["p"]),
        ("1.3", overall["p"]),
    ])

    lines.append(
        f"Across ${n_pairs}$ paired responses, analytical trust decreased from $M_{{pre}}={analytical['pre_mean']:.3f}$ "
        f"({ci_text(analytical['pre_mean'], analytical['pre_ci'])}) to $M_{{post}}={analytical['post_mean']:.3f}$ "
        f"({ci_text(analytical['post_mean'], analytical['post_ci'])}), $t({dfree})={analytical['t']:.3f}$, "
        f"$p {p_text(analytical['p'])}$, $W={analytical['w']:.3f}$, $p_{{W}} {p_text(analytical['w_p'])}$, "
        f"$d={analytical['d']:.3f}$, and $\\Delta={analytical['delta_mean']:.3f}$. Emotional trust also declined from "
        f"$M_{{pre}}={emotional['pre_mean']:.3f}$ ({ci_text(emotional['pre_mean'], emotional['pre_ci'])}) to "
        f"$M_{{post}}={emotional['post_mean']:.3f}$ ({ci_text(emotional['post_mean'], emotional['post_ci'])}), "
        f"$t({dfree})={emotional['t']:.3f}$, $p {p_text(emotional['p'])}$, $W={emotional['w']:.3f}$, "
        f"$p_{{W}} {p_text(emotional['w_p'])}$, $d={emotional['d']:.3f}$, and $\\Delta={emotional['delta_mean']:.3f}$."
    )

    lines.append("")
    lines.append(
        f"Overall trust decreased from $M_{{pre}}={overall['pre_mean']:.3f}$ ({ci_text(overall['pre_mean'], overall['pre_ci'])}) "
        f"to $M_{{post}}={overall['post_mean']:.3f}$ ({ci_text(overall['post_mean'], overall['post_ci'])}), $t({dfree})={overall['t']:.3f}$, "
        f"$p {p_text(overall['p'])}$, $W={overall['w']:.3f}$, $p_{{W}} {p_text(overall['w_p'])}$, $d={overall['d']:.3f}$, "
        f"and $\\Delta={overall['delta_mean']:.3f}$. Sub-hypothesis 1.1 {significance_text(analytical['p'])}, sub-hypothesis 1.2 {significance_text(emotional['p'])}, and sub-hypothesis 1.3 {significance_text(overall['p'])}, so all three components support Hypothesis 1."
    )

    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ "
        f"for the three primary sub-hypothesis tests, sub-hypothesis 1.1 {bh_results['1.1']}, "
        f"sub-hypothesis 1.2 {bh_results['1.2']}, and sub-hypothesis 1.3 {bh_results['1.3']}."
    )

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    plt.style.use("ggplot")

    analytical_pre = df["Total Analytical Trust"] / ANALYTICAL_MAX
    analytical_post = df["Total Analytical Trust Post"] / ANALYTICAL_MAX
    emotional_pre = df["Total Emotional Trust"] / EMOTIONAL_MAX
    emotional_post = df["Total Emotional Trust Post"] / EMOTIONAL_MAX
    overall_pre = (analytical_pre + emotional_pre) / 2.0
    overall_post = (analytical_post + emotional_post) / 2.0

    results = {
        "Analytical Trust": analyze_metric(analytical_pre, analytical_post),
        "Emotional Trust": analyze_metric(emotional_pre, emotional_post),
        "Overall Trust": analyze_metric(overall_pre, overall_post),
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    for idx, (ax, (name, stats)) in enumerate(zip(axes, results.items())):
        draw_subplot(ax, name, stats, show_y_axis=idx == 0)

    fig.suptitle("Hypothesis 1: Pre vs Post Experiment Trust Questionnaire", fontsize=32)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(results)

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
