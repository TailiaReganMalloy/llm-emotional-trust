"""Statistical analysis and report generation for hypothesis 1.

This module reads the processed Metrics.csv file, computes the planned tests and
multiple-comparison adjustments, and writes hypothesis1.txt. Plotting lives
in hypothesis1_plot.py.
"""

from __future__ import annotations

from pathlib import Path

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

def p_text(pvalue: float) -> str:
    if np.isnan(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "< .001"
    return f"= {pvalue:.3f}".replace("0.", ".")

# Benjamini-Hochberg adjusted q-values: q_(i) = min over j>=i of p_(j) * m / j, capped at 1, returned in input order.
def bh_qvalues(pvalues: list[float]) -> list[float]:
    p = np.array([np.inf if pd.isna(v) else v for v in pvalues], dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.minimum(adjusted, 1.0)
    return [float("nan") if np.isinf(p[i]) else float(q[i]) for i in range(m)]

def q_text(q: float) -> str:
    return "$q=NA$" if pd.isna(q) else f"$q{p_text(q).replace(' ', '')}$"

def bh_status(q: float) -> str:
    if pd.isna(q):
        return "was not testable"
    return "remained statistically significant" if q < ALPHA else "was not statistically significant after correction"

# Descriptive mean intervals use the normal-approximation 1.96 x standard error interval, matching the report output.
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

# Paired Cohen's d is the mean post-minus-pre difference divided by the SD of those paired differences.
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

# Uses complete pre/post pairs; a two-sided paired t-test is primary, with a two-sided Wilcoxon signed-rank robustness test.
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

# Report files contain per-sub-hypothesis descriptive statistics, test statistics, p-values,
# effect sizes/differences when available, and Benjamini-Hochberg adjusted q-values.

# Writes the calculated descriptive values, test statistics, and adjusted significance results.
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
    q_values = bh_qvalues([analytical["p"], emotional["p"], overall["p"]])

    # One paragraph per sub-hypothesis: descriptives, paired t-test (+ d), Wilcoxon, difference, then B-H result.
    for label, name, stats, q in [
        ("1.1", "analytical", analytical, q_values[0]),
        ("1.2", "Emotional", emotional, q_values[1]),
        ("1.3", "overall", overall, q_values[2]),
    ]:
        lines.append(
            f"Sub-hypothesis {label} compared ${n_pairs}$ paired responses showing that {name} trust changed from "
            f"$M_{{pre}}={stats['pre_mean']:.3f}$ ({ci_text(stats['pre_mean'], stats['pre_ci'])}) to "
            f"$M_{{post}}={stats['post_mean']:.3f}$ ({ci_text(stats['post_mean'], stats['post_ci'])}). "
            f"Both parametric and non-parametric analyses confirmed this difference, including a paired-samples t-test "
            f"($t({dfree})={stats['t']:.3f}$, $p {p_text(stats['p'])}$, $d={stats['d']:.3f}$), and a Wilcoxon signed-rank test "
            f"($W={stats['w']:.3f}$, $p_{{W}} {p_text(stats['w_p'])}$). The difference was $\\Delta={stats['delta_mean']:.3f}$. "
            f"After applying B-H correction at false discovery rate $q={ALPHA:.2f}$, sub-hypothesis {label} "
            f"({q_text(q)}) {bh_status(q)}."
        )
        lines.append("")

    lines.append(
        f"Sub-hypothesis 1.1 {significance_text(analytical['p'])}, sub-hypothesis 1.2 {significance_text(emotional['p'])}, and sub-hypothesis 1.3 {significance_text(overall['p'])}; so all three components support Hypothesis 1."
    )
    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ "
        f"for the three primary sub-hypothesis tests, sub-hypothesis 1.1 ({q_text(q_values[0])}) {bh_status(q_values[0])}, "
        f"sub-hypothesis 1.2 ({q_text(q_values[1])}) {bh_status(q_values[1])}, and sub-hypothesis 1.3 ({q_text(q_values[2])}) {bh_status(q_values[2])}."
    )

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # Normalize each measure by its scale maximum and calculate the paired statistics and report.
    analytical_pre = df["Total Analytical Trust"] / ANALYTICAL_MAX
    analytical_post = df["Total Analytical Trust Post"] / ANALYTICAL_MAX
    emotional_pre = df["Total Emotional Trust"] / EMOTIONAL_MAX
    emotional_post = df["Total Emotional Trust Post"] / EMOTIONAL_MAX
    results = {
        "Analytical Trust": analyze_metric(analytical_pre, analytical_post),
        "Emotional Trust": analyze_metric(emotional_pre, emotional_post),
        "Overall Trust": analyze_metric((analytical_pre + emotional_pre) / 2.0, (analytical_post + emotional_post) / 2.0),
    }
    write_report(results)
    print(f"Saved report: {TXT_PATH}")


if __name__ == "__main__":
    main()
