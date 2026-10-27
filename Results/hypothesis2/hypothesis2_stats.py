"""Statistical analysis and report generation for hypothesis 2.

This module reads the processed Metrics.csv file, computes the planned tests and
multiple-comparison adjustments, and writes hypothesis2.txt. Plotting lives
in hypothesis2_plot.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import t, ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

FIGURES_DIR = REPO_ROOT / "Figures"

PLOT_OUTPUT_PATH = FIGURES_DIR / "hypothesis2.png"

REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis2.txt"

ALPHA = 0.05

SUBPLOT_TITLE_FONTSIZE = 26

AXIS_LABEL_FONTSIZE = 26

TICK_LABEL_FONTSIZE = 26

ANNOTATION_FONTSIZE = 24

SIGNIFICANCE_FONTSIZE = 24

CONDITION_ORDER = ["Interactive", "Text"]

CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static",
}

CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}

def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")

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
    return "$q=NA$" if pd.isna(q) else f"$q{format_p(q).replace(' ', '')}$"

def bh_status(q: float) -> str:
    if pd.isna(q):
        return "was not testable"
    return "remained statistically significant" if q < ALPHA else "was not statistically significant after correction"

def normalize_condition(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "interactive": "Interactive",
                "text": "Text",
                "static": "Text",
            }
        )
    )

def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan

    mean_val = float(clean.mean())
    if n == 1:
        return mean_val, mean_val, mean_val

    sem = float(clean.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf((1 + confidence) / 2, df=n - 1) * sem)
    return mean_val, mean_val - margin, mean_val + margin

# Computes Welch-Satterthwaite degrees of freedom from the two sample variances and sizes.
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

# Trust changes are normalized by each scale's maximum before the two independent condition groups are compared.
def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Condition",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = normalize_condition(work["Condition"])
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["overall_change"] = (work["analytical_change"] + work["emotional_change"]) / 2.0

    return work

# Welch's independent-samples t-test does not assume equal variances; the result stores group means, 95% t intervals, t, df, p, and mean difference.
def condition_welch_test(data: pd.DataFrame, value_col: str) -> dict[str, float]:
    interactive_vals = data.loc[data["Condition"] == "Interactive", value_col].dropna().astype(float)
    text_vals = data.loc[data["Condition"] == "Text", value_col].dropna().astype(float)

    i_mean, i_low, i_high = mean_ci(interactive_vals)
    t_mean, t_low, t_high = mean_ci(text_vals)

    if len(interactive_vals) < 2 or len(text_vals) < 2:
        t_stat = np.nan
        p_value = np.nan
        dfree = np.nan
        cohens_d = np.nan
    else:
        t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")
        dfree = welch_df(interactive_vals, text_vals)
        # Cohen's d for independent groups: (M_Interactive - M_Text) / pooled SD.
        pooled_var = ((len(interactive_vals) - 1) * interactive_vals.var(ddof=1) + (len(text_vals) - 1) * text_vals.var(ddof=1)) / (len(interactive_vals) + len(text_vals) - 2)
        cohens_d = float((i_mean - t_mean) / np.sqrt(pooled_var)) if pooled_var > 0 else np.nan

    return {
        "n_interactive": float(len(interactive_vals)),
        "n_text": float(len(text_vals)),
        "mean_interactive": i_mean,
        "mean_text": t_mean,
        "ci_low_interactive": i_low,
        "ci_high_interactive": i_high,
        "ci_low_text": t_low,
        "ci_high_text": t_high,
        "df": dfree,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(p_value) if pd.notna(p_value) else np.nan,
        "d": cohens_d,
        "delta_interactive_minus_text": i_mean - t_mean,
    }

# Report files contain per-sub-hypothesis descriptive statistics, test statistics, p-values,
# effect sizes/differences when available, and Benjamini-Hochberg adjusted q-values.

# Writes the calculated descriptive values, test statistics, and adjusted significance results.
def write_report(
    overall_test: dict[str, float],
    emotional_test: dict[str, float],
    analytical_test: dict[str, float],
) -> str:
    def ci_text(low: float, high: float) -> str:
        if np.isnan(low) or np.isnan(high):
            return "95\\% CI [NA, NA]"
        return f"95\\% CI [{low:.3f}, {high:.3f}]"

    def significance_text(p_value: float, threshold: float = ALPHA) -> str:
        if pd.isna(p_value):
            return "was not testable"
        if p_value < threshold:
            return "was statistically significant"
        return "was not statistically significant"

    def df_text(value: float) -> str:
        if pd.isna(value):
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

    lines: list[str] = []
    q_values = bh_qvalues([analytical_test["p"], emotional_test["p"], overall_test["p"]])

    def condition_paragraph(label: str, name: str, stats: dict[str, float], q: float) -> str:
        return (
            f"Sub-hypothesis {label} compared $n_I={int(stats['n_interactive'])}$ Interactive and $n_T={int(stats['n_text'])}$ Static participants, "
            f"showing that {name} trust change was $M_{{Interactive}}={stats['mean_interactive']:.3f}$ "
            f"({ci_text(stats['ci_low_interactive'], stats['ci_high_interactive'])}) in the Interactive condition and "
            f"$M_{{Text}}={stats['mean_text']:.3f}$ ({ci_text(stats['ci_low_text'], stats['ci_high_text'])}) in the Static condition. "
            f"Welch's two-sample t-test found $t({df_text(stats['df'])})={stats['t']:.3f}$, $p {format_p(stats['p'])}$, "
            f"$d={stats['d']:.3f}$, and a difference of $\\Delta(M_I-M_T)={stats['delta_interactive_minus_text']:.3f}$. "
            f"The result {significance_text(stats['p'])}. "
            f"After B-H correction at false discovery rate $q={ALPHA:.2f}$, sub-hypothesis {label} "
            f"({q_text(q)}) {bh_status(q)}."
        )

    lines.append("\\subsection{Hypothesis 2}")
    lines.append("\\begin{center}")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{Figures/hypothesis2.png}")
    lines.append("\\vspace{-20pt}")
    lines.append("\\captionof{figure}{Visualization for Hypothesis 2.}")
    lines.append("\\label{fig:hypothesis2}")
    lines.append("\\end{center}")
    lines.append("")
    lines.append(
        "Hypothesis 2 tested whether experimental condition changed the magnitude of trust change across analytical, emotional, and overall trust outcomes. Figure \\ref{fig:hypothesis2} summarizes the mean change scores for the Interactive and Static conditions, and the primary analyses used Welch's two-sample t-tests to compare conditions for each outcome."
    )
    lines.append("")

    lines.append(condition_paragraph("2.1", "analytical", analytical_test, q_values[0]))
    lines.append("")
    lines.append(condition_paragraph("2.2", "Emotional", emotional_test, q_values[1]))
    lines.append("")
    lines.append(condition_paragraph("2.3", "overall", overall_test, q_values[2]))
    lines.append("")
    lines.append(
        "Taken together, these results indicate that the Interactive and Static conditions produced comparable changes in trust, so Hypothesis 2 was not supported."
    )
    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the three primary sub-hypothesis tests, sub-hypothesis 2.1 ({q_text(q_values[0])}) {bh_status(q_values[0])}, sub-hypothesis 2.2 ({q_text(q_values[1])}) {bh_status(q_values[1])}, and sub-hypothesis 2.3 ({q_text(q_values[2])}) {bh_status(q_values[2])}."
    )
        
    return "\n".join(lines) + "\n"
def main() -> None:
    data = pd.read_csv(DATA_PATH)
    # Analyze changes for each trust domain and compare Interactive against Text/Static participants.
    analysis = build_analysis_frame(data)
    overall_test = condition_welch_test(analysis, "overall_change")
    emotional_test = condition_welch_test(analysis, "emotional_change")
    analytical_test = condition_welch_test(analysis, "analytical_change")
    report_text = write_report(overall_test, emotional_test, analytical_test)
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
