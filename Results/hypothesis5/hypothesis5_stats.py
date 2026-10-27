"""Statistical analysis and report generation for hypothesis 5.

This module reads the processed Metrics.csv file, computes the planned tests and
multiple-comparison adjustments, and writes hypothesis5.txt. Plotting lives
in hypothesis5_plot.py.
"""

from __future__ import annotations

from pathlib import Path

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

# Assigns a WEIRD score from the configured demographic indicators; scores of four or more define the WEIRD analysis group.
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

# Normalizes analytical and emotional trust by their scale maxima, then forms overall scores and pre/post change scores.
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

# Compares independent WEIRD and Normal groups with a two-sided Welch t-test; reports each group's mean and 95% t interval, df, t, p, and mean difference.
def welch_group_test(frame: pd.DataFrame, metric_col: str) -> dict[str, float]:
    weird = frame.loc[frame["Group"] == "WEIRD", metric_col].dropna().astype(float)
    normal = frame.loc[frame["Group"] == "NORMAL", metric_col].dropna().astype(float)
    weird_mean, weird_low, weird_high = mean_ci(weird)
    normal_mean, normal_low, normal_high = mean_ci(normal)

    if len(weird) < 2 or len(normal) < 2:
        t_stat, pvalue, dfree, cohens_d = np.nan, np.nan, np.nan, np.nan
    else:
        t_stat, pvalue = ttest_ind(weird, normal, equal_var=False, nan_policy="omit")
        dfree = welch_df(weird, normal)
        # Cohen's d for independent groups: (M_WEIRD - M_Normal) / pooled SD.
        pooled_var = ((len(weird) - 1) * weird.var(ddof=1) + (len(normal) - 1) * normal.var(ddof=1)) / (len(weird) + len(normal) - 2)
        cohens_d = float((weird_mean - normal_mean) / np.sqrt(pooled_var)) if pooled_var > 0 else np.nan

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
        "d": cohens_d,
        "delta_weird_minus_normal": weird_mean - normal_mean,
    }

def build_plot_stats(
    frame: pd.DataFrame,
    pre_col: str,
    post_col: str,
    ) -> dict[str, dict[str, float]]:
    return {
        "pre": welch_group_test(frame, pre_col),
        "post": welch_group_test(frame, post_col),
    }

# Report files contain per-sub-hypothesis descriptive statistics, test statistics, p-values,
# effect sizes/differences when available, and Benjamini-Hochberg adjusted q-values.

# Writes the calculated descriptive values, test statistics, and adjusted significance results.
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

    def timepoint_sentence(label: str, stats: dict[str, float], subhypothesis: str, q: float) -> str:
        return (
            f"Sub-hypothesis {subhypothesis} compared WEIRD and Normal participants, showing that {label} trust was "
            f"$M_{{WEIRD}}={stats['mean_weird']:.3f}$ ({ci_text(stats['ci_low_weird'], stats['ci_high_weird'])}) for WEIRD participants and "
            f"$M_{{Normal}}={stats['mean_normal']:.3f}$ ({ci_text(stats['ci_low_normal'], stats['ci_high_normal'])}) for Normal participants. "
            f"Welch's two-sample t-test found $t({df_text(stats['df'])})={stats['t']:.3f}$, $p {p_text(stats['p'])}$, "
            f"$d={stats['d']:.3f}$, and a difference of $\\Delta(M_W-M_N)={stats['delta_weird_minus_normal']:.3f}$. "
            f"The result {significance_text(stats['p'])}. "
            f"After B-H correction at false discovery rate $q={ALPHA:.2f}$, sub-hypothesis {subhypothesis} "
            f"({q_text(q)}) {bh_status(q)}."
        )

    lines: list[str] = []
    q_values = bh_qvalues([
        analytical_plot["pre"]["p"],
        analytical_plot["post"]["p"],
        emotional_plot["pre"]["p"],
        emotional_plot["post"]["p"],
        overall_plot["pre"]["p"],
        overall_plot["post"]["p"],
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
    lines.append(f"Across $n_{{WEIRD}}={n_weird}$ and $n_{{Normal}}={n_normal}$ participants:")
    for sub, label, plot_stats, key in [
        ("5.1", "analytical pre", analytical_plot, "pre"),
        ("5.2", "analytical post", analytical_plot, "post"),
        ("5.3", "emotional pre", emotional_plot, "pre"),
        ("5.4", "emotional post", emotional_plot, "post"),
        ("5.5", "overall pre", overall_plot, "pre"),
        ("5.6", "overall post", overall_plot, "post"),
    ]:
        lines.append(timepoint_sentence(label, plot_stats[key], sub, q_values[int(sub[-1]) - 1]))
        lines.append("")
    lines.append("")
    bh_parts = [f"sub-hypothesis 5.{i + 1} ({q_text(q_values[i])}) {bh_status(q_values[i])}" for i in range(6)]
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the six primary sub-hypothesis tests, "
        + ", ".join(bh_parts[:5]) + f", and {bh_parts[5]}."
    )
    lines.append("")
    lines.append("")
    lines.append("\\textbf{Hypothesis 5.} WEIRD and Normal participants will differ in trust across three main domains, each evaluated at both the pre- and post-experiment timepoints.")
    lines.append("5.1 Analytical trust will differ between WEIRD and Normal participants at both the pre-experiment and post-experiment timepoints.")
    lines.append("5.2 Emotional trust will differ between WEIRD and Normal participants at both the pre-experiment and post-experiment timepoints.")
    lines.append("5.3 Overall trust will differ between WEIRD and Normal participants at both the pre-experiment and post-experiment timepoints.")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # Generate the six independent group contrasts: three domains, each at pre and post.
    analysis = build_analysis_frame(df)
    analytical_plot = build_plot_stats(analysis, "analytical_pre", "analytical_post")
    emotional_plot = build_plot_stats(analysis, "emotional_pre", "emotional_post")
    overall_plot = build_plot_stats(analysis, "overall_pre", "overall_post")
    write_report(analytical_plot, emotional_plot, overall_plot)
    print(f"Saved report: {TXT_PATH}")


if __name__ == "__main__":
    main()
