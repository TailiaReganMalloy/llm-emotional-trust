"""Statistical analysis and report generation for hypothesis 6.

This module reads the processed Metrics.csv file, computes the planned tests and
multiple-comparison adjustments, and writes hypothesis6.txt. Plotting lives
in hypothesis6_plot.py.
"""

from __future__ import annotations

from pathlib import Path

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

# Compares independent WEIRD and Normal groups with a two-sided Welch t-test; reports group means, 95% t intervals, df, t, p, and mean difference.
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

# Within each cultural group, compares paired analytical and emotional change scores with a two-sided paired t-test and Wilcoxon robustness test.
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

# Report files contain per-sub-hypothesis descriptive statistics, test statistics, p-values,
# effect sizes/differences when available, and Benjamini-Hochberg adjusted q-values.

# Writes the calculated descriptive values, test statistics, and adjusted significance results.
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

    def group_sentence(group_label: str, subhypothesis: str, stats: dict[str, float], q: float) -> str:
        dfree = int(stats["n"]) - 1 if pd.notna(stats["n"]) and stats["n"] >= 2 else np.nan
        return (
            f"Sub-hypothesis {subhypothesis} compared analytical and emotional trust change for {group_label} participants ($n={int(stats['n'])}$): "
            f"analytical trust change was $M_{{A}}={stats['an_mean']:.3f}$ ({ci_text(stats['an_low'], stats['an_high'])}) and "
            f"emotional trust change was $M_{{E}}={stats['em_mean']:.3f}$ ({ci_text(stats['em_low'], stats['em_high'])}). "
            f"The paired-samples t-test found $t({df_text(dfree)})={stats['t']:.3f}$, $p {p_text(stats['p'])}$, "
            f"$d={stats['d']:.3f}$, and a Wilcoxon signed-rank test had $W={stats['w']:.3f}$, $p_W {p_text(stats['w_p'])}$. "
            f"The difference was $\\Delta(M_A-M_E)={stats['diff_mean']:.3f}$. The result {significance_text(stats['p'])}. "
            f"After B-H correction at false discovery rate $q={ALPHA:.2f}$, sub-hypothesis {subhypothesis} "
            f"({q_text(q)}) {bh_status(q)}."
        )

    def welch_sentence(label: str, stats: dict[str, float], subhypothesis: str, q: float) -> str:
        return (
            f"Sub-hypothesis {subhypothesis} compared WEIRD and Normal participants, showing that {label} was "
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
        analytical_pre_test["p"],
        emotional_pre_test["p"],
        analytical_post_test["p"],
        emotional_post_test["p"],
        weird_stats["p"],
        normal_stats["p"],
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

    lines.append(f"Across $n_{{WEIRD}}={n_weird}$ and $n_{{Normal}}={n_normal}$ participants:")
    for sub, label, test in [
        ("6.1", "analytical pre-trust", analytical_pre_test),
        ("6.2", "emotional pre-trust", emotional_pre_test),
        ("6.3", "analytical post-trust", analytical_post_test),
        ("6.4", "emotional post-trust", emotional_post_test),
    ]:
        lines.append(welch_sentence(label, test, sub, q_values[int(sub[-1]) - 1]))
        lines.append("")
    lines.append(group_sentence("WEIRD", "6.5", weird_stats, q_values[4]))
    lines.append("")
    lines.append(group_sentence("Normal", "6.6", normal_stats, q_values[5]))
    lines.append("")
    bh_parts = [f"sub-hypothesis 6.{i + 1} ({q_text(q_values[i])}) {bh_status(q_values[i])}" for i in range(6)]
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the six primary sub-hypothesis tests, "
        + ", ".join(bh_parts[:5]) + f", and {bh_parts[5]}."
    )
    lines.append("")
    lines.append("")
    lines.append("\\textbf{Hypothesis 6.} WEIRD and Normal participants will differ across three planned analytical-emotional contrast families.")
    lines.append("6.1 A pre-experiment component contrast analysis of analytical and emotional trust will demonstrate significant differences between WEIRD and Normal participants.")
    lines.append("6.2 A post-experiment component contrast analysis of analytical and emotional trust will demonstrate significant differences between WEIRD and Normal participants.")
    lines.append("6.3 A change contrast analysis will test whether the analytical-versus-emotional trust-change pattern differs within the WEIRD and Normal groups.")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # Run four cross-group baseline/post contrasts and paired analytical-vs-emotional change tests in both groups.
    analysis = build_analysis_frame(df)
    analytical_pre_test = welch_group_test(analysis, "analytical_pre")
    emotional_pre_test = welch_group_test(analysis, "emotional_pre")
    analytical_post_test = welch_group_test(analysis, "analytical_post")
    emotional_post_test = welch_group_test(analysis, "emotional_post")
    weird = analysis[analysis["Group"] == "WEIRD"].copy()
    normal = analysis[analysis["Group"] == "NORMAL"].copy()
    weird_stats = compare_change_types(weird)
    normal_stats = compare_change_types(normal)
    write_report(analytical_pre_test, emotional_pre_test, analytical_post_test, emotional_post_test, weird_stats, normal_stats)
    print(f"Saved report: {TXT_PATH}")


if __name__ == "__main__":
    main()
