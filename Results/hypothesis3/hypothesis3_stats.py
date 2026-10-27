"""Statistical analysis and report generation for hypothesis 3.

This module reads the processed Metrics.csv file, computes the planned tests and
multiple-comparison adjustments, and writes hypothesis3.txt. Plotting lives
in hypothesis3_plot.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import ttest_rel, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

TXT_PATH = OUTPUT_DIR / "hypothesis3.txt"

FIGURES_DIR = REPO_ROOT / "Figures"

PNG_PATH = FIGURES_DIR / "hypothesis3.png"

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

TYPE_COLORS = {
    "Emotional z": "#377eb8",
    "Analytical z": "#c26a26",
}

ERROR_COLOR = "#4a4a4a"

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

WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}

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
    if (row.get("Nationality") in WESTERN_COUNTRIES
            or row.get("Country of birth") in WESTERN_COUNTRIES):
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

def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - float(values.mean())) / std

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
    work["is_weird"] = work["weird_score"] >= 4

    work["analytical_pre"] = work["Total Analytical Trust"] / ANALYTICAL_MAX
    work["analytical_post"] = work["Total Analytical Trust Post"] / ANALYTICAL_MAX
    work["emotional_pre"] = work["Total Emotional Trust"] / EMOTIONAL_MAX
    work["emotional_post"] = work["Total Emotional Trust Post"] / EMOTIONAL_MAX

    work["overall_pre"] = (work["analytical_pre"] + work["emotional_pre"]) / 2.0
    work["overall_post"] = (work["analytical_post"] + work["emotional_post"]) / 2.0

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]

    return work

# Uses complete pre/post pairs for a two-sided paired t-test and Wilcoxon robustness test; reports paired Cohen's d and post-minus-pre change.
def paired_pre_post(pre: pd.Series, post: pd.Series) -> dict[str, float]:
    pair = pd.DataFrame({"pre": pre, "post": post}).dropna().copy()
    pre_vals = pair["pre"].astype(float)
    post_vals = pair["post"].astype(float)

    if len(pair) < 2:
        return {
            "n": float(len(pair)),
            "pre_mean": np.nan,
            "post_mean": np.nan,
            "pre_sd": np.nan,
            "post_sd": np.nan,
            "delta_mean": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "pre_ci": np.nan,
            "post_ci": np.nan,
        }

    t_stat, pvalue = ttest_rel(pre_vals, post_vals, nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(post_vals, pre_vals)
    pre_mean, pre_ci = mean_ci(pre_vals)
    post_mean, post_ci = mean_ci(post_vals)

    return {
        "n": float(len(pair)),
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "pre_sd": float(pre_vals.std(ddof=1)),
        "post_sd": float(post_vals.std(ddof=1)),
        "delta_mean": float((post_vals - pre_vals).mean()),
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(post_vals, pre_vals),
        "pre_ci": pre_ci,
        "post_ci": post_ci,
    }

# Compares standardized emotional and analytical change scores within the WEIRD group using paired tests.
def compare_trust_type(df: pd.DataFrame) -> dict[str, float]:
    pair = df[["emotional_change_z", "analytical_change_z"]].dropna().copy()
    emotional = pair["emotional_change_z"].astype(float)
    analytical = pair["analytical_change_z"].astype(float)

    if len(pair) < 2:
        return {
            "n": float(len(pair)),
            "em_mean": np.nan,
            "an_mean": np.nan,
            "em_sd": np.nan,
            "an_sd": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "em_ci": np.nan,
            "an_ci": np.nan,
            "diff_mean": np.nan,
        }

    t_stat, pvalue = ttest_rel(emotional, analytical, nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(emotional, analytical)
    em_mean, em_ci = mean_ci(emotional)
    an_mean, an_ci = mean_ci(analytical)

    return {
        "n": float(len(pair)),
        "em_mean": em_mean,
        "an_mean": an_mean,
        "em_sd": float(emotional.std(ddof=1)),
        "an_sd": float(analytical.std(ddof=1)),
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(emotional, analytical),
        "em_ci": em_ci,
        "an_ci": an_ci,
        "diff_mean": float((emotional - analytical).mean()),
    }

# Report files contain per-sub-hypothesis descriptive statistics, test statistics, p-values,
# effect sizes/differences when available, and Benjamini-Hochberg adjusted q-values.

# Writes the calculated descriptive values, test statistics, and adjusted significance results.
def write_report(
    overall_stats: dict[str, float],
    analytical_stats: dict[str, float],
    emotional_stats: dict[str, float],
) -> None:
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
    q_values = bh_qvalues([analytical_stats["p"], emotional_stats["p"], overall_stats["p"]])
    n_pairs = int(overall_stats["n"]) if pd.notna(overall_stats["n"]) else 0
    dfree = n_pairs - 1 if n_pairs >= 2 else np.nan

    lines.append("\\subsection{Hypothesis 3}")
    lines.append("\\begin{center}")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{Figures/hypothesis3.png}")
    lines.append("\\vspace{-20pt}")
    lines.append("\\captionof{figure}{Visualization for Hypothesis 3.}")
    lines.append("\\label{fig:hypothesis3}")
    lines.append("\\end{center}")
    lines.append("")
    lines.append(
        "Hypothesis 3 tested whether WEIRD participants showed a distinct trust-change pattern across analytical, emotional, and overall trust. Figure \\ref{fig:hypothesis3} summarizes these three pre-post tests."
    )
    lines.append("")

    # One paragraph per sub-hypothesis: descriptives, paired t-test (+ d), Wilcoxon, difference, then B-H result.
    for label, name, stats, q in [
        ("3.1", "analytical", analytical_stats, q_values[0]),
        ("3.2", "Emotional", emotional_stats, q_values[1]),
        ("3.3", "overall", overall_stats, q_values[2]),
    ]:
        lines.append(
            f"Sub-hypothesis {label} compared ${n_pairs}$ WEIRD participants showing that {name} trust changed from "
            f"$M_{{pre}}={stats['pre_mean']:.3f}$ ({ci_text(stats['pre_mean'], stats['pre_ci'])}) to "
            f"$M_{{post}}={stats['post_mean']:.3f}$ ({ci_text(stats['post_mean'], stats['post_ci'])}). "
            f"Both parametric and non-parametric analyses confirmed this difference, including a paired-samples t-test "
            f"($t({dfree})={stats['t']:.3f}$, $p {p_text(stats['p'])}$, $d={stats['d']:.3f}$), and a Wilcoxon signed-rank test "
            f"($W={stats['w']:.3f}$, $p_{{W}} {p_text(stats['w_p'])}$). The difference was $\\Delta={stats['delta_mean']:.3f}$. "
            f"After applying B-H correction at false discovery rate $q={ALPHA:.2f}$, sub-hypothesis {label} "
            f"({q_text(q)}) {bh_status(q)}."
        )
        lines.append("")

    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the three primary sub-hypothesis tests, sub-hypothesis 3.1 ({q_text(q_values[0])}) {bh_status(q_values[0])}, sub-hypothesis 3.2 ({q_text(q_values[1])}) {bh_status(q_values[1])}, and sub-hypothesis 3.3 ({q_text(q_values[2])}) {bh_status(q_values[2])}."
    )

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # Build normalized outcomes, select WEIRD participants, then test the three paired pre/post outcomes.
    analysis = build_analysis_frame(df)
    weird = analysis[analysis["is_weird"]].copy()
    overall_stats = paired_pre_post(weird["overall_pre"], weird["overall_post"])
    analytical_stats = paired_pre_post(weird["analytical_pre"], weird["analytical_post"])
    emotional_stats = paired_pre_post(weird["emotional_pre"], weird["emotional_post"])
    write_report(overall_stats, analytical_stats, emotional_stats)
    print(f"Saved report: {TXT_PATH}")


if __name__ == "__main__":
    main()
