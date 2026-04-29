#!/usr/bin/env python3
"""
Generate a preliminary WEIRD-vs-condition trust report with plots.

Outputs:
- figures/prelim_emotional_change_by_weird_condition.png
- figures/prelim_analytical_post_by_weird_condition.png
- figures/preliminary_weird_summary.csv
- report.md
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Combined.csv"
FIG_DIR = ROOT / "figures"
REPORT_PATH = ROOT / "report.md"
SUMMARY_CSV_PATH = FIG_DIR / "preliminary_weird_summary.csv"

EMOTIONAL_FIG_PATH = FIG_DIR / "prelim_emotional_change_by_weird_condition.png"
ANALYTICAL_FIG_PATH = FIG_DIR / "prelim_analytical_post_by_weird_condition.png"

CONDITION_INTERACTIVE = "Interactive"
CONDITION_STATIC = "Static (Text)"

POSITIVE_LIKERT_MAP = {
    "strongly disagree": -2,
    "disagree": -1,
    "neutral": 0,
    "neither agree nor disagree": 0,
    "agree": 1,
    "strongly agree": 2,
}
NEGATIVE_LIKERT_MAP = {label: -score for label, score in POSITIVE_LIKERT_MAP.items()}

EMOTIONAL_POSITIVE_TOKENS = {
    "empathetic",
    "sensitive",
    "personal",
    "caring",
    "altruistic",
    "cordial",
    "responsive",
    "open-minded",
    "patient",
}
EMOTIONAL_NEGATIVE_TOKENS = {
    "apathetic",
    "insensitive",
    "impersonal",
    "ignoring",
    "self-serving",
    "rude",
    "indifferent",
    "judgmental",
    "impatient",
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


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def map_likert(series: pd.Series, negative_polarity: bool = False) -> pd.Series:
    value_map = NEGATIVE_LIKERT_MAP if negative_polarity else POSITIVE_LIKERT_MAP
    return series.map(lambda v: value_map.get(normalize_text(v), np.nan)).astype(float)


def map_emotional_token(series: pd.Series) -> pd.Series:
    def _map(value: object) -> float:
        token = normalize_text(value)
        if token in EMOTIONAL_POSITIVE_TOKENS:
            return 1.0
        if token in EMOTIONAL_NEGATIVE_TOKENS:
            return -1.0
        return np.nan

    return series.map(_map).astype(float)


def mean_of_available(series_list: list[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=1).mean(axis=1)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Condition",
        "Country of residence",
        "Language",
        "AI Dishonest",
        "AI Suspicious",
        "AI Harm",
        "AI Confident",
        "AI Security",
        "AI Trustworthy",
        "AI Reliable",
        "AI Trust",
        "AI Wary",
        "AI Deceptive",
        "AI Dishonest Post",
        "AI Suspicious Post",
        "AI Harm Post 1",
        "AI Harm Post 2",
        "AI Confident Post",
        "AI Security Post",
        "AI Trustworthy Post",
        "AI Reliable Post",
        "AI Trust Post",
        "AI Trust Post 2",
        "AI Wary/Deceptive Post",
        "AI Wary Post",
        "AI Deceptive Post",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise KeyError(f"Missing expected column: {column}")

    pre_emotional_cols = [f"AI systems are {index}" for index in range(1, 10)]
    post_emotional_cols = [f"AI systems are Post {index}" for index in range(1, 10)]
    for column in pre_emotional_cols + post_emotional_cols:
        if column not in df.columns:
            raise KeyError(f"Missing expected emotional column: {column}")

    pre_analytical = pd.DataFrame(
        {
            "dishonest": map_likert(df["AI Dishonest"], negative_polarity=True),
            "suspicious": map_likert(df["AI Suspicious"], negative_polarity=True),
            "harm": map_likert(df["AI Harm"], negative_polarity=True),
            "confident": map_likert(df["AI Confident"], negative_polarity=False),
            "security": map_likert(df["AI Security"], negative_polarity=False),
            "trustworthy": map_likert(df["AI Trustworthy"], negative_polarity=False),
            "reliable": map_likert(df["AI Reliable"], negative_polarity=False),
            "trust": map_likert(df["AI Trust"], negative_polarity=False),
            "wary_deceptive": mean_of_available(
                [
                    map_likert(df["AI Wary"], negative_polarity=True),
                    map_likert(df["AI Deceptive"], negative_polarity=True),
                ]
            ),
        }
    )

    post_analytical = pd.DataFrame(
        {
            "dishonest": map_likert(df["AI Dishonest Post"], negative_polarity=True),
            "suspicious": map_likert(df["AI Suspicious Post"], negative_polarity=True),
            "harm": mean_of_available(
                [
                    map_likert(df["AI Harm Post 1"], negative_polarity=True),
                    map_likert(df["AI Harm Post 2"], negative_polarity=True),
                ]
            ),
            "confident": map_likert(df["AI Confident Post"], negative_polarity=False),
            "security": map_likert(df["AI Security Post"], negative_polarity=False),
            "trustworthy": map_likert(df["AI Trustworthy Post"], negative_polarity=False),
            "reliable": map_likert(df["AI Reliable Post"], negative_polarity=False),
            "trust": mean_of_available(
                [
                    map_likert(df["AI Trust Post"], negative_polarity=False),
                    map_likert(df["AI Trust Post 2"], negative_polarity=False),
                ]
            ),
            "wary_deceptive": mean_of_available(
                [
                    map_likert(df["AI Wary/Deceptive Post"], negative_polarity=True),
                    map_likert(df["AI Wary Post"], negative_polarity=True),
                    map_likert(df["AI Deceptive Post"], negative_polarity=True),
                ]
            ),
        }
    )

    pre_emotional = pd.concat([map_emotional_token(df[col]) for col in pre_emotional_cols], axis=1)
    post_emotional = pd.concat([map_emotional_token(df[col]) for col in post_emotional_cols], axis=1)

    analysis = pd.DataFrame(index=df.index)
    analysis["analytical_pre"] = pre_analytical.mean(axis=1)
    analysis["analytical_post"] = post_analytical.mean(axis=1)
    analysis["analytical_change"] = analysis["analytical_post"] - analysis["analytical_pre"]

    analysis["emotional_pre"] = pre_emotional.mean(axis=1)
    analysis["emotional_post"] = post_emotional.mean(axis=1)
    analysis["emotional_change"] = analysis["emotional_post"] - analysis["emotional_pre"]

    raw_condition = df["Condition"].astype(str).str.strip()
    analysis["Condition"] = raw_condition.replace({"Text": CONDITION_STATIC, "Interactive": CONDITION_INTERACTIVE})
    analysis = analysis[analysis["Condition"].isin([CONDITION_INTERACTIVE, CONDITION_STATIC])].copy()

    language = df["Language"].astype(str).str.strip().str.lower()
    western_residence = df["Country of residence"].isin(WESTERN_COUNTRIES)

    # Preliminary WEIRD proxy for this dataset: Western country residence + English language.
    analysis["weird_like"] = western_residence & language.eq("english")
    analysis["WEIRD Group"] = np.where(analysis["weird_like"], "WEIRD-like", "Non-WEIRD-like")

    return analysis


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def cohen_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna().to_numpy(dtype=float)
    b = sample_b.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = np.sqrt((var_a + var_b) / 2.0)
    if np.isclose(pooled_sd, 0.0):
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_sd


def welch_condition_tests(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for group_name in ["WEIRD-like", "Non-WEIRD-like"]:
        subset = data[data["WEIRD Group"] == group_name]
        interactive = subset[subset["Condition"] == CONDITION_INTERACTIVE][metric].dropna()
        static = subset[subset["Condition"] == CONDITION_STATIC][metric].dropna()

        t_stat, p_value = stats.ttest_ind(interactive, static, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "Group": group_name,
                "n (Interactive)": int(len(interactive)),
                "n (Static)": int(len(static)),
                "Interactive Mean": float(interactive.mean()),
                "Static Mean": float(static.mean()),
                "Mean Difference (Interactive - Static)": float(interactive.mean() - static.mean()),
                "Welch t": float(t_stat),
                "p": float(p_value),
                "Cohen d": float(cohen_d(interactive, static)),
            }
        )

    return pd.DataFrame(rows)


def interaction_permutation_test(
    data: pd.DataFrame,
    metric: str,
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    subset = data[[metric, "weird_like", "Condition"]].dropna().copy()
    group_means = subset.groupby(["weird_like", "Condition"])[metric].mean().unstack()

    observed = (
        (group_means.loc[True, CONDITION_INTERACTIVE] - group_means.loc[True, CONDITION_STATIC])
        - (group_means.loc[False, CONDITION_INTERACTIVE] - group_means.loc[False, CONDITION_STATIC])
    )

    rng = np.random.default_rng(seed)
    extreme_count = 0
    weird_values = subset["weird_like"].to_numpy()

    for _ in range(n_permutations):
        permuted = subset.copy()
        permuted["weird_like"] = rng.permutation(weird_values)
        perm_means = permuted.groupby(["weird_like", "Condition"])[metric].mean().unstack()
        perm_stat = (
            (perm_means.loc[True, CONDITION_INTERACTIVE] - perm_means.loc[True, CONDITION_STATIC])
            - (perm_means.loc[False, CONDITION_INTERACTIVE] - perm_means.loc[False, CONDITION_STATIC])
        )
        if abs(perm_stat) >= abs(observed):
            extreme_count += 1

    p_value = (extreme_count + 1) / (n_permutations + 1)
    return float(observed), float(p_value)


def plot_metric(
    data: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    output_path: Path,
    draw_zero_line: bool = False,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(9.5, 5.5))
    ax = sns.barplot(
        data=data,
        x="WEIRD Group",
        y=metric,
        hue="Condition",
        order=["WEIRD-like", "Non-WEIRD-like"],
        hue_order=[CONDITION_INTERACTIVE, CONDITION_STATIC],
        palette={CONDITION_INTERACTIVE: "#2C7FB8", CONDITION_STATIC: "#D95F0E"},
        errorbar=("ci", 95),
        capsize=0.08,
    )
    if draw_zero_line:
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.65)

    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel(y_label)
    ax.legend(title="Condition", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def to_markdown_table(df: pd.DataFrame, float_columns: list[str] | None = None, precision: int = 3) -> str:
    float_columns = float_columns or []
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values: list[str] = []
        for column in headers:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.{precision}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_summary_table(data: pd.DataFrame, metric: str, metric_label: str) -> pd.DataFrame:
    rows = []
    for group in ["WEIRD-like", "Non-WEIRD-like"]:
        for condition in [CONDITION_INTERACTIVE, CONDITION_STATIC]:
            subset = data[(data["WEIRD Group"] == group) & (data["Condition"] == condition)][metric].dropna()
            rows.append(
                {
                    "Metric": metric_label,
                    "Group": group,
                    "Condition": condition,
                    "n": int(len(subset)),
                    "Mean": float(subset.mean()),
                    "SD": float(subset.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    analysis = analysis.dropna(subset=["emotional_change", "analytical_post", "Condition", "WEIRD Group", "weird_like"])

    emotional_tests = welch_condition_tests(analysis, "emotional_change")
    analytical_tests = welch_condition_tests(analysis, "analytical_post")

    emotional_did, emotional_interaction_p = interaction_permutation_test(analysis, "emotional_change")
    analytical_did, analytical_interaction_p = interaction_permutation_test(analysis, "analytical_post")

    plot_metric(
        analysis,
        metric="emotional_change",
        title="Emotional Trust Change by Condition and WEIRD-like Group",
        y_label="Emotional Trust Change (Post - Pre)",
        output_path=EMOTIONAL_FIG_PATH,
        draw_zero_line=True,
    )
    plot_metric(
        analysis,
        metric="analytical_post",
        title="Analytical Post-Trust by Condition and WEIRD-like Group",
        y_label="Analytical Post-Trust Score",
        output_path=ANALYTICAL_FIG_PATH,
        draw_zero_line=True,
    )

    summary_df = pd.concat(
        [
            build_summary_table(analysis, "emotional_change", "Emotional Trust Change (Post - Pre)"),
            build_summary_table(analysis, "analytical_post", "Analytical Post-Trust Score"),
        ],
        ignore_index=True,
    )
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    weird_counts = (
        analysis.groupby(["WEIRD Group", "Condition"]).size().reset_index(name="n").sort_values(["WEIRD Group", "Condition"])
    )

    key_results = pd.DataFrame(
        [
            {
                "Metric": "Emotional Trust Change (Post - Pre)",
                "Interaction (Diff-in-Diff)": emotional_did,
                "Permutation p": emotional_interaction_p,
            },
            {
                "Metric": "Analytical Post-Trust Score",
                "Interaction (Diff-in-Diff)": analytical_did,
                "Permutation p": analytical_interaction_p,
            },
        ]
    )

    report_lines = [
        "# Preliminary Trust Analysis: WEIRD-like Background x Condition",
        "",
        f"Generated on {date.today().isoformat()} from `data/Combined.csv`.",
        "",
        "## Research Focus",
        "",
        "This preliminary analysis tests whether participants with WEIRD-like backgrounds respond differently to the explanation condition (Interactive vs Static/Text) for emotional versus analytical trust in AI.",
        "",
        "## Operationalization Used",
        "",
        "- Condition: `Interactive` vs `Text` (shown below as `Static (Text)`).",
        "- WEIRD-like proxy: participant lives in a Western country and reports English as language.",
        "- Emotional trust score: average of mapped emotional adjective choices ($+1$ for pro-trust adjectives, $-1$ for anti-trust adjectives).",
        "- Analytical trust score: harmonized average of analytical trust items on a $[-2, 2]$ scale, including handling split post columns by averaging available variants.",
        "",
        "## Sample Sizes",
        "",
        to_markdown_table(weird_counts),
        "",
        "## Key Interaction Results",
        "",
        to_markdown_table(
            key_results.assign(**{"Permutation p": key_results["Permutation p"].map(format_p_value)}),
            float_columns=["Interaction (Diff-in-Diff)"],
            precision=3,
        ),
        "",
        "Interpretation: negative interaction values indicate that Interactive (vs Static) produced a larger downward shift for the WEIRD-like group relative to the Non-WEIRD-like group.",
        "",
        "## Figure 1: Emotional Trust Change",
        "",
        "![Emotional Trust Change by Condition and WEIRD-like Group](figures/prelim_emotional_change_by_weird_condition.png)",
        "",
        "### Welch Tests (Interactive vs Static within each group)",
        "",
        to_markdown_table(
            emotional_tests.assign(
                **{
                    "p": emotional_tests["p"].map(format_p_value),
                }
            ),
            float_columns=[
                "Interactive Mean",
                "Static Mean",
                "Mean Difference (Interactive - Static)",
                "Welch t",
                "Cohen d",
            ],
            precision=3,
        ),
        "",
        "## Figure 2: Analytical Post-Trust",
        "",
        "![Analytical Post-Trust by Condition and WEIRD-like Group](figures/prelim_analytical_post_by_weird_condition.png)",
        "",
        "### Welch Tests (Interactive vs Static within each group)",
        "",
        to_markdown_table(
            analytical_tests.assign(
                **{
                    "p": analytical_tests["p"].map(format_p_value),
                }
            ),
            float_columns=[
                "Interactive Mean",
                "Static Mean",
                "Mean Difference (Interactive - Static)",
                "Welch t",
                "Cohen d",
            ],
            precision=3,
        ),
        "",
        "## Notes for Presentation",
        "",
        "- These are exploratory, preliminary results intended for presentation-level discussion.",
        "- The WEIRD-like proxy is an approximation based on available fields and should be sensitivity-checked in a final analysis plan.",
        "- A reproducible numeric summary was also saved to `figures/preliminary_weird_summary.csv`.",
    ]

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved figure: {EMOTIONAL_FIG_PATH}")
    print(f"Saved figure: {ANALYTICAL_FIG_PATH}")
    print(f"Saved summary CSV: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
