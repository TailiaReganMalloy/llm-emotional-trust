"""Hypothesis 5a additional analyses.

Hypothesis 5a tests subgroup simple effects within each condition:
1) WEIRD-like vs Non-WEIRD-like in the Interactive condition.
2) WEIRD-like vs Non-WEIRD-like in the Text condition.

Outcome: Overall Trust Change (post - pre).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


DATA_PATH = Path("./data/Metrics.csv")
OUTPUT_PATH = Path("./plots/hypothesis5a_within_condition_group_effects.csv")
ALPHA = 0.05

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


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def welch_df(group_a: pd.Series, group_b: pd.Series) -> float:
    n1, n2 = len(group_a), len(group_b)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1 = group_a.var(ddof=1)
    v2 = group_b.var(ddof=1)
    numerator = (v1 / n1 + v2 / n2) ** 2
    denominator = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    if np.isclose(denominator, 0):
        return np.nan
    return float(numerator / denominator)


def cohens_d_independent(group_a: pd.Series, group_b: pd.Series) -> float:
    n1, n2 = len(group_a), len(group_b)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1 = group_a.var(ddof=1)
    v2 = group_b.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if np.isclose(pooled_sd, 0):
        return np.nan
    return float((group_a.mean() - group_b.mean()) / pooled_sd)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Condition",
        "Country of residence",
        "Language",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    analysis = df.copy()
    analysis["Condition"] = analysis["Condition"].astype(str).str.strip()
    analysis = analysis[analysis["Condition"].isin(["Interactive", "Text"])].copy()

    analysis["overall_change"] = (
        (analysis["Total Analytical Trust Post"] - analysis["Total Analytical Trust"])
        + (analysis["Total Emotional Trust Post"] - analysis["Total Emotional Trust"])
    )

    language = analysis["Language"].astype(str).str.strip().str.lower()
    analysis["weird_like"] = analysis["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith(
        "english"
    )
    analysis["WEIRD Group"] = np.where(analysis["weird_like"], "WEIRD-like", "Non-WEIRD-like")
    return analysis


def analyze_within_condition_group_effect(df: pd.DataFrame, condition: str) -> dict[str, object]:
    subset = df[df["Condition"] == condition].copy()

    weird = subset.loc[subset["weird_like"], "overall_change"].dropna().astype(float)
    non_weird = subset.loc[~subset["weird_like"], "overall_change"].dropna().astype(float)

    if len(weird) < 2 or len(non_weird) < 2:
        return {
            "Condition": condition,
            "n WEIRD-like": int(len(weird)),
            "n Non-WEIRD-like": int(len(non_weird)),
            "WEIRD Mean": float(weird.mean()) if len(weird) else np.nan,
            "WEIRD SD": float(weird.std(ddof=1)) if len(weird) > 1 else np.nan,
            "Non-WEIRD Mean": float(non_weird.mean()) if len(non_weird) else np.nan,
            "Non-WEIRD SD": float(non_weird.std(ddof=1)) if len(non_weird) > 1 else np.nan,
            "Mean Difference (WEIRD - Non-WEIRD)": np.nan,
            "Welch t": np.nan,
            "Welch df": np.nan,
            "Welch p": np.nan,
            "Welch Significant (p<0.05)": False,
            "Mann-Whitney U": np.nan,
            "Mann-Whitney p": np.nan,
            "Mann-Whitney Significant (p<0.05)": False,
            "Cohen d": np.nan,
        }

    t_stat, t_p = ttest_ind(weird, non_weird, equal_var=False, nan_policy="omit")
    u_stat, u_p = mannwhitneyu(weird, non_weird, alternative="two-sided")

    return {
        "Condition": condition,
        "n WEIRD-like": int(len(weird)),
        "n Non-WEIRD-like": int(len(non_weird)),
        "WEIRD Mean": float(weird.mean()),
        "WEIRD SD": float(weird.std(ddof=1)),
        "Non-WEIRD Mean": float(non_weird.mean()),
        "Non-WEIRD SD": float(non_weird.std(ddof=1)),
        "Mean Difference (WEIRD - Non-WEIRD)": float(weird.mean() - non_weird.mean()),
        "Welch t": float(t_stat),
        "Welch df": welch_df(weird, non_weird),
        "Welch p": float(t_p),
        "Welch Significant (p<0.05)": bool(t_p < ALPHA),
        "Mann-Whitney U": float(u_stat),
        "Mann-Whitney p": float(u_p),
        "Mann-Whitney Significant (p<0.05)": bool(u_p < ALPHA),
        "Cohen d": cohens_d_independent(weird, non_weird),
    }


def latex_line(row: pd.Series) -> str:
    if pd.isna(row["Welch t"]):
        return (
            f"A Welch's independent-samples $t$-test was planned within the {row['Condition']} "
            "condition, but there were insufficient observations in one or both subgroup levels."
        )

    if row["Welch p"] < ALPHA:
        direction = "higher" if row["WEIRD Mean"] > row["Non-WEIRD Mean"] else "lower"
        comparison = f"were significantly {direction} than"
    else:
        comparison = "were not significantly different from"

    return (
        f"A Welch's independent-samples $t$-test was conducted within the {row['Condition']} "
        "condition to determine if Overall Trust Change differed between WEIRD-like and "
        "Non-WEIRD-like participants. Results indicated that WEIRD-like scores ($M = "
        f"{row['WEIRD Mean']:.2f}$, $SD = {row['WEIRD SD']:.2f}$) {comparison} Non-WEIRD-like "
        f"scores ($M = {row['Non-WEIRD Mean']:.2f}$, $SD = {row['Non-WEIRD SD']:.2f}$), "
        f"$t({row['Welch df']:.2f}) = {row['Welch t']:.2f}$, $p {format_p(row['Welch p'])}$, "
        f"$d = {row['Cohen d']:.2f}$."
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    results = pd.DataFrame(
        [
            analyze_within_condition_group_effect(analysis, "Interactive"),
            analyze_within_condition_group_effect(analysis, "Text"),
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Hypothesis 5a: Within-condition subgroup effects")
    print(results.to_string(index=False))

    print("\nLaTeX narrative lines:")
    for _, row in results.iterrows():
        print(latex_line(row))

    print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()