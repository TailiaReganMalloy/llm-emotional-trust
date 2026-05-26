"""Hypothesis 2a additional analyses.

Hypothesis 2a decomposes Hypothesis 2 into condition-level simple effects:
1) Pre vs post trust change within the Interactive condition.
2) Pre vs post trust change within the Text (static) condition.

Primary outcome: Overall Trust (Analytical + Emotional totals).
Supplementary outcomes: Emotional Trust and Analytical Trust.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


DATA_PATH = Path("./data/Metrics.csv")
OUTPUT_PATH = Path("./plots/hypothesis2a_condition_split_pre_post.csv")
ALPHA = 0.05

METRICS = [
    ("Overall Trust", "overall_pre", "overall_post"),
    ("Emotional Trust", "Total Emotional Trust", "Total Emotional Trust Post"),
    ("Analytical Trust", "Total Analytical Trust", "Total Analytical Trust Post"),
]


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def cohens_d_paired(pre: pd.Series, post: pd.Series) -> float:
    diff = post - pre
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    if np.isclose(sd, 0):
        return np.nan
    return float(diff.mean() / sd)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Condition",
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

    analysis["overall_pre"] = (
        analysis["Total Analytical Trust"] + analysis["Total Emotional Trust"]
    )
    analysis["overall_post"] = (
        analysis["Total Analytical Trust Post"] + analysis["Total Emotional Trust Post"]
    )
    return analysis


def analyze_condition_metric(
    df: pd.DataFrame, condition: str, metric_label: str, pre_col: str, post_col: str
) -> dict[str, object]:
    subset = df[df["Condition"] == condition][[pre_col, post_col]].dropna().copy()
    pre = subset[pre_col].astype(float)
    post = subset[post_col].astype(float)
    delta = post - pre

    if len(subset) < 2:
        return {
            "Condition": condition,
            "Metric": metric_label,
            "n (paired)": int(len(subset)),
            "Pre Mean": float(pre.mean()) if len(pre) else np.nan,
            "Pre SD": float(pre.std(ddof=1)) if len(pre) > 1 else np.nan,
            "Post Mean": float(post.mean()) if len(post) else np.nan,
            "Post SD": float(post.std(ddof=1)) if len(post) > 1 else np.nan,
            "Mean Delta (Post-Pre)": float(delta.mean()) if len(delta) else np.nan,
            "Paired t": np.nan,
            "df": np.nan,
            "Paired t p": np.nan,
            "Paired t Significant (p<0.05)": False,
            "Wilcoxon": np.nan,
            "Wilcoxon p": np.nan,
            "Wilcoxon Significant (p<0.05)": False,
            "Cohen d (paired)": np.nan,
            "Abs Mean Delta": float(abs(delta.mean())) if len(delta) else np.nan,
        }

    t_stat, t_p = ttest_rel(pre, post, nan_policy="omit")
    if np.allclose(delta, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = wilcoxon(pre, post, alternative="two-sided")

    return {
        "Condition": condition,
        "Metric": metric_label,
        "n (paired)": int(len(subset)),
        "Pre Mean": float(pre.mean()),
        "Pre SD": float(pre.std(ddof=1)),
        "Post Mean": float(post.mean()),
        "Post SD": float(post.std(ddof=1)),
        "Mean Delta (Post-Pre)": float(delta.mean()),
        "Paired t": float(t_stat),
        "df": int(len(subset) - 1),
        "Paired t p": float(t_p),
        "Paired t Significant (p<0.05)": bool(t_p < ALPHA),
        "Wilcoxon": float(w_stat),
        "Wilcoxon p": float(w_p),
        "Wilcoxon Significant (p<0.05)": bool(w_p < ALPHA),
        "Cohen d (paired)": cohens_d_paired(pre, post),
        "Abs Mean Delta": float(abs(delta.mean())),
    }


def latex_line(row: pd.Series) -> str:
    if pd.isna(row["Paired t"]):
        return (
            f"A paired-samples $t$-test was planned for {row['Metric']} in the "
            f"{row['Condition']} condition, but there were insufficient paired observations."
        )

    if row["Paired t p"] < ALPHA:
        direction = "higher" if row["Post Mean"] > row["Pre Mean"] else "lower"
        comparison = f"was significantly {direction} than"
    else:
        comparison = "was not significantly different from"

    return (
        f"A paired-samples $t$-test was conducted to determine if there was a significant "
        f"difference between pre and post {row['Metric']} scores within the {row['Condition']} "
        f"condition. Results indicated that post scores ($M = {row['Post Mean']:.2f}$, "
        f"$SD = {row['Post SD']:.2f}$) {comparison} pre scores ($M = {row['Pre Mean']:.2f}$, "
        f"$SD = {row['Pre SD']:.2f}$), $t({int(row['df'])}) = {row['Paired t']:.2f}$, "
        f"$p {format_p(row['Paired t p'])}$, $d = {row['Cohen d (paired)']:.2f}$."
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    rows: list[dict[str, object]] = []
    for condition in ["Interactive", "Text"]:
        for metric_label, pre_col, post_col in METRICS:
            rows.append(analyze_condition_metric(analysis, condition, metric_label, pre_col, post_col))

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Hypothesis 2a: Condition-split pre/post simple effects")
    print(results.to_string(index=False))

    print("\nLaTeX narrative lines (Overall Trust simple effects):")
    for condition in ["Interactive", "Text"]:
        row = results[
            (results["Condition"] == condition) & (results["Metric"] == "Overall Trust")
        ].iloc[0]
        print(latex_line(row))

    print("\nDescriptive comparison of absolute mean overall change:")
    overall = results[results["Metric"] == "Overall Trust"][
        ["Condition", "Mean Delta (Post-Pre)", "Abs Mean Delta"]
    ]
    print(overall.to_string(index=False))

    print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()