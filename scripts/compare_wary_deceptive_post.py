#!/usr/bin/env python3
"""
Compare the AI wary/deceptive post item across conditions and save a plot.

Writes output to: figures/wary_deceptive_post_by_condition.png

This script tries to locate the post column heuristically (matching words 'wary',
'deceptive' and 'post' in the header). It maps common Likert labels to numeric
scores 1-5, prints a group summary, runs basic tests (t-test or ANOVA/Kruskal),
and saves a bar + swarm plot.

Usage: python3 scripts/compare_wary_deceptive_post.py
"""
from pathlib import Path
import sys

import numpy as np

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
except Exception as e:
    print("Missing dependencies or import error:", e, file=sys.stderr)
    print("Install required packages: pandas seaborn matplotlib scipy numpy", file=sys.stderr)
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Combined.csv"
OUTPUT_PNG = ROOT / "figures" / "wary_deceptive_post_by_condition.png"


def find_column(df, keywords):
    """Find first column that contains all keywords (case-insensitive)."""
    keys = [k.lower() for k in keywords]
    for col in df.columns:
        col_l = str(col).lower()
        if all(k in col_l for k in keys):
            return col
    return None


def likert_to_numeric(series):
    """Map common Likert labels to numeric 1..5. Return float Series with NaN for non-mapped."""
    base_map = {
        "strongly disagree": 1,
        "disagree": 2,
        "neither agree nor disagree": 3,
        "neither agree nor disagree.": 3,
        "neither agree nor disagree ": 3,
        "agree": 4,
        "strongly agree": 5,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    }

    s = series.astype(str).str.strip().str.lower()
    mapped = s.map(base_map)

    # If some remained unmapped, try to extract a leading digit 1-5
    unmapped_mask = mapped.isna()
    if unmapped_mask.any():
        digits = s[unmapped_mask].str.extract(r"([1-5])")
        if not digits.empty:
            mapped.loc[unmapped_mask] = digits[0].astype(float).values

    return mapped.astype(float)


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(3)

    df = pd.read_csv(DATA_PATH)

    # Try find the likely post column. README lists it as 'AI Wary/Deceptive Post' in one form.
    post_col = find_column(df, ["wary", "deceptive", "post"]) or find_column(df, ["wary", "post"]) or find_column(df, ["deceptive", "post"]) or find_column(df, ["wary/deceptive", "post"]) or "AI Wary/Deceptive Post"

    if post_col not in df.columns:
        # If last-resort string doesn't exist, try to locate any column containing 'wary' and 'post'
        post_col = find_column(df, ["wary", "post"]) or find_column(df, ["deceptive", "post"]) or None

    if post_col is None or post_col not in df.columns:
        print("ERROR: Could not locate the AI wary/deceptive post column in the CSV header.", file=sys.stderr)
        print("Headers available:", ", ".join(map(str, df.columns.tolist()[:200])), file=sys.stderr)
        sys.exit(4)

    # Condition column
    cond_col = "Condition" if "Condition" in df.columns else (find_column(df, ["condition"]) or None)
    if cond_col is None or cond_col not in df.columns:
        print("ERROR: Could not locate a Condition column.", file=sys.stderr)
        sys.exit(5)

    df["wary_post_num"] = likert_to_numeric(df[post_col])
    clean = df.dropna(subset=["wary_post_num", cond_col]).copy()
    if clean.empty:
        print("No valid rows after mapping Likert to numeric.", file=sys.stderr)
        sys.exit(6)

    # Print summary
    grp = clean.groupby(cond_col)["wary_post_num"]
    summary = grp.agg(["count", "mean", "std"]).reset_index()
    print("Group summary:")
    print(summary.to_string(index=False))

    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=clean, x=cond_col, y="wary_post_num", estimator="mean", ci=95, palette="Set2")
    try:
        sns.swarmplot(data=clean, x=cond_col, y="wary_post_num", color="k", alpha=0.5, size=3)
    except Exception:
        # fallback: jittered stripplot
        sns.stripplot(data=clean, x=cond_col, y="wary_post_num", color="k", alpha=0.5, size=3, jitter=True)

    ax.set_ylabel("AI Wary/Deceptive Post (1-5)")
    ax.set_xlabel("Condition")
    ax.set_ylim(0.8, 5.2)
    ax.set_title("AI Wary/Deceptive (Post) by Condition")
    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved figure to {OUTPUT_PNG}")

    # Statistical tests
    groups = [g["wary_post_num"].dropna().values for _, g in clean.groupby(cond_col)]
    n_groups = len(groups)
    if n_groups == 2:
        tstat, p_t = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
        try:
            ustat, p_u = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        except Exception:
            ustat, p_u = np.nan, np.nan
        print(f"Two-group tests (n={len(groups[0])}, {len(groups[1])}):")
        print(f" Welch t-stat = {tstat:.4f}, p = {p_t:.4g}")
        print(f" Mann-Whitney U = {ustat:.4f}, p = {p_u:.4g}")
    else:
        try:
            fstat, p_a = stats.f_oneway(*groups)
        except Exception:
            fstat, p_a = np.nan, np.nan
        try:
            kstat, p_k = stats.kruskal(*groups)
        except Exception:
            kstat, p_k = np.nan, np.nan
        print(f"{n_groups}-group tests:")
        print(f" ANOVA F = {fstat:.4f}, p = {p_a:.4g}")
        print(f" Kruskal-Wallis H = {kstat:.4f}, p = {p_k:.4g}")


if __name__ == "__main__":
    main()
