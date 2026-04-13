#!/usr/bin/env python3
"""
Compare AI wary/deceptive post scores across ethnicity groups and save a plot.

This script will try to locate an ethnicity column (preferring 'Ethnicity simplified'
if present), map Likert labels to numeric scores, group low-count ethnicities into
'Other' (to keep the figure readable), and save a bar+swarm plot to
`figures/wary_by_ethnicity.png`.

Usage: python3 scripts/compare_ethnicity_wary_post.py
"""
from pathlib import Path
import sys

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
except Exception as e:
    print("Missing dependencies or import error:", e, file=sys.stderr)
    print("Install required packages: pandas seaborn matplotlib scipy numpy", file=sys.stderr)
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Combined.csv"
OUTPUT_PNG = ROOT / "figures" / "wary_by_ethnicity.png"


def find_column(df, keywords):
    keys = [k.lower() for k in keywords]
    for col in df.columns:
        col_l = str(col).lower()
        if all(k in col_l for k in keys):
            return col
    return None


def likert_to_numeric(series):
    base_map = {
        "strongly disagree": 1,
        "disagree": 2,
        "neither agree nor disagree": 3,
        "neither agree nor disagree.": 3,
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
    unmapped = mapped.isna()
    if unmapped.any():
        digits = s[unmapped].str.extract(r"([1-5])")
        if not digits.empty:
            mapped.loc[unmapped] = digits[0].astype(float).values
    return mapped.astype(float)


def collapse_small_categories(series, min_count=20, top_n=None):
    """Collapse categories with counts < min_count into 'Other'. Optionally keep only top_n by count."""
    counts = series.value_counts(dropna=False)
    if top_n is not None and len(counts) > top_n:
        top = counts.nlargest(top_n).index
        return series.where(series.isin(top), other="Other")

    to_keep = counts[counts >= min_count].index
    return series.where(series.isin(to_keep), other="Other")


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(3)

    df = pd.read_csv(DATA_PATH)

    # locate post column (wary/deceptive post)
    post_col = find_column(df, ["wary", "deceptive", "post"]) or find_column(df, ["wary", "post"]) or find_column(df, ["deceptive", "post"]) or None
    if post_col is None:
        print("ERROR: Could not find wary/deceptive post column. Available headers:", file=sys.stderr)
        print(", ".join(map(str, df.columns.tolist()[:200])), file=sys.stderr)
        sys.exit(4)

    # locate ethnicity column (prefer simplified)
    eth_col = None
    for cand in ["Ethnicity simplified", "Ethnicity (Demographics)", "Ethnicity", "ethnicity simplified", "ethnicity"]:
        if cand in df.columns:
            eth_col = cand
            break
    if eth_col is None:
        eth_col = find_column(df, ["ethnicity"]) or None

    if eth_col is None or eth_col not in df.columns:
        print("ERROR: Could not locate an Ethnicity column.", file=sys.stderr)
        sys.exit(5)

    df["wary_post_num"] = likert_to_numeric(df[post_col])
    df["eth_group"] = df[eth_col].astype(str).str.strip()

    # Collapse small categories: prefer to keep top 8, others -> Other (adjustable)
    df["eth_group_collapsed"] = collapse_small_categories(df["eth_group"], min_count=15, top_n=8)

    clean = df.dropna(subset=["wary_post_num", "eth_group_collapsed"]).copy()
    if clean.empty:
        print("No valid rows after mapping and collapsing categories.", file=sys.stderr)
        sys.exit(6)

    # Compute summary by collapsed ethnicity
    grp = clean.groupby("eth_group_collapsed")["wary_post_num"]
    summary = grp.agg(["count", "mean", "std"]).reset_index().sort_values("count", ascending=False)
    print("Ethnicity group summary (collapsed):")
    print(summary.to_string(index=False))

    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    order = summary.sort_values("mean", ascending=False)["eth_group_collapsed"].tolist()
    ax = sns.barplot(data=clean, x="eth_group_collapsed", y="wary_post_num", order=order, estimator="mean", ci=95, palette="Spectral")
    try:
        sns.swarmplot(data=clean, x="eth_group_collapsed", y="wary_post_num", order=order, color="k", alpha=0.5, size=3)
    except Exception:
        sns.stripplot(data=clean, x="eth_group_collapsed", y="wary_post_num", order=order, color="k", alpha=0.5, size=3, jitter=True)

    ax.set_ylabel("AI Wary/Deceptive Post (1-5)")
    ax.set_xlabel("Ethnicity (collapsed)")
    ax.set_ylim(0.8, 5.2)
    plt.xticks(rotation=30, ha="right")
    ax.set_title("AI Wary/Deceptive (Post) by Ethnicity")
    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved figure to {OUTPUT_PNG}")

    # Statistical tests across groups
    groups = [g["wary_post_num"].dropna().values for _, g in clean.groupby("eth_group_collapsed")]
    n_groups = len(groups)
    if n_groups < 2:
        print("Not enough groups for statistical comparison.")
        return

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
