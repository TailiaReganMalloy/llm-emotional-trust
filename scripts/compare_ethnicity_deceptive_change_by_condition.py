#!/usr/bin/env python3
"""
Compare AI deceptivity change (post - pre) across ethnicity groups and condition.

Writes outputs to:
- figures/deceptive_change_by_ethnicity_condition.png
- figures/deceptive_change_summary_by_ethnicity_condition.csv

Usage: python3 scripts/compare_ethnicity_deceptive_change_by_condition.py
"""
from pathlib import Path
import sys

try:
    import numpy as np
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
OUTPUT_PNG = ROOT / "figures" / "deceptive_change_by_ethnicity_condition.png"
OUTPUT_CSV = ROOT / "figures" / "deceptive_change_summary_by_ethnicity_condition.csv"


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


def collapse_small_categories(series, min_count=15, top_n=8):
    counts = series.value_counts(dropna=False)
    if top_n is not None and len(counts) > top_n:
        keep = counts.nlargest(top_n).index
        return series.where(series.isin(keep), other="Other")

    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), other="Other")


def is_prefer_not_to_say(value):
    s = str(value).strip().lower()
    blocked = {
        "prefer not to say",
        "prefer not to answer",
        "rather not say",
        "decline to state",
        "declined",
    }
    return s in blocked


def map_white_vs_nonwhite(value):
    s = str(value).strip().lower()

    if "mixed" in s:
        return None
    if "white" in s and "non-white" not in s and "nonwhite" not in s:
        return "White"
    if any(token in s for token in ["asian", "black", "other"]):
        return "Non-white"
    return None


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(3)

    df = pd.read_csv(DATA_PATH)

    pre_col = "AI Deceptive" if "AI Deceptive" in df.columns else find_column(df, ["deceptive"])
    post_col = "AI Deceptive Post" if "AI Deceptive Post" in df.columns else find_column(df, ["deceptive", "post"])

    if pre_col is None or pre_col not in df.columns:
        print("ERROR: Could not locate pre deceptivity column (expected 'AI Deceptive').", file=sys.stderr)
        sys.exit(4)
    if post_col is None or post_col not in df.columns:
        print("ERROR: Could not locate post deceptivity column (expected 'AI Deceptive Post').", file=sys.stderr)
        sys.exit(5)

    cond_col = "Condition" if "Condition" in df.columns else find_column(df, ["condition"])
    if cond_col is None or cond_col not in df.columns:
        print("ERROR: Could not locate Condition column.", file=sys.stderr)
        sys.exit(6)

    eth_col = None
    for cand in ["Ethnicity simplified", "Ethnicity (Demographics)", "Ethnicity", "ethnicity simplified", "ethnicity"]:
        if cand in df.columns:
            eth_col = cand
            break
    if eth_col is None:
        eth_col = find_column(df, ["ethnicity"])
    if eth_col is None or eth_col not in df.columns:
        print("ERROR: Could not locate Ethnicity column.", file=sys.stderr)
        sys.exit(7)

    df["deceptive_pre_num"] = likert_to_numeric(df[pre_col])
    df["deceptive_post_num"] = likert_to_numeric(df[post_col])
    df["deceptive_change"] = df["deceptive_post_num"] - df["deceptive_pre_num"]

    df["eth_group"] = df[eth_col].astype(str).str.strip()
    df = df.loc[~df["eth_group"].map(is_prefer_not_to_say)].copy()
    df["eth_group_collapsed"] = df["eth_group"].map(map_white_vs_nonwhite)

    clean = df.dropna(subset=["deceptive_change", "eth_group_collapsed", cond_col]).copy()
    if clean.empty:
        print("No valid rows after numeric mapping and filtering.", file=sys.stderr)
        sys.exit(8)

    clean[cond_col] = clean[cond_col].astype(str).str.strip()

    pair_means_raw = (
        clean.groupby(["eth_group_collapsed", cond_col])["deceptive_change"]
        .mean()
        .reset_index(name="pair_mean_raw")
    )
    pair_min = pair_means_raw["pair_mean_raw"].min()
    pair_max = pair_means_raw["pair_mean_raw"].max()
    if pd.isna(pair_min) or pd.isna(pair_max):
        print("ERROR: unable to compute pair means for normalization.", file=sys.stderr)
        sys.exit(9)

    if np.isclose(pair_max, pair_min):
        clean["deceptive_change_norm"] = 0.0
    else:
        # Map pair-min to -1 and pair-max to +1.
        clean["deceptive_change_norm"] = 2 * (clean["deceptive_change"] - pair_min) / (pair_max - pair_min) - 1

    summary = (
        clean.groupby(["eth_group_collapsed", cond_col])
        .agg(
            count=("deceptive_change_norm", "count"),
            mean_norm=("deceptive_change_norm", "mean"),
            std_norm=("deceptive_change_norm", "std"),
            mean_raw=("deceptive_change", "mean"),
            std_raw=("deceptive_change", "std"),
        )
        .reset_index()
        .sort_values(["eth_group_collapsed", cond_col])
    )
    print("Summary: normalized deceptivity change (post - pre) by ethnicity and condition")
    print(f"Normalization anchors from pair means: min={pair_min:.4f} -> -1, max={pair_max:.4f} -> +1")
    print(summary.to_string(index=False))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved summary to {OUTPUT_CSV}")

    order = (
        clean.groupby("eth_group_collapsed")["deceptive_change_norm"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    cond_order = sorted(clean[cond_col].dropna().unique().tolist())

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=clean,
        x="eth_group_collapsed",
        y="deceptive_change_norm",
        hue=cond_col,
        order=order,
        hue_order=cond_order,
        estimator="mean",
        errorbar=("ci", 95),
        palette="Set2",
    )

    handles, labels = ax.get_legend_handles_labels()
    keep = len(cond_order)
    ax.legend(handles[:keep], labels[:keep], title="Condition", frameon=True)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Ethnicity (collapsed)")
    ax.set_ylabel("Normalized AI Deceptive Change (Post - Pre)")
    ax.set_title("Normalized AI Deceptivity Change by Ethnicity Group and Condition")
    ax.set_ylim(-1.0, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(OUTPUT_PNG, dpi=220)
    print(f"Saved figure to {OUTPUT_PNG}")

    # Per-ethnicity two-condition tests when possible
    cond_levels = cond_order
    if len(cond_levels) == 2:
        c1, c2 = cond_levels
        print("\nPer-ethnicity condition tests (Welch t-test and Mann-Whitney):")
        for group_name, gdf in clean.groupby("eth_group_collapsed"):
            g1 = gdf.loc[gdf[cond_col] == c1, "deceptive_change_norm"].dropna().values
            g2 = gdf.loc[gdf[cond_col] == c2, "deceptive_change_norm"].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                print(f" {group_name}: insufficient data ({c1} n={len(g1)}, {c2} n={len(g2)})")
                continue

            tstat, p_t = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
            try:
                ustat, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            except Exception:
                ustat, p_u = np.nan, np.nan

            print(
                f" {group_name}: {c1} n={len(g1)} mean={np.mean(g1):.3f} | "
                f"{c2} n={len(g2)} mean={np.mean(g2):.3f} | "
                f"t={tstat:.3f} p={p_t:.4g} | U={ustat:.3f} p={p_u:.4g}"
            )
    else:
        print(f"\nCondition has {len(cond_levels)} levels; skipping pairwise two-condition tests.")


if __name__ == "__main__":
    main()
