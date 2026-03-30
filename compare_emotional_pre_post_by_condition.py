import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, wilcoxon


DATA_PATH = "data/Combined.csv"
SUMMARY_OUTPUT_PATH = "plots/emotional_pre_post_significance_by_condition.csv"
PLOT_OUTPUT_PATH = "plots/emotional_pre_post_mean_difference_by_condition.png"
CONDITION_COLUMN = "Condition"


ITEM_PAIRS = [
    ("AI systems are 1", "Item 1", "Apathetic", "Empathetic"),
    ("AI systems are 2", "Item 2", "Insensitive", "Sensitive"),
    ("AI systems are 3", "Item 3", "Impersonal", "Personal"),
    ("AI systems are 4", "Item 4", "Ignoring", "Caring"),
    ("AI systems are 5", "Item 5", "Self-Serving", "Altruistic"),
    ("AI systems are 6", "Item 6", "Rude", "Cordial"),
    ("AI systems are 7", "Item 7", "Indifferent", "Responsive"),
    ("AI systems are 8", "Item 8", "Judgmental", "Open-Minded"),
    ("AI systems are 9", "Item 9", "Impatient", "Patient"),
]

POST_COLS = [f"AI systems are Post {i}" for i in range(1, 10)]


def to_binary_score(series: pd.Series, negative_label: str, positive_label: str) -> pd.Series:
    # Direction convention used throughout this script:
    # bad adjective = 0, good adjective = 1.
    # Therefore, positive (Post - Pre) means shift from bad -> good.
    mapped = series.astype(str).str.strip().map({negative_label: 0, positive_label: 1})
    return pd.to_numeric(mapped, errors="coerce")


def extract_post_series(df: pd.DataFrame, options: set[str]) -> pd.Series:
    cleaned = df[POST_COLS].astype(str).apply(lambda col: col.str.strip())
    match_mask = cleaned.isin(options)

    def pick_value(row_values: pd.Series, row_mask: pd.Series) -> str | float:
        matches = row_values[row_mask]
        if matches.empty:
            return np.nan
        return matches.iloc[0]

    return cleaned.apply(lambda row: pick_value(row, match_mask.loc[row.name]), axis=1)


def safe_wilcoxon(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    diffs = post - pre
    if len(diffs) == 0:
        return np.nan, np.nan
    if np.allclose(diffs, 0):
        return 0.0, 1.0
    stat, pvalue = wilcoxon(pre, post, alternative="two-sided")
    return float(stat), float(pvalue)


def safe_paired_ttest(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    if len(pre) < 2:
        return np.nan, np.nan
    stat, pvalue = ttest_rel(pre, post, nan_policy="omit")
    return float(stat), float(pvalue)


def safe_independent_ttest(group_a: pd.Series, group_b: pd.Series) -> tuple[float, float]:
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan, np.nan
    stat, pvalue = ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
    return float(stat), float(pvalue)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col
        for pair in ITEM_PAIRS
        for col in (pair[0],)
        if col not in df.columns
    ]
    if CONDITION_COLUMN not in df.columns:
        raise KeyError(f"Column not found: {CONDITION_COLUMN}")
    missing_post_columns = [col for col in POST_COLS if col not in df.columns]
    if missing_columns or missing_post_columns:
        raise KeyError(f"Missing required columns: {sorted(set(missing_columns))}")

    conditions = sorted(df[CONDITION_COLUMN].dropna().unique())
    records = []
    deltas_by_item_condition: dict[tuple[str, str], pd.Series] = {}

    for condition in conditions:
        condition_df = df[df[CONDITION_COLUMN] == condition].copy()

        for pre_col, label, negative_label, positive_label in ITEM_PAIRS:
            pair_df = condition_df[[pre_col] + POST_COLS].copy()
            post_raw = extract_post_series(pair_df, {negative_label, positive_label})
            pair_df["Pre"] = to_binary_score(pair_df[pre_col], negative_label, positive_label)
            pair_df["Post"] = to_binary_score(post_raw, negative_label, positive_label)
            pair_df = pair_df.dropna(subset=["Pre", "Post"])

            pre_scores = pair_df["Pre"]
            post_scores = pair_df["Post"]
            deltas = post_scores - pre_scores
            mean_pre = pre_scores.mean() if len(pre_scores) else np.nan
            mean_post = post_scores.mean() if len(post_scores) else np.nan
            mean_delta = deltas.mean() if len(pre_scores) else np.nan

            deltas_by_item_condition[(label, condition)] = deltas

            t_stat, t_p = safe_paired_ttest(pre_scores, post_scores)
            w_stat, w_p = safe_wilcoxon(pre_scores, post_scores)

            records.append(
                {
                    "Condition": condition,
                    "Item": label,
                    "Item Options": f"{negative_label} vs {positive_label}",
                    "X Label": f"{label}: {negative_label} vs {positive_label}",
                    "Pre Column": pre_col,
                    "Post Columns Searched": ", ".join(POST_COLS),
                    "N (paired)": len(pair_df),
                    "Mean Pre": mean_pre,
                    "Mean Post": mean_post,
                    "Mean Delta (Post-Pre)": mean_delta,
                    "Paired t-stat": t_stat,
                    "Paired t p-value": t_p,
                    "Wilcoxon stat": w_stat,
                    "Wilcoxon p-value": w_p,
                }
            )

    results_df = pd.DataFrame(records)

    across_condition_records = []
    for _, label, _, _ in ITEM_PAIRS:
        interactive_deltas = deltas_by_item_condition.get((label, "Interactive"), pd.Series(dtype=float))
        text_deltas = deltas_by_item_condition.get((label, "Text"), pd.Series(dtype=float))
        t_stat, pvalue = safe_independent_ttest(interactive_deltas, text_deltas)
        across_condition_records.append(
            {
                "Item": label,
                "Across-condition t-stat": t_stat,
                "Across-condition t p-value": pvalue,
            }
        )

    across_condition_df = pd.DataFrame(across_condition_records)
    results_df = results_df.merge(across_condition_df, on="Item", how="left")

    item_order = [
        f"{label}: {negative_label} vs {positive_label}"
        for _, label, negative_label, positive_label in ITEM_PAIRS
    ]
    results_df["X Label"] = pd.Categorical(results_df["X Label"], categories=item_order, ordered=True)
    results_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(
        data=results_df,
        x="X Label",
        y="Mean Delta (Post-Pre)",
        hue="Condition",
        errorbar=None,
        palette="Set2",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Mean Pre-to-Post Emotional Item Difference by Condition")
    ax.set_xlabel("Emotional Adjective Item")
    ax.set_ylabel("Mean Delta (Post - Pre, where 0=Bad and 1=Good)")
    ax.tick_params(axis="x", labelrotation=25)

    y_min, y_max = ax.get_ylim()
    y_top = y_max - 0.03 * (y_max - y_min)
    y_bottom = y_min + 0.03 * (y_max - y_min)
    ax.text(
        0.01,
        y_top,
        "Positive delta: Bad -> Good",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        fontsize=10,
        color="black",
    )
    ax.text(
        0.01,
        y_bottom,
        "Negative delta: Good -> Bad",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        color="black",
    )

    # Add across-condition p-values (Welch t-test on deltas) above each item's pair of bars.
    x_positions = ax.get_xticks()
    for i, (_, label, _, _) in enumerate(ITEM_PAIRS):
        row = across_condition_df.loc[across_condition_df["Item"] == label]
        if row.empty:
            continue
        pvalue = row["Across-condition t p-value"].iloc[0]
        if pd.isna(pvalue):
            p_text = "p=NA"
        elif pvalue < 0.001:
            p_text = "p<0.001"
        else:
            p_text = f"p={pvalue:.3f}"

        item_rows = results_df[results_df["Item"] == label]
        y_pair_max = item_rows["Mean Delta (Post-Pre)"].max()
        y_text = y_pair_max + 0.012 * (y_max - y_min)
        ax.text(x_positions[i], y_text, p_text, ha="center", va="bottom", fontsize=9, color="black")

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150)
    plt.close()

    display_df = results_df[
        [
            "Condition",
            "Item",
            "N (paired)",
            "Mean Pre",
            "Mean Post",
            "Mean Delta (Post-Pre)",
            "Paired t p-value",
            "Wilcoxon p-value",
        ]
    ].copy()

    print("Saved summary:", SUMMARY_OUTPUT_PATH)
    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("\nPre/Post comparison with significance by condition:")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
