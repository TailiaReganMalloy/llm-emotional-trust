import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal


DATA_PATH = "data/Combined.csv"
OUTPUT_PATH = "plots/ai_wary_deceptive_post_by_gender.png"
COLUMN = "AI Wary/Deceptive Post"
GENDER_COLUMN = "Gender"


LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neither Agree nor Disagree": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}


def to_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    return series.map(LIKERT_MAP)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    if COLUMN not in df.columns:
        raise KeyError(f"Column not found: {COLUMN}")
    if GENDER_COLUMN not in df.columns:
        raise KeyError(f"Column not found: {GENDER_COLUMN}")

    plot_df = df[[GENDER_COLUMN, COLUMN]].copy()
    plot_df["Score"] = to_score(plot_df[COLUMN])
    plot_df[GENDER_COLUMN] = plot_df[GENDER_COLUMN].astype(str).str.strip()
    plot_df = plot_df.dropna(subset=[GENDER_COLUMN, "Score"])
    plot_df = plot_df[plot_df[GENDER_COLUMN] != ""]

    group_counts = plot_df[GENDER_COLUMN].value_counts()
    valid_groups = group_counts[group_counts > 0].index.tolist()
    if len(valid_groups) < 2:
        raise ValueError("At least two gender groups with valid scores are required.")

    plot_df = plot_df[plot_df[GENDER_COLUMN].isin(valid_groups)].copy()
    plot_order = group_counts.index.tolist()

    grouped_scores = [
        plot_df.loc[plot_df[GENDER_COLUMN] == gender, "Score"] for gender in plot_order
    ]

    # One-way ANOVA compares mean differences across multiple groups.
    f_stat, anova_pvalue = f_oneway(*grouped_scores)
    # Kruskal-Wallis provides a non-parametric check for ordinal/Likert-style data.
    h_stat, kruskal_pvalue = kruskal(*grouped_scores)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(
        data=plot_df,
        x=GENDER_COLUMN,
        hue=GENDER_COLUMN,
        y="Score",
        order=plot_order,
        estimator="mean",
        errorbar=("ci", 95),
        palette="Set2",
        legend=False,
    )

    ax.set_title("Mean AI Wary/Deceptive Post by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Mean Score (1-5)")
    ax.set_ylim(1, 5)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved plot to {OUTPUT_PATH}")
    print("\nGroup sizes:")
    for gender, count in group_counts.items():
        print(f"- {gender}: n={count}")

    print("\nStatistical comparison (all gender groups):")
    print(f"One-way ANOVA: F={f_stat:.4f}, p={anova_pvalue:.6f}")
    print(f"Kruskal-Wallis: H={h_stat:.4f}, p={kruskal_pvalue:.6f}")


if __name__ == "__main__":
    main()