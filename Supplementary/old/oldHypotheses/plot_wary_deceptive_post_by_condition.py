import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind


DATA_PATH = "data/Combined.csv"
OUTPUT_PATH = "plots/ai_wary_deceptive_post_by_condition.png"
COLUMN = "AI Wary/Deceptive Post"
CONDITION_COLUMN = "Condition"


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
    if CONDITION_COLUMN not in df.columns:
        raise KeyError(f"Column not found: {CONDITION_COLUMN}")

    plot_df = df[[CONDITION_COLUMN, COLUMN]].copy()
    plot_df["Score"] = to_score(plot_df[COLUMN])
    plot_df = plot_df.dropna(subset=[CONDITION_COLUMN, "Score"])

    interactive_scores = plot_df.loc[plot_df[CONDITION_COLUMN] == "Interactive", "Score"]
    text_scores = plot_df.loc[plot_df[CONDITION_COLUMN] == "Text", "Score"]

    if interactive_scores.empty or text_scores.empty:
        raise ValueError("Both Interactive and Text groups need at least one valid score.")

    # Welch's t-test compares means without assuming equal variance.
    t_stat, t_pvalue = ttest_ind(interactive_scores, text_scores, equal_var=False, nan_policy="omit")
    # Mann-Whitney U provides a non-parametric check for ordinal/Likert-style data.
    u_stat, u_pvalue = mannwhitneyu(interactive_scores, text_scores, alternative="two-sided")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_df,
        x=CONDITION_COLUMN,
        hue=CONDITION_COLUMN,
        y="Score",
        estimator="mean",
        errorbar=("ci", 95),
        palette="Set2",
        legend=False,
    )

    ax.set_title("Mean AI Wary/Deceptive Post by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Score (1-5)")
    ax.set_ylim(1, 5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved plot to {OUTPUT_PATH}")
    print("\nStatistical comparison (Interactive vs Text):")
    print(
        f"Welch's t-test: n_interactive={len(interactive_scores)}, n_text={len(text_scores)}, "
        f"t={t_stat:.4f}, p={t_pvalue:.6f}"
    )
    print(f"Mann-Whitney U: U={u_stat:.4f}, p={u_pvalue:.6f}")


if __name__ == "__main__":
    main()
