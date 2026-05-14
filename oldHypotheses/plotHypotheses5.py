import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t


DATA_PATH = "data/Metrics.csv"
OUTPUT_PATH = "plots/h5_condition_by_weird_interaction.png"
ALPHA = 0.05
N_PERMUTATIONS = 5000

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


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(clean.mean())
    if n == 1:
        return mean_val, mean_val, mean_val

    sem = float(clean.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf((1 + confidence) / 2, df=n - 1) * sem)
    return mean_val, mean_val - margin, mean_val + margin


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
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = work["Condition"].astype(str).str.strip()
    work = work[work["Condition"].isin(["Interactive", "Text"])].copy()

    work["analytical_change"] = work["Total Analytical Trust Post"] - work["Total Analytical Trust"]
    work["emotional_change"] = work["Total Emotional Trust Post"] - work["Total Emotional Trust"]
    work["overall_change"] = work["analytical_change"] + work["emotional_change"]

    language = work["Language"].astype(str).str.strip().str.lower()
    work["weird_like"] = work["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith(
        "english"
    )
    work["WEIRD Group"] = np.where(work["weird_like"], "WEIRD-like", "Non-WEIRD-like")
    return work


def interaction_ols_test(data: pd.DataFrame) -> dict[str, float]:
    fit_df = data[["overall_change", "Condition", "weird_like"]].dropna().copy()
    y = fit_df["overall_change"].astype(float).to_numpy()
    condition = (fit_df["Condition"] == "Interactive").astype(float).to_numpy()
    group = fit_df["weird_like"].astype(float).to_numpy()
    interaction = condition * group

    X = np.column_stack([np.ones(len(fit_df)), condition, group, interaction])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    df = n - p
    if df <= 0:
        return {
            "interaction_coef": np.nan,
            "interaction_t": np.nan,
            "interaction_df": np.nan,
            "interaction_p": np.nan,
        }

    rss = float(np.sum(residuals**2))
    sigma2 = rss / df
    x_inv = np.linalg.inv(X.T @ X)
    cov = sigma2 * x_inv
    se = np.sqrt(np.diag(cov))

    b_interaction = float(beta[3])
    se_interaction = float(se[3])
    t_interaction = b_interaction / se_interaction if not np.isclose(se_interaction, 0) else np.nan
    p_interaction = (
        2 * (1 - stats.t.cdf(abs(t_interaction), df)) if pd.notna(t_interaction) else np.nan
    )

    return {
        "interaction_coef": b_interaction,
        "interaction_t": float(t_interaction),
        "interaction_df": float(df),
        "interaction_p": float(p_interaction),
    }


def did_permutation_test(data: pd.DataFrame, n_permutations: int = N_PERMUTATIONS, seed: int = 42) -> dict[str, float]:
    subset = data[["overall_change", "Condition", "weird_like"]].dropna().copy()
    means = subset.groupby(["weird_like", "Condition"])["overall_change"].mean().unstack()

    observed_did = (
        (means.loc[True, "Interactive"] - means.loc[True, "Text"])
        - (means.loc[False, "Interactive"] - means.loc[False, "Text"])
    )

    rng = np.random.default_rng(seed)
    weird_values = subset["weird_like"].to_numpy()
    extreme = 0

    for _ in range(n_permutations):
        permuted = subset.copy()
        permuted["weird_like"] = rng.permutation(weird_values)
        perm_means = permuted.groupby(["weird_like", "Condition"])["overall_change"].mean().unstack()
        perm_did = (
            (perm_means.loc[True, "Interactive"] - perm_means.loc[True, "Text"])
            - (perm_means.loc[False, "Interactive"] - perm_means.loc[False, "Text"])
        )
        if abs(perm_did) >= abs(observed_did):
            extreme += 1

    p_value = (extreme + 1) / (n_permutations + 1)
    return {
        "did_observed": float(observed_did),
        "did_p": float(p_value),
    }


def summarize_cells(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in ["WEIRD-like", "Non-WEIRD-like"]:
        for condition in ["Interactive", "Text"]:
            vals = data.loc[
                (data["WEIRD Group"] == group) & (data["Condition"] == condition), "overall_change"
            ]
            mean_val, ci_low, ci_high = mean_ci(vals)
            rows.append(
                {
                    "WEIRD Group": group,
                    "Condition": condition,
                    "n": int(vals.notna().sum()),
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def plot_h5(summary: pd.DataFrame, interaction: dict[str, float], did: dict[str, float]) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7.5))

    group_order = ["WEIRD-like", "Non-WEIRD-like"]
    condition_order = ["Interactive", "Text"]
    colors = {
        "Interactive": "#377eb8",
        "Text": "#c26a26",
    }

    x_positions = np.arange(len(group_order), dtype=float)
    bar_width = 0.34
    offsets = {
        "Interactive": -bar_width / 2,
        "Text": bar_width / 2,
    }

    for condition in condition_order:
        for i, group in enumerate(group_order):
            row = summary[(summary["WEIRD Group"] == group) & (summary["Condition"] == condition)].iloc[0]
            mean_val = float(row["mean"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            x_val = x_positions[i] + offsets[condition]

            label = "Interactive" if (condition == "Interactive" and i == 0) else None
            if condition == "Text" and i == 0:
                label = "Static (Text)"

            ax.bar(
                x_val,
                mean_val,
                width=bar_width,
                color=colors[condition],
                edgecolor="none",
                label=label,
                zorder=2,
            )

            ax.errorbar(
                x_val,
                mean_val,
                yerr=[[mean_val - ci_low], [ci_high - mean_val]],
                fmt="none",
                ecolor="#4a4a4a",
                elinewidth=4,
                capsize=10,
                capthick=4,
                zorder=3,
            )

    y_min = float(summary["ci_low"].min())
    y_max = float(summary["ci_high"].max())
    y_span = max(y_max - y_min, 0.2)

    x_left = x_positions[0] + offsets["Interactive"]
    x_right = x_positions[-1] + offsets["Text"]
    bracket_y = y_max + 0.14 * y_span
    h = 0.06 * y_span

    ax.plot(
        [x_left, x_left, x_right, x_right],
        [bracket_y, bracket_y + h, bracket_y + h, bracket_y],
        color="black",
        lw=2,
    )

    interaction_p = float(interaction["interaction_p"])
    ax.text(
        (x_left + x_right) / 2,
        bracket_y + h + 0.02 * y_span,
        significance_stars(interaction_p),
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="black",
    )

    p_text = (
        f"Condition x WEIRD interaction: t({int(interaction['interaction_df'])}) = {interaction['interaction_t']:.2f}, "
        f"p = {interaction_p:.3f}; permutation DID p = {did['did_p']:.3f}"
    ).replace("p = 0.000", "p < .001")
    ax.text(
        (x_left + x_right) / 2,
        bracket_y - 0.05 * y_span,
        p_text,
        ha="center",
        va="top",
        fontsize=10.5,
        color="black",
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.5, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_order, fontsize=16)
    ax.set_xlabel("Group", fontsize=18)
    ax.set_ylabel("Mean Overall Trust Change (Post - Pre)", fontsize=18)
    ax.set_title("Hypothesis 5: Condition x WEIRD Interaction", fontsize=28)

    ax.legend(title="Condition", loc="lower right", fontsize=14, title_fontsize=18)

    ax.text(
        0.01,
        0.99,
        "Asterisks: * p < .05, ** p < .01, *** p < .001",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="black",
    )

    lower = min(y_min, 0.0) - 0.12 * y_span
    upper = bracket_y + h + 0.14 * y_span
    ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)
    summary = summarize_cells(analysis)
    interaction = interaction_ols_test(analysis)
    did = did_permutation_test(analysis)
    plot_h5(summary, interaction, did)

    print("Saved plot:", OUTPUT_PATH)
    print("\nGroup-condition summary used for plotting:")
    print(summary.to_string(index=False))
    print(
        f"\nInteraction test: t({int(interaction['interaction_df'])}) = {interaction['interaction_t']:.2f}, "
        f"p = {interaction['interaction_p']:.6f}"
    )
    print(f"Permutation DID: {did['did_observed']:.6f}, p = {did['did_p']:.6f}")


if __name__ == "__main__":
    main()