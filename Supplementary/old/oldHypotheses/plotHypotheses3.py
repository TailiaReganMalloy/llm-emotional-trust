import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel


DATA_PATH = "data/Metrics.csv"
OUTPUT_PATH = "plots/h3_weird_emotional_vs_analytical.png"
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


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def zscore(series: pd.Series) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if np.isclose(std, 0):
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((series - mean) / std).astype(float)


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


def build_weird_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [
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
    work["analytical_change"] = work["Total Analytical Trust Post"] - work["Total Analytical Trust"]
    work["emotional_change"] = work["Total Emotional Trust Post"] - work["Total Emotional Trust"]
    work["analytical_change_z"] = zscore(work["analytical_change"].astype(float))
    work["emotional_change_z"] = zscore(work["emotional_change"].astype(float))

    language = work["Language"].astype(str).str.strip().str.lower()
    work["weird_like"] = work["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith(
        "english"
    )

    weird_df = work[work["weird_like"]].copy()
    pair_df = weird_df[["emotional_change_z", "analytical_change_z"]].dropna()
    if len(pair_df) < 2:
        raise ValueError("Not enough paired WEIRD-like observations for Hypothesis 3 visualization.")
    return pair_df


def collect_h3_results(pair_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, int]:
    emotional = pair_df["emotional_change_z"].astype(float)
    analytical = pair_df["analytical_change_z"].astype(float)

    t_stat, p_value = ttest_rel(emotional, analytical, nan_policy="omit")
    n = len(pair_df)

    em_mean, em_low, em_high = mean_ci(emotional)
    an_mean, an_low, an_high = mean_ci(analytical)

    results = pd.DataFrame(
        [
            {
                "Measure": "Emotional Change (z)",
                "mean": em_mean,
                "ci_low": em_low,
                "ci_high": em_high,
            },
            {
                "Measure": "Analytical Change (z)",
                "mean": an_mean,
                "ci_low": an_low,
                "ci_high": an_high,
            },
        ]
    )

    return results, float(t_stat), float(p_value), int(n)


def plot_h3(results: pd.DataFrame, t_stat: float, p_value: float, n: int) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = {
        "Emotional Change (z)": "#377eb8",
        "Analytical Change (z)": "#c26a26",
    }

    x_positions = np.arange(len(results), dtype=float)
    bar_width = 0.55

    for i, row in results.reset_index(drop=True).iterrows():
        mean_val = float(row["mean"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        label = row["Measure"]

        ax.bar(
            x_positions[i],
            mean_val,
            width=bar_width,
            color=colors[label],
            edgecolor="none",
            label=label,
            zorder=2,
        )

        ax.errorbar(
            x_positions[i],
            mean_val,
            yerr=[[mean_val - ci_low], [ci_high - mean_val]],
            fmt="none",
            ecolor="#4a4a4a",
            elinewidth=4,
            capsize=10,
            capthick=4,
            zorder=3,
        )

    y_min = float(results["ci_low"].min())
    y_max = float(results["ci_high"].max())
    y_span = max(y_max - y_min, 0.2)

    bracket_y = y_max + 0.10 * y_span
    h = 0.05 * y_span
    x1, x2 = x_positions[0], x_positions[1]

    ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + h, bracket_y + h, bracket_y], color="black", lw=2)
    ax.text(
        (x1 + x2) / 2,
        bracket_y + h + 0.02 * y_span,
        significance_stars(p_value),
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="black",
    )

    p_text = f"paired t({n - 1}) = {t_stat:.2f}, p = {p_value:.3f}".replace("p = 0.000", "p < .001")
    ax.text(
        (x1 + x2) / 2,
        bracket_y - 0.04 * y_span,
        p_text,
        ha="center",
        va="top",
        fontsize=11,
        color="black",
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.5, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(results["Measure"].tolist(), fontsize=15)
    ax.set_xlabel("Trust-Type Change (WEIRD-like Group)", fontsize=18)
    ax.set_ylabel("Standardized Mean Change (Post - Pre)", fontsize=18)
    ax.set_title("Hypothesis 3: WEIRD-like Emotional vs Analytical Change", fontsize=28)

    ax.legend(title="Hypothesis / Measure", loc="lower right", fontsize=14, title_fontsize=18)

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

    lower = min(y_min, 0.0) - 0.10 * y_span
    upper = bracket_y + h + 0.12 * y_span
    ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    pair_df = build_weird_dataframe(df)
    results, t_stat, p_value, n = collect_h3_results(pair_df)
    plot_h3(results, t_stat, p_value, n)

    display = results.copy()
    print("Saved plot:", OUTPUT_PATH)
    print("\nHypothesis 3 visualization values:")
    print(display.to_string(index=False))
    print(f"\nPaired t-test: t({n - 1}) = {t_stat:.2f}, p = {p_value:.6f}")


if __name__ == "__main__":
    main()