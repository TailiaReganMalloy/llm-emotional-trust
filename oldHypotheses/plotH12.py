import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel


DATA_PATH = "data/Metrics.csv"
OUTPUT_PATH = "plots/h1_h2_significant_results.png"
ALPHA = 0.05


METRIC_SPECS = [
    ("Overall Trust", "overall_pre", "overall_post"),
    ("Emotional Trust", "Total Emotional Trust", "Total Emotional Trust Post"),
    ("Analytical Trust", "Total Analytical Trust", "Total Analytical Trust Post"),
]

GROUP_SPECS = [
    ("H1: All Participants", None),
    ("H2: Interactive", "Interactive"),
    ("H2: Static (Text)", "Text"),
]


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


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


def paired_effect(pre: pd.Series, post: pd.Series) -> dict[str, float]:
    pairs = pd.DataFrame({"pre": pre, "post": post}).dropna()
    if len(pairs) < 2:
        return {
            "n": float(len(pairs)),
            "mean_change": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }

    pre_clean = pairs["pre"].astype(float)
    post_clean = pairs["post"].astype(float)
    diff = post_clean - pre_clean

    t_stat, p_value = ttest_rel(pre_clean, post_clean, nan_policy="omit")
    mean_change, ci_low, ci_high = mean_ci(diff)

    return {
        "n": float(len(pairs)),
        "mean_change": mean_change,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def collect_significant_results(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "Condition",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = work["Condition"].astype(str).str.strip()
    work = work[work["Condition"].isin(["Interactive", "Text"])].copy()

    work["overall_pre"] = work["Total Analytical Trust"] + work["Total Emotional Trust"]
    work["overall_post"] = work["Total Analytical Trust Post"] + work["Total Emotional Trust Post"]

    rows: list[dict[str, float | str]] = []
    for group_label, condition in GROUP_SPECS:
        subset = work if condition is None else work[work["Condition"] == condition]

        for metric_label, pre_col, post_col in METRIC_SPECS:
            stats = paired_effect(subset[pre_col], subset[post_col])
            rows.append(
                {
                    "Group": group_label,
                    "Metric": metric_label,
                    **stats,
                }
            )

    out = pd.DataFrame(rows)
    significant = out[out["p_value"] < ALPHA].copy()
    if significant.empty:
        raise ValueError("No significant H1/H2 effects were found with p < 0.05.")
    return significant


def plot_significant_results(results: pd.DataFrame) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(14, 8))

    metric_order = [metric for metric, _, _ in METRIC_SPECS]
    group_order = [group for group, _ in GROUP_SPECS]
    colors = {
        "H1: All Participants": "#6f6f6f",
        "H2: Interactive": "#377eb8",
        "H2: Static (Text)": "#c26a26",
    }

    x_positions = np.arange(len(metric_order), dtype=float)
    bar_width = 0.23
    offsets = np.array([-bar_width, 0.0, bar_width])

    star_positions: list[float] = []
    y_min = float(results["ci_low"].min())
    y_max = float(results["ci_high"].max())
    y_span = max(y_max - y_min, 0.2)
    star_pad = 0.03 * y_span

    for group_idx, group_label in enumerate(group_order):
        for metric_idx, metric_label in enumerate(metric_order):
            match = results[(results["Group"] == group_label) & (results["Metric"] == metric_label)]
            if match.empty:
                continue

            row = match.iloc[0]
            x_val = x_positions[metric_idx] + offsets[group_idx]
            mean_val = float(row["mean_change"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])

            label = group_label if metric_idx == 0 else None
            ax.bar(
                x_val,
                mean_val,
                width=bar_width,
                color=colors[group_label],
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

            stars = significance_stars(float(row["p_value"]))
            if stars:
                if mean_val >= 0:
                    star_y = ci_high + star_pad
                    va = "bottom"
                else:
                    star_y = ci_low - star_pad
                    va = "top"
                star_positions.append(star_y)
                ax.text(
                    x_val,
                    star_y,
                    stars,
                    ha="center",
                    va=va,
                    fontsize=18,
                    fontweight="bold",
                    color="black",
                    zorder=4,
                )

    ax.axhline(0, color="#4a4a4a", linewidth=1.5, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_order, fontsize=16)
    ax.set_xlabel("Trust Metric", fontsize=18)
    ax.set_ylabel("Mean Change (Post - Pre)", fontsize=18)
    ax.set_title("Significant Results Supporting Hypotheses 1 and 2", fontsize=30)

    ax.legend(title="Hypothesis / Condition", loc="lower right", fontsize=16, title_fontsize=22)

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

    all_y = [y_min, y_max, 0.0, *star_positions]
    lower = min(all_y) - 0.08 * y_span
    upper = max(all_y) + 0.08 * y_span
    ax.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    significant_results = collect_significant_results(df)
    plot_significant_results(significant_results)

    display = significant_results[
        ["Group", "Metric", "n", "mean_change", "ci_low", "ci_high", "t_stat", "p_value"]
    ].copy()
    display["n"] = display["n"].astype(int)

    print("Saved plot:", OUTPUT_PATH)
    print("\nSignificant effects visualized:")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()