from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
PLOT_OUTPUT_PATH = FIGURES_DIR / "hypothesis2.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis2.txt"
ALPHA = 0.05

SUBPLOT_TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20

CONDITION_ORDER = ["Interactive", "Text"]
CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static",
}
CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def normalize_condition(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "interactive": "Interactive",
                "text": "Text",
                "static": "Text",
            }
        )
    )


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
    required = {
        "Condition",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = normalize_condition(work["Condition"])
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["overall_change"] = (work["analytical_change"] + work["emotional_change"]) / 2.0

    return work


def condition_welch_test(data: pd.DataFrame, value_col: str) -> dict[str, float]:
    interactive_vals = data.loc[data["Condition"] == "Interactive", value_col].dropna().astype(float)
    text_vals = data.loc[data["Condition"] == "Text", value_col].dropna().astype(float)

    i_mean, i_low, i_high = mean_ci(interactive_vals)
    t_mean, t_low, t_high = mean_ci(text_vals)

    if len(interactive_vals) < 2 or len(text_vals) < 2:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")

    return {
        "n_interactive": float(len(interactive_vals)),
        "n_text": float(len(text_vals)),
        "mean_interactive": i_mean,
        "mean_text": t_mean,
        "ci_low_interactive": i_low,
        "ci_high_interactive": i_high,
        "ci_low_text": t_low,
        "ci_high_text": t_high,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(p_value) if pd.notna(p_value) else np.nan,
        "delta_interactive_minus_text": i_mean - t_mean,
    }


def add_pair_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    star_text: str,
    y_step: float,
) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=2.0, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, star_text or "ns", ha="center", va="bottom", fontsize=18)


def draw_condition_subplot(
    ax: plt.Axes,
    test_result: dict[str, float],
    title: str,
    ylabel: str,
) -> None:
    x = np.array([0.0, 1.0])
    means = np.array([test_result["mean_interactive"], test_result["mean_text"]], dtype=float)
    lows = np.array([test_result["ci_low_interactive"], test_result["ci_low_text"]], dtype=float)
    highs = np.array([test_result["ci_high_interactive"], test_result["ci_high_text"]], dtype=float)

    ax.bar(
        x,
        means,
        width=0.58,
        color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=3.0,
        capsize=7,
        capthick=3.0,
        zorder=3,
    )

    finite_lows = lows[np.isfinite(lows)]
    finite_highs = highs[np.isfinite(highs)]
    if finite_lows.size == 0 or finite_highs.size == 0:
        finite_means = means[np.isfinite(means)]
        if finite_means.size == 0:
            y_min, y_max = -0.1, 0.1
        else:
            y_min = float(np.nanmin(finite_means)) - 0.1
            y_max = float(np.nanmax(finite_means)) + 0.1
    else:
        y_min = float(np.nanmin(finite_lows))
        y_max = float(np.nanmax(finite_highs))

    y_span = max(y_max - y_min, 0.2)

    add_pair_bracket(
        ax,
        x[0],
        x[1],
        y_max + 0.10 * y_span,
        significance_stars(float(test_result["p"])),
        0.03 * y_span,
    )

    interpretation = "No significant main condition effect" if float(test_result["p"]) >= ALPHA else "Significant main condition effect"
    summary = (
        f"t={test_result['t']:.3f}, p {format_p(float(test_result['p']))}\n"
        f"{interpretation}"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(-0.5, 0.2)
    ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1])


def write_report(
    overall_test: dict[str, float],
    emotional_test: dict[str, float],
    analytical_test: dict[str, float],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 7 Results}")
    lines.append(
        "Hypothesis 7 tested whether there is a main effect of condition across three trust-change outcomes: overall change, emotional change, and analytical change."
    )
    lines.append("")

    lines.append("7.1 Main condition effect on overall change:")
    lines.append(
        "Overall change condition effect: "
        f"$M_{{Interactive}}={overall_test['mean_interactive']:.3f}$, "
        f"$M_{{Text}}={overall_test['mean_text']:.3f}$, "
        f"$t={overall_test['t']:.3f}$, "
        f"$p {format_p(overall_test['p'])}$, "
        f"$\\Delta(M_I-M_T)={overall_test['delta_interactive_minus_text']:.3f}$."
    )
    if float(overall_test["p"]) >= ALPHA:
        lines.append("Interpretation: no significant main condition effect on overall change.")
    else:
        lines.append("Interpretation: a significant main condition effect on overall change.")

    lines.append("")
    lines.append("7.2 Main condition effect on emotional change:")
    lines.append(
        "Emotional change condition effect: "
        f"$M_{{Interactive}}={emotional_test['mean_interactive']:.3f}$, "
        f"$M_{{Text}}={emotional_test['mean_text']:.3f}$, "
        f"$t={emotional_test['t']:.3f}$, "
        f"$p {format_p(emotional_test['p'])}$, "
        f"$\\Delta(M_I-M_T)={emotional_test['delta_interactive_minus_text']:.3f}$."
    )
    if float(emotional_test["p"]) >= ALPHA:
        lines.append("Interpretation: no significant main condition effect on emotional change.")
    else:
        lines.append("Interpretation: a significant main condition effect on emotional change.")

    lines.append("")
    lines.append("7.3 Main condition effect on analytical change:")
    lines.append(
        "Analytical change condition effect: "
        f"$M_{{Interactive}}={analytical_test['mean_interactive']:.3f}$, "
        f"$M_{{Text}}={analytical_test['mean_text']:.3f}$, "
        f"$t={analytical_test['t']:.3f}$, "
        f"$p {format_p(analytical_test['p'])}$, "
        f"$\\Delta(M_I-M_T)={analytical_test['delta_interactive_minus_text']:.3f}$."
    )
    if float(analytical_test["p"]) >= ALPHA:
        lines.append("Interpretation: no significant main condition effect on analytical change.")
    else:
        lines.append("Interpretation: a significant main condition effect on analytical change.")
        
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    overall_test = condition_welch_test(analysis, "overall_change")
    emotional_test = condition_welch_test(analysis, "emotional_change")
    analytical_test = condition_welch_test(analysis, "analytical_change")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)

    draw_condition_subplot(
        axes[0],
        overall_test,
        title="2.1 Main Condition Effect: Overall Change",
        ylabel="Overall Change",
    )
    draw_condition_subplot(
        axes[1],
        emotional_test,
        title="2.2 Main Condition Effect: Emotional Change",
        ylabel="Emotional Change",
    )
    draw_condition_subplot(
        axes[2],
        analytical_test,
        title="2.3 Main Condition Effect: Analytical Change",
        ylabel="Analytical Change",
    )

    fig.suptitle("Hypothesis 2: Main Condition Effects Across Trust Change Outcomes", fontsize=26)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close(fig)

    report_text = write_report(
        overall_test=overall_test,
        emotional_test=emotional_test,
        analytical_test=analytical_test,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
