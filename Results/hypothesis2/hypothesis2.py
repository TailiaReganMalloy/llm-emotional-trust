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

SUBPLOT_TITLE_FONTSIZE = 26
AXIS_LABEL_FONTSIZE = 26
TICK_LABEL_FONTSIZE = 26
ANNOTATION_FONTSIZE = 24
SIGNIFICANCE_FONTSIZE = 24

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


def welch_df(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna().astype(float)
    b = sample_b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan

    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    n_a = len(a)
    n_b = len(b)

    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    if denominator <= np.finfo(float).tiny:
        return np.nan
    return float(numerator / denominator)


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
        dfree = np.nan
    else:
        t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")
        dfree = welch_df(interactive_vals, text_vals)

    return {
        "n_interactive": float(len(interactive_vals)),
        "n_text": float(len(text_vals)),
        "mean_interactive": i_mean,
        "mean_text": t_mean,
        "ci_low_interactive": i_low,
        "ci_high_interactive": i_high,
        "ci_low_text": t_low,
        "ci_high_text": t_high,
        "df": dfree,
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
    text_above: bool = True,
    point_up: bool = False,
) -> None:
    y_ends = [y + y_step, y, y, y + y_step] if point_up else [y - y_step, y, y, y - y_step]
    ax.plot([x1, x1, x2, x2], y_ends, color="black", lw=2.0, zorder=4)
    text_y = y + 0.6 * y_step if text_above else y - 0.6 * y_step
    va = "bottom" if text_above else "top"
    ax.text(
        (x1 + x2) / 2,
        text_y,
        star_text or "ns",
        ha="center",
        va=va,
        fontsize=SIGNIFICANCE_FONTSIZE,
        zorder=5,
    )


def draw_condition_subplot(
    ax: plt.Axes,
    test_result: dict[str, float],
    title: str,
    show_y_axis: bool,
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
    y_bracket = y_min - 0.10 * y_span

    add_pair_bracket(
        ax,
        x[0],
        x[1],
        y_bracket,
        significance_stars(float(test_result["p"])),
        0.03 * y_span,
        text_above=False,
        point_up=True,
    )

    summary = f"t = {test_result['t']:.3f} p {format_p(float(test_result['p']))}"
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=TICK_LABEL_FONTSIZE + 2)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE + 2, pad=14)
    if show_y_axis:
        ax.set_ylabel("Trust Score Change", fontsize=AXIS_LABEL_FONTSIZE + 2)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE + 2)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(-0.3, 0.05)
    ax.set_yticks([-0.3, -0.2, -0.1, 0.0])


def write_report(
    overall_test: dict[str, float],
    emotional_test: dict[str, float],
    analytical_test: dict[str, float],
) -> str:
    def ci_text(low: float, high: float) -> str:
        if np.isnan(low) or np.isnan(high):
            return "95\\% CI [NA, NA]"
        return f"95\\% CI [{low:.3f}, {high:.3f}]"

    def significance_text(p_value: float, threshold: float = ALPHA) -> str:
        if pd.isna(p_value):
            return "was not testable"
        if p_value < threshold:
            return "was statistically significant"
        return "was not statistically significant"

    def df_text(value: float) -> str:
        if pd.isna(value):
            return "NA"
        return f"{value:.2f}"

    def bh_significance_map(named_pvalues: list[tuple[str, float]]) -> dict[str, str]:
        pvalues = np.array([np.inf if np.isnan(pvalue) else pvalue for _, pvalue in named_pvalues], dtype=float)
        order = np.argsort(pvalues)
        sorted_p = pvalues[order]
        thresholds = ALPHA * (np.arange(1, len(named_pvalues) + 1, dtype=float) / len(named_pvalues))
        rejected = np.zeros(len(named_pvalues), dtype=bool)
        passing = np.where(sorted_p <= thresholds)[0]
        if len(passing):
            rejected[order[: passing[-1] + 1]] = True

        results_map: dict[str, str] = {}
        for idx, (name, pvalue) in enumerate(named_pvalues):
            if np.isnan(pvalue):
                results_map[name] = "was not testable"
            elif rejected[idx]:
                results_map[name] = "was statistically significant"
            else:
                results_map[name] = "was not statistically significant"
        return results_map

    lines: list[str] = []
    bh_results = bh_significance_map([
        ("2.1", analytical_test["p"]),
        ("2.2", emotional_test["p"]),
        ("2.3", overall_test["p"]),
    ])

    lines.append("\\subsection{Hypothesis 2}")
    lines.append("\\begin{center}")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{Figures/hypothesis2.png}")
    lines.append("\\vspace{-20pt}")
    lines.append("\\captionof{figure}{Visualization for Hypothesis 2.}")
    lines.append("\\label{fig:hypothesis2}")
    lines.append("\\end{center}")
    lines.append("")
    lines.append(
        "Hypothesis 2 tested whether experimental condition changed the magnitude of trust change across analytical, emotional, and overall trust outcomes. Figure \\ref{fig:hypothesis2} summarizes the mean change scores for the Interactive and Static conditions, and the primary analyses used Welch's two-sample t-tests to compare conditions for each outcome."
    )
    lines.append("")

    lines.append(
        f"Across $n_{{Interactive}}={int(analytical_test['n_interactive'])}$ and $n_{{Text}}={int(analytical_test['n_text'])}$ participants, analytical trust change was $M_{{Interactive}}={analytical_test['mean_interactive']:.3f}$ "
        f"({ci_text(analytical_test['ci_low_interactive'], analytical_test['ci_high_interactive'])}) in the Interactive condition and $M_{{Text}}={analytical_test['mean_text']:.3f}$ "
        f"({ci_text(analytical_test['ci_low_text'], analytical_test['ci_high_text'])}) in the Static condition, Welch's $t({df_text(analytical_test['df'])})={analytical_test['t']:.3f}$, "
        f"$p {format_p(analytical_test['p'])}$, and $\\Delta(M_I-M_T)={analytical_test['delta_interactive_minus_text']:.3f}$. Sub-hypothesis 2.1 {significance_text(analytical_test['p'])}."
    )

    lines.append("")
    lines.append(
        f"Emotional trust change followed the same pattern: $M_{{Interactive}}={emotional_test['mean_interactive']:.3f}$ "
        f"({ci_text(emotional_test['ci_low_interactive'], emotional_test['ci_high_interactive'])}) for Interactive participants and $M_{{Text}}={emotional_test['mean_text']:.3f}$ "
        f"({ci_text(emotional_test['ci_low_text'], emotional_test['ci_high_text'])}) for Static participants, Welch's $t({df_text(emotional_test['df'])})={emotional_test['t']:.3f}$, "
        f"$p {format_p(emotional_test['p'])}$, and $\\Delta(M_I-M_T)={emotional_test['delta_interactive_minus_text']:.3f}$. Sub-hypothesis 2.2 {significance_text(emotional_test['p'])}."
    )

    lines.append("")
    lines.append(
        f"Overall trust change also did not differ meaningfully by condition, with $M_{{Interactive}}={overall_test['mean_interactive']:.3f}$ "
        f"({ci_text(overall_test['ci_low_interactive'], overall_test['ci_high_interactive'])}) in the Interactive condition and $M_{{Text}}={overall_test['mean_text']:.3f}$ "
        f"({ci_text(overall_test['ci_low_text'], overall_test['ci_high_text'])}) in the Static condition, Welch's $t({df_text(overall_test['df'])})={overall_test['t']:.3f}$, "
        f"$p {format_p(overall_test['p'])}$, and $\\Delta(M_I-M_T)={overall_test['delta_interactive_minus_text']:.3f}$. Sub-hypothesis 2.3 {significance_text(overall_test['p'])}. Taken together, these results indicate that the Interactive and Static conditions produced comparable changes in trust, so Hypothesis 2 was not supported."
    )

    lines.append("")
    lines.append(
        f"\\textbf{{Benjamini-Hochberg Procedure:}} Using the Benjamini-Hochberg procedure at false discovery rate $q={ALPHA:.2f}$ for the three primary sub-hypothesis tests, sub-hypothesis 2.1 {bh_results['2.1']}, sub-hypothesis 2.2 {bh_results['2.2']}, and sub-hypothesis 2.3 {bh_results['2.3']}."
    )
        
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    overall_test = condition_welch_test(analysis, "overall_change")
    emotional_test = condition_welch_test(analysis, "emotional_change")
    analytical_test = condition_welch_test(analysis, "analytical_change")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True, sharey=True)

    draw_condition_subplot(
        axes[0],
        analytical_test,
        title="Analytical Change",
        show_y_axis=True,
    )
    draw_condition_subplot(
        axes[1],
        emotional_test,
        title="Emotional Change",
        show_y_axis=False,
    )
    draw_condition_subplot(
        axes[2],
        overall_test,
        title="Overall Change",
        show_y_axis=False,
    )

    fig.suptitle("Hypothesis 2: Main Condition Effects Across Trust Change Outcomes", fontsize=32)
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
