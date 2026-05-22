from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, t, ttest_ind, ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis4.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis4.txt"
ALPHA = 0.05

GROUP_ORDER = ["WEIRD", "NORMAL"]
CONDITION_ORDER = ["Interactive", "Text"]
CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static (Text)",
}
CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}
STAR_FONT_SIZE = 30

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
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    language = work["Language"].astype(str).str.strip().str.lower()
    work["weird_like"] = work["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith(
        "english"
    )
    work["Group"] = np.where(work["weird_like"], "WEIRD", "NORMAL")

    # Per-item average change keeps analytical and emotional scales comparable.
    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["analytical_vs_emotional_diff"] = work["analytical_change"] - work["emotional_change"]

    work["condition_bin"] = (work["Condition"] == "Interactive").astype(float)
    work["group_bin"] = (work["Group"] == "WEIRD").astype(float)
    return work


def summarize_cells(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            vals = data.loc[(data["Group"] == group) & (data["Condition"] == condition), value_col]
            mean_val, ci_low, ci_high = mean_ci(vals)
            rows.append(
                {
                    "Group": group,
                    "Condition": condition,
                    "n": int(vals.notna().sum()),
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def paired_pre_post_tests(
    data: pd.DataFrame,
    pre_col: str,
    post_col: str,
    scale_divisor: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in GROUP_ORDER:
        for condition in CONDITION_ORDER:
            pair = data.loc[
                (data["Group"] == group) & (data["Condition"] == condition),
                [pre_col, post_col],
            ].dropna()

            if len(pair) < 2:
                t_stat = np.nan
                p_value = np.nan
            else:
                pre_vals = pair[pre_col].astype(float) / scale_divisor
                post_vals = pair[post_col].astype(float) / scale_divisor
                t_stat, p_value = ttest_rel(pre_vals, post_vals, nan_policy="omit")

            rows.append(
                {
                    "Group": group,
                    "Condition": condition,
                    "n": int(len(pair)),
                    "Mean Pre": float(pair[pre_col].mean() / scale_divisor) if len(pair) else np.nan,
                    "Mean Post": float(pair[post_col].mean() / scale_divisor) if len(pair) else np.nan,
                    "Mean Change (Post-Pre)": float(
                        (pair[post_col].mean() - pair[pre_col].mean()) / scale_divisor
                    )
                    if len(pair)
                    else np.nan,
                    "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p": float(p_value) if pd.notna(p_value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def welch_group_within_condition_tests(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for condition in CONDITION_ORDER:
        weird_vals = data.loc[
            (data["Group"] == "WEIRD") & (data["Condition"] == condition), value_col
        ].dropna()
        non_weird_vals = data.loc[
            (data["Group"] == "NORMAL") & (data["Condition"] == condition), value_col
        ].dropna()

        if len(weird_vals) < 2 or len(non_weird_vals) < 2:
            t_stat = np.nan
            p_value = np.nan
        else:
            t_stat, p_value = ttest_ind(weird_vals, non_weird_vals, equal_var=False, nan_policy="omit")

        rows.append(
            {
                "Condition": condition,
                "Mean WEIRD": float(weird_vals.mean()) if len(weird_vals) else np.nan,
                "Mean NORMAL": float(non_weird_vals.mean()) if len(non_weird_vals) else np.nan,
                "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                "p": float(p_value) if pd.notna(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def one_way_anova_four_cells(data: pd.DataFrame, value_col: str) -> tuple[float, float]:
    cells = [
        data.loc[(data["Group"] == group) & (data["Condition"] == condition), value_col].dropna()
        for group in GROUP_ORDER
        for condition in CONDITION_ORDER
    ]
    f_stat, p_value = f_oneway(*cells)
    return float(f_stat), float(p_value)


def interaction_effect(data: pd.DataFrame, value_col: str) -> dict[str, float]:
    fit_df = data[[value_col, "condition_bin", "group_bin"]].dropna().copy()
    y = fit_df[value_col].astype(float).to_numpy()
    condition = fit_df["condition_bin"].to_numpy(dtype=float)
    group = fit_df["group_bin"].to_numpy(dtype=float)
    interaction = condition * group

    X = np.column_stack([np.ones(len(fit_df)), condition, group, interaction])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        return {
            "coef": np.nan,
            "t": np.nan,
            "df": np.nan,
            "p": np.nan,
        }

    rss = float(np.sum(residuals**2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    coef = float(beta[3])
    t_stat = coef / float(se[3]) if not np.isclose(float(se[3]), 0) else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if pd.notna(t_stat) else np.nan
    return {
        "coef": coef,
        "t": float(t_stat),
        "df": float(dof),
        "p": float(p_value),
    }


def draw_grouped_subplot(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
    ylabel: str,
    show_legend: bool,
) -> dict[tuple[str, str], float]:
    x_positions = np.arange(len(GROUP_ORDER), dtype=float)
    bar_width = 0.34
    offsets = {
        "Interactive": -bar_width / 2,
        "Text": bar_width / 2,
    }
    x_map: dict[tuple[str, str], float] = {}

    for condition in CONDITION_ORDER:
        for idx, group in enumerate(GROUP_ORDER):
            row = summary[(summary["Group"] == group) & (summary["Condition"] == condition)].iloc[0]
            mean_val = float(row["mean"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            x_val = x_positions[idx] + offsets[condition]
            x_map[(group, condition)] = x_val

            label = CONDITION_LABELS[condition] if idx == 0 else None
            ax.bar(
                x_val,
                mean_val,
                width=bar_width,
                color=CONDITION_COLORS[condition],
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
                elinewidth=3.5,
                capsize=8,
                capthick=3.5,
                zorder=3,
            )

    y_min = float(summary["ci_low"].min())
    y_max = float(summary["ci_high"].max())
    y_span = max(y_max - y_min, 0.2)
    ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.22 * y_span)

    ax.axhline(0, color="#4a4a4a", linewidth=1.5, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(GROUP_ORDER, fontsize=16)
    ax.set_xlabel("Group", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=24)
    if show_legend:
        ax.legend(title="Condition", loc="lower right", fontsize=15, title_fontsize=18)
    return x_map


def annotate_significant_bars(
    ax: plt.Axes,
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    x_map: dict[tuple[str, str], float],
) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 0.2)
    pad = 0.02 * y_span
    marker_bottoms: list[float] = []

    for _, row in tests.iterrows():
        stars = significance_stars(float(row["p"]))
        if not stars:
            continue
        group = str(row["Group"])
        condition = str(row["Condition"])
        x_val = x_map[(group, condition)]
        ci_low = float(
            summary[(summary["Group"] == group) & (summary["Condition"] == condition)]["ci_low"].iloc[0]
        )
        marker_y = ci_low - pad
        marker_bottoms.append(marker_y)
        ax.text(
            x_val,
            marker_y,
            stars,
            ha="center",
            va="top",
            fontsize=STAR_FONT_SIZE,
            fontweight="bold",
            color="black",
            zorder=4,
        )

    if marker_bottoms:
        ax.set_ylim(min(y_min, min(marker_bottoms) - 0.06 * y_span), y_max)


def annotate_diff_group_brackets(
    ax: plt.Axes,
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    x_map: dict[tuple[str, str], float],
) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 0.2)
    pad = 0.02 * y_span
    marker_bottoms: list[float] = []

    for condition in CONDITION_ORDER:
        row = tests[tests["Condition"] == condition].iloc[0]
        stars = significance_stars(float(row["p"]))
        if not stars:
            continue

        x1 = x_map[("WEIRD", condition)]
        x2 = x_map[("NORMAL", condition)]
        ci_low_left = float(
            summary[(summary["Group"] == "WEIRD") & (summary["Condition"] == condition)]["ci_low"].iloc[0]
        )
        ci_low_right = float(
            summary[(summary["Group"] == "NORMAL") & (summary["Condition"] == condition)]["ci_low"].iloc[0]
        )
        marker_y = min(ci_low_left, ci_low_right) - pad
        marker_bottoms.append(marker_y)
        color = CONDITION_COLORS[condition]
        ax.text(
            (x1 + x2) / 2,
            marker_y,
            stars,
            ha="center",
            va="top",
            fontsize=STAR_FONT_SIZE,
            fontweight="bold",
            color=color,
        )

    if marker_bottoms:
        ax.set_ylim(min(y_min, min(marker_bottoms) - 0.06 * y_span), y_max)


def annotate_weird_no_difference_emotional(
    ax: plt.Axes,
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    x_map: dict[tuple[str, str], float],
) -> None:
    weird_rows = tests[tests["Group"] == "WEIRD"]
    if weird_rows.empty or not bool((weird_rows["p"] >= ALPHA).all()):
        return

    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 0.2)
    pad = 0.02 * y_span
    x_center = (x_map[("WEIRD", "Interactive")] + x_map[("WEIRD", "Text")]) / 2
    ci_low_weird = float(summary[summary["Group"] == "WEIRD"]["ci_low"].min())
    y_note = ci_low_weird - pad

    ax.text(
        x_center,
        y_note,
        "ns",
        ha="center",
        va="top",
        fontsize=STAR_FONT_SIZE,
        fontweight="bold",
        color="black",
    )
    ax.set_ylim(min(y_min, y_note - 0.06 * y_span), y_max)


def write_report(
    h4a_tests: pd.DataFrame,
    h4a_anova: tuple[float, float],
    h4a_interaction: dict[str, float],
    h4b_tests: pd.DataFrame,
    h4b_anova: tuple[float, float],
    h4b_interaction: dict[str, float],
    h4c_tests: pd.DataFrame,
    h4c_anova: tuple[float, float],
    h4c_interaction: dict[str, float],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 4 Results}")
    lines.append("Hypothesis 4 tested group x condition effects across three trust-change dimensions in a three-panel visualization.")
    lines.append("All trust-change outcomes are computed as per-item average change (Post - Pre).")
    lines.append("")

    lines.append(
        "4.1 Analytical trust change by Group x Condition (left panel): paired pre-post tests were "
        + "; ".join(
            [
                (
                    f"{row['Group']} / {CONDITION_LABELS[row['Condition']]}: "
                    f"$t({int(row['n']) - 1})={row['t']:.2f}$, "
                    f"$p {format_p(row['p'])}$, "
                    f"$\\Delta M={row['Mean Change (Post-Pre)']:.3f}$"
                )
                for _, row in h4a_tests.iterrows()
            ]
        )
        + ". "
        + f"Omnibus four-cell ANOVA: $F={h4a_anova[0]:.2f}$, $p {format_p(h4a_anova[1])}$. "
        + f"Interaction estimate: $b={h4a_interaction['coef']:.3f}$, $t({int(h4a_interaction['df'])})={h4a_interaction['t']:.2f}$, $p {format_p(h4a_interaction['p'])}$."
    )

    lines.append(
        "4.2 Emotional trust change by Group x Condition (middle panel): paired pre-post tests were "
        + "; ".join(
            [
                (
                    f"{row['Group']} / {CONDITION_LABELS[row['Condition']]}: "
                    f"$t({int(row['n']) - 1})={row['t']:.2f}$, "
                    f"$p {format_p(row['p'])}$, "
                    f"$\\Delta M={row['Mean Change (Post-Pre)']:.3f}$"
                )
                for _, row in h4b_tests.iterrows()
            ]
        )
        + ". "
        + f"Omnibus four-cell ANOVA: $F={h4b_anova[0]:.2f}$, $p {format_p(h4b_anova[1])}$. "
        + f"Interaction estimate: $b={h4b_interaction['coef']:.3f}$, $t({int(h4b_interaction['df'])})={h4b_interaction['t']:.2f}$, $p {format_p(h4b_interaction['p'])}$."
    )

    lines.append(
        "4.3 Analytical-vs-emotional change difference by Group x Condition (right panel): Welch tests by condition were "
        + "; ".join(
            [
                (
                    f"{CONDITION_LABELS[row['Condition']]}: "
                    f"$t={row['t']:.2f}$, "
                    f"$p {format_p(row['p'])}$, "
                    f"$M_{{WEIRD}}={row['Mean WEIRD']:.3f}$, "
                    f"$M_{{NORMAL}}={row['Mean NORMAL']:.3f}$"
                )
                for _, row in h4c_tests.iterrows()
            ]
        )
        + ". "
        + f"Omnibus four-cell ANOVA: $F={h4c_anova[0]:.2f}$, $p {format_p(h4c_anova[1])}$. "
        + f"Interaction estimate: $b={h4c_interaction['coef']:.3f}$, $t({int(h4c_interaction['df'])})={h4c_interaction['t']:.2f}$, $p {format_p(h4c_interaction['p'])}$."
    )

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    # Subhypothesis 4.1: Analytical pre-post change.
    h4a_summary = summarize_cells(analysis, "analytical_change")
    h4a_tests = paired_pre_post_tests(
        analysis,
        pre_col="Total Analytical Trust",
        post_col="Total Analytical Trust Post",
        scale_divisor=10.0,
    )
    h4a_anova = one_way_anova_four_cells(analysis, "analytical_change")
    h4a_interaction = interaction_effect(analysis, "analytical_change")

    # Subhypothesis 4.2: Emotional pre-post change.
    h4b_summary = summarize_cells(analysis, "emotional_change")
    h4b_tests = paired_pre_post_tests(
        analysis,
        pre_col="Total Emotional Trust",
        post_col="Total Emotional Trust Post",
        scale_divisor=9.0,
    )
    h4b_anova = one_way_anova_four_cells(analysis, "emotional_change")
    h4b_interaction = interaction_effect(analysis, "emotional_change")

    # Subhypothesis 4.3: Analytical vs emotional trust difference.
    h4c_summary = summarize_cells(analysis, "analytical_vs_emotional_diff")
    h4c_tests = welch_group_within_condition_tests(analysis, "analytical_vs_emotional_diff")
    h4c_anova = one_way_anova_four_cells(analysis, "analytical_vs_emotional_diff")
    h4c_interaction = interaction_effect(analysis, "analytical_vs_emotional_diff")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))

    xmap_left = draw_grouped_subplot(
        axes[0],
        h4a_summary,
        title="Analytical Trust Change by Group and Condition",
        ylabel="Analytical Change (Post - Pre, per-item)",
        show_legend=True,
    )
    annotate_significant_bars(axes[0], h4a_tests, h4a_summary, xmap_left)

    xmap_mid = draw_grouped_subplot(
        axes[1],
        h4b_summary,
        title="Emotional Trust Change by Group and Condition",
        ylabel="Emotional Change (Post - Pre, per-item)",
        show_legend=False,
    )
    annotate_significant_bars(axes[1], h4b_tests, h4b_summary, xmap_mid)
    annotate_weird_no_difference_emotional(axes[1], h4b_tests, h4b_summary, xmap_mid)

    xmap_right = draw_grouped_subplot(
        axes[2],
        h4c_summary,
        title="Analytical vs Emotional Trust Difference",
        ylabel="Analytical - Emotional Change Difference",
        show_legend=False,
    )
    annotate_diff_group_brackets(axes[2], h4c_tests, h4c_summary, xmap_right)

    axes[0].text(
        0.01,
        0.99,
        "Asterisks: * p < .05, ** p < .01, *** p < .001",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=15,
        color="black",
    )

    fig.suptitle("Hypothesis 4: Group x Condition Effects Across Trust Dimensions", fontsize=30)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOT_OUTPUT_PATH, dpi=150)
    plt.close(fig)

    report_text = write_report(
        h4a_tests=h4a_tests,
        h4a_anova=h4a_anova,
        h4a_interaction=h4a_interaction,
        h4b_tests=h4b_tests,
        h4b_anova=h4b_anova,
        h4b_interaction=h4b_interaction,
        h4c_tests=h4c_tests,
        h4c_anova=h4c_anova,
        h4c_interaction=h4c_interaction,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
