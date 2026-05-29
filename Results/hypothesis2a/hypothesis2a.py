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
PLOT_OUTPUT_PATH = FIGURES_DIR / "hypothesis2a.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis2a.txt"
ALPHA = 0.05

SUBPLOT_TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 18

CONDITION_ORDER = ["Interactive", "Text"]
CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static",
}
CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}
GROUP_ORDER = ["WEIRD", "NORMAL"]
GROUP_DISPLAY = {
    "WEIRD": "WEIRD",
    "NORMAL": "non-WEIRD",
}

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

HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}
WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}


def score_weird(row: pd.Series) -> int:
    score = 0
    if row.get("Country of residence") in WESTERN_COUNTRIES:
        score += 1
    if str(row.get("Language", "")).lower().startswith("english"):
        score += 1
    if row.get("Education") in HIGH_EDUCATION:
        score += 1
    if row.get("Employment status") in WEIRD_EMPLOYMENT:
        score += 1
    if str(row.get("Ethnicity simplified", "")).strip().lower() == "white":
        score += 1
    if (row.get("Nationality") in WESTERN_COUNTRIES
            or row.get("Country of birth") in WESTERN_COUNTRIES):
        score += 1
    return score


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


def significance_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    stars = significance_stars(p_value)
    return stars if stars else "ns"


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
        "Country of residence",
        "Language",
        "Education",
        "Employment status",
        "Ethnicity simplified",
        "Nationality",
        "Country of birth",
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

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Group"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")

    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["overall_change"] = (work["analytical_change"] + work["emotional_change"]) / 2.0

    return work


def condition_welch_test(data: pd.DataFrame, value_col: str, group: str) -> dict[str, float | str]:
    subset = data[data["Group"] == group].copy()

    interactive_vals = subset.loc[subset["Condition"] == "Interactive", value_col].dropna().astype(float)
    text_vals = subset.loc[subset["Condition"] == "Text", value_col].dropna().astype(float)

    i_mean, i_low, i_high = mean_ci(interactive_vals)
    t_mean, t_low, t_high = mean_ci(text_vals)

    if len(interactive_vals) < 2 or len(text_vals) < 2:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")

    return {
        "group": group,
        "n_group": float(len(subset)),
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
    label: str,
    y_step: float,
) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=2.0, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, label, ha="center", va="bottom", fontsize=18)


def draw_group_split_subplot(
    ax: plt.Axes,
    weird_test: dict[str, float | str],
    normal_test: dict[str, float | str],
    title: str,
    ylabel: str,
) -> None:
    x = np.array([0.0, 1.0, 3.0, 4.0], dtype=float)
    means = np.array(
        [
            weird_test["mean_interactive"],
            weird_test["mean_text"],
            normal_test["mean_interactive"],
            normal_test["mean_text"],
        ],
        dtype=float,
    )
    lows = np.array(
        [
            weird_test["ci_low_interactive"],
            weird_test["ci_low_text"],
            normal_test["ci_low_interactive"],
            normal_test["ci_low_text"],
        ],
        dtype=float,
    )
    highs = np.array(
        [
            weird_test["ci_high_interactive"],
            weird_test["ci_high_text"],
            normal_test["ci_high_interactive"],
            normal_test["ci_high_text"],
        ],
        dtype=float,
    )

    ax.bar(
        x,
        means,
        width=0.58,
        color=[CONDITION_COLORS["Interactive"], CONDITION_COLORS["Text"]] * 2,
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
    bracket_y = y_max + 0.10 * y_span
    step = 0.03 * y_span

    add_pair_bracket(
        ax,
        x[0],
        x[1],
        bracket_y,
        significance_label(float(weird_test["p"])),
        step,
    )
    add_pair_bracket(
        ax,
        x[2],
        x[3],
        bracket_y,
        significance_label(float(normal_test["p"])),
        step,
    )

    summary = (
        f"{GROUP_DISPLAY['WEIRD']}: t={float(weird_test['t']):.3f}, p {format_p(float(weird_test['p']))}\n"
        f"{GROUP_DISPLAY['NORMAL']}: t={float(normal_test['t']):.3f}, p {format_p(float(normal_test['p']))}"
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
    ax.set_xlim(-0.8, 4.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            CONDITION_LABELS["Interactive"],
            CONDITION_LABELS["Text"],
            CONDITION_LABELS["Interactive"],
            CONDITION_LABELS["Text"],
        ],
        fontsize=TICK_LABEL_FONTSIZE,
    )

    ax.text(
        0.5,
        -0.08,
        GROUP_DISPLAY["WEIRD"],
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=24,
    )
    ax.text(
        3.5,
        -0.08,
        GROUP_DISPLAY["NORMAL"],
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=24,
    )

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(-0.4, 0.2)
    ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2])


def interpretation_line(weird_test: dict[str, float | str], normal_test: dict[str, float | str]) -> str:
    significant_groups: list[str] = []

    weird_p = float(weird_test["p"])
    if pd.notna(weird_p) and weird_p < ALPHA:
        significant_groups.append("WEIRD")

    normal_p = float(normal_test["p"])
    if pd.notna(normal_p) and normal_p < ALPHA:
        significant_groups.append("NORMAL")

    if not significant_groups:
        return "Interpretation: no significant condition effect detected in either WEIRD or NORMAL."
    return "Interpretation: significant condition effect detected in " + ", ".join(significant_groups) + "."


def report_group_line(test_result: dict[str, float | str]) -> str:
    return (
        f"- {test_result['group']}: "
        f"$n_{{Interactive}}={int(float(test_result['n_interactive']))}$, "
        f"$n_{{Text}}={int(float(test_result['n_text']))}$, "
        f"$M_{{Interactive}}={float(test_result['mean_interactive']):.3f}$, "
        f"$M_{{Text}}={float(test_result['mean_text']):.3f}$, "
        f"$t={float(test_result['t']):.3f}$, "
        f"$p {format_p(float(test_result['p']))}$, "
        f"$\\Delta(M_I-M_T)={float(test_result['delta_interactive_minus_text']):.3f}$."
    )


def write_report(
    overall_tests: dict[str, dict[str, float | str]],
    emotional_tests: dict[str, dict[str, float | str]],
    analytical_tests: dict[str, dict[str, float | str]],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 8 Results}")
    lines.append(
        "Hypothesis 8 tested whether the main effect of condition differs across overall, emotional, and analytical trust change when analyzed separately within WEIRD and NORMAL participants."
    )
    lines.append("")

    lines.append("8.1 Main condition effect on overall change (split by WEIRD/NORMAL):")
    lines.append(report_group_line(overall_tests["WEIRD"]))
    lines.append(report_group_line(overall_tests["NORMAL"]))
    lines.append(interpretation_line(overall_tests["WEIRD"], overall_tests["NORMAL"]))

    lines.append("")
    lines.append("8.2 Main condition effect on emotional change (split by WEIRD/NORMAL):")
    lines.append(report_group_line(emotional_tests["WEIRD"]))
    lines.append(report_group_line(emotional_tests["NORMAL"]))
    lines.append(interpretation_line(emotional_tests["WEIRD"], emotional_tests["NORMAL"]))

    lines.append("")
    lines.append("8.3 Main condition effect on analytical change (split by WEIRD/NORMAL):")
    lines.append(report_group_line(analytical_tests["WEIRD"]))
    lines.append(report_group_line(analytical_tests["NORMAL"]))
    lines.append(interpretation_line(analytical_tests["WEIRD"], analytical_tests["NORMAL"]))

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    overall_tests = {
        group: condition_welch_test(analysis, "overall_change", group)
        for group in GROUP_ORDER
    }
    emotional_tests = {
        group: condition_welch_test(analysis, "emotional_change", group)
        for group in GROUP_ORDER
    }
    analytical_tests = {
        group: condition_welch_test(analysis, "analytical_change", group)
        for group in GROUP_ORDER
    }

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)

    draw_group_split_subplot(
        axes[0],
        overall_tests["WEIRD"],
        overall_tests["NORMAL"],
        title="2.1 Follow-Up Main Condition Effect:\nOverall Change (WEIRD vs non-WEIRD)",
        ylabel="Overall Change",
    )
    draw_group_split_subplot(
        axes[1],
        emotional_tests["WEIRD"],
        emotional_tests["NORMAL"],
        title="2.2 Follow-Up Main Condition Effect:\nEmotional Change (WEIRD vs non-WEIRD)",
        ylabel="Emotional Change",
    )
    draw_group_split_subplot(
        axes[2],
        analytical_tests["WEIRD"],
        analytical_tests["NORMAL"],
        title="2.3 Follow-Up Main Condition Effect:\nAnalytical Change (WEIRD vs non-WEIRD)",
        ylabel="Analytical Change",
    )

    fig.suptitle("Hypothesis 2 Follow-Up Condition Effects by WEIRD/non-WEIRD Across Trust Change Outcomes.", fontsize=30)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close(fig)

    report_text = write_report(
        overall_tests=overall_tests,
        emotional_tests=emotional_tests,
        analytical_tests=analytical_tests,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
