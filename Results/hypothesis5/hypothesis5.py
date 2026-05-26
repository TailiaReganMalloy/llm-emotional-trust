from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "hypothesis5.txt"
FIGURES_DIR = REPO_ROOT / "Figures"
PNG_PATH = FIGURES_DIR / "hypothesis5.png"
ALPHA = 0.05

ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0

SUBPLOT_TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14

GROUP_COLORS = {
    "WEIRD": "#377eb8",
    "NORMAL": "#c26a26",
}
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


def p_to_stars(pvalue: float) -> str:
    if np.isnan(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def p_text(pvalue: float) -> str:
    if np.isnan(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "< .001"
    return f"= {pvalue:.3f}".replace("0.", ".")


def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan, np.nan, np.nan

    mean = float(values.mean())
    n = len(values)
    if n < 2:
        return mean, mean, mean

    sem = float(values.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf(0.975, df=n - 1) * sem)
    return mean, mean - margin, mean + margin


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Country of residence",
        "Language",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    language = work["Language"].astype(str).str.strip().str.lower()
    work["Group"] = np.where(
        work["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith("english"),
        "WEIRD",
        "NORMAL",
    )

    work["analytical_pre"] = work["Total Analytical Trust"] / ANALYTICAL_MAX
    work["analytical_post"] = work["Total Analytical Trust Post"] / ANALYTICAL_MAX
    work["emotional_pre"] = work["Total Emotional Trust"] / EMOTIONAL_MAX
    work["emotional_post"] = work["Total Emotional Trust Post"] / EMOTIONAL_MAX

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]
    work["overall_change"] = (work["analytical_change"] + work["emotional_change"]) / 2.0

    # Match the standardized-change framing used in Hypothesis 2.4 / 3.4 style analyses.
    work["analytical_change_z"] = zscore(work["analytical_change"])
    work["emotional_change_z"] = zscore(work["emotional_change"])
    work["overall_change_z"] = zscore(work["overall_change"])

    return work


def welch_group_test(frame: pd.DataFrame, metric_col: str) -> dict[str, float]:
    weird = frame.loc[frame["Group"] == "WEIRD", metric_col].dropna().astype(float)
    normal = frame.loc[frame["Group"] == "NORMAL", metric_col].dropna().astype(float)

    weird_mean, weird_low, weird_high = mean_ci(weird)
    normal_mean, normal_low, normal_high = mean_ci(normal)

    if len(weird) < 2 or len(normal) < 2:
        t_stat, pvalue = np.nan, np.nan
    else:
        t_stat, pvalue = ttest_ind(weird, normal, equal_var=False, nan_policy="omit")

    return {
        "n_weird": float(len(weird)),
        "n_normal": float(len(normal)),
        "mean_weird": weird_mean,
        "mean_normal": normal_mean,
        "ci_low_weird": weird_low,
        "ci_high_weird": weird_high,
        "ci_low_normal": normal_low,
        "ci_high_normal": normal_high,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(pvalue) if pd.notna(pvalue) else np.nan,
        "delta_weird_minus_normal": weird_mean - normal_mean,
    }


def add_pair_bracket(ax: plt.Axes, x1: float, x2: float, y: float, label: str, y_step: float) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=2.0, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, label, ha="center", va="bottom", fontsize=18)


def draw_left_subplot(
    ax: plt.Axes,
    emotional_test: dict[str, float],
    analytical_test: dict[str, float],
) -> None:
    x = np.array([0.0, 1.0, 3.0, 4.0], dtype=float)
    means = np.array(
        [
            emotional_test["mean_weird"],
            emotional_test["mean_normal"],
            analytical_test["mean_weird"],
            analytical_test["mean_normal"],
        ],
        dtype=float,
    )
    lows = np.array(
        [
            emotional_test["ci_low_weird"],
            emotional_test["ci_low_normal"],
            analytical_test["ci_low_weird"],
            analytical_test["ci_low_normal"],
        ],
        dtype=float,
    )
    highs = np.array(
        [
            emotional_test["ci_high_weird"],
            emotional_test["ci_high_normal"],
            analytical_test["ci_high_weird"],
            analytical_test["ci_high_normal"],
        ],
        dtype=float,
    )

    ax.bar(
        x,
        means,
        width=0.58,
        color=[GROUP_COLORS["WEIRD"], GROUP_COLORS["NORMAL"]] * 2,
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=2.8,
        capsize=8,
        capthick=2.8,
        zorder=3,
    )

    y_min = float(np.nanmin(lows))
    y_max = float(np.nanmax(highs))
    y_span = max(y_max - y_min, 0.25)

    bracket_y = y_max + 0.10 * y_span
    add_pair_bracket(ax, x[0], x[1], bracket_y, p_to_stars(emotional_test["p"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], bracket_y, p_to_stars(analytical_test["p"]), 0.03 * y_span)

    summary = (
        f"5.1 Emotional z: t={emotional_test['t']:.3f}, p {p_text(emotional_test['p'])}\n"
        f"5.2 Analytical z: t={analytical_test['t']:.3f}, p {p_text(analytical_test['p'])}"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [GROUP_DISPLAY["WEIRD"], GROUP_DISPLAY["NORMAL"], GROUP_DISPLAY["WEIRD"], GROUP_DISPLAY["NORMAL"]],
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.text(0.5, -0.14, "Emotional z", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=14)
    ax.text(3.5, -0.14, "Analytical z", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=14)
    ax.set_ylabel("Standardized Change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "5.1 and 5.2 Group Contrasts (Emotional z and Analytical z)",
        fontsize=SUBPLOT_TITLE_FONTSIZE,
    )
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.40 * y_span)


def draw_right_subplot(ax: plt.Axes, overall_test: dict[str, float]) -> None:
    x = np.array([0.0, 1.0], dtype=float)
    means = np.array([overall_test["mean_weird"], overall_test["mean_normal"]], dtype=float)
    lows = np.array([overall_test["ci_low_weird"], overall_test["ci_low_normal"]], dtype=float)
    highs = np.array([overall_test["ci_high_weird"], overall_test["ci_high_normal"]], dtype=float)

    ax.bar(
        x,
        means,
        width=0.58,
        color=[GROUP_COLORS["WEIRD"], GROUP_COLORS["NORMAL"]],
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=[means - lows, highs - means],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=2.8,
        capsize=8,
        capthick=2.8,
        zorder=3,
    )

    y_min = float(np.nanmin(lows))
    y_max = float(np.nanmax(highs))
    y_span = max(y_max - y_min, 0.25)

    add_pair_bracket(ax, x[0], x[1], y_max + 0.10 * y_span, p_to_stars(overall_test["p"]), 0.03 * y_span)

    summary = f"5.3 Overall z: t={overall_test['t']:.3f}, p {p_text(overall_test['p'])}"
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_DISPLAY["WEIRD"], GROUP_DISPLAY["NORMAL"]], fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Standardized Change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("5.3 Group Contrast (Overall z)", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.40 * y_span)


def write_report(
    emotional_test: dict[str, float],
    analytical_test: dict[str, float],
    overall_test: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("\\subsubsection{Hypothesis 5 Results}")
    lines.append(
        "Hypothesis 5 compared WEIRD and NORMAL participants on standardized change metrics: emotional standardized change, analytical standardized change, and overall standardized change."
    )
    lines.append("")

    lines.append(
        "5.1 WEIRD vs NORMAL emotional standardized change: "
        f"$n_{{WEIRD}}={int(emotional_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(emotional_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={emotional_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={emotional_test['mean_normal']:.3f}$, "
        f"$t={emotional_test['t']:.3f}$, "
        f"$p {p_text(emotional_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={emotional_test['delta_weird_minus_normal']:.3f}$."
    )

    lines.append(
        "5.2 WEIRD vs NORMAL analytical standardized change: "
        f"$n_{{WEIRD}}={int(analytical_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(analytical_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={analytical_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={analytical_test['mean_normal']:.3f}$, "
        f"$t={analytical_test['t']:.3f}$, "
        f"$p {p_text(analytical_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={analytical_test['delta_weird_minus_normal']:.3f}$."
    )

    lines.append(
        "5.3 WEIRD vs NORMAL overall standardized change: "
        f"$n_{{WEIRD}}={int(overall_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(overall_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={overall_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={overall_test['mean_normal']:.3f}$, "
        f"$t={overall_test['t']:.3f}$, "
        f"$p {p_text(overall_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={overall_test['delta_weird_minus_normal']:.3f}$."
    )

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    emotional_test = welch_group_test(analysis, "emotional_change_z")
    analytical_test = welch_group_test(analysis, "analytical_change_z")
    overall_test = welch_group_test(analysis, "overall_change_z")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), constrained_layout=True)

    draw_left_subplot(axes[0], emotional_test, analytical_test)
    draw_right_subplot(axes[1], overall_test)

    axes[0].text(
        0.01,
        0.90,
        "Asterisks: * p < .05, ** p < .01, *** p < .001",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
    )

    fig.suptitle("Hypothesis 5: WEIRD vs non-WEIRD Standardized Trust-Change Contrasts", fontsize=26)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(emotional_test, analytical_test, overall_test)

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
