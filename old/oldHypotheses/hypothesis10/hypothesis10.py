from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis10.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis10.txt"
ALPHA = 0.05

GROUP_ORDER = ["WEIRD", "NORMAL"]
GROUP_LABELS = {
    "WEIRD": "WEIRD",
    "NORMAL": "NORMAL",
}
GROUP_COLORS = {
    "WEIRD": "#377eb8",
    "NORMAL": "#c26a26",
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
        "Country of residence",
        "Language",
        "Total Analytical Trust",
        "Total Emotional Trust",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    language = work["Language"].astype(str).str.strip().str.lower()
    residence = work["Country of residence"].astype(str).str.strip()
    work["Group"] = np.where(
        residence.isin(WESTERN_COUNTRIES) & language.str.startswith("english"),
        "WEIRD",
        "NORMAL",
    )

    work["overall_pre"] = work["Total Analytical Trust"] + work["Total Emotional Trust"]
    work["analytical_pre"] = work["Total Analytical Trust"]
    work["emotional_pre"] = work["Total Emotional Trust"]

    return work


def group_welch_test(data: pd.DataFrame, value_col: str) -> dict[str, float]:
    weird_vals = data.loc[data["Group"] == "WEIRD", value_col].dropna().astype(float)
    normal_vals = data.loc[data["Group"] == "NORMAL", value_col].dropna().astype(float)

    w_mean, w_low, w_high = mean_ci(weird_vals)
    n_mean, n_low, n_high = mean_ci(normal_vals)

    if len(weird_vals) < 2 or len(normal_vals) < 2:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = ttest_ind(weird_vals, normal_vals, equal_var=False, nan_policy="omit")

    return {
        "n_weird": float(len(weird_vals)),
        "n_normal": float(len(normal_vals)),
        "mean_weird": w_mean,
        "mean_normal": n_mean,
        "ci_low_weird": w_low,
        "ci_high_weird": w_high,
        "ci_low_normal": n_low,
        "ci_high_normal": n_high,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(p_value) if pd.notna(p_value) else np.nan,
        "delta_weird_minus_normal": w_mean - n_mean,
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
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, star_text, ha="center", va="bottom", fontsize=18)


def draw_group_subplot(
    ax: plt.Axes,
    test_result: dict[str, float],
    title: str,
    ylabel: str,
) -> None:
    x = np.array([0.0, 1.0])
    means = np.array([test_result["mean_weird"], test_result["mean_normal"]], dtype=float)
    lows = np.array([test_result["ci_low_weird"], test_result["ci_low_normal"]], dtype=float)
    highs = np.array([test_result["ci_high_weird"], test_result["ci_high_normal"]], dtype=float)

    ax.bar(
        x,
        means,
        width=0.58,
        color=[GROUP_COLORS[g] for g in GROUP_ORDER],
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
        significance_label(float(test_result["p"])),
        0.03 * y_span,
    )

    interpretation = "No significant WEIRD vs NORMAL pre-trust difference" if float(test_result["p"]) >= ALPHA else "Significant WEIRD vs NORMAL pre-trust difference"
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
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(y_min - 0.18 * y_span, y_max + 0.40 * y_span)


def write_report(
    overall_test: dict[str, float],
    analytical_test: dict[str, float],
    emotional_test: dict[str, float],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 10 Results}")
    lines.append(
        "Hypothesis 10 tested whether WEIRD and NORMAL participants differ in pre-experiment overall trust, analytical trust, and emotional trust."
    )
    lines.append("")

    lines.append("10.1 WEIRD vs NORMAL difference in pre-experiment overall trust:")
    lines.append(
        "Overall pre-trust group contrast: "
        f"$n_{{WEIRD}}={int(overall_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(overall_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={overall_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={overall_test['mean_normal']:.3f}$, "
        f"$t={overall_test['t']:.3f}$, "
        f"$p {format_p(overall_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={overall_test['delta_weird_minus_normal']:.3f}$."
    )
    if float(overall_test["p"]) >= ALPHA:
        lines.append("Interpretation: no significant WEIRD vs NORMAL difference in pre-experiment overall trust.")
    else:
        lines.append("Interpretation: significant WEIRD vs NORMAL difference in pre-experiment overall trust.")

    lines.append("")
    lines.append("10.2 WEIRD vs NORMAL difference in pre-experiment analytical trust:")
    lines.append(
        "Analytical pre-trust group contrast: "
        f"$n_{{WEIRD}}={int(analytical_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(analytical_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={analytical_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={analytical_test['mean_normal']:.3f}$, "
        f"$t={analytical_test['t']:.3f}$, "
        f"$p {format_p(analytical_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={analytical_test['delta_weird_minus_normal']:.3f}$."
    )
    if float(analytical_test["p"]) >= ALPHA:
        lines.append("Interpretation: no significant WEIRD vs NORMAL difference in pre-experiment analytical trust.")
    else:
        lines.append("Interpretation: significant WEIRD vs NORMAL difference in pre-experiment analytical trust.")

    lines.append("")
    lines.append("10.3 WEIRD vs NORMAL difference in pre-experiment emotional trust:")
    lines.append(
        "Emotional pre-trust group contrast: "
        f"$n_{{WEIRD}}={int(emotional_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(emotional_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={emotional_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={emotional_test['mean_normal']:.3f}$, "
        f"$t={emotional_test['t']:.3f}$, "
        f"$p {format_p(emotional_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={emotional_test['delta_weird_minus_normal']:.3f}$."
    )
    if float(emotional_test["p"]) >= ALPHA:
        lines.append("Interpretation: no significant WEIRD vs NORMAL difference in pre-experiment emotional trust.")
    else:
        lines.append("Interpretation: significant WEIRD vs NORMAL difference in pre-experiment emotional trust.")

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    overall_test = group_welch_test(analysis, "overall_pre")
    analytical_test = group_welch_test(analysis, "analytical_pre")
    emotional_test = group_welch_test(analysis, "emotional_pre")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(25, 9), constrained_layout=True)

    draw_group_subplot(
        axes[0],
        overall_test,
        title="10.1 Pre-Experiment Overall Trust (WEIRD vs NORMAL)",
        ylabel="Overall Pre-Trust Score",
    )
    draw_group_subplot(
        axes[1],
        analytical_test,
        title="10.2 Pre-Experiment Analytical Trust (WEIRD vs NORMAL)",
        ylabel="Analytical Pre-Trust Score",
    )
    draw_group_subplot(
        axes[2],
        emotional_test,
        title="10.3 Pre-Experiment Emotional Trust (WEIRD vs NORMAL)",
        ylabel="Emotional Pre-Trust Score",
    )

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

    fig.suptitle("Hypothesis 10: WEIRD vs NORMAL Differences in Pre-Experiment Trust", fontsize=22)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=160)
    plt.close(fig)

    report_text = write_report(
        overall_test=overall_test,
        analytical_test=analytical_test,
        emotional_test=emotional_test,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
