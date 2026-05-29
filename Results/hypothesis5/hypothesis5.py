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

GROUP_COLORS = {"WEIRD": "#377eb8", "NORMAL": "#c26a26"}
GROUP_DISPLAY = {"WEIRD": "WEIRD", "NORMAL": "non-WEIRD"}

UPDATED_DESCRIPTION = (
    "Hypothesis 5 focuses on contrasts that can explain directional differences between analytical and emotional trust outcomes. "
    "The figure includes four analyses in this order: analytical-emotional change contrast, pre-trust component source contrast, "
    "post-trust component source contrast, and change component source contrast."
)

WESTERN_COUNTRIES = {
    "United States", "United Kingdom", "Canada", "Australia", "New Zealand", "Ireland", "Germany", "France",
    "Netherlands", "Belgium", "Switzerland", "Austria", "Denmark", "Sweden", "Norway", "Finland", "Iceland",
    "Luxembourg", "Italy", "Spain", "Portugal",
}
HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}
WEIRD_EMPLOYMENT = {"Full-Time", "Not in paid work (e.g. homemaker', 'retired or disabled)"}


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
    if row.get("Nationality") in WESTERN_COUNTRIES or row.get("Country of birth") in WESTERN_COUNTRIES:
        score += 1
    return score


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


def add_stat_box(ax: plt.Axes, lines: list[str]) -> None:
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )


def sig_phrase(pvalue: float, label: str) -> str:
    if pd.isna(pvalue):
        return f"{label}: significance unavailable"
    if pvalue < ALPHA:
        return f"Significant {label}"
    return f"No significant {label}"


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


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Group"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")

    work["analytical_pre"] = pd.to_numeric(work["Total Analytical Trust"], errors="coerce") / ANALYTICAL_MAX
    work["analytical_post"] = pd.to_numeric(work["Total Analytical Trust Post"], errors="coerce") / ANALYTICAL_MAX
    work["emotional_pre"] = pd.to_numeric(work["Total Emotional Trust"], errors="coerce") / EMOTIONAL_MAX
    work["emotional_post"] = pd.to_numeric(work["Total Emotional Trust Post"], errors="coerce") / EMOTIONAL_MAX

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]
    work["diff_change"] = work["analytical_change"] - work["emotional_change"]
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
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, label, ha="center", va="bottom", fontsize=14)


def draw_diff_subplot(ax: plt.Axes, diff_group_test: dict[str, float]) -> None:
    x = np.array([0.0, 1.0], dtype=float)
    means = np.array([diff_group_test["mean_weird"], diff_group_test["mean_normal"]], dtype=float)
    lows = np.array([diff_group_test["ci_low_weird"], diff_group_test["ci_low_normal"]], dtype=float)
    highs = np.array([diff_group_test["ci_high_weird"], diff_group_test["ci_high_normal"]], dtype=float)

    ax.bar(x, means, width=0.58, color=[GROUP_COLORS["WEIRD"], GROUP_COLORS["NORMAL"]], edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#4a4a4a", elinewidth=2.2, capsize=7, capthick=2.2, zorder=3)
    y_min = float(np.nanmin(lows))
    y_max = float(np.nanmax(highs))
    y_span = max(y_max - y_min, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_max + 0.10 * y_span, p_to_stars(diff_group_test["p"]), 0.03 * y_span)

    add_stat_box(
        ax,
        [
            f"t={diff_group_test['t']:.3f}, p {p_text(diff_group_test['p'])}",
            sig_phrase(diff_group_test["p"], "group effect"),
        ],
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.2, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_DISPLAY["WEIRD"], GROUP_DISPLAY["NORMAL"]], fontsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Analytical - Emotional Change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Analytical-Emotional Change Contrast", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.80 * y_span)


def draw_component_subplot(
    ax: plt.Axes,
    test_a: dict[str, float],
    test_b: dict[str, float],
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
) -> None:
    x = np.array([0.0, 1.0, 3.0, 4.0], dtype=float)
    means = np.array([test_a["mean_weird"], test_a["mean_normal"], test_b["mean_weird"], test_b["mean_normal"]], dtype=float)
    lows = np.array([test_a["ci_low_weird"], test_a["ci_low_normal"], test_b["ci_low_weird"], test_b["ci_low_normal"]], dtype=float)
    highs = np.array([test_a["ci_high_weird"], test_a["ci_high_normal"], test_b["ci_high_weird"], test_b["ci_high_normal"]], dtype=float)

    ax.bar(x, means, width=0.58, color=[GROUP_COLORS["WEIRD"], GROUP_COLORS["NORMAL"]] * 2, edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="none", ecolor="#4a4a4a", elinewidth=2.2, capsize=7, capthick=2.2, zorder=3)
    y_min = float(np.nanmin(lows))
    y_max = float(np.nanmax(highs))
    y_span = max(y_max - y_min, 0.25)
    add_pair_bracket(ax, x[0], x[1], y_max + 0.10 * y_span, p_to_stars(test_a["p"]), 0.03 * y_span)
    add_pair_bracket(ax, x[2], x[3], y_max + 0.10 * y_span, p_to_stars(test_b["p"]), 0.03 * y_span)

    add_stat_box(
        ax,
        [
            f"{label_a}: t={test_a['t']:.3f}, p {p_text(test_a['p'])}",
            f"{label_b}: t={test_b['t']:.3f}, p {p_text(test_b['p'])}",
        ],
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.2, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["WEIRD", "non-\nWEIRD", "WEIRD", "non-\nWEIRD"], fontsize=TICK_LABEL_FONTSIZE)
    ax.text(0.5, -0.14, label_a, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
    ax.text(3.5, -0.14, label_b, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.95 * y_span)


def write_report(
    diff_group_test: dict[str, float],
    analytical_pre_test: dict[str, float],
    emotional_pre_test: dict[str, float],
    analytical_post_test: dict[str, float],
    emotional_post_test: dict[str, float],
    analytical_component_test: dict[str, float],
    emotional_component_test: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("\\subsubsection{Hypothesis 5 Results}")
    lines.append(UPDATED_DESCRIPTION)
    lines.append("")
    lines.append(
        "5.1 Group contrast on $(\\Delta_{analytical} - \\Delta_{emotional})$: "
        f"$n_{{WEIRD}}={int(diff_group_test['n_weird'])}$, "
        f"$n_{{NORMAL}}={int(diff_group_test['n_normal'])}$, "
        f"$M_{{WEIRD}}={diff_group_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={diff_group_test['mean_normal']:.3f}$, "
        f"$t={diff_group_test['t']:.3f}$, "
        f"$p {p_text(diff_group_test['p'])}$, "
        f"$\\Delta(M_W-M_N)={diff_group_test['delta_weird_minus_normal']:.3f}$."
    )
    lines.append(
        "5.2 Pre component source test (analytical pre, WEIRD vs non-WEIRD): "
        f"$M_{{WEIRD}}={analytical_pre_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={analytical_pre_test['mean_normal']:.3f}$, "
        f"$t={analytical_pre_test['t']:.3f}$, "
        f"$p {p_text(analytical_pre_test['p'])}$."
    )
    lines.append(
        "5.3 Pre component source test (emotional pre, WEIRD vs non-WEIRD): "
        f"$M_{{WEIRD}}={emotional_pre_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={emotional_pre_test['mean_normal']:.3f}$, "
        f"$t={emotional_pre_test['t']:.3f}$, "
        f"$p {p_text(emotional_pre_test['p'])}$."
    )
    lines.append(
        "5.4 Post component source test (analytical post, WEIRD vs non-WEIRD): "
        f"$M_{{WEIRD}}={analytical_post_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={analytical_post_test['mean_normal']:.3f}$, "
        f"$t={analytical_post_test['t']:.3f}$, "
        f"$p {p_text(analytical_post_test['p'])}$."
    )
    lines.append(
        "5.5 Post component source test (emotional post, WEIRD vs non-WEIRD): "
        f"$M_{{WEIRD}}={emotional_post_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={emotional_post_test['mean_normal']:.3f}$, "
        f"$t={emotional_post_test['t']:.3f}$, "
        f"$p {p_text(emotional_post_test['p'])}$."
    )
    lines.append(
        "5.6 Change component source test (analytical change, WEIRD vs non-WEIRD): "
        f"$M_{{WEIRD}}={analytical_component_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={analytical_component_test['mean_normal']:.3f}$, "
        f"$t={analytical_component_test['t']:.3f}$, "
        f"$p {p_text(analytical_component_test['p'])}$."
    )
    lines.append(
        "5.7 Change component source test (emotional change, WEIRD vs non-WEIRD): "
        f"$M_{{WEIRD}}={emotional_component_test['mean_weird']:.3f}$, "
        f"$M_{{NORMAL}}={emotional_component_test['mean_normal']:.3f}$, "
        f"$t={emotional_component_test['t']:.3f}$, "
        f"$p {p_text(emotional_component_test['p'])}$."
    )
    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(df)

    diff_group_test = welch_group_test(analysis, "diff_change")
    analytical_pre_test = welch_group_test(analysis, "analytical_pre")
    emotional_pre_test = welch_group_test(analysis, "emotional_pre")
    analytical_post_test = welch_group_test(analysis, "analytical_post")
    emotional_post_test = welch_group_test(analysis, "emotional_post")
    analytical_component_test = welch_group_test(analysis, "analytical_change")
    emotional_component_test = welch_group_test(analysis, "emotional_change")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)

    draw_diff_subplot(axes[0], diff_group_test)
    draw_component_subplot(
        axes[1],
        analytical_pre_test,
        emotional_pre_test,
        label_a="Analytical pre",
        label_b="Emotional pre",
        ylabel="Pre trust (normalized)",
        title="Pre-Trust Component\nSource Contrast",
    )
    draw_component_subplot(
        axes[2],
        analytical_post_test,
        emotional_post_test,
        label_a="Analytical post",
        label_b="Emotional post",
        ylabel="Post trust (normalized)",
        title="Post-Trust Component\nSource Contrast",
    )
    draw_component_subplot(
        axes[3],
        analytical_component_test,
        emotional_component_test,
        label_a="Analytical change",
        label_b="Emotional change",
        ylabel="Change (Post - Pre, normalized)",
        title="Change Component\nSource Contrast",
    )

    fig.suptitle("Hypothesis 5: Group Contrast and Component Source Contrasts", fontsize=20)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(
        diff_group_test,
        analytical_pre_test,
        emotional_pre_test,
        analytical_post_test,
        emotional_post_test,
        analytical_component_test,
        emotional_component_test,
    )

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
