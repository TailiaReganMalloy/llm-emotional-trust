from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "hypothesis4.txt"
FIGURES_DIR = REPO_ROOT / "Figures"
PNG_PATH = FIGURES_DIR / "hypothesis4.png"
ALPHA = 0.05

ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0

SUBPLOT_TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 20

PREPOST_COLORS = {
    "Pre": "#377eb8",
    "Post": "#c26a26",
}
TYPE_COLORS = {
    "Emotional z": "#377eb8",
    "Analytical z": "#c26a26",
}
ERROR_COLOR = "#4a4a4a"

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


def add_stat_box(ax: plt.Axes, stats: dict[str, float], effect_label: str) -> None:
    pvalue = float(stats["p"]) if pd.notna(stats.get("p")) else np.nan
    tvalue = float(stats["t"]) if pd.notna(stats.get("t")) else np.nan
    interpretation = (
        f"Significant {effect_label}" if pd.notna(pvalue) and pvalue < ALPHA else f"No significant {effect_label}"
    )
    summary = f"t={tvalue:.3f}, p {p_text(pvalue)}\n{interpretation}" if pd.notna(tvalue) else f"t=NA, p {p_text(pvalue)}\n{interpretation}"
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )


def mean_ci(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, ci


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def cohens_d_paired(a: pd.Series, b: pd.Series) -> float:
    diff = a - b
    if len(diff) < 2:
        return np.nan
    sd = float(diff.std(ddof=1))
    if np.isclose(sd, 0.0):
        return np.nan
    return float(diff.mean() / sd)


def safe_wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 1:
        return np.nan, np.nan
    diff = a - b
    if np.allclose(diff.to_numpy(dtype=float), 0.0):
        return 0.0, 1.0
    stat, pvalue = wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(pvalue)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["weird_score"] = work.apply(score_weird, axis=1)
    work["is_weird"] = work["weird_score"] >= 4

    work["analytical_pre"] = work["Total Analytical Trust"] / ANALYTICAL_MAX
    work["analytical_post"] = work["Total Analytical Trust Post"] / ANALYTICAL_MAX
    work["emotional_pre"] = work["Total Emotional Trust"] / EMOTIONAL_MAX
    work["emotional_post"] = work["Total Emotional Trust Post"] / EMOTIONAL_MAX

    work["overall_pre"] = (work["analytical_pre"] + work["emotional_pre"]) / 2.0
    work["overall_post"] = (work["analytical_post"] + work["emotional_post"]) / 2.0

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]

    work["analytical_change_z"] = zscore(work["analytical_change"].astype(float))
    work["emotional_change_z"] = zscore(work["emotional_change"].astype(float))
    return work


def paired_pre_post(pre: pd.Series, post: pd.Series) -> dict[str, float]:
    pair = pd.DataFrame({"pre": pre, "post": post}).dropna().copy()
    pre_vals = pair["pre"].astype(float)
    post_vals = pair["post"].astype(float)

    if len(pair) < 2:
        return {
            "n": float(len(pair)),
            "pre_mean": np.nan,
            "post_mean": np.nan,
            "pre_sd": np.nan,
            "post_sd": np.nan,
            "delta_mean": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "pre_ci": np.nan,
            "post_ci": np.nan,
        }

    t_stat, pvalue = ttest_rel(pre_vals, post_vals, nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(post_vals, pre_vals)
    pre_mean, pre_ci = mean_ci(pre_vals)
    post_mean, post_ci = mean_ci(post_vals)

    return {
        "n": float(len(pair)),
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "pre_sd": float(pre_vals.std(ddof=1)),
        "post_sd": float(post_vals.std(ddof=1)),
        "delta_mean": float((post_vals - pre_vals).mean()),
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(post_vals, pre_vals),
        "pre_ci": pre_ci,
        "post_ci": post_ci,
    }


def compare_trust_type(df: pd.DataFrame) -> dict[str, float]:
    pair = df[["emotional_change_z", "analytical_change_z"]].dropna().copy()
    emotional = pair["emotional_change_z"].astype(float)
    analytical = pair["analytical_change_z"].astype(float)

    if len(pair) < 2:
        return {
            "n": float(len(pair)),
            "em_mean": np.nan,
            "an_mean": np.nan,
            "em_sd": np.nan,
            "an_sd": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "em_ci": np.nan,
            "an_ci": np.nan,
            "diff_mean": np.nan,
        }

    t_stat, pvalue = ttest_rel(emotional, analytical, nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(emotional, analytical)
    em_mean, em_ci = mean_ci(emotional)
    an_mean, an_ci = mean_ci(analytical)

    return {
        "n": float(len(pair)),
        "em_mean": em_mean,
        "an_mean": an_mean,
        "em_sd": float(emotional.std(ddof=1)),
        "an_sd": float(analytical.std(ddof=1)),
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(emotional, analytical),
        "em_ci": em_ci,
        "an_ci": an_ci,
        "diff_mean": float((emotional - analytical).mean()),
    }


def draw_pre_post_subplot(ax: plt.Axes, title: str, stats: dict[str, float]) -> None:
    x = np.array([0.0, 1.0], dtype=float)
    labels = ["Pre", "Post"]
    means = np.array([stats["pre_mean"], stats["post_mean"]], dtype=float)
    cis = np.array([stats["pre_ci"], stats["post_ci"]], dtype=float)
    cis_clean = np.nan_to_num(cis, nan=0.0)

    for idx, label in enumerate(labels):
        ax.bar(
            x[idx],
            means[idx],
            width=0.8,
            color=PREPOST_COLORS[label],
            edgecolor="none",
            zorder=2,
        )
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=cis_clean[idx],
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=2.8,
            capsize=8,
            capthick=2.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE, pad=14)
    ax.set_ylabel("Trust Score (0-1)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    y_top = float(np.nanmax(means + cis_clean))
    y_bottom = float(np.nanmin(means - cis_clean))
    y_span = max(y_top - y_bottom, 0.25)

    y_bracket = y_top + 0.07 * y_span
    y_text = y_bracket + 0.02 * y_span
    y_low = y_bottom - 0.15 * y_span
    y_high = y_bracket + 0.50 * y_span

    ax.plot(
        [0, 0, 1, 1],
        [y_bracket - 0.015 * y_span, y_bracket, y_bracket, y_bracket - 0.015 * y_span],
        color="black",
        lw=2.0,
        zorder=4,
    )
    ax.text(0.5, y_text, p_to_stars(stats["p"]), ha="center", va="bottom", fontsize=24, zorder=5)
    add_stat_box(ax, stats, "pre-post effect")
    ax.set_ylim(bottom=y_low, top=y_high)


def draw_type_subplot(ax: plt.Axes, title: str, stats: dict[str, float]) -> None:
    labels = ["Emotional z", "Analytical z"]
    x = np.array([0.0, 1.0], dtype=float)
    means = np.array([stats["em_mean"], stats["an_mean"]], dtype=float)
    cis = np.array([stats["em_ci"], stats["an_ci"]], dtype=float)
    cis_clean = np.nan_to_num(cis, nan=0.0)

    for idx, label in enumerate(labels):
        ax.bar(
            x[idx],
            means[idx],
            width=0.8,
            color=TYPE_COLORS[label],
            edgecolor="none",
            zorder=2,
        )
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=cis_clean[idx],
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=2.8,
            capsize=8,
            capthick=2.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE, pad=14)
    ax.set_ylabel("Standardized Change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.axhline(0.0, color=ERROR_COLOR, linewidth=1.6, zorder=1)

    y_top = float(np.nanmax(means + cis_clean))
    y_bottom = float(np.nanmin(means - cis_clean))
    y_span = max(y_top - y_bottom, 0.25)

    y_bracket = y_top + 0.10 * y_span
    y_text = y_bracket + 0.03 * y_span
    y_low = y_bottom - 0.18 * y_span
    y_high = y_bracket + 0.52 * y_span

    ax.plot(
        [0, 0, 1, 1],
        [y_bracket - 0.02 * y_span, y_bracket, y_bracket, y_bracket - 0.02 * y_span],
        color="black",
        lw=2.0,
        zorder=4,
    )
    ax.text(0.5, y_text, p_to_stars(stats["p"]), ha="center", va="bottom", fontsize=24, zorder=5)
    add_stat_box(ax, stats, "trust-type difference")
    ax.set_ylim(bottom=y_low, top=y_high)


def write_report(
    overall_stats: dict[str, float],
    analytical_stats: dict[str, float],
    emotional_stats: dict[str, float],
    type_stats: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("\\subsubsection{Hypothesis 3 Results}")
    lines.append(
        "Hypothesis 3 tested NORMAL/non-WEIRD participants with four subtests: 3.1 overall pre-post significance, 3.2 analytical pre-post significance, 3.3 emotional pre-post significance, and 3.4 whether emotional and analytical standardized change do not differ."
    )
    lines.append("")

    lines.append(
        "3.1 NORMAL Overall pre-post: "
        f"$n={int(overall_stats['n'])}$, "
        f"$M_{{pre}}={overall_stats['pre_mean']:.3f}$, "
        f"$M_{{post}}={overall_stats['post_mean']:.3f}$, "
        f"$t({int(overall_stats['n'] - 1)})={overall_stats['t']:.3f}$, "
        f"$p {p_text(overall_stats['p'])}$, "
        f"$W={overall_stats['w']:.3f}$, "
        f"$p_{{W}} {p_text(overall_stats['w_p'])}$, "
        f"$d={overall_stats['d']:.3f}$, "
        f"$\\Delta={overall_stats['delta_mean']:.3f}$."
    )

    lines.append(
        "3.2 NORMAL Analytical pre-post: "
        f"$n={int(analytical_stats['n'])}$, "
        f"$M_{{pre}}={analytical_stats['pre_mean']:.3f}$, "
        f"$M_{{post}}={analytical_stats['post_mean']:.3f}$, "
        f"$t({int(analytical_stats['n'] - 1)})={analytical_stats['t']:.3f}$, "
        f"$p {p_text(analytical_stats['p'])}$, "
        f"$W={analytical_stats['w']:.3f}$, "
        f"$p_{{W}} {p_text(analytical_stats['w_p'])}$, "
        f"$d={analytical_stats['d']:.3f}$, "
        f"$\\Delta={analytical_stats['delta_mean']:.3f}$."
    )

    lines.append(
        "3.3 NORMAL Emotional pre-post: "
        f"$n={int(emotional_stats['n'])}$, "
        f"$M_{{pre}}={emotional_stats['pre_mean']:.3f}$, "
        f"$M_{{post}}={emotional_stats['post_mean']:.3f}$, "
        f"$t({int(emotional_stats['n'] - 1)})={emotional_stats['t']:.3f}$, "
        f"$p {p_text(emotional_stats['p'])}$, "
        f"$W={emotional_stats['w']:.3f}$, "
        f"$p_{{W}} {p_text(emotional_stats['w_p'])}$, "
        f"$d={emotional_stats['d']:.3f}$, "
        f"$\\Delta={emotional_stats['delta_mean']:.3f}$."
    )

    no_difference = pd.notna(type_stats["p"]) and type_stats["p"] >= ALPHA
    lines.append(
        "3.4 NORMAL emotional-vs-analytical standardized change: "
        f"$n={int(type_stats['n'])}$, "
        f"$M_{{em,z}}={type_stats['em_mean']:.3f}$, "
        f"$M_{{an,z}}={type_stats['an_mean']:.3f}$, "
        f"$t({int(type_stats['n'] - 1)})={type_stats['t']:.3f}$, "
        f"$p {p_text(type_stats['p'])}$, "
        f"$W={type_stats['w']:.3f}$, "
        f"$p_{{W}} {p_text(type_stats['w_p'])}$, "
        f"$d={type_stats['d']:.3f}$, "
        f"$\\Delta_{{z}}={type_stats['diff_mean']:.3f}$. "
        + (
            "This supports the predicted no-difference test."
            if no_difference
            else "This does not support the predicted no-difference test."
        )
    )

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    plt.style.use("ggplot")

    analysis = build_analysis_frame(df)
    normal = analysis[~analysis["is_weird"]].copy()

    overall_stats = paired_pre_post(normal["overall_pre"], normal["overall_post"])
    analytical_stats = paired_pre_post(normal["analytical_pre"], normal["analytical_post"])
    emotional_stats = paired_pre_post(normal["emotional_pre"], normal["emotional_post"])
    type_stats = compare_trust_type(normal)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    draw_pre_post_subplot(axes[0], "4.1 Overall Pre vs Post", overall_stats)
    draw_pre_post_subplot(axes[1], "4.2 Analytical Pre vs Post", analytical_stats)
    draw_pre_post_subplot(axes[2], "4.3 Emotional Pre vs Post", emotional_stats)
    draw_type_subplot(axes[3], "4.4 Emotional vs Analytical Difference", type_stats)

    fig.suptitle("Hypothesis 4: non-WEIRD Trust Change Pattern", fontsize=27)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(overall_stats, analytical_stats, emotional_stats, type_stats)

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
