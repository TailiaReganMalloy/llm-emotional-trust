from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "hypothesis1.txt"
FIGURES_DIR = REPO_ROOT / "Figures"
PNG_PATH = FIGURES_DIR / "hypothesis1.png"
ALPHA = 0.05

ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0

SUBPLOT_TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 22

PREPOST_COLORS = {
    "Pre": "#377eb8",
    "Post": "#c26a26",
}
ERROR_COLOR = "#4a4a4a"


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


def mean_ci(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, ci


def cohens_d_paired(pre: pd.Series, post: pd.Series) -> float:
    diffs = post - pre
    if len(diffs) < 2:
        return np.nan
    sd = float(diffs.std(ddof=1))
    if np.isclose(sd, 0.0):
        return np.nan
    return float(diffs.mean() / sd)


def safe_wilcoxon(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    if len(pre) < 1:
        return np.nan, np.nan
    diffs = post - pre
    if np.allclose(diffs.to_numpy(dtype=float), 0.0):
        return 0.0, 1.0
    stat, pvalue = wilcoxon(pre, post, alternative="two-sided")
    return float(stat), float(pvalue)


def analyze_metric(pre: pd.Series, post: pd.Series) -> dict[str, float]:
    pairs = pd.DataFrame({"pre": pre, "post": post}).dropna()
    if len(pairs) < 2:
        return {
            "n": float(len(pairs)),
            "pre_mean": np.nan,
            "pre_ci": np.nan,
            "post_mean": np.nan,
            "post_ci": np.nan,
            "pre_sd": np.nan,
            "post_sd": np.nan,
            "t": np.nan,
            "p": np.nan,
            "w": np.nan,
            "w_p": np.nan,
            "d": np.nan,
            "delta_mean": np.nan,
        }

    t_stat, pvalue = ttest_rel(pairs["pre"], pairs["post"], nan_policy="omit")
    w_stat, w_pvalue = safe_wilcoxon(pairs["pre"], pairs["post"])
    pre_mean, pre_ci = mean_ci(pairs["pre"])
    post_mean, post_ci = mean_ci(pairs["post"])

    return {
        "n": float(len(pairs)),
        "pre_mean": pre_mean,
        "pre_ci": pre_ci,
        "post_mean": post_mean,
        "post_ci": post_ci,
        "pre_sd": float(pairs["pre"].std(ddof=1)),
        "post_sd": float(pairs["post"].std(ddof=1)),
        "t": float(t_stat),
        "p": float(pvalue),
        "w": float(w_stat),
        "w_p": float(w_pvalue),
        "d": cohens_d_paired(pairs["pre"], pairs["post"]),
        "delta_mean": float((pairs["post"] - pairs["pre"]).mean()),
    }


def draw_subplot(ax: plt.Axes, title: str, stats: dict[str, float]) -> None:
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
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

    y_top = float(np.nanmax(means + cis_clean))
    y_bottom = float(np.nanmin(means - cis_clean))
    y_span = max(y_top - y_bottom, 0.25)

    y_bracket = y_top + 0.07 * y_span
    y_text = y_bracket + 0.02 * y_span
    y_low = y_bottom - 0.15 * y_span
    y_high = y_bracket + 0.20 * y_span

    ax.plot(
        [0, 0, 1, 1],
        [y_bracket - 0.015 * y_span, y_bracket, y_bracket, y_bracket - 0.015 * y_span],
        color="black",
        lw=2.0,
        zorder=4,
    )
    ax.text(0.5, y_text, p_to_stars(stats["p"]), ha="center", va="bottom", fontsize=24, zorder=5)

    interpretation = "Significant pre-post effect" if float(stats["p"]) < ALPHA else "No significant pre-post effect"
    summary = f"t={stats['t']:.3f}, p {p_text(float(stats['p']))}\n{interpretation}"
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

    ax.set_ylim(bottom=0, top=0.5)


def write_report(results: dict[str, dict[str, float]]) -> None:
    lines: list[str] = []
    lines.append("\\subsubsection{Hypothesis 1 Results}")
    lines.append(
        "Hypothesis 1 tested pre-post differences in aggregate overall trust, emotional trust, and analytical trust."
    )
    lines.append("")

    order = [
        ("Overall Trust", "1.1"),
        ("Analytical Trust", "1.2"),
        ("Emotional Trust", "1.3"),
    ]

    for metric, label in order:
        r = results[metric]
        dfree = int(r["n"] - 1) if pd.notna(r["n"]) and r["n"] >= 2 else np.nan
        support_text = "supported" if pd.notna(r["p"]) and r["p"] < ALPHA else "not supported"
        lines.append(
            f"{label} {metric}: "
            f"$n={int(r['n'])}$, "
            f"$M_{{pre}}={r['pre_mean']:.3f}$, "
            f"$SD_{{pre}}={r['pre_sd']:.3f}$, "
            f"$M_{{post}}={r['post_mean']:.3f}$, "
            f"$SD_{{post}}={r['post_sd']:.3f}$, "
            f"$t({dfree})={r['t']:.3f}$, "
            f"$p {p_text(r['p'])}$, "
            f"$W={r['w']:.3f}$, "
            f"$p_{{W}} {p_text(r['w_p'])}$, "
            f"$d={r['d']:.3f}$, "
            f"$\\Delta={r['delta_mean']:.3f}$, "
            f"subhypothesis {support_text}."
        )

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    plt.style.use("ggplot")

    analytical_pre = df["Total Analytical Trust"] / ANALYTICAL_MAX
    analytical_post = df["Total Analytical Trust Post"] / ANALYTICAL_MAX
    emotional_pre = df["Total Emotional Trust"] / EMOTIONAL_MAX
    emotional_post = df["Total Emotional Trust Post"] / EMOTIONAL_MAX
    overall_pre = (analytical_pre + emotional_pre) / 2.0
    overall_post = (analytical_post + emotional_post) / 2.0

    results = {
        "Overall Trust": analyze_metric(overall_pre, overall_post),
        "Analytical Trust": analyze_metric(analytical_pre, analytical_post),
        "Emotional Trust": analyze_metric(emotional_pre, emotional_post),
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    for ax, (name, stats) in zip(axes, results.items()):
        draw_subplot(ax, name, stats)

    fig.suptitle("Hypothesis 1: Pre vs Post Experiment Trust Questionnaire", fontsize=28)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)

    write_report(results)

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
