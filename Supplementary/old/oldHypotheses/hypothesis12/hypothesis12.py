from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis12.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis12.txt"
ALPHA = 0.05

GROUP_ORDER = ["WEIRD", "NORMAL"]
GROUP_COLORS = {
    "Pre": "#377eb8",
    "Post": "#c26a26",
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
        "Total Analytical Trust Post",
        "Total Emotional Trust Post",
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
    work["overall_post"] = work["Total Analytical Trust Post"] + work["Total Emotional Trust Post"]
    work["analytical_pre"] = work["Total Analytical Trust"]
    work["analytical_post"] = work["Total Analytical Trust Post"]
    work["emotional_pre"] = work["Total Emotional Trust"]
    work["emotional_post"] = work["Total Emotional Trust Post"]

    return work


def paired_prepost_test(data: pd.DataFrame, group: str, pre_col: str, post_col: str) -> dict[str, float]:
    pairs = data.loc[data["Group"] == group, [pre_col, post_col]].dropna().copy()
    pre_vals = pairs[pre_col].astype(float)
    post_vals = pairs[post_col].astype(float)

    pre_mean, pre_low, pre_high = mean_ci(pre_vals)
    post_mean, post_low, post_high = mean_ci(post_vals)

    if len(pairs) < 2:
        t_stat, p_value = np.nan, np.nan
    else:
        t_stat, p_value = ttest_rel(pre_vals, post_vals, nan_policy="omit")

    return {
        "n": float(len(pairs)),
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "ci_low_pre": pre_low,
        "ci_high_pre": pre_high,
        "ci_low_post": post_low,
        "ci_high_post": post_high,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(p_value) if pd.notna(p_value) else np.nan,
        "delta_post_minus_pre": post_mean - pre_mean,
    }


def did_interaction_test(data: pd.DataFrame, pre_col: str, post_col: str) -> dict[str, float]:
    use = data[["Group", pre_col, post_col]].dropna().copy().reset_index(drop=True)
    if use.empty:
        return {"coef": np.nan, "t": np.nan, "df": np.nan, "p": np.nan}

    rows: list[dict[str, float]] = []
    for _, row in use.iterrows():
        group_bin = 1.0 if row["Group"] == "WEIRD" else 0.0
        rows.append({"score": float(row[pre_col]), "group_bin": group_bin, "time_bin": 0.0})
        rows.append({"score": float(row[post_col]), "group_bin": group_bin, "time_bin": 1.0})

    long_df = pd.DataFrame(rows)
    y = long_df["score"].to_numpy(dtype=float)
    group = long_df["group_bin"].to_numpy(dtype=float)
    time = long_df["time_bin"].to_numpy(dtype=float)
    interaction = group * time

    X = np.column_stack([np.ones(len(long_df)), group, time, interaction])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    n, p = X.shape
    dof = n - p
    if dof <= 0:
        return {"coef": np.nan, "t": np.nan, "df": np.nan, "p": np.nan}

    rss = float(np.sum(residuals**2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    coef = float(beta[3])
    t_stat = coef / float(se[3]) if not np.isclose(float(se[3]), 0.0) else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if pd.notna(t_stat) else np.nan

    return {
        "coef": coef,
        "t": float(t_stat),
        "df": float(dof),
        "p": float(p_value),
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
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, label, ha="center", va="bottom", fontsize=16)


def draw_did_subplot(
    ax: plt.Axes,
    weird_stats: dict[str, float],
    normal_stats: dict[str, float],
    interaction: dict[str, float],
    title: str,
    ylabel: str,
) -> None:
    x = np.array([0.0, 1.0, 3.0, 4.0], dtype=float)
    means = np.array(
        [
            weird_stats["pre_mean"],
            weird_stats["post_mean"],
            normal_stats["pre_mean"],
            normal_stats["post_mean"],
        ],
        dtype=float,
    )
    lows = np.array(
        [
            weird_stats["ci_low_pre"],
            weird_stats["ci_low_post"],
            normal_stats["ci_low_pre"],
            normal_stats["ci_low_post"],
        ],
        dtype=float,
    )
    highs = np.array(
        [
            weird_stats["ci_high_pre"],
            weird_stats["ci_high_post"],
            normal_stats["ci_high_pre"],
            normal_stats["ci_high_post"],
        ],
        dtype=float,
    )

    ax.bar(
        x,
        means,
        width=0.58,
        color=[GROUP_COLORS["Pre"], GROUP_COLORS["Post"]] * 2,
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

    add_pair_bracket(
        ax,
        x[0],
        x[1],
        bracket_y,
        significance_label(float(weird_stats["p"])),
        0.03 * y_span,
    )
    add_pair_bracket(
        ax,
        x[2],
        x[3],
        bracket_y,
        significance_label(float(normal_stats["p"])),
        0.03 * y_span,
    )

    summary = (
        f"WEIRD pre-post: t={weird_stats['t']:.3f}, p {format_p(float(weird_stats['p']))}\n"
        f"NORMAL pre-post: t={normal_stats['t']:.3f}, p {format_p(float(normal_stats['p']))}\n"
        f"Group x Time interaction: b={interaction['coef']:.3f}, t={interaction['t']:.3f}, p {format_p(float(interaction['p']))}"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )

    ax.set_xlim(-0.8, 4.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pre", "Post", "Pre", "Post"], fontsize=11)
    ax.text(0.5, -0.14, "WEIRD", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.text(3.5, -0.14, "NORMAL", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(y_min - 0.23 * y_span, y_max + 0.42 * y_span)


def write_report(
    overall_weird: dict[str, float],
    overall_normal: dict[str, float],
    overall_interaction: dict[str, float],
    analytical_weird: dict[str, float],
    analytical_normal: dict[str, float],
    analytical_interaction: dict[str, float],
    emotional_weird: dict[str, float],
    emotional_normal: dict[str, float],
    emotional_interaction: dict[str, float],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 12 Results}")
    lines.append(
        "Hypothesis 12 tested whether WEIRD and NORMAL participants differ in pre-to-post trust change using Group x Time interaction models (difference-in-differences) for overall, analytical, and emotional trust."
    )
    lines.append("")

    lines.append("12.1 Group x Time interaction for overall trust:")
    lines.append(
        f"WEIRD pre-post: $n={int(overall_weird['n'])}$, $\\Delta(M_{{post}}-M_{{pre}})={overall_weird['delta_post_minus_pre']:.3f}$, "
        f"$t={overall_weird['t']:.3f}$, $p {format_p(overall_weird['p'])}$."
    )
    lines.append(
        f"NORMAL pre-post: $n={int(overall_normal['n'])}$, $\\Delta(M_{{post}}-M_{{pre}})={overall_normal['delta_post_minus_pre']:.3f}$, "
        f"$t={overall_normal['t']:.3f}$, $p {format_p(overall_normal['p'])}$."
    )
    lines.append(
        f"Interaction estimate: $b={overall_interaction['coef']:.3f}$, $t({int(overall_interaction['df'])})={overall_interaction['t']:.3f}$, $p {format_p(overall_interaction['p'])}$."
    )

    lines.append("")
    lines.append("12.2 Group x Time interaction for analytical trust:")
    lines.append(
        f"WEIRD pre-post: $n={int(analytical_weird['n'])}$, $\\Delta(M_{{post}}-M_{{pre}})={analytical_weird['delta_post_minus_pre']:.3f}$, "
        f"$t={analytical_weird['t']:.3f}$, $p {format_p(analytical_weird['p'])}$."
    )
    lines.append(
        f"NORMAL pre-post: $n={int(analytical_normal['n'])}$, $\\Delta(M_{{post}}-M_{{pre}})={analytical_normal['delta_post_minus_pre']:.3f}$, "
        f"$t={analytical_normal['t']:.3f}$, $p {format_p(analytical_normal['p'])}$."
    )
    lines.append(
        f"Interaction estimate: $b={analytical_interaction['coef']:.3f}$, $t({int(analytical_interaction['df'])})={analytical_interaction['t']:.3f}$, $p {format_p(analytical_interaction['p'])}$."
    )

    lines.append("")
    lines.append("12.3 Group x Time interaction for emotional trust:")
    lines.append(
        f"WEIRD pre-post: $n={int(emotional_weird['n'])}$, $\\Delta(M_{{post}}-M_{{pre}})={emotional_weird['delta_post_minus_pre']:.3f}$, "
        f"$t={emotional_weird['t']:.3f}$, $p {format_p(emotional_weird['p'])}$."
    )
    lines.append(
        f"NORMAL pre-post: $n={int(emotional_normal['n'])}$, $\\Delta(M_{{post}}-M_{{pre}})={emotional_normal['delta_post_minus_pre']:.3f}$, "
        f"$t={emotional_normal['t']:.3f}$, $p {format_p(emotional_normal['p'])}$."
    )
    lines.append(
        f"Interaction estimate: $b={emotional_interaction['coef']:.3f}$, $t({int(emotional_interaction['df'])})={emotional_interaction['t']:.3f}$, $p {format_p(emotional_interaction['p'])}$."
    )

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    overall_weird = paired_prepost_test(analysis, "WEIRD", "overall_pre", "overall_post")
    overall_normal = paired_prepost_test(analysis, "NORMAL", "overall_pre", "overall_post")
    overall_interaction = did_interaction_test(analysis, "overall_pre", "overall_post")

    analytical_weird = paired_prepost_test(analysis, "WEIRD", "analytical_pre", "analytical_post")
    analytical_normal = paired_prepost_test(analysis, "NORMAL", "analytical_pre", "analytical_post")
    analytical_interaction = did_interaction_test(analysis, "analytical_pre", "analytical_post")

    emotional_weird = paired_prepost_test(analysis, "WEIRD", "emotional_pre", "emotional_post")
    emotional_normal = paired_prepost_test(analysis, "NORMAL", "emotional_pre", "emotional_post")
    emotional_interaction = did_interaction_test(analysis, "emotional_pre", "emotional_post")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(26, 9), constrained_layout=True)

    draw_did_subplot(
        axes[0],
        overall_weird,
        overall_normal,
        overall_interaction,
        title="12.1 Overall Trust: WEIRD vs NORMAL Pre/Post",
        ylabel="Overall Trust Score",
    )
    draw_did_subplot(
        axes[1],
        analytical_weird,
        analytical_normal,
        analytical_interaction,
        title="12.2 Analytical Trust: WEIRD vs NORMAL Pre/Post",
        ylabel="Analytical Trust Score",
    )
    draw_did_subplot(
        axes[2],
        emotional_weird,
        emotional_normal,
        emotional_interaction,
        title="12.3 Emotional Trust: WEIRD vs NORMAL Pre/Post",
        ylabel="Emotional Trust Score",
    )

    axes[0].text(
        0.01,
        0.90,
        "Brackets: within-group pre vs post.\nText box includes Group x Time interaction.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
    )

    fig.suptitle("Hypothesis 12: Difference-in-Differences (WEIRD vs NORMAL, Pre vs Post)", fontsize=21)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=160)
    plt.close(fig)

    report_text = write_report(
        overall_weird=overall_weird,
        overall_normal=overall_normal,
        overall_interaction=overall_interaction,
        analytical_weird=analytical_weird,
        analytical_normal=analytical_normal,
        analytical_interaction=analytical_interaction,
        emotional_weird=emotional_weird,
        emotional_normal=emotional_normal,
        emotional_interaction=emotional_interaction,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
