from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis9.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis9.txt"
ALPHA = 0.05

CONDITION_ORDER = ["Interactive", "Text"]
CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static (Text)",
}
CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}
GROUP_ORDER = ["WEIRD", "NORMAL"]

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


def format_stat(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


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

    language = work["Language"].astype(str).str.strip().str.lower()
    residence = work["Country of residence"].astype(str).str.strip()
    work["Group"] = np.where(
        residence.isin(WESTERN_COUNTRIES) & language.str.startswith("english"),
        "WEIRD",
        "NORMAL",
    )

    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["analytical_minus_emotional_change"] = work["analytical_change"] - work["emotional_change"]

    work["condition_bin"] = (work["Condition"] == "Interactive").astype(float)
    work["group_bin"] = (work["Group"] == "WEIRD").astype(float)

    return work


def condition_welch_test(data: pd.DataFrame, group: str) -> dict[str, float | str]:
    subset = data[data["Group"] == group].copy()

    interactive_vals = (
        subset.loc[subset["Condition"] == "Interactive", "analytical_minus_emotional_change"]
        .dropna()
        .astype(float)
    )
    text_vals = (
        subset.loc[subset["Condition"] == "Text", "analytical_minus_emotional_change"]
        .dropna()
        .astype(float)
    )

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


def interaction_effect(data: pd.DataFrame) -> dict[str, float]:
    fit_df = data[["analytical_minus_emotional_change", "condition_bin", "group_bin"]].dropna().copy()

    y = fit_df["analytical_minus_emotional_change"].to_numpy(dtype=float)
    condition = fit_df["condition_bin"].to_numpy(dtype=float)
    group = fit_df["group_bin"].to_numpy(dtype=float)
    interaction = condition * group

    X = np.column_stack([np.ones(len(fit_df)), condition, group, interaction])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    n, p = X.shape
    dof = n - p
    if dof <= 0:
        return {"coef": np.nan, "t": np.nan, "df": np.nan, "p": np.nan}

    rss = float(np.sum(residuals**2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
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


def build_long_form_frame(data: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["Condition", "Group", "condition_bin", "group_bin", "analytical_change", "emotional_change"]
    base = data[base_cols].copy().reset_index(drop=True)
    base["participant_id"] = np.arange(len(base), dtype=int)

    analytic = base[["participant_id", "Condition", "Group", "condition_bin", "group_bin"]].copy()
    analytic["trust_type"] = "Analytical"
    analytic["trust_type_bin"] = 0.0
    analytic["change_score"] = base["analytical_change"].astype(float)

    emotional = base[["participant_id", "Condition", "Group", "condition_bin", "group_bin"]].copy()
    emotional["trust_type"] = "Emotional"
    emotional["trust_type_bin"] = 1.0
    emotional["change_score"] = base["emotional_change"].astype(float)

    long_df = pd.concat([analytic, emotional], ignore_index=True)
    return long_df.dropna(subset=["change_score", "condition_bin", "group_bin", "trust_type_bin"]).copy()


def long_form_three_way_effect(data: pd.DataFrame) -> dict[str, float]:
    fit_df = build_long_form_frame(data)
    if fit_df.empty:
        return {
            "coef": np.nan,
            "t": np.nan,
            "df": np.nan,
            "p": np.nan,
            "n_obs": 0.0,
            "n_participants": 0.0,
        }

    y = fit_df["change_score"].to_numpy(dtype=float)
    condition = fit_df["condition_bin"].to_numpy(dtype=float)
    group = fit_df["group_bin"].to_numpy(dtype=float)
    trust_type = fit_df["trust_type_bin"].to_numpy(dtype=float)

    c_by_g = condition * group
    c_by_t = condition * trust_type
    g_by_t = group * trust_type
    c_by_g_by_t = condition * group * trust_type

    X = np.column_stack(
        [
            np.ones(len(fit_df)),
            condition,
            group,
            trust_type,
            c_by_g,
            c_by_t,
            g_by_t,
            c_by_g_by_t,
        ]
    )
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
            "n_obs": float(n),
            "n_participants": float(fit_df["participant_id"].nunique()),
        }

    rss = float(np.sum(residuals**2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    coef = float(beta[7])
    t_stat = coef / float(se[7]) if not np.isclose(float(se[7]), 0.0) else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if pd.notna(t_stat) else np.nan

    return {
        "coef": coef,
        "t": float(t_stat),
        "df": float(dof),
        "p": float(p_value),
        "n_obs": float(n),
        "n_participants": float(fit_df["participant_id"].nunique()),
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


def draw_single_group_condition_subplot(
    ax: plt.Axes,
    test_result: dict[str, float | str],
    title: str,
) -> None:
    x = np.array([0.0, 1.0], dtype=float)
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
        significance_label(float(test_result["p"])),
        0.03 * y_span,
    )

    summary = (
        f"t={format_stat(float(test_result['t']))}, p {format_p(float(test_result['p']))}\n"
        f"Delta(I-T)={format_stat(float(test_result['delta_interactive_minus_text']))}"
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

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=11)
    ax.set_ylabel("Analytical - Emotional Change")
    ax.set_title(title)
    ax.set_ylim(y_min - 0.20 * y_span, y_max + 0.40 * y_span)


def draw_group_split_subplot(
    ax: plt.Axes,
    weird_test: dict[str, float | str],
    normal_test: dict[str, float | str],
    interaction: dict[str, float],
    long_form_model: dict[str, float],
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

    add_pair_bracket(ax, x[0], x[1], bracket_y, significance_label(float(weird_test["p"])), step)
    add_pair_bracket(ax, x[2], x[3], bracket_y, significance_label(float(normal_test["p"])), step)

    summary = (
        f"Group x Condition interaction: b={format_stat(interaction['coef'])}, "
        f"t={format_stat(interaction['t'])}, p {format_p(interaction['p'])}\n"
        f"Long-form C x G x Type: b={format_stat(long_form_model['coef'])}, "
        f"t={format_stat(long_form_model['t'])}, p {format_p(long_form_model['p'])}"
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
        fontsize=10,
    )
    ax.text(0.5, -0.14, "WEIRD", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.text(3.5, -0.14, "NORMAL", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    ax.set_ylabel("Analytical - Emotional Change")
    ax.set_title("9.3 Grouped Condition Contrast + Long-Form Interaction")
    ax.set_ylim(y_min - 0.23 * y_span, y_max + 0.40 * y_span)


def report_group_line(test_result: dict[str, float | str]) -> str:
    return (
        f"- {test_result['group']}: "
        f"$n_{{Interactive}}={int(float(test_result['n_interactive']))}$, "
        f"$n_{{Text}}={int(float(test_result['n_text']))}$, "
        f"$M_{{Interactive}}={float(test_result['mean_interactive']):.3f}$, "
        f"$M_{{Text}}={float(test_result['mean_text']):.3f}$, "
        f"$t={format_stat(float(test_result['t']))}$, "
        f"$p {format_p(float(test_result['p']))}$, "
        f"$\\Delta(M_I-M_T)={float(test_result['delta_interactive_minus_text']):.3f}$."
    )


def write_report(
    weird_test: dict[str, float | str],
    normal_test: dict[str, float | str],
    interaction: dict[str, float],
    long_form_model: dict[str, float],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 9 Results}")
    lines.append(
        "Hypothesis 9 tested whether the condition effect on trust-change magnitude depends on the interaction between WEIRD vs NORMAL grouping and trust type (analytical vs emotional), operationalized as analytical-minus-emotional change."
    )
    lines.append("")

    lines.append("9.1 Condition effect on analytical-minus-emotional change within WEIRD participants:")
    lines.append(report_group_line(weird_test))
    if pd.notna(float(weird_test["p"])) and float(weird_test["p"]) < ALPHA:
        lines.append("Interpretation: significant condition effect within WEIRD participants.")
    else:
        lines.append("Interpretation: no significant condition effect within WEIRD participants.")

    lines.append("")
    lines.append("9.2 Condition effect on analytical-minus-emotional change within NORMAL participants:")
    lines.append(report_group_line(normal_test))
    if pd.notna(float(normal_test["p"])) and float(normal_test["p"]) < ALPHA:
        lines.append("Interpretation: significant condition effect within NORMAL participants.")
    else:
        lines.append("Interpretation: no significant condition effect within NORMAL participants.")

    lines.append("")
    lines.append("9.3 Group x Condition interaction on analytical-minus-emotional change:")
    lines.append(
        "Interaction estimate: "
        f"$b={format_stat(interaction['coef'])}$, "
        f"$t({int(interaction['df'])})={format_stat(interaction['t'])}$, "
        f"$p {format_p(interaction['p'])}$."
    )
    if pd.notna(interaction["p"]) and interaction["p"] < ALPHA:
        lines.append("Interpretation: the condition effect differs significantly between WEIRD and NORMAL participants.")
    else:
        lines.append("Interpretation: no significant evidence that the condition effect differs between WEIRD and NORMAL participants.")

    lines.append("")
    lines.append("9.4 Long-form Condition x Group x Trust-Type model on trust change:")
    lines.append(
        "Three-way interaction estimate (Condition x Group x Trust Type): "
        f"$b={format_stat(long_form_model['coef'])}$, "
        f"$t({int(long_form_model['df'])})={format_stat(long_form_model['t'])}$, "
        f"$p {format_p(long_form_model['p'])}$, "
        f"$n_{{obs}}={int(long_form_model['n_obs'])}$, "
        f"$n_{{participants}}={int(long_form_model['n_participants'])}$.")
    if pd.notna(long_form_model["p"]) and long_form_model["p"] < ALPHA:
        lines.append("Interpretation: significant three-way interaction, indicating condition effects differ by both group and trust type.")
    else:
        lines.append("Interpretation: no significant three-way interaction in the long-form model.")

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    weird_test = condition_welch_test(analysis, "WEIRD")
    normal_test = condition_welch_test(analysis, "NORMAL")
    interaction = interaction_effect(analysis)
    long_form_model = long_form_three_way_effect(analysis)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(25, 9), constrained_layout=True)

    draw_single_group_condition_subplot(
        axes[0],
        weird_test,
        title="9.1 WEIRD: Condition Effect On Analytical-Emotional Change",
    )
    draw_single_group_condition_subplot(
        axes[1],
        normal_test,
        title="9.2 NORMAL: Condition Effect On Analytical-Emotional Change",
    )
    draw_group_split_subplot(axes[2], weird_test, normal_test, interaction, long_form_model)

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

    fig.suptitle("Hypothesis 9: Condition Effects On Analytical-Emotional Change By WEIRD/NORMAL", fontsize=22)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=160)
    plt.close(fig)

    report_text = write_report(
        weird_test=weird_test,
        normal_test=normal_test,
        interaction=interaction,
        long_form_model=long_form_model,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
