from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, t, ttest_ind


ALPHA = 0.05

METRIC_SPECS = [
    ("analytical_change", "Analytical Change (Post - Pre, per-item)", "#377eb8"),
    ("emotional_change", "Emotional Change (Post - Pre, per-item)", "#c26a26"),
    (
        "analytical_vs_emotional_diff",
        "Analytical - Emotional Change Difference",
        "#4daf4a",
    ),
]

CONDITION_ORDER = ["Interactive", "Text"]


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
        series.astype("string")
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


def coalesce_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in columns:
        if col not in df.columns:
            continue
        candidate = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        candidate = candidate.mask(candidate.eq(""), pd.NA)
        out = out.fillna(candidate)
    return out


def coalesce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        out = out.fillna(candidate)
    return out


def collapse_top_categories(
    series: pd.Series,
    max_groups: int = 8,
    min_count: int = 10,
    other_label: str = "Other",
) -> pd.Series:
    clean = series.astype("string").str.strip()
    clean = clean.mask(clean.eq(""), pd.NA)
    counts = clean.value_counts(dropna=True)

    keep = counts[counts >= min_count].index.tolist()
    if len(keep) > max_groups:
        keep = keep[:max_groups]

    collapsed = clean.where(clean.isin(keep), other_label)
    if (collapsed == other_label).sum() == 0:
        return clean
    return collapsed


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


def build_change_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns for trust-change analysis: {missing}")

    work = df.copy()
    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["analytical_vs_emotional_diff"] = work["analytical_change"] - work["emotional_change"]
    return work


def _as_float_series(values: Iterable[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def metric_comparisons(
    frame: pd.DataFrame,
    group_col: str,
    metric_col: str,
    group_order: list[str],
    min_group_n_for_tests: int,
) -> dict[str, object]:
    summary_rows: list[dict[str, float | str | int]] = []
    test_samples: dict[str, pd.Series] = {}

    for group in group_order:
        values = frame.loc[frame[group_col] == group, metric_col].dropna().astype(float)
        mean_val, ci_low, ci_high = mean_ci(values)
        summary_rows.append(
            {
                "Group": group,
                "n": int(len(values)),
                "mean": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        if len(values) >= max(min_group_n_for_tests, 2):
            test_samples[group] = values

    summary = pd.DataFrame(summary_rows)

    anova_f = np.nan
    anova_p = np.nan
    if len(test_samples) >= 2:
        anova_f, anova_p = f_oneway(*[vals for vals in test_samples.values()])

    welch_t = np.nan
    welch_p = np.nan
    welch_groups: tuple[str, str] | None = None
    if len(test_samples) == 2:
        group_a, group_b = list(test_samples.keys())
        welch_t, welch_p = ttest_ind(
            test_samples[group_a],
            test_samples[group_b],
            equal_var=False,
            nan_policy="omit",
        )
        welch_groups = (group_a, group_b)

    pairwise_rows: list[dict[str, float | str]] = []
    combos = list(combinations(test_samples.keys(), 2))
    if combos:
        for group_a, group_b in combos:
            t_stat, p_val = ttest_ind(
                test_samples[group_a],
                test_samples[group_b],
                equal_var=False,
                nan_policy="omit",
            )
            p_adj = min(float(p_val) * len(combos), 1.0)
            pairwise_rows.append(
                {
                    "Group A": group_a,
                    "Group B": group_b,
                    "t": float(t_stat),
                    "p": float(p_val),
                    "p_adj": p_adj,
                    "stars": significance_stars(p_adj),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)

    return {
        "summary": summary,
        "anova_f": float(anova_f) if pd.notna(anova_f) else np.nan,
        "anova_p": float(anova_p) if pd.notna(anova_p) else np.nan,
        "welch_t": float(welch_t) if pd.notna(welch_t) else np.nan,
        "welch_p": float(welch_p) if pd.notna(welch_p) else np.nan,
        "welch_groups": welch_groups,
        "pairwise": pairwise_df,
    }


def condition_effect_tests(
    frame: pd.DataFrame,
    group_col: str,
    metric_col: str,
    group_order: list[str],
    min_n_per_condition: int,
) -> dict[str, object]:
    def _single_test(values: pd.DataFrame) -> dict[str, float]:
        interactive_vals = (
            values.loc[values["Condition"] == "Interactive", metric_col].dropna().astype(float)
        )
        text_vals = values.loc[values["Condition"] == "Text", metric_col].dropna().astype(float)

        mean_i, ci_i_low, ci_i_high = mean_ci(interactive_vals)
        mean_t, ci_t_low, ci_t_high = mean_ci(text_vals)

        if len(interactive_vals) >= min_n_per_condition and len(text_vals) >= min_n_per_condition:
            t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")
        else:
            t_stat, p_value = np.nan, np.nan

        return {
            "n_interactive": float(len(interactive_vals)),
            "n_text": float(len(text_vals)),
            "mean_interactive": mean_i,
            "mean_text": mean_t,
            "ci_low_interactive": ci_i_low,
            "ci_high_interactive": ci_i_high,
            "ci_low_text": ci_t_low,
            "ci_high_text": ci_t_high,
            "t": float(t_stat) if pd.notna(t_stat) else np.nan,
            "p": float(p_value) if pd.notna(p_value) else np.nan,
            "delta_i_minus_t": mean_i - mean_t,
        }

    overall = _single_test(frame)

    by_group_rows: list[dict[str, float | str]] = []
    for group_name in group_order:
        subset = frame.loc[frame[group_col] == group_name].copy()
        if subset.empty:
            continue
        stats_row = _single_test(subset)
        stats_row["Group"] = group_name
        by_group_rows.append(stats_row)

    by_group = pd.DataFrame(by_group_rows)

    return {
        "overall": overall,
        "by_group": by_group,
    }


def draw_metric_subplot(
    ax: plt.Axes,
    metric_label: str,
    metric_color: str,
    metric_result: dict[str, object],
    grouping_label: str,
) -> None:
    summary = metric_result["summary"]
    x_positions = np.arange(len(summary), dtype=float)
    means = summary["mean"].astype(float).to_numpy()
    ci_lows = summary["ci_low"].astype(float).to_numpy()
    ci_highs = summary["ci_high"].astype(float).to_numpy()

    err_low = np.nan_to_num(means - ci_lows, nan=0.0)
    err_high = np.nan_to_num(ci_highs - means, nan=0.0)

    ax.bar(
        x_positions,
        means,
        color=metric_color,
        edgecolor="none",
        width=0.75,
        zorder=2,
    )
    ax.errorbar(
        x_positions,
        means,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=2.5,
        capsize=7,
        capthick=2.5,
        zorder=3,
    )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.3, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["Group"].tolist(), rotation=28, ha="right", fontsize=10)
    ax.set_title(metric_label, fontsize=15)
    ax.set_xlabel(grouping_label, fontsize=12)
    ax.set_ylabel("Mean Change", fontsize=12)

    y_min = float(np.nanmin(ci_lows)) if len(ci_lows) else -0.1
    y_max = float(np.nanmax(ci_highs)) if len(ci_highs) else 0.1
    y_span = max(y_max - y_min, 0.2)

    ax.set_ylim(y_min - 0.16 * y_span, y_max + 0.26 * y_span)

    for idx, n_val in enumerate(summary["n"].tolist()):
        ax.text(
            x_positions[idx],
            y_min - 0.08 * y_span,
            f"n={int(n_val)}",
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )

    annotation = f"ANOVA: F={metric_result['anova_f']:.2f}, p {format_p(metric_result['anova_p'])}"
    if pd.notna(metric_result["welch_p"]):
        group_a, group_b = metric_result["welch_groups"]
        annotation += (
            "\n"
            f"Welch ({group_a} vs {group_b}): t={metric_result['welch_t']:.2f}, "
            f"p {format_p(metric_result['welch_p'])}"
        )

    ax.text(
        0.01,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def write_report_text(
    analysis_title: str,
    grouping_label: str,
    group_counts: pd.Series,
    metric_results: dict[str, dict[str, object]],
    condition_results: dict[str, dict[str, object]] | None,
    output_txt: Path,
    min_group_n_for_tests: int,
    min_n_per_condition: int,
) -> None:
    lines: list[str] = []
    lines.append(f"\\subsubsection{{{analysis_title} Results}}")
    lines.append(
        f"Compared analytical trust change, emotional trust change, and their difference across {grouping_label} categories."
    )
    lines.append("")
    lines.append("Included groups and counts:")
    for group_name, n in group_counts.items():
        lines.append(f"- {group_name}: n={int(n)}")
    lines.append("")

    for metric_col, metric_label, _ in METRIC_SPECS:
        result = metric_results[metric_col]
        lines.append(
            f"{metric_label}: ANOVA across groups with n >= {min_group_n_for_tests}: "
            f"F={result['anova_f']:.3f}, p {format_p(result['anova_p'])}."
        )

        summary = result["summary"]
        lines.append("Group means (95% CI):")
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['Group']}: n={int(row['n'])}, "
                f"M={row['mean']:.3f}, "
                f"CI=[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            )

        pairwise = result["pairwise"]
        if pairwise.empty:
            lines.append("Pairwise Welch tests (Bonferroni-adjusted): not available.")
        else:
            lines.append("Pairwise Welch tests (Bonferroni-adjusted):")
            for _, row in pairwise.iterrows():
                star_text = f" {row['stars']}" if row["stars"] else ""
                lines.append(
                    f"- {row['Group A']} vs {row['Group B']}: "
                    f"t={row['t']:.3f}, p={row['p']:.3f}, p_adj={row['p_adj']:.3f}{star_text}"
                )

        lines.append("")

    if condition_results is not None:
        lines.append("Condition-effect follow-up tests (Interactive vs Text):")
        lines.append(
            "These tests are additive to the trust-change group comparisons above and evaluate condition effects overall and within each group category."
        )
        lines.append("")

        for metric_col, metric_label, _ in METRIC_SPECS:
            result = condition_results[metric_col]
            overall = result["overall"]
            lines.append(
                f"{metric_label} overall condition effect "
                f"(min n per condition = {min_n_per_condition}): "
                f"$n_{{Interactive}}={int(overall['n_interactive'])}$, "
                f"$n_{{Text}}={int(overall['n_text'])}$, "
                f"$M_{{Interactive}}={overall['mean_interactive']:.3f}$, "
                f"$M_{{Text}}={overall['mean_text']:.3f}$, "
                f"$t={overall['t']:.3f}$, "
                f"$p {format_p(overall['p'])}$, "
                f"$\\Delta(M_I-M_T)={overall['delta_i_minus_t']:.3f}$."
            )

            by_group = result["by_group"]
            if by_group.empty:
                lines.append("Within-group condition tests: not available.")
            else:
                lines.append("Within-group condition tests:")
                for _, row in by_group.iterrows():
                    lines.append(
                        f"- {row['Group']}: "
                        f"$n_{{Interactive}}={int(row['n_interactive'])}$, "
                        f"$n_{{Text}}={int(row['n_text'])}$, "
                        f"$M_{{Interactive}}={row['mean_interactive']:.3f}$, "
                        f"$M_{{Text}}={row['mean_text']:.3f}$, "
                        f"$t={row['t']:.3f}$, "
                        f"$p {format_p(row['p'])}$, "
                        f"$\\Delta(M_I-M_T)={row['delta_i_minus_t']:.3f}$."
                    )
            lines.append("")

    output_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def significance_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    stars = significance_stars(p_value)
    return stars if stars else "ns"


def condition_effect_by_group(
    frame: pd.DataFrame,
    group_col: str,
    metric_col: str,
    group_order: list[str],
    min_n_per_condition: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for group_name in group_order:
        subset = frame.loc[frame[group_col] == group_name].copy()
        if subset.empty:
            continue

        interactive_vals = (
            subset.loc[subset["Condition"] == "Interactive", metric_col].dropna().astype(float)
        )
        text_vals = subset.loc[subset["Condition"] == "Text", metric_col].dropna().astype(float)

        m_i, l_i, h_i = mean_ci(interactive_vals)
        m_t, l_t, h_t = mean_ci(text_vals)

        if len(interactive_vals) >= min_n_per_condition and len(text_vals) >= min_n_per_condition:
            t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")
        else:
            t_stat, p_value = np.nan, np.nan

        rows.append(
            {
                "Group": group_name,
                "n_interactive": float(len(interactive_vals)),
                "n_text": float(len(text_vals)),
                "mean_interactive": m_i,
                "mean_text": m_t,
                "ci_low_interactive": l_i,
                "ci_high_interactive": h_i,
                "ci_low_text": l_t,
                "ci_high_text": h_t,
                "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                "p": float(p_value) if pd.notna(p_value) else np.nan,
                "delta_i_minus_t": m_i - m_t,
            }
        )

    return pd.DataFrame(rows)


def _add_pair_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    y_step: float,
) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=1.8, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.5 * y_step, label, ha="center", va="bottom", fontsize=10)


def draw_condition_split_subplot(
    ax: plt.Axes,
    metric_label: str,
    metric_result: pd.DataFrame,
    grouping_label: str,
) -> None:
    if metric_result.empty:
        ax.set_title(metric_label)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    groups = metric_result["Group"].astype(str).tolist()
    x_center = np.arange(len(groups), dtype=float)
    bar_width = 0.36
    x_i = x_center - bar_width / 2
    x_t = x_center + bar_width / 2

    m_i = metric_result["mean_interactive"].astype(float).to_numpy()
    m_t = metric_result["mean_text"].astype(float).to_numpy()
    l_i = metric_result["ci_low_interactive"].astype(float).to_numpy()
    h_i = metric_result["ci_high_interactive"].astype(float).to_numpy()
    l_t = metric_result["ci_low_text"].astype(float).to_numpy()
    h_t = metric_result["ci_high_text"].astype(float).to_numpy()

    ax.bar(x_i, m_i, width=bar_width, color="#377eb8", edgecolor="none", label="Interactive", zorder=2)
    ax.bar(x_t, m_t, width=bar_width, color="#c26a26", edgecolor="none", label="Static (Text)", zorder=2)

    ax.errorbar(
        x_i,
        m_i,
        yerr=[np.nan_to_num(m_i - l_i, nan=0.0), np.nan_to_num(h_i - m_i, nan=0.0)],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=2.2,
        capsize=6,
        capthick=2.2,
        zorder=3,
    )
    ax.errorbar(
        x_t,
        m_t,
        yerr=[np.nan_to_num(m_t - l_t, nan=0.0), np.nan_to_num(h_t - m_t, nan=0.0)],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=2.2,
        capsize=6,
        capthick=2.2,
        zorder=3,
    )

    y_candidates = np.concatenate([l_i, h_i, l_t, h_t])
    y_candidates = y_candidates[np.isfinite(y_candidates)]
    if y_candidates.size == 0:
        y_min, y_max = -0.1, 0.1
    else:
        y_min = float(np.nanmin(y_candidates))
        y_max = float(np.nanmax(y_candidates))
    y_span = max(y_max - y_min, 0.2)

    for idx, row in metric_result.iterrows():
        bracket_y = max(
            float(row["ci_high_interactive"]) if pd.notna(row["ci_high_interactive"]) else y_max,
            float(row["ci_high_text"]) if pd.notna(row["ci_high_text"]) else y_max,
        ) + 0.06 * y_span
        _add_pair_bracket(
            ax,
            x_i[idx],
            x_t[idx],
            bracket_y,
            significance_label(float(row["p"])),
            0.02 * y_span,
        )

        ax.text(
            x_i[idx],
            y_min - 0.10 * y_span,
            f"n={int(row['n_interactive'])}",
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )
        ax.text(
            x_t[idx],
            y_min - 0.10 * y_span,
            f"n={int(row['n_text'])}",
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )

    ax.axhline(0.0, color="#4a4a4a", linewidth=1.3, zorder=1)
    ax.set_xticks(x_center)
    ax.set_xticklabels(groups, rotation=28, ha="right", fontsize=10)
    ax.set_title(metric_label, fontsize=14)
    ax.set_xlabel(grouping_label, fontsize=11)
    ax.set_ylabel("Mean Change", fontsize=11)
    ax.set_ylim(y_min - 0.18 * y_span, y_max + 0.32 * y_span)
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def write_condition_report_text(
    analysis_title: str,
    grouping_label: str,
    condition_results: dict[str, pd.DataFrame],
    output_txt: Path,
    min_n_per_condition: int,
) -> None:
    lines: list[str] = []
    lines.append(f"\\subsubsection{{{analysis_title} Condition-Effect Results}}")
    lines.append(
        f"Condition-effect tests (Interactive vs Text) were run within each {grouping_label} category for analytical change, emotional change, and analytical-minus-emotional difference."
    )
    lines.append("")

    for metric_col, metric_label, _ in METRIC_SPECS:
        lines.append(f"{metric_label} (min n per condition = {min_n_per_condition}):")
        result = condition_results[metric_col]
        if result.empty:
            lines.append("- No valid groups available.")
            lines.append("")
            continue

        for _, row in result.iterrows():
            lines.append(
                f"- {row['Group']}: "
                f"$n_{{Interactive}}={int(row['n_interactive'])}$, "
                f"$n_{{Text}}={int(row['n_text'])}$, "
                f"$M_{{Interactive}}={row['mean_interactive']:.3f}$, "
                f"$M_{{Text}}={row['mean_text']:.3f}$, "
                f"$t={row['t']:.3f}$, "
                f"$p {format_p(row['p'])}$, "
                f"$\\Delta(M_I-M_T)={row['delta_i_minus_t']:.3f}$, "
                f"{significance_label(float(row['p']))}."
            )
        lines.append("")

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001; ns = non-significant.")
    output_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_grouping_condition_effect_analysis(
    df: pd.DataFrame,
    group_series: pd.Series,
    grouping_label: str,
    analysis_title: str,
    output_png: Path,
    output_txt: Path,
    preferred_order: list[str] | None = None,
    min_n_per_condition: int = 8,
) -> None:
    frame = build_change_frame(df)

    group = pd.Series(group_series, index=frame.index, dtype="string")
    group = group.str.replace(r"\s+", " ", regex=True).str.strip()
    group = group.mask(group.eq(""), pd.NA)

    frame["Grouping"] = group
    frame = frame.dropna(subset=["Grouping"]).copy()

    if "Condition" not in frame.columns:
        raise KeyError("Missing required column for condition-effect analysis: ['Condition']")

    frame["Condition"] = normalize_condition(frame["Condition"])
    frame = frame[frame["Condition"].isin(CONDITION_ORDER)].copy()

    if frame.empty:
        raise ValueError(f"No usable rows after grouping by {grouping_label} with valid condition labels.")

    counts = frame["Grouping"].value_counts()
    if preferred_order:
        ordered = [g for g in preferred_order if g in counts.index]
        ordered.extend([g for g in counts.index.tolist() if g not in ordered])
    else:
        ordered = counts.index.tolist()

    condition_results: dict[str, pd.DataFrame] = {}
    for metric_col, _, _ in METRIC_SPECS:
        condition_results[metric_col] = condition_effect_by_group(
            frame,
            group_col="Grouping",
            metric_col=metric_col,
            group_order=ordered,
            min_n_per_condition=min_n_per_condition,
        )

    fig_width = max(19.0, 6.5 + 0.9 * len(ordered))
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 7.1), constrained_layout=True)

    for ax, (metric_col, metric_label, _) in zip(axes, METRIC_SPECS):
        draw_condition_split_subplot(
            ax,
            metric_label=metric_label,
            metric_result=condition_results[metric_col],
            grouping_label=grouping_label,
        )

    axes[0].text(
        0.01,
        0.98,
        "Per-group brackets test Interactive vs Static (Text).\nAsterisks: * p < .05, ** p < .01, *** p < .001; ns = non-significant.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    fig.suptitle(f"{analysis_title}: Condition Effects By Group", fontsize=22)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    write_condition_report_text(
        analysis_title=analysis_title,
        grouping_label=grouping_label,
        condition_results=condition_results,
        output_txt=output_txt,
        min_n_per_condition=min_n_per_condition,
    )

    print(f"Saved condition report: {output_txt}")
    print(f"Saved condition figure: {output_png}")


def run_grouping_analysis(
    df: pd.DataFrame,
    group_series: pd.Series,
    grouping_label: str,
    analysis_title: str,
    output_png: Path,
    output_txt: Path,
    preferred_order: list[str] | None = None,
    min_group_n_for_tests: int = 10,
    include_condition_effect_tests: bool = False,
    min_n_per_condition: int = 8,
) -> None:
    frame = build_change_frame(df)

    group = pd.Series(group_series, index=frame.index, dtype="string")
    group = group.str.replace(r"\s+", " ", regex=True).str.strip()
    group = group.mask(group.eq(""), pd.NA)

    frame["Grouping"] = group
    frame = frame.dropna(subset=["Grouping"]).copy()

    if "Condition" in frame.columns:
        frame["Condition"] = normalize_condition(frame["Condition"])
        frame = frame[frame["Condition"].isin(CONDITION_ORDER)].copy()

    if frame.empty:
        raise ValueError(f"No usable rows after grouping by {grouping_label}.")

    counts = frame["Grouping"].value_counts()

    if preferred_order:
        ordered = [g for g in preferred_order if g in counts.index]
        ordered.extend([g for g in counts.index.tolist() if g not in ordered])
    else:
        ordered = counts.index.tolist()

    metric_results: dict[str, dict[str, object]] = {}
    for metric_col, _, _ in METRIC_SPECS:
        metric_results[metric_col] = metric_comparisons(
            frame,
            group_col="Grouping",
            metric_col=metric_col,
            group_order=ordered,
            min_group_n_for_tests=min_group_n_for_tests,
        )

    condition_results: dict[str, dict[str, object]] | None = None
    if include_condition_effect_tests:
        condition_results = {}
        for metric_col, _, _ in METRIC_SPECS:
            condition_results[metric_col] = condition_effect_tests(
                frame,
                group_col="Grouping",
                metric_col=metric_col,
                group_order=ordered,
                min_n_per_condition=min_n_per_condition,
            )

    fig_width = max(19.0, 6.5 + 0.8 * len(ordered))
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 6.8), constrained_layout=True)

    for ax, (metric_col, metric_label, metric_color) in zip(axes, METRIC_SPECS):
        draw_metric_subplot(
            ax,
            metric_label=metric_label,
            metric_color=metric_color,
            metric_result=metric_results[metric_col],
            grouping_label=grouping_label,
        )

    fig.suptitle(f"{analysis_title}: Trust-Change Comparisons", fontsize=22)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    write_report_text(
        analysis_title=analysis_title,
        grouping_label=grouping_label,
        group_counts=counts,
        metric_results=metric_results,
        condition_results=condition_results,
        output_txt=output_txt,
        min_group_n_for_tests=min_group_n_for_tests,
        min_n_per_condition=min_n_per_condition,
    )

    print(f"Saved report: {output_txt}")
    print(f"Saved figure: {output_png}")
