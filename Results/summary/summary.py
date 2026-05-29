from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "summary.tex"
MEASURES_PATH = REPO_ROOT / "Methodology" / "Measures" / "measures.py"
DEMOGRAPHICS_PATH = REPO_ROOT / "Methodology" / "Participants" / "demographics.py"

GROUPS = [
    ("Interactive", "WEIRD"),
    ("Interactive", "non-WEIRD"),
    ("Text", "WEIRD"),
    ("Text", "non-WEIRD"),
]

PRE_METRICS = [
    ("analytical_pre_01", "Analytical"),
    ("emotional_pre_01", "Emotional"),
    ("overall_pre_01", "Overall"),
]

POST_CHANGE_METRICS = [
    ("analytical_post_01", "Analytical trust post"),
    ("analytical_change_01", "Analytical trust change"),
    ("emotional_post_01", "Emotional trust post"),
    ("emotional_change_01", "Emotional trust change"),
    ("overall_post_01", "Overall trust post"),
    ("overall_change_01", "Overall trust change"),
]

CONDITION_POST_CHANGE_METRICS = [
    ("analytical_post_01", "Analytical post"),
    ("analytical_change_01", "Analytical change"),
    ("emotional_post_01", "Emotional post"),
    ("emotional_change_01", "Emotional change"),
    ("overall_post_01", "Overall post"),
    ("overall_change_01", "Overall change"),
]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measures = load_module(MEASURES_PATH, "summary_measures")
demographics = load_module(DEMOGRAPHICS_PATH, "summary_demographics")


def build_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Condition"] = df["Condition"].astype(str).str.strip()
    df = measures.add_trust_scores(df)
    df["overall_pre_01"] = (df["analytical_pre_01"] + df["emotional_pre_01"]) / 2.0
    df["overall_post_01"] = (df["analytical_post_01"] + df["emotional_post_01"]) / 2.0
    df["overall_change_01"] = df["overall_post_01"] - df["overall_pre_01"]
    df["weird_score"] = df.apply(demographics.score_weird, axis=1)
    df["Culture"] = np.where(df["weird_score"] >= 4, "WEIRD", "non-WEIRD")
    return df


def group_series(df: pd.DataFrame, metric: str, condition: str, culture: str) -> pd.Series:
    mask = (df["Condition"] == condition) & (df["Culture"] == culture)
    return pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()


def culture_series(df: pd.DataFrame, metric: str, culture: str) -> pd.Series:
    mask = df["Culture"] == culture
    return pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()


def condition_series(df: pd.DataFrame, metric: str, condition: str) -> pd.Series:
    mask = df["Condition"] == condition
    return pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()


def format_cell(value: str, higher: bool, lower: bool) -> str:
    if higher:
        value = f"\\textbf{{{value}}}"
    if lower:
        value = f"\\textit{{{value}}}"
    return value


def significance_stars(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.0005:
        return "***"
    if p_value < 0.005:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_p_value(p_value: float) -> str:
    if np.isnan(p_value):
        return "NA"
    if p_value < 0.0001:
        return "< .0001"
    return f"= {p_value:.4f}".replace("0.", ".")


def metric_phrase(label: str) -> str:
    return f"{label.lower()} trust" if label in {"Analytical", "Emotional", "Overall"} else label.lower()


def summarize_metric(series: pd.Series) -> str:
    if len(series) == 0:
        return "--"
    return f"{series.mean():.2f} ({series.std(ddof=1):.2f})"


def pairwise_directions(metric_groups: dict[tuple[str, str], pd.Series], alpha: float = 0.05):
    higher: dict[tuple[str, str], set[tuple[str, str]]] = {group: set() for group in GROUPS}
    lower: dict[tuple[str, str], set[tuple[str, str]]] = {group: set() for group in GROUPS}
    results: list[dict[str, object]] = []

    for idx, left in enumerate(GROUPS):
        for right in GROUPS[idx + 1:]:
            a = metric_groups[left]
            b = metric_groups[right]
            if len(a) < 2 or len(b) < 2:
                continue

            test = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            t_stat = float(test.statistic)
            p_two = float(test.pvalue)
            direction = "ns"
            p_one = p_two / 2.0

            if p_two < alpha and t_stat > 0 and p_one < alpha:
                higher[left].add(right)
                lower[right].add(left)
                direction = f"{left[0]} {left[1]} > {right[0]} {right[1]}"
            elif p_two < alpha and t_stat < 0 and p_one < alpha:
                higher[right].add(left)
                lower[left].add(right)
                direction = f"{right[0]} {right[1]} > {left[0]} {left[1]}"

            results.append(
                {
                    "left": left,
                    "right": right,
                    "mean_left": float(a.mean()),
                    "mean_right": float(b.mean()),
                    "t": t_stat,
                    "p_two": p_two,
                    "p_one": p_one,
                    "direction": direction,
                }
            )

    return higher, lower, results


def within_condition_direction(
    weird: pd.Series,
    non_weird: pd.Series,
    alpha: float = 0.05,
) -> tuple[bool, bool, bool, bool, dict[str, float | str]]:
    result = {
        "mean_weird": float(weird.mean()) if len(weird) else np.nan,
        "mean_non_weird": float(non_weird.mean()) if len(non_weird) else np.nan,
        "t": np.nan,
        "p_two": np.nan,
        "p_one": np.nan,
        "direction": "ns",
    }
    if len(weird) < 2 or len(non_weird) < 2:
        return False, False, False, False, result

    test = stats.ttest_ind(weird, non_weird, equal_var=False, nan_policy="omit")
    t_stat = float(test.statistic)
    p_two = float(test.pvalue)
    p_one = p_two / 2.0
    result.update({"t": t_stat, "p_two": p_two, "p_one": p_one})

    if p_two < alpha and t_stat > 0 and p_one < alpha:
        result["direction"] = "WEIRD > non-WEIRD"
        return True, False, False, True, result
    if p_two < alpha and t_stat < 0 and p_one < alpha:
        result["direction"] = "non-WEIRD > WEIRD"
        return False, True, True, False, result
    return False, False, False, False, result


def render_pre_table(df: pd.DataFrame) -> tuple[str, dict[str, dict[str, object]]]:
    pairwise_by_metric: dict[str, dict[str, object]] = {}
    weird_n = int((df["Culture"] == "WEIRD").sum())
    non_weird_n = int((df["Culture"] == "non-WEIRD").sum())

    lines = [
        r"\centering",
        r"\small",
        r"\textbf{Pre-Experiment Trust By Cultural Group}\\",
        r"\emph{Cells show mean (SD); bold marks the higher significant value, italics the lower one; asterisks denote WEIRD vs. non-WEIRD comparisons.}\\[0.4em]",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        "Metric & WEIRD ($n={}$) & non-WEIRD ($n={}$) ".format(weird_n, non_weird_n) + ("\\" * 2),
        r"\midrule",
    ]

    for metric, label in PRE_METRICS:
        weird = culture_series(df, metric, "WEIRD")
        non_weird = culture_series(df, metric, "non-WEIRD")
        weird_hi, weird_lo, non_hi, non_lo, result = within_condition_direction(weird, non_weird)
        pairwise_by_metric[metric] = result

        weird_cell = format_cell(summarize_metric(weird), higher=weird_hi, lower=weird_lo)
        non_weird_cell = format_cell(summarize_metric(non_weird), higher=non_hi, lower=non_lo)
        stars = significance_stars(float(result["p_one"])) if result["direction"] != "ns" else ""
        metric_label = f"{label}{stars}"
        lines.append(" & ".join([metric_label, weird_cell, non_weird_cell]) + " " + ("\\" * 2))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    return "\n".join(lines) + "\n", pairwise_by_metric


def render_post_change_table(df: pd.DataFrame) -> tuple[str, dict[str, list[dict[str, object]]]]:
    group_ns = {group: len(group_series(df, "overall_pre_01", *group)) for group in GROUPS}
    pairwise_by_metric: dict[str, list[dict[str, object]]] = {}

    lines = [
        r"\centering",
        r"\small",
        r"\textbf{Post And Change By Condition And Cultural Group}\\",
        r"\emph{Cells show mean (SD); bold indicates values significantly higher than at least one comparison group, and italics indicate values significantly lower than at least one comparison group.}\\[0.4em]",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Interactive WEIRD & Interactive non-WEIRD & Text WEIRD & Text non-WEIRD \\",
        r"\midrule",
        "n & {} & {} & {} & {} \\\\".format(*(group_ns[group] for group in GROUPS)),
        r"\midrule",
    ]

    for metric, label in POST_CHANGE_METRICS:
        metric_groups = {group: group_series(df, metric, *group) for group in GROUPS}
        higher, lower, results = pairwise_directions(metric_groups)
        pairwise_by_metric[metric] = results

        cells: list[str] = []
        for group in GROUPS:
            value = summarize_metric(metric_groups[group])
            cells.append(
                format_cell(
                    value,
                    higher=bool(higher[group]),
                    lower=bool(lower[group]),
                )
            )

        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    return "\n".join(lines) + "\n", pairwise_by_metric


def render_condition_post_change_table(df: pd.DataFrame) -> tuple[str, dict[str, dict[str, object]]]:
    pairwise_by_metric: dict[str, dict[str, object]] = {}
    interactive_n = int((df["Condition"] == "Interactive").sum())
    text_n = int((df["Condition"] == "Text").sum())

    lines = [
        r"\centering",
        r"\small",
        r"\textbf{Post And Change By Condition}\\",
        r"\emph{Averaged across all participants. Cells show mean (SD); bold marks the higher significant value, italics the lower one; asterisks denote Interactive vs. Text comparisons.}\\[0.4em]",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        "Metric & Interactive ($n={}$) & Text ($n={}$) ".format(interactive_n, text_n) + ("\\" * 2),
        r"\midrule",
    ]

    for metric, label in CONDITION_POST_CHANGE_METRICS:
        interactive = condition_series(df, metric, "Interactive")
        text = condition_series(df, metric, "Text")
        interactive_hi, interactive_lo, text_hi, text_lo, result = within_condition_direction(interactive, text)
        pairwise_by_metric[metric] = result

        interactive_cell = format_cell(summarize_metric(interactive), higher=interactive_hi, lower=interactive_lo)
        text_cell = format_cell(summarize_metric(text), higher=text_hi, lower=text_lo)
        stars = significance_stars(float(result["p_one"])) if result["direction"] != "ns" else ""
        metric_label = f"{label}{stars}"
        lines.append(" & ".join([metric_label, interactive_cell, text_cell]) + " " + ("\\" * 2))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    return "\n".join(lines) + "\n", pairwise_by_metric


def render_pre_summary(pairwise_by_metric: dict[str, dict[str, object]]) -> str:
    sentences: list[str] = []
    for metric, label in PRE_METRICS:
        result = pairwise_by_metric[metric]
        phrase = metric_phrase(label)
        if result["direction"] == "ns":
            sentences.append(
                f"No pre-experiment cultural-group difference was detected for {phrase} (Welch $t={result['t']:.2f}$, two-sided $p {format_p_value(float(result['p_two']))}$)."
            )
            continue

        if result["direction"] == "non-WEIRD > WEIRD":
            higher_group = "non-WEIRD participants"
            lower_group = "WEIRD participants"
            higher_mean = float(result["mean_non_weird"])
            lower_mean = float(result["mean_weird"])
        else:
            higher_group = "WEIRD participants"
            lower_group = "non-WEIRD participants"
            higher_mean = float(result["mean_weird"])
            lower_mean = float(result["mean_non_weird"])

        sentences.append(
            f"For {phrase}, {higher_group} scored higher than {lower_group} ($M={higher_mean:.2f}$ vs. $M={lower_mean:.2f}$; Welch $t={result['t']:.2f}$, two-sided $p {format_p_value(float(result['p_two']))}$, one-sided $p {format_p_value(float(result['p_one']))}$)."
        )

    return "\\noindent " + " ".join(sentences) + "\n"


def comparison_phrase(result: dict[str, object]) -> str:
    left_condition, left_group = result["left"]
    right_condition, right_group = result["right"]
    if result["direction"] == "ns":
        return (
            f"{left_condition} {left_group} did not differ from {right_condition} {right_group} "
            f"(Welch $t={result['t']:.2f}$, two-sided $p {format_p_value(float(result['p_two']))}$)"
        )

    if result["direction"] == f"{left_condition} {left_group} > {right_condition} {right_group}":
        higher_label = f"{left_condition} {left_group}"
        lower_label = f"{right_condition} {right_group}"
    else:
        higher_label = f"{right_condition} {right_group}"
        lower_label = f"{left_condition} {left_group}"

    return (
        f"{higher_label} was higher than {lower_label} "
        f"(Welch $t={result['t']:.2f}$, two-sided $p {format_p_value(float(result['p_two']))}$, "
        f"one-sided $p {format_p_value(float(result['p_one']))}$)"
    )


def render_post_change_summary(pairwise_by_metric: dict[str, list[dict[str, object]]]) -> str:
    sentences: list[str] = []
    for metric, label in POST_CHANGE_METRICS:
        significant = [result for result in pairwise_by_metric[metric] if result["direction"] != "ns"]
        phrase = label.lower()
        if not significant:
            sentences.append(f"No pairwise group differences reached significance for {phrase}.")
            continue

        comparisons = "; ".join(comparison_phrase(result) for result in significant)
        sentences.append(f"For {phrase}, {comparisons}.")

    return "\\noindent " + " ".join(sentences) + "\n"


def render_condition_post_change_summary(pairwise_by_metric: dict[str, dict[str, object]]) -> str:
    sentences: list[str] = []
    for metric, label in CONDITION_POST_CHANGE_METRICS:
        result = pairwise_by_metric[metric]
        phrase = label.lower()
        if result["direction"] == "ns":
            sentences.append(
                f"No condition difference was detected for {phrase} when averaging across participants (Welch $t={result['t']:.2f}$, two-sided $p {format_p_value(float(result['p_two']))}$)."
            )
            continue

        if result["direction"] == "WEIRD > non-WEIRD":
            higher_group = "Interactive participants"
            lower_group = "Text participants"
            higher_mean = float(result["mean_weird"])
            lower_mean = float(result["mean_non_weird"])
        else:
            higher_group = "Text participants"
            lower_group = "Interactive participants"
            higher_mean = float(result["mean_non_weird"])
            lower_mean = float(result["mean_weird"])

        sentences.append(
            f"For {phrase}, {higher_group} scored higher than {lower_group} ($M={higher_mean:.2f}$ vs. $M={lower_mean:.2f}$; Welch $t={result['t']:.2f}$, two-sided $p {format_p_value(float(result['p_two']))}$, one-sided $p {format_p_value(float(result['p_one']))}$)."
        )

    return "\\noindent " + " ".join(sentences) + "\n"


def print_pre_results(pairwise_by_metric: dict[str, dict[str, object]]) -> None:
    for metric, label in PRE_METRICS:
        print(f"\n{label}:")
        result = pairwise_by_metric[metric]
        print(
            f"  WEIRD vs non-WEIRD: "
            f"means={result['mean_weird']:.3f}/{result['mean_non_weird']:.3f}, "
            f"t={result['t']:.3f}, p(two-sided)={result['p_two']:.4f}, "
            f"p(one-sided)={result['p_one']:.4f}, direction={result['direction']}"
        )


def print_post_change_results(pairwise_by_metric: dict[str, list[dict[str, object]]]) -> None:
    for metric, label in POST_CHANGE_METRICS:
        print(f"\n{label}:")
        for result in pairwise_by_metric[metric]:
            left = f"{result['left'][0]} {result['left'][1]}"
            right = f"{result['right'][0]} {result['right'][1]}"
            print(
                f"  {left} vs {right}: "
                f"means={result['mean_left']:.3f}/{result['mean_right']:.3f}, "
                f"t={result['t']:.3f}, p(two-sided)={result['p_two']:.4f}, "
                f"p(one-sided)={result['p_one']:.4f}, direction={result['direction']}"
            )


def print_condition_post_change_results(pairwise_by_metric: dict[str, dict[str, object]]) -> None:
    for metric, label in CONDITION_POST_CHANGE_METRICS:
        print(f"\n{label}:")
        result = pairwise_by_metric[metric]
        print(
            f"  Interactive vs Text: "
            f"means={result['mean_weird']:.3f}/{result['mean_non_weird']:.3f}, "
            f"t={result['t']:.3f}, p(two-sided)={result['p_two']:.4f}, "
            f"p(one-sided)={result['p_one']:.4f}, direction={result['direction']}"
        )


def main() -> None:
    df = build_df()
    pre_table_tex, pre_results = render_pre_table(df)
    condition_post_change_table_tex, condition_post_change_results = render_condition_post_change_table(df)
    post_change_table_tex, post_change_results = render_post_change_table(df)
    pre_summary_tex = render_pre_summary(pre_results)
    condition_post_change_summary_tex = render_condition_post_change_summary(condition_post_change_results)
    post_change_summary_tex = render_post_change_summary(post_change_results)

    combined_top_tables_tex = "\n".join([
        r"\begin{center}",
        r"\begin{minipage}[t]{0.48\linewidth}",
        pre_table_tex.rstrip(),
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.48\linewidth}",
        condition_post_change_table_tex.rstrip(),
        r"\end{minipage}",
        r"\end{center}",
    ]) + "\n"

    centered_post_change_table_tex = "\n".join([
        r"\begin{center}",
        post_change_table_tex.rstrip(),
        r"\end{center}",
    ]) + "\n"

    OUTPUT_PATH.write_text(
        "\\subsection{Summary}\n"
        + combined_top_tables_tex
        + "\n"
        + pre_summary_tex
        + "\n"
        + condition_post_change_summary_tex
        + "\n"
        + centered_post_change_table_tex
        + "\n"
        + post_change_summary_tex
    )
    print(f"Saved: {OUTPUT_PATH}")
    print("\nPre table tests:")
    print_pre_results(pre_results)
    print("\nCondition post/change table tests:")
    print_condition_post_change_results(condition_post_change_results)
    print("\nPost/change table tests:")
    print_post_change_results(post_change_results)


if __name__ == "__main__":
    main()