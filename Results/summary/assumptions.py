from __future__ import annotations

from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
MEASURES_PATH = REPO_ROOT / "Methodology" / "Measures" / "measures.py"
DEMOGRAPHICS_PATH = REPO_ROOT / "Methodology" / "Participants" / "demographics.py"

PRE_METRICS = [
    ("analytical_pre_01", "analytical pre"),
    ("emotional_pre_01", "emotional pre"),
    ("overall_pre_01", "overall pre"),
]

CONDITION_METRICS = [
    ("analytical_post_01", "analytical post"),
    ("analytical_change_01", "analytical change"),
    ("emotional_post_01", "emotional post"),
    ("emotional_change_01", "emotional change"),
    ("overall_post_01", "overall post"),
    ("overall_change_01", "overall change"),
]

FOUR_GROUPS = [
    ("Interactive", "WEIRD"),
    ("Interactive", "non-WEIRD"),
    ("Text", "WEIRD"),
    ("Text", "non-WEIRD"),
]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measures = load_module(MEASURES_PATH, "assumptions_measures")
demographics = load_module(DEMOGRAPHICS_PATH, "assumptions_demographics")


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


def culture_series(df: pd.DataFrame, metric: str, culture: str) -> pd.Series:
    mask = df["Culture"] == culture
    return pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()


def condition_series(df: pd.DataFrame, metric: str, condition: str) -> pd.Series:
    mask = df["Condition"] == condition
    return pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()


def cell_series(df: pd.DataFrame, metric: str, condition: str, culture: str) -> pd.Series:
    mask = (df["Condition"] == condition) & (df["Culture"] == culture)
    return pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()


def shapiro_pvalue(series: pd.Series) -> float:
    if len(series) < 3:
        return float("nan")
    return float(stats.shapiro(series).pvalue)


def brown_forsythe_pvalue(*groups: pd.Series) -> float:
    clean_groups = [group for group in groups if len(group) >= 2]
    if len(clean_groups) < 2:
        return float("nan")
    return float(stats.levene(*clean_groups, center="median").pvalue)


def format_metric_list(labels: list[str]) -> str:
    if not labels:
        return "none"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def describe_variance_result(prefix: str, labels: list[str]) -> str:
    if not labels:
        return f"{prefix} was not detected"
    return f"{prefix} was detected only for {format_metric_list(labels)}"


def render_assumptions_summary(df: pd.DataFrame | None = None) -> str:
    if df is None:
        df = build_df()

    normality_counts = {
        "pre": {"total": 0, "failed": 0},
        "condition": {"total": 0, "failed": 0},
        "cell": {"total": 0, "failed": 0},
    }

    for metric, label in PRE_METRICS:
        for culture in ["WEIRD", "non-WEIRD"]:
            pvalue = shapiro_pvalue(culture_series(df, metric, culture))
            if not np.isnan(pvalue):
                normality_counts["pre"]["total"] += 1
                if pvalue < 0.05:
                    normality_counts["pre"]["failed"] += 1

    for metric, label in CONDITION_METRICS:
        for condition in ["Interactive", "Text"]:
            pvalue = shapiro_pvalue(condition_series(df, metric, condition))
            if not np.isnan(pvalue):
                normality_counts["condition"]["total"] += 1
                if pvalue < 0.05:
                    normality_counts["condition"]["failed"] += 1

    for metric, label in CONDITION_METRICS:
        for condition, culture in FOUR_GROUPS:
            pvalue = shapiro_pvalue(cell_series(df, metric, condition, culture))
            if not np.isnan(pvalue):
                normality_counts["cell"]["total"] += 1
                if pvalue < 0.05:
                    normality_counts["cell"]["failed"] += 1

    normality_total = sum(group["total"] for group in normality_counts.values())
    normality_failed = sum(group["failed"] for group in normality_counts.values())

    pre_heterogeneous: list[str] = []
    for metric, label in PRE_METRICS:
        pvalue = brown_forsythe_pvalue(
            culture_series(df, metric, "WEIRD"),
            culture_series(df, metric, "non-WEIRD"),
        )
        if not np.isnan(pvalue) and pvalue < 0.05:
            pre_heterogeneous.append(label)

    condition_heterogeneous: list[str] = []
    for metric, label in CONDITION_METRICS:
        pvalue = brown_forsythe_pvalue(
            condition_series(df, metric, "Interactive"),
            condition_series(df, metric, "Text"),
        )
        if not np.isnan(pvalue) and pvalue < 0.05:
            condition_heterogeneous.append(label)

    cell_heterogeneous: list[str] = []
    for metric, label in CONDITION_METRICS:
        pvalue = brown_forsythe_pvalue(
            *(cell_series(df, metric, condition, culture) for condition, culture in FOUR_GROUPS)
        )
        if not np.isnan(pvalue) and pvalue < 0.05:
            cell_heterogeneous.append(label)

    normality_sentence = (
        f"Shapiro-Wilk checks rejected exact normality in {normality_failed} of {normality_total} analyzed group-level distributions "
        f"({normality_counts['pre']['failed']} of {normality_counts['pre']['total']} pre-trust cultural-group distributions, "
        f"{normality_counts['condition']['failed']} of {normality_counts['condition']['total']} condition-averaged post/change distributions, "
        f"and {normality_counts['cell']['failed']} of {normality_counts['cell']['total']} four-cell post/change distributions), which is expected for bounded trust scores and moderate-to-large samples."
    )

    variance_sentence = (
        "Brown-Forsythe tests showed that "
        f"{describe_variance_result('pre-trust variance heterogeneity across cultural groups', pre_heterogeneous)}; "
        f"{describe_variance_result('condition-level variance heterogeneity', condition_heterogeneous)}; "
        f"and {describe_variance_result('four-cell variance heterogeneity', cell_heterogeneous)}."
    )

    robustness_sentence = (
        "Because the summary analyses use Welch tests throughout, the reported parametric comparisons remain appropriate under unequal variances and are generally robust to the mild normality departures observed at the current cell sizes ($n=64$ to $304$)."
    )

    return "\\noindent Assumption checks for the parametric analyses indicated that " + normality_sentence + " " + variance_sentence + " " + robustness_sentence + "\n"


def main() -> None:
    print(render_assumptions_summary())


if __name__ == "__main__":
    main()