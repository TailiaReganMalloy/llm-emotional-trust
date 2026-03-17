from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EMOTIONAL_PAIRS = {
    "AI systems are 1": {"Apathetic": 0, "Empathetic": 1},
    "AI systems are 2": {"Insensitive": 0, "Sensitive": 1},
    "AI systems are 3": {"Impersonal": 0, "Personal": 1},
    "AI systems are 4": {"Ignoring": 0, "Caring": 1},
    "AI systems are 5": {"Self Serving": 0, "Self-Serving": 0, "Altruistic": 1},
    "AI systems are 6": {"Rude": 0, "Cordial": 1},
    "AI systems are 7": {"Indifferent": 0, "Responsive": 1},
    "AI systems are 8": {"Judgemental": 0, "Judgmental": 0, "Open-Minded": 1},
    "AI systems are 9": {"Impatient": 0, "Patient": 1},
}

ANALYTICAL_PAIRS = [
    ("AI Weary", "AI Weary Post"),
    ("AI Confident", "AI Confident Post"),
    ("AI Suspicious", "AI Suspicious Post"),
    ("AI Trust", "AI Trust Post"),
    ("AI Harm", "AI Harm Post"),
    ("AI Honest", "AI Honest Post"),
    ("AI Security", "AI Security Post"),
    ("AI Deceptive", "AI Deceptive Post"),
    ("AI Reliable", "AI Reliable Post"),
    ("AI Trustworthy", "AI Trustworthy Post"),
]

NEGATIVE_TRUST_ITEMS = {"AI Weary", "AI Suspicious", "AI Harm", "AI Deceptive"}

LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Agree": 1,
    "Strongly Agree": 2,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot education-split emotional/analytical trust impact by condition."
    )
    parser.add_argument(
        "--input-file",
        default="./data/combined.csv",
        help="Input CSV path (default: ./data/combined.csv)",
    )
    return parser.parse_args()


def _normalize_text(value: object) -> str:
    text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
    text = text.replace("judgmental", "judgemental")
    return " ".join(text.split())


LIKERT_VALUE_MAP = {_normalize_text(k): v for k, v in LIKERT_MAP.items()}

EMOTIONAL_VALUE_MAP: dict[str, int] = {}
for mapping in EMOTIONAL_PAIRS.values():
    for key, val in mapping.items():
        EMOTIONAL_VALUE_MAP[_normalize_text(key)] = val


def _to_likert_numeric(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).map(_normalize_text).map(LIKERT_VALUE_MAP)
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def _to_emotional_numeric(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).map(_normalize_text).map(EMOTIONAL_VALUE_MAP)
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def _build_emotional_delta(df: pd.DataFrame) -> pd.Series:
    deltas = []
    for pre_col in EMOTIONAL_PAIRS:
        post_col = f"{pre_col} Post"
        if pre_col not in df.columns or post_col not in df.columns:
            continue
        pre = _to_emotional_numeric(df[pre_col])
        post = _to_emotional_numeric(df[post_col])
        deltas.append(post - pre)
    if not deltas:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(deltas, axis=1).mean(axis=1, skipna=True)


def _build_analytical_delta(df: pd.DataFrame) -> pd.Series:
    deltas = []
    for pre_col, post_col in ANALYTICAL_PAIRS:
        if pre_col not in df.columns or post_col not in df.columns:
            continue
        pre = _to_likert_numeric(df[pre_col])
        post = _to_likert_numeric(df[post_col])
        delta = post - pre
        if pre_col in NEGATIVE_TRUST_ITEMS:
            delta = -delta
        deltas.append(delta)
    if not deltas:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(deltas, axis=1).mean(axis=1, skipna=True)


def _summarize_by_condition(df: pd.DataFrame, impact_col: str) -> pd.DataFrame:
    conditions = df["Condition"].dropna().unique().tolist()
    if "Interactive" in conditions and "Static" in conditions:
        ordered_conditions = ["Interactive", "Static"]
    else:
        ordered_conditions = sorted(conditions)

    rows = []
    for condition in ordered_conditions:
        values = df.loc[df["Condition"] == condition, impact_col].to_numpy()
        values = values[np.isfinite(values)]
        if len(values) < 2:
            t_stat = float("nan")
            p_val = float("nan")
            mean_val = float(np.mean(values)) if len(values) else float("nan")
            sem_val = float("nan")
        else:
            t_stat, p_val = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
            mean_val = float(np.mean(values))
            sem_val = float(stats.sem(values, nan_policy="omit"))

        rows.append(
            {
                "Condition": condition,
                "n": int(len(values)),
                "mean_delta": mean_val,
                "sem": sem_val,
                "t_vs_zero": float(t_stat),
                "p_vs_zero": float(p_val),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["Condition", "n", "mean_delta", "sem", "t_vs_zero", "p_vs_zero"],
    )


def _condition_effect_pvalue(df: pd.DataFrame, impact_col: str) -> float:
    conditions = df["Condition"].dropna().unique().tolist()
    if "Interactive" in conditions and "Static" in conditions:
        ordered_conditions = ["Interactive", "Static"]
    else:
        ordered_conditions = sorted(conditions)

    if len(ordered_conditions) != 2:
        return float("nan")

    group_a = df.loc[df["Condition"] == ordered_conditions[0], impact_col].to_numpy()
    group_b = df.loc[df["Condition"] == ordered_conditions[1], impact_col].to_numpy()
    group_a = group_a[np.isfinite(group_a)]
    group_b = group_b[np.isfinite(group_b)]

    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")

    _, p_val = stats.ttest_ind(group_a, group_b, equal_var=True, nan_policy="omit")
    return float(p_val)


def _education_to_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("'", "")
    if normalized in {"bachelor", "bachelor degree", "bachelors degree"}:
        return "Bachelor degree"
    return "Other education"


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input_file)
    combined = pd.read_csv(input_path)

	assert

    if "Status" in combined.columns:
        combined = combined[combined["Status"].astype(str).str.upper() == "APPROVED"].copy()

    education_source = "Education"
    if education_source not in combined.columns:
        raise ValueError("Expected 'Education' column in input data.")

    combined["Education Group"] = combined[education_source].apply(_education_to_group)
    combined = combined[combined["Education Group"].notna()].copy()

    combined["Emotional Impact Delta"] = _build_emotional_delta(combined)
    combined["Analytical Trust Impact Delta"] = _build_analytical_delta(combined)

    panel_specs = [
        ("Emotional Impact Delta", "Emotional", "Bachelor degree", 0, 0),
        ("Emotional Impact Delta", "Emotional", "Other education", 0, 1),
        ("Analytical Trust Impact Delta", "Analytical", "Bachelor degree", 1, 0),
        ("Analytical Trust Impact Delta", "Analytical", "Other education", 1, 1),
    ]

    all_summaries = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    for impact_col, impact_label, education_label, row_i, col_i in panel_specs:
        ax = axes[row_i, col_i]
        subset = combined[combined["Education Group"] == education_label].copy()
        subset = subset[["Condition", impact_col]].dropna()

        summary_df = _summarize_by_condition(subset, impact_col)
        panel_p_val = _condition_effect_pvalue(subset, impact_col)
        summary_df["impact_type"] = impact_label
        summary_df["education_group"] = education_label
        summary_df["p_condition_effect_within_split"] = panel_p_val
        all_summaries.append(summary_df)

        x = summary_df["Condition"].tolist()
        means = summary_df["mean_delta"].to_numpy()
        sems = summary_df["sem"].fillna(0.0).to_numpy()
        bars = ax.bar(x, means, yerr=sems, capsize=6, color=["#4C78A8", "#72B7B2"])
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{impact_label} - {education_label}")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Trust Impact (Post - Pre)")

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min if y_max != y_min else 1.0
        offset = 0.06 * y_range

        if len(bars) >= 2:
            first_center = bars[0].get_x() + bars[0].get_width() / 2
            last_center = bars[-1].get_x() + bars[-1].get_width() / 2
            x_center = (first_center + last_center) / 2
        elif len(bars) == 1:
            x_center = bars[0].get_x() + bars[0].get_width() / 2
        else:
            x_center = 0.0

        label = f"p={panel_p_val:.3g}" if np.isfinite(panel_p_val) else "p=nan"
        y_label = y_max + offset
        ax.text(x_center, y_label, label, ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.set_ylim(y_min, y_max + (0.14 * y_range))

    fig.suptitle("Overall Trust Impact by Condition and Education Group", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = Path("./analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.stem == "combined":
        plot_filename = "education_split_emotional_analytical_2x2.png"
        summary_filename = "education_split_emotional_analytical_summary.csv"
    else:
        plot_filename = f"education_split_emotional_analytical_2x2_{input_path.stem}.png"
        summary_filename = f"education_split_emotional_analytical_summary_{input_path.stem}.csv"

    plot_path = out_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    summary_all_df = pd.concat(all_summaries, ignore_index=True)
    summary_path = out_dir / summary_filename
    summary_all_df.to_csv(summary_path, index=False)

    print("Education-split emotional and analytical impact summary:")
    print(summary_all_df.to_string(index=False))
    print(f"\nInput file used: {input_path}")
    print(f"Education source column used: {education_source}")
    print(f"Saved 2x2 plot to: {plot_path}")
    print(f"Saved summary table to: {summary_path}")


if __name__ == "__main__":
    main()
