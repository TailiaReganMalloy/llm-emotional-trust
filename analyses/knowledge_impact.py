from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


combined = pd.read_csv("./data/combined.csv")

if "Status" in combined.columns:
    combined = combined[combined["Status"].astype(str).str.upper() == "APPROVED"].copy()


EMOTIONAL_PAIRS = {
    "AI systems are 1": {"Apathetic": -1, "Empathetic": 1},
    "AI systems are 2": {"Insensitive": -1, "Sensitive": 1},
    "AI systems are 3": {"Impersonal": -1, "Personal": 1},
    "AI systems are 4": {"Ignoring": -1, "Caring": 1},
    "AI systems are 5": {"Self Serving": -1, "Self-Serving": -1, "Altruistic": 1},
    "AI systems are 6": {"Rude": -1, "Cordial": 1},
    "AI systems are 7": {"Indifferent": -1, "Responsive": 1},
    "AI systems are 8": {"Judgemental": -1, "Judgmental": -1, "Open-Minded": 1},
    "AI systems are 9": {"Impatient": -1, "Patient": 1},
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


LOW_KNOWLEDGE = {"beginner knowledge", "conceptual understanding", "no knowledge"}
HIGH_KNOWLEDGE = {"advanced", "expert"}


def _normalize_text(value: object) -> str:
    text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
    text = text.replace("judgmental", "judgemental")
    return " ".join(text.split())


def _knowledge_group(value: object) -> str | None:
    key = _normalize_text(value)
    if key in LOW_KNOWLEDGE:
        return "Low Knowledge"
    if key in HIGH_KNOWLEDGE:
        return "High Knowledge"
    return None


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

    return pd.DataFrame(rows)


combined["Knowledge Group"] = combined["AI Knowledge"].apply(_knowledge_group)
combined = combined[combined["Knowledge Group"].notna()].copy()

combined["Emotional Impact Delta"] = _build_emotional_delta(combined)
combined["Analytical Trust Impact Delta"] = _build_analytical_delta(combined)

panel_specs = [
    ("Emotional Impact Delta", "Emotional", "Low Knowledge", 0, 0),
    ("Emotional Impact Delta", "Emotional", "High Knowledge", 0, 1),
    ("Analytical Trust Impact Delta", "Analytical", "Low Knowledge", 1, 0),
    ("Analytical Trust Impact Delta", "Analytical", "High Knowledge", 1, 1),
]

all_summaries = []
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for impact_col, impact_label, knowledge_label, row_i, col_i in panel_specs:
    ax = axes[row_i, col_i]
    subset = combined[combined["Knowledge Group"] == knowledge_label].copy()
    subset = subset[["Condition", impact_col]].dropna()

    summary_df = _summarize_by_condition(subset, impact_col)
    summary_df["impact_type"] = impact_label
    summary_df["knowledge_group"] = knowledge_label
    all_summaries.append(summary_df)

    x = summary_df["Condition"].tolist()
    means = summary_df["mean_delta"].to_numpy()
    sems = summary_df["sem"].fillna(0.0).to_numpy()
    p_vals = summary_df["p_vs_zero"].to_numpy()

    bars = ax.bar(x, means, yerr=sems, capsize=6, color=["#4C78A8", "#72B7B2"])
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_title(f"{impact_label} - {knowledge_label}")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Trust Impact (Post - Pre)")

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max != y_min else 1.0
    offset = 0.03 * y_range

    for bar, p_val in zip(bars, p_vals):
        height = bar.get_height()
        label = f"p={p_val:.3g}" if np.isfinite(p_val) else "p=nan"
        y = height + offset if height >= 0 else height - offset
        va = "bottom" if height >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va=va, fontsize=10)

fig.suptitle("Overall Trust Impact by Condition and AI Knowledge Group", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

out_dir = Path("./analysis")
out_dir.mkdir(parents=True, exist_ok=True)

plot_path = out_dir / "knowledge_split_emotional_analytical_2x2.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

summary_all_df = pd.concat(all_summaries, ignore_index=True)
summary_path = out_dir / "knowledge_split_emotional_analytical_summary.csv"
summary_all_df.to_csv(summary_path, index=False)

print("Knowledge-split emotional and analytical impact summary:")
print(summary_all_df.to_string(index=False))
print(f"\nSaved 2x2 plot to: {plot_path}")
print(f"Saved summary table to: {summary_path}")
