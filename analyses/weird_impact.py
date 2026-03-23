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

# Proxies for Industrialized/Rich/Democratic via nationality.
INDUSTRIALIZED_NATIONALITIES = {
    "Australia",
    "Canada",
    "Chile",
    "Croatia",
    "Czech Republic",
    "Estonia",
    "France",
    "Germany",
    "Greece",
    "Hong Kong",
    "Hungary",
    "Israel",
    "Italy",
    "Japan",
    "Korea",
    "Lithuania",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovenia",
    "Spain",
    "Sweden",
    "United Kingdom",
    "United States",
}

RICH_NATIONALITIES = {
    "Australia",
    "Canada",
    "Chile",
    "Czech Republic",
    "Estonia",
    "France",
    "Germany",
    "Greece",
    "Hong Kong",
    "Hungary",
    "Israel",
    "Italy",
    "Japan",
    "Korea",
    "Lithuania",
    "Netherlands",
    "Poland",
    "Portugal",
    "Slovenia",
    "Spain",
    "Sweden",
    "Trinidad and Tobago",
    "United Kingdom",
    "United States",
}

DEMOCRATIC_NATIONALITIES = {
    "Australia",
    "Brazil",
    "Canada",
    "Chile",
    "Colombia",
    "Croatia",
    "Czech Republic",
    "Estonia",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "India",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Korea",
    "Lithuania",
    "Malaysia",
    "Mexico",
    "Netherlands",
    "Philippines",
    "Poland",
    "Portugal",
    "Romania",
    "Slovenia",
    "South Africa",
    "Spain",
    "Sweden",
    "Tunisia",
    "United Kingdom",
    "United States",
}

EDUCATED_LEVELS = {"Bachelor", "Master", "PhD", "Graduate Professional Degree"}


def _normalize_text(value: object) -> str:
    text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
    text = text.replace("judgmental", "judgemental")
    return " ".join(text.split())


def _is_white_ethnicity(row: pd.Series) -> bool:
    simplified = _normalize_text(row.get("Ethnicity simplified", ""))
    if simplified == "white":
        return True

    ethnicity = _normalize_text(row.get("Ethnicity", ""))
    return ethnicity.startswith("white")


def _is_educated(value: object) -> bool:
    return str(value).strip() in EDUCATED_LEVELS


def _weird_flags(row: pd.Series) -> dict[str, bool]:
    nationality = str(row.get("Nationality", "")).strip()
    return {
        "W": _is_white_ethnicity(row),
        "E": _is_educated(row.get("Education", "")),
        "I": nationality in INDUSTRIALIZED_NATIONALITIES,
        "R": nationality in RICH_NATIONALITIES,
        "D": nationality in DEMOCRATIC_NATIONALITIES,
    }


def _weird_score(row: pd.Series) -> int:
    flags = _weird_flags(row)
    return int(sum(flags.values()))


def _is_weird(row: pd.Series) -> bool:
    # Inclusive threshold requested: classify as WEIRD when >=3 of 5 criteria are met.
    return _weird_score(row) >= 3


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


required_cols = ["Education", "Ethnicity", "Ethnicity simplified", "Nationality", "Condition"]
missing = [col for col in required_cols if col not in combined.columns]
if missing:
    raise ValueError(f"Expected columns missing from dataset: {missing}")

combined["WEIRD Group"] = combined.apply(_is_weird, axis=1).map({True: "WEIRD", False: "Non-WEIRD"})
combined["WEIRD Score"] = combined.apply(_weird_score, axis=1)

combined["Emotional Impact Delta"] = _build_emotional_delta(combined)
combined["Analytical Trust Impact Delta"] = _build_analytical_delta(combined)

panel_specs = [
    ("Emotional Impact Delta", "Emotional", "WEIRD", 0, 0),
    ("Emotional Impact Delta", "Emotional", "Non-WEIRD", 0, 1),
    ("Analytical Trust Impact Delta", "Analytical", "WEIRD", 1, 0),
    ("Analytical Trust Impact Delta", "Analytical", "Non-WEIRD", 1, 1),
]

all_summaries = []
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for impact_col, impact_label, weird_label, row_i, col_i in panel_specs:
    ax = axes[row_i, col_i]
    subset = combined[combined["WEIRD Group"] == weird_label].copy()
    subset = subset[["Condition", impact_col]].dropna()

    summary_df = _summarize_by_condition(subset, impact_col)
    summary_df["impact_type"] = impact_label
    summary_df["weird_group"] = weird_label
    all_summaries.append(summary_df)

    x = summary_df["Condition"].tolist()
    means = summary_df["mean_delta"].to_numpy()
    sems = summary_df["sem"].fillna(0.0).to_numpy()
    p_vals = summary_df["p_vs_zero"].to_numpy()

    bars = ax.bar(x, means, yerr=sems, capsize=6, color=["#4C78A8", "#72B7B2"])
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_title(f"{impact_label} - {weird_label}")
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

fig.suptitle("Overall Trust Impact by Condition and WEIRD Group", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])

out_dir = Path("./analysis")
out_dir.mkdir(parents=True, exist_ok=True)

plot_path = out_dir / "weird_split_emotional_analytical_2x2.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

summary_all_df = pd.concat(all_summaries, ignore_index=True)
summary_path = out_dir / "weird_split_emotional_analytical_summary.csv"
summary_all_df.to_csv(summary_path, index=False)

group_counts = combined.groupby(["WEIRD Group", "Condition"], as_index=False).size()
score_counts = combined.groupby(["WEIRD Score", "WEIRD Group"], as_index=False).size()

print("WEIRD-split emotional and analytical impact summary:")
print(summary_all_df.to_string(index=False))
print("\nWEIRD group sample sizes by condition:")
print(group_counts.to_string(index=False))
print("\nWEIRD score distribution:")
print(score_counts.sort_values(["WEIRD Score", "WEIRD Group"]).to_string(index=False))
print(f"\nSaved 2x2 plot to: {plot_path}")
print(f"Saved summary table to: {summary_path}")
