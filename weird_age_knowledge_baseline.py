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

ANALYTICAL_PRE_ITEMS = [
    "AI Weary",
    "AI Confident",
    "AI Suspicious",
    "AI Trust",
    "AI Harm",
    "AI Honest",
    "AI Security",
    "AI Deceptive",
    "AI Reliable",
    "AI Trustworthy",
]

NEGATIVE_TRUST_ITEMS = {"AI Weary", "AI Suspicious", "AI Harm", "AI Deceptive"}

LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Agree": 1,
    "Strongly Agree": 2,
}

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

UNKNOWLEDGEABLE_AI = {"beginner knowledge", "no knowledge"}
KNOWLEDGEABLE_AI = {"advanced", "expert", "conceptual understanding"}


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


def _knowledge_band(value: object) -> str | None:
    key = _normalize_text(value)
    if key in KNOWLEDGEABLE_AI:
        return "Knowledgeable"
    if key in UNKNOWLEDGEABLE_AI:
        return "Unknowledgeable"
    return None


def _age_band(row: pd.Series) -> str | None:
    age_val = row.get("Age_demographic", np.nan)
    age_num = pd.to_numeric(age_val, errors="coerce")
    if not np.isfinite(age_num):
        age_num = pd.to_numeric(row.get("Age", np.nan), errors="coerce")
    if not np.isfinite(age_num):
        return None
    return "Young" if age_num <= 38 else "Old"


def _weird6_score(row: pd.Series) -> int:
    nationality = str(row.get("Nationality", "")).strip()
    flags = {
        "W": _is_white_ethnicity(row),
        "E": _is_educated(row.get("Education", "")),
        "I": nationality in INDUSTRIALIZED_NATIONALITIES,
        "R": nationality in RICH_NATIONALITIES,
        "D": nationality in DEMOCRATIC_NATIONALITIES,
        "K": _knowledge_band(row.get("AI Knowledge", "")) == "Knowledgeable",
    }
    return int(sum(flags.values()))


def _is_weird_4_of_6(row: pd.Series) -> bool:
    return _weird6_score(row) >= 4


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


def _build_emotional_baseline(df: pd.DataFrame) -> pd.Series:
    vals = []
    for pre_col in EMOTIONAL_PAIRS:
        if pre_col not in df.columns:
            continue
        vals.append(_to_emotional_numeric(df[pre_col]))
    if not vals:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(vals, axis=1).mean(axis=1, skipna=True)


def _build_analytical_baseline(df: pd.DataFrame) -> pd.Series:
    vals = []
    for pre_col in ANALYTICAL_PRE_ITEMS:
        if pre_col not in df.columns:
            continue
        item = _to_likert_numeric(df[pre_col])
        if pre_col in NEGATIVE_TRUST_ITEMS:
            item = -item
        vals.append(item)
    if not vals:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(vals, axis=1).mean(axis=1, skipna=True)


def _summarize_by_group(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    groups = ["WEIRD+Young+Knowledgeable", "Non-WEIRD"]
    rows = []
    for group in groups:
        values = df.loc[df["Group"] == group, value_col].to_numpy()
        values = values[np.isfinite(values)]
        mean_val = float(np.mean(values)) if len(values) else float("nan")
        sem_val = float(stats.sem(values, nan_policy="omit")) if len(values) >= 2 else float("nan")
        rows.append(
            {
                "group": group,
                "n": int(len(values)),
                "mean": mean_val,
                "sem": sem_val,
            }
        )

    summary_df = pd.DataFrame(rows)

    weird_vals = df.loc[df["Group"] == "WEIRD+Young+Knowledgeable", value_col].to_numpy()
    non_weird_vals = df.loc[df["Group"] == "Non-WEIRD", value_col].to_numpy()
    weird_vals = weird_vals[np.isfinite(weird_vals)]
    non_weird_vals = non_weird_vals[np.isfinite(non_weird_vals)]

    if len(weird_vals) >= 2 and len(non_weird_vals) >= 2:
        _, p_val = stats.ttest_ind(weird_vals, non_weird_vals, equal_var=True, nan_policy="omit")
    else:
        p_val = float("nan")

    summary_df["p_group_difference"] = float(p_val)
    return summary_df


required_cols = [
    "Education",
    "Ethnicity",
    "Ethnicity simplified",
    "Nationality",
    "Condition",
    "AI Knowledge",
]
missing = [col for col in required_cols if col not in combined.columns]
if missing:
    raise ValueError(f"Expected columns missing from dataset: {missing}")

combined["Age Band"] = combined.apply(_age_band, axis=1)
combined["Knowledge Band"] = combined["AI Knowledge"].apply(_knowledge_band)
combined["WEIRD6 Score"] = combined.apply(_weird6_score, axis=1)
combined["WEIRD 4of6"] = combined.apply(_is_weird_4_of_6, axis=1)

combined["Group"] = np.where(
    (
        (combined["WEIRD 4of6"])
        & (combined["Age Band"] == "Young")
        & (combined["Knowledge Band"] == "Knowledgeable")
    ),
    "WEIRD+Young+Knowledgeable",
    "Non-WEIRD",
)

combined["Emotional Baseline"] = _build_emotional_baseline(combined)
combined["Analytical Baseline"] = _build_analytical_baseline(combined)

panel_specs = [
    ("Emotional Baseline", "Emotional Baseline", 0),
    ("Analytical Baseline", "Analytical Baseline", 1),
]

all_summaries = []
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for value_col, title, ax_i in panel_specs:
    ax = axes[ax_i]
    summary_df = _summarize_by_group(combined[["Group", value_col]].dropna(), value_col)
    summary_df["measure"] = title
    all_summaries.append(summary_df)

    x = summary_df["group"].tolist()
    means = summary_df["mean"].to_numpy()
    sems = summary_df["sem"].fillna(0.0).to_numpy()
    p_val = summary_df["p_group_difference"].iloc[0]

    bars = ax.bar(x, means, yerr=sems, capsize=6, color=["#4C78A8", "#72B7B2"])
    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel("Baseline Score")

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max != y_min else 1.0
    offset = 0.05 * y_range

    label = f"p={p_val:.3g}" if np.isfinite(p_val) else "p=nan"
    if len(bars) == 2:
        x_center = (bars[0].get_x() + bars[0].get_width() / 2 + bars[1].get_x() + bars[1].get_width() / 2) / 2
    else:
        x_center = 0.0
    ax.text(x_center, y_max + offset, label, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(y_min, y_max + 0.14 * y_range)

fig.suptitle("Baseline Trust Comparison: WEIRD+Young+Knowledgeable vs Non-WEIRD", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dir = Path("./analysis")
out_dir.mkdir(parents=True, exist_ok=True)

plot_path = out_dir / "weird4of6_age_knowledge_baseline_weirdplus_vs_nonweird.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

summary_all_df = pd.concat(all_summaries, ignore_index=True)
summary_path = out_dir / "weird4of6_age_knowledge_baseline_weirdplus_vs_nonweird_summary.csv"
summary_all_df.to_csv(summary_path, index=False)

group_counts = combined.groupby(["Group", "Condition"], as_index=False).size()

print("Baseline comparison summary:")
print(summary_all_df.to_string(index=False))
print("\nGroup sample sizes by condition:")
print(group_counts.to_string(index=False))
print(f"\nSaved baseline plot to: {plot_path}")
print(f"Saved summary table to: {summary_path}")
