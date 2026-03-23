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


def _normalize_text(value: object) -> str:
	text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
	text = text.replace("judgmental", "judgemental")
	return " ".join(text.split())


EMOTIONAL_VALUE_MAP: dict[str, int] = {}
for mapping in EMOTIONAL_PAIRS.values():
	for key, val in mapping.items():
		EMOTIONAL_VALUE_MAP[_normalize_text(key)] = val


def _to_emotional_numeric(series: pd.Series) -> pd.Series:
	mapped = series.astype(str).map(_normalize_text).map(EMOTIONAL_VALUE_MAP)
	return mapped.fillna(pd.to_numeric(series, errors="coerce"))


participant_deltas = []
for pre_col in EMOTIONAL_PAIRS:
	post_col = f"{pre_col} Post"
	if pre_col not in combined.columns or post_col not in combined.columns:
		continue
	pre = _to_emotional_numeric(combined[pre_col])
	post = _to_emotional_numeric(combined[post_col])
	participant_deltas.append(post - pre)

if not participant_deltas:
	raise ValueError("No emotional pre/post column pairs found in dataset.")

delta_df = pd.concat(participant_deltas, axis=1)
combined["Emotional Impact Delta"] = delta_df.mean(axis=1, skipna=True)

analysis_df = combined[["Condition", "Emotional Impact Delta"]].dropna()
conditions = analysis_df["Condition"].dropna().unique().tolist()
if len(conditions) != 2:
	raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)}: {conditions}")

if "Interactive" in conditions and "Static" in conditions:
	ordered_conditions = ["Interactive", "Static"]
else:
	ordered_conditions = sorted(conditions)

summary_rows: list[dict[str, float | str | int]] = []
means = []
sems = []
p_labels = []

for condition in ordered_conditions:
	values = analysis_df.loc[analysis_df["Condition"] == condition, "Emotional Impact Delta"].to_numpy()
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

	summary_rows.append(
		{
			"Condition": condition,
			"n": int(len(values)),
			"mean_delta": mean_val,
			"sem": sem_val,
			"t_vs_zero": float(t_stat),
			"p_vs_zero": float(p_val),
		}
	)
	means.append(mean_val)
	sems.append(sem_val if np.isfinite(sem_val) else 0.0)
	p_labels.append(p_val)

summary_df = pd.DataFrame(summary_rows)
print("Overall emotional impact by condition (Post - Pre):")
print(summary_df.to_string(index=False))

cond_a_vals = analysis_df.loc[
	analysis_df["Condition"] == ordered_conditions[0], "Emotional Impact Delta"
].to_numpy()
cond_b_vals = analysis_df.loc[
	analysis_df["Condition"] == ordered_conditions[1], "Emotional Impact Delta"
].to_numpy()
if len(cond_a_vals) >= 2 and len(cond_b_vals) >= 2:
	t_between, p_between = stats.ttest_ind(cond_a_vals, cond_b_vals, equal_var=True, nan_policy="omit")
	print(
		f"\nBetween-condition test on overall emotional impact: "
		f"t={t_between:.4f}, p={p_between:.4g}"
	)

out_dir = Path("./analysis")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
bars = plt.bar(ordered_conditions, means, yerr=sems, capsize=6, color=["#4C78A8", "#72B7B2"])
plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
plt.title("Overall Emotional Impact by Condition")
plt.xlabel("Condition")
plt.ylabel("Mean Emotional Impact (Post - Pre)")

ax = plt.gca()
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min if y_max != y_min else 1.0
offset = 0.03 * y_range

for bar, p_val in zip(bars, p_labels):
	height = bar.get_height()
	label = f"p={p_val:.3g}" if np.isfinite(p_val) else "p=nan"
	y = height + offset if height >= 0 else height - offset
	va = "bottom" if height >= 0 else "top"
	plt.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va=va, fontsize=10)

plt.tight_layout()
plot_path = out_dir / "overall_emotional_impact_by_condition.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

summary_path = out_dir / "overall_emotional_impact_by_condition.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved emotional summary to: {summary_path}")
print(f"Saved emotional impact plot to: {plot_path}")

