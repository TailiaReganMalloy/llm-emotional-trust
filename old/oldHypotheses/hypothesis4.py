"""Hypothesis 4 tests.

Hypothesis 4: The Non-WEIRD-like group shows a significant difference between
emotional and analytical trust change magnitudes.

Because analytical and emotional totals are on different scales, this script
compares z-scored change values (paired within participant).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


DATA_PATH = Path("./data/Metrics.csv")
OUTPUT_PATH = Path("./plots/hypothesis4_nonweird_type_difference.csv")
ALPHA = 0.05

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


def format_p(p_value: float) -> str:
	if pd.isna(p_value):
		return "NA"
	if p_value < 0.001:
		return "< .001"
	return f"= {p_value:.3f}".replace("0.", ".")


def cohens_d_paired(series_a: pd.Series, series_b: pd.Series) -> float:
	diff = series_a - series_b
	if len(diff) < 2:
		return np.nan
	sd = diff.std(ddof=1)
	if np.isclose(sd, 0):
		return np.nan
	return float(diff.mean() / sd)


def zscore(series: pd.Series) -> pd.Series:
	mean = float(series.mean())
	std = float(series.std(ddof=0))
	if np.isclose(std, 0):
		return pd.Series(0.0, index=series.index, dtype=float)
	return ((series - mean) / std).astype(float)


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
	required = [
		"Country of residence",
		"Language",
		"Total Analytical Trust",
		"Total Analytical Trust Post",
		"Total Emotional Trust",
		"Total Emotional Trust Post",
	]
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise KeyError(f"Missing required columns: {missing}")

	analysis = df.copy()
	analysis["analytical_change"] = (
		analysis["Total Analytical Trust Post"] - analysis["Total Analytical Trust"]
	)
	analysis["emotional_change"] = (
		analysis["Total Emotional Trust Post"] - analysis["Total Emotional Trust"]
	)

	analysis["analytical_change_z"] = zscore(analysis["analytical_change"].astype(float))
	analysis["emotional_change_z"] = zscore(analysis["emotional_change"].astype(float))

	language = analysis["Language"].astype(str).str.strip().str.lower()
	analysis["weird_like"] = analysis["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith(
		"english"
	)
	analysis["WEIRD Group"] = np.where(analysis["weird_like"], "WEIRD-like", "Non-WEIRD-like")
	return analysis


def analyze_nonweird_group(df: pd.DataFrame) -> dict[str, object]:
	subset = df[~df["weird_like"]].copy()
	pair_df = subset[["emotional_change_z", "analytical_change_z"]].dropna()

	emotional = pair_df["emotional_change_z"].astype(float)
	analytical = pair_df["analytical_change_z"].astype(float)

	if len(pair_df) < 2:
		return {
			"Group": "Non-WEIRD-like",
			"n (paired)": int(len(pair_df)),
			"Emotional z Mean": np.nan,
			"Emotional z SD": np.nan,
			"Analytical z Mean": np.nan,
			"Analytical z SD": np.nan,
			"Mean Difference (Emotional z - Analytical z)": np.nan,
			"Paired t": np.nan,
			"df": np.nan,
			"Paired t p": np.nan,
			"Paired t Significant (p<0.05)": False,
			"Wilcoxon": np.nan,
			"Wilcoxon p": np.nan,
			"Wilcoxon Significant (p<0.05)": False,
			"Cohen d (paired)": np.nan,
		}

	t_stat, t_p = ttest_rel(emotional, analytical, nan_policy="omit")
	diffs = emotional - analytical
	if np.allclose(diffs, 0):
		w_stat, w_p = 0.0, 1.0
	else:
		w_stat, w_p = wilcoxon(emotional, analytical, alternative="two-sided")

	return {
		"Group": "Non-WEIRD-like",
		"n (paired)": int(len(pair_df)),
		"Emotional z Mean": float(emotional.mean()),
		"Emotional z SD": float(emotional.std(ddof=1)),
		"Analytical z Mean": float(analytical.mean()),
		"Analytical z SD": float(analytical.std(ddof=1)),
		"Mean Difference (Emotional z - Analytical z)": float((emotional - analytical).mean()),
		"Paired t": float(t_stat),
		"df": int(len(pair_df) - 1),
		"Paired t p": float(t_p),
		"Paired t Significant (p<0.05)": bool(t_p < ALPHA),
		"Wilcoxon": float(w_stat),
		"Wilcoxon p": float(w_p),
		"Wilcoxon Significant (p<0.05)": bool(w_p < ALPHA),
		"Cohen d (paired)": cohens_d_paired(emotional, analytical),
	}


def latex_line(row: pd.Series) -> str:
	if pd.isna(row["Paired t"]):
		return (
			"A paired-samples $t$-test was planned to compare emotional and analytical trust change "
			"within the Non-WEIRD-like group, but there were insufficient paired observations."
		)

	if row["Paired t p"] < ALPHA:
		direction = "higher" if row["Emotional z Mean"] > row["Analytical z Mean"] else "lower"
		comparison = f"was significantly {direction} than"
	else:
		comparison = "was not significantly different from"

	return (
		"A paired-samples $t$-test was conducted to determine if there was a significant difference "
		"between standardized Emotional Trust Change and standardized Analytical Trust Change within "
		"the Non-WEIRD-like group. Results indicated that Emotional Trust Change ($M = "
		f"{row['Emotional z Mean']:.2f}$, $SD = {row['Emotional z SD']:.2f}$) {comparison} "
		"Analytical Trust Change ($M = "
		f"{row['Analytical z Mean']:.2f}$, $SD = {row['Analytical z SD']:.2f}$), "
		f"$t({int(row['df'])}) = {row['Paired t']:.2f}$, $p {format_p(row['Paired t p'])}$, "
		f"$d = {row['Cohen d (paired)']:.2f}$."
	)


def main() -> None:
	df = pd.read_csv(DATA_PATH)
	analysis = build_analysis_frame(df)
	result = pd.DataFrame([analyze_nonweird_group(analysis)])

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(OUTPUT_PATH, index=False)

	print("Hypothesis 4: Non-WEIRD-like type-comparison test")
	print(result.to_string(index=False))
	print("\nLaTeX narrative line:")
	print(latex_line(result.iloc[0]))
	print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
