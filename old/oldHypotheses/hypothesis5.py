"""Hypothesis 5 tests.

Hypothesis 5: The condition effect on trust change size differs between
WEIRD-like and Non-WEIRD-like groups.

Primary outcome: overall trust change (post - pre).
Primary inferential test: OLS interaction term (Condition x WEIRD Group).
Robustness test: permutation Difference-in-Differences (DID).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA_PATH = Path("./data/Metrics.csv")
OUTPUT_PATH = Path("./plots/hypothesis5_interaction_tests.csv")
ALPHA = 0.05
N_PERMUTATIONS = 10000

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


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
	required = [
		"Condition",
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
	analysis["Condition"] = analysis["Condition"].astype(str).str.strip()
	analysis = analysis[analysis["Condition"].isin(["Interactive", "Text"])].copy()

	analysis["analytical_change"] = (
		analysis["Total Analytical Trust Post"] - analysis["Total Analytical Trust"]
	)
	analysis["emotional_change"] = (
		analysis["Total Emotional Trust Post"] - analysis["Total Emotional Trust"]
	)
	analysis["overall_change"] = analysis["analytical_change"] + analysis["emotional_change"]

	language = analysis["Language"].astype(str).str.strip().str.lower()
	analysis["weird_like"] = analysis["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith(
		"english"
	)
	analysis["WEIRD Group"] = np.where(analysis["weird_like"], "WEIRD-like", "Non-WEIRD-like")
	return analysis


def interaction_ols_test(data: pd.DataFrame) -> dict[str, float]:
	fit_df = data[["overall_change", "Condition", "weird_like"]].dropna().copy()
	y = fit_df["overall_change"].astype(float).to_numpy()
	condition = (fit_df["Condition"] == "Interactive").astype(float).to_numpy()
	group = fit_df["weird_like"].astype(float).to_numpy()
	interaction = condition * group

	X = np.column_stack([np.ones(len(fit_df)), condition, group, interaction])
	beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
	residuals = y - X @ beta
	n, p = X.shape
	df = n - p
	if df <= 0:
		return {
			"interaction_coef": np.nan,
			"interaction_se": np.nan,
			"interaction_t": np.nan,
			"interaction_df": np.nan,
			"interaction_p": np.nan,
		}

	rss = float(np.sum(residuals**2))
	sigma2 = rss / df
	x_inv = np.linalg.inv(X.T @ X)
	cov = sigma2 * x_inv
	se = np.sqrt(np.diag(cov))

	b_interaction = float(beta[3])
	se_interaction = float(se[3])
	t_interaction = b_interaction / se_interaction if not np.isclose(se_interaction, 0) else np.nan
	p_interaction = (
		2 * (1 - stats.t.cdf(abs(t_interaction), df)) if pd.notna(t_interaction) else np.nan
	)

	return {
		"interaction_coef": b_interaction,
		"interaction_se": se_interaction,
		"interaction_t": float(t_interaction),
		"interaction_df": float(df),
		"interaction_p": float(p_interaction),
	}


def did_permutation_test(data: pd.DataFrame, n_permutations: int = N_PERMUTATIONS, seed: int = 42) -> dict[str, float]:
	subset = data[["overall_change", "Condition", "weird_like"]].dropna().copy()
	means = subset.groupby(["weird_like", "Condition"])["overall_change"].mean().unstack()

	observed_did = (
		(means.loc[True, "Interactive"] - means.loc[True, "Text"])
		- (means.loc[False, "Interactive"] - means.loc[False, "Text"])
	)

	rng = np.random.default_rng(seed)
	weird_values = subset["weird_like"].to_numpy()
	extreme = 0

	for _ in range(n_permutations):
		permuted = subset.copy()
		permuted["weird_like"] = rng.permutation(weird_values)
		perm_means = permuted.groupby(["weird_like", "Condition"])["overall_change"].mean().unstack()
		perm_did = (
			(perm_means.loc[True, "Interactive"] - perm_means.loc[True, "Text"])
			- (perm_means.loc[False, "Interactive"] - perm_means.loc[False, "Text"])
		)
		if abs(perm_did) >= abs(observed_did):
			extreme += 1

	p_value = (extreme + 1) / (n_permutations + 1)
	return {
		"did_observed": float(observed_did),
		"did_permutation_p": float(p_value),
	}


def group_condition_summary(data: pd.DataFrame) -> pd.DataFrame:
	return (
		data.groupby(["WEIRD Group", "Condition"])["overall_change"]
		.agg(["count", "mean", "std"])
		.reset_index()
		.rename(columns={"count": "n", "mean": "Overall Change Mean", "std": "Overall Change SD"})
	)


def main() -> None:
	df = pd.read_csv(DATA_PATH)
	analysis = build_analysis_frame(df)

	ols = interaction_ols_test(analysis)
	did = did_permutation_test(analysis)
	summary = group_condition_summary(analysis)

	result = pd.DataFrame(
		[
			{
				"Metric": "Overall Trust Change",
				"Interaction Coef (Condition x WEIRD)": ols["interaction_coef"],
				"Interaction SE": ols["interaction_se"],
				"Interaction t": ols["interaction_t"],
				"Interaction df": ols["interaction_df"],
				"Interaction p": ols["interaction_p"],
				"Interaction Significant (p<0.05)": bool(ols["interaction_p"] < ALPHA)
				if pd.notna(ols["interaction_p"])
				else False,
				"Observed DID": did["did_observed"],
				"Permutation p (DID)": did["did_permutation_p"],
				"Permutation Significant (p<0.05)": bool(did["did_permutation_p"] < ALPHA),
			}
		]
	)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(OUTPUT_PATH, index=False)

	print("Hypothesis 5: Condition x WEIRD interaction on overall trust change")
	print("\nGroup x condition summary:")
	print(summary.to_string(index=False))
	print("\nInferential results:")
	print(result.to_string(index=False))

	row = result.iloc[0]
	if pd.notna(row["Interaction p"]) and row["Interaction p"] < ALPHA:
		interpretation = "indicating a significant condition-by-group interaction"
	else:
		interpretation = "indicating no statistically significant condition-by-group interaction"

	latex_line = (
		"A linear interaction model was conducted to test whether condition had a different effect "
		"on overall trust change for WEIRD-like vs Non-WEIRD-like participants. The interaction term "
		f"($Condition \\times WEIRD$) was $b = {row['Interaction Coef (Condition x WEIRD)']:.2f}$ "
		f"($SE = {row['Interaction SE']:.2f}$), $t({row['Interaction df']:.0f}) = {row['Interaction t']:.2f}$, "
		f"$p {format_p(row['Interaction p'])}$, {interpretation}. A permutation DID robustness test "
		f"yielded observed $DID = {row['Observed DID']:.2f}$ with $p {format_p(row['Permutation p (DID)'])}$."
	)

	print("\nLaTeX narrative line:")
	print(latex_line)
	print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
