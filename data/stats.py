from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "Metrics.csv"
LATEX_OUTPUT_PATH = Path(__file__).resolve().parent / "weird_normal_gender_table.tex"

CONDITION_ORDER = ["Interactive", "Text"]
GROUP_ORDER = ["WEIRD", "NORMAL"]
GENDER_ORDER = ["Man", "Woman", "NB/GD", "Unknown"]

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


def normalize_condition(series: pd.Series) -> pd.Series:
	mapped = (
		series.astype(str)
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
	return mapped


def normalize_gender_value(value: object) -> str | float:
	if pd.isna(value):
		return np.nan

	text = str(value).strip()
	if not text:
		return np.nan

	lower = text.lower()
	if "non-binary" in lower or "nonbinary" in lower:
		return "NB/GD"
	if "prefer not" in lower or "rather not" in lower:
		return "NB/GD"
	if lower.startswith("woman") or lower == "female":
		return "Woman"
	if lower.startswith("man") or lower == "male":
		return "Man"
	return text


def derive_gender(df: pd.DataFrame) -> pd.Series:
	gender_sources = [
		"Gender",
		"Sex",
		"Gender (Demographics)",
		"Sex (Demographics)",
	]

	combined = pd.Series(pd.NA, index=df.index, dtype="string")
	for col in gender_sources:
		if col not in df.columns:
			continue
		candidate = df[col].astype("string").str.strip()
		candidate = candidate.mask(candidate.eq(""), pd.NA)
		combined = combined.fillna(candidate)

	return combined.map(normalize_gender_value)


def combine_first_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
	out = pd.Series(np.nan, index=df.index, dtype=float)
	for col in columns:
		if col not in df.columns:
			continue
		numeric = pd.to_numeric(df[col], errors="coerce")
		out = out.fillna(numeric)
	return out


def assign_group(df: pd.DataFrame) -> pd.Series:
	language = df["Language"].astype(str).str.strip().str.lower()
	residence = df["Country of residence"].astype(str).str.strip()
	weird_like = residence.isin(WESTERN_COUNTRIES) & language.str.startswith("english")
	return pd.Series(np.where(weird_like, "WEIRD", "NORMAL"), index=df.index)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		df.groupby(["Condition", "Gender", "Group"], dropna=False)
		.agg(n=("PID", "size"), avg_age=("Age Numeric", "mean"))
		.reset_index()
	)
	grouped["total_by_condition_gender"] = grouped.groupby(["Condition", "Gender"])["n"].transform("sum")
	grouped["pct"] = grouped["n"] / grouped["total_by_condition_gender"] * 100.0

	rows: list[dict[str, object]] = []
	for (condition, gender), sub in grouped.groupby(["Condition", "Gender"], sort=False):
		row: dict[str, object] = {
			"Condition": condition,
			"Gender": gender,
			"Total n": int(sub["n"].sum()),
		}

		for group in GROUP_ORDER:
			match = sub[sub["Group"] == group]
			if match.empty:
				row[f"{group} n"] = 0
				row[f"{group} %"] = 0.0
				row[f"{group} Avg Age"] = np.nan
			else:
				row[f"{group} n"] = int(match["n"].iloc[0])
				row[f"{group} %"] = float(match["pct"].iloc[0])
				row[f"{group} Avg Age"] = (
					float(match["avg_age"].iloc[0]) if pd.notna(match["avg_age"].iloc[0]) else np.nan
				)

		rows.append(row)

	summary = pd.DataFrame(rows)
	condition_rank = {cond: i for i, cond in enumerate(CONDITION_ORDER)}
	gender_rank = {gender: i for i, gender in enumerate(GENDER_ORDER)}
	summary["_condition_rank"] = summary["Condition"].map(condition_rank).fillna(999)
	summary["_gender_rank"] = summary["Gender"].map(gender_rank).fillna(999)
	summary = summary.sort_values(["_condition_rank", "_gender_rank", "Gender"]).drop(
		columns=["_condition_rank", "_gender_rank"]
	)
	return summary


def format_age(value: float) -> str:
	if pd.isna(value):
		return "NA"
	return f"{value:.2f}"


def build_latex_table(summary: pd.DataFrame) -> str:
	display = summary.copy()
	display["WEIRD n (%)"] = display.apply(
		lambda row: f"{int(row['WEIRD n'])} ({row['WEIRD %']:.2f}\\%)",
		axis=1,
	)
	display["NORMAL n (%)"] = display.apply(
		lambda row: f"{int(row['NORMAL n'])} ({row['NORMAL %']:.2f}\\%)",
		axis=1,
	)
	display["WEIRD Avg Age"] = display["WEIRD Avg Age"].map(format_age)
	display["NORMAL Avg Age"] = display["NORMAL Avg Age"].map(format_age)

	display = display[
		[
			"Condition",
			"Gender",
			"Total n",
			"WEIRD n (%)",
			"WEIRD Avg Age",
			"NORMAL n (%)",
			"NORMAL Avg Age",
		]
	]

	latex = display.to_latex(
		index=False,
		escape=False,
		caption="WEIRD and NORMAL composition by gender within each condition, with average age.",
		label="tab:weird-normal-gender-condition",
	)
	return latex


def main() -> None:
	data = pd.read_csv(DATA_PATH)

	required = {"PID", "Condition", "Country of residence", "Language"}
	missing = sorted(required - set(data.columns))
	if missing:
		raise KeyError(f"Missing required columns in Metrics.csv: {missing}")

	work = data.copy()
	work["Condition"] = normalize_condition(work["Condition"])
	work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

	work["Gender"] = derive_gender(work).fillna("Unknown")
	work["Age Numeric"] = combine_first_numeric(
		work,
		["Age (Demographics)", "Age", "Age.1", "Age.2"],
	)
	work["Group"] = assign_group(work)

	summary = build_summary(work)
	if summary.empty:
		print("No records found for Interactive/Text conditions.")
		return

	latex_table = build_latex_table(summary)
	LATEX_OUTPUT_PATH.write_text(latex_table, encoding="utf-8")

	print("LaTeX table:")
	print(latex_table)
	print(f"Saved LaTeX table to: {LATEX_OUTPUT_PATH}")


if __name__ == "__main__":
	main()