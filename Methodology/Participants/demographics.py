from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "demographics.tex"

WESTERN_COUNTRIES = {
    "United States", "United Kingdom", "Canada", "Australia", "New Zealand",
    "Ireland", "Germany", "France", "Netherlands", "Belgium", "Switzerland",
    "Austria", "Denmark", "Sweden", "Norway", "Finland", "Iceland",
    "Luxembourg", "Italy", "Spain", "Portugal",
}

HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}
WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}


def score_weird(row: pd.Series) -> int:
    score = 0
    if row.get("Country of residence") in WESTERN_COUNTRIES:
        score += 1
    if str(row.get("Language", "")).lower().startswith("english"):
        score += 1
    if row.get("Education") in HIGH_EDUCATION:
        score += 1
    if row.get("Employment status") in WEIRD_EMPLOYMENT:
        score += 1
    if str(row.get("Ethnicity simplified", "")).strip().lower() == "white":
        score += 1
    if (row.get("Nationality") in WESTERN_COUNTRIES
            or row.get("Country of birth") in WESTERN_COUNTRIES):
        score += 1
    return score

CONDITIONS = ["Interactive", "Text"]
GENDER_ROWS = ["Man", "Woman", "NB/GD"]


def classify_gender(value: object) -> str:
    v = str(value).strip().lower() if not (isinstance(value, float) and np.isnan(value)) else ""
    if v.startswith("man") or v == "male":
        return "Man"
    if v.startswith("woman") or v == "female":
        return "Woman"
    return "NB/GD"


def select_age_column(df: pd.DataFrame) -> str:
    for col in ["Age", "Age (Demographics)"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                return col
    raise KeyError("Expected an age column: 'Age (Demographics)' or 'Age'")


def build_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Condition"] = df["Condition"].str.strip()
    df["weird_score"] = df.apply(score_weird, axis=1)
    df["is_weird"] = df["weird_score"] >= 4
    df["Gender_cat"] = df["Gender"].map(classify_gender)
    age_col = select_age_column(df)
    df["Age_numeric"] = pd.to_numeric(df[age_col], errors="coerce")
    return df


def fmt_mean(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.2f}"


def demographics_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for condition in CONDITIONS:
        for gender in GENDER_ROWS:
            subset = df[(df["Condition"] == condition) & (df["Gender_cat"] == gender)]
            total_n = int(len(subset))
            weird_subset = subset[subset["is_weird"]]
            normal_subset = subset[~subset["is_weird"]]
            weird_n = int(len(weird_subset))
            normal_n = int(len(normal_subset))

            weird_pct = (100.0 * weird_n / total_n) if total_n else 0.0
            normal_pct = (100.0 * normal_n / total_n) if total_n else 0.0

            rows.append(
                {
                    "Condition": condition,
                    "Gender": gender,
                    "Total n": str(total_n),
                    "WEIRD n (%)": f"{weird_n} ({weird_pct:.2f}\\%)",
                    "WEIRD Avg Age": fmt_mean(weird_subset["Age_numeric"].mean()),
                    "non-WEIRD n (%)": f"{normal_n} ({normal_pct:.2f}\\%)",
                    "non-WEIRD Avg Age": fmt_mean(normal_subset["Age_numeric"].mean()),
                }
            )
    return rows


def latex_table(rows: list[dict[str, str]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrllll}",
        r"\toprule",
        r"Condition & Gender & Total n & WEIRD n (\%) & WEIRD Avg Age & non-WEIRD n (\%) & non-WEIRD Avg Age \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["Condition"],
                row["Gender"],
                row["Total n"],
                row["WEIRD n (%)"],
                row["WEIRD Avg Age"],
                row["non-WEIRD n (%)"],
                row["non-WEIRD Avg Age"],
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    df = build_df()

    rows = demographics_rows(df)
    output = latex_table(
        rows,
        caption="WEIRD and non-WEIRD composition by gender within each condition, with average age.",
        label="tab:weird-normal-gender-condition",
    ) + "\n"

    OUTPUT_PATH.write_text(output)
    print(f"Saved: {OUTPUT_PATH}")
    print("\nRows:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
