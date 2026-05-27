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


def classify_gender(value: object) -> str:
    v = str(value).strip().lower() if not (isinstance(value, float) and np.isnan(value)) else ""
    if "man (including trans male" in v or v == "man":
        return "Male"
    if "woman (including trans female" in v or v == "woman":
        return "Female"
    return "Non-binary / Not specified"


def build_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Condition"] = df["Condition"].str.strip()
    df["weird_score"] = df.apply(score_weird, axis=1)
    df["is_weird"] = df["weird_score"] >= 4
    df["Group"] = np.where(df["is_weird"], "WEIRD", "Normal")
    df["Gender_cat"] = df["Gender"].map(classify_gender)
    return df


def counts_table(df: pd.DataFrame, category_col: str, categories: list[str]) -> pd.DataFrame:
    rows = {}
    for cat in categories:
        row: dict[str, int] = {}
        for cond in CONDITIONS:
            n = int(((df["Condition"] == cond) & (df[category_col] == cat)).sum())
            row[cond] = n
        row["Total"] = int((df[category_col] == cat).sum())
        rows[cat] = row
    result = pd.DataFrame(rows).T
    result.loc["Total"] = result.sum()
    return result[CONDITIONS + ["Total"]]


def latex_table(df_counts: pd.DataFrame, caption: str, label: str) -> str:
    n_cols = len(df_counts.columns)
    col_spec = "l" + "r" * n_cols
    col_header = " & ".join(df_counts.columns.tolist())
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        f"\\textbf{{Group}} & {col_header} \\\\",
        r"\hline",
    ]
    for i, (row_label, row) in enumerate(df_counts.iterrows()):
        if row_label == "Total":
            lines.append(r"\hline")
        values = " & ".join(str(int(v)) for v in row)
        lines.append(f"{row_label} & {values} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    df = build_df()

    gender_cats = ["Male", "Female", "Non-binary / Not specified"]
    gender_counts = counts_table(df, "Gender_cat", gender_cats)

    group_cats = ["WEIRD", "Normal"]
    group_counts = counts_table(df, "Group", group_cats)

    gender_tex = latex_table(
        gender_counts,
        caption="Participant gender breakdown by experiment condition.",
        label="tab:demographics_gender",
    )
    group_tex = latex_table(
        group_counts,
        caption="Participant WEIRD/Normal population breakdown by experiment condition.",
        label="tab:demographics_group",
    )

    output = "\n\n".join([gender_tex, group_tex]) + "\n"
    OUTPUT_PATH.write_text(output)
    print(f"Saved: {OUTPUT_PATH}")
    print("\nGender counts:")
    print(gender_counts.to_string())
    print("\nGroup counts:")
    print(group_counts.to_string())


if __name__ == "__main__":
    main()
