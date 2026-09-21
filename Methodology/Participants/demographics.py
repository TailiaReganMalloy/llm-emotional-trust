from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "demographics.tex"
REVIEW_RESPONSES = r"""2.1  Describe participant exclusion criteria more explicitly.
\begin{itemize}
\item[Response:] Records without a completion timestamp were removed before response-time analysis ($n=49$). Among the remaining completed records, response time had a mean of 3,633.25 seconds and a sample standard deviation of 51,083.46 seconds. Participants were then excluded when response time fell outside the mean $\pm$ two standard deviations, corresponding to an upper cutoff of 105,800.17 seconds; one approved Interactive record with a response time of 1,089,866 seconds (approximately 12.6 days) was excluded as an outlier. No below-lower-bound outliers were identified. No other participant-level exclusion criteria were applied. The full flow is reported in Section~3.1.2 and Figure~\ref{fig:participant-flow}.
\end{itemize}

2.2  Clarify randomization procedures for participant assignment.
\begin{itemize}
\item[Response:] Participants were analyzed in the Interactive and Text conditions recorded in the compiled study data. The repository does not preserve a randomization seed or document the assignment mechanism, so no stronger claim about the randomization procedure is made here; the resulting condition counts and exclusions are shown in Section~3.1.2 and Figure~\ref{fig:participant-flow}.
\end{itemize}

2.3  Describe data preprocessing and missing-data handling.
\begin{itemize}
\item[Response:] The source exports were merged and trust responses were mapped to the analysis scales by the dataset preprocessing script. Records missing a completion timestamp were removed before calculating response-time bounds, and the one completed response-time outlier beyond the mean $\pm$ two standard deviations was removed. Other missing values were retained in the analysis dataset and omitted only when a particular statistical test required the corresponding variable.
\end{itemize}

2.4  State software versions and package dependencies used for statistical analyses.
\begin{itemize}
\item[Response:] Analyses were implemented in Python using the dependencies listed in \texttt{requirements.txt}: pandas (\textgreater=1.3), NumPy (\textgreater=1.21), SciPy (\textgreater=1.7), statsmodels (\textgreater=0.14), seaborn (\textgreater=0.12), matplotlib (\textgreater=3.5), Jinja2 (\textgreater=3.1), and vaderSentiment (\textgreater=3.3.2). The repository records minimum supported package versions rather than a frozen environment file.
\end{itemize}

2.5  Consider including a preregistered protocol or clearly explain why preregistration was not performed.
\begin{itemize}
\item[Response:] No preregistered protocol is included in the repository. The study materials, preprocessing scripts, analysis scripts, and generated outputs are retained in the project for transparency, and the results should therefore be interpreted in light of the absence of a preregistration record.
\end{itemize}

\subsubsection{Participant flow}\label{sec:participant-flow}
The compiled dataset contained 504 records. Forty-nine records without a completion timestamp were removed before response-time analysis, leaving 455 completed records. One completed record, an approved Interactive submission with a response time of 1,089,866 seconds, exceeded the upper mean-plus-two-standard-deviations cutoff and was excluded. The final analysis cohort comprised 454 participants: 223 in the Interactive condition and 231 in the Text condition. Figure~\ref{fig:participant-flow} summarizes this flow.

\begin{figure}[htbp]
\centering
\fbox{\begin{minipage}{0.78\linewidth}
\centering
	extbf{Compiled records: $n=504$}\\[0.5em]
$\downarrow$\\[0.5em]
Records without completion timestamp excluded: $n=49$\\[0.5em]
$\downarrow$\\[0.5em]
Completed records assessed for response-time outliers: $n=455$\\[0.5em]
$\downarrow$\\[0.5em]
Response-time outlier excluded: $n=1$\\[0.5em]
$\downarrow$\\[0.5em]
Final analysis cohort: $n=454$\\
Interactive: $n=223$; Text: $n=231$
\end{minipage}}
\caption{Participant flow from the compiled dataset to the final analysis cohort.}
\label{fig:participant-flow}
\end{figure}
"""
EXCLUSION_DESCRIPTION = (
    "Records without a completion timestamp were removed before response-time analysis "
    "($n=49$). Among the remaining completed records, response time had a mean of 3,633.25 "
    "seconds and a sample standard deviation of 51,083.46 seconds. Participants were then "
    "excluded when response time fell outside the mean $\\pm$ two standard deviations, "
    "corresponding to an upper cutoff of 105,800.17 seconds; one approved Interactive "
    "record with a response time of 1,089,866 seconds (approximately 12.6 days) was "
    "excluded as an outlier. No below-lower-bound outliers were identified."
)

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
    df["Gender_cat"] = derive_gender(df)
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
    output = REVIEW_RESPONSES + "\n\n" + EXCLUSION_DESCRIPTION + "\n\n" + output

    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nRows:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
