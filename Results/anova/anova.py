from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
REPORT_OUTPUT_PATH = OUTPUT_DIR / "anova.txt"
ALPHA = 0.05

ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0

CONDITION_ORDER = ["Interactive", "Text"]

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

HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}
WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}


def score_weird(row: pd.Series) -> int:
    score = 0
    if row.get("Country of residence") in WESTERN_COUNTRIES:
        score += 1
    if str(row.get("Language", "")).strip().lower().startswith("english"):
        score += 1
    if row.get("Education") in HIGH_EDUCATION:
        score += 1
    if row.get("Employment status") in WEIRD_EMPLOYMENT:
        score += 1
    if str(row.get("Ethnicity simplified", "")).strip().lower() == "white":
        score += 1
    if (
        row.get("Nationality") in WESTERN_COUNTRIES
        or row.get("Country of birth") in WESTERN_COUNTRIES
    ):
        score += 1
    return score


def normalize_condition(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "interactive": "Interactive",
            "text": "Text",
            "static": "Text",
        })
    )


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def effect_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "insufficient data"
    if p_value < ALPHA:
        return "significant"
    return "not significant"


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Condition",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
        "Country of residence",
        "Language",
        "Education",
        "Employment status",
        "Ethnicity simplified",
        "Nationality",
        "Country of birth",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = normalize_condition(work["Condition"])
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Demographic"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")

    work["analytical_pre"] = work["Total Analytical Trust"] / ANALYTICAL_MAX
    work["analytical_post"] = work["Total Analytical Trust Post"] / ANALYTICAL_MAX
    work["emotional_pre"] = work["Total Emotional Trust"] / EMOTIONAL_MAX
    work["emotional_post"] = work["Total Emotional Trust Post"] / EMOTIONAL_MAX

    work["analytical_change"] = work["analytical_post"] - work["analytical_pre"]
    work["emotional_change"] = work["emotional_post"] - work["emotional_pre"]
    work["overall_change"] = (work["analytical_change"] + work["emotional_change"]) / 2.0

    work["Condition"] = pd.Categorical(work["Condition"], categories=CONDITION_ORDER)
    work["Demographic"] = pd.Categorical(work["Demographic"], categories=["WEIRD", "NORMAL"])

    return work


def run_two_way_anova(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    model = ols(f"{outcome} ~ C(Condition) * C(Demographic)", data=frame).fit()
    table = anova_lm(model, typ=2)

    residual_df = float(table.loc["Residual", "df"])
    residual_ss = float(table.loc["Residual", "sum_sq"])

    def extract(term: str) -> dict[str, float]:
        sum_sq = float(table.loc[term, "sum_sq"])
        df_num = float(table.loc[term, "df"])
        f_value = float(table.loc[term, "F"])
        p_value = float(table.loc[term, "PR(>F)"])
        partial_eta_sq = sum_sq / (sum_sq + residual_ss) if (sum_sq + residual_ss) > 0 else np.nan
        return {
            "sum_sq": sum_sq,
            "df_num": df_num,
            "df_den": residual_df,
            "F": f_value,
            "p": p_value,
            "eta_p2": partial_eta_sq,
        }

    return {
        "condition": extract("C(Condition)"),
        "demographic": extract("C(Demographic)"),
        "interaction": extract("C(Condition):C(Demographic)"),
    }


def format_outcome_section(title: str, results: dict[str, dict[str, float]]) -> list[str]:
    lines: list[str] = []
    lines.append(title)

    condition = results["condition"]
    demographic = results["demographic"]
    interaction = results["interaction"]

    lines.append(
        "Condition main effect: "
        f"$F({condition['df_num']:.0f}, {condition['df_den']:.0f})={condition['F']:.3f}$, "
        f"$p {p_text(condition['p'])}$, "
        f"$\\eta_p^2={condition['eta_p2']:.3f}$, "
        f"{effect_label(condition['p'])}."
    )
    lines.append(
        "Demographic main effect (WEIRD vs NORMAL): "
        f"$F({demographic['df_num']:.0f}, {demographic['df_den']:.0f})={demographic['F']:.3f}$, "
        f"$p {p_text(demographic['p'])}$, "
        f"$\\eta_p^2={demographic['eta_p2']:.3f}$, "
        f"{effect_label(demographic['p'])}."
    )
    lines.append(
        "Condition x Demographic interaction: "
        f"$F({interaction['df_num']:.0f}, {interaction['df_den']:.0f})={interaction['F']:.3f}$, "
        f"$p {p_text(interaction['p'])}$, "
        f"$\\eta_p^2={interaction['eta_p2']:.3f}$, "
        f"{effect_label(interaction['p'])}."
    )
    lines.append("")

    return lines


def write_report(frame: pd.DataFrame, overall: dict[str, dict[str, float]], emotional: dict[str, dict[str, float]], analytical: dict[str, dict[str, float]]) -> str:
    n_total = len(frame)
    weird_n = int((frame["Demographic"] == "WEIRD").sum())
    normal_n = int((frame["Demographic"] == "NORMAL").sum())

    lines: list[str] = []
    lines.append("\\subsection{ANOVA: Condition and Demographic Effects}")
    lines.append(
        "A two-way ANOVA tested main effects of condition (Interactive vs. Text), "
        "demographic group (WEIRD vs. NORMAL), and their interaction for trust-change outcomes."
    )
    lines.append(
        f"Sample: $n={n_total}$ ($n_{{WEIRD}}={weird_n}$, $n_{{NORMAL}}={normal_n}$)."
    )
    lines.append("")

    lines.extend(format_outcome_section("Overall Trust Change", overall))
    lines.extend(format_outcome_section("Emotional Trust Change", emotional))
    lines.extend(format_outcome_section("Analytical Trust Change", analytical))

    all_results = {
        "Overall": overall,
        "Emotional": emotional,
        "Analytical": analytical,
    }

    significant_terms: list[str] = []
    for outcome_name, outcome_results in all_results.items():
        for term_name, term_results in outcome_results.items():
            if term_results["p"] < ALPHA:
                significant_terms.append(
                    f"{outcome_name} {term_name.replace('_', ' ')}"
                )

    lines.append("Interpretation summary:")
    if significant_terms:
        lines.append("Significant ANOVA effects were detected for: " + ", ".join(significant_terms) + ".")
    else:
        lines.append(
            "No significant ANOVA main effects or interactions were detected across "
            "overall, emotional, and analytical trust change."
        )

    report_text = "\n".join(lines) + "\n"
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    return report_text


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    frame = build_analysis_frame(df)

    overall = run_two_way_anova(frame, "overall_change")
    emotional = run_two_way_anova(frame, "emotional_change")
    analytical = run_two_way_anova(frame, "analytical_change")

    report = write_report(frame, overall, emotional, analytical)
    print(report)
    print(f"Saved report: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
