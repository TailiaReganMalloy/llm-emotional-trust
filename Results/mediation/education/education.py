from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ANALYSIS_DIR = REPO_ROOT / "Supplementary" / "mediations"
if str(SHARED_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_ANALYSIS_DIR))

from group_change_common import (
    collapse_top_categories,
    coalesce_text_columns,
    run_grouping_analysis,
    run_grouping_condition_effect_analysis,
)


DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "education.txt"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "Figures"
PNG_PATH = FIGURES_DIR / "education.png"
CONDITION_TXT_PATH = OUTPUT_DIR / "education_condition.txt"
CONDITION_PNG_PATH = FIGURES_DIR / "education_condition.png"


def normalize_education(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    if "high school" in text or "secondary" in text:
        return "High School"
    if "associate" in text:
        return "Associate"
    if "some college" in text:
        return "Some College"
    if "bachelor" in text:
        return "Bachelor"
    if "master" in text:
        return "Master"
    if "phd" in text or "doctor" in text:
        return "PhD"
    if "graduate professional" in text:
        return "Graduate Professional Degree"

    return str(value).strip().title()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    raw = coalesce_text_columns(df, ["Education", "Education (Demographics)"])
    normalized = raw.map(normalize_education)
    grouped = collapse_top_categories(normalized, max_groups=7, min_count=8, other_label="Other")

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Education",
        analysis_title="Education-Based Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        preferred_order=[
            "High School",
            "Some College",
            "Associate",
            "Bachelor",
            "Master",
            "Graduate Professional Degree",
            "PhD",
            "Other",
        ],
        min_group_n_for_tests=8,
        include_condition_effect_tests=True,
        min_n_per_condition=8,
    )

    run_grouping_condition_effect_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Education",
        analysis_title="Education-Based Trust-Change Comparison",
        output_png=CONDITION_PNG_PATH,
        output_txt=CONDITION_TXT_PATH,
        preferred_order=[
            "High School",
            "Some College",
            "Associate",
            "Bachelor",
            "Master",
            "Graduate Professional Degree",
            "PhD",
            "Other",
        ],
        min_n_per_condition=8,
    )


if __name__ == "__main__":
    main()
