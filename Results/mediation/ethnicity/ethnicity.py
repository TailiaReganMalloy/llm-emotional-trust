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
TXT_PATH = OUTPUT_DIR / "ethnicity.txt"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "Figures"
PNG_PATH = FIGURES_DIR / "ethnicity.png"
CONDITION_TXT_PATH = OUTPUT_DIR / "ethnicity_condition.txt"
CONDITION_PNG_PATH = FIGURES_DIR / "ethnicity_condition.png"


def normalize_ethnicity(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    if "prefer not" in text or "rather not" in text:
        return "Prefer not to say"
    if "white" in text:
        return "White"
    if "asian" in text:
        return "Asian"
    if "black" in text or "african" in text:
        return "Black"
    if "mixed" in text or "multiracial" in text:
        return "Mixed"
    if "hispanic" in text or "latino" in text:
        return "Latino/Hispanic"
    if "middle eastern" in text or "arab" in text:
        return "Middle Eastern/North African"
    if "indigenous" in text or "native" in text:
        return "Indigenous"
    if "other" in text:
        return "Other"

    return str(value).strip().title()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    raw = coalesce_text_columns(
        df,
        [
            "Ethnicity simplified",
            "Ethnicity simplified (Demographics)",
            "Ethnicity",
            "Ethnicity (Demographics)",
        ],
    )
    normalized = raw.map(normalize_ethnicity)
    grouped = collapse_top_categories(normalized, max_groups=8, min_count=8, other_label="Other")

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Ethnicity",
        analysis_title="Ethnicity-Based Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        preferred_order=[
            "White",
            "Asian",
            "Black",
            "Latino/Hispanic",
            "Middle Eastern/North African",
            "Mixed",
            "Indigenous",
            "Other",
            "Prefer not to say",
        ],
        min_group_n_for_tests=8,
        include_condition_effect_tests=True,
        min_n_per_condition=8,
    )

    run_grouping_condition_effect_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Ethnicity",
        analysis_title="Ethnicity-Based Trust-Change Comparison",
        output_png=CONDITION_PNG_PATH,
        output_txt=CONDITION_TXT_PATH,
        preferred_order=[
            "White",
            "Asian",
            "Black",
            "Latino/Hispanic",
            "Middle Eastern/North African",
            "Mixed",
            "Indigenous",
            "Other",
            "Prefer not to say",
        ],
        min_n_per_condition=8,
    )


if __name__ == "__main__":
    main()
