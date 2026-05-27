from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ANALYSIS_DIR = REPO_ROOT / "Supplementary" / "mediations"
if str(SHARED_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_ANALYSIS_DIR))

from group_change_common import (
    coalesce_text_columns,
    run_grouping_analysis,
    run_grouping_condition_effect_analysis,
)


DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "gender.txt"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "Figures"
PNG_PATH = FIGURES_DIR / "gender.png"
CONDITION_TXT_PATH = OUTPUT_DIR / "gender_condition.txt"
CONDITION_PNG_PATH = FIGURES_DIR / "gender_condition.png"


def normalize_gender(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    if text in {"man", "male"} or text.startswith("man"):
        return "Man"
    if text in {"woman", "female"} or text.startswith("woman"):
        return "Woman"
    if "non-binary" in text or "nonbinary" in text:
        return "NB/GD"
    if "prefer not" in text or "rather not" in text:
        return "NB/GD"

    return "Unknown"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    raw_gender = coalesce_text_columns(
        df,
        [
            "Gender",
            "Gender (Demographics)",
            "Sex",
            "Sex (Demographics)",
        ],
    )
    grouped = raw_gender.map(normalize_gender)

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Gender",
        analysis_title="Gender-Based Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        preferred_order=["Man", "Woman", "NB/GD", "Unknown"],
        min_group_n_for_tests=8,
        include_condition_effect_tests=True,
        min_n_per_condition=8,
    )

    run_grouping_condition_effect_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Gender",
        analysis_title="Gender-Based Trust-Change Comparison",
        output_png=CONDITION_PNG_PATH,
        output_txt=CONDITION_TXT_PATH,
        preferred_order=["Man", "Woman", "NB/GD", "Unknown"],
        min_n_per_condition=8,
    )


if __name__ == "__main__":
    main()
