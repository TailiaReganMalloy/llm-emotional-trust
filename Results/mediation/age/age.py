from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ANALYSIS_DIR = REPO_ROOT / "Supplementary" / "mediations"
if str(SHARED_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_ANALYSIS_DIR))

from group_change_common import (
    coalesce_numeric_columns,
    run_grouping_analysis,
    run_grouping_condition_effect_analysis,
)


DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "age.txt"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "Figures"
PNG_PATH = FIGURES_DIR / "age.png"
CONDITION_TXT_PATH = OUTPUT_DIR / "age_condition.txt"
CONDITION_PNG_PATH = FIGURES_DIR / "age_condition.png"


def build_age_group(df: pd.DataFrame) -> pd.Series:
    age = coalesce_numeric_columns(df, ["Age (Demographics)", "Age", "Age.1", "Age.2"])
    age = age.where((age >= 18) & (age <= 100), np.nan)

    labels = ["18-24", "25-34", "35-44", "45+"]
    grouped = pd.cut(
        age,
        bins=[17, 24, 34, 44, 100],
        labels=labels,
        include_lowest=True,
    )
    return grouped.astype("string")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    grouped = build_age_group(df)

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Age Group",
        analysis_title="Age-Based Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        preferred_order=["18-24", "25-34", "35-44", "45+"],
        min_group_n_for_tests=8,
        include_condition_effect_tests=True,
        min_n_per_condition=8,
    )

    run_grouping_condition_effect_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Age Group",
        analysis_title="Age-Based Trust-Change Comparison",
        output_png=CONDITION_PNG_PATH,
        output_txt=CONDITION_TXT_PATH,
        preferred_order=["18-24", "25-34", "35-44", "45+"],
        min_n_per_condition=8,
    )


if __name__ == "__main__":
    main()
