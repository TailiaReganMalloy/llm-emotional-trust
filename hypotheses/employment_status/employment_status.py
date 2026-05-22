from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESES_ROOT = Path(__file__).resolve().parents[1]
if str(HYPOTHESES_ROOT) not in sys.path:
    sys.path.insert(0, str(HYPOTHESES_ROOT))

from old.oldHypotheses.group_change_common import (
    collapse_top_categories,
    coalesce_text_columns,
    run_grouping_analysis,
    run_grouping_condition_effect_analysis,
)


DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "employment_status.txt"
PNG_PATH = OUTPUT_DIR / "employment_status.png"
CONDITION_TXT_PATH = OUTPUT_DIR / "employment_status_condition.txt"
CONDITION_PNG_PATH = OUTPUT_DIR / "employment_status_condition.png"


def normalize_employment(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    if "full-time" in text or "full time" in text:
        return "Full-Time"
    if "part-time" in text or "part time" in text:
        return "Part-Time"
    if "self-employed" in text:
        return "Self-Employed"
    if "student" in text:
        return "Student"
    if "retired" in text:
        return "Retired"
    if "unemployed" in text:
        return "Unemployed"
    if "not in paid work" in text:
        return "Not in paid work"
    if "due to start" in text:
        return "Starting new job soon"

    return str(value).strip().title()


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    raw = coalesce_text_columns(
        df,
        [
            "Employment status",
            "Employment Status",
            "Employment status (Demographics)",
        ],
    )
    normalized = raw.map(normalize_employment)
    grouped = collapse_top_categories(normalized, max_groups=7, min_count=8, other_label="Other")

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Employment Status",
        analysis_title="Employment-Status Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        preferred_order=[
            "Student",
            "Full-Time",
            "Part-Time",
            "Self-Employed",
            "Unemployed",
            "Retired",
            "Not in paid work",
            "Starting new job soon",
            "Other",
        ],
        min_group_n_for_tests=8,
        include_condition_effect_tests=True,
        min_n_per_condition=8,
    )

    run_grouping_condition_effect_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Employment Status",
        analysis_title="Employment-Status Trust-Change Comparison",
        output_png=CONDITION_PNG_PATH,
        output_txt=CONDITION_TXT_PATH,
        preferred_order=[
            "Student",
            "Full-Time",
            "Part-Time",
            "Self-Employed",
            "Unemployed",
            "Retired",
            "Not in paid work",
            "Starting new job soon",
            "Other",
        ],
        min_n_per_condition=8,
    )


if __name__ == "__main__":
    main()
