from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ANALYSIS_DIR = REPO_ROOT / "Supplementary" / "hypotheses"
if str(SHARED_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_ANALYSIS_DIR))

from group_change_common import (
    coalesce_text_columns,
    run_grouping_analysis,
    run_grouping_condition_effect_analysis,
)


DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "ai_knowledge.txt"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "Figures"
PNG_PATH = FIGURES_DIR / "ai_knowledge.png"
CONDITION_TXT_PATH = OUTPUT_DIR / "ai_knowledge_condition.txt"
CONDITION_PNG_PATH = FIGURES_DIR / "ai_knowledge_condition.png"


def normalize_ai_knowledge(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    if "no knowledge" in text:
        return "No knowledge"
    if "beginner" in text:
        return "Beginner"
    if "conceptual" in text:
        return "Conceptual"
    if "advanced" in text:
        return "Advanced"
    if "expert" in text:
        return "Expert"

    return str(value).strip().title()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    raw = coalesce_text_columns(df, ["AI Knowledge", "AI Knowledge (Demographics)"])
    grouped = raw.map(normalize_ai_knowledge)

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="AI Knowledge",
        analysis_title="AI-Knowledge Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        preferred_order=["No knowledge", "Beginner", "Conceptual", "Advanced", "Expert"],
        min_group_n_for_tests=8,
        include_condition_effect_tests=True,
        min_n_per_condition=8,
    )

    run_grouping_condition_effect_analysis(
        df=df,
        group_series=grouped,
        grouping_label="AI Knowledge",
        analysis_title="AI-Knowledge Trust-Change Comparison",
        output_png=CONDITION_PNG_PATH,
        output_txt=CONDITION_TXT_PATH,
        preferred_order=["No knowledge", "Beginner", "Conceptual", "Advanced", "Expert"],
        min_n_per_condition=8,
    )


if __name__ == "__main__":
    main()
