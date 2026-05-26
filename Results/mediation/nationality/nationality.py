from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ANALYSIS_DIR = REPO_ROOT / "Supplementary" / "hypotheses"
if str(SHARED_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_ANALYSIS_DIR))

from group_change_common import collapse_top_categories, coalesce_text_columns, run_grouping_analysis


DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "nationality.txt"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "Figures"
PNG_PATH = FIGURES_DIR / "nationality.png"


def normalize_nationality(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    aliases = {
        "us": "United States",
        "u.s.": "United States",
        "u.s.a": "United States",
        "usa": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }

    if text in aliases:
        return aliases[text]

    return str(value).strip().title()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    raw = coalesce_text_columns(df, ["Nationality", "Nationality (Demographics)"])
    normalized = raw.map(normalize_nationality)
    grouped = collapse_top_categories(normalized, max_groups=8, min_count=8, other_label="Other")

    run_grouping_analysis(
        df=df,
        group_series=grouped,
        grouping_label="Nationality",
        analysis_title="Nationality-Based Trust-Change Comparison",
        output_png=PNG_PATH,
        output_txt=TXT_PATH,
        min_group_n_for_tests=8,
    )


if __name__ == "__main__":
    main()
