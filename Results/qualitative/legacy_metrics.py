from __future__ import annotations

import io
from pathlib import Path
import subprocess

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINED_PATH = REPO_ROOT / "Dataset" / "Combined.csv"
LEGACY_QUALITATIVE_COMMIT = "5e0b564"

AI_ANALYTICAL_TRUST = [
    "AI Deceptive",
    "AI Dishonest",
    "AI Suspicious",
    "AI Wary",
    "AI Harm",
    "AI Confident",
    "AI Security",
    "AI Trustworthy",
    "AI Reliable",
    "AI Trust",
]

AI_ANALYTICAL_TRUST_POST = [
    "AI Deceptive Post",
    "AI Dishonest Post",
    "AI Suspicious Post",
    "AI Wary Post",
    "AI Harm Post",
    "AI Confident Post",
    "AI Security Post",
    "AI Trustworthy Post",
    "AI Reliable Post",
    "AI Trust Post",
]

AI_EMOTIONAL_TRUST = [f"AI systems are {i}" for i in range(1, 10)]
AI_EMOTIONAL_TRUST_POST = [f"AI systems are {i} Post" for i in range(1, 10)]

LIKERT_MAP = {
    "strongly disagree": -2,
    "disagree": -1,
    "neutral": 0,
    "neither agree nor disagree": 0,
    "agree": 1,
    "strongly agree": 2,
}

REVERSE_LIKERT_MAP = {
    "strongly disagree": 2,
    "disagree": 1,
    "neutral": 0,
    "neither agree nor disagree": 0,
    "agree": -1,
    "strongly agree": -2,
}

ANALYTICAL_COLUMN_CANDIDATES = {
    "AI Deceptive": ["AI Deceptive"],
    "AI Dishonest": ["AI Dishonest", "AI Honest"],
    "AI Suspicious": ["AI Suspicious"],
    "AI Wary": ["AI Wary", "AI Weary"],
    "AI Harm": ["AI Harm"],
    "AI Confident": ["AI Confident"],
    "AI Security": ["AI Security"],
    "AI Trustworthy": ["AI Trustworthy"],
    "AI Reliable": ["AI Reliable"],
    "AI Trust": ["AI Trust"],
    "AI Deceptive Post": ["AI Deceptive Post", "AI Wary/Deceptive Post"],
    "AI Dishonest Post": ["AI Dishonest Post", "AI Honest Post"],
    "AI Suspicious Post": ["AI Suspicious Post"],
    "AI Wary Post": ["AI Wary Post", "AI Wary/Deceptive Post", "AI Weary Post"],
    "AI Harm Post": ["AI Harm Post", "AI Harm Post 1", "AI Harm Post 2"],
    "AI Confident Post": ["AI Confident Post"],
    "AI Security Post": ["AI Security Post"],
    "AI Trustworthy Post": ["AI Trustworthy Post"],
    "AI Reliable Post": ["AI Reliable Post"],
    "AI Trust Post": ["AI Trust Post", "AI Trust Post 2"],
}

EMOTIONAL_COLUMN_CANDIDATES = {
    **{f"AI systems are {i}": [f"AI systems are {i}"] for i in range(1, 10)},
    **{
        f"AI systems are {i} Post": [f"AI systems are {i} Post", f"AI systems are Post {i}"]
        for i in range(1, 10)
    },
}

POSITIVE_ANALYTICAL_COLS = {
    "AI Confident",
    "AI Security",
    "AI Trustworthy",
    "AI Reliable",
    "AI Trust",
    "AI Confident Post",
    "AI Security Post",
    "AI Trustworthy Post",
    "AI Reliable Post",
    "AI Trust Post",
}

EMOTIONAL_POLARITY = {
    "AI systems are 1": {"empathetic": 1, "apathetic": -1},
    "AI systems are 2": {"sensitive": 1, "insensitive": -1},
    "AI systems are 3": {"personal": 1, "impersonal": -1},
    "AI systems are 4": {"caring": 1, "ignoring": -1},
    "AI systems are 5": {"altruistic": 1, "self-serving": -1},
    "AI systems are 6": {"cordial": 1, "rude": -1},
    "AI systems are 7": {"responsive": 1, "indifferent": -1},
    "AI systems are 8": {"open-minded": 1, "judgmental": -1},
    "AI systems are 9": {"patient": 1, "impatient": -1},
}


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _coalesce_columns(df: pd.DataFrame, target: str, candidates: list[str]) -> None:
    available = [column for column in candidates if column in df.columns]
    if not available:
        df[target] = pd.NA
        return
    df[target] = df[available].bfill(axis=1).iloc[:, 0]


def _load_historical_metrics_snapshot() -> pd.DataFrame | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{LEGACY_QUALITATIVE_COMMIT}:Dataset/Metrics.csv"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    try:
        snapshot = pd.read_csv(io.StringIO(result.stdout))
    except Exception:
        return None

    if len(snapshot) != 504:
        return None
    return snapshot


def load_legacy_qualitative_frame() -> pd.DataFrame:
    historical_snapshot = _load_historical_metrics_snapshot()
    if historical_snapshot is not None:
        return historical_snapshot

    combined = pd.read_csv(COMBINED_PATH).copy()

    for target, candidates in {**ANALYTICAL_COLUMN_CANDIDATES, **EMOTIONAL_COLUMN_CANDIDATES}.items():
        _coalesce_columns(combined, target, candidates)

    negative_analytical_cols = set(AI_ANALYTICAL_TRUST + AI_ANALYTICAL_TRUST_POST) - POSITIVE_ANALYTICAL_COLS

    for column in POSITIVE_ANALYTICAL_COLS:
        if column in combined.columns:
            combined[column] = combined[column].map(lambda value: LIKERT_MAP.get(_normalize_text(value), 0))

    for column in negative_analytical_cols:
        if column in combined.columns:
            combined[column] = combined[column].map(lambda value: REVERSE_LIKERT_MAP.get(_normalize_text(value), 0))

    for pre_col, polarity in EMOTIONAL_POLARITY.items():
        post_col = f"{pre_col} Post"
        if pre_col in combined.columns:
            combined[pre_col] = combined[pre_col].map(lambda value: polarity.get(_normalize_text(value), 0))
        if post_col in combined.columns:
            combined[post_col] = combined[post_col].map(lambda value: polarity.get(_normalize_text(value), 0))

    combined["Total Analytical Trust"] = combined[AI_ANALYTICAL_TRUST].sum(axis=1)
    combined["Total Analytical Trust Post"] = combined[AI_ANALYTICAL_TRUST_POST].sum(axis=1)
    combined["Total Emotional Trust"] = combined[AI_EMOTIONAL_TRUST].sum(axis=1)
    combined["Total Emotional Trust Post"] = combined[AI_EMOTIONAL_TRUST_POST].sum(axis=1)
    combined["Emotional Trust Difference"] = combined["Total Emotional Trust"] - combined["Total Emotional Trust Post"]
    combined["Analytical Trust Difference"] = combined["Total Analytical Trust"] - combined["Total Analytical Trust Post"]

    return combined