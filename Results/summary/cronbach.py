from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"

ANALYTICAL_PRE = [
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

ANALYTICAL_POST = [
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

EMOTIONAL_PRE = [f"AI systems are {i}" for i in range(1, 10)]
EMOTIONAL_POST = [f"AI systems are {i} Post" for i in range(1, 10)]

CONDITION_LABELS = {"Interactive": "Interactive", "Text": "Static"}


def cronbach_alpha(df: pd.DataFrame, columns: list[str]) -> float:
    # The analytic file already uses polarity-aligned, reverse-scored items:
    # positively worded trust items are scored in the trust direction, while
    # negatively worded items (e.g., deceptive, dishonest, suspicious, wary, harmful)
    # were reverse-coded before reliability estimation. We therefore compute alpha on
    # the scored item set used in the primary analyses.
    X = df.loc[:, columns].dropna().to_numpy(dtype=float)
    if X.shape[0] == 0 or X.shape[1] <= 1:
        return float("nan")
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    k = X.shape[1]
    return (k / (k - 1)) * (1 - (item_vars.sum() / total_var))


def summarize(df: pd.DataFrame, name: str, columns: list[str]) -> dict[str, float]:
    results = {"overall": cronbach_alpha(df, columns)}
    for cond in ["Interactive", "Text"]:
        sub = df[df["Condition"] == cond]
        results[CONDITION_LABELS[cond]] = cronbach_alpha(sub, columns)
    print(
        f"{name}: overall alpha = {results['overall']:.4f}; "
        f"Interactive = {results['Interactive']:.4f}; Static = {results['Static']:.4f}"
    )
    return results


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Analytic sample: N = {len(df)}")
    summarize(df, "analytical_pre", ANALYTICAL_PRE)
    summarize(df, "analytical_post", ANALYTICAL_POST)
    summarize(df, "emotional_pre", EMOTIONAL_PRE)
    summarize(df, "emotional_post", EMOTIONAL_POST)


if __name__ == "__main__":
    main()
