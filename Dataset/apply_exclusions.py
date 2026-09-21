"""Build the participant cohort used by all analyses."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "Metrics.csv"
OUTPUT_PATH = ROOT / "Metrics.csv"
EXCLUSIONS_PATH = ROOT / "AnalysisExclusions.csv"

def build_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(INPUT_PATH)
    source["_source_row"] = source.index + 2
    reasons = pd.Series("included", index=source.index, dtype="string")

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal reasons
        reasons.loc[mask & reasons.eq("included")] = reason

    exclude(source["Completed at"].isna(), "missing completion timestamp")

    response_time = pd.to_numeric(source["Time taken"], errors="coerce")
    response_mean = response_time[reasons.eq("included")].mean()
    response_sd = response_time[reasons.eq("included")].std()
    lower_bound = response_mean - (2 * response_sd)
    upper_bound = response_mean + (2 * response_sd)
    exclude(response_time.lt(lower_bound), "response time below mean - 2 SD")
    exclude(response_time.gt(upper_bound), "response time above mean + 2 SD")

    exclusions = source.loc[reasons.ne("included"), ["_source_row", "Submission id", "Condition"]].copy()
    exclusions["reason"] = reasons.loc[exclusions.index].astype(str)

    cohort = source.loc[reasons.eq("included")].drop(columns="_source_row").copy()
    return cohort, exclusions


def main() -> None:
    cohort, exclusions = build_cohort()
    cohort.to_csv(OUTPUT_PATH, index=False)
    exclusions.to_csv(EXCLUSIONS_PATH, index=False)
    print(f"Analysis cohort: {len(cohort)} participants")
    print(f"Excluded records: {len(exclusions)}")
    print(f"Saved exclusions: {EXCLUSIONS_PATH}")


if __name__ == "__main__":
    main()