"""Run mediation analyses for selected hypothesis families.

This script focuses on hypotheses retained in the current report structure:
- H1, H2, H3, H4, H5a

For each model, it estimates:
- path a: X -> M
- path b: M -> Y (controlling for X)
- path c: total effect X -> Y
- path c': direct effect X -> Y (controlling for M)
- indirect effect: a*b with bootstrap CI/p-value

Important: these mediation models provide evidence consistent with causal pathways,
but they do not by themselves prove causality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DATA_PATH = Path("./data/Metrics.csv")
OUTPUT_PATH = Path("./plots/mediation_significant_hypotheses.csv")
N_BOOT = 3000
SEED = 42

WESTERN_COUNTRIES = {
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "New Zealand",
    "Ireland",
    "Germany",
    "France",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Austria",
    "Denmark",
    "Sweden",
    "Norway",
    "Finland",
    "Iceland",
    "Luxembourg",
    "Italy",
    "Spain",
    "Portugal",
}


def zscore(series: pd.Series) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if np.isclose(std, 0):
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((series - mean) / std).astype(float)


def _fit_ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    rank = int(np.linalg.matrix_rank(X))
    dof = n - rank

    if dof <= 0:
        se = np.full(beta.shape, np.nan, dtype=float)
        t_vals = np.full(beta.shape, np.nan, dtype=float)
        p_vals = np.full(beta.shape, np.nan, dtype=float)
    else:
        rss = float(np.sum(residuals**2))
        sigma2 = rss / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(cov))
        t_vals = np.divide(beta, se, out=np.full(beta.shape, np.nan), where=~np.isclose(se, 0))
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), dof))

    return {
        "beta": beta,
        "se": se,
        "t": t_vals,
        "p": p_vals,
        "dof": dof,
    }


def _design_matrix(df: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    cols = [np.ones(len(df), dtype=float)]
    for pred in predictors:
        cols.append(df[pred].astype(float).to_numpy())
    return np.column_stack(cols)


def mediation_bootstrap(
    df: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    covars: list[str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    covars = covars or []
    needed = [x, m, y, *covars]
    work = df[needed].dropna().copy()

    # Ensure all modeled columns are numeric for stable OLS fitting.
    for col in needed:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()

    n = len(work)
    if n < 20:
        return {
            "n": n,
            "a": np.nan,
            "a_p": np.nan,
            "b": np.nan,
            "b_p": np.nan,
            "c_total": np.nan,
            "c_total_p": np.nan,
            "c_prime": np.nan,
            "c_prime_p": np.nan,
            "indirect_ab": np.nan,
            "indirect_ci_low": np.nan,
            "indirect_ci_high": np.nan,
            "indirect_p_boot": np.nan,
            "supports_mediation": False,
        }

    # a path: M ~ X + covars
    Xa = _design_matrix(work, [x, *covars])
    ya = work[m].to_numpy(dtype=float)
    fit_a = _fit_ols(ya, Xa)
    a = float(fit_a["beta"][1])
    a_p = float(fit_a["p"][1])

    # b and c' paths: Y ~ X + M + covars
    Xb = _design_matrix(work, [x, m, *covars])
    yb = work[y].to_numpy(dtype=float)
    fit_b = _fit_ols(yb, Xb)
    c_prime = float(fit_b["beta"][1])
    c_prime_p = float(fit_b["p"][1])
    b = float(fit_b["beta"][2])
    b_p = float(fit_b["p"][2])

    # total effect c: Y ~ X + covars
    Xc = _design_matrix(work, [x, *covars])
    yc = work[y].to_numpy(dtype=float)
    fit_c = _fit_ols(yc, Xc)
    c_total = float(fit_c["beta"][1])
    c_total_p = float(fit_c["p"][1])

    indirect = a * b

    rng = np.random.default_rng(seed)
    boot_indirect: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bdf = work.iloc[idx]

        Xa_b = _design_matrix(bdf, [x, *covars])
        ya_b = bdf[m].to_numpy(dtype=float)
        fit_a_b = _fit_ols(ya_b, Xa_b)
        a_b = float(fit_a_b["beta"][1])

        Xb_b = _design_matrix(bdf, [x, m, *covars])
        yb_b = bdf[y].to_numpy(dtype=float)
        fit_b_b = _fit_ols(yb_b, Xb_b)
        b_b = float(fit_b_b["beta"][2])

        if np.isfinite(a_b) and np.isfinite(b_b):
            boot_indirect.append(a_b * b_b)

    if len(boot_indirect) < max(100, int(0.1 * n_boot)):
        ci_low = np.nan
        ci_high = np.nan
        p_boot = np.nan
        supports_mediation = False
    else:
        boot_arr = np.array(boot_indirect, dtype=float)
        ci_low, ci_high = np.percentile(boot_arr, [2.5, 97.5])
        p_boot = 2 * min(np.mean(boot_arr <= 0), np.mean(boot_arr >= 0))
        supports_mediation = bool((ci_low > 0) or (ci_high < 0))

    return {
        "n": n,
        "a": a,
        "a_p": a_p,
        "b": b,
        "b_p": b_p,
        "c_total": c_total,
        "c_total_p": c_total_p,
        "c_prime": c_prime,
        "c_prime_p": c_prime_p,
        "indirect_ab": indirect,
        "indirect_ci_low": float(ci_low),
        "indirect_ci_high": float(ci_high),
        "indirect_p_boot": float(p_boot),
        "supports_mediation": supports_mediation,
    }


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["Condition"] = out["Condition"].astype(str).str.strip()
    out = out[out["Condition"].isin(["Interactive", "Text"])].copy()
    out["condition_bin"] = (out["Condition"] == "Interactive").astype(int)

    language = out["Language"].astype(str).str.strip().str.lower()
    out["weird_like"] = (
        out["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith("english")
    ).astype(int)

    out["emotional_change"] = out["Total Emotional Trust Post"] - out["Total Emotional Trust"]
    out["analytical_change"] = out["Total Analytical Trust Post"] - out["Total Analytical Trust"]
    out["overall_change"] = out["emotional_change"] + out["analytical_change"]
    out["emotional_change_z"] = zscore(out["emotional_change"].astype(float))
    out["analytical_change_z"] = zscore(out["analytical_change"].astype(float))

    pre_rows = pd.DataFrame(
        {
            "time": 0,
            "condition_bin": out["condition_bin"],
            "emotional": out["Total Emotional Trust"],
            "analytical": out["Total Analytical Trust"],
        }
    )
    post_rows = pd.DataFrame(
        {
            "time": 1,
            "condition_bin": out["condition_bin"],
            "emotional": out["Total Emotional Trust Post"],
            "analytical": out["Total Analytical Trust Post"],
        }
    )
    long_df = pd.concat([pre_rows, post_rows], ignore_index=True)

    return out, long_df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    wide_df, long_df = prepare_data(df)

    models: list[dict[str, Any]] = [
        {
            "hypothesis": "H1",
            "analysis": "Time -> Analytical via Emotional (all, condition-adjusted)",
            "data": long_df,
            "x": "time",
            "m": "emotional",
            "y": "analytical",
            "covars": ["condition_bin"],
        },
        {
            "hypothesis": "H2",
            "analysis": "Time -> Analytical via Emotional (Interactive only)",
            "data": long_df[long_df["condition_bin"] == 1],
            "x": "time",
            "m": "emotional",
            "y": "analytical",
            "covars": [],
        },
        {
            "hypothesis": "H2",
            "analysis": "Time -> Analytical via Emotional (Text only)",
            "data": long_df[long_df["condition_bin"] == 0],
            "x": "time",
            "m": "emotional",
            "y": "analytical",
            "covars": [],
        },
        {
            "hypothesis": "H3",
            "analysis": "Condition -> Emotional change (z) via Analytical change (z), WEIRD-like",
            "data": wide_df[wide_df["weird_like"] == 1],
            "x": "condition_bin",
            "m": "analytical_change_z",
            "y": "emotional_change_z",
            "covars": [],
        },
        {
            "hypothesis": "H4",
            "analysis": "Condition -> Emotional change (z) via Analytical change (z), Non-WEIRD-like",
            "data": wide_df[wide_df["weird_like"] == 0],
            "x": "condition_bin",
            "m": "analytical_change_z",
            "y": "emotional_change_z",
            "covars": [],
        },
        {
            "hypothesis": "H5a",
            "analysis": "WEIRD-like -> Emotional change via Analytical change (Text only)",
            "data": wide_df[wide_df["condition_bin"] == 0],
            "x": "weird_like",
            "m": "analytical_change",
            "y": "emotional_change",
            "covars": [],
        },
        {
            "hypothesis": "H5a",
            "analysis": "WEIRD-like -> Emotional change via Analytical change (Interactive only)",
            "data": wide_df[wide_df["condition_bin"] == 1],
            "x": "weird_like",
            "m": "analytical_change",
            "y": "emotional_change",
            "covars": [],
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in models:
        res = mediation_bootstrap(
            df=spec["data"],
            x=spec["x"],
            m=spec["m"],
            y=spec["y"],
            covars=spec["covars"],
            n_boot=N_BOOT,
            seed=SEED,
        )
        rows.append(
            {
                "Hypothesis": spec["hypothesis"],
                "Analysis": spec["analysis"],
                "X": spec["x"],
                "M": spec["m"],
                "Y": spec["y"],
                **res,
            }
        )

    out_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    pd.set_option("display.max_colwidth", 100)
    print("Mediation analyses for hypotheses H1, H2, H3, H4, H5a")
    print(out_df.to_string(index=False))
    print("\nSaved mediation summary:", OUTPUT_PATH)
    print(
        "\nNote: mediation results are suggestive evidence of pathway structure and are "
        "not definitive proof of causality without stronger design assumptions."
    )


if __name__ == "__main__":
    main()