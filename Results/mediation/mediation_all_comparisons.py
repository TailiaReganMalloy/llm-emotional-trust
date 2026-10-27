"""Supplementary table: mediation of the WEIRD -> (analytical - emotional change) effect for
every pairwise group comparison of each demographic variable (country of origin excluded).

For a pair of groups (A, B) of one demographic variable, only participants in A or B are kept,
M = 1 if the participant is in B (reference group is A), X = WEIRD indicator, and
Y = analytical change - emotional change. The indirect effect a*b is estimated with the same
non-parametric bootstrap used in mediation.py.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import mediation as med

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "mediation_all_comparisons.csv"
TEX_PATH = REPO_ROOT / "Supplementary" / "mediation_table.tex"

MIN_GROUP_N = 10


def age_group(df: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(df["Age"], errors="coerce")
    age = age.fillna(pd.to_numeric(df.get("Age (Demographics)"), errors="coerce"))
    age = age.where((age >= 18) & (age <= 100))
    return pd.cut(age, bins=[17, 24, 34, 44, 100], labels=["18-24", "25-34", "35-44", "45+"]).astype(object)


def demographic_groups(work: pd.DataFrame) -> dict[str, pd.Series]:
    gender = work["Gender"].fillna(work.get("Gender (Demographics)"))
    gender = gender.map(
        lambda v: np.nan
        if pd.isna(v)
        else "Woman"
        if str(v).startswith("Woman")
        else "Man"
        if str(v).startswith("Man")
        else "Non-binary"
        if str(v).startswith("Non-binary")
        else "Rather not say"
    )
    employment = work["Employment status"].replace({"DATA_EXPIRED": np.nan}).map(
        lambda v: "Not in paid work" if str(v).startswith("Not in paid work") else v
    )
    english = work["Language"].map(lambda v: np.nan if pd.isna(v) else "English" if str(v).lower().startswith("english") else "Non-English")
    return {
        "Age group": age_group(work),
        "Gender": gender,
        "AI knowledge": work["AI Knowledge"],
        "Education": work["Education"],
        "Employment status": employment,
        "Ethnicity": work["Ethnicity simplified"],
        "Language": english,
    }


def run_pair(work: pd.DataFrame, groups: pd.Series, ref: str, other: str, rng: np.random.Generator) -> dict | None:
    use = work.assign(_g=groups)
    use = use[use["_g"].isin([ref, other])][["weird_like", "_g", "analytical_vs_emotional_diff"]].dropna()
    n_ref = int((use["_g"] == ref).sum())
    n_other = int((use["_g"] == other).sum())
    if min(n_ref, n_other) < MIN_GROUP_N:
        return None

    x = use["weird_like"].to_numpy(dtype=float)
    m = (use["_g"] == other).to_numpy(dtype=float)
    y = use["analytical_vs_emotional_diff"].to_numpy(dtype=float)
    if np.isclose(x.std(), 0.0):
        return None

    _, a = med.ols_coef(m, x)
    _, c_total = med.ols_coef(y, x)
    _, c_prime, b = med.ols_coef_with_mediator(y, x, m)
    ab, lo, hi, p, t, _ = med.bootstrap_indirect(x, m, y, med.BOOTSTRAP_SAMPLES, rng)
    return {
        "n": len(use),
        "n_ref": n_ref,
        "n_other": n_other,
        "a": a,
        "b": b,
        "c": c_total,
        "c_prime": c_prime,
        "ab": ab,
        "ci_low": lo,
        "ci_high": hi,
        "p": p,
    }


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return adjusted.tolist()


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%")


def main() -> None:
    df = pd.read_csv(med.DATA_PATH)
    work = med.prepare_analysis_frame(df)
    rng = np.random.default_rng(med.RNG_SEED)

    rows = []
    for variable, groups in demographic_groups(work).items():
        categories = [c for c in groups.dropna().unique() if (groups == c).sum() >= MIN_GROUP_N]
        categories = sorted(categories, key=lambda c: -(groups == c).sum())
        for ref, other in itertools.combinations(categories, 2):
            res = run_pair(work, groups, ref, other, rng)
            if res is None:
                continue
            rows.append({"variable": variable, "comparison": f"{other} vs. {ref}", **res})

    table = pd.DataFrame(rows)
    # B-H is applied within each demographic variable (family = comparisons of that variable).
    table["q"] = table.groupby("variable")["p"].transform(lambda p: pd.Series(bh_adjust(p.tolist()), index=p.index))
    table["significant"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    table.to_csv(CSV_PATH, index=False)

    header = r"Variable & Comparison & $n$ & $a$ & $b$ & $ab$ & 95\% CI & $p$ & $q$ \\"
    lines = [
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{longtable}{L{0.12\linewidth}L{0.25\linewidth}R{0.05\linewidth}R{0.065\linewidth}R{0.065\linewidth}R{0.065\linewidth}R{0.16\linewidth}R{0.055\linewidth}R{0.055\linewidth}}",
        r"\caption{Mediation of the effect of WEIRD group on the analytical-versus-emotional trust-change difference for every pairwise comparison of each demographic variable (country of origin excluded). Only participants in the two compared groups are included; $M=1$ for the first group listed and $M=0$ for the second (reference) group. $a$: $X\rightarrow M$; $b$: $M\rightarrow Y\mid X$; $ab$: indirect effect with 95\% bootstrap CI (5000 resamples); $p$: bootstrap $p$-value; $q$: Benjamini-Hochberg adjusted $p$ within each demographic variable. Bold rows have a bootstrap CI that excludes zero.}",
        r"\label{tab:mediation_all}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    previous = None
    for row in table.itertuples():
        if previous is not None and row.variable != previous:
            lines.append(r"\midrule")
        previous = row.variable
        cells = [
            tex_escape(row.variable),
            tex_escape(row.comparison),
            str(row.n),
            fmt(row.a),
            fmt(row.b),
            fmt(row.ab),
            f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]",
            fmt_p(row.p),
            fmt_p(row.q),
        ]
        if row.significant:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", "}", ""]
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Comparisons: {len(table)}; CI excludes zero: {int(table['significant'].sum())}")
    print(f"Saved table: {TEX_PATH}")
    print(f"Saved csv: {CSV_PATH}")


if __name__ == "__main__":
    main()
