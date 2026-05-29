from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TXT_PATH = OUTPUT_DIR / "mediation.txt"
FIGURES_DIR = REPO_ROOT / "Figures"
PNG_PATH = FIGURES_DIR / "mediation.png"

CONDITION_ORDER = ["Interactive", "Text"]
ALPHA = 0.05
BOOTSTRAP_SAMPLES = 5000
RNG_SEED = 42

SUBPLOT_TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14

MEDIATORS = [
    ("western_geo", "Western geographic indicator (residence or origin)"),
    ("english_language", "English language indicator"),
    ("high_education", "High education indicator"),
    ("weird_employment", "WEIRD employment indicator"),
    ("white_ethnicity", "White ethnicity indicator"),
]

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

HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}
WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}


def score_weird(row: pd.Series) -> int:
    score = 0
    # Geographic WEIRD criterion counted once: residence OR origin.
    if (
        row.get("Country of residence") in WESTERN_COUNTRIES
        or row.get("Nationality") in WESTERN_COUNTRIES
        or row.get("Country of birth") in WESTERN_COUNTRIES
    ):
        score += 1
    if str(row.get("Language", "")).lower().startswith("english"):
        score += 1
    if row.get("Education") in HIGH_EDUCATION:
        score += 1
    if row.get("Employment status") in WEIRD_EMPLOYMENT:
        score += 1
    if str(row.get("Ethnicity simplified", "")).strip().lower() == "white":
        score += 1
    return score


@dataclass
class MediationResult:
    mediator_key: str
    mediator_label: str
    n: int
    a_path: float
    b_path: float
    c_total: float
    c_prime: float
    indirect_effect: float
    indirect_t: float
    indirect_ci_low: float
    indirect_ci_high: float
    indirect_p: float
    valid: bool
    note: str


def p_to_stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def p_text(pvalue: float) -> str:
    if pd.isna(pvalue):
        return "NA"
    if pvalue < 0.001:
        return "< .001"
    return f"= {pvalue:.3f}".replace("0.", ".")


def t_text(tvalue: float) -> str:
    if pd.isna(tvalue):
        return "NA"
    return f"{tvalue:.3f}"


def mediation_sig_text(pvalue: float) -> str:
    if pd.isna(pvalue):
        return "Mediation significance unavailable"
    if pvalue < ALPHA:
        return "Significant mediation effect"
    return "No significant mediation effect"


def normalize_condition(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "interactive": "Interactive",
                "text": "Text",
                "static": "Text",
            }
        )
    )
    return normalized


def ols_coef(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    # Returns intercept and slope from y = b0 + b1*x.
    design = np.column_stack([np.ones(len(x), dtype=float), x.astype(float)])
    beta, _, rank, _ = np.linalg.lstsq(design, y.astype(float), rcond=None)
    if rank < 2:
        return np.nan, np.nan
    return float(beta[0]), float(beta[1])


def ols_coef_with_mediator(y: np.ndarray, x: np.ndarray, m: np.ndarray) -> tuple[float, float, float]:
    # Returns intercept, coefficient on x (direct path), and coefficient on m (b path).
    design = np.column_stack([np.ones(len(x), dtype=float), x.astype(float), m.astype(float)])
    beta, _, rank, _ = np.linalg.lstsq(design, y.astype(float), rcond=None)
    if rank < 3:
        return np.nan, np.nan, np.nan
    return float(beta[0]), float(beta[1]), float(beta[2])


def bootstrap_indirect(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, np.ndarray]:
    n = len(x)
    estimates: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        mb = m[idx]
        yb = y[idx]

        _, a = ols_coef(mb, xb)
        _, _, b = ols_coef_with_mediator(yb, xb, mb)
        if pd.notna(a) and pd.notna(b):
            estimates.append(float(a * b))

    if not estimates:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.array([], dtype=float)

    boot = np.array(estimates, dtype=float)
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))

    p_lower = float(np.mean(boot <= 0.0))
    p_upper = float(np.mean(boot >= 0.0))
    p_value = float(2.0 * min(p_lower, p_upper))
    p_value = min(max(p_value, 0.0), 1.0)

    boot_sd = float(np.std(boot, ddof=1)) if len(boot) > 1 else np.nan
    if pd.notna(boot_sd) and not np.isclose(boot_sd, 0.0):
        t_stat = float(np.mean(boot) / boot_sd)
    else:
        t_stat = np.nan

    return float(np.mean(boot)), ci_low, ci_high, p_value, t_stat, boot


def prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Condition",
        "Country of residence",
        "Language",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = normalize_condition(work["Condition"])
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["weird_like"] = (work["weird_score"] >= 4).astype(float)

    # Scale by number of items so analytical and emotional trust deltas are comparable.
    # Keep individual WEIRD-component columns for use as mediator variables.
    language = work["Language"].astype(str).str.strip().str.lower()
    residence = work["Country of residence"].astype(str).str.strip()
    nationality = work["Nationality"].astype(str).str.strip()
    country_of_birth = work["Country of birth"].astype(str).str.strip()

    work["western_geo"] = (
        residence.isin(WESTERN_COUNTRIES)
        | nationality.isin(WESTERN_COUNTRIES)
        | country_of_birth.isin(WESTERN_COUNTRIES)
    ).astype(float)
    work["english_language"] = language.str.startswith("english").astype(float)
    work["high_education"] = work["Education"].isin(HIGH_EDUCATION).astype(float)
    work["weird_employment"] = work["Employment status"].isin(WEIRD_EMPLOYMENT).astype(float)
    work["white_ethnicity"] = (
        work["Ethnicity simplified"].astype(str).str.strip().str.lower() == "white"
    ).astype(float)

    # Scale by number of items so analytical and emotional trust deltas are comparable.
    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["analytical_vs_emotional_diff"] = work["analytical_change"] - work["emotional_change"]

    return work


def run_single_mediation(
    subset: pd.DataFrame,
    condition: str,
    mediator_key: str,
    mediator_label: str,
    rng: np.random.Generator,
) -> MediationResult:
    use = subset[["weird_like", mediator_key, "analytical_vs_emotional_diff"]].dropna().copy()
    n = len(use)

    if n < 20:
        return MediationResult(
            mediator_key=mediator_key,
            mediator_label=mediator_label,
            n=n,
            a_path=np.nan,
            b_path=np.nan,
            c_total=np.nan,
            c_prime=np.nan,
            indirect_effect=np.nan,
            indirect_t=np.nan,
            indirect_ci_low=np.nan,
            indirect_ci_high=np.nan,
            indirect_p=np.nan,
            valid=False,
            note="Insufficient rows for mediation (n < 20).",
        )

    x = use["weird_like"].to_numpy(dtype=float)
    m = use[mediator_key].to_numpy(dtype=float)
    y = use["analytical_vs_emotional_diff"].to_numpy(dtype=float)

    if np.isclose(np.std(x, ddof=0), 0.0):
        return MediationResult(
            mediator_key=mediator_key,
            mediator_label=mediator_label,
            n=n,
            a_path=np.nan,
            b_path=np.nan,
            c_total=np.nan,
            c_prime=np.nan,
            indirect_effect=np.nan,
            indirect_t=np.nan,
            indirect_ci_low=np.nan,
            indirect_ci_high=np.nan,
            indirect_p=np.nan,
            valid=False,
            note="No variance in WEIRD indicator within condition.",
        )

    if np.isclose(np.std(m, ddof=0), 0.0):
        return MediationResult(
            mediator_key=mediator_key,
            mediator_label=mediator_label,
            n=n,
            a_path=np.nan,
            b_path=np.nan,
            c_total=np.nan,
            c_prime=np.nan,
            indirect_effect=np.nan,
            indirect_t=np.nan,
            indirect_ci_low=np.nan,
            indirect_ci_high=np.nan,
            indirect_p=np.nan,
            valid=False,
            note="No variance in mediator within condition.",
        )

    _, a_path = ols_coef(m, x)
    _, c_total = ols_coef(y, x)
    _, c_prime, b_path = ols_coef_with_mediator(y, x, m)

    if pd.isna(a_path) or pd.isna(c_total) or pd.isna(c_prime) or pd.isna(b_path):
        return MediationResult(
            mediator_key=mediator_key,
            mediator_label=mediator_label,
            n=n,
            a_path=np.nan,
            b_path=np.nan,
            c_total=np.nan,
            c_prime=np.nan,
            indirect_effect=np.nan,
            indirect_t=np.nan,
            indirect_ci_low=np.nan,
            indirect_ci_high=np.nan,
            indirect_p=np.nan,
            valid=False,
            note="Singular design matrix prevented coefficient estimation.",
        )

    indirect_mean, ci_low, ci_high, p_value, t_stat, _ = bootstrap_indirect(
        x=x,
        m=m,
        y=y,
        n_boot=BOOTSTRAP_SAMPLES,
        rng=rng,
    )

    return MediationResult(
        mediator_key=mediator_key,
        mediator_label=mediator_label,
        n=n,
        a_path=float(a_path),
        b_path=float(b_path),
        c_total=float(c_total),
        c_prime=float(c_prime),
        indirect_effect=float(indirect_mean),
        indirect_t=float(t_stat),
        indirect_ci_low=float(ci_low),
        indirect_ci_high=float(ci_high),
        indirect_p=float(p_value),
        valid=True,
        note="",
    )


def write_report(results: list[MediationResult]) -> None:
    lines: list[str] = []
    lines.append("\\subsubsection{Mediation Of WEIRD Attributes Across Conditions}")
    lines.append(
        "Across both conditions combined (Interactive and Text), separate mediation models tested whether each WEIRD-defining attribute mediated the association between WEIRD group membership and the analytical-vs-emotional change difference."
    )
    lines.append(
        "Model per test: X = WEIRD indicator, M = attribute indicator, Y = analytical_change - emotional_change; indirect effect estimated by non-parametric bootstrap."
    )
    lines.append("")

    for res in results:
        if not res.valid:
            lines.append(
                f"For the {res.mediator_label.lower()} as a mediator, analysis of (n={res.n}) could not estimate the mediation model because {res.note.lower()}"
            )
            lines.append("")
            continue

        lines.append(
            f"For the {res.mediator_label.lower()} as a mediator, analysis of (n={res.n}) showed an a path (X->M) of {res.a_path:.4f}, "
            f"a b path (M->Y | X) of {res.b_path:.4f}, a c total path (X->Y) of {res.c_total:.4f}, and a c' direct path (X->Y | M) of {res.c_prime:.4f}."
        )
        lines.append(
            f"The estimated indirect effect (a*b) was {res.indirect_effect:.4f}, with a 95% CI of [{res.indirect_ci_low:.4f}, {res.indirect_ci_high:.4f}] and p {p_text(res.indirect_p)}."
        )
        lines.append(
            (
                "This indicates evidence of mediation because the bootstrap confidence interval excludes zero."
                if (res.indirect_ci_low > 0 and res.indirect_ci_high > 0)
                or (res.indirect_ci_low < 0 and res.indirect_ci_high < 0)
                else "This indicates no clear evidence of mediation because the bootstrap confidence interval includes zero."
            )
        )
        lines.append("")

    TXT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def draw_results(results: list[MediationResult]) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    labels = [r.mediator_label for r in results]

    x = np.arange(len(results), dtype=float)
    means = np.array([r.indirect_effect for r in results], dtype=float)
    lows = np.array([r.indirect_ci_low for r in results], dtype=float)
    highs = np.array([r.indirect_ci_high for r in results], dtype=float)

    err_low = np.nan_to_num(means - lows, nan=0.0)
    err_high = np.nan_to_num(highs - means, nan=0.0)

    bar_colors = ["#377eb8", "#c26a26", "#4daf4a", "#984ea3", "#ff7f00"]
    ax.bar(x, means, color=bar_colors[: len(results)], width=0.68, edgecolor="none", zorder=2)
    ax.errorbar(
        x,
        means,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="#3a3a3a",
        elinewidth=2.4,
        capsize=7,
        capthick=2.4,
        zorder=3,
    )

    for idx, res in enumerate(results):
        star = p_to_stars(res.indirect_p)
        if star:
            anchor = highs[idx] if pd.notna(highs[idx]) else means[idx]
            ax.text(idx, anchor + 0.01, star, ha="center", va="bottom", fontsize=19, color="black")

        box_text = (
            f"t={t_text(res.indirect_t)}, p {p_text(res.indirect_p)}\n"
            f"{mediation_sig_text(res.indirect_p)}"
        )
        ax.text(
            idx,
            (highs[idx] if pd.notna(highs[idx]) else means[idx]) + 0.13 * max(np.nanmax(highs) - np.nanmin(lows), 0.05),
            box_text,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "boxstyle": "round,pad=0.25"},
            zorder=5,
        )

        n_text = f"n={res.n}"
        ax.text(idx, ax.get_ylim()[0], n_text, ha="center", va="bottom", fontsize=9, color="#333333")

    ax.axhline(0.0, color="#444444", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_title("All Conditions Combined", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_xlabel("Mediator", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Indirect Effect (a*b)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    fig.suptitle(
        "Mediation Of WEIRD-Defining Attributes On Analytical-Emotional Change Difference",
        fontsize=24,
    )

    ymin = min(np.nanmin([r.indirect_ci_low for r in results if r.valid]), -0.02)
    ymax = max(np.nanmax([r.indirect_ci_high for r in results if r.valid]), 0.02)
    span = max(ymax - ymin, 0.05)
    ax.set_ylim(ymin - 0.15 * span, ymax + 0.65 * span)

    ax.text(
        0.01,
        0.99,
        "Asterisks: * p < .05, ** p < .01, *** p < .001",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
    )

    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    analysis = prepare_analysis_frame(df)

    rng = np.random.default_rng(RNG_SEED)
    results: list[MediationResult] = []
    subset = analysis.copy()
    for mediator_key, mediator_label in MEDIATORS:
        res = run_single_mediation(
            subset=subset,
            condition="All",
            mediator_key=mediator_key,
            mediator_label=mediator_label,
            rng=rng,
        )
        results.append(res)

    write_report(results)
    draw_results(results)

    print(f"Saved report: {TXT_PATH}")
    print(f"Saved figure: {PNG_PATH}")


if __name__ == "__main__":
    main()
