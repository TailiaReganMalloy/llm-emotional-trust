from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
FIGURES_DIR = REPO_ROOT / "Figures"
RESULTS_DIR = Path(__file__).resolve().parent

ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0

METRICS_TEXT_COLS: dict[str, str] = {
    "AI Interaction Feeling": "interaction_feeling",
    "Need Model Understanding": "need_model_understanding",
    "Job Screening Feeling": "job_screening_feeling",
    "Explanation Comment": "explanation_comment",
}

WESTERN_COUNTRIES: set[str] = {
    "United States", "United Kingdom", "Canada", "Australia", "New Zealand",
    "Ireland", "Germany", "France", "Netherlands", "Belgium", "Switzerland",
    "Austria", "Denmark", "Sweden", "Norway", "Finland", "Iceland",
    "Luxembourg", "Italy", "Spain", "Portugal",
}
HIGH_EDUCATION: set[str] = {
    "Bachelor", "Master", "Graduate Professional Degree", "PhD",
}
WEIRD_EMPLOYMENT: set[str] = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}

INDIVIDUALISTIC: dict[str, float] = {
    "i": 2.0, "me": 2.0, "my": 2.0, "mine": 2.0, "myself": 2.5,
    "individual": 1.5, "individuality": 1.5, "individualistic": 2.0,
    "personal": 1.2, "privately": 1.2, "private": 1.0,
    "independent": 1.5, "independently": 1.5, "independence": 1.5,
    "autonomous": 1.5, "autonomy": 1.5,
    "self-reliant": 1.5, "self-reliance": 1.5,
    "alone": 1.2, "solo": 1.0,
    "freedom": 1.0, "liberty": 1.0,
    "rights": 1.0, "choice": 0.8,
    "unique": 0.8, "uniqueness": 0.8,
    "achievement": 1.0, "accomplish": 0.8, "accomplishment": 0.8,
    "competition": 1.0, "compete": 1.0, "competitive": 1.0,
    "self": 0.8, "self-interest": 1.5, "self-focused": 1.5,
    "own": 0.6, "ownership": 0.8,
}

COLLECTIVIST: dict[str, float] = {
    "we": 2.0, "us": 2.0, "our": 2.0, "ours": 2.0, "ourselves": 2.5,
    "community": 1.8, "communities": 1.8,
    "society": 1.5, "societal": 1.5, "social": 1.0,
    "group": 1.2, "groups": 1.2,
    "team": 1.2, "teamwork": 1.5,
    "together": 1.5, "collectively": 1.5, "collective": 1.5,
    "shared": 1.2, "share": 1.0, "sharing": 1.2,
    "cooperation": 1.5, "cooperate": 1.5, "cooperative": 1.5,
    "collaboration": 1.5, "collaborate": 1.5, "collaborative": 1.5,
    "solidarity": 1.8,
    "belonging": 1.2, "belong": 1.0,
    "interdependence": 1.8, "interdependent": 1.5,
    "people": 0.8, "everyone": 1.2, "everybody": 1.2,
    "mutual": 1.2, "mutually": 1.2,
    "common": 0.8, "commons": 1.0,
    "harmony": 1.5, "harmonious": 1.2,
    "duty": 1.2, "obligation": 1.2, "responsibility": 0.8,
    "public": 0.8, "citizens": 1.0,
}

WORD_RE = re.compile(r"[a-z']+")

OUTCOMES = ["overall_change", "analytical_change", "emotional_change"]
OUTCOME_LABELS = {
    "overall_change": "Overall",
    "analytical_change": "Analytical",
    "emotional_change": "Emotional",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if np.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(clean.mean())
    if n == 1:
        return mean_val, mean_val, mean_val
    sem = float(clean.std(ddof=1) / np.sqrt(n))
    margin = float(stats.t.ppf((1 + confidence) / 2, n - 1) * sem)
    return mean_val, mean_val - margin, mean_val + margin


def cohens_d_independent(a: pd.Series, b: pd.Series) -> float:
    av = a.dropna().astype(float)
    bv = b.dropna().astype(float)
    if len(av) < 2 or len(bv) < 2:
        return np.nan
    pooled_den = len(av) + len(bv) - 2
    if pooled_den <= 0:
        return np.nan
    pooled_sd = np.sqrt(
        ((len(av) - 1) * float(av.var(ddof=1)) + (len(bv) - 1) * float(bv.var(ddof=1)))
        / pooled_den
    )
    if np.isclose(pooled_sd, 0.0):
        return np.nan
    return float((av.mean() - bv.mean()) / pooled_sd)


def welch_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    av = a.dropna().astype(float)
    bv = b.dropna().astype(float)
    if len(av) < 2 or len(bv) < 2:
        return {
            "n_a": float(len(av)), "n_b": float(len(bv)),
            "mean_a": float(av.mean()) if len(av) else np.nan,
            "mean_b": float(bv.mean()) if len(bv) else np.nan,
            "t": np.nan, "p": np.nan, "d": np.nan,
        }
    t_stat, p_value = ttest_ind(av, bv, equal_var=False, nan_policy="omit")
    return {
        "n_a": float(len(av)), "n_b": float(len(bv)),
        "mean_a": float(av.mean()), "mean_b": float(bv.mean()),
        "t": float(t_stat), "p": float(p_value),
        "d": cohens_d_independent(av, bv),
    }


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def score_ic(text: str) -> float:
    if not text:
        return np.nan
    tokens = WORD_RE.findall(text.lower())
    i_weight = sum(INDIVIDUALISTIC.get(token, 0.0) for token in tokens)
    c_weight = sum(COLLECTIVIST.get(token, 0.0) for token in tokens)
    total = i_weight + c_weight
    if total < 1e-9:
        return 0.0
    return (c_weight - i_weight) / total


def score_weird(row: pd.Series) -> int:
    score = 0
    if row.get("Country of residence") in WESTERN_COUNTRIES:
        score += 1
    if str(row.get("Language", "")).lower().startswith("english"):
        score += 1
    if row.get("Education") in HIGH_EDUCATION:
        score += 1
    if row.get("Employment status") in WEIRD_EMPLOYMENT:
        score += 1
    if str(row.get("Ethnicity simplified", "")).strip().lower() == "white":
        score += 1
    if (
        row.get("Nationality") in WESTERN_COUNTRIES
        or row.get("Country of birth") in WESTERN_COUNTRIES
    ):
        score += 1
    return score


def assign_ic_groups_terciles(series: pd.Series) -> pd.Series:
    result = pd.Series("missing", index=series.index, dtype=object)
    clean = series.dropna().astype(float)
    if clean.empty:
        return result
    ordered = clean.sort_values(kind="mergesort")

    # Rank-based thirds to keep group sizes as equal as possible.
    low_idx, mid_idx, high_idx = np.array_split(ordered.index.to_numpy(), 3)
    result.loc[low_idx] = "individualistic"
    result.loc[mid_idx] = "middle"
    result.loc[high_idx] = "collectivist"
    return result


def build_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["Condition"] = work["Condition"].map(normalize_text).replace("", np.nan)
    work["condition_bin"] = np.where(work["Condition"] == "Interactive", 1.0, 0.0)

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Group"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")
    work["group_bin"] = np.where(work["Group"] == "WEIRD", 1.0, 0.0)

    ic_columns: list[str] = []
    for source_col, suffix in METRICS_TEXT_COLS.items():
        if source_col not in work.columns:
            continue
        score_col = f"{suffix}_ic"
        work[score_col] = work[source_col].map(normalize_text).map(score_ic)
        ic_columns.append(score_col)

    if not ic_columns:
        raise ValueError("No IC source text columns found in Metrics.csv")

    work["ic_mean"] = work[ic_columns].mean(axis=1, skipna=True)

    if {"Total Analytical Trust", "Total Analytical Trust Post"}.issubset(work.columns):
        analytical_raw = (
            pd.to_numeric(work["Total Analytical Trust Post"], errors="coerce") / ANALYTICAL_MAX
            - pd.to_numeric(work["Total Analytical Trust"], errors="coerce") / ANALYTICAL_MAX
        )
    else:
        analytical_raw = pd.to_numeric(work.get("Analytical Trust Difference"), errors="coerce")

    if {"Total Emotional Trust", "Total Emotional Trust Post"}.issubset(work.columns):
        emotional_raw = (
            pd.to_numeric(work["Total Emotional Trust Post"], errors="coerce") / EMOTIONAL_MAX
            - pd.to_numeric(work["Total Emotional Trust"], errors="coerce") / EMOTIONAL_MAX
        )
    else:
        emotional_raw = pd.to_numeric(work.get("Emotional Trust Difference"), errors="coerce")

    work["analytical_change"] = zscore(analytical_raw)
    work["emotional_change"] = zscore(emotional_raw)
    work["overall_change"] = zscore(analytical_raw + emotional_raw)
    work["analytical_minus_emotional_change"] = work["analytical_change"] - work["emotional_change"]

    work["ic_group_full"] = assign_ic_groups_terciles(work["ic_mean"])

    return work


def ancova_ic_on_change(df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    fit_df = df[
        df["ic_mean"].notna()
        & df["analytical_minus_emotional_change"].notna()
        & df["Group"].isin(["WEIRD", "NORMAL"])
        & df["Condition"].isin(["Interactive", "Text"])
    ][[
        "ic_mean",
        "analytical_minus_emotional_change",
        "group_bin",
        "condition_bin",
    ]].copy()

    if len(fit_df) < 12:
        return {
            "n": float(len(fit_df)),
            "coef": np.full(5, np.nan),
            "se": np.full(5, np.nan),
            "t": np.full(5, np.nan),
            "p": np.full(5, np.nan),
            "condition_mean": np.nan,
        }

    y = fit_df["analytical_minus_emotional_change"].to_numpy(dtype=float)
    ic = fit_df["ic_mean"].to_numpy(dtype=float)
    grp = fit_df["group_bin"].to_numpy(dtype=float)
    cond = fit_df["condition_bin"].to_numpy(dtype=float)
    inter = ic * grp

    X = np.column_stack([np.ones(len(fit_df)), ic, grp, cond, inter])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - (X @ beta)
    n_obs, n_params = X.shape
    dof = n_obs - n_params
    if dof <= 0:
        return {
            "n": float(len(fit_df)),
            "coef": beta,
            "se": np.full(5, np.nan),
            "t": np.full(5, np.nan),
            "p": np.full(5, np.nan),
            "condition_mean": float(cond.mean()),
        }

    sigma2 = float(np.sum(resid ** 2) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = beta / se
    p_vals = np.array([2.0 * stats.t.sf(abs(float(tv)), dof) if np.isfinite(tv) else np.nan for tv in t_vals])

    return {
        "n": float(len(fit_df)),
        "coef": beta,
        "se": se,
        "t": t_vals,
        "p": p_vals,
        "condition_mean": float(cond.mean()),
    }


def draw_left_panel_full_sample(ax: plt.Axes, full_df: pd.DataFrame) -> None:
    x = np.arange(len(OUTCOMES), dtype=float)
    bar_width = 0.34

    i_means: list[float] = []
    i_err_low: list[float] = []
    i_err_high: list[float] = []
    c_means: list[float] = []
    c_err_low: list[float] = []
    c_err_high: list[float] = []

    summary_lines = [
        f"n_indiv={(full_df['ic_group_full'] == 'individualistic').sum()}, "
        f"n_coll={(full_df['ic_group_full'] == 'collectivist').sum()}"
    ]

    for outcome in OUTCOMES:
        i_vals = full_df.loc[full_df["ic_group_full"] == "individualistic", outcome]
        c_vals = full_df.loc[full_df["ic_group_full"] == "collectivist", outcome]

        i_mean, i_low, i_high = mean_ci(i_vals)
        c_mean, c_low, c_high = mean_ci(c_vals)

        i_means.append(i_mean)
        i_err_low.append(i_mean - i_low)
        i_err_high.append(i_high - i_mean)
        c_means.append(c_mean)
        c_err_low.append(c_mean - c_low)
        c_err_high.append(c_high - c_mean)

        stat = welch_test(c_vals, i_vals)
        p_value = float(stat["p"])
        summary_lines.append(
            f"{OUTCOME_LABELS[outcome]} C-I: p {format_p_text(p_value)} ({significance_stars(p_value)})"
        )

    ax.bar(
        x - bar_width / 2,
        i_means,
        width=bar_width,
        color="#c26a26",
        label="Individualistic",
        yerr=[i_err_low, i_err_high],
        capsize=4,
        edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2,
        c_means,
        width=bar_width,
        color="#377eb8",
        label="Collectivist",
        yerr=[c_err_low, c_err_high],
        capsize=4,
        edgecolor="white",
    )

    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES], fontsize=11)
    ax.set_title("Full Sample: Trust-Change by IC Orientation", fontsize=14)
    ax.set_ylabel("Normalized trust change (z-score)", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)

    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def draw_middle_panel_counts(ax: plt.Axes, df: pd.DataFrame) -> None:
    grouped = df[df["ic_group_full"].isin(["individualistic", "collectivist"])].copy()

    weird_indiv = int(((grouped["Group"] == "WEIRD") & (grouped["ic_group_full"] == "individualistic")).sum())
    weird_coll = int(((grouped["Group"] == "WEIRD") & (grouped["ic_group_full"] == "collectivist")).sum())
    normal_indiv = int(((grouped["Group"] == "NORMAL") & (grouped["ic_group_full"] == "individualistic")).sum())
    normal_coll = int(((grouped["Group"] == "NORMAL") & (grouped["ic_group_full"] == "collectivist")).sum())

    x = np.array([0.0, 1.0])
    width = 0.36

    ax.bar(
        x - width / 2,
        [weird_indiv, normal_indiv],
        width=width,
        color="#c26a26",
        edgecolor="white",
        label="More individualistic",
    )
    ax.bar(
        x + width / 2,
        [weird_coll, normal_coll],
        width=width,
        color="#377eb8",
        edgecolor="white",
        label="More collectivist",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["WEIRD", "non-WEIRD"], fontsize=11)
    ax.set_ylabel("Participants", fontsize=12)
    ax.set_title("IC Orientation Counts by Demographic Group", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)

    for xpos, yval in zip(x - width / 2, [weird_indiv, normal_indiv]):
        ax.text(xpos, yval + 1.0, str(yval), ha="center", va="bottom", fontsize=9)
    for xpos, yval in zip(x + width / 2, [weird_coll, normal_coll]):
        ax.text(xpos, yval + 1.0, str(yval), ha="center", va="bottom", fontsize=9)


def draw_right_panel_ancova(ax: plt.Axes, df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    model_df = df[
        df["ic_mean"].notna()
        & df["analytical_minus_emotional_change"].notna()
        & df["Group"].isin(["WEIRD", "NORMAL"])
        & df["Condition"].isin(["Interactive", "Text"])
    ].copy()

    ancova = ancova_ic_on_change(model_df)
    beta = ancova["coef"]
    p_vals = ancova["p"]
    condition_mean = float(ancova["condition_mean"])

    weird_df = model_df[model_df["Group"] == "WEIRD"]
    normal_df = model_df[model_df["Group"] == "NORMAL"]

    ax.scatter(
        weird_df["ic_mean"],
        weird_df["analytical_minus_emotional_change"],
        s=18,
        alpha=0.35,
        color="#377eb8",
        label="WEIRD",
    )
    ax.scatter(
        normal_df["ic_mean"],
        normal_df["analytical_minus_emotional_change"],
        s=18,
        alpha=0.35,
        color="#c26a26",
        label="non-WEIRD",
    )

    x_grid = np.linspace(float(model_df["ic_mean"].min()), float(model_df["ic_mean"].max()), 200)

    # Predicted lines at mean condition (covariate-adjusted visualization).
    y_normal = beta[0] + beta[1] * x_grid + beta[3] * condition_mean
    y_weird = beta[0] + beta[1] * x_grid + beta[2] + beta[3] * condition_mean + beta[4] * x_grid

    ax.plot(x_grid, y_weird, color="#1f5fa0", linewidth=2.5)
    ax.plot(x_grid, y_normal, color="#a1541c", linewidth=2.5)

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_title("ANCOVA Fit: IC Score vs Analytical-Emotional Change", fontsize=14)
    ax.set_xlabel("IC score (−1 = individualistic, +1 = collectivist)", fontsize=11)
    ax.set_ylabel("Analytical − Emotional change (z-score)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)

    lines = [
        f"n = {int(ancova['n'])}",
        f"IC main: p {format_p_text(float(p_vals[1]))} ({significance_stars(float(p_vals[1]))})",
        f"Group main: p {format_p_text(float(p_vals[2]))} ({significance_stars(float(p_vals[2]))})",
        f"Condition covariate: p {format_p_text(float(p_vals[3]))} ({significance_stars(float(p_vals[3]))})",
        f"IC x Group: p {format_p_text(float(p_vals[4]))} ({significance_stars(float(p_vals[4]))})",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    return ancova


def make_figure(df: pd.DataFrame, output_path: Path) -> dict[str, float | np.ndarray]:
    full_df = df[df["ic_group_full"].isin(["individualistic", "collectivist"])].copy()

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.8), constrained_layout=True)

    draw_left_panel_full_sample(axes[0], full_df)
    draw_middle_panel_counts(axes[1], df)
    ancova = draw_right_panel_ancova(axes[2], df)

    fig.suptitle("IC Follow-up: Group Contrasts, Counts, and ANCOVA", fontsize=18)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)

    return ancova


def write_report(df: pd.DataFrame, ancova: dict[str, float | np.ndarray], report_path: Path) -> None:
    full_df = df[df["ic_group_full"].isin(["individualistic", "collectivist"])].copy()
    p_vals = ancova["p"]
    beta = ancova["coef"]

    lines: list[str] = []
    lines.append("IC3 analysis summary")
    lines.append("===================")
    lines.append(f"Total participants: {len(df)}")
    lines.append(
        "Full-sample IC split: "
        f"n_individualistic={(full_df['ic_group_full'] == 'individualistic').sum()}, "
        f"n_collectivist={(full_df['ic_group_full'] == 'collectivist').sum()}"
    )
    lines.append("")
    lines.append("Left panel (full-sample trust-change comparisons, Collectivist - Individualistic):")
    for outcome in OUTCOMES:
        c_vals = full_df.loc[full_df["ic_group_full"] == "collectivist", outcome]
        i_vals = full_df.loc[full_df["ic_group_full"] == "individualistic", outcome]
        stat = welch_test(c_vals, i_vals)
        lines.append(
            f"- {OUTCOME_LABELS[outcome]}: p={float(stat['p']):.6f}, d={float(stat['d']):.3f}"
        )

    lines.append("")
    lines.append("Middle panel counts (using full-sample IC split):")
    lines.append(
        "- WEIRD: "
        f"individualistic={int(((full_df['Group'] == 'WEIRD') & (full_df['ic_group_full'] == 'individualistic')).sum())}, "
        f"collectivist={int(((full_df['Group'] == 'WEIRD') & (full_df['ic_group_full'] == 'collectivist')).sum())}"
    )
    lines.append(
        "- non-WEIRD: "
        f"individualistic={int(((full_df['Group'] == 'NORMAL') & (full_df['ic_group_full'] == 'individualistic')).sum())}, "
        f"collectivist={int(((full_df['Group'] == 'NORMAL') & (full_df['ic_group_full'] == 'collectivist')).sum())}"
    )

    lines.append("")
    lines.append("Right panel ANCOVA model:")
    lines.append(
        "Outcome = analytical_minus_emotional_change; predictors = "
        "IC score, Group (WEIRD vs non-WEIRD), Condition (Interactive vs Text), and IC x Group"
    )
    lines.append(f"- n={int(ancova['n'])}")
    lines.append(f"- IC main effect: beta={float(beta[1]):.4f}, p={float(p_vals[1]):.6f}")
    lines.append(f"- Group main effect: beta={float(beta[2]):.4f}, p={float(p_vals[2]):.6f}")
    lines.append(f"- Condition covariate: beta={float(beta[3]):.4f}, p={float(p_vals[3]):.6f}")
    lines.append(f"- IC x Group interaction: beta={float(beta[4]):.4f}, p={float(p_vals[4]):.6f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    df = build_analysis_frame(raw)

    figure_path = FIGURES_DIR / "ic3.png"
    report_path = RESULTS_DIR / "ic3.txt"

    ancova = make_figure(df, figure_path)
    write_report(df, ancova, report_path)

    p_vals = ancova["p"]
    print("IC3 complete")
    print("============")
    print(f"Saved figure: {figure_path}")
    print(f"Saved report: {report_path}")
    print("ANCOVA p-values:")
    print(f"  IC main effect:      {float(p_vals[1]):.6f}")
    print(f"  Group main effect:   {float(p_vals[2]):.6f}")
    print(f"  Condition covariate: {float(p_vals[3]):.6f}")
    print(f"  IC x Group:          {float(p_vals[4]):.6f}")


if __name__ == "__main__":
    main()
