"""ic1.py – Individualism-Collectivism overview panels.

Mirrors the four-panel layout of sentiment1.py but replaces VADER sentiment
scoring with a lexicon-based Individualism-Collectivism (IC) score.

Score convention
----------------
  -1.0  →  maximally individualistic language
   0.0  →  neutral / no IC-marked words
  +1.0  →  maximally collectivist language

Formula
-------
For each text the script counts weighted occurrences of individualistic (I-lexicon)
and collectivist (C-lexicon) terms after lowercasing and tokenising with [a-z']+.

    raw = C_weight_sum − I_weight_sum
    norm = I_weight_sum + C_weight_sum          (denominator)

If norm < 1e-9 (no marked words found) the raw score is 0.0 (neutral).
Otherwise  score = raw / norm  which naturally lives in [−1, +1].

Participant-level ic_mean is the mean across the four free-text fields, skipna.

Panels
------
1. Histogram of participant ic_mean scores.
2. Condition bar chart (Interactive vs Text/Static).
3. Demographic-group bar chart (WEIRD vs Normal).
4. Group × Condition interaction bar chart.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
FIGURES_DIR = REPO_ROOT / "Figures"
RESULTS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# IC Lexicons
# Weights reflect the directional strength of each marker.
# Pronouns are higher weight (most reliable linguistic signal).
# Value/concept words are weighted somewhat lower as they are ambiguous.
# ---------------------------------------------------------------------------

# Words that signal an individualistic orientation
INDIVIDUALISTIC: dict[str, float] = {
    # first-person singular pronouns
    "i": 2.0, "me": 2.0, "my": 2.0, "mine": 2.0, "myself": 2.5,
    # autonomy & independence values
    "individual": 1.5, "individuality": 1.5, "individualistic": 2.0,
    "personal": 1.2, "privately": 1.2, "private": 1.0,
    "independent": 1.5, "independently": 1.5, "independence": 1.5,
    "autonomous": 1.5, "autonomy": 1.5,
    "self-reliant": 1.5, "self-reliance": 1.5,
    "alone": 1.2, "myself": 1.5, "solo": 1.0,
    "freedom": 1.0, "liberty": 1.0,
    "rights": 1.0, "choice": 0.8,
    "unique": 0.8, "uniqueness": 0.8,
    "achievement": 1.0, "accomplish": 0.8, "accomplishment": 0.8,
    "competition": 1.0, "compete": 1.0, "competitive": 1.0,
    "self": 0.8, "self-interest": 1.5, "self-focused": 1.5,
    "own": 0.6, "ownership": 0.8,
}

# Words that signal a collectivist orientation
COLLECTIVIST: dict[str, float] = {
    # first-person plural pronouns
    "we": 2.0, "us": 2.0, "our": 2.0, "ours": 2.0, "ourselves": 2.5,
    # community & group values
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


# ---------------------------------------------------------------------------
# Utility helpers (mirrored from sentiment1.py)
# ---------------------------------------------------------------------------

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
    var_a, var_b = float(av.var(ddof=1)), float(bv.var(ddof=1))
    pooled_den = len(av) + len(bv) - 2
    if pooled_den <= 0:
        return np.nan
    pooled_sd = np.sqrt(((len(av) - 1) * var_a + (len(bv) - 1) * var_b) / pooled_den)
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


def fit_condition_group_interaction(
    df: pd.DataFrame, outcome_col: str = "ic_mean"
) -> dict[str, float]:
    fit_df = df[
        df["Condition"].isin(["Interactive", "Text"])
        & df["Group"].isin(["WEIRD", "NORMAL"])
    ][[outcome_col, "Condition", "Group"]].dropna()

    if len(fit_df) < 8:
        return {"n": float(len(fit_df)), "coef_interaction": np.nan,
                "t_interaction": np.nan, "p_interaction": np.nan}

    y_vec = fit_df[outcome_col].to_numpy(dtype=float)
    condition_bin = (fit_df["Condition"] == "Interactive").to_numpy(dtype=float)
    group_bin = (fit_df["Group"] == "WEIRD").to_numpy(dtype=float)
    interaction = condition_bin * group_bin
    X = np.column_stack([np.ones(len(fit_df)), condition_bin, group_bin, interaction])

    beta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
    resid = y_vec - X @ beta
    n_obs, n_params = X.shape
    dof = n_obs - n_params
    if dof <= 0:
        return {"n": float(len(fit_df)), "coef_interaction": np.nan,
                "t_interaction": np.nan, "p_interaction": np.nan}

    sigma2 = float(np.sum(resid ** 2) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    if np.isclose(float(se[3]), 0.0):
        t_inter, p_inter = np.nan, np.nan
    else:
        t_inter = float(beta[3] / se[3])
        p_inter = float(2.0 * stats.t.sf(abs(t_inter), dof))

    return {
        "n": float(len(fit_df)),
        "coef_interaction": float(beta[3]),
        "t_interaction": t_inter,
        "p_interaction": p_inter,
    }


# ---------------------------------------------------------------------------
# IC scoring
# ---------------------------------------------------------------------------

def score_ic(text: str) -> float:
    """Score a single text on the IC dimension.

    Returns a value in [−1, +1]:
        negative → individualistic
        positive → collectivist
        0.0      → neutral or no IC-relevant words found
    Returns np.nan for empty/missing text.
    """
    if not text:
        return np.nan
    tokens = WORD_RE.findall(text.lower())
    i_weight = sum(INDIVIDUALISTIC.get(t, 0.0) for t in tokens)
    c_weight = sum(COLLECTIVIST.get(t, 0.0) for t in tokens)
    total = i_weight + c_weight
    if total < 1e-9:
        return 0.0  # no IC-relevant language; treat as neutral
    return (c_weight - i_weight) / total


# ---------------------------------------------------------------------------
# WEIRD scoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Build analysis frame
# ---------------------------------------------------------------------------

def build_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["Condition"] = work["Condition"].map(normalize_text).replace("", np.nan)

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["is_weird"] = work["weird_score"] >= 4
    work["Group"] = np.where(work["is_weird"], "WEIRD", "NORMAL")

    ic_columns: list[str] = []
    for source_col, suffix in METRICS_TEXT_COLS.items():
        if source_col not in work.columns:
            continue
        text_col = f"{suffix}_text"
        score_col = f"{suffix}_ic"
        work[text_col] = work[source_col].map(normalize_text)
        work[score_col] = work[text_col].map(score_ic)
        ic_columns.append(score_col)

    if not ic_columns:
        raise ValueError("No text source columns found in Metrics.csv")

    work["ic_mean"] = work[ic_columns].mean(axis=1, skipna=True)

    # Trust-change outcomes (z-scored)
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

    return work


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_ic_overview_panels(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["ic_mean"].notna()].copy()

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes

    # --- Panel 0: Histogram of ic_mean ---
    ax0.hist(
        plot_df["ic_mean"], bins=24,
        color="#4e79a7", edgecolor="white", alpha=0.9,
    )
    ax0.axvline(float(plot_df["ic_mean"].mean()), color="#1f2937", linewidth=2, linestyle="--")
    ax0.axvline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax0.set_title("Participant Mean IC Score", fontsize=14)
    ax0.set_xlabel("IC score  (−1 = individualistic, +1 = collectivist)", fontsize=10)
    ax0.set_ylabel("Participants", fontsize=11)

    # --- Panel 1: Condition comparison ---
    condition_order = ["Interactive", "Text"]
    cond_means, cond_err_low, cond_err_high = [], [], []
    for cond in condition_order:
        m, lo, hi = mean_ci(plot_df.loc[plot_df["Condition"] == cond, "ic_mean"])
        cond_means.append(m)
        cond_err_low.append(m - lo)
        cond_err_high.append(hi - m)

    cond_test = welch_test(
        plot_df.loc[plot_df["Condition"] == "Interactive", "ic_mean"],
        plot_df.loc[plot_df["Condition"] == "Text", "ic_mean"],
    )

    x_cond = np.arange(len(condition_order), dtype=float)
    ax1.bar(
        x_cond, cond_means,
        yerr=[cond_err_low, cond_err_high],
        capsize=4, color=["#377eb8", "#c26a26"], edgecolor="white",
    )
    ax1.axhline(0.0, color="#333333", linewidth=1.0)
    ax1.set_xticks(x_cond)
    ax1.set_xticklabels(condition_order)
    ax1.set_title("Condition Difference in Mean IC Score", fontsize=14)
    ax1.set_ylabel("Mean IC score", fontsize=11)
    ax1.text(
        0.02, 0.98,
        (
            f"p {format_p_text(float(cond_test['p']))} ({significance_stars(float(cond_test['p']))})\n"
            f"d = {float(cond_test['d']):.2f}"
        ),
        transform=ax1.transAxes, ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    # --- Panel 2: Demographic group comparison ---
    group_order = ["WEIRD", "NORMAL"]
    group_labels_display = ["WEIRD", "Normal"]
    grp_means, grp_err_low, grp_err_high = [], [], []
    for grp in group_order:
        m, lo, hi = mean_ci(plot_df.loc[plot_df["Group"] == grp, "ic_mean"])
        grp_means.append(m)
        grp_err_low.append(m - lo)
        grp_err_high.append(hi - m)

    group_test = welch_test(
        plot_df.loc[plot_df["Group"] == "WEIRD", "ic_mean"],
        plot_df.loc[plot_df["Group"] == "NORMAL", "ic_mean"],
    )

    x_grp = np.arange(len(group_order), dtype=float)
    ax2.bar(
        x_grp, grp_means,
        yerr=[grp_err_low, grp_err_high],
        capsize=4, color=["#377eb8", "#c26a26"], edgecolor="white",
    )
    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.set_xticks(x_grp)
    ax2.set_xticklabels(group_labels_display)
    ax2.set_title("Demographic Group Difference in Mean IC Score", fontsize=14)
    ax2.set_ylabel("Mean IC score", fontsize=11)
    ax2.text(
        0.02, 0.98,
        (
            f"p {format_p_text(float(group_test['p']))} ({significance_stars(float(group_test['p']))})\n"
            f"d = {float(group_test['d']):.2f}"
        ),
        transform=ax2.transAxes, ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    # --- Panel 3: Group × Condition interaction ---
    condition_colors = {"Interactive": "#377eb8", "Text": "#c26a26"}
    bar_width = 0.34
    for offset, cond in zip([-bar_width / 2, bar_width / 2], condition_order):
        means, err_low, err_high = [], [], []
        for grp in group_order:
            m, lo, hi = mean_ci(
                plot_df.loc[
                    (plot_df["Group"] == grp) & (plot_df["Condition"] == cond), "ic_mean"
                ]
            )
            means.append(m)
            err_low.append(m - lo)
            err_high.append(hi - m)
        ax3.bar(
            x_grp + offset, means, width=bar_width,
            yerr=[err_low, err_high], capsize=4,
            color=condition_colors[cond], edgecolor="white", label=cond,
        )

    interaction_test = fit_condition_group_interaction(plot_df, "ic_mean")
    ax3.axhline(0.0, color="#333333", linewidth=1.0)
    ax3.set_xticks(x_grp)
    ax3.set_xticklabels(group_labels_display)
    ax3.set_title("Group × Condition Interaction", fontsize=14)
    ax3.set_ylabel("Mean IC score", fontsize=11)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.text(
        0.02, 0.98,
        (
            f"interaction p {format_p_text(float(interaction_test['p_interaction']))} "
            f"({significance_stars(float(interaction_test['p_interaction']))})\n"
            f"beta = {float(interaction_test['coef_interaction']):.3f}"
        ),
        transform=ax3.transAxes, ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle(
        "Individualism-Collectivism Language Overview  "
        "(−1 = Individualistic, +1 = Collectivist)",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    df = build_analysis_frame(raw)

    output_path = FIGURES_DIR / "ic1.png"
    make_ic_overview_panels(df, output_path)

    cond_test = welch_test(
        df.loc[df["Condition"] == "Interactive", "ic_mean"],
        df.loc[df["Condition"] == "Text", "ic_mean"],
    )
    group_test = welch_test(
        df.loc[df["Group"] == "WEIRD", "ic_mean"],
        df.loc[df["Group"] == "NORMAL", "ic_mean"],
    )
    interaction_test = fit_condition_group_interaction(df, "ic_mean")

    n_scored = int(df["ic_mean"].notna().sum())
    n_nonzero = int((df["ic_mean"].dropna() != 0.0).sum())
    mean_score = float(df["ic_mean"].dropna().mean())

    print("Individualism-Collectivism Overview (ic1)")
    print("==========================================")
    print(f"Participants with IC data:              {n_scored}")
    print(f"Participants with non-neutral IC score: {n_nonzero}")
    print(f"Overall mean IC score:                  {mean_score:.4f}")
    print(
        f"Condition (Interactive vs Text):       "
        f"t({int(cond_test['n_a'] + cond_test['n_b'] - 2)})="
        f"{float(cond_test['t']):.3f}, "
        f"p={float(cond_test['p']):.6f}, "
        f"d={float(cond_test['d']):.3f}"
    )
    print(
        f"Group (WEIRD vs Normal):               "
        f"t={float(group_test['t']):.3f}, "
        f"p={float(group_test['p']):.6f}, "
        f"d={float(group_test['d']):.3f}"
    )
    print(
        f"Group × Condition interaction:         "
        f"beta={float(interaction_test['coef_interaction']):.4f}, "
        f"p={float(interaction_test['p_interaction']):.6f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
