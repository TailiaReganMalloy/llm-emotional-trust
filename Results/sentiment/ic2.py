"""ic2.py – Trust-change comparison by IC (Individualism-Collectivism) group.

Mirrors the three-panel layout of sentiment2.py, but instead of splitting
participants by low/high sentiment, splits them by IC orientation:

  Individualistic  →  ic_mean in the lower half of the distribution
  Collectivist     →  ic_mean in the upper half of the distribution

Splitting strategy
------------------
Full-sample panel: even median split (lower 50% = Individualistic,
                   upper 50% = Collectivist).
WEIRD/Normal panels: the same even median split is applied within each
                     sub-group independently so group size is balanced.

Panels
------
1. Full sample  – Individualistic vs Collectivist across Overall /
                  Analytical / Emotional trust change.
2. WEIRD sub-sample  – same comparison.
3. Normal sub-sample – same comparison.

Each panel shows mean ± 95 % CI bars and an in-plot summary of Welch
t-test p-values for each outcome.
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

# ---------------------------------------------------------------------------
# Constants (duplicated from ic1 to keep this script standalone)
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

IC_GROUP_COLORS = {
    "individualistic": "#c26a26",   # orange-brown
    "collectivist": "#377eb8",      # blue
}


# ---------------------------------------------------------------------------
# Utilities
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


# ---------------------------------------------------------------------------
# IC scoring
# ---------------------------------------------------------------------------

def score_ic(text: str) -> float:
    if not text:
        return np.nan
    tokens = WORD_RE.findall(text.lower())
    i_weight = sum(INDIVIDUALISTIC.get(t, 0.0) for t in tokens)
    c_weight = sum(COLLECTIVIST.get(t, 0.0) for t in tokens)
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


# ---------------------------------------------------------------------------
# Build analysis frame
# ---------------------------------------------------------------------------

def build_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["Condition"] = work["Condition"].map(normalize_text).replace("", np.nan)

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["Group"] = np.where(work["weird_score"] >= 4, "WEIRD", "NORMAL")

    ic_columns: list[str] = []
    for source_col, suffix in METRICS_TEXT_COLS.items():
        if source_col not in work.columns:
            continue
        score_col = f"{suffix}_ic"
        work[score_col] = work[source_col].map(normalize_text).map(score_ic)
        ic_columns.append(score_col)

    if not ic_columns:
        raise ValueError("No text source columns found in Metrics.csv")

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

    return work


# ---------------------------------------------------------------------------
# IC group assignment (even median split within a subset)
# ---------------------------------------------------------------------------

def assign_ic_groups(series: pd.Series, label_col: pd.Series) -> pd.Series:
    """Assign 'individualistic' / 'collectivist' via even median split.

    Participants with ic_mean == median are assigned to the upper half.
    Returns a Series aligned to `series.index`.
    """
    result = pd.Series("missing", index=series.index, dtype=object)
    valid = series.dropna()
    if valid.empty:
        return result
    ordered = valid.sort_values(kind="mergesort")
    n = len(ordered)
    lower_n = n // 2
    result.loc[ordered.index[:lower_n]] = "individualistic"
    result.loc[ordered.index[lower_n:]] = "collectivist"
    return result


def add_ic_groups(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    # Full-sample split
    work["ic_group_full"] = assign_ic_groups(work["ic_mean"], work["ic_mean"])
    # Group-wise split
    work["ic_group_groupwise"] = "missing"
    for grp in ["WEIRD", "NORMAL"]:
        mask = work["Group"] == grp
        work.loc[mask, "ic_group_groupwise"] = assign_ic_groups(
            work.loc[mask, "ic_mean"], work.loc[mask, "ic_mean"]
        ).values
    return work


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_panel_summary(df: pd.DataFrame, group_col: str) -> list[str]:
    n_i = int((df[group_col] == "individualistic").sum())
    n_c = int((df[group_col] == "collectivist").sum())
    lines = [f"n_indiv={n_i}, n_coll={n_c}"]
    for outcome in OUTCOMES:
        i_vals = df.loc[df[group_col] == "individualistic", outcome]
        c_vals = df.loc[df[group_col] == "collectivist", outcome]
        result = welch_test(c_vals, i_vals)  # Collectivist − Individualistic
        p = float(result["p"])
        lines.append(
            f"{OUTCOME_LABELS[outcome]}: C−I p {format_p_text(p)} ({significance_stars(p)})"
        )
    return lines


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    title: str,
    show_legend: bool = False,
) -> None:
    x = np.arange(len(OUTCOMES), dtype=float)
    bar_width = 0.34

    i_means, i_err_low, i_err_high = [], [], []
    c_means, c_err_low, c_err_high = [], [], []

    for outcome in OUTCOMES:
        i_vals = df.loc[df[group_col] == "individualistic", outcome]
        c_vals = df.loc[df[group_col] == "collectivist", outcome]

        im, il, ih = mean_ci(i_vals)
        cm, cl, ch = mean_ci(c_vals)

        i_means.append(im); i_err_low.append(im - il); i_err_high.append(ih - im)
        c_means.append(cm); c_err_low.append(cm - cl); c_err_high.append(ch - cm)

    ax.bar(
        x - bar_width / 2, i_means, width=bar_width,
        color=IC_GROUP_COLORS["individualistic"],
        label="Individualistic",
        yerr=[i_err_low, i_err_high], capsize=4,
        ecolor="#8f4f1d", edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2, c_means, width=bar_width,
        color=IC_GROUP_COLORS["collectivist"],
        label="Collectivist",
        yerr=[c_err_low, c_err_high], capsize=4,
        ecolor="#2b5f8c", edgecolor="white",
    )

    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES], fontsize=11)
    ax.set_title(title, fontsize=14)

    summary = _build_panel_summary(df, group_col)
    ax.text(
        0.02, 0.98, "\n".join(summary),
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    if show_legend:
        ax.legend(loc="lower left", fontsize=9)


def make_ic_trust_change_plot(df: pd.DataFrame, output_path: Path) -> None:
    full_df = df[df["ic_group_full"].isin(["individualistic", "collectivist"])].copy()
    weird_df = df[df["Group"] == "WEIRD"].copy()
    normal_df = df[df["Group"] == "NORMAL"].copy()

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True, constrained_layout=True)

    _draw_panel(axes[0], full_df, "ic_group_full", "Full Sample", show_legend=True)
    _draw_panel(axes[1], weird_df, "ic_group_groupwise", "WEIRD")
    _draw_panel(axes[2], normal_df, "ic_group_groupwise", "Normal")

    axes[0].set_ylabel("Normalized trust change (z-score)", fontsize=13)
    fig.suptitle(
        "Trust-Change by IC Orientation  "
        "(Individualistic vs Collectivist Language)",
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
    df = add_ic_groups(df)

    output_path = FIGURES_DIR / "ic2.png"
    make_ic_trust_change_plot(df, output_path)

    full_df = df[df["ic_group_full"].isin(["individualistic", "collectivist"])]
    weird_df = df[df["Group"] == "WEIRD"]
    normal_df = df[df["Group"] == "NORMAL"]

    print("Individualism-Collectivism Trust-Change Comparison (ic2)")
    print("=========================================================")
    print(f"Total participants: {len(df)}")
    print(
        f"Full-sample split: "
        f"n_individualistic={(full_df['ic_group_full'] == 'individualistic').sum()}, "
        f"n_collectivist={(full_df['ic_group_full'] == 'collectivist').sum()}"
    )
    for outcome in OUTCOMES:
        i_vals = full_df.loc[full_df["ic_group_full"] == "individualistic", outcome]
        c_vals = full_df.loc[full_df["ic_group_full"] == "collectivist", outcome]
        r = welch_test(c_vals, i_vals)
        print(f"  Full {outcome}: C−I p={float(r['p']):.6f}, d={float(r['d']):.3f}")

    for grp_name, sub_df, col in [
        ("WEIRD", weird_df, "ic_group_groupwise"),
        ("Normal", normal_df, "ic_group_groupwise"),
    ]:
        print(
            f"{grp_name} n={len(sub_df)} "
            f"(individualistic={(sub_df[col] == 'individualistic').sum()}, "
            f"collectivist={(sub_df[col] == 'collectivist').sum()})"
        )
        for outcome in OUTCOMES:
            i_vals = sub_df.loc[sub_df[col] == "individualistic", outcome]
            c_vals = sub_df.loc[sub_df[col] == "collectivist", outcome]
            r = welch_test(c_vals, i_vals)
            print(f"  {grp_name} {outcome}: C−I p={float(r['p']):.6f}, d={float(r['d']):.3f}")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
