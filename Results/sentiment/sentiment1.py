from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
FIGURES_DIR = REPO_ROOT / "Figures"

ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0

METRICS_TEXT_COLS = {
    "AI Interaction Feeling": "interaction_feeling",
    "Need Model Understanding": "need_model_understanding",
    "Job Screening Feeling": "job_screening_feeling",
    "Explanation Comment": "explanation_comment",
}

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
    if (row.get("Nationality") in WESTERN_COUNTRIES
            or row.get("Country of birth") in WESTERN_COUNTRIES):
        score += 1
    return score

WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class SentimentEngine:
    backend: str
    analyzer: object | None


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_sentiment_engine() -> SentimentEngine:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentEngine(backend="vader", analyzer=SentimentIntensityAnalyzer())
    except Exception as exc:
        raise ModuleNotFoundError(
            "vaderSentiment is required for this analysis. Install it with: pip install vaderSentiment"
        ) from exc


def score_sentiment(text: str, engine: SentimentEngine) -> float:
    if not text:
        return np.nan

    if engine.analyzer is None:
        raise RuntimeError("Sentiment engine is not initialized. Ensure vaderSentiment is installed.")

    return float(engine.analyzer.polarity_scores(text)["compound"])


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

    var_a = float(av.var(ddof=1))
    var_b = float(bv.var(ddof=1))
    pooled_num = (len(av) - 1) * var_a + (len(bv) - 1) * var_b
    pooled_den = len(av) + len(bv) - 2
    if pooled_den <= 0:
        return np.nan
    pooled_sd = np.sqrt(pooled_num / pooled_den)
    if np.isclose(pooled_sd, 0.0):
        return np.nan
    return float((av.mean() - bv.mean()) / pooled_sd)


def welch_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    av = a.dropna().astype(float)
    bv = b.dropna().astype(float)

    if len(av) < 2 or len(bv) < 2:
        return {
            "n_a": float(len(av)),
            "n_b": float(len(bv)),
            "mean_a": float(av.mean()) if len(av) else np.nan,
            "mean_b": float(bv.mean()) if len(bv) else np.nan,
            "t": np.nan,
            "p": np.nan,
            "d": np.nan,
        }

    t_stat, p_value = ttest_ind(av, bv, equal_var=False, nan_policy="omit")
    return {
        "n_a": float(len(av)),
        "n_b": float(len(bv)),
        "mean_a": float(av.mean()),
        "mean_b": float(bv.mean()),
        "t": float(t_stat),
        "p": float(p_value),
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


def fit_condition_group_interaction(df: pd.DataFrame, outcome_col: str = "sentiment_mean") -> dict[str, float]:
    fit_df = df[
        df["Condition"].isin(["Interactive", "Text"]) & df["Group"].isin(["WEIRD", "NORMAL"])
    ][[outcome_col, "Condition", "Group"]].dropna()

    if len(fit_df) < 8:
        return {
            "n": float(len(fit_df)),
            "coef_interaction": np.nan,
            "t_interaction": np.nan,
            "p_interaction": np.nan,
        }

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
        return {
            "n": float(len(fit_df)),
            "coef_interaction": np.nan,
            "t_interaction": np.nan,
            "p_interaction": np.nan,
        }

    sigma2 = float(np.sum(resid**2) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    if np.isclose(float(se[3]), 0.0):
        t_inter = np.nan
        p_inter = np.nan
    else:
        t_inter = float(beta[3] / se[3])
        p_inter = float(2.0 * stats.t.sf(abs(t_inter), dof))

    return {
        "n": float(len(fit_df)),
        "coef_interaction": float(beta[3]),
        "t_interaction": t_inter,
        "p_interaction": p_inter,
    }


def build_analysis_frame(raw: pd.DataFrame, engine: SentimentEngine) -> pd.DataFrame:
    work = raw.copy()
    work["Condition"] = work["Condition"].map(normalize_text).replace("", np.nan)

    work["weird_score"] = work.apply(score_weird, axis=1)
    work["is_weird"] = work["weird_score"] >= 4
    work["Group"] = np.where(work["is_weird"], "WEIRD", "NORMAL")

    sentiment_columns: list[str] = []
    for source_col, suffix in METRICS_TEXT_COLS.items():
        if source_col not in work.columns:
            continue
        text_col = f"{suffix}_text"
        score_col = f"{suffix}_sentiment"
        work[text_col] = work[source_col].map(normalize_text)
        work[score_col] = work[text_col].map(lambda txt: score_sentiment(txt, engine))
        sentiment_columns.append(score_col)

    if not sentiment_columns:
        raise ValueError("No sentiment source columns were found in Metrics.csv")

    work["sentiment_mean"] = work[sentiment_columns].mean(axis=1, skipna=True)

    if {"Total Analytical Trust", "Total Analytical Trust Post"}.issubset(work.columns):
        analytical_pre = pd.to_numeric(work["Total Analytical Trust"], errors="coerce") / ANALYTICAL_MAX
        analytical_post = pd.to_numeric(work["Total Analytical Trust Post"], errors="coerce") / ANALYTICAL_MAX
        analytical_raw = analytical_post - analytical_pre
    else:
        analytical_raw = pd.to_numeric(work.get("Analytical Trust Difference"), errors="coerce")

    if {"Total Emotional Trust", "Total Emotional Trust Post"}.issubset(work.columns):
        emotional_pre = pd.to_numeric(work["Total Emotional Trust"], errors="coerce") / EMOTIONAL_MAX
        emotional_post = pd.to_numeric(work["Total Emotional Trust Post"], errors="coerce") / EMOTIONAL_MAX
        emotional_raw = emotional_post - emotional_pre
    else:
        emotional_raw = pd.to_numeric(work.get("Emotional Trust Difference"), errors="coerce")

    work["analytical_change"] = zscore(analytical_raw)
    work["emotional_change"] = zscore(emotional_raw)
    work["overall_change"] = zscore(analytical_raw + emotional_raw)

    return work


def make_old_sentiment_panels(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["sentiment_mean"].notna()].copy()

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)

    ax0, ax1, ax2, ax3 = axes

    ax0.hist(plot_df["sentiment_mean"], bins=24, color="#4e79a7", edgecolor="white", alpha=0.9)
    ax0.axvline(float(plot_df["sentiment_mean"].mean()), color="#1f2937", linewidth=2, linestyle="--")
    ax0.set_title("Participant Mean Sentiment", fontsize=14)
    ax0.set_xlabel("Mean sentiment", fontsize=11)
    ax0.set_ylabel("Participants", fontsize=11)

    condition_order = ["Interactive", "Text"]
    condition_means: list[float] = []
    condition_err_low: list[float] = []
    condition_err_high: list[float] = []
    for condition in condition_order:
        vals = plot_df.loc[plot_df["Condition"] == condition, "sentiment_mean"]
        mean_val, low_ci, high_ci = mean_ci(vals)
        condition_means.append(mean_val)
        condition_err_low.append(mean_val - low_ci)
        condition_err_high.append(high_ci - mean_val)

    cond_test = welch_test(
        plot_df.loc[plot_df["Condition"] == "Interactive", "sentiment_mean"],
        plot_df.loc[plot_df["Condition"] == "Text", "sentiment_mean"],
    )

    x_cond = np.arange(len(condition_order), dtype=float)
    ax1.bar(
        x_cond,
        condition_means,
        yerr=[condition_err_low, condition_err_high],
        capsize=4,
        color=["#377eb8", "#c26a26"],
        edgecolor="white",
    )
    ax1.set_xticks(x_cond)
    ax1.set_xticklabels(condition_order)
    ax1.set_title("Condition Difference in Mean Sentiment", fontsize=14)
    ax1.set_ylabel("Mean sentiment", fontsize=11)
    ax1.text(
        0.02,
        0.98,
        (
            f"p {format_p_text(float(cond_test['p']))} ({significance_stars(float(cond_test['p']))})\n"
            f"d = {float(cond_test['d']):.2f}"
        ),
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    group_order = ["WEIRD", "NORMAL"]
    group_labels = ["WEIRD", "Normal"]
    group_means: list[float] = []
    group_err_low: list[float] = []
    group_err_high: list[float] = []
    for group_name in group_order:
        vals = plot_df.loc[plot_df["Group"] == group_name, "sentiment_mean"]
        mean_val, low_ci, high_ci = mean_ci(vals)
        group_means.append(mean_val)
        group_err_low.append(mean_val - low_ci)
        group_err_high.append(high_ci - mean_val)

    group_test = welch_test(
        plot_df.loc[plot_df["Group"] == "WEIRD", "sentiment_mean"],
        plot_df.loc[plot_df["Group"] == "NORMAL", "sentiment_mean"],
    )

    x_group = np.arange(len(group_order), dtype=float)
    ax2.bar(
        x_group,
        group_means,
        yerr=[group_err_low, group_err_high],
        capsize=4,
        color=["#377eb8", "#c26a26"],
        edgecolor="white",
    )
    ax2.set_xticks(x_group)
    ax2.set_xticklabels(group_labels)
    ax2.set_title("Demographic Group Difference in Mean Sentiment", fontsize=14)
    ax2.set_ylabel("Mean sentiment", fontsize=11)
    ax2.text(
        0.02,
        0.98,
        (
            f"p {format_p_text(float(group_test['p']))} ({significance_stars(float(group_test['p']))})\n"
            f"d = {float(group_test['d']):.2f}"
        ),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    condition_order = ["Interactive", "Text"]
    condition_colors = {"Interactive": "#377eb8", "Text": "#c26a26"}
    bar_width = 0.34
    for offset, condition in zip([-bar_width / 2, bar_width / 2], condition_order):
        means: list[float] = []
        err_low: list[float] = []
        err_high: list[float] = []
        for group_name in group_order:
            vals = plot_df.loc[
                (plot_df["Group"] == group_name) & (plot_df["Condition"] == condition),
                "sentiment_mean",
            ]
            mean_val, low_ci, high_ci = mean_ci(vals)
            means.append(mean_val)
            err_low.append(mean_val - low_ci)
            err_high.append(high_ci - mean_val)

        ax3.bar(
            x_group + offset,
            means,
            width=bar_width,
            yerr=[err_low, err_high],
            capsize=4,
            color=condition_colors[condition],
            edgecolor="white",
            label=condition,
        )

    interaction_test = fit_condition_group_interaction(plot_df, "sentiment_mean")
    ax3.set_xticks(x_group)
    ax3.set_xticklabels(group_labels)
    ax3.set_title("Group x Condition Interaction", fontsize=14)
    ax3.set_ylabel("Mean sentiment", fontsize=11)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.text(
        0.02,
        0.98,
        (
            f"interaction p {format_p_text(float(interaction_test['p_interaction']))} "
            f"({significance_stars(float(interaction_test['p_interaction']))})\n"
            f"beta = {float(interaction_test['coef_interaction']):.3f}"
        ),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle("Sentiment Overview", fontsize=17)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    engine = get_sentiment_engine()
    analysis_df = build_analysis_frame(raw, engine)

    output_path = FIGURES_DIR / "sentiment1.png"
    make_old_sentiment_panels(analysis_df, output_path)

    cond_test = welch_test(
        analysis_df.loc[analysis_df["Condition"] == "Interactive", "sentiment_mean"],
        analysis_df.loc[analysis_df["Condition"] == "Text", "sentiment_mean"],
    )
    group_test = welch_test(
        analysis_df.loc[analysis_df["Group"] == "WEIRD", "sentiment_mean"],
        analysis_df.loc[analysis_df["Group"] == "NORMAL", "sentiment_mean"],
    )
    interaction_test = fit_condition_group_interaction(analysis_df, "sentiment_mean")

    print("Sentiment overview")
    print("==================")
    print(f"Participants with sentiment data: {int(analysis_df['sentiment_mean'].notna().sum())}")
    print(
        "Condition (Interactive vs Text): "
        f"p={float(cond_test['p']):.6f}, d={float(cond_test['d']):.3f}"
    )
    print(
        "Demographic group (WEIRD vs Normal): "
        f"p={float(group_test['p']):.6f}, d={float(group_test['d']):.3f}"
    )
    print(
        "Group x Condition interaction: "
        f"p={float(interaction_test['p_interaction']):.6f}, "
        f"beta={float(interaction_test['coef_interaction']):.3f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
