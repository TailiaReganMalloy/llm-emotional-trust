from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

ALPHA = 0.05

TEXT_COLS = {
    "AI Interaction Feeling": "interaction_feeling",
    "AI Definition Feeling": "definition_feeling",
    "Explanation Comment": "explanation_comment",
}

CONDITION_ORDER = ["Interactive", "Text"]
CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
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

WORD_RE = re.compile(r"[a-zA-Z']+")

POSITIVE_LEXICON = {
    "good",
    "great",
    "impressed",
    "clear",
    "helpful",
    "useful",
    "trust",
    "trusted",
    "confident",
    "confidence",
    "transparent",
    "reliable",
    "accurate",
    "safe",
    "fair",
    "insightful",
    "informative",
    "better",
    "excellent",
    "strong",
    "comfortable",
    "positive",
    "understand",
    "understanding",
}

NEGATIVE_LEXICON = {
    "bad",
    "confusing",
    "unclear",
    "unhelpful",
    "harmful",
    "unsafe",
    "biased",
    "bias",
    "deceptive",
    "dishonest",
    "wary",
    "suspicious",
    "distrust",
    "worried",
    "concerned",
    "negative",
    "frustrating",
    "frustrated",
    "unreliable",
    "opaque",
    "misleading",
    "doubt",
    "skeptical",
    "skeptic",
    "afraid",
}


@dataclass
class SentimentEngine:
    backend: str
    analyzer: object | None


@dataclass
class OLSResult:
    coef: float
    se: float
    t: float
    p: float
    df: float


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


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_sentiment_engine() -> SentimentEngine:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentEngine(backend="vader", analyzer=SentimentIntensityAnalyzer())
    except Exception:
        return SentimentEngine(backend="fallback_lexicon", analyzer=None)


def fallback_sentiment(text: str) -> float:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return np.nan

    positive_hits = sum(token in POSITIVE_LEXICON for token in tokens)
    negative_hits = sum(token in NEGATIVE_LEXICON for token in tokens)
    raw = (positive_hits - negative_hits) / max(len(tokens), 1)
    return float(np.clip(raw * 4.0, -1.0, 1.0))


def score_sentiment(text: str, engine: SentimentEngine) -> float:
    if not text:
        return np.nan

    if engine.backend == "vader" and engine.analyzer is not None:
        return float(engine.analyzer.polarity_scores(text)["compound"])

    return fallback_sentiment(text)


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


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> dict[str, float]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return {
            "n": float(len(pair)),
            "r": np.nan,
            "p": np.nan,
        }

    if method == "pearson":
        r_val, p_val = pearsonr(pair["x"], pair["y"])
    elif method == "spearman":
        r_val, p_val = spearmanr(pair["x"], pair["y"])
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    return {
        "n": float(len(pair)),
        "r": float(r_val),
        "p": float(p_val),
    }


def fit_interaction_ols(y: pd.Series, condition_bin: pd.Series, group_bin: pd.Series) -> dict[str, OLSResult]:
    fit_df = pd.DataFrame(
        {
            "y": y.astype(float),
            "condition_bin": condition_bin.astype(float),
            "group_bin": group_bin.astype(float),
        }
    ).dropna()

    x_condition = fit_df["condition_bin"].to_numpy(dtype=float)
    x_group = fit_df["group_bin"].to_numpy(dtype=float)
    x_inter = x_condition * x_group

    y_vec = fit_df["y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(fit_df)), x_condition, x_group, x_inter])

    beta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
    resid = y_vec - X @ beta

    n, p = X.shape
    dof = n - p

    if dof <= 0:
        nan_result = OLSResult(np.nan, np.nan, np.nan, np.nan, np.nan)
        return {
            "condition": nan_result,
            "group": nan_result,
            "interaction": nan_result,
            "model_df": OLSResult(np.nan, np.nan, np.nan, np.nan, np.nan),
        }

    rss = float(np.sum(resid**2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    results: dict[str, OLSResult] = {}
    for idx, key in zip([1, 2, 3], ["condition", "group", "interaction"]):
        coef = float(beta[idx])
        se_val = float(se[idx])
        if np.isclose(se_val, 0.0):
            t_val = np.nan
            p_val = np.nan
        else:
            t_val = coef / se_val
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), dof))

        results[key] = OLSResult(coef=coef, se=se_val, t=float(t_val), p=float(p_val), df=float(dof))

    results["model_df"] = OLSResult(coef=np.nan, se=np.nan, t=np.nan, p=np.nan, df=float(dof))
    return results


def build_analysis_frame(df: pd.DataFrame, engine: SentimentEngine) -> pd.DataFrame:
    required_cols = [
        "Condition",
        "Country of residence",
        "Language",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
        *TEXT_COLS.keys(),
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = work["Condition"].astype(str).str.strip()
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    language = work["Language"].astype(str).str.strip().str.lower()
    work["is_weird"] = work["Country of residence"].isin(WESTERN_COUNTRIES) & language.str.startswith("english")
    work["Group"] = np.where(work["is_weird"], "WEIRD", "NORMAL")

    for source_col, suffix in TEXT_COLS.items():
        normalized_col = f"{suffix}_text"
        sentiment_col = f"{suffix}_sentiment"
        work[normalized_col] = work[source_col].map(normalize_text)
        work[sentiment_col] = work[normalized_col].map(lambda txt: score_sentiment(txt, engine))

    sentiment_cols = [f"{suffix}_sentiment" for suffix in TEXT_COLS.values()]
    text_cols = [f"{suffix}_text" for suffix in TEXT_COLS.values()]

    work["combined_text"] = work[text_cols].apply(
        lambda row: " || ".join([value for value in row if value]), axis=1
    )
    work["sentiment_mean"] = work[sentiment_cols].mean(axis=1, skipna=True)
    work["response_count_non_empty"] = work[text_cols].apply(
        lambda row: float(sum(bool(value) for value in row)), axis=1
    )
    work["combined_text_length"] = work["combined_text"].map(len).astype(float)

    work["analytical_change"] = work["Total Analytical Trust Post"] - work["Total Analytical Trust"]
    work["emotional_change"] = work["Total Emotional Trust Post"] - work["Total Emotional Trust"]
    work["overall_change"] = work["analytical_change"] + work["emotional_change"]
    work["analytical_minus_emotional_change"] = work["analytical_change"] - work["emotional_change"]

    work["condition_bin"] = (work["Condition"] == "Interactive").astype(float)
    work["group_bin"] = (work["Group"] == "WEIRD").astype(float)

    keep = [
        "Condition",
        "Group",
        "is_weird",
        "condition_bin",
        "group_bin",
        "sentiment_mean",
        *sentiment_cols,
        "response_count_non_empty",
        "combined_text_length",
        "analytical_change",
        "emotional_change",
        "overall_change",
        "analytical_minus_emotional_change",
        *text_cols,
        "combined_text",
    ]
    return work[keep].copy()


def run_primary_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    sentiment_vars = [
        "sentiment_mean",
        "interaction_feeling_sentiment",
        "definition_feeling_sentiment",
        "explanation_comment_sentiment",
    ]

    for sentiment_var in sentiment_vars:
        interactive_vals = df.loc[df["Condition"] == "Interactive", sentiment_var]
        text_vals = df.loc[df["Condition"] == "Text", sentiment_var]
        condition_test = welch_test(interactive_vals, text_vals)
        rows.append(
            {
                "analysis_stage": "primary",
                "analysis_type": "welch",
                "metric": sentiment_var,
                "contrast": "condition_interactive_vs_text",
                "subset": "all",
                **condition_test,
            }
        )

        weird_vals = df.loc[df["Group"] == "WEIRD", sentiment_var]
        normal_vals = df.loc[df["Group"] == "NORMAL", sentiment_var]
        group_test = welch_test(weird_vals, normal_vals)
        rows.append(
            {
                "analysis_stage": "primary",
                "analysis_type": "welch",
                "metric": sentiment_var,
                "contrast": "group_weird_vs_normal",
                "subset": "all",
                **group_test,
            }
        )

        ols = fit_interaction_ols(df[sentiment_var], df["condition_bin"], df["group_bin"])
        for effect_name in ["condition", "group", "interaction"]:
            effect = ols[effect_name]
            rows.append(
                {
                    "analysis_stage": "primary",
                    "analysis_type": "ols",
                    "metric": sentiment_var,
                    "contrast": f"ols_{effect_name}_effect",
                    "subset": "all",
                    "n_a": float(df[sentiment_var].notna().sum()),
                    "n_b": np.nan,
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "t": effect.t,
                    "p": effect.p,
                    "d": np.nan,
                    "coef": effect.coef,
                    "se": effect.se,
                    "df": effect.df,
                }
            )

    trust_outcomes = [
        "overall_change",
        "analytical_change",
        "emotional_change",
        "analytical_minus_emotional_change",
    ]

    for outcome in trust_outcomes:
        for method in ["pearson", "spearman"]:
            corr = safe_correlation(df["sentiment_mean"], df[outcome], method)
            rows.append(
                {
                    "analysis_stage": "primary",
                    "analysis_type": method,
                    "metric": outcome,
                    "contrast": "sentiment_mean_association",
                    "subset": "all",
                    "n_a": corr["n"],
                    "n_b": np.nan,
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "t": np.nan,
                    "p": corr["p"],
                    "d": np.nan,
                    "coef": corr["r"],
                    "se": np.nan,
                    "df": np.nan,
                }
            )

    return pd.DataFrame(rows)


def run_follow_up_tests(df: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def has_sig(metric: str, contrast: str) -> bool:
        subset = primary[(primary["metric"] == metric) & (primary["contrast"] == contrast)]
        return bool((subset["p"] < ALPHA).any())

    sentiment_vars = [
        "sentiment_mean",
        "interaction_feeling_sentiment",
        "definition_feeling_sentiment",
        "explanation_comment_sentiment",
    ]

    for sentiment_var in sentiment_vars:
        trigger = (
            has_sig(sentiment_var, "condition_interactive_vs_text")
            or has_sig(sentiment_var, "group_weird_vs_normal")
            or has_sig(sentiment_var, "ols_interaction_effect")
        )
        if not trigger:
            continue

        for group_name in ["WEIRD", "NORMAL"]:
            subset = df[df["Group"] == group_name]
            test = welch_test(
                subset.loc[subset["Condition"] == "Interactive", sentiment_var],
                subset.loc[subset["Condition"] == "Text", sentiment_var],
            )
            rows.append(
                {
                    "analysis_stage": "follow_up",
                    "analysis_type": "welch",
                    "metric": sentiment_var,
                    "contrast": "condition_interactive_vs_text",
                    "subset": f"group={group_name}",
                    **test,
                }
            )

        for condition in CONDITION_ORDER:
            subset = df[df["Condition"] == condition]
            test = welch_test(
                subset.loc[subset["Group"] == "WEIRD", sentiment_var],
                subset.loc[subset["Group"] == "NORMAL", sentiment_var],
            )
            rows.append(
                {
                    "analysis_stage": "follow_up",
                    "analysis_type": "welch",
                    "metric": sentiment_var,
                    "contrast": "group_weird_vs_normal",
                    "subset": f"condition={condition}",
                    **test,
                }
            )

    for outcome in [
        "overall_change",
        "analytical_change",
        "emotional_change",
        "analytical_minus_emotional_change",
    ]:
        sig_primary = primary[
            (primary["analysis_type"].isin(["pearson", "spearman"]))
            & (primary["metric"] == outcome)
            & (primary["p"] < ALPHA)
        ]
        if sig_primary.empty:
            continue

        for condition in CONDITION_ORDER:
            subset = df[df["Condition"] == condition]
            corr = safe_correlation(subset["sentiment_mean"], subset[outcome], "pearson")
            rows.append(
                {
                    "analysis_stage": "follow_up",
                    "analysis_type": "pearson",
                    "metric": outcome,
                    "contrast": "sentiment_mean_association",
                    "subset": f"condition={condition}",
                    "n_a": corr["n"],
                    "n_b": np.nan,
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "t": np.nan,
                    "p": corr["p"],
                    "d": np.nan,
                    "coef": corr["r"],
                    "se": np.nan,
                    "df": np.nan,
                }
            )

        for group_name in ["WEIRD", "NORMAL"]:
            subset = df[df["Group"] == group_name]
            corr = safe_correlation(subset["sentiment_mean"], subset[outcome], "pearson")
            rows.append(
                {
                    "analysis_stage": "follow_up",
                    "analysis_type": "pearson",
                    "metric": outcome,
                    "contrast": "sentiment_mean_association",
                    "subset": f"group={group_name}",
                    "n_a": corr["n"],
                    "n_b": np.nan,
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "t": np.nan,
                    "p": corr["p"],
                    "d": np.nan,
                    "coef": corr["r"],
                    "se": np.nan,
                    "df": np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "analysis_stage",
                "analysis_type",
                "metric",
                "contrast",
                "subset",
                "n_a",
                "n_b",
                "mean_a",
                "mean_b",
                "t",
                "p",
                "d",
                "coef",
                "se",
                "df",
            ]
        )

    return pd.DataFrame(rows)


def compute_sentiment_mean_contrasts(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}

    overall_condition = welch_test(
        df.loc[df["Condition"] == "Interactive", "sentiment_mean"],
        df.loc[df["Condition"] == "Text", "sentiment_mean"],
    )
    results["overall_condition"] = overall_condition

    overall_group = welch_test(
        df.loc[df["Group"] == "WEIRD", "sentiment_mean"],
        df.loc[df["Group"] == "NORMAL", "sentiment_mean"],
    )
    results["overall_group"] = overall_group

    for group_name in ["WEIRD", "NORMAL"]:
        res = welch_test(
            df.loc[(df["Group"] == group_name) & (df["Condition"] == "Interactive"), "sentiment_mean"],
            df.loc[(df["Group"] == group_name) & (df["Condition"] == "Text"), "sentiment_mean"],
        )
        results[f"condition_within_{group_name}"] = res

    for condition in CONDITION_ORDER:
        res = welch_test(
            df.loc[(df["Condition"] == condition) & (df["Group"] == "WEIRD"), "sentiment_mean"],
            df.loc[(df["Condition"] == condition) & (df["Group"] == "NORMAL"), "sentiment_mean"],
        )
        results[f"group_within_{condition}"] = res

    return results


def make_group_condition_sentiment_plot(df: pd.DataFrame, primary: pd.DataFrame, output_path: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    for group in ["WEIRD", "NORMAL"]:
        for condition in CONDITION_ORDER:
            vals = df.loc[(df["Group"] == group) & (df["Condition"] == condition), "sentiment_mean"]
            mean_val, ci_low, ci_high = mean_ci(vals)
            summary_rows.append(
                {
                    "Group": group,
                    "Condition": condition,
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    summary = pd.DataFrame(summary_rows)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

    groups = ["WEIRD", "NORMAL"]
    x_positions = np.arange(len(groups), dtype=float)
    bar_width = 0.34
    offsets = {
        "Interactive": -bar_width / 2,
        "Text": bar_width / 2,
    }

    for condition in CONDITION_ORDER:
        for idx, group in enumerate(groups):
            row = summary[(summary["Group"] == group) & (summary["Condition"] == condition)].iloc[0]
            x_val = x_positions[idx] + offsets[condition]

            label = condition if idx == 0 else None
            ax.bar(
                x_val,
                float(row["mean"]),
                width=bar_width,
                color=CONDITION_COLORS[condition],
                edgecolor="none",
                label=label,
                zorder=2,
            )
            ax.errorbar(
                x_val,
                float(row["mean"]),
                yerr=[[float(row["mean"]) - float(row["ci_low"])], [float(row["ci_high"]) - float(row["mean"])]],
                fmt="none",
                ecolor="#4a4a4a",
                elinewidth=2.8,
                capsize=7,
                capthick=2.8,
                zorder=3,
            )

    ax.axhline(0, color="#4a4a4a", linewidth=1.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, fontsize=14)
    ax.set_ylabel("Mean sentiment score", fontsize=15)
    ax.set_xlabel("Participant group", fontsize=15)
    ax.set_title("Open-response sentiment by Group and Condition", fontsize=18, pad=12)
    ax.legend(title="Condition", fontsize=12, title_fontsize=13, loc="best")

    contrasts = compute_sentiment_mean_contrasts(df)
    interaction_row = primary[
        (primary["metric"] == "sentiment_mean")
        & (primary["contrast"] == "ols_interaction_effect")
    ]
    interaction_p = float(interaction_row["p"].iloc[0]) if not interaction_row.empty else np.nan

    sig_lines = [
        (
            "Overall condition (Interactive vs Text): "
            f"{significance_stars(contrasts['overall_condition']['p'])} "
            f"(p {format_p_text(contrasts['overall_condition']['p'])})"
        ),
        (
            "Overall group (WEIRD vs NORMAL): "
            f"{significance_stars(contrasts['overall_group']['p'])} "
            f"(p {format_p_text(contrasts['overall_group']['p'])})"
        ),
        (
            "Interaction (Condition x Group): "
            f"{significance_stars(interaction_p)} "
            f"(p {format_p_text(interaction_p)})"
        ),
        (
            "Within WEIRD (Interactive vs Text): "
            f"{significance_stars(contrasts['condition_within_WEIRD']['p'])}"
        ),
        (
            "Within NORMAL (Interactive vs Text): "
            f"{significance_stars(contrasts['condition_within_NORMAL']['p'])}"
        ),
        (
            "Within Interactive (WEIRD vs NORMAL): "
            f"{significance_stars(contrasts['group_within_Interactive']['p'])}"
        ),
        (
            "Within Text (WEIRD vs NORMAL): "
            f"{significance_stars(contrasts['group_within_Text']['p'])}"
        ),
    ]

    ax.text(
        0.01,
        0.99,
        "Significance summary:\n" + "\n".join(sig_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def make_sentiment_scatter_panels(df: pd.DataFrame, output_path: Path) -> None:
    outcomes = [
        "overall_change",
        "analytical_change",
        "emotional_change",
        "analytical_minus_emotional_change",
    ]
    titles = {
        "overall_change": "Overall trust change",
        "analytical_change": "Analytical trust change",
        "emotional_change": "Emotional trust change",
        "analytical_minus_emotional_change": "Analytical minus emotional change",
    }

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    flat_axes = axes.flatten()

    for ax, outcome in zip(flat_axes, outcomes):
        plot_df = df[["sentiment_mean", outcome, "Condition"]].dropna().copy()

        for condition in CONDITION_ORDER:
            cdf = plot_df[plot_df["Condition"] == condition]
            ax.scatter(
                cdf["sentiment_mean"],
                cdf[outcome],
                s=26,
                alpha=0.65,
                color=CONDITION_COLORS[condition],
                label=condition,
            )

        if len(plot_df) >= 3:
            x = plot_df["sentiment_mean"].to_numpy(dtype=float)
            y = plot_df[outcome].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, deg=1)
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=1.8)

            all_corr = safe_correlation(plot_df["sentiment_mean"], plot_df[outcome], method="pearson")

            interactive_corr = safe_correlation(
                plot_df.loc[plot_df["Condition"] == "Interactive", "sentiment_mean"],
                plot_df.loc[plot_df["Condition"] == "Interactive", outcome],
                method="pearson",
            )
            text_corr = safe_correlation(
                plot_df.loc[plot_df["Condition"] == "Text", "sentiment_mean"],
                plot_df.loc[plot_df["Condition"] == "Text", outcome],
                method="pearson",
            )
            weird_corr = safe_correlation(
                df.loc[df["Group"] == "WEIRD", "sentiment_mean"],
                df.loc[df["Group"] == "WEIRD", outcome],
                method="pearson",
            )
            normal_corr = safe_correlation(
                df.loc[df["Group"] == "NORMAL", "sentiment_mean"],
                df.loc[df["Group"] == "NORMAL", outcome],
                method="pearson",
            )

            corr_lines = [
                (
                    f"All: r={all_corr['r']:.3f}, p {format_p_text(all_corr['p'])} "
                    f"({significance_stars(all_corr['p'])})"
                ),
                (
                    f"Interactive: {significance_stars(interactive_corr['p'])} | "
                    f"Text: {significance_stars(text_corr['p'])}"
                ),
                (
                    f"WEIRD: {significance_stars(weird_corr['p'])} | "
                    f"NORMAL: {significance_stars(normal_corr['p'])}"
                ),
            ]
            ax.text(
                0.02,
                0.98,
                "\n".join(corr_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        ax.set_title(titles[outcome], fontsize=13)
        ax.set_xlabel("Sentiment mean")
        ax.set_ylabel("Change")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        flat_axes[0].legend(
            handles,
            labels,
            title="Condition",
            loc="lower right",
            fontsize=10,
            title_fontsize=11,
            frameon=True,
        )

    fig.suptitle("Sentiment vs Trust-Change Outcomes", fontsize=18)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def summarize_significant_results(primary: pd.DataFrame, follow_up: pd.DataFrame) -> list[str]:
    sig_lines: list[str] = []

    all_results = pd.concat([primary, follow_up], ignore_index=True)
    sig = all_results[all_results["p"] < ALPHA].copy()

    if sig.empty:
        return ["No tests reached p < .05 in this exploratory run."]

    for _, row in sig.iterrows():
        line = (
            f"{row['analysis_stage']} | {row['analysis_type']} | {row['metric']} | "
            f"{row['contrast']} | subset={row['subset']} | p={row['p']:.4f}"
        )
        sig_lines.append(line)

    return sig_lines


def get_primary_p(primary: pd.DataFrame, metric: str, contrast: str, analysis_type: str | None = None) -> float:
    subset = primary[(primary["metric"] == metric) & (primary["contrast"] == contrast)]
    if analysis_type is not None:
        subset = subset[subset["analysis_type"] == analysis_type]
    if subset.empty:
        return np.nan
    return float(subset["p"].iloc[0])


def write_sentiment_latex(
    analysis_df: pd.DataFrame,
    primary: pd.DataFrame,
    sentiment_plot_path: Path,
    scatter_plot_path: Path,
    output_path: Path,
) -> None:
    sentiment_contrasts = compute_sentiment_mean_contrasts(analysis_df)

    overall_cond_p = sentiment_contrasts["overall_condition"]["p"]
    overall_group_p = sentiment_contrasts["overall_group"]["p"]
    interaction_p = get_primary_p(primary, "sentiment_mean", "ols_interaction_effect", analysis_type="ols")

    outcomes = [
        "overall_change",
        "analytical_change",
        "emotional_change",
        "analytical_minus_emotional_change",
    ]
    outcome_labels = {
        "overall_change": "Overall trust change",
        "analytical_change": "Analytical trust change",
        "emotional_change": "Emotional trust change",
        "analytical_minus_emotional_change": "Analytical minus emotional trust change",
    }

    lines: list[str] = []
    lines.append("\\section*{Exploratory Sentiment Figures}")
    lines.append("This section summarizes the two exploratory sentiment plots with explicit significance status for each main test.")
    lines.append("")

    lines.append("\\subsection*{Figure 1: Sentiment by Group and Condition}")
    lines.append(
        "Figure 1 compares participant-level mean sentiment scores across WEIRD and NORMAL groups and across Interactive and Text conditions."
    )
    lines.append("\\begin{itemize}")
    lines.append(
        "\\item Overall condition contrast (Interactive vs Text): "
        f"{significance_stars(overall_cond_p)} (\\(p {format_p_text(overall_cond_p)}\\))."
    )
    lines.append(
        "\\item Overall group contrast (WEIRD vs NORMAL): "
        f"{significance_stars(overall_group_p)} (\\(p {format_p_text(overall_group_p)}\\))."
    )
    lines.append(
        "\\item Condition-by-group interaction (OLS interaction term): "
        f"{significance_stars(interaction_p)} (\\(p {format_p_text(interaction_p)}\\))."
    )
    lines.append(
        "\\item Within-group condition contrasts: "
        f"WEIRD = {significance_stars(sentiment_contrasts['condition_within_WEIRD']['p'])}, "
        f"NORMAL = {significance_stars(sentiment_contrasts['condition_within_NORMAL']['p'])}."
    )
    lines.append(
        "\\item Within-condition group contrasts: "
        f"Interactive = {significance_stars(sentiment_contrasts['group_within_Interactive']['p'])}, "
        f"Text = {significance_stars(sentiment_contrasts['group_within_Text']['p'])}."
    )
    lines.append("\\end{itemize}")
    lines.append("\\begin{figure}[h]")
    lines.append("\\centering")
    lines.append(
        "\\includegraphics[width=0.9\\linewidth]{"
        + str(sentiment_plot_path.relative_to(REPO_ROOT)).replace("\\", "/")
        + "}"
    )
    lines.append("\\caption{Open-response sentiment by group and condition with explicit significance labels (stars or ns).}")
    lines.append("\\label{fig:exploratory_sentiment_group_condition}")
    lines.append("\\end{figure}")
    lines.append("")

    lines.append("\\subsection*{Figure 2: Sentiment Associations With Trust Change}")
    lines.append(
        "Figure 2 shows sentiment-to-trust-change associations with panel-specific significance labeling for all-participant and subgroup/condition correlations."
    )
    lines.append("\\begin{itemize}")
    for outcome in outcomes:
        p_val = get_primary_p(primary, outcome, "sentiment_mean_association", analysis_type="pearson")
        lines.append(
            "\\item "
            + outcome_labels[outcome]
            + ": "
            + f"{significance_stars(p_val)} (\\(p {format_p_text(p_val)}\\)) in the all-sample Pearson test."
        )
    lines.append("\\end{itemize}")
    lines.append("\\begin{figure}[h]")
    lines.append("\\centering")
    lines.append(
        "\\includegraphics[width=0.92\\linewidth]{"
        + str(scatter_plot_path.relative_to(REPO_ROOT)).replace("\\", "/")
        + "}"
    )
    lines.append("\\caption{Sentiment versus trust-change outcomes with significance annotations (stars or ns) per panel.}")
    lines.append("\\label{fig:exploratory_sentiment_trust_changes}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\noindent Significance key: \\(* p<.05\\), \\(** p<.01\\), \\(*** p<.001\\), ns = not significant.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_self_prompt(primary: pd.DataFrame, follow_up: pd.DataFrame) -> list[str]:
    prompts: list[str] = []

    sig_primary = primary[primary["p"] < ALPHA]

    sig_condition = sig_primary[sig_primary["contrast"] == "condition_interactive_vs_text"]
    sig_group = sig_primary[sig_primary["contrast"] == "group_weird_vs_normal"]
    sig_interaction = sig_primary[sig_primary["contrast"] == "ols_interaction_effect"]
    sig_corr = sig_primary[
        (sig_primary["analysis_type"].isin(["pearson", "spearman"]))
        & (sig_primary["contrast"] == "sentiment_mean_association")
    ]

    if not sig_condition.empty:
        prompts.append(
            "Run token-level keyness (log-odds or chi-square) between Interactive and Text responses for each text field and inspect whether specific lexical cues explain condition sentiment differences."
        )

    if not sig_group.empty:
        prompts.append(
            "Compare thematic language between WEIRD and NORMAL groups using keyword extraction and manual coding on high-impact terms, then test whether theme prevalence remains after conditioning on Condition."
        )

    if not sig_interaction.empty:
        prompts.append(
            "Fit simple-slope models for condition effects within WEIRD and NORMAL separately and bootstrap confidence intervals to quantify interaction robustness."
        )

    if not sig_corr.empty:
        significant_outcomes = sorted(set(sig_corr["metric"].tolist()))
        outcome_text = ", ".join(significant_outcomes)
        prompts.append(
            f"For significant sentiment-trust links ({outcome_text}), run multivariable models adjusting for condition and group, then evaluate whether sentiment still predicts trust change."
        )
        prompts.append(
            "Check non-linear effects by adding quadratic sentiment terms and compare model fit with AIC or adjusted R-squared."
        )

    if follow_up[follow_up["p"] < ALPHA].empty and sig_primary.empty:
        prompts.append(
            "No significant effects were detected. Next, increase sensitivity by using richer sentiment features (subjectivity, uncertainty words, negation rate) and rerun with robust regression."
        )

    prompts.append(
        "Validate exploratory findings with false-discovery-rate correction and report which conclusions remain stable."
    )

    return prompts


def write_report(
    engine: SentimentEngine,
    analysis_df: pd.DataFrame,
    primary: pd.DataFrame,
    follow_up: pd.DataFrame,
    report_path: Path,
    self_prompt_path: Path,
) -> None:
    significant_lines = summarize_significant_results(primary, follow_up)
    prompts = build_self_prompt(primary, follow_up)

    lines: list[str] = []
    lines.append("# Exploratory Qualitative Sentiment Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Sentiment backend: {engine.backend}")
    lines.append(f"- Participants analyzed: {len(analysis_df)}")
    lines.append("- Text sources: AI Interaction Feeling, AI Definition Feeling, Explanation Comment")
    lines.append("- Grouping: WEIRD vs NORMAL (Western country + English language)")
    lines.append("- Outcomes: overall, analytical, emotional, and analytical-minus-emotional trust change")
    lines.append("")

    lines.append("## Primary Test Counts")
    lines.append(f"- Number of primary tests: {len(primary)}")
    lines.append(f"- Number of significant primary tests (p < .05): {int((primary['p'] < ALPHA).sum())}")
    lines.append("")

    lines.append("## Follow-Up Test Counts")
    lines.append(f"- Number of triggered follow-up tests: {len(follow_up)}")
    lines.append(f"- Number of significant follow-up tests (p < .05): {int((follow_up['p'] < ALPHA).sum())}")
    lines.append("")

    lines.append("## Significant Findings")
    for line in significant_lines:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## Evaluation")
    lines.append("- This module is exploratory and not pre-registered; interpret p-values as hypothesis-generating.")
    lines.append("- Sentiment is based on short open responses, so lexical noise and sparse text can attenuate effects.")
    lines.append("- Effect sizes and directionality should be prioritized over binary significance in follow-up work.")
    lines.append("")

    lines.append("## Self-Prompted Next Analyses")
    for prompt in prompts:
        lines.append(f"- {prompt}")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prompt_lines = [
        "# Self Prompt For Next Analysis Round",
        "",
        "Use the significant findings from primary and follow-up tests to run targeted next analyses:",
        "",
    ]
    for idx, prompt in enumerate(prompts, start=1):
        prompt_lines.append(f"{idx}. {prompt}")
    prompt_lines.append("")
    self_prompt_path.write_text("\n".join(prompt_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    engine = get_sentiment_engine()
    analysis_df = build_analysis_frame(raw, engine)

    primary = run_primary_tests(analysis_df)
    follow_up = run_follow_up_tests(analysis_df, primary)

    scored_path = OUTPUT_DIR / "scored_participant_text.csv"
    primary_path = OUTPUT_DIR / "primary_tests.csv"
    follow_up_path = OUTPUT_DIR / "follow_up_tests.csv"
    report_path = OUTPUT_DIR / "report.md"
    self_prompt_path = OUTPUT_DIR / "self_prompt_for_next_round.md"
    sentiment_latex_path = OUTPUT_DIR / "sentiment.txt"
    sentiment_plot_path = OUTPUT_DIR / "sentiment_by_group_condition.png"
    scatter_plot_path = OUTPUT_DIR / "sentiment_vs_trust_changes.png"

    analysis_df.to_csv(scored_path, index=False)
    primary.to_csv(primary_path, index=False)
    follow_up.to_csv(follow_up_path, index=False)

    make_group_condition_sentiment_plot(analysis_df, primary, sentiment_plot_path)
    make_sentiment_scatter_panels(analysis_df, scatter_plot_path)
    write_sentiment_latex(
        analysis_df=analysis_df,
        primary=primary,
        sentiment_plot_path=sentiment_plot_path,
        scatter_plot_path=scatter_plot_path,
        output_path=sentiment_latex_path,
    )

    write_report(
        engine=engine,
        analysis_df=analysis_df,
        primary=primary,
        follow_up=follow_up,
        report_path=report_path,
        self_prompt_path=self_prompt_path,
    )

    print(f"Saved: {scored_path}")
    print(f"Saved: {primary_path}")
    print(f"Saved: {follow_up_path}")
    print(f"Saved: {sentiment_plot_path}")
    print(f"Saved: {scatter_plot_path}")
    print(f"Saved: {sentiment_latex_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {self_prompt_path}")


if __name__ == "__main__":
    main()
