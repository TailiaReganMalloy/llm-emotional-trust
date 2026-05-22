from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, ttest_ind, ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis6.png"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "hypothesis6.txt"
ALPHA = 0.05

CONDITION_ORDER = ["Interactive", "Text"]
GROUP_ORDER = ["WEIRD", "NORMAL"]
CONDITION_LABELS = {
    "Interactive": "Interactive",
    "Text": "Static (Text)",
}
CONDITION_COLORS = {
    "Interactive": "#377eb8",
    "Text": "#c26a26",
}
METRIC_COLORS = {
    "analytical_change": "#377eb8",
    "emotional_change": "#c26a26",
    "analytical_vs_emotional_diff": "#4daf4a",
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


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def normalize_condition(series: pd.Series) -> pd.Series:
    return (
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


def coalesce_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in columns:
        if col not in df.columns:
            continue
        candidate = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        candidate = candidate.mask(candidate.eq(""), pd.NA)
        out = out.fillna(candidate)
    return out


def coalesce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        out = out.fillna(candidate)
    return out


def collapse_top_categories(
    series: pd.Series,
    max_groups: int = 8,
    min_count: int = 12,
    other_label: str = "Other",
) -> pd.Series:
    clean = series.astype("string").str.strip()
    clean = clean.mask(clean.eq(""), pd.NA)
    counts = clean.value_counts(dropna=True)

    keep = counts[counts >= min_count].index.tolist()
    if len(keep) > max_groups:
        keep = keep[:max_groups]

    collapsed = clean.where(clean.isin(keep), other_label)
    if (collapsed == other_label).sum() == 0:
        return clean
    return collapsed


def normalize_gender(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if not text:
        return "Unknown"
    if text in {"man", "male"} or text.startswith("man"):
        return "Man"
    if text in {"woman", "female"} or text.startswith("woman"):
        return "Woman"
    if "non-binary" in text or "nonbinary" in text:
        return "NB/GD"
    if "prefer not" in text or "rather not" in text:
        return "NB/GD"
    return "Unknown"


def normalize_ai_knowledge(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if not text:
        return "Unknown"
    if "no knowledge" in text:
        return "No knowledge"
    if "beginner" in text:
        return "Beginner"
    if "conceptual" in text:
        return "Conceptual"
    if "advanced" in text:
        return "Advanced"
    if "expert" in text:
        return "Expert"
    return str(value).strip().title()


def normalize_education(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if not text:
        return "Unknown"
    if "high school" in text or "secondary" in text:
        return "High School"
    if "associate" in text:
        return "Associate"
    if "some college" in text:
        return "Some College"
    if "bachelor" in text:
        return "Bachelor"
    if "master" in text:
        return "Master"
    if "phd" in text or "doctor" in text:
        return "PhD"
    if "graduate professional" in text:
        return "Graduate Professional Degree"
    return str(value).strip().title()


def normalize_employment(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if not text:
        return "Unknown"
    if "full-time" in text or "full time" in text:
        return "Full-Time"
    if "part-time" in text or "part time" in text:
        return "Part-Time"
    if "self-employed" in text:
        return "Self-Employed"
    if "student" in text:
        return "Student"
    if "retired" in text:
        return "Retired"
    if "unemployed" in text:
        return "Unemployed"
    if "not in paid work" in text:
        return "Not in paid work"
    if "due to start" in text:
        return "Starting new job soon"
    return str(value).strip().title()


def normalize_nationality(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if not text:
        return "Unknown"
    aliases = {
        "us": "United States",
        "u.s.": "United States",
        "u.s.a": "United States",
        "usa": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }
    if text in aliases:
        return aliases[text]
    return str(value).strip().title()


def normalize_ethnicity(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if not text:
        return "Unknown"
    if "prefer not" in text or "rather not" in text:
        return "Prefer not to say"
    if "white" in text:
        return "White"
    if "asian" in text:
        return "Asian"
    if "black" in text or "african" in text:
        return "Black"
    if "mixed" in text or "multiracial" in text:
        return "Mixed"
    if "hispanic" in text or "latino" in text:
        return "Latino/Hispanic"
    if "middle eastern" in text or "arab" in text:
        return "Middle Eastern/North African"
    if "indigenous" in text or "native" in text:
        return "Indigenous"
    if "other" in text:
        return "Other"
    return str(value).strip().title()


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan

    mean_val = float(clean.mean())
    if n == 1:
        return mean_val, mean_val, mean_val

    sem = float(clean.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf((1 + confidence) / 2, df=n - 1) * sem)
    return mean_val, mean_val - margin, mean_val + margin


def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Condition",
        "Country of residence",
        "Language",
        "Total Analytical Trust",
        "Total Analytical Trust Post",
        "Total Emotional Trust",
        "Total Emotional Trust Post",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work["Condition"] = normalize_condition(work["Condition"])
    work = work[work["Condition"].isin(CONDITION_ORDER)].copy()

    language = work["Language"].astype(str).str.strip().str.lower()
    residence = work["Country of residence"].astype(str).str.strip()
    weird_like = residence.isin(WESTERN_COUNTRIES) & language.str.startswith("english")

    work["Group"] = np.where(weird_like, "WEIRD", "NORMAL")
    work["condition_bin"] = (work["Condition"] == "Interactive").astype(float)
    work["group_bin"] = (work["Group"] == "WEIRD").astype(float)

    # Per-item average change keeps analytical and emotional scales comparable.
    work["analytical_change"] = (
        (work["Total Analytical Trust Post"] - work["Total Analytical Trust"]) / 10.0
    )
    work["emotional_change"] = (
        (work["Total Emotional Trust Post"] - work["Total Emotional Trust"]) / 9.0
    )
    work["analytical_vs_emotional_diff"] = work["analytical_change"] - work["emotional_change"]

    age_numeric = coalesce_numeric_columns(work, ["Age (Demographics)", "Age", "Age.1", "Age.2"])
    age_numeric = age_numeric.where((age_numeric >= 18) & (age_numeric <= 100), np.nan)
    work["Age Numeric"] = age_numeric
    work["Age Group"] = pd.cut(
        age_numeric,
        bins=[17, 24, 34, 44, 100],
        labels=["18-24", "25-34", "35-44", "45+"],
        include_lowest=True,
    ).astype("string").fillna("Unknown")

    gender_raw = coalesce_text_columns(work, ["Gender", "Gender (Demographics)", "Sex", "Sex (Demographics)"])
    work["Gender Norm"] = gender_raw.map(normalize_gender)

    ai_raw = coalesce_text_columns(work, ["AI Knowledge", "AI Knowledge (Demographics)"])
    ai_norm = ai_raw.map(normalize_ai_knowledge)
    work["AI Knowledge Norm"] = (
        collapse_top_categories(ai_norm, max_groups=6, min_count=12, other_label="Other")
        .astype("string")
        .fillna("Unknown")
    )

    edu_raw = coalesce_text_columns(work, ["Education", "Education (Demographics)"])
    edu_norm = edu_raw.map(normalize_education)
    work["Education Norm"] = (
        collapse_top_categories(edu_norm, max_groups=7, min_count=12, other_label="Other")
        .astype("string")
        .fillna("Unknown")
    )

    emp_raw = coalesce_text_columns(
        work,
        ["Employment status", "Employment Status", "Employment status (Demographics)"],
    )
    emp_norm = emp_raw.map(normalize_employment)
    work["Employment Norm"] = (
        collapse_top_categories(emp_norm, max_groups=7, min_count=12, other_label="Other")
        .astype("string")
        .fillna("Unknown")
    )

    nat_raw = coalesce_text_columns(work, ["Nationality", "Nationality (Demographics)"])
    nat_norm = nat_raw.map(normalize_nationality)
    work["Nationality Top"] = (
        collapse_top_categories(nat_norm, max_groups=8, min_count=15, other_label="Other")
        .astype("string")
        .fillna("Unknown")
    )

    eth_raw = coalesce_text_columns(
        work,
        [
            "Ethnicity simplified",
            "Ethnicity simplified (Demographics)",
            "Ethnicity",
            "Ethnicity (Demographics)",
        ],
    )
    eth_norm = eth_raw.map(normalize_ethnicity)
    work["Ethnicity Norm"] = (
        collapse_top_categories(eth_norm, max_groups=7, min_count=12, other_label="Other")
        .astype("string")
        .fillna("Unknown")
    )

    return work


def condition_welch_test(
    data: pd.DataFrame,
    value_col: str,
    mask: pd.Series | None = None,
) -> dict[str, float]:
    work = data[mask].copy() if mask is not None else data.copy()

    interactive_vals = work.loc[work["Condition"] == "Interactive", value_col].dropna().astype(float)
    text_vals = work.loc[work["Condition"] == "Text", value_col].dropna().astype(float)

    i_mean, i_low, i_high = mean_ci(interactive_vals)
    t_mean, t_low, t_high = mean_ci(text_vals)

    if len(interactive_vals) < 2 or len(text_vals) < 2:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat, p_value = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")

    return {
        "n_interactive": float(len(interactive_vals)),
        "n_text": float(len(text_vals)),
        "mean_interactive": i_mean,
        "mean_text": t_mean,
        "ci_low_interactive": i_low,
        "ci_high_interactive": i_high,
        "ci_low_text": t_low,
        "ci_high_text": t_high,
        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
        "p": float(p_value) if pd.notna(p_value) else np.nan,
        "delta_interactive_minus_text": i_mean - t_mean,
    }


def paired_analytical_vs_emotional_by_condition(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for condition in CONDITION_ORDER:
        pair = data.loc[
            data["Condition"] == condition,
            ["analytical_change", "emotional_change"],
        ].dropna()

        if len(pair) < 2:
            t_stat = np.nan
            p_value = np.nan
        else:
            t_stat, p_value = ttest_rel(
                pair["analytical_change"].astype(float),
                pair["emotional_change"].astype(float),
                nan_policy="omit",
            )

        rows.append(
            {
                "Condition": condition,
                "n": float(len(pair)),
                "mean_analytical": float(pair["analytical_change"].mean()) if len(pair) else np.nan,
                "mean_emotional": float(pair["emotional_change"].mean()) if len(pair) else np.nan,
                "mean_diff": float((pair["analytical_change"] - pair["emotional_change"]).mean())
                if len(pair)
                else np.nan,
                "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                "p": float(p_value) if pd.notna(p_value) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def interaction_effect(data: pd.DataFrame, value_col: str) -> dict[str, float]:
    fit_df = data[[value_col, "condition_bin", "group_bin"]].dropna().copy()
    y = fit_df[value_col].astype(float).to_numpy()
    condition = fit_df["condition_bin"].to_numpy(dtype=float)
    group = fit_df["group_bin"].to_numpy(dtype=float)
    interaction = condition * group

    X = np.column_stack([np.ones(len(fit_df)), condition, group, interaction])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    rank = np.linalg.matrix_rank(X)
    dof = n - rank

    if dof <= 0:
        return {
            "coef": np.nan,
            "se": np.nan,
            "t": np.nan,
            "df": np.nan,
            "p": np.nan,
        }

    rss = float(np.sum(residuals**2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    coef = float(beta[3])
    se_val = float(se[3]) if len(se) > 3 else np.nan
    t_stat = coef / se_val if pd.notna(se_val) and not np.isclose(se_val, 0) else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if pd.notna(t_stat) else np.nan
    return {
        "coef": coef,
        "se": se_val,
        "t": float(t_stat),
        "df": float(dof),
        "p": float(p_value),
    }


def demographic_adjusted_condition_effect(data: pd.DataFrame, value_col: str) -> dict[str, float]:
    demographic_cols = [
        "Age Numeric",
        "Gender Norm",
        "AI Knowledge Norm",
        "Education Norm",
        "Employment Norm",
        "Nationality Top",
        "Ethnicity Norm",
    ]

    model_df = data[[value_col, "condition_bin"] + demographic_cols].copy()
    model_df = model_df.dropna(subset=[value_col, "condition_bin"]).copy()

    if model_df.empty:
        return {
            "n": 0.0,
            "coef": np.nan,
            "se": np.nan,
            "t": np.nan,
            "df": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "k": np.nan,
        }

    age = model_df["Age Numeric"].astype(float)
    age = age.fillna(float(age.median()) if age.notna().any() else 0.0)

    cat_df = pd.DataFrame(
        {
            col: model_df[col].astype("string").fillna("Unknown")
            for col in demographic_cols
            if col != "Age Numeric"
        }
    )
    dummies = pd.get_dummies(cat_df, drop_first=True, dtype=float)

    y = model_df[value_col].astype(float).to_numpy()
    condition = model_df["condition_bin"].to_numpy(dtype=float)

    if dummies.shape[1] == 0:
        X = np.column_stack([np.ones(len(model_df)), condition, age.to_numpy(dtype=float)])
    else:
        X = np.column_stack(
            [
                np.ones(len(model_df)),
                condition,
                age.to_numpy(dtype=float),
                dummies.to_numpy(dtype=float),
            ]
        )

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted

    n, p = X.shape
    rank = np.linalg.matrix_rank(X)
    dof = n - rank

    if dof <= 0:
        return {
            "n": float(n),
            "coef": np.nan,
            "se": np.nan,
            "t": np.nan,
            "df": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "k": float(p - 1),
        }

    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    cond_coef = float(beta[1])
    cond_se = float(se[1])
    cond_t = cond_coef / cond_se if not np.isclose(cond_se, 0) else np.nan
    cond_p = 2 * (1 - stats.t.cdf(abs(cond_t), dof)) if pd.notna(cond_t) else np.nan

    r2 = 1 - rss / tss if not np.isclose(tss, 0) else np.nan

    return {
        "n": float(n),
        "coef": cond_coef,
        "se": cond_se,
        "t": float(cond_t),
        "df": float(dof),
        "p": float(cond_p),
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "k": float(p - 1),
    }


def demographic_stratified_tests(
    data: pd.DataFrame,
    value_col: str,
    demographic_col: str,
    min_per_condition: int = 12,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    level_counts = data[demographic_col].astype("string").value_counts(dropna=False)
    for level in level_counts.index.tolist():
        sub = data[data[demographic_col].astype("string") == level]

        interactive_vals = sub.loc[sub["Condition"] == "Interactive", value_col].dropna().astype(float)
        text_vals = sub.loc[sub["Condition"] == "Text", value_col].dropna().astype(float)

        if len(interactive_vals) < min_per_condition or len(text_vals) < min_per_condition:
            continue

        t_stat, p_val = ttest_ind(interactive_vals, text_vals, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "demographic": demographic_col,
                "level": str(level),
                "metric": value_col,
                "n_interactive": float(len(interactive_vals)),
                "n_text": float(len(text_vals)),
                "mean_interactive": float(interactive_vals.mean()),
                "mean_text": float(text_vals.mean()),
                "delta_interactive_minus_text": float(interactive_vals.mean() - text_vals.mean()),
                "t": float(t_stat),
                "p": float(p_val),
            }
        )

    return pd.DataFrame(rows)


def collect_demographic_stratified_significant(
    data: pd.DataFrame,
    value_col: str,
    min_per_condition: int = 12,
) -> pd.DataFrame:
    demographic_cols = [
        "Age Group",
        "Gender Norm",
        "AI Knowledge Norm",
        "Education Norm",
        "Employment Norm",
        "Nationality Top",
        "Ethnicity Norm",
    ]

    all_rows: list[pd.DataFrame] = []
    for col in demographic_cols:
        result = demographic_stratified_tests(
            data=data,
            value_col=value_col,
            demographic_col=col,
            min_per_condition=min_per_condition,
        )
        if not result.empty:
            all_rows.append(result)

    if not all_rows:
        return pd.DataFrame()

    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.sort_values("p", ascending=True)
    return merged


def add_pair_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    star_text: str,
    y_step: float,
) -> None:
    ax.plot([x1, x1, x2, x2], [y - y_step, y, y, y - y_step], color="black", lw=2.0, zorder=4)
    ax.text((x1 + x2) / 2, y + 0.6 * y_step, star_text or "ns", ha="center", va="bottom", fontsize=18)


def draw_overall_condition_subplot(
    ax: plt.Axes,
    analytical_test: dict[str, float],
    emotional_test: dict[str, float],
) -> None:
    metric_order = ["Analytical", "Emotional"]
    test_map = {
        "Analytical": analytical_test,
        "Emotional": emotional_test,
    }

    x_base = np.arange(len(metric_order), dtype=float)
    width = 0.34
    offsets = {
        "Interactive": -width / 2,
        "Text": width / 2,
    }

    for condition in CONDITION_ORDER:
        for idx, metric_name in enumerate(metric_order):
            test_res = test_map[metric_name]
            if condition == "Interactive":
                mean_val = float(test_res["mean_interactive"])
                ci_low = float(test_res["ci_low_interactive"])
                ci_high = float(test_res["ci_high_interactive"])
            else:
                mean_val = float(test_res["mean_text"])
                ci_low = float(test_res["ci_low_text"])
                ci_high = float(test_res["ci_high_text"])

            x_val = x_base[idx] + offsets[condition]
            label = CONDITION_LABELS[condition] if idx == 0 else None
            ax.bar(
                x_val,
                mean_val,
                width=width,
                color=CONDITION_COLORS[condition],
                edgecolor="none",
                label=label,
                zorder=2,
            )
            ax.errorbar(
                x_val,
                mean_val,
                yerr=[[mean_val - ci_low], [ci_high - mean_val]],
                fmt="none",
                ecolor="#4a4a4a",
                elinewidth=3.0,
                capsize=7,
                capthick=3.0,
                zorder=3,
            )

    all_lows = [
        analytical_test["ci_low_interactive"],
        analytical_test["ci_low_text"],
        emotional_test["ci_low_interactive"],
        emotional_test["ci_low_text"],
    ]
    all_highs = [
        analytical_test["ci_high_interactive"],
        analytical_test["ci_high_text"],
        emotional_test["ci_high_interactive"],
        emotional_test["ci_high_text"],
    ]
    y_min = float(np.nanmin(all_lows))
    y_max = float(np.nanmax(all_highs))
    y_span = max(y_max - y_min, 0.2)

    add_pair_bracket(
        ax,
        x_base[0] + offsets["Interactive"],
        x_base[0] + offsets["Text"],
        y_max + 0.10 * y_span,
        significance_stars(float(analytical_test["p"])),
        0.03 * y_span,
    )
    add_pair_bracket(
        ax,
        x_base[1] + offsets["Interactive"],
        x_base[1] + offsets["Text"],
        y_max + 0.22 * y_span,
        significance_stars(float(emotional_test["p"])),
        0.03 * y_span,
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels(metric_order, fontsize=13)
    ax.set_ylabel("Mean Change (Post - Pre, per-item)")
    ax.set_title("6.1 Condition Effect Across Trust Types")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(y_min - 0.18 * y_span, y_max + 0.40 * y_span)


def draw_group_condition_subplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    group_tests_analytical: pd.DataFrame,
    group_tests_emotional: pd.DataFrame,
    interaction_analytical: dict[str, float],
    interaction_emotional: dict[str, float],
) -> None:
    categories = [("WEIRD", "Interactive"), ("WEIRD", "Text"), ("NORMAL", "Interactive"), ("NORMAL", "Text")]
    labels = ["WEIRD\nInteractive", "WEIRD\nText", "NORMAL\nInteractive", "NORMAL\nText"]
    x_base = np.arange(len(categories), dtype=float)
    width = 0.36

    for metric, offset, color in [
        ("analytical_change", -width / 2, METRIC_COLORS["analytical_change"]),
        ("emotional_change", width / 2, METRIC_COLORS["emotional_change"]),
    ]:
        for idx, (group, condition) in enumerate(categories):
            vals = data.loc[(data["Group"] == group) & (data["Condition"] == condition), metric].dropna()
            mean_val, ci_low, ci_high = mean_ci(vals)
            x_val = x_base[idx] + offset
            label = "Analytical" if (idx == 0 and metric == "analytical_change") else None
            if idx == 0 and metric == "emotional_change":
                label = "Emotional"

            ax.bar(
                x_val,
                mean_val,
                width=width,
                color=color,
                edgecolor="none",
                label=label,
                zorder=2,
            )
            ax.errorbar(
                x_val,
                mean_val,
                yerr=[[mean_val - ci_low], [ci_high - mean_val]],
                fmt="none",
                ecolor="#4a4a4a",
                elinewidth=2.6,
                capsize=6,
                capthick=2.6,
                zorder=3,
            )

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Change")
    ax.set_title("6.2 Condition Effect Within WEIRD vs NORMAL")
    ax.legend(loc="lower right", fontsize=11)

    text_lines = [
        "Condition Welch p within group:",
        (
            f"WEIRD: analytical p {format_p(float(group_tests_analytical.loc[group_tests_analytical['Group']=='WEIRD', 'p'].iloc[0]))}, "
            f"emotional p {format_p(float(group_tests_emotional.loc[group_tests_emotional['Group']=='WEIRD', 'p'].iloc[0]))}"
        ),
        (
            f"NORMAL: analytical p {format_p(float(group_tests_analytical.loc[group_tests_analytical['Group']=='NORMAL', 'p'].iloc[0]))}, "
            f"emotional p {format_p(float(group_tests_emotional.loc[group_tests_emotional['Group']=='NORMAL', 'p'].iloc[0]))}"
        ),
        (
            f"Interaction p (Analytical): {format_p(float(interaction_analytical['p']))}; "
            f"(Emotional): {format_p(float(interaction_emotional['p']))}"
        ),
    ]

    ax.text(
        0.01,
        0.99,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def draw_diff_subplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    diff_condition_test: dict[str, float],
    paired_df: pd.DataFrame,
) -> None:
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    means = []
    lows = []
    highs = []

    for condition in CONDITION_ORDER:
        vals = data.loc[data["Condition"] == condition, "analytical_vs_emotional_diff"].dropna()
        mean_val, ci_low, ci_high = mean_ci(vals)
        means.append(mean_val)
        lows.append(ci_low)
        highs.append(ci_high)

    means_arr = np.array(means, dtype=float)
    lows_arr = np.array(lows, dtype=float)
    highs_arr = np.array(highs, dtype=float)

    ax.bar(
        x,
        means_arr,
        width=0.58,
        color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        means_arr,
        yerr=[means_arr - lows_arr, highs_arr - means_arr],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=3.0,
        capsize=7,
        capthick=3.0,
        zorder=3,
    )

    y_min = float(np.nanmin(lows_arr))
    y_max = float(np.nanmax(highs_arr))
    y_span = max(y_max - y_min, 0.2)

    add_pair_bracket(
        ax,
        x[0],
        x[1],
        y_max + 0.10 * y_span,
        significance_stars(float(diff_condition_test["p"])),
        0.03 * y_span,
    )

    paired_i = paired_df[paired_df["Condition"] == "Interactive"].iloc[0]
    paired_t = paired_df[paired_df["Condition"] == "Text"].iloc[0]

    text = (
        f"Paired analytical vs emotional:\n"
        f"Interactive p {format_p(float(paired_i['p']))}; "
        f"Text p {format_p(float(paired_t['p']))}\n"
        f"Condition Welch on difference p {format_p(float(diff_condition_test['p']))}"
    )

    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )

    ax.axhline(0, color="#4a4a4a", linewidth=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=12)
    ax.set_ylabel("Analytical - Emotional Change")
    ax.set_title("6.3 Analytical vs Emotional Difference")
    ax.set_ylim(y_min - 0.18 * y_span, y_max + 0.36 * y_span)


def write_report(
    overall_analytical: dict[str, float],
    overall_emotional: dict[str, float],
    adjusted_analytical: dict[str, float],
    adjusted_emotional: dict[str, float],
    strat_analytical_sig: pd.DataFrame,
    strat_emotional_sig: pd.DataFrame,
    group_tests_analytical: pd.DataFrame,
    group_tests_emotional: pd.DataFrame,
    interaction_analytical: dict[str, float],
    interaction_emotional: dict[str, float],
    paired_df: pd.DataFrame,
    diff_condition_test: dict[str, float],
) -> str:
    lines: list[str] = []

    lines.append("\\subsubsection{Hypothesis 6 Results}")
    lines.append(
        "Hypothesis 6 tested whether Interactive and Static (Text) explanations produced different analytical and emotional trust-change outcomes, including demographic coverage, WEIRD/NORMAL subgroup tests, and analytical-vs-emotional comparisons."
    )
    lines.append("")

    lines.append("6.1 Across both trust types and demographics:")
    lines.append(
        "Overall condition effect (analytical): "
        f"$M_{{Interactive}}={overall_analytical['mean_interactive']:.3f}$, "
        f"$M_{{Text}}={overall_analytical['mean_text']:.3f}$, "
        f"$t={overall_analytical['t']:.3f}$, "
        f"$p {format_p(overall_analytical['p'])}$."
    )
    lines.append(
        "Overall condition effect (emotional): "
        f"$M_{{Interactive}}={overall_emotional['mean_interactive']:.3f}$, "
        f"$M_{{Text}}={overall_emotional['mean_text']:.3f}$, "
        f"$t={overall_emotional['t']:.3f}$, "
        f"$p {format_p(overall_emotional['p'])}$."
    )
    lines.append(
        "Demographic-adjusted OLS condition effect (analytical): "
        f"$b={adjusted_analytical['coef']:.3f}$, "
        f"$SE={adjusted_analytical['se']:.3f}$, "
        f"$t({int(adjusted_analytical['df'])})={adjusted_analytical['t']:.3f}$, "
        f"$p {format_p(adjusted_analytical['p'])}$, "
        f"$R^2={adjusted_analytical['r2']:.3f}$."
    )
    lines.append(
        "Demographic-adjusted OLS condition effect (emotional): "
        f"$b={adjusted_emotional['coef']:.3f}$, "
        f"$SE={adjusted_emotional['se']:.3f}$, "
        f"$t({int(adjusted_emotional['df'])})={adjusted_emotional['t']:.3f}$, "
        f"$p {format_p(adjusted_emotional['p'])}$, "
        f"$R^2={adjusted_emotional['r2']:.3f}$."
    )

    lines.append("Significant demographic-stratified Interactive vs Text effects (p < .05):")
    if strat_analytical_sig.empty and strat_emotional_sig.empty:
        lines.append("- None detected under the minimum per-condition sample threshold.")
    else:
        if not strat_analytical_sig.empty:
            lines.append("- Analytical change:")
            for _, row in strat_analytical_sig.head(10).iterrows():
                lines.append(
                    f"  {row['demographic']}={row['level']}: "
                    f"$\\Delta(M_I-M_T)={row['delta_interactive_minus_text']:.3f}$, "
                    f"$t={row['t']:.3f}$, $p {format_p(row['p'])}$."
                )
        if not strat_emotional_sig.empty:
            lines.append("- Emotional change:")
            for _, row in strat_emotional_sig.head(10).iterrows():
                lines.append(
                    f"  {row['demographic']}={row['level']}: "
                    f"$\\Delta(M_I-M_T)={row['delta_interactive_minus_text']:.3f}$, "
                    f"$t={row['t']:.3f}$, $p {format_p(row['p'])}$."
                )

    lines.append("")
    lines.append("6.2 Between WEIRD and NORMAL:")

    for metric_name, metric_df in [
        ("Analytical", group_tests_analytical),
        ("Emotional", group_tests_emotional),
    ]:
        weird_row = metric_df[metric_df["Group"] == "WEIRD"].iloc[0]
        normal_row = metric_df[metric_df["Group"] == "NORMAL"].iloc[0]
        lines.append(
            f"{metric_name} condition effect within WEIRD: "
            f"$M_I={weird_row['mean_interactive']:.3f}$, $M_T={weird_row['mean_text']:.3f}$, "
            f"$t={weird_row['t']:.3f}$, $p {format_p(weird_row['p'])}$."
        )
        lines.append(
            f"{metric_name} condition effect within NORMAL: "
            f"$M_I={normal_row['mean_interactive']:.3f}$, $M_T={normal_row['mean_text']:.3f}$, "
            f"$t={normal_row['t']:.3f}$, $p {format_p(normal_row['p'])}$."
        )

    lines.append(
        "Analytical group x condition interaction: "
        f"$b={interaction_analytical['coef']:.3f}$, "
        f"$t({int(interaction_analytical['df'])})={interaction_analytical['t']:.3f}$, "
        f"$p {format_p(interaction_analytical['p'])}$."
    )
    lines.append(
        "Emotional group x condition interaction: "
        f"$b={interaction_emotional['coef']:.3f}$, "
        f"$t({int(interaction_emotional['df'])})={interaction_emotional['t']:.3f}$, "
        f"$p {format_p(interaction_emotional['p'])}$."
    )

    lines.append("")
    lines.append("6.3 Between analytical vs emotional trust:")

    for _, row in paired_df.iterrows():
        lines.append(
            f"{CONDITION_LABELS[row['Condition']]} paired analytical vs emotional: "
            f"$M_A={row['mean_analytical']:.3f}$, "
            f"$M_E={row['mean_emotional']:.3f}$, "
            f"$t({int(row['n']) - 1})={row['t']:.3f}$, "
            f"$p {format_p(row['p'])}$."
        )

    lines.append(
        "Condition contrast on analytical-emotional difference score: "
        f"$M_I={diff_condition_test['mean_interactive']:.3f}$, "
        f"$M_T={diff_condition_test['mean_text']:.3f}$, "
        f"$t={diff_condition_test['t']:.3f}$, "
        f"$p {format_p(diff_condition_test['p'])}$."
    )

    lines.append("Asterisk key: * p < .05, ** p < .01, *** p < .001.")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    analysis = build_analysis_frame(data)

    # 6.1 Across trust types and demographics.
    overall_analytical = condition_welch_test(analysis, "analytical_change")
    overall_emotional = condition_welch_test(analysis, "emotional_change")

    adjusted_analytical = demographic_adjusted_condition_effect(analysis, "analytical_change")
    adjusted_emotional = demographic_adjusted_condition_effect(analysis, "emotional_change")

    strat_analytical = collect_demographic_stratified_significant(analysis, "analytical_change", min_per_condition=12)
    strat_emotional = collect_demographic_stratified_significant(analysis, "emotional_change", min_per_condition=12)

    strat_analytical_sig = strat_analytical[strat_analytical["p"] < ALPHA].copy() if not strat_analytical.empty else strat_analytical
    strat_emotional_sig = strat_emotional[strat_emotional["p"] < ALPHA].copy() if not strat_emotional.empty else strat_emotional

    # 6.2 Between WEIRD and NORMAL.
    group_tests_analytical_rows: list[dict[str, float | str]] = []
    group_tests_emotional_rows: list[dict[str, float | str]] = []

    for group in GROUP_ORDER:
        group_mask = analysis["Group"] == group
        a_test = condition_welch_test(analysis, "analytical_change", mask=group_mask)
        e_test = condition_welch_test(analysis, "emotional_change", mask=group_mask)

        group_tests_analytical_rows.append({"Group": group, **a_test})
        group_tests_emotional_rows.append({"Group": group, **e_test})

    group_tests_analytical = pd.DataFrame(group_tests_analytical_rows)
    group_tests_emotional = pd.DataFrame(group_tests_emotional_rows)

    interaction_analytical = interaction_effect(analysis, "analytical_change")
    interaction_emotional = interaction_effect(analysis, "emotional_change")

    # 6.3 Analytical vs emotional trust.
    paired_df = paired_analytical_vs_emotional_by_condition(analysis)
    diff_condition_test = condition_welch_test(analysis, "analytical_vs_emotional_diff")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), constrained_layout=True)

    draw_overall_condition_subplot(axes[0], overall_analytical, overall_emotional)
    draw_group_condition_subplot(
        axes[1],
        analysis,
        group_tests_analytical,
        group_tests_emotional,
        interaction_analytical,
        interaction_emotional,
    )
    draw_diff_subplot(axes[2], analysis, diff_condition_test, paired_df)

    axes[0].text(
        0.01,
        0.99,
        "Asterisks: * p < .05, ** p < .01, *** p < .001",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="black",
    )

    fig.suptitle("Hypothesis 6: Interactive vs Static Effects Across Trust Dimensions", fontsize=24)
    fig.savefig(PLOT_OUTPUT_PATH, dpi=160)
    plt.close(fig)

    report_text = write_report(
        overall_analytical=overall_analytical,
        overall_emotional=overall_emotional,
        adjusted_analytical=adjusted_analytical,
        adjusted_emotional=adjusted_emotional,
        strat_analytical_sig=strat_analytical_sig,
        strat_emotional_sig=strat_emotional_sig,
        group_tests_analytical=group_tests_analytical,
        group_tests_emotional=group_tests_emotional,
        interaction_analytical=interaction_analytical,
        interaction_emotional=interaction_emotional,
        paired_df=paired_df,
        diff_condition_test=diff_condition_test,
    )
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print("Saved plot:", PLOT_OUTPUT_PATH)
    print("Saved report:", REPORT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
