from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from Results.qualitative.sentiment1 import (
        DATA_PATH,
        FIGURES_DIR,
        METRICS_TEXT_COLS,
        build_analysis_frame,
        get_sentiment_engine,
        mean_ci,
        normalize_text,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from Results.qualitative.sentiment1 import (
        DATA_PATH,
        FIGURES_DIR,
        METRICS_TEXT_COLS,
        build_analysis_frame,
        get_sentiment_engine,
        mean_ci,
        normalize_text,
    )

ic_mod = import_module("Results.qualitative.1c3")

OUTCOMES = ["analytical_change", "overall_change", "emotional_change"]
OUTCOME_LABELS = {
    "overall_change": "Overall",
    "analytical_change": "Analytical",
    "emotional_change": "Emotional",
}


def profile_mean(df: pd.DataFrame, group_col: str, group_value: str) -> np.ndarray:
    return np.array(
        [df.loc[df[group_col] == group_value, outcome].mean() for outcome in OUTCOMES],
        dtype=float,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.nan
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.dot(a, b) / denom)


def add_ic_orientation(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    ic_columns: list[str] = []

    for source_col, suffix in METRICS_TEXT_COLS.items():
        if source_col not in work.columns:
            continue
        score_col = f"{suffix}_ic"
        work[score_col] = work[source_col].map(normalize_text).map(ic_mod.score_ic)
        ic_columns.append(score_col)

    if not ic_columns:
        raise ValueError("No IC source text columns found in Metrics.csv")

    work["ic_mean"] = work[ic_columns].mean(axis=1, skipna=True)
    work["ic_group"] = ic_mod.assign_ic_groups_terciles(work["ic_mean"])
    return work


def add_sentiment_polarity(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["sentiment_polarity"] = "neutral"
    work.loc[work["sentiment_mean"] < 0.0, "sentiment_polarity"] = "negative"
    work.loc[work["sentiment_mean"] > 0.0, "sentiment_polarity"] = "positive"
    return work


def compute_group_lines(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    weird = np.array([df.loc[df["Group"] == "WEIRD", outcome].mean() for outcome in OUTCOMES], dtype=float)
    non_weird = np.array([df.loc[df["Group"] == "NORMAL", outcome].mean() for outcome in OUTCOMES], dtype=float)
    return weird, non_weird


def plot_two_group_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    left_label: str,
    right_label: str,
    title: str,
    weird_line: np.ndarray,
    non_weird_line: np.ndarray,
    left_color: str,
    right_color: str,
    show_legend: bool = False,
    legend_groups_only: bool = False,
) -> None:
    x = np.arange(len(OUTCOMES), dtype=float)
    width = 0.34

    left_means: list[float] = []
    left_err_low: list[float] = []
    left_err_high: list[float] = []
    right_means: list[float] = []
    right_err_low: list[float] = []
    right_err_high: list[float] = []

    for outcome in OUTCOMES:
        left_vals = df.loc[df[group_col] == left_label, outcome]
        right_vals = df.loc[df[group_col] == right_label, outcome]

        l_mean, l_low, l_high = mean_ci(left_vals)
        r_mean, r_low, r_high = mean_ci(right_vals)

        left_means.append(l_mean)
        left_err_low.append(l_mean - l_low)
        left_err_high.append(l_high - l_mean)

        right_means.append(r_mean)
        right_err_low.append(r_mean - r_low)
        right_err_high.append(r_high - r_mean)

    left_bar = ax.bar(
        x - width / 2,
        left_means,
        width=width,
        color=left_color,
        label=left_label.capitalize(),
        yerr=[left_err_low, left_err_high],
        capsize=4,
        edgecolor="white",
    )
    right_bar = ax.bar(
        x + width / 2,
        right_means,
        width=width,
        color=right_color,
        label=right_label.capitalize(),
        yerr=[right_err_low, right_err_high],
        capsize=4,
        edgecolor="white",
    )

    ax.plot(x, weird_line, color="#222222", marker="o", linewidth=1.8, label="WEIRD mean")
    ax.plot(x, non_weird_line, color="#555555", marker="o", linestyle="--", linewidth=1.8, label="non-WEIRD mean")

    ax.axhline(0.0, color="#333333", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES], fontsize=11)
    ax.set_title(title, fontsize=14)

    if show_legend:
        if legend_groups_only:
            ax.legend([left_bar, right_bar], [left_label.capitalize(), right_label.capitalize()], loc="lower left", fontsize=9)
        else:
            ax.legend(loc="lower left", fontsize=9)


def plot_similarity_panel(
    ax: plt.Axes,
    left_profile: np.ndarray,
    right_profile: np.ndarray,
    weird_profile: np.ndarray,
    non_weird_profile: np.ndarray,
    left_label: str,
    right_label: str,
    title: str,
) -> None:
    weird_left = cosine_similarity(weird_profile, left_profile)
    weird_right = cosine_similarity(weird_profile, right_profile)
    non_left = cosine_similarity(non_weird_profile, left_profile)
    non_right = cosine_similarity(non_weird_profile, right_profile)

    x = np.arange(2, dtype=float)
    width = 0.32

    ax.bar(
        x - width / 2,
        [weird_left, non_left],
        width=width,
        color="#377eb8",
        edgecolor="white",
        label=left_label.capitalize(),
    )
    ax.bar(
        x + width / 2,
        [weird_right, non_right],
        width=width,
        color="#c26a26",
        edgecolor="white",
        label=right_label.capitalize(),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["WEIRD", "non-WEIRD"], fontsize=11)
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)

    ax.text(
        0.02,
        0.98,
        (
            f"WEIRD: {left_label[:3]}={weird_left:.3f}, {right_label[:3]}={weird_right:.3f}\n"
            f"non-WEIRD: {left_label[:3]}={non_left:.3f}, {right_label[:3]}={non_right:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def participant_similarity_frame(
    df: pd.DataFrame,
    left_profile: np.ndarray,
    right_profile: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        vec = np.array([row[outcome] for outcome in OUTCOMES], dtype=float)
        rows.append(
            {
                "Group": row["Group"],
                "sim_left": cosine_similarity(vec, left_profile),
                "sim_right": cosine_similarity(vec, right_profile),
            }
        )
    return pd.DataFrame(rows)


def build_profile_contrast_frame(
    df: pd.DataFrame,
    negative_profile: np.ndarray,
    positive_profile: np.ndarray,
    individualistic_profile: np.ndarray,
    collectivist_profile: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        vec = np.array([row[outcome] for outcome in OUTCOMES], dtype=float)
        sim_negative = cosine_similarity(vec, negative_profile)
        sim_positive = cosine_similarity(vec, positive_profile)
        sim_individualistic = cosine_similarity(vec, individualistic_profile)
        sim_collectivist = cosine_similarity(vec, collectivist_profile)

        rows.append(
            {
                "Group": row["Group"],
                "sentiment_contrast": sim_positive - sim_negative,
                "ic_contrast": sim_individualistic - sim_collectivist,
                "interaction_contrast": 0.5 * (sim_positive + sim_individualistic)
                - 0.5 * (sim_negative + sim_collectivist),
            }
        )
    return pd.DataFrame(rows)


def welch_group_test(values_a: pd.Series, values_b: pd.Series) -> float:
    clean_a = values_a.dropna().astype(float)
    clean_b = values_b.dropna().astype(float)
    if len(clean_a) < 2 or len(clean_b) < 2:
        return np.nan
    _, p_value = stats.ttest_ind(clean_a, clean_b, equal_var=False, nan_policy="omit")
    return float(p_value)


def add_sig_bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str) -> None:
    height = 0.035
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y], color="#333333", linewidth=1.1)
    ax.text((x0 + x1) / 2, y + height + 0.015, text, ha="center", va="bottom", fontsize=9)


def plot_profile_summary_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    negative_profile: np.ndarray,
    positive_profile: np.ndarray,
    individualistic_profile: np.ndarray,
    collectivist_profile: np.ndarray,
) -> None:
    contrast_df = build_profile_contrast_frame(
        df,
        negative_profile=negative_profile,
        positive_profile=positive_profile,
        individualistic_profile=individualistic_profile,
        collectivist_profile=collectivist_profile,
    )

    group_order = ["WEIRD", "NORMAL"]
    group_labels = ["WEIRD", "non-WEIRD"]
    categories = [
        ("ic_contrast", "IC"),
        ("sentiment_contrast", "Sentiment"),
        ("interaction_contrast", "Interaction"),
    ]
    x = np.arange(len(categories), dtype=float)
    offset = 0.13

    weird_means: list[float] = []
    weird_err_low: list[float] = []
    weird_err_high: list[float] = []
    non_means: list[float] = []
    non_err_low: list[float] = []
    non_err_high: list[float] = []

    for column, _ in categories:
        weird_vals = contrast_df.loc[contrast_df["Group"] == "WEIRD", column]
        non_vals = contrast_df.loc[contrast_df["Group"] == "NORMAL", column]

        weird_mean, weird_low, weird_high = mean_ci(weird_vals)
        non_mean, non_low, non_high = mean_ci(non_vals)

        weird_means.append(weird_mean)
        weird_err_low.append(weird_mean - weird_low)
        weird_err_high.append(weird_high - weird_mean)
        non_means.append(non_mean)
        non_err_low.append(non_mean - non_low)
        non_err_high.append(non_high - non_mean)

    ax.errorbar(
        x - offset,
        weird_means,
        yerr=[weird_err_low, weird_err_high],
        fmt="o-",
        color="#222222",
        ecolor="#222222",
        elinewidth=1.4,
        capsize=4,
        markersize=7,
        linewidth=1.6,
        label="WEIRD",
    )
    ax.errorbar(
        x + offset,
        non_means,
        yerr=[non_err_low, non_err_high],
        fmt="o--",
        color="#666666",
        ecolor="#666666",
        elinewidth=1.4,
        capsize=4,
        markersize=7,
        linewidth=1.6,
        label="non-WEIRD",
    )

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in categories], fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Profile contrast", fontsize=11)
    ax.set_title("Profile Contrast Summary", fontsize=14)
    ax.legend(loc="lower left", fontsize=9)

    for idx, (column, _) in enumerate(categories):
        weird_vals = contrast_df.loc[contrast_df["Group"] == "WEIRD", column]
        non_vals = contrast_df.loc[contrast_df["Group"] == "NORMAL", column]
        p_value = welch_group_test(weird_vals, non_vals)
        p_text = f"p {ic_mod.format_p_text(p_value)} ({ic_mod.significance_stars(p_value)})"
        y_top = max(
            weird_means[idx] + weird_err_high[idx],
            non_means[idx] + non_err_high[idx],
        ) + 0.07
        add_sig_bracket(ax, x[idx] - offset, x[idx] + offset, y_top, p_text)

def make_qualitative_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), constrained_layout=True)

    weird_line, non_weird_line = compute_group_lines(df)
    weird_profile = profile_mean(df, "Group", "WEIRD")
    non_weird_profile = profile_mean(df, "Group", "NORMAL")

    sentiment_df = df[df["sentiment_polarity"].isin(["negative", "positive"])].copy()
    ic_df = df[df["ic_group"].isin(["individualistic", "collectivist"])].copy()

    negative_profile = profile_mean(sentiment_df, "sentiment_polarity", "negative")
    positive_profile = profile_mean(sentiment_df, "sentiment_polarity", "positive")
    individualistic_profile = profile_mean(ic_df, "ic_group", "individualistic")
    collectivist_profile = profile_mean(ic_df, "ic_group", "collectivist")

    plot_two_group_panel(
        axes[0],
        sentiment_df,
        group_col="sentiment_polarity",
        left_label="negative",
        right_label="positive",
        title="Negative vs Positive Sentiment",
        weird_line=weird_line,
        non_weird_line=non_weird_line,
        left_color="#377eb8",
        right_color="#c26a26",
        show_legend=True,
    )

    plot_two_group_panel(
        axes[1],
        ic_df,
        group_col="ic_group",
        left_label="individualistic",
        right_label="collectivist",
        title="Individualistic vs Collectivist",
        weird_line=weird_line,
        non_weird_line=non_weird_line,
        left_color="#c26a26",
        right_color="#377eb8",
        show_legend=True,
        legend_groups_only=True,
    )

    plot_profile_summary_panel(
        axes[2],
        df,
        negative_profile=negative_profile,
        positive_profile=positive_profile,
        individualistic_profile=individualistic_profile,
        collectivist_profile=collectivist_profile,
    )

    axes[0].set_ylabel("Normalized trust change (z-score)", fontsize=13)
    axes[1].set_ylabel("Normalized trust change (z-score)", fontsize=13)
    fig.suptitle("Qualitative Trust-Change and Profile Summary", fontsize=18)

    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    engine = get_sentiment_engine()
    analysis_df = build_analysis_frame(raw, engine)
    analysis_df = add_ic_orientation(analysis_df)
    analysis_df = add_sentiment_polarity(analysis_df)

    output_path = FIGURES_DIR / "qualitative.png"
    make_qualitative_plot(analysis_df, output_path)

    print("Qualitative trust-change figure")
    print("==============================")
    print(f"Participants total: {len(analysis_df)}")
    print(
        "Sentiment groups: "
        f"negative={(analysis_df['sentiment_polarity'] == 'negative').sum()}, "
        f"positive={(analysis_df['sentiment_polarity'] == 'positive').sum()}, "
        f"neutral={(analysis_df['sentiment_polarity'] == 'neutral').sum()}"
    )
    print(
        "IC groups: "
        f"individualistic={(analysis_df['ic_group'] == 'individualistic').sum()}, "
        f"middle={(analysis_df['ic_group'] == 'middle').sum()}, "
        f"collectivist={(analysis_df['ic_group'] == 'collectivist').sum()}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
