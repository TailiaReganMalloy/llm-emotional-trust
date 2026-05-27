"""Bias analysis: across-group comparison of emotional and analytical trust change by
age, race, gender,
and all pairwise combinations.

Layout of the single output figure (2 rows × 3 columns):
  Row 0: Age  |  Race  |  Gender
  Row 1: Age × Race  |  Age × Gender  |  Race × Gender
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPO_ROOT / "Figures"
PNG_PATH = OUTPUT_DIR / "bias.png"
LEGACY_PNG_PATH = FIGURES_DIR / "bias.png"
TXT_PATH = OUTPUT_DIR / "bias.txt"
ALPHA = 0.05

MIN_N = 8  # minimum group size to include in plot / test


# ── palette: each demographic gets a colour family ──────────────────────────
PALETTE_AGE = ["#a6cee3", "#1f78b4"]
PALETTE_RACE = ["#d73027", "#4575b4"]
PALETTE_GENDER = ["#4393c3", "#d6604d", "#92c5de"]
ERROR_COLOR = "#333333"


# ── normalisation helpers ────────────────────────────────────────────────────

def coalesce_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in columns:
        if col not in df.columns:
            continue
        cand = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        cand = cand.mask(cand.eq(""), pd.NA)
        out = out.fillna(cand)
    return out


def coalesce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col not in df.columns:
            continue
        out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def build_age_group(df: pd.DataFrame) -> pd.Series:
    age = coalesce_numeric(df, ["Age (Demographics)", "Age", "Age.1", "Age.2"])
    age = age.where((age >= 18) & (age <= 100), np.nan)

    grouped = pd.Series(pd.NA, index=df.index, dtype="string")
    grouped.loc[age.notna() & (age < 40)] = "<40"
    grouped.loc[age.notna() & (age >= 40)] = ">40"
    return grouped


def build_race_group(df: pd.DataFrame) -> pd.Series:
    raw = coalesce_text(df, [
        "Ethnicity simplified",
        "Ethnicity simplified (Demographics)",
        "Ethnicity",
        "Ethnicity (Demographics)",
    ])

    def _norm(v: object) -> str:
        if pd.isna(v):
            return pd.NA
        t = str(v).strip().lower()
        if "prefer not" in t or "rather not" in t:
            return pd.NA
        if "white" in t:
            return "White"
        return "Non-White"

    return raw.map(_norm).astype("string")


def build_gender_group(df: pd.DataFrame) -> pd.Series:
    raw = coalesce_text(df, [
        "Gender",
        "Gender (Demographics)",
        "Sex",
        "Sex (Demographics)",
    ])

    def _norm(v: object) -> str:
        if pd.isna(v):
            return pd.NA
        t = str(v).strip().lower()
        if t in {"man", "male"} or t.startswith("man"):
            return "Man"
        if t in {"woman", "female"} or t.startswith("woman"):
            return "Woman"
        if "prefer not" in t or "rather not" in t:
            return "NB/Prefer not"
        if "non-binary" in t or "nonbinary" in t:
            return "NB/Prefer not"
        return "NB/Prefer not"

    return raw.map(_norm).astype("string")


# ── core computation ─────────────────────────────────────────────────────────

def group_means(frame: pd.DataFrame, group_col: str, preferred_order: list[str] | None = None) -> tuple[list[dict], list[str]]:
    rows = []
    groups = [
        g
        for g in frame[group_col].dropna().unique()
        if (frame[group_col] == g).sum() >= MIN_N
    ]
    if preferred_order:
        pref = [g for g in preferred_order if g in groups]
        extras = sorted([g for g in groups if g not in pref])
        groups = pref + extras
    else:
        groups = sorted(groups)

    for g in groups:
        subset = frame[frame[group_col] == g]
        em = subset["emotional_change"].dropna().astype(float)
        an = subset["analytical_change"].dropna().astype(float)
        rows.append(
            {
                "group": str(g),
                "n": float(len(subset)),
                "em_mean": float(em.mean()) if len(em) else np.nan,
                "an_mean": float(an.mean()) if len(an) else np.nan,
            }
        )
    return rows, [str(g) for g in groups]


def metric_between_group_test(frame: pd.DataFrame, group_col: str, metric_col: str, groups: list[str]) -> dict[str, object]:
    samples = []
    used_groups: list[str] = []
    for g in groups:
        vals = frame.loc[frame[group_col] == g, metric_col].dropna().astype(float)
        if len(vals) >= MIN_N:
            samples.append(vals)
            used_groups.append(g)

    if len(samples) < 2:
        return {
            "kind": "insufficient",
            "p": np.nan,
            "stat": np.nan,
            "groups": used_groups,
            "posthoc": [],
        }

    if len(samples) == 2:
        t_stat, p_val = ttest_ind(samples[0], samples[1], equal_var=False, nan_policy="omit")
        return {
            "kind": "welch_t",
            "stat": float(t_stat),
            "p": float(p_val),
            "groups": used_groups,
            "posthoc": [],
        }

    f_stat, p_val = f_oneway(*samples)
    posthoc_rows = []
    n_pairs = len(used_groups) * (len(used_groups) - 1) // 2
    for g_a, g_b in combinations(used_groups, 2):
        vals_a = frame.loc[frame[group_col] == g_a, metric_col].dropna().astype(float)
        vals_b = frame.loc[frame[group_col] == g_b, metric_col].dropna().astype(float)
        if len(vals_a) < MIN_N or len(vals_b) < MIN_N:
            continue
        t_stat, p_raw = ttest_ind(vals_a, vals_b, equal_var=False, nan_policy="omit")
        p_adj = min(float(p_raw) * n_pairs, 1.0)  # Bonferroni correction
        posthoc_rows.append(
            {
                "group_a": str(g_a),
                "group_b": str(g_b),
                "t": float(t_stat),
                "p_raw": float(p_raw),
                "p_adj": p_adj,
                "significant": p_adj < ALPHA,
            }
        )

    return {
        "kind": "anova",
        "stat": float(f_stat),
        "p": float(p_val),
        "groups": used_groups,
        "posthoc": posthoc_rows,
    }


def panel_results(
    frame: pd.DataFrame,
    group_col: str,
    preferred_order: list[str] | None = None,
) -> tuple[list[dict], dict[str, object], dict[str, object]]:
    means, groups = group_means(frame, group_col, preferred_order=preferred_order)
    em_test = metric_between_group_test(frame, group_col, "emotional_change", groups)
    an_test = metric_between_group_test(frame, group_col, "analytical_change", groups)
    return means, em_test, an_test


def combo_group_results(frame: pd.DataFrame, col_a: str, col_b: str) -> list[dict]:
    frame = frame.copy()
    frame["_combo"] = (
        frame[col_a].astype(str).str.strip() + " / "
        + frame[col_b].astype(str).str.strip()
    )
    # drop cells where either factor was NA
    frame = frame[~frame[col_a].isna() & ~frame[col_b].isna()]
    means, em_test, an_test = panel_results(frame, "_combo")
    return means, em_test, an_test


# ── plotting helpers ──────────────────────────────────────────────────────────

def p_to_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p(p: float) -> str:
    if np.isnan(p):
        return "NA"
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".")


def format_between_test(test: dict[str, object], metric_label: str) -> str:
    if test["kind"] == "insufficient":
        return f"{metric_label}: insufficient data"
    if test["kind"] == "welch_t":
        return f"{metric_label}: Welch t={test['stat']:.2f}, p {format_p(float(test['p']))}"
    return f"{metric_label}: ANOVA F={test['stat']:.2f}, p {format_p(float(test['p']))}"


def format_posthoc_line(row: dict[str, object]) -> str:
    sig = "significant" if bool(row["significant"]) else "not significant"
    return (
        f"{row['group_a']} vs {row['group_b']}: "
        f"t={float(row['t']):.2f}, p_raw {format_p(float(row['p_raw']))}, "
        f"p_adj {format_p(float(row['p_adj']))} ({sig})"
    )


def significant_groups_from_posthoc(test: dict[str, object]) -> set[str]:
    groups: set[str] = set()
    for row in test.get("posthoc", []):
        if bool(row.get("significant", False)):
            groups.add(str(row["group_a"]))
            groups.add(str(row["group_b"]))
    return groups


def describe_change_value(metric_label: str, value: float) -> str:
    direction = "decreased" if value < 0 else "increased"
    return (
        f"{metric_label} trust {direction} by {abs(value):.3f} on the normalised "
        "post-pre scale."
    )


def draw_panel(
    ax: plt.Axes,
    results: list[dict],
    em_test: dict[str, object],
    an_test: dict[str, object],
    title: str,
    palette: list[str],
) -> None:
    """Bar chart of group means for emotional and analytical trust change."""
    if not results:
        ax.set_visible(False)
        return

    groups = [r["group"] for r in results]
    n_groups = len(groups)
    x = np.arange(n_groups, dtype=float)
    width = 0.32

    colors = (palette * ((n_groups // len(palette)) + 1))[:n_groups]

    em_means = np.array([r["em_mean"] for r in results], dtype=float)
    an_means = np.array([r["an_mean"] for r in results], dtype=float)

    ax.bar(
        x - width / 2,
        em_means,
        width=width,
        color=colors,
        alpha=0.85,
        label="Emotional change",
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        an_means,
        width=width,
        color=colors,
        alpha=0.45,
        label="Analytical change",
        zorder=2,
        hatch="//",
    )

    ax.axhline(0.0, color="#555555", linewidth=0.9, zorder=1)
    ax.set_xticks(x)

    # shorten long combo labels for legibility
    short_labels = []
    for g in groups:
        parts = g.split(" / ")
        if len(parts) == 2:
            short_labels.append(f"{parts[0][:6]}\n/{parts[1][:6]}")
        else:
            short_labels.append(g)
    ax.set_xticklabels(short_labels, fontsize=7, rotation=25, ha="right")
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylabel("Trust change (Post - Pre, normalised)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    em_text = format_between_test(em_test, "Emotional")
    an_text = format_between_test(an_test, "Analytical")
    ax.text(
        0.01,
        0.99,
        f"{em_text}\n{an_text}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#bbbbbb", "pad": 2.5},
    )


# ── report ────────────────────────────────────────────────────────────────────

def write_report(panels: list[tuple[str, list[dict], dict[str, object], dict[str, object]]]) -> None:
    lines: list[str] = []
    lines.append("\\subsection{Bias Analysis: Across-Group Trust-Change Comparisons}")
    lines.append(
        "For each panel, group differences were tested separately for emotional change and "
        "analytical change. Two-group panels use Welch's t-test; panels with three or more "
        "groups use one-way ANOVA with Bonferroni-corrected post-hoc pairwise Welch tests."
    )
    lines.append(f"Minimum group size for inclusion: $n \\geq {MIN_N}$.")
    lines.append("")
    lines.append("\\paragraph{How to interpret the significance tests}")
    lines.append(
        "Welch's t-test (two-group panels): tests whether the mean trust change differs between "
        "the two groups (e.g., $<40$ vs $>40$). A significant $p$ indicates those two group means "
        "are statistically different for that metric."
    )
    lines.append(
        "One-way ANOVA (three-or-more-group panels): tests whether at least one group mean differs "
        "from the others. A significant ANOVA $p$ means there is an overall between-group effect, "
        "but it does not by itself identify which specific groups differ."
    )
    lines.append(
        "Post-hoc pairwise tests: follow-up Welch tests between every pair of groups in ANOVA panels. "
        "$p_{raw}$ is the unadjusted pairwise p-value. $p_{adj}$ is the Bonferroni-corrected p-value "
        "that controls family-wise error across all pairwise comparisons in that panel/metric."
    )
    lines.append(
        "Significance threshold: tests are marked significant when $p < .05$ (or $p_{adj} < .05$ for "
        "post-hoc tests). 'Not significant' means the data did not provide sufficient evidence of a "
        "difference at this threshold; it does not prove the groups are identical."
    )
    lines.append("")

    for label, results, em_test, an_test in panels:
        lines.append(f"\\subsubsection{{{label}}}")
        lines.append(format_between_test(em_test, "Emotional"))
        if em_test.get("kind") == "anova":
            lines.append("Emotional post-hoc pairwise (Bonferroni-adjusted):")
            for row in em_test.get("posthoc", []):
                lines.append(f"- {format_posthoc_line(row)}")
        lines.append(format_between_test(an_test, "Analytical"))
        if an_test.get("kind") == "anova":
            lines.append("Analytical post-hoc pairwise (Bonferroni-adjusted):")
            for row in an_test.get("posthoc", []):
                lines.append(f"- {format_posthoc_line(row)}")

        # Add plain-language interpretations only for significant groups in 1x panels.
        one_x_panel = "×" not in label
        sig_em_groups = significant_groups_from_posthoc(em_test) if one_x_panel else set()
        sig_an_groups = significant_groups_from_posthoc(an_test) if one_x_panel else set()

        for r in results:
            lines.append(
                f"{r['group']}: $n={int(r['n'])}$, "
                f"$M_{{em,chg}}={r['em_mean']:.3f}$, $M_{{an,chg}}={r['an_mean']:.3f}$."
            )
            group_name = str(r["group"])
            if one_x_panel and (group_name in sig_em_groups or group_name in sig_an_groups):
                em_text = describe_change_value("Emotional", float(r["em_mean"]))
                an_text = describe_change_value("Analytical", float(r["an_mean"]))
                lines.append(
                    f"Interpretation ({group_name}): $M_{{em,chg}}={r['em_mean']:.3f}$ means {em_text} "
                    f"$M_{{an,chg}}={r['an_mean']:.3f}$ means {an_text}"
                )
        lines.append("")

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # build trust-change metrics
    df["analytical_change"] = (
        pd.to_numeric(df["Total Analytical Trust Post"], errors="coerce")
        - pd.to_numeric(df["Total Analytical Trust"], errors="coerce")
    ) / 10.0
    df["emotional_change"] = (
        pd.to_numeric(df["Total Emotional Trust Post"], errors="coerce")
        - pd.to_numeric(df["Total Emotional Trust"], errors="coerce")
    ) / 9.0

    # build demographic columns
    df["age_grp"] = build_age_group(df)
    df["race_grp"] = build_race_group(df)
    df["gender_grp"] = build_gender_group(df)

    # compute results for all 6 panels
    res_age, age_em_test, age_an_test = panel_results(
        df,
        "age_grp",
        preferred_order=["<40", ">40"],
    )
    res_race, race_em_test, race_an_test = panel_results(
        df,
        "race_grp",
        preferred_order=["White", "Non-White"],
    )
    res_gender, gender_em_test, gender_an_test = panel_results(
        df,
        "gender_grp",
        preferred_order=["Man", "Woman", "NB/Prefer not"],
    )
    res_age_race = combo_group_results(df, "age_grp", "race_grp")
    res_age_gender = combo_group_results(df, "age_grp", "gender_grp")
    res_race_gender = combo_group_results(df, "race_grp", "gender_grp")

    panels = [
        ("Age", res_age, age_em_test, age_an_test, PALETTE_AGE),
        ("Race / Ethnicity", res_race, race_em_test, race_an_test, PALETTE_RACE),
        ("Gender", res_gender, gender_em_test, gender_an_test, PALETTE_GENDER),
        ("Age × Race", res_age_race[0], res_age_race[1], res_age_race[2], PALETTE_AGE + PALETTE_RACE),
        ("Age × Gender", res_age_gender[0], res_age_gender[1], res_age_gender[2], PALETTE_AGE + PALETTE_GENDER),
        ("Race × Gender", res_race_gender[0], res_race_gender[1], res_race_gender[2], PALETTE_RACE + PALETTE_GENDER),
    ]

    # ── figure ────────────────────────────────────────────────────────────────
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)

    for idx, (title, results, em_test, an_test, palette) in enumerate(panels):
        row, col = divmod(idx, 3)
        draw_panel(axes[row, col], results, em_test, an_test, title, palette)

    # shared legend
    legend_handles = [
        mpatches.Patch(color="#888888", alpha=0.85, label="Emotional change"),
        mpatches.Patch(color="#888888", alpha=0.45, hatch="//", label="Analytical change"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Across-Group Emotional and Analytical Trust-Change Comparisons\n"
        "Panel notes show between-group tests per metric",
        fontsize=11, y=1.03,
    )

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(LEGACY_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ── report ────────────────────────────────────────────────────────────────
    report_panels = [(t, r, em_test, an_test) for t, r, em_test, an_test, _ in panels]
    write_report(report_panels)

    print(f"Saved figure: {PNG_PATH}")
    print(f"Saved figure copy: {LEGACY_PNG_PATH}")
    print(f"Saved report: {TXT_PATH}")


if __name__ == "__main__":
    main()
