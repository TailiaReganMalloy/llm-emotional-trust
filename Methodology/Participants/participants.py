from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "Dataset" / "Metrics.csv"
TEX_PATH = Path(__file__).resolve().parent / "participants.tex"
FIGURES_DIR = REPO_ROOT / "Figures"

WESTERN_COUNTRIES = {
    "United States", "United Kingdom", "Canada", "Australia", "New Zealand",
    "Ireland", "Germany", "France", "Netherlands", "Belgium", "Switzerland",
    "Austria", "Denmark", "Sweden", "Norway", "Finland", "Iceland",
    "Luxembourg", "Italy", "Spain", "Portugal",
}

HIGH_EDUCATION = {"Bachelor", "Master", "Graduate Professional Degree", "PhD"}

WEIRD_EMPLOYMENT = {
    "Full-Time",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def latex_table(rows: list[tuple[str, str | int]], col1: str, col2: str,
                caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lr}",
        r"\hline",
        f"\\textbf{{{col1}}} & \\textbf{{{col2}}} \\\\",
        r"\hline",
    ]
    for row_label, value in rows:
        lines.append(f"{row_label} & {value} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def counts_rows(series: pd.Series, exclude: list[str] | None = None) -> list[tuple[str, int]]:
    vc = series.value_counts()
    rows = []
    for k, v in vc.items():
        if exclude and k in exclude:
            continue
        rows.append((str(k), int(v)))
    return rows


# ── load & derive ─────────────────────────────────────────────────────────────

def score_weird(row: pd.Series) -> int:
    """Return the number of WEIRD criteria (0–6) satisfied by a participant."""
    score = 0
    # 1. Country of residence
    if row["Country of residence"] in WESTERN_COUNTRIES:
        score += 1
    # 2. English as first language
    if str(row["Language"]).lower().startswith("english"):
        score += 1
    # 3. High education (Bachelor or above)
    if row["Education"] in HIGH_EDUCATION:
        score += 1
    # 4. Full-time employed or retired/not in paid work
    if row["Employment status"] in WEIRD_EMPLOYMENT:
        score += 1
    # 5. White ethnicity
    if str(row["Ethnicity simplified"]).strip().lower() == "white":
        score += 1
    # 6. Western nationality OR western country of birth
    if (row["Nationality"] in WESTERN_COUNTRIES
            or row["Country of birth"] in WESTERN_COUNTRIES):
        score += 1
    return score


def build_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["weird_score"] = df.apply(score_weird, axis=1)
    df["is_weird"] = df["weird_score"] >= 4
    return df


# ── table builders ────────────────────────────────────────────────────────────

def age_table(df: pd.DataFrame) -> str:
    age = df["Age"].dropna()
    rows = [
        ("Mean", f"{age.mean():.1f}"),
        ("SD", f"{age.std():.1f}"),
        ("Median", f"{age.median():.0f}"),
        ("Min", f"{int(age.min())}"),
        ("Max", f"{int(age.max())}"),
    ]
    return latex_table(rows, "Statistic", "Value",
                       caption="Participant age summary ($N=504$).",
                       label="tab:demographics_age")


def education_table(df: pd.DataFrame) -> str:
    ORDER = [
        "High School",
        "Bachelor",
        "Master",
        "Graduate Professional Degree",
        "PhD",
    ]
    vc = df["Education"].value_counts()
    rows = [(lvl, int(vc.get(lvl, 0))) for lvl in ORDER]
    return latex_table(rows, "Education Level", "$n$",
                       caption="Participant highest education level ($N=504$).",
                       label="tab:demographics_education")


def ai_knowledge_table(df: pd.DataFrame) -> str:
    ORDER = ["No knowledge", "Beginner knowledge", "Conceptual understanding",
             "Advanced", "Expert"]
    vc = df["AI Knowledge"].value_counts()
    rows = [(lvl, int(vc.get(lvl, 0))) for lvl in ORDER]
    return latex_table(rows, "AI Knowledge Level", "$n$",
                       caption="Participant self-reported AI knowledge ($N=504$).",
                       label="tab:demographics_ai_knowledge")


def employment_table(df: pd.DataFrame) -> str:
    EXCLUDE = ["DATA_EXPIRED"]
    rows = counts_rows(df["Employment status"], exclude=EXCLUDE)
    valid_n = int((~df["Employment status"].isin(EXCLUDE)).sum())
    return latex_table(rows, "Employment Status", "$n$",
                       caption=f"Participant employment status ($n={valid_n}$; {504 - valid_n} responses expired).",
                       label="tab:demographics_employment")


def ethnicity_table(df: pd.DataFrame) -> str:
    rows = counts_rows(df["Ethnicity simplified"])
    return latex_table(rows, "Ethnicity", "$n$",
                       caption="Participant ethnicity ($N=504$).",
                       label="tab:demographics_ethnicity")


def weird_table(df: pd.DataFrame) -> str:
    weird_n = int(df["is_weird"].sum())
    normal_n = int((~df["is_weird"]).sum())
    rows = [("WEIRD ($\\geq$4/6 criteria)", weird_n),
            ("Non-WEIRD ($<$4/6 criteria)", normal_n),
            ("Total", 504)]
    return latex_table(rows, "Population Group", "$n$",
                       caption=r"WEIRD vs.\ non-WEIRD participant breakdown using the 4/6 scoring rule ($N=504$).",
                       label="tab:demographics_weird")


def plot_weird_scores(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    scores = df["weird_score"]
    counts = [int((scores == i).sum()) for i in range(7)]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if i >= 4 else "#2980b9" for i in range(7)]
    bars = ax.bar(range(7), counts, color=colors, edgecolor="white", linewidth=0.8)

    ax.axvline(x=3.5, color="black", linestyle="--", linewidth=1.5, label="Threshold (≥4 = WEIRD)")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(count), ha="center", va="bottom", fontsize=11)

    weird_n = int(df["is_weird"].sum())
    normal_n = int((~df["is_weird"]).sum())

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", label=f"Non-WEIRD (<4 criteria, n={normal_n})"),
        Patch(facecolor="#c0392b", label=f"WEIRD (≥4 criteria, n={weird_n})"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Threshold"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xlabel("Number of WEIRD criteria met (out of 6)", fontsize=12)
    ax.set_ylabel("Number of participants", fontsize=12)
    ax.set_title("WEIRD score distribution (4/6 threshold)", fontsize=13)
    ax.set_xticks(range(7))
    ax.set_ylim(0, max(counts) + 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "weird_score_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

PREAMBLE = (
    r"We recruited 504 participants through the online Prolific platform. "
    r"For each condition (Interactive and Static), the Prolific participant recruiting tools were used "
    r"to control for the race and nationality of pooled participants to conform to our desired distribution "
    r"of participants. We controlled for a specific distribution of age, race, gender, and nationality to "
    r"ensure a 25\% WEIRD and 75 \% non-WEIRD participants within an even gender split of 1/3 men, 1/3 women, "
    r"and 1/3 non-binary, gender-diverse, or prefer-not-to-say participants. These sampling preferences are "
    r"used to pool participants as a guide for the Prolific system, the final sampling of participants with "
    r"approved experiment completion is shown in Table \ref{tab:demographics_gender}."
)


def main() -> None:
    df = build_df()

    weird_n = int(df["is_weird"].sum())
    normal_n = int((~df["is_weird"]).sum())
    weird_pct = weird_n / 504 * 100
    normal_pct = normal_n / 504 * 100

    weird_definition = (
        r"\paragraph{WEIRD classification.} "
        r"Participants were classified as \emph{WEIRD} "
        r"(Western, Educated, Industrialised, Rich, and Democratic~\cite{henrich2010weirdest}) "
        r"using a multi-criterion scoring rule. "
        r"Each participant was assessed against six binary criteria: "
        r"(1)~\textit{Country of residence} --- residing in one of 21 Western nations "
        r"(United States, United Kingdom, Canada, Australia, New Zealand, Ireland, Germany, France, "
        r"Netherlands, Belgium, Switzerland, Austria, Denmark, Sweden, Norway, Finland, Iceland, "
        r"Luxembourg, Italy, Spain, or Portugal); "
        r"(2)~\textit{Language} --- self-reported first language beginning with \emph{English} (case-insensitive); "
        r"(3)~\textit{Education} --- highest qualification of Bachelor's degree or above "
        r"(Bachelor, Master, Graduate Professional Degree, or PhD); "
        r"(4)~\textit{Employment} --- employed full-time or not in paid work (including retired); "
        r"(5)~\textit{Ethnicity} --- self-identified as White; "
        r"(6)~\textit{Origin} --- nationality \emph{or} country of birth in one of the same 21 Western nations. "
        rf"Participants satisfying at least 4 of these 6 criteria were labelled WEIRD; "
        rf"all others were labelled non-WEIRD. "
        rf"Under this rule, {weird_n} participants ({weird_pct:.1f}\%) were classified as WEIRD "
        rf"and {normal_n} ({normal_pct:.1f}\%) as non-WEIRD (see Table~\ref{{tab:demographics_weird}} "
        rf"and Figure~\ref{{fig:weird_score_distribution}})."
    )

    tables = "\n\n".join([
        age_table(df),
        education_table(df),
        ai_knowledge_table(df),
        employment_table(df),
        ethnicity_table(df),
        weird_table(df),
    ])

    output = "\n\n".join([PREAMBLE, weird_definition, tables]) + "\n"
    TEX_PATH.write_text(output)
    print(f"Updated: {TEX_PATH}")

    plot_weird_scores(df)


if __name__ == "__main__":
    main()
