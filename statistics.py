from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


combined = pd.read_csv("./data/combined.csv")

TARGET_COLUMNS = [
      "AI Weary Post",
      "AI Confident Post",
      "AI Suspicious Post",
      "AI Trust Post",
      "AI Harm Post",
      "AI Honest Post",
      "AI Security Post",
      "AI Deceptive Post",
      "AI Reliable Post",
      "AI Trustworthy Post",
      "AI systems are 1 Post",
      "AI systems are 2 Post",
      "AI systems are 3 Post",
      "AI systems are 4 Post",
      "AI systems are 5 Post",
      "AI systems are 6 Post",
      "AI systems are 7 Post",
      "AI systems are 8 Post",
      "AI systems are 9 Post",
      "AI Feel Post",
      "AI Understand Post",
      "AI Job Post",
]

AI_SYSTEMS_ARE_MAP = {
      "AI systems are 1": {"Apathetic": 0, "Empathetic": 1},
      "AI systems are 2": {"Insensitive": 0, "Sensitive": 1},
      "AI systems are 3": {"Impersonal": 0, "Personal": 1},
      "AI systems are 4": {"Ignoring": 0, "Caring": 1},
      "AI systems are 5": {"Self Serving": 0, "Altruistic": 1},
      "AI systems are 6": {"Rude": 0, "Cordial": 1},
      "AI systems are 7": {"Indifferent": 0, "Responsive": 1},
      "AI systems are 8": {"Judgemental": 0, "Open-Minded": 1},
      "AI systems are 9": {"Impatient": 0, "Patient": 1},
}

LIKERT_MAP = {
      "Strongly Disagree": -2,
      "Disagree": -1,
      "Agree": 1,
      "Strongly Agree": 2,
}


def clean_text(value: object) -> str:
      text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
      text = text.replace("judgmental", "judgemental")
      return " ".join(text.split())


LIKERT_VALUE_MAP = {clean_text(k): v for k, v in LIKERT_MAP.items()}

AI_SYSTEMS_VALUE_MAP: dict[str, int] = {}
for pair in AI_SYSTEMS_ARE_MAP.values():
      for key, value in pair.items():
            AI_SYSTEMS_VALUE_MAP[clean_text(key)] = value


def normalize_likert(series: pd.Series) -> pd.Series:
      mapped = series.astype(str).map(clean_text).map(LIKERT_VALUE_MAP)
      return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def normalize_ai_systems_are(series: pd.Series) -> pd.Series:
      mapped = series.astype(str).map(clean_text).map(AI_SYSTEMS_VALUE_MAP)
      return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def base_col_name(col_name: str) -> str:
      return col_name.replace(" Post", "")


def normalize_for_column(series: pd.Series, col_name: str) -> pd.Series:
      base = base_col_name(col_name)
      if base in AI_SYSTEMS_ARE_MAP:
            return pd.to_numeric(normalize_ai_systems_are(series), errors="coerce")
      return pd.to_numeric(normalize_likert(series), errors="coerce")


conditions = [c for c in combined["Condition"].dropna().unique().tolist()]
if len(conditions) != 2:
      raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)}: {conditions}")

# Fix order for consistent reporting/plotting.
if "Interactive" in conditions and "Static" in conditions:
      cond_a, cond_b = "Interactive", "Static"
else:
      cond_a, cond_b = sorted(conditions)

results: list[dict[str, float | str | int]] = []

for column in TARGET_COLUMNS:
      pre_col = base_col_name(column)
      post_col = column

      if pre_col not in combined.columns or post_col not in combined.columns:
            results.append(
                  {
                        "pre_column": pre_col,
                        "post_column": post_col,
                        "column": post_col,
                        "n": 0,
                        "mean_delta_" + cond_a: float("nan"),
                        "mean_delta_" + cond_b: float("nan"),
                        "t_statistic": float("nan"),
                        "p_value": float("nan"),
                        "r_squared": float("nan"),
                  }
            )
            continue

      pre_values = normalize_for_column(combined[pre_col], pre_col)
      post_values = normalize_for_column(combined[post_col], post_col)
      delta = post_values - pre_values

      tmp = pd.DataFrame({"Condition": combined["Condition"], "delta": delta}).dropna()
      tmp = tmp[tmp["Condition"].isin([cond_a, cond_b])]
      if tmp.empty:
            results.append(
                  {
                        "pre_column": pre_col,
                        "post_column": post_col,
                        "column": post_col,
                        "n": 0,
                        "mean_delta_" + cond_a: float("nan"),
                        "mean_delta_" + cond_b: float("nan"),
                        "t_statistic": float("nan"),
                        "p_value": float("nan"),
                        "r_squared": float("nan"),
                  }
            )
            continue

      # Binary encoding for two-condition significance test.
      tmp["x"] = (tmp["Condition"] == cond_a).astype(int)

      group_a = tmp.loc[tmp["Condition"] == cond_a, "delta"].to_numpy()
      group_b = tmp.loc[tmp["Condition"] == cond_b, "delta"].to_numpy()

      if len(group_a) < 2 or len(group_b) < 2:
            p_value = float("nan")
            r_squared = float("nan")
            statistic = float("nan")
      else:
            # t-test for significance + R^2 from point-biserial correlation.
            statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=True, nan_policy="omit")
            lr = stats.linregress(tmp["x"], tmp["delta"])
            r_squared = float(lr.rvalue ** 2)

      results.append(
            {
                  "pre_column": pre_col,
                  "post_column": post_col,
                  "column": post_col,
                  "n": int(len(tmp)),
                  "mean_delta_" + cond_a: float(np.mean(group_a)) if len(group_a) else float("nan"),
                  "mean_delta_" + cond_b: float(np.mean(group_b)) if len(group_b) else float("nan"),
                  "t_statistic": float(statistic),
                  "p_value": float(p_value),
                  "r_squared": float(r_squared),
            }
      )

results_df = pd.DataFrame(results)
if results_df.empty:
      raise ValueError("No valid target columns were available for testing.")

results_df = results_df.sort_values("p_value", na_position="last").reset_index(drop=True)

print("Condition significance results for pre/post deltas (Post - Pre):")
print(results_df.to_string(index=False))

out_dir = Path("./analysis")
out_dir.mkdir(parents=True, exist_ok=True)
results_path = out_dir / "post_delta_condition_significance.csv"
results_df.to_csv(results_path, index=False)

plot_df = results_df.dropna(subset=["r_squared"]).copy()
plt.figure(figsize=(16, 7))
bars = plt.bar(plot_df["post_column"], plot_df["r_squared"], color="#4C78A8")
plt.title("Condition Effect Size (R^2) on Post-Pre Delta")
plt.xlabel("Post Columns (delta tested as Post - Pre)")
plt.ylabel("R^2")
plt.xticks(rotation=45, ha="right")

ax = plt.gca()
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min if y_max != y_min else 1.0
label_offset = 0.02 * y_range

for bar, row in zip(bars, plot_df.itertuples(index=False)):
      label = f"p={row.p_value:.3g}" if np.isfinite(row.p_value) else "p=nan"
      ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
      )

plt.tight_layout()
plot_path = out_dir / "post_delta_condition_r2.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved significance table to: {results_path}")
print(f"Saved R^2 plot to: {plot_path}")