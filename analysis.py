import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

combined = pd.read_pickle("./data/grouped.pkl")
#print(combined.columns)
"""
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
       'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
       'Consent 4', 'Prolific or Email', 'Education', 'AI Knowledge', 'Age',
       'AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm',
       'AI Confident', 'AI Security', 'AI Trustworthy', 'AI Reliable',
       'AI Trust', 'AI systems are 1', 'AI systems are 2', 'AI systems are 3',
       'AI systems are 4', 'AI systems are 5', 'AI systems are 6',
       'AI systems are 7', 'AI systems are 8', 'AI systems are 9',
       'AI Harm Post', 'AI Security Post', 'AI Suspicious Post',
       'AI Weary Post', 'AI Deceptive Post', 'AI Confident Post',
       'AI Honest Post', 'AI Trustworthy Post', 'AI Reliable Post',
       'AI Trust Post', 'AI systems are 1 Post', 'AI systems are 2 Post',
       'AI systems are 3 Post', 'AI systems are 4 Post',
       'AI systems are 5 Post', 'AI systems are 6 Post',
       'AI systems are 7 Post', 'AI systems are 8 Post',
       'AI systems are 9 Post', 'AI Feel Post', 'AI Understand Post',
       'AI Job Post', 'Condition', 'Interaction Name List',
       'Interaction Original List', 'Interaction Updated List',
       'Interaction Type List', 'Response Name List', 'Response Text List'],
      dtype='str')
"""

pre_to_post = {
      "AI Harm": "AI Harm Post",
      "AI Security": "AI Security Post",
      "AI Suspicious": "AI Suspicious Post",
      "AI Weary": "AI Weary Post",
      "AI Deceptive": "AI Deceptive Post",
      "AI Confident": "AI Confident Post",
      "AI Honest": "AI Honest Post",
      "AI Trustworthy": "AI Trustworthy Post",
      "AI Reliable": "AI Reliable Post",
      "AI Trust": "AI Trust Post",
      "AI systems are 1": "AI systems are 1 Post",
      "AI systems are 2": "AI systems are 2 Post",
      "AI systems are 3": "AI systems are 3 Post",
      "AI systems are 4": "AI systems are 4 Post",
      "AI systems are 5": "AI systems are 5 Post",
      "AI systems are 6": "AI systems are 6 Post",
      "AI systems are 7": "AI systems are 7 Post",
      "AI systems are 8": "AI systems are 8 Post",
      "AI systems are 9": "AI systems are 9 Post",
}

AI_systems_are_map = {
	"AI systems are 1": {"Apathetic":0, "Empathetic":1},
	"AI systems are 2": {"Insensitive":0, "Sensitive":1},
	"AI systems are 3": {"Impersonal":0, "Personal":1},
	"AI systems are 4": {"Ignoring":0, "Caring":1},
	"AI systems are 5": {"Self Serving":0, "Altruistic":1},
	"AI systems are 6": {"Rude":0, "Cordial":1},
	"AI systems are 7": {"Indifferent":0, "Responsive":1},
	"AI systems are 8": {"Judgemental":0, "Open-Minded":1},
	"AI systems are 9": {"Impatient":0, "Patient":1},
}


def clean_text(value: str) -> str:
      text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
      text = text.replace("judgmental", "judgemental")
      return " ".join(text.split())


AI_systems_value_map = {}
for pair_map in AI_systems_are_map.values():
      for key, val in pair_map.items():
            AI_systems_value_map[clean_text(key)] = val

likert_map = {
      "Strongly Disagree": -2,
      "Disagree": -1,
      "Agree": 1,
      "Strongly Agree": 2,
}

likert_value_map = {clean_text(key): val for key, val in likert_map.items()}


def normalize_likert(series: pd.Series) -> pd.Series:
      mapped = series.astype(str).map(clean_text).map(likert_value_map)
      return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def normalize_ai_systems_are(series: pd.Series, col_name: str) -> pd.Series:
      mapped = series.astype(str).map(clean_text).map(AI_systems_value_map)
      return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def get_ai_base_col(col_name: str) -> str:
      return col_name.replace(" Post", "")

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
      if len(a) < 2 or len(b) < 2:
            return float("nan")
      var_a = np.var(a, ddof=1)
      var_b = np.var(b, ddof=1)
      pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
      return (np.mean(a) - np.mean(b)) / np.sqrt(pooled) if pooled > 0 else float("nan")


results = []
conditions = combined["Condition"].dropna().unique().tolist()

for pre_col, post_col in pre_to_post.items():
      if pre_col not in combined.columns or post_col not in combined.columns:
            continue
      
      pre_values = pd.to_numeric(combined[pre_col], errors="coerce")
      post_values = pd.to_numeric(combined[post_col], errors="coerce")

      if get_ai_base_col(pre_col) in AI_systems_are_map:
            pre_values = pd.to_numeric(normalize_ai_systems_are(combined[pre_col], pre_col), errors="coerce")
            post_values = pd.to_numeric(normalize_ai_systems_are(combined[post_col], post_col), errors="coerce")

            pre_missing = pre_values.isna().sum()
            post_missing = post_values.isna().sum()
            if pre_missing or post_missing:
                  print(
                        f"AI systems mapping gaps for '{pre_col}': "
                        f"pre NaN={pre_missing}, post NaN={post_missing}"
                  )

      if pre_values.isna().all():
            pre_values = pd.to_numeric(normalize_likert(combined[pre_col]), errors="coerce")
      if post_values.isna().all():
            post_values = pd.to_numeric(normalize_likert(combined[post_col]), errors="coerce")
      delta = post_values - pre_values

      valid = combined[["Condition"]].copy()
      valid["delta"] = delta
      valid = valid.dropna(subset=["Condition", "delta"])

      if len(valid) == 0:
            continue

      if len(conditions) == 2:
            cond_a = conditions[0]
            cond_b = conditions[1]
            group_a = valid.loc[valid["Condition"] == cond_a, "delta"].to_numpy()
            group_b = valid.loc[valid["Condition"] == cond_b, "delta"].to_numpy()
            if len(group_a) < 2 or len(group_b) < 2:
                  p_value = float("nan")
                  stat = float("nan")
                  mean_a = float("nan")
                  mean_b = float("nan")
                  direction = ""
            else:
                  stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True, nan_policy="omit")
                  mean_a = np.mean(group_a)
                  mean_b = np.mean(group_b)
                  if np.isfinite(mean_a) and np.isfinite(mean_b):
                        if mean_a > mean_b:
                              direction = f"{cond_a} > {cond_b}"
                        elif mean_b > mean_a:
                              direction = f"{cond_b} > {cond_a}"
                        else:
                              direction = "tie"
                  else:
                        direction = ""
            results.append({
                  "pre_column": pre_col,
                  "post_column": post_col,
                  "test": "student_t",
                  "statistic": stat,
                  "p_value": p_value,
                  "n_total": len(valid),
                  "condition_a": cond_a,
                  "condition_b": cond_b,
                  "mean_delta_group_a": mean_a,
                  "mean_delta_group_b": mean_b,
                  "direction": direction,
                  "effect_size_d": cohen_d(group_a, group_b),
            })
      else:
            groups = [valid.loc[valid["Condition"] == cond, "delta"].to_numpy() for cond in conditions]
            if any(len(group) < 2 for group in groups):
                  stat = float("nan")
                  p_value = float("nan")
            else:
                  stat, p_value = stats.f_oneway(*groups)
            results.append({
                  "pre_column": pre_col,
                  "post_column": post_col,
                  "test": "anova",
                  "statistic": stat,
                  "p_value": p_value,
                  "n_total": len(valid),
            })

results_df = pd.DataFrame(results)
if results_df.empty:
      print("No valid pre/post pairs found for analysis.")
else:
      results_df = results_df.sort_values("p_value")
      print(results_df)
      results_df.to_csv("./analysis/condition_delta_stats.csv", index=False)

      ANALYSIS_DIR = Path("./analysis")
      ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
      plot_df = results_df.dropna(subset=["statistic"]).copy()
      plot_df = plot_df[~plot_df["pre_column"].str.startswith("AI systems are ")]
      plt.figure(figsize=(14, 6))
      bars = plt.bar(plot_df["pre_column"], plot_df["statistic"])
      plt.title("Condition Effect on Pre/Post Change", fontsize=16)
      plt.xlabel("Column", fontsize=14)
      plt.ylabel("Test statistic", fontsize=14)
      plt.xticks(rotation=45, ha="right", fontsize=12)
      plt.yticks(fontsize=12)

      ax = plt.gca()
      
      y_min, y_max = ax.get_ylim()
      y_range = y_max - y_min if y_max != y_min else 1.0
      label_offset = 0.05 * y_range

      for bar, row in zip(bars, plot_df.itertuples(index=False)):
            height = bar.get_height()
            label = f"p={row.p_value:.3g}" if np.isfinite(row.p_value) else "p=nan"
            if hasattr(row, "direction") and row.direction:
                  label = f"{label}\n{row.direction}"

            if height >= 0:
                  y_pos = height + label_offset
                  v_align = "bottom"
            else:
                  y_pos = height - label_offset
                  v_align = "top"

            plt.text(
                  bar.get_x() + bar.get_width() / 2,
                  y_pos,
                  label,
                  ha="center",
                  va=v_align,
                  fontsize=12,
            )
      
      ax.set_ylim(-3, 3)
      plt.tight_layout()
      plt.savefig(ANALYSIS_DIR / "condition_delta_stats.png", dpi=300, bbox_inches="tight")
      plt.close()

      ai_systems_cols = [col for col in pre_to_post.keys() if get_ai_base_col(col) in AI_systems_are_map]
      ai_plot_df = results_df[results_df["pre_column"].isin(ai_systems_cols)].copy()
      if not ai_plot_df.empty:
            plt.figure(figsize=(14, 6))
            x_labels = [col.replace("AI systems are ", "") for col in ai_plot_df["pre_column"]]
            bars = plt.bar(x_labels, ai_plot_df["statistic"])
            plt.title("Condition Effect on AI Systems Are (Pre/Post)", fontsize=16)
            plt.xlabel("Item", fontsize=14)
            plt.ylabel("Test statistic", fontsize=14)
            plt.xticks(rotation=0, ha="center", fontsize=12)
            plt.yticks(fontsize=12)

            ax = plt.gca()
            max_abs = float(np.nanmax(np.abs(ai_plot_df["statistic"].to_numpy())))
            if not np.isfinite(max_abs):
                  max_abs = 1.0
            pad = max_abs * 0.6
            ax.set_ylim(-(max_abs + pad), max_abs + pad)
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min if y_max != y_min else 1.0
            label_offset = 0.08 * y_range

            for bar, row in zip(bars, ai_plot_df.itertuples(index=False)):
                  height = bar.get_height()
                  label = f"p={row.p_value:.3g}" if np.isfinite(row.p_value) else "p=nan"
                  if hasattr(row, "direction") and row.direction:
                        label = f"{label}\n{row.direction}"

                  if height >= 0:
                        y_pos = height + label_offset
                        v_align = "bottom"
                  else:
                        y_pos = height - label_offset
                        v_align = "top"

                  plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        label,
                        ha="center",
                        va=v_align,
                        fontsize=12,
                  )

            for bar, col_name in zip(bars, ai_plot_df["pre_column"]):
                  mapping = AI_systems_are_map.get(col_name, {})
                  if len(mapping) == 2:
                        bad_word, good_word = list(mapping.keys())
                        x_center = bar.get_x() + bar.get_width() / 2
                        ax.text(
                              x_center,
                              -0.14,
                              bad_word,
                              ha="center",
                              va="top",
                              fontsize=10,
                              transform=ax.get_xaxis_transform(),
                        )
                        ax.text(
                              x_center,
                              1.06,
                              good_word,
                              ha="center",
                              va="bottom",
                              fontsize=10,
                              transform=ax.get_xaxis_transform(),
                        )

            plt.tight_layout()
            plt.savefig(ANALYSIS_DIR / "ai_systems_are_delta_stats.png", dpi=300, bbox_inches="tight")
            plt.close()

      logical_cols = [
            "AI Weary",
            "AI Deceptive",
            "AI Honest",
            "AI Suspicious",
            "AI Harm",
            "AI Confident",
            "AI Security",
            "AI Trustworthy",
            "AI Reliable",
            "AI Trust",
      ]
      emotional_cols = [
            "AI systems are 1",
            "AI systems are 2",
            "AI systems are 3",
            "AI systems are 4",
            "AI systems are 5",
            "AI systems are 6",
            "AI systems are 7",
            "AI systems are 8",
            "AI systems are 9",
      ]

      logical_pre = pd.concat(
            [
                  pd.to_numeric(
                        normalize_ai_systems_are(combined[col], col),
                        errors="coerce",
                  )
                  if get_ai_base_col(col) in AI_systems_are_map
                  else pd.to_numeric(normalize_likert(combined[col]), errors="coerce")
                  for col in logical_cols
            ],
            axis=1,
      )
      logical_post = pd.concat(
            [
                  pd.to_numeric(
                        normalize_ai_systems_are(combined[f"{col} Post"], f"{col} Post"),
                        errors="coerce",
                  )
                  if get_ai_base_col(col) in AI_systems_are_map
                  else pd.to_numeric(normalize_likert(combined[f"{col} Post"]), errors="coerce")
                  for col in logical_cols
            ],
            axis=1,
      )
      logical_delta = (logical_post - logical_pre).mean(axis=1, skipna=True)

      emotional_pre = pd.concat(
            [
                  pd.to_numeric(
                        normalize_ai_systems_are(combined[col], col),
                        errors="coerce",
                  )
                  if get_ai_base_col(col) in AI_systems_are_map
                  else pd.to_numeric(normalize_likert(combined[col]), errors="coerce")
                  for col in emotional_cols
            ],
            axis=1,
      )
      emotional_post = pd.concat(
            [
                  pd.to_numeric(
                        normalize_ai_systems_are(combined[f"{col} Post"], f"{col} Post"),
                        errors="coerce",
                  )
                  if get_ai_base_col(col) in AI_systems_are_map
                  else pd.to_numeric(normalize_likert(combined[f"{col} Post"]), errors="coerce")
                  for col in emotional_cols
            ],
            axis=1,
      )
      emotional_delta = (emotional_post - emotional_pre).mean(axis=1, skipna=True)

      def condition_ttest(delta_series: pd.Series) -> dict:
            valid = combined[["Condition"]].copy()
            valid["delta"] = delta_series
            valid = valid.dropna(subset=["Condition", "delta"])
            if len(valid) == 0 or len(conditions) != 2:
                  return {"statistic": np.nan, "p_value": np.nan, "direction": ""}

            cond_a = conditions[0]
            cond_b = conditions[1]
            group_a = valid.loc[valid["Condition"] == cond_a, "delta"].to_numpy()
            group_b = valid.loc[valid["Condition"] == cond_b, "delta"].to_numpy()
            if len(group_a) < 2 or len(group_b) < 2:
                  return {"statistic": np.nan, "p_value": np.nan, "direction": ""}

            stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True, nan_policy="omit")
            mean_a = np.mean(group_a)
            mean_b = np.mean(group_b)
            if np.isfinite(mean_a) and np.isfinite(mean_b):
                  if mean_a > mean_b:
                        direction = f"{cond_a} > {cond_b}"
                  elif mean_b > mean_a:
                        direction = f"{cond_b} > {cond_a}"
                  else:
                        direction = "tie"
            else:
                  direction = ""

            return {"statistic": stat, "p_value": p_value, "direction": direction}
      
      logical_stats = condition_ttest(logical_delta)
      emotional_stats = condition_ttest(emotional_delta)

      group_plot_df = pd.DataFrame(
            [
                  {
                        "label": "Logical Distrust of AI",
                        "statistic": logical_stats["statistic"],
                        "p_value": logical_stats["p_value"],
                        "direction": logical_stats["direction"],
                  },
                  {
                        "label": "Emotional Distrust of AI",
                        "statistic": emotional_stats["statistic"],
                        "p_value": emotional_stats["p_value"],
                        "direction": emotional_stats["direction"],
                  },
            ]
      ).dropna(subset=["statistic"])
      if group_plot_df.empty:
            print("No valid logical/emotional group stats to plot.")
      else:
            plt.figure(figsize=(8, 5))
            bars = plt.bar(group_plot_df["label"], group_plot_df["statistic"])
            plt.title("Condition Effect on Logical vs Emotional Distrust", fontsize=16)
            plt.xlabel("Group", fontsize=14)
            plt.ylabel("Test statistic", fontsize=14)
            plt.xticks(rotation=0, ha="center", fontsize=12)
            plt.yticks(fontsize=12)

            ax = plt.gca()
            max_abs = float(np.nanmax(np.abs(group_plot_df["statistic"].to_numpy())))
            if not np.isfinite(max_abs):
                  max_abs = 1.0
            pad = max_abs * 0.6
            ax.set_ylim(-(max_abs + pad), max_abs + pad)
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min if y_max != y_min else 1.0
            label_offset = 0.08 * y_range

            for bar, row in zip(bars, group_plot_df.itertuples(index=False)):
                  height = bar.get_height()
                  label = f"p={row.p_value:.3g}" if np.isfinite(row.p_value) else "p=nan"
                  if row.direction:
                        label = f"{label}\n{row.direction}"

                  if height >= 0:
                        y_pos = height + label_offset
                        v_align = "bottom"
                  else:
                        y_pos = height - label_offset
                        v_align = "top"

                  plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        label,
                        ha="center",
                        va=v_align,
                        fontsize=12,
                  )

            plt.tight_layout()
            plt.savefig(ANALYSIS_DIR / "distrust_group_delta_stats.png", dpi=300, bbox_inches="tight")
            plt.close()




