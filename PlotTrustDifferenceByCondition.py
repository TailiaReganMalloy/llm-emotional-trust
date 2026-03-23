import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt

Cleaned = pd.read_csv("./data/Cleaned.csv")

""" print(Cleaned.columns)
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
       'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
       'Consent 4', 'Education', 'AI Knowledge', 'Age', 'AI Deceptive',
       'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm', 'AI Confident',
       'AI Security', 'AI Trustworthy', 'AI Reliable', 'AI Trust',
       'AI systems are 1', 'AI systems are 2', 'AI systems are 3',
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
       'AI Job Post', 'Condition', 'Submission id', 'Participant id', 'Status',
       'Custom study tncs accepted at', 'Started at', 'Completed at',
       'Reviewed at', 'Archived at', 'Time taken', 'Completion code',
       'Total approvals', 'Gender', 'Ethnicity', 'Age (Demographics)', 'Sex',
       'Ethnicity simplified', 'Country of birth', 'Country of residence',
       'Nationality', 'Language', 'Student status', 'Employment status',
       'Responses', 'Submissions'],
      dtype='str')
"""


assert(False)
"""
Research Question:
Does experimental condition (Interactive vs Text) influence trust change outcomes,
specifically 'Analytical Trust Difference' and 'Emotional Trust Difference'?

RQ Prediction: Emotional Trust Difference but no Analytical Trust Difference. 

Directional Hypothesis:
Participants in the Interactive condition will show higher emotional trust change,
but not analytical trust change, compared to participants in the Text condition.
1. No meaningful difference in 'Analytical Trust Difference'
2. Higher 'Emotional Trust Difference' in the Interactive condition

Null Hypothesis 1 (H0_Analytical):
There is no mean difference in 'Analytical Trust Difference' between Interactive and Text conditions. 

Null Hypothesis 2 (H0_Emotional):
There is no significant mean difference in 'Emotional Trust Difference' between Interactive and Text conditions.

Prediction: 
1. We will fail to reject the Null Hypothesis H0_Analytical due to there being no statistically significant difference between the conditions. 

2. We will reject the Null Hypotheses H0_Emotional due to there being a statistically significant difference between the conditions. 
"""

def _to_numeric(series: pd.Series) -> pd.Series:
      return pd.to_numeric(series, errors="coerce")


interactive = Cleaned[Cleaned["Condition"] == "Interactive"].copy()
text = Cleaned[Cleaned["Condition"] == "Text"].copy()

analytical_interactive = _to_numeric(interactive["Analytical Trust Difference"]).dropna()
analytical_text = _to_numeric(text["Analytical Trust Difference"]).dropna()

emotional_interactive = _to_numeric(interactive["Emotional Trust Difference"]).dropna()
emotional_text = _to_numeric(text["Emotional Trust Difference"]).dropna()

# Two-sided Welch's t-test for H0_Analytical:
# H0: mean(Interactive) - mean(Text) = 0
t_analytical, p_analytical = stats.ttest_ind(
      analytical_interactive,
      analytical_text,
      equal_var=False,
)

# Welch's t-test for H0_Emotional
t_emotional, p_emotional = stats.ttest_ind(
      emotional_interactive,
      emotional_text,
      equal_var=False,
)

alpha = 0.05

print("\n--- Group Sizes ---")
print(f"Interactive n (analytical): {len(analytical_interactive)}")
print(f"Text n (analytical): {len(analytical_text)}")
print(f"Interactive n (emotional): {len(emotional_interactive)}")
print(f"Text n (emotional): {len(emotional_text)}")

print("\n--- Null Hypothesis 1: Analytical Trust Difference ---")
print(f"Interactive mean: {analytical_interactive.mean():.4f}")
print(f"Text mean: {analytical_text.mean():.4f}")
print(f"Two-sided Welch t-statistic: {t_analytical:.4f}")
print(f"p-value: {p_analytical:.6f}")
if p_analytical < alpha:
      print("Decision: Reject H0_Analytical at alpha=0.05")
else:
      print("Decision: Fail to reject H0_Analytical at alpha=0.05")

print("\n--- Null Hypothesis 2: Emotional Trust Difference ---")
print(f"Interactive mean: {emotional_interactive.mean():.4f}")
print(f"Text mean: {emotional_text.mean():.4f}")
print(f"Welch t-statistic: {t_emotional:.4f}")
print(f"p-value: {p_emotional:.6f}")
if p_emotional < alpha:
      print("Decision: Reject H0_Emotional at alpha=0.05")
else:
      print("Decision: Fail to reject H0_Emotional at alpha=0.05")


def _sig_label(p_value: float) -> str:
      if p_value < 0.001:
            return "***"
      if p_value < 0.01:
            return "**"
      if p_value < 0.05:
            return "*"
      return "ns"


analytical_means = [analytical_interactive.mean(), analytical_text.mean()]
analytical_stds = [analytical_interactive.std(), analytical_text.std()]
emotional_means = [emotional_interactive.mean(), emotional_text.mean()]
emotional_stds = [emotional_interactive.std(), emotional_text.std()]

metric_positions = [0, 1]
bar_width = 0.35
interactive_x = [x - (bar_width / 2) for x in metric_positions]
text_x = [x + (bar_width / 2) for x in metric_positions]

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(
      interactive_x,
      [analytical_means[0], emotional_means[0]],
      yerr=[analytical_stds[0], emotional_stds[0]],
      width=bar_width,
      label="Interactive",
      capsize=6,
)
ax.bar(
      text_x,
      [analytical_means[1], emotional_means[1]],
      yerr=[analytical_stds[1], emotional_stds[1]],
      width=bar_width,
      label="Text",
      capsize=6,
)

ax.set_xticks(metric_positions)
ax.set_xticklabels(["Analytical Trust Difference", "Emotional Trust Difference"])
ax.set_ylabel("Mean (with SD)")
ax.set_title("Trust Difference by Condition")
ax.legend()


def _draw_significance_bar(x1: float, x2: float, y: float, label: str) -> None:
      h = 0.2
      ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.2)
      ax.text((x1 + x2) / 2, y + h + 0.03, label, ha="center", va="bottom")


analytical_top = max(analytical_means[0] + analytical_stds[0], analytical_means[1] + analytical_stds[1])
emotional_top = max(emotional_means[0] + emotional_stds[0], emotional_means[1] + emotional_stds[1])

_draw_significance_bar(interactive_x[0], text_x[0], analytical_top + 0.2, _sig_label(p_analytical))
_draw_significance_bar(interactive_x[1], text_x[1], emotional_top + 0.2, _sig_label(p_emotional))

plt.tight_layout()
plot_path = "./plots/trust_difference_by_condition.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot to {plot_path}")