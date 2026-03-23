import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import binomtest


DemographicsInteractive1 = pd.read_csv("DemographicsInteractive1.csv")
DemographicsInteractive2 = pd.read_csv("DemographicsInteractive2.csv")
DemographicsText1 = pd.read_csv("DemographicsText1.csv")
DemographicsText2 = pd.read_csv("DemographicsText2.csv")

DemographicsInteractive1 = DemographicsInteractive1[DemographicsInteractive1['Nationality'] != "CONSENT_REVOKED"]
DemographicsInteractive2 = DemographicsInteractive2[DemographicsInteractive2['Nationality'] != "CONSENT_REVOKED"]
DemographicsText1 = DemographicsText1[DemographicsText1['Nationality'] != "CONSENT_REVOKED"]
DemographicsText2 = DemographicsText2[DemographicsText2['Nationality'] != "CONSENT_REVOKED"]

Responses = pd.read_csv("Responses.csv")
Submissions = pd.read_csv("Submissions.csv")

responseIDs = Responses['prolific_id'].unique()

DemographicsInteractive1 = DemographicsInteractive1[DemographicsInteractive1['Participant id'].isin(responseIDs)]
DemographicsInteractive2 = DemographicsInteractive2[DemographicsInteractive2['Participant id'].isin(responseIDs)]
DemographicsText1 = DemographicsText1[DemographicsText1['Participant id'].isin(responseIDs)]
DemographicsText2 = DemographicsText2[DemographicsText2['Participant id'].isin(responseIDs)]

Interactive = pd.concat([DemographicsInteractive1, DemographicsInteractive2])
Text = pd.concat([DemographicsText1, DemographicsText2])

Interactive['Condition'] = "Interactive"
Text['Condition'] = "Text"

Combined = pd.concat([Interactive, Text])

print(Combined['Participant id'].unique())
print(Responses[prolific_id])

assert(False)

QuestionnaireInteractive = pd.read_excel("QuestionnaireInteractive.xlsx")
QuestionnaireText= pd.read_excel("QuestionnaireText.xlsx")
Questionnaire = pd.concat([QuestionnaireInteractive, QuestionnaireText])

#print(Questionnaire.columns)
WeirdEducation = ['Bachelor', 'Graduate Professional Degree', 'Master', 'PhD']
Questionnaire = Questionnaire.dropna(subset=['What is your education level?'])
WeirdAIKnowledge = ['Conceptual understanding', 'Advanced', 'Expert']

age_int = pd.to_numeric(Combined["Age"], errors="coerce").dropna().astype(int)
age_mean = age_int.mean()

WeirdCountries = ['United Kingdom', 'United States', 'Spain', 'Italy', 'Greece', 'Germany', 'France', 'Sweden', 'New Zealand', 'Australia', 'Romania', 'Turkey']

WeirdEthicities = ['White', 'Black/African American', 'Black/British', 'White Sephardic Jew','White Mexican']

"""
1 weird point if 'Country of birth' is in weirdCountries
1 weird point if 'nationality' is in weirdCountries 
1 weird point if Ethnicity is in weirdEthnicities 
1 weird point is 'Country of Residence' is in weirdCountries.  
1 weird point if 'Age' is below 32.59381
1 weird point if education in WeirdEducation
1 weird point if AI Knowledge is in WeirdAIKnowledge
"""


def _normalize_id(value: object) -> str:
	if pd.isna(value):
		return ""
	text = str(value).strip().lower()
	if text.endswith("@email.prolific.com"):
		text = text[: -len("@email.prolific.com")]
	return text


q_id_col = "What is your email adress or Prolific ID?"
q_edu_col = "What is your education level?"
q_ai_col = "What is your Artificial Intelligence Knowledge?"

Questionnaire = Questionnaire.copy()
Questionnaire["_join_key"] = Questionnaire[q_id_col].apply(_normalize_id)
Questionnaire["_email_key"] = Questionnaire["Email"].apply(_normalize_id)

# Replace non-identifying questionnaire IDs with email fallback where available.
invalid_ids = {"", "anonymous", "nan", "none"}
use_email_mask = Questionnaire["_join_key"].isin(invalid_ids) & ~Questionnaire["_email_key"].isin(invalid_ids)
Questionnaire.loc[use_email_mask, "_join_key"] = Questionnaire.loc[use_email_mask, "_email_key"]

questionnaire_lookup = (
	Questionnaire.dropna(subset=[q_edu_col, q_ai_col])
	.drop_duplicates(subset=["_join_key"], keep="first")
	[["_join_key", q_edu_col, q_ai_col]]
)

Combined["_join_key"] = Combined["Participant id"].apply(_normalize_id)
Combined = Combined.merge(questionnaire_lookup, on="_join_key", how="left")

Combined["Weird Score"] = 0
Combined["Weird Score"] += Combined["Country of birth"].isin(WeirdCountries).astype(int)
Combined["Weird Score"] += Combined["Nationality"].isin(WeirdCountries).astype(int)
Combined["Weird Score"] += Combined["Ethnicity"].isin(WeirdEthicities).astype(int)
Combined["Weird Score"] += Combined["Country of residence"].isin(WeirdCountries).astype(int)
Combined["Weird Score"] += (pd.to_numeric(Combined["Age"], errors="coerce") < age_mean).fillna(False).astype(int)
Combined["Weird Score"] += Combined[q_edu_col].isin(WeirdEducation).astype(int)
Combined["Weird Score"] += Combined[q_ai_col].isin(WeirdAIKnowledge).astype(int)

print("\n--- Weird Score By Condition ---")
print(
	Combined.groupby("Condition")["Weird Score"]
	.agg(["count", "mean", "std", "min", "max"])
)

print("\n--- Weird Score Distribution By Condition ---")
print(pd.crosstab(Combined["Weird Score"], Combined["Condition"]))


def _balance_gap_for_threshold(df: pd.DataFrame, threshold: int) -> float:
	weird_flag = df["Weird Score"] >= threshold
	weird_rate_by_condition = weird_flag.groupby(df["Condition"]).mean()
	if "Interactive" not in weird_rate_by_condition or "Text" not in weird_rate_by_condition:
		return float("inf")

	# Objective 1: keep Weird rate similar across conditions.
	condition_gap = abs(float(weird_rate_by_condition["Interactive"]) - float(weird_rate_by_condition["Text"]))

	# Objective 2: keep Weird True/False globally close to a 50/50 split.
	overall_true_rate = float(weird_flag.mean())
	class_balance_gap = abs(overall_true_rate - 0.5)

	# Equal-weight joint objective: lower is better.
	return condition_gap + class_balance_gap


min_score = int(Combined["Weird Score"].min())
max_score = int(Combined["Weird Score"].max())
candidate_thresholds = []
for t in range(min_score, max_score + 2):
	weird_flag = Combined["Weird Score"] >= t
	if weird_flag.any() and (~weird_flag).any():
		candidate_thresholds.append(t)

best_threshold = min(candidate_thresholds, key=lambda t: _balance_gap_for_threshold(Combined, t))

# Weird=True for scores at or above the chosen cutoff, else False.
Combined["Weird"] = Combined["Weird Score"] >= best_threshold

print("\n--- Best Weird Split Search ---")
print("Rule: score >= threshold -> Weird=True; score < threshold -> Weird=False")
print(f"Best threshold: {best_threshold}")
print(f"Balance gap (absolute difference in Weird rates): {_balance_gap_for_threshold(Combined, best_threshold):.6f}")

print("\n--- Weird (Boolean) Split By Condition ---")
split_counts = pd.crosstab(Combined["Weird"], Combined["Condition"])
print(split_counts)

print("\n--- Weird Rate By Condition ---")
print(Combined.groupby("Condition")["Weird"].mean())