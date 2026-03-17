import pandas as pd 

grouped_with_interactions = pd.read_csv("./data/all/grouped_with_interactions.csv")


def _normalize_identifier(value: object) -> str:
      if pd.isna(value):
            return ""
      text = str(value).strip().lower()
      if text.endswith("@email.prolific.com"):
            text = text[: -len("@email.prolific.com")]
      return text


df = grouped_with_interactions[["Condition", "Prolific or Email"]].copy()
df["_normalized_id"] = df["Prolific or Email"].apply(_normalize_identifier)
df = df[df["_normalized_id"] != ""]

cross_condition_ids = (
      df.groupby("_normalized_id")["Condition"]
      .nunique()
      .loc[lambda s: s > 1]
      .index
)

print("BOTH CONDITIONS", len(cross_condition_ids))
if len(cross_condition_ids) > 0:
      overlapping = (
            df[df["_normalized_id"].isin(cross_condition_ids)]
            .sort_values(["_normalized_id", "Condition"]) 
            [["_normalized_id", "Condition", "Prolific or Email"]]
            .drop_duplicates()
      )

grouped_with_interactions["_normalized_id"] = grouped_with_interactions[
      "Prolific or Email"
].apply(_normalize_identifier)

cleaned = grouped_with_interactions[
      ~grouped_with_interactions["_normalized_id"].isin(cross_condition_ids)
].copy()

cleaned = cleaned.drop(columns=["_normalized_id"])

output_path = "./data/all/grouped_with_interactions_no_cross_condition.csv"
cleaned.to_csv(output_path, index=False)

#print(grouped_with_interactions.columns)
"""
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
       'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
       'Consent 4', 'Prolific or Email', 'Education', 'AI Knowledge', 'Age',
       'AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm',
       'AI Confident', 'AI Security', 'AI Trustworthy', 'AI Reliable',
       'AI Trust', 'AI systems are 1', 'AI systems are 2', 'AI systems are 3',
       'AI systems are 4', 'AI systems are 5', 'AI systems are 6',
       'AI systems are 7', 'AI systems are 8', 'AI systems are 9',
       'AI Weary Post', 'AI Confident Post', 'AI Suspicious Post',
       'AI Trust Post', 'AI Harm Post', 'AI Honest Post', 'AI Security Post',
       'AI Deceptive Post', 'AI Reliable Post', 'AI Trustworthy Post',
       'AI systems are 1 Post', 'AI systems are 2 Post',
       'AI systems are 3 Post', 'AI systems are 4 Post',
       'AI systems are 5 Post', 'AI systems are 6 Post',
       'AI systems are 7 Post', 'AI systems are 8 Post',
       'AI systems are 9 Post', 'AI Feel Post', 'AI Understand Post',
       'AI Job Post', 'Condition', 'Interactive Page Title List',
       'Interactive Section Title List', 'Interactive Section Index List',
       'Interactive Original Text List', 'Interactive Updated Text List',
       'Response Section Title List', 'Response Section Index List',
       'Response Text List', 'Response Created At List',
       'Response Updated At List'],
      dtype='str')
"""


demographics_interactive = pd.read_csv("data/all/demographic_interactive.csv")
#print(demographics_interactive.columns)
"""
Index(['Submission id', 'Participant id', 'Status',
       'Custom study tncs accepted at', 'Started at', 'Completed at',
       'Reviewed at', 'Archived at', 'Time taken', 'Completion code',
       'Total approvals', 'Gender', 'Ethnicity', 'Age', 'Sex',
       'Ethnicity simplified', 'Country of birth', 'Country of residence',
       'Nationality', 'Language', 'Student status', 'Employment status'],
      dtype='str')
"""
demographics_text = pd.read_csv("data/all/demographics_text.csv")
#print(demographics_text.columns)
"""
Index(['Submission id', 'Participant id', 'Status',
       'Custom study tncs accepted at', 'Started at', 'Completed at',
       'Reviewed at', 'Archived at', 'Time taken', 'Completion code',
       'Total approvals', 'Gender', 'Ethnicity', 'Age', 'Sex',
       'Ethnicity simplified', 'Country of birth', 'Country of residence',
       'Nationality', 'Language', 'Student status', 'Employment status'],
      dtype='str')
"""


cleaned_with_id = cleaned.copy()
cleaned_with_id["_normalized_id"] = cleaned_with_id["Prolific or Email"].apply(_normalize_identifier)

demographics_all = pd.concat(
      [demographics_interactive, demographics_text],
      ignore_index=True,
      sort=False,
)
demographics_all["_normalized_id"] = demographics_all["Participant id"].apply(_normalize_identifier)
demographics_all = demographics_all.drop_duplicates(subset=["_normalized_id"], keep="first")

# Keep all demographic columns; if there is a name collision (e.g., Age), use _demographic suffix.
combined = cleaned_with_id.merge(
      demographics_all,
      on="_normalized_id",
      how="left",
      suffixes=("", "_demographic"),
)
combined = combined.drop(columns=["_normalized_id"])

combined_output_path = "./data/all/combined_with_demographics.csv"
combined.to_csv(combined_output_path, index=False)

#print(f"Combined shape: {combined.shape}")
#print(f"Saved combined dataset to: {combined_output_path}")



""" print(combined.columns)
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
       'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
       'Consent 4', 'Prolific or Email', 'Education', 'AI Knowledge', 'Age',
       'AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm',
       'AI Confident', 'AI Security', 'AI Trustworthy', 'AI Reliable',
       'AI Trust', 'AI systems are 1', 'AI systems are 2', 'AI systems are 3',
       'AI systems are 4', 'AI systems are 5', 'AI systems are 6',
       'AI systems are 7', 'AI systems are 8', 'AI systems are 9',
       'AI Weary Post', 'AI Confident Post', 'AI Suspicious Post',
       'AI Trust Post', 'AI Harm Post', 'AI Honest Post', 'AI Security Post',
       'AI Deceptive Post', 'AI Reliable Post', 'AI Trustworthy Post',
       'AI systems are 1 Post', 'AI systems are 2 Post',
       'AI systems are 3 Post', 'AI systems are 4 Post',
       'AI systems are 5 Post', 'AI systems are 6 Post',
       'AI systems are 7 Post', 'AI systems are 8 Post',
       'AI systems are 9 Post', 'AI Feel Post', 'AI Understand Post',
       'AI Job Post', 'Condition', 'Interactive Page Title List',
       'Interactive Section Title List', 'Interactive Section Index List',
       'Interactive Original Text List', 'Interactive Updated Text List',
       'Response Section Title List', 'Response Section Index List',
       'Response Text List', 'Response Created At List',
       'Response Updated At List', 'Submission id', 'Participant id', 'Status',
       'Custom study tncs accepted at', 'Started at', 'Completed at',
       'Reviewed at', 'Archived at', 'Time taken', 'Completion code',
       'Total approvals', 'Gender', 'Ethnicity', 'Age_demographic', 'Sex',
       'Ethnicity simplified', 'Country of birth', 'Country of residence',
       'Nationality', 'Language', 'Student status', 'Employment status'],
      dtype='str')
"""



combined = combined[combined["Status"].astype(str).str.upper() == "APPROVED"].copy()

print("Condition counts in combined (APPROVED only):")
print(combined["Condition"].value_counts(dropna=False))

combined.to_csv("./data/combined.csv")
combined.to_pickle("./data/combined.pkl")

