from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path(__file__).resolve().parent / "raw"


def _latest_csv(prefix: str) -> Path:
	pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8}})_(\d{{6}})\.csv$")
	candidates = []
	for path in RAW_DIR.glob(f"{prefix}_*.csv"):
		match = pattern.match(path.name)
		if match:
			candidates.append((match.group(1) + match.group(2), path))

	if not candidates:
		raise FileNotFoundError(f"No CSV files found for prefix '{prefix}' in {RAW_DIR}")

	candidates.sort(key=lambda item: item[0])
	return candidates[-1][1]


def _remove_special_chars(columns: pd.Index) -> pd.Index:
	return columns.str.replace(r"[^A-Za-z0-9_]+", "", regex=True)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


interactive_csv = _latest_csv("interactive_submissions")
student_csv = _latest_csv("student_responses")

interactive_df = pd.read_csv(interactive_csv)
student_responses_df = pd.read_csv(student_csv)



for prolific_id in interactive_df['prolific_id'].unique():
	email = interactive_df['email'].unique()[0]
	print(email)
	user_interactions = interactive_df[interactive_df['prolific_id'] == prolific_id]
	user_responses = student_responses_df[student_responses_df['prolific_id'] == prolific_id]
	user_survey = trust_group_1_df[trust_group_1_df['Email'] == prolific_id] 
	if(len(user_survey) == 0):
		user_survey = trust_group_2_df[trust_group_2_df['Email'] == prolific_id] 
	if(len(user_survey) == 0):
		user_survey = trust_group_1_df[trust_group_1_df['Email'] == email] 
	if(len(user_survey) == 0):
		user_survey = trust_group_2_df[trust_group_2_df['Email'] == email] 
	if(len(user_survey) == 0):
		continue 
	count += 1 

print(count)

assert(False)

trust_group_1_df = trust_group_1_df.drop(columns=["Email", "Name"], errors="ignore")
trust_group_2_df = trust_group_2_df.drop(columns=["Email", "Name"], errors="ignore")

"""
Index(['ID', 'Starttime', 'Completiontime', 'Lastmodifiedtime', 'Consent',
       'Education', 'AIKnowledge', 'Age', 'AIDeceptive', 'AIHonest',
       'AISuspicious', 'AIWeary', 'AIHarmful', 'AIConfident', 'AISecurity',
       'AITrustworthy', 'AIReliable', 'AITrust', 'AIsystemsare1',
       'AIsystemsare2', 'AIsystemsare3', 'AIsystemsare4', 'AIsystemsare5',
       'AIsystemsare6', 'AIsystemsare7', 'AIsystemsare8', 'AIsystemsare9',
       'AIHarmPost', 'AISecurityPost', 'AISuspiciousPost', 'AIWearyPost',
       'AIDeceptivePost', 'AIConfidentPost', 'AIHonestPost',
       'AITrustworthyPost', 'AIReliablePost', 'AITrustPost',
       'AIsystemsare1Post', 'AIsystemsare2Post', 'AIsystemsare3Post',
       'AIsystemsare4Post', 'AIsystemsare5Post', 'AIsystemsare7Post',
       'AIsystemsare8Post', 'AIsystemsare9Post', 'AIFeelPost',
       'AIUnderstandPost', 'AIJobPost', 'Condition'],
      dtype='str')
"""

adjust = ['AIDeceptive', 'AIHonest', 'AISuspicious', 'AIWeary', 'AIHarmful', 'AIConfident', 'AISecurity', 'AITrustworthy', 'AIReliable', 'AITrust']

Columns = ['AIDeceptive', 'AIHonest', 'AISuspicious', 'AIWeary', 'AIHarmful', 'AIConfident', 'AISecurity', 'AITrustworthy', 'AIReliable', 'AITrust', 'AIsystemsare1', 'AIsystemsare2', 'AIsystemsare3', 'AIsystemsare4', 'AIsystemsare5', 'AIsystemsare6', 'AIsystemsare7', 'AIsystemsare8', 'AIsystemsare9',]

avg_differences = {}
prob_changes = {}

for column in Columns:
    post_column = f"{column}Post"
    if column not in grouped.columns or post_column not in grouped.columns:
        print(f"Skipping {column}: missing {column} or {post_column}")
        continue
    
    print(grouped[column])
    assert(False)
    pre_values = pd.to_numeric(grouped[column], errors="coerce")
    post_values = pd.to_numeric(grouped[post_column], errors="coerce")
    diff = post_values - pre_values
    avg_difference = diff.mean()
    prob_change = (diff != 0).mean()

    avg_differences[column] = avg_difference
    prob_changes[column] = prob_change

    print(f"{column}: average difference = {avg_difference:.4f}")

plt.figure(figsize=(12, 6))
plt.bar(prob_changes.keys(), prob_changes.values())
plt.xlabel("Column")
plt.ylabel("Probability of changing opinion")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
