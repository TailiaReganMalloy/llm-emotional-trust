from __future__ import annotations

from pathlib import Path

import pandas as pd
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
ALL_DIR = BASE_DIR / "all"

condition1 = pd.read_excel(ALL_DIR / "Condition1.xlsx")
condition2 = pd.read_excel(ALL_DIR / "Condition2.xlsx")

submissions = pd.read_csv("./all/public.interactive_submissions.csv")
responses = pd.read_csv("./all/public.student_responses.csv")



def _remove_special_chars(columns: pd.Index) -> pd.Index:
	return columns.str.replace(r"[^A-Za-z0-9_]+", "", regex=True)

condition1.columns = _remove_special_chars(condition1.columns)
condition2.columns = _remove_special_chars(condition2.columns)

columns_to_drop = {
	"PleaseheadovertothefollowinglinkandinteractwiththeAIexplanationbeforeyoumoveontothesecondroundofquestionshttpstailiamalloya64ce4a49feeherokuappcombaseExplanationDes",
	"PleaseheadovertothefollowinglinkandinteractwiththeAIexplanationbeforeyoumoveontothesecondroundofquestionshttpstailiamalloya64ce4a49feeherokuappcomtextExplanationDes",
}

condition1 = condition1.drop(columns=columns_to_drop, errors="ignore")
condition2 = condition2.drop(columns=columns_to_drop, errors="ignore")

column_aliases = {
	# Condition2 variants -> shared canonical names expected by column_rename_map.
	"IamwaryofAIsystemsAIsystemsaredeceptive": "IamwaryofAIsystems2",
	"AIsystemsactionswillhaveaharmfulorinjuriousoutcome3": "AIsystemsaredeceptive2",
	"IcantrustAIsystemsAIsystemsaretrustworthy": "IcantrustAIsystems2",
	"HowdidyoufeelwheninteractingwiththisAI": "HowdidyoufeelwhenreadingthedefinitionofAI",
}

condition1 = condition1.rename(columns=column_aliases)
condition2 = condition2.rename(columns=column_aliases)

condition1_cols = list(condition1.columns)
condition2_cols = list(condition2.columns)

missing_in_2 = [col for col in condition1_cols if col not in set(condition2_cols)]
missing_in_1 = [col for col in condition2_cols if col not in set(condition1_cols)]

if missing_in_2 or missing_in_1:
	raise ValueError(
		"Trust group columns do not match after normalization. "
		f"Only in condition1: {missing_in_2}. "
		f"Only in condition2: {missing_in_1}."
	)


column_rename_map = {
	"ID": "ID",
	"Starttime": "Start time",
	"Completiontime": "Completion time", 
	"Name": "Name",
	"Lastmodifiedtime": "Last modified time",
	"IconsenttothecollectionanduseofmypersonaldatainrelationtotheResearchProject": "Consent 1",
	"IagreetothedataIprovidebeingarchivedattheuniversityofLuxembourgandbeingusedinpseudonymizedformfortheResearchProject": "Consent 2",
	"IconsenttomypersonaldataasdescribedintheinformationsheetbeingprocessedforthepurposesofexplainableassessmentinAI": "Consent 3",
	"DataProtectionAgreementIhavereadtheinformationsheetandIhavebeeninformedbytheresearcherJulesWaxaboutthenatureandthepotentialconsequencesandrisksoftheabovementionedRe": "Consent 4",
	"WhatisyouremailadressorProlificID": "Prolific or Email",
	"Whatisyoureducationlevel": "Education",
	"WhatisyourArtificialIntelligenceKnowledge": "AI Knowledge",
	"Whatisyourage": "Age",
	"AIsystemsaredeceptive": "AI Deceptive",
	"AIsystemsbehaveinandishonestmanner": "AI Honest",
	"IamsuspiciousofAIsystemsintentactionoroutputs": "AI Suspicious",
	"IamwaryofAIsystems": "AI Weary",
	"AIsystemsactionswillhaveaharmfulorinjuriousoutcome": "AI Harm",
	"IamconfidentinAIsystems": "AI Confident",
	"AIsystemsprovidesecurity": "AI Security",
	"AIsystemsaretrustworthy": "AI Trustworthy",
	"AIsystemarereliable": "AI Reliable",
	"IcantrustAIsystems": "AI Trust",
	"AIsystemsare": "AI systems are 1",
	"AIsystemsare2": "AI systems are 2",
	"AIsystemsare3": "AI systems are 3",
	"AIsystemsare4": "AI systems are 4",	
	"AIsystemsare5": "AI systems are 5",
	"AIsystemsare6": "AI systems are 6",
	"AIsystemsare7": "AI systems are 7",
	"AIsystemsare8": "AI systems are 8",
	"AIsystemsare9": "AI systems are 9",
	"PleaseheadovertothefollowinglinkandinteractwiththeAIexplanationbeforeyoumoveontothesecondroundofquestionshttpstailiamalloya64ce4a49feeherokuappcombaseExplanationDes": "Open Response",
	"AIsystemsactionswillhaveaharmfulorinjuriousoutcome2": "AI Harm Post",
	"AIsystemsprovidesecurity2": "AI Security Post",
	"IamsuspiciousofAIsystemsintentactionoroutputs2": "AI Suspicious Post",
	"IamwaryofAIsystems2": "AI Weary Post",
	"AIsystemsaredeceptive2": "AI Deceptive Post",
	"IamconfidentinAIsystems2": "AI Confident Post",
	"AIsystemsbehaveinandishonestmanner2": "AI Honest Post",
	"IcantrustAIsystems2": "AI Trustworthy Post",
	"AIsystemarereliable2": "AI Reliable Post",
	"AIsystemsaretrustworthy2": "AI Trust Post",
	"AIsystemsare10": "AI systems are 1 Post",
	"AIsystemsare11": "AI systems are 2 Post",
	"AIsystemsare12": "AI systems are 3 Post",
	"AIsystemsare13": "AI systems are 4 Post",
	"AIsystemsare14": "AI systems are 5 Post",
	"AIsystemsare15": "AI systems are 6 Post",
	"AIsystemsare16": "AI systems are 7 Post",
	"AIsystemsare17": "AI systems are 8 Post",
	"AIsystemsare18": "AI systems are 9 Post",
	"HowdidyoufeelwhenreadingthedefinitionofAI": "AI Feel Post",
	"Doyouneedtounderstandthemodelinordertotrustit": "AI Understand Post",
	"HowwouldyoufeelifanAIsystemwereusedtoscreenyourapplicationforajobposition": "AI Job Post",
}

condition1 = condition1.rename(columns=column_rename_map)
condition2 = condition2.rename(columns=column_rename_map)

if "Prolific or Email" in condition1.columns:
	series = condition1["Prolific or Email"]
	mask = series.astype(str).str.match(r"^[^@]+@email\.prolific\.com$")
	condition1.loc[mask, "Prolific or Email"] = series[mask].str.replace(
		r"@email\.prolific\.com$", "", regex=True
	)

if "Prolific or Email" in condition2.columns:
	series = condition2["Prolific or Email"]
	mask = series.astype(str).str.match(r"^[^@]+@email\.prolific\.com$")
	condition2.loc[mask, "Prolific or Email"] = series[mask].str.replace(
		r"@email\.prolific\.com$", "", regex=True
	)

condition1['Condition'] = "Interactive"
condition2['Condition'] = "Static"


condition1["RandomID"] = [uuid4().hex for _ in range(len(condition1))]
condition2["RandomID"] = [uuid4().hex for _ in range(len(condition2))]

condition1 = condition1.set_index("RandomID")
condition2 = condition2.set_index("RandomID")

condition1_cols = list(condition1.columns)
condition2_cols = list(condition2.columns)

missing_in_2 = [col for col in condition1_cols if col not in set(condition2_cols)]
missing_in_1 = [col for col in condition2_cols if col not in set(condition1_cols)]

if missing_in_2 or missing_in_1:
	raise ValueError(
		"Trust group columns do not match after normalization. "
		f"Only in condition1: {missing_in_2}. "
		f"Only in condition2: {missing_in_1}."
	)

grouped = pd.concat([condition1, condition2], ignore_index=True)


"""print(grouped.columns)
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
       'AI Job Post', 'Condition'],
      dtype='str')
"""




"""print(interactive_submissions.columns)
Index(['id', 'page_title', 'section_title', 'section_index', 'original_text',
       'updated_text', 'submission_type', 'created_at', 'user_identifier',
       'completion_token', 'email', 'prolific_id', 'study_type', 'session_id'],
      dtype='str')
"""

"""print(student_responses.columns)
Index(['id', 'page_title', 'section_title', 'section_index', 'response_text',
	   'created_at', 'updated_at', 'user_identifier', 'completion_token',
	   'email', 'prolific_id', 'study_type', 'session_id'],
	  dtype='str')
"""


def _normalize_identifier(value: object) -> str:
	if pd.isna(value):
		return ""
	text = str(value).strip()
	if text.lower().endswith("@email.prolific.com"):
		text = text[: -len("@email.prolific.com")]
	return text.lower()


interactive_submissions = interactive_submissions.copy()
student_responses = student_responses.copy()

interactive_submissions["_match_email"] = interactive_submissions["email"].apply(_normalize_identifier)
interactive_submissions["_match_prolific"] = interactive_submissions["prolific_id"].apply(_normalize_identifier)

student_responses["_match_email"] = student_responses["email"].apply(_normalize_identifier)
student_responses["_match_prolific"] = student_responses["prolific_id"].apply(_normalize_identifier)

interactive_list_columns = [
	"Interactive Page Title List",
	"Interactive Section Title List",
	"Interactive Section Index List",
	"Interactive Original Text List",
	"Interactive Updated Text List",
]
student_list_columns = [
	"Response Section Title List",
	"Response Section Index List",
	"Response Text List",
	"Response Created At List",
	"Response Updated At List",
]

for column in interactive_list_columns + student_list_columns:
	grouped[column] = pd.Series([None] * len(grouped), dtype="object")


for idx, row in grouped.iterrows():
	participant_id = _normalize_identifier(row["Prolific or Email"])

	matched_interactive = interactive_submissions[
		(interactive_submissions["_match_email"] == participant_id)
		| (interactive_submissions["_match_prolific"] == participant_id)
	]

	matched_responses = student_responses[
		(student_responses["_match_email"] == participant_id)
		| (student_responses["_match_prolific"] == participant_id)
	]

	grouped.at[idx, "Interactive Page Title List"] = matched_interactive["page_title"].astype(str).tolist()
	grouped.at[idx, "Interactive Section Title List"] = matched_interactive["section_title"].astype(str).tolist()
	grouped.at[idx, "Interactive Section Index List"] = matched_interactive["section_index"].tolist()
	grouped.at[idx, "Interactive Original Text List"] = matched_interactive["original_text"].astype(str).tolist()
	grouped.at[idx, "Interactive Updated Text List"] = matched_interactive["updated_text"].astype(str).tolist()

	grouped.at[idx, "Response Section Title List"] = matched_responses["section_title"].astype(str).tolist()
	grouped.at[idx, "Response Section Index List"] = matched_responses["section_index"].tolist()
	grouped.at[idx, "Response Text List"] = matched_responses["response_text"].astype(str).tolist()
	grouped.at[idx, "Response Created At List"] = matched_responses["created_at"].astype(str).tolist()
	grouped.at[idx, "Response Updated At List"] = matched_responses["updated_at"].astype(str).tolist()


output_path = ALL_DIR / "grouped_with_interactions.csv"
grouped.to_csv(output_path, index=False)
print(f"Saved grouped data with interaction/response list columns to: {output_path}")