from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path(__file__).resolve().parent / "all"


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


def _print_duplicate_columns(df: pd.DataFrame, label: str) -> None:
	duplicates = df.columns[df.columns.duplicated()].tolist()
	if duplicates:
		print(f"Duplicate columns in {label}: {duplicates}")


interactive_csv = _latest_csv("interactive_submissions")
student_csv = _latest_csv("student_responses")

interactive_df = pd.read_csv(interactive_csv)
student_responses_df = pd.read_csv(student_csv)


trust_group_1_df = pd.read_excel(RAW_DIR / "TrustGroup1(1-188).xlsx")
trust_group_2_df = pd.read_excel(RAW_DIR / "TrustGroup2(1-167).xlsx")

assert(False)


trust_group_1_df.columns = _remove_special_chars(trust_group_1_df.columns)
trust_group_2_df.columns = _remove_special_chars(trust_group_2_df.columns)

columns_to_drop = {
	"PleaseheadovertothefollowinglinkandinteractwiththeAIexplanationbeforeyoumoveontothesecondroundofquestionshttpstailiamalloya64ce4a49feeherokuappcombaseExplanationDes",
	"PleaseheadovertothefollowinglinkandinteractwiththeAIexplanationbeforeyoumoveontothesecondroundofquestionshttpstailiamalloya64ce4a49feeherokuappcomtextExplanationDes",
}

trust_group_1_df = trust_group_1_df.drop(columns=columns_to_drop, errors="ignore")
trust_group_2_df = trust_group_2_df.drop(columns=columns_to_drop, errors="ignore")

trust_group_1_rename_map = {
	"IamwaryofAIsystemsAIsystemsaredeceptive": "IamwaryofAIsystems2",
	"AIsystemsactionswillhaveaharmfulorinjuriousoutcome3": "AIsystemsaredeceptive2",
	"IcantrustAIsystemsAIsystemsaretrustworthy": "IcantrustAIsystems2",
	"HowdidyoufeelwheninteractingwiththisAI": "HowdidyoufeelwhenreadingthedefinitionofAI",
}

trust_group_1_df = trust_group_1_df.rename(columns=trust_group_1_rename_map)

trust_group_1_cols = list(trust_group_1_df.columns)
trust_group_2_cols = list(trust_group_2_df.columns)

missing_in_2 = [col for col in trust_group_1_cols if col not in set(trust_group_2_cols)]
missing_in_1 = [col for col in trust_group_2_cols if col not in set(trust_group_1_cols)]

if missing_in_2 or missing_in_1:
	raise ValueError(
		"Trust group columns do not match after normalization. "
		f"Only in trust_group_1: {missing_in_2}. "
		f"Only in trust_group_2: {missing_in_1}."
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
	"PleaseheadovertothefollowinglinkandinteractwiththeAIexplanationbeforeyoumoveontothesecondroundofquestionshttpstailiamalloya64ce4a49feeherokuappcombaseExplanationDes": "NA",
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

trust_group_1_df = trust_group_1_df.rename(columns=column_rename_map)
trust_group_2_df = trust_group_2_df.rename(columns=column_rename_map)

if "Prolific or Email" in trust_group_1_df.columns:
	series = trust_group_1_df["Prolific or Email"]
	mask = series.astype(str).str.match(r"^[^@]+@email\.prolific\.com$")
	trust_group_1_df.loc[mask, "Prolific or Email"] = series[mask].str.replace(
		r"@email\.prolific\.com$", "", regex=True
	)

if "Prolific or Email" in trust_group_2_df.columns:
	series = trust_group_2_df["Prolific or Email"]
	mask = series.astype(str).str.match(r"^[^@]+@email\.prolific\.com$")
	trust_group_2_df.loc[mask, "Prolific or Email"] = series[mask].str.replace(
		r"@email\.prolific\.com$", "", regex=True
	)

trust_group_1_df['Condition'] = "Interactive"
trust_group_2_df['Condition'] = "Static"


trust_group_1_df["RandomID"] = [uuid4().hex for _ in range(len(trust_group_1_df))]
trust_group_2_df["RandomID"] = [uuid4().hex for _ in range(len(trust_group_2_df))]

trust_group_1_df = trust_group_1_df.set_index("RandomID")
trust_group_2_df = trust_group_2_df.set_index("RandomID")

missing_in_2 = [col for col in trust_group_1_cols if col not in set(trust_group_2_cols)]
missing_in_1 = [col for col in trust_group_2_cols if col not in set(trust_group_1_cols)]

if missing_in_2 or missing_in_1:
	raise ValueError(
		"Trust group columns do not match after normalization. "
		f"Only in trust_group_1: {missing_in_2}. "
		f"Only in trust_group_2: {missing_in_1}."
	)

grouped = pd.concat([trust_group_1_df, trust_group_2_df], ignore_index=True)


print(grouped.columns)
"""
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
	   'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
	   'Consent 4', 'Prolific or Email', 'Education', 'AI Knowledge', 'Age',
	   'AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harmful',
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
	   'AI Job Post', 'Condition'],
"""


"""
Index(['id', 'page_title', 'section_title', 'section_index', 'original_text',
	   'updated_text', 'submission_type', 'created_at', 'user_identifier',
	   'completion_token', 'email', 'prolific_id', 'study_type', 'session_id'],
	  dtype='str')
Index(['id', 'page_title', 'section_title', 'section_index', 'response_text',
	   'created_at', 'updated_at', 'user_identifier', 'completion_token',
	   'email', 'prolific_id', 'study_type', 'session_id'],
	  dtype='str')
"""

list_columns = [
	"Interaction Name List",
	"Interaction Original List",
	"Interaction Updated List",
	"Interaction Type List",
	"Response Name List",
	"Response Text List",
]
for column in list_columns:
	grouped[column] = pd.Series([None] * len(grouped), dtype="object")

count = 0
for outer_idx, participant in grouped.iterrows():
	user_interactions = interactive_df[interactive_df['prolific_id'] == participant['Prolific or Email']]
	if(len(user_interactions) == 0):
		user_interactions = interactive_df[interactive_df['email'] == participant['Prolific or Email']]
		
	user_responses = student_responses_df[student_responses_df['prolific_id'] == participant['Prolific or Email']]
	if(len(user_responses) == 0):
		user_responses = student_responses_df[student_responses_df['email'] == participant['Prolific or Email']]
	if(len(user_interactions) == 0 and len(user_responses) == 0):
		grouped = grouped.drop(index=outer_idx)
		continue

	if(len(user_responses) == 0):
		user_responses = student_responses_df[student_responses_df['session_id'] == user_interactions['session_id'].to_list()[0]]
	if(len(user_interactions) == 0):
		user_interactions = interactive_df[interactive_df['session_id'] == user_responses['session_id'].to_list()[0]]
	
	Interaction_Names = []
	Interaction_Originals = []
	Interaction_Updateds = []
	Interaction_Types = []

	for interaction_idx, interaction in user_interactions.iterrows():
		Interaction_Names.append(str(interaction['page_title']) + " " + str(interaction['section_title']) + " " + str(interaction['section_index']))
		Interaction_Originals.append(str(interaction['original_text']))
		Interaction_Updateds.append(str(interaction['updated_text']))
		Interaction_Types.append(str(interaction['submission_type']))
		
	grouped.at[outer_idx, "Interaction Name List"] = Interaction_Names
	grouped.at[outer_idx, "Interaction Original List"] = Interaction_Originals
	grouped.at[outer_idx, "Interaction Updated List"] = Interaction_Updateds
	grouped.at[outer_idx, "Interaction Type List"] = Interaction_Types


	Response_Names = []
	Response_Texts = []

	for response_idx, user_response in user_responses.iterrows():
		Response_Names.append(str(user_response['page_title']) + " " + str(user_response['section_title']) + " " + str(user_response['section_index']))
		Response_Texts.append(str(user_response['response_text']))
	grouped.at[outer_idx, "Response Name List"] = Response_Names
	grouped.at[outer_idx, "Response Text List"] = Response_Texts

grouped.to_pickle("./grouped.pkl")
grouped.to_csv("./grouped.csv")