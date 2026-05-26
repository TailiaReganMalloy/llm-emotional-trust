import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
import unicodedata

DATA_DIR = './data/raw/'
QuestionnaireInteractive = pd.read_excel(DATA_DIR + 'QuestionnaireInteractive.xlsx')
QuestionnaireText= pd.read_excel(DATA_DIR + 'QuestionnaireText.xlsx')

DemographicsInteractive1 = pd.read_csv(DATA_DIR + 'DemographicsInteractive1.csv')
DemographicsInteractive1['Condition'] = 'Interactive'
DemographicsInteractive2 = pd.read_csv(DATA_DIR + 'DemographicsInteractive2.csv')
DemographicsInteractive2['Condition'] = 'Interactive'
DemographicsText1 = pd.read_csv(DATA_DIR + 'DemographicsText1.csv')
DemographicsText1['Condition'] = 'Text'
DemographicsText2 = pd.read_csv(DATA_DIR + 'DemographicsText2.csv')
DemographicsText2['Condition'] = 'Text'

Responses = pd.read_csv(DATA_DIR + 'Responses.csv')
Submissions = pd.read_csv(DATA_DIR + 'Submissions.csv')
Demographics = pd.concat([
	DemographicsInteractive1,
	DemographicsInteractive2, 
	DemographicsText1, 
	DemographicsText2
])


def _normalize_identifier(value: object) -> str:
	if pd.isna(value):
		return ""
	text = str(value).strip().lower()
	if text.endswith("@email.prolific.com"):
		text = text[: -len("@email.prolific.com")]
	return text


def _questionnaire_keys(row: pd.Series) -> tuple[str, str]:
	prolific_key = _normalize_identifier(row.get("Prolific or Email", ""))
	email_key = _normalize_identifier(row.get("Email", ""))

	# Some rows store email in the prolific field. Treat that as email fallback.
	if "@" in prolific_key:
		if not email_key:
			email_key = prolific_key
		prolific_key = ""

	return prolific_key, email_key


def _find_rows_by_participant(frame: pd.DataFrame, prolific_key: str, email_key: str) -> pd.DataFrame:
	if prolific_key:
		prolific_match = frame[frame["_match_prolific"] == prolific_key]
		if not prolific_match.empty:
			return prolific_match

	if email_key:
		email_match = frame[frame["_match_email"] == email_key]
		if not email_match.empty:
			return email_match

	return frame.iloc[0:0]


def _normalize_column_name(value: object) -> str:
	if pd.isna(value):
		return ""
	text = unicodedata.normalize("NFKD", str(value))
	text = text.replace("\xa0", " ")
	return "".join(ch for ch in text.lower() if ch.isalnum())



"""
Index(['Submission id', 'Participant id', 'Status',
       'Custom study tncs accepted at', 'Started at', 'Completed at',
       'Reviewed at', 'Archived at', 'Time taken', 'Completion code',
       'Total approvals', 'Gender', 'Ethnicity', 'Age', 'Sex',
       'Ethnicity simplified', 'Country of birth', 'Country of residence',
       'Nationality', 'Language', 'Student status', 'Employment status'],
      dtype='str')
"""

Questionnaire = pd.concat([
	QuestionnaireInteractive, 
	QuestionnaireText
])

Responses.rename(columns={'prolific_id': 'PID'}, inplace=True)
Submissions.rename(columns={'prolific_id': 'PID'}, inplace=True)
Demographics.rename(columns={'Participant id': 'PID'}, inplace=True)
Questionnaire.rename(columns={'What is your email adress or Prolific ID?': 'PID'}, inplace=True)

Combined = merged_df = pd.merge(Demographics, Questionnaire, on='PID')
Combined = Combined[Combined['Ethnicity simplified'] != 'CONSENT_REVOKED']
Combined = Combined.reset_index(drop=True)

white_ethnicities = ['White']

Combined['Is White'] = Combined['Ethnicity simplified'].isin(white_ethnicities).astype(int)

normalized_column_rename_map = {
	_normalize_column_name("What is your email adress or Prolific ID?"): "Prolific or Email",
	_normalize_column_name("What is your education level?"): "Education",
	_normalize_column_name("What is your Artificial Intelligence Knowledge?"): "AI Knowledge",
	_normalize_column_name("What is your age"): "Age",
	_normalize_column_name("What is your age?"): "Age",
	_normalize_column_name("AI-systems are deceptive"): "AI Deceptive",
	_normalize_column_name("AI-systems behave in an dishonest manner"): "AI Dishonest",
	_normalize_column_name("I am suspicious of AI-system's intent, action, or, outputs"): "AI Suspicious",
	_normalize_column_name("I am wary of AI-systems"): "AI Wary",
	_normalize_column_name("AI-system's actions will have a harmful or injurious outcome"): "AI Harm",
	_normalize_column_name("I am confident in AI-systems"): "AI Confident",
	_normalize_column_name("AI-systems provide security"): "AI Security",
	_normalize_column_name("AI-systems are trustworthy"): "AI Trustworthy",
	_normalize_column_name("AI-system are reliable"): "AI Reliable",
	_normalize_column_name("I can trust AI-systems"): "AI Trust",
	_normalize_column_name("AI-systems are"): "AI systems are 1",
	_normalize_column_name("AI-systems are2"): "AI systems are 2",
	_normalize_column_name("AI-systems are3"): "AI systems are 3",
	_normalize_column_name("AI-systems are4"): "AI systems are 4",
	_normalize_column_name("AI-systems are5"): "AI systems are 5",
	_normalize_column_name("AI-systems are6"): "AI systems are 6",
	_normalize_column_name("AI-systems are7"): "AI systems are 7",
	_normalize_column_name("AI-systems are8"): "AI systems are 8",
	_normalize_column_name("AI-systems are9"): "AI systems are 9",
	_normalize_column_name("AI-system's actions will have a harmful or injurious outcome 2"): "AI Harm Post 1",
	_normalize_column_name("AI-systems provide security 2"): "AI Security Post",
	_normalize_column_name("I am suspicious of AI-system's intent, action, or, outputs2"): "AI Suspicious Post",
	_normalize_column_name("I am wary of AI-systemsAI-systems are deceptive"): "AI Wary/Deceptive Post",
	_normalize_column_name("AI-system's actions will have a harmful or injurious outcome 3"): "AI Harm Post 2",
	_normalize_column_name("I am confident in AI-systems2"): "AI Confident Post",
	_normalize_column_name("AI-systems behave in an dishonest manner2"): "AI Dishonest Post",
	_normalize_column_name("I can trust AI-systemsAI-systems are trustworthy"): "AI Trust Post",
	_normalize_column_name("AI-system are reliable 2"): "AI Reliable Post",
	_normalize_column_name("AI-systems are trustworthy2"): "AI Trustworthy Post",
	_normalize_column_name("AI-systems are10"): "AI systems are Post 1",
	_normalize_column_name("AI-systems are11"): "AI systems are Post 2",
	_normalize_column_name("AI-systems are12"): "AI systems are Post 3",
	_normalize_column_name("AI-systems are13"): "AI systems are Post 4",
	_normalize_column_name("AI-systems are14"): "AI systems are Post 5",
	_normalize_column_name("AI-systems are15"): "AI systems are Post 6",
	_normalize_column_name("AI-systems are16"): "AI systems are Post 7",
	_normalize_column_name("AI-systems are17"): "AI systems are Post 8",
	_normalize_column_name("AI-systems are18"): "AI systems are Post 9",
	_normalize_column_name("I am wary of AI-systems2"): "AI Wary Post",
	_normalize_column_name("AI-systems are deceptive2"): "AI Deceptive Post",
	_normalize_column_name("I can trust AI-systems2"): "AI Trust Post 2",
	_normalize_column_name("How did you feel when interacting with this AI?"): "AI Interaction Feeling",
	_normalize_column_name("Do you need to understand the model in order to trust it?"): "Need Model Understanding",
	_normalize_column_name("How would you feel if an AI system were used to screen your application for a job position?"): "Job Screening Feeling",
	_normalize_column_name("How did you feel when reading the definition of AI?"): "AI Definition Feeling",
}

explanation_column_norms = {
	_normalize_column_name("Please head over to the following link and interact with the AI explanation before you move on to the second round of questions https://tailia-malloy-a64ce4a49fee.herokuapp.com/baseExplanation Des..."),
	_normalize_column_name("Please head over to the following link and interact with the AI explanation before you move on to the second round of questions https://tailia-malloy-a64ce4a49fee.herokuapp.com/textExplanation Des..."),
}

explanation_source_columns = [
	column for column in Combined.columns if _normalize_column_name(column) in explanation_column_norms
]
if explanation_source_columns:
	Combined["Explanation Comment"] = Combined[explanation_source_columns].bfill(axis=1).iloc[:, 0]

rename_map = {}
for column in Combined.columns:
	normalized_name = _normalize_column_name(column)
	if normalized_name in normalized_column_rename_map:
		rename_map[column] = normalized_column_rename_map[normalized_name]

Combined.rename(columns=rename_map, inplace=True)

drop_column_norms = {
	_normalize_column_name("I consent to the collection and use of my personal data in relation to the Research Project"),
	_normalize_column_name("I agree to the data I provide being archived at the university of Luxembourg and being used in pseudonymized form for the Research Project"),
	_normalize_column_name("I consent to my personal data, as described in the information sheet, being processed for the purposes of explainable assessment in AI."),
	_normalize_column_name("Data Protection Agreement I have read the information sheet and I have been informed by the researcher Jules Wax about the nature and the potential consequences and risks of the above-mentioned Re..."),
	_normalize_column_name("Custom study tncs accepted at"),
	_normalize_column_name("Please head over to the following link and interact with the AI explanation before you move on to the second round of questions https://tailia-malloy-a64ce4a49fee.herokuapp.com/baseExplanation Des..."),
	_normalize_column_name("Please head over to the following link and interact with the AI explanation before you move on to the second round of questions https://tailia-malloy-a64ce4a49fee.herokuapp.com/textExplanation Des..."),
}

columns_to_drop = [column for column in Combined.columns if _normalize_column_name(column) in drop_column_norms]
Combined.drop(columns=columns_to_drop, inplace=True, errors="ignore")


Demographics_lookup = Demographics.copy()
Demographics_lookup["_match_prolific"] = Demographics_lookup.get("Participant id", pd.Series(dtype="object")).apply(_normalize_identifier)
if "Email" in Demographics_lookup.columns:
	Demographics_lookup["_match_email"] = Demographics_lookup["Email"].apply(_normalize_identifier)
else:
	Demographics_lookup["_match_email"] = ""

# If participant id is actually an email for some rows, make it usable for fallback matching.
participant_is_email = Demographics_lookup["_match_prolific"].astype(str).str.contains("@", regex=False)
Demographics_lookup.loc[participant_is_email & (Demographics_lookup["_match_email"] == ""), "_match_email"] = Demographics_lookup.loc[
	participant_is_email & (Demographics_lookup["_match_email"] == ""), "_match_prolific"
]

demographic_column_map: dict[str, str] = {}
for col in Demographics.columns:
	if _normalize_column_name(col) == _normalize_column_name("Custom study tncs accepted at"):
		continue
	output_col = col if col not in Combined.columns else f"{col} (Demographics)"
	Combined[output_col] = None
	demographic_column_map[output_col] = col

Responses_lookup = Responses.copy()
Responses_lookup["_match_prolific"] = Responses_lookup.get("prolific_id", pd.Series(dtype="object")).apply(_normalize_identifier)
Responses_lookup["_match_email"] = Responses_lookup.get("email", pd.Series(dtype="object")).apply(_normalize_identifier)

Submissions_lookup = Submissions.copy()
Submissions_lookup["_match_prolific"] = Submissions_lookup.get("prolific_id", pd.Series(dtype="object")).apply(_normalize_identifier)
Submissions_lookup["_match_email"] = Submissions_lookup.get("email", pd.Series(dtype="object")).apply(_normalize_identifier)

Combined["Responses"] = pd.Series([None] * len(Combined), dtype="object")
Combined["Submissions"] = pd.Series([None] * len(Combined), dtype="object")

responses_payload: list[object] = []
submissions_payload: list[object] = []

for idx, row in Combined.iterrows():
	prolific_key, email_key = _questionnaire_keys(row)

	matched_demographics = _find_rows_by_participant(Demographics_lookup, prolific_key, email_key)
	if not matched_demographics.empty:
		demographics_row = matched_demographics.iloc[0]
		for output_col, source_col in demographic_column_map.items():
			Combined.at[idx, output_col] = demographics_row[source_col]

	matched_responses = _find_rows_by_participant(Responses_lookup, prolific_key, email_key)
	responses_payload.append([
		{
			"Study Type": response_row.get("study_type"),
			"Page Title": response_row.get("page_title"),
			"Section Title": response_row.get("section_title"),
			"Section Index": response_row.get("section_index"),
			"Response Text": response_row.get("response_text"),
		}
		for _, response_row in matched_responses.iterrows()
	])

	matched_submissions = _find_rows_by_participant(Submissions_lookup, prolific_key, email_key)
	submissions_payload.append([
		{
			"Study Type": submission_row.get("study_type"),
			"Page Title": submission_row.get("page_title"),
			"Section Title": submission_row.get("section_title"),
			"Section Index": submission_row.get("section_index"),
			"Original Text": submission_row.get("original_text"),
			"Updated Text": submission_row.get("updated_text"),
		}
		for _, submission_row in matched_submissions.iterrows()
	])

Combined["Responses"] = responses_payload
Combined["Submissions"] = submissions_payload

for column in Combined.columns:
	print(column)
output_path = Path("./data/Combined.csv")
Combined.to_csv(output_path, index=False)
print(f"Saved combined dataframe to: {output_path}")