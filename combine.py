from pathlib import Path

import pandas as pd



Demographics = pd.read_csv("./data/Demographics.csv")
Questionnaires = pd.read_csv("./data/Questionnaires.csv")
Responses = pd.read_csv("./data/Responses.csv")
Submissions = pd.read_csv("./data/Submissions.csv")


"""
'Responses' : {'Study Type': study_type, 'Page Title': page_title, 'Section Title': section_title, 'Section Index': section_index, 'Response Text': response_text}
"""


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


Combined = Questionnaires.copy()

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

for idx, row in Combined.iterrows():
	prolific_key, email_key = _questionnaire_keys(row)

	matched_demographics = _find_rows_by_participant(Demographics_lookup, prolific_key, email_key)
	if not matched_demographics.empty:
		demographics_row = matched_demographics.iloc[0]
		for output_col, source_col in demographic_column_map.items():
			Combined.at[idx, output_col] = demographics_row[source_col]

	matched_responses = _find_rows_by_participant(Responses_lookup, prolific_key, email_key)
	Combined.at[idx, "Responses"] = [
		{
			"Study Type": response_row.get("study_type"),
			"Page Title": response_row.get("page_title"),
			"Section Title": response_row.get("section_title"),
			"Section Index": response_row.get("section_index"),
			"Response Text": response_row.get("response_text"),
		}
		for _, response_row in matched_responses.iterrows()
	]

	matched_submissions = _find_rows_by_participant(Submissions_lookup, prolific_key, email_key)
	Combined.at[idx, "Submissions"] = [
		{
			"Study Type": submission_row.get("study_type"),
			"Page Title": submission_row.get("page_title"),
			"Section Title": submission_row.get("section_title"),
			"Section Index": submission_row.get("section_index"),
			"Original Text": submission_row.get("original_text"),
			"Updated Text": submission_row.get("updated_text"),
		}
		for _, submission_row in matched_submissions.iterrows()
	]
output_path = Path("./data/Combined.csv")
Combined.to_csv(output_path, index=False)
print(f"Saved combined dataframe to: {output_path}")