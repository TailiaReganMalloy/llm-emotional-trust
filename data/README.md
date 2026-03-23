# Data Dictionary

This file documents columns in `data/Combined.csv`.

Notes:
- `dtype` values below are pandas inferred dtypes when reading `data/Combined.csv` with default `pd.read_csv(...)` settings.
- `Source raw file(s)` lists where values were originally collected.
- `Condition`, `Responses`, and `Submissions` are assembled/derived during processing in `analysis.py`.

| Column | Description | dtype | Source raw file(s) |
|---|---|---|---|
| ID | Questionnaire record identifier. | int64 | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Start time | Questionnaire start timestamp. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Completion time | Questionnaire completion timestamp. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Email | Participant email from questionnaire export. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Name | Participant name field from questionnaire export. | float64 | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Last modified time | Last modification timestamp of questionnaire row. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Consent 1 | Consent item response 1. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Consent 2 | Consent item response 2. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Consent 3 | Consent item response 3. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Consent 4 | Consent item response 4. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Prolific or Email | Participant identifier entered in questionnaire (Prolific ID or email). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Education | Reported education level. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Knowledge | Self-reported AI knowledge level. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Age | Age from questionnaire. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Deceptive | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Honest | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Suspicious | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Weary | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Harm | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Confident | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Security | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Trustworthy | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Reliable | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Trust | Trust/safety item (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 1 | Semantic differential item 1 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 2 | Semantic differential item 2 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 3 | Semantic differential item 3 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 4 | Semantic differential item 4 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 5 | Semantic differential item 5 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 6 | Semantic differential item 6 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 7 | Semantic differential item 7 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 8 | Semantic differential item 8 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 9 | Semantic differential item 9 (pre). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Harm Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Security Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Suspicious Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Weary Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Deceptive Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Confident Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Honest Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Trustworthy Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Reliable Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Trust Post | Trust/safety item (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 1 Post | Semantic differential item 1 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 2 Post | Semantic differential item 2 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 3 Post | Semantic differential item 3 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 4 Post | Semantic differential item 4 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 5 Post | Semantic differential item 5 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 6 Post | Semantic differential item 6 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 7 Post | Semantic differential item 7 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 8 Post | Semantic differential item 8 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI systems are 9 Post | Semantic differential item 9 (post). | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Feel Post | Post prompt emotional reaction item. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Understand Post | Post prompt model-understanding/trust item. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| AI Job Post | Post prompt job-screening attitude item. | str | `data/raw/QuestionnaireInteractive.xlsx`, `data/raw/QuestionnaireText.xlsx` |
| Condition | Experimental condition label added during processing. | str | Derived from questionnaire file (`Interactive` vs `Text`) |
| Submission id | Demographics submission identifier. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Participant id | Demographics participant identifier. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Status | Demographics record status. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Custom study tncs accepted at | Consent acceptance timestamp in demographics export. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Started at | Demographics survey start timestamp. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Completed at | Demographics survey completion timestamp. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Reviewed at | Demographics review timestamp. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Archived at | Demographics archive timestamp. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Time taken | Time spent on demographics survey. | float64 | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Completion code | Completion code from demographics platform. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Total approvals | Approval count in demographics export. | float64 | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Gender | Participant gender. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Ethnicity | Participant ethnicity. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Age (Demographics) | Age from demographics export (kept separate from questionnaire age). | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Sex | Participant sex. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Ethnicity simplified | Simplified ethnicity category. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Country of birth | Participant country of birth. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Country of residence | Participant country of residence. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Nationality | Participant nationality. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Language | Participant language. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Student status | Participant student status. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Employment status | Participant employment status. | str | `data/raw/DemographicsInteractive1.csv`, `data/raw/DemographicsInteractive2.csv`, `data/raw/DemographicsText1.csv`, `data/raw/DemographicsText2.csv` |
| Responses | List of response dictionaries per participant with keys: `Study Type`, `Page Title`, `Section Title`, `Section Index`, `Response Text`. Stored as stringified list in CSV. | str | `data/raw/WebsiteResponses.csv` |
| Submissions | List of submission dictionaries per participant with keys: `Study Type`, `Page Title`, `Section Title`, `Section Index`, `Original Text`, `Updated Text`. Stored as stringified list in CSV. | str | `data/raw/WebsiteSubmissions.csv` |
