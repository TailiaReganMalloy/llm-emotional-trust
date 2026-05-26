# Metrics Column Dictionary

Generated from data/Metrics.csv.

Total columns documented: 119

Grouping strategy:
- First by origin (metrics.py derived, combine.py derived, or raw carried fields from Combined.csv).
- Then by pandas dtype in Metrics.csv.

## Derived in metrics.py: Aggregate trust scores

### Dtype: int64

#### Total Analytical Trust
- What it is: Row-level aggregate metric for analytical trust.
- Source/Calculation: Calculated in metrics.py as the row-wise sum of AI_Analytical_Trust canonical columns.
- Data type: int64

#### Total Analytical Trust Post
- What it is: Row-level aggregate metric for analytical trust post.
- Source/Calculation: Calculated in metrics.py as the row-wise sum of AI_Analytical_Trust_Post canonical columns.
- Data type: int64

#### Total Emotional Trust
- What it is: Row-level aggregate metric for emotional trust.
- Source/Calculation: Calculated in metrics.py as the row-wise sum of AI_Emotional_Trust canonical columns.
- Data type: int64

#### Total Emotional Trust Post
- What it is: Row-level aggregate metric for emotional trust post.
- Source/Calculation: Calculated in metrics.py as the row-wise sum of AI_Emotional_Trust_Post canonical columns.
- Data type: int64

#### Emotional Trust Difference
- What it is: Row-level aggregate metric for emotional trust.
- Source/Calculation: Calculated in metrics.py as Total Emotional Trust - Total Emotional Trust Post.
- Data type: int64

#### Analytical Trust Difference
- What it is: Row-level aggregate metric for analytical trust.
- Source/Calculation: Calculated in metrics.py as Total Analytical Trust - Total Analytical Trust Post.
- Data type: int64

## Derived in metrics.py: Canonical trust item scores

### Dtype: int64

#### AI Deceptive
- What it is: Pre-intervention analytical trust item measuring perceived deceptiveness of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Deceptive'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Dishonest
- What it is: Pre-intervention analytical trust item measuring perceived dishonesty of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Dishonest', 'AI Honest'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Suspicious
- What it is: Pre-intervention analytical trust item measuring suspicion about AI intentions/actions/outputs.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Suspicious'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Wary
- What it is: Pre-intervention analytical trust item measuring wariness toward AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Wary', 'AI Weary'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Harm
- What it is: Pre-intervention analytical trust item measuring expected harmfulness of AI actions.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Harm'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Confident
- What it is: Pre-intervention analytical trust item measuring confidence in AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Confident'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Security
- What it is: Pre-intervention analytical trust item measuring perceived security provided by AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Security'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Trustworthy
- What it is: Pre-intervention analytical trust item measuring perceived trustworthiness of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Trustworthy'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Reliable
- What it is: Pre-intervention analytical trust item measuring perceived reliability of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Reliable'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Trust
- What it is: Pre-intervention analytical trust item measuring willingness to trust AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Trust'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI systems are 1
- What it is: Pre-intervention emotional trust item 1 (empathetic vs apathetic).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 1'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 2
- What it is: Pre-intervention emotional trust item 2 (sensitive vs insensitive).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 2'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 3
- What it is: Pre-intervention emotional trust item 3 (personal vs impersonal).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 3'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 4
- What it is: Pre-intervention emotional trust item 4 (caring vs ignoring).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 4'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 5
- What it is: Pre-intervention emotional trust item 5 (altruistic vs self-serving).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 5'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 6
- What it is: Pre-intervention emotional trust item 6 (cordial vs rude).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 6'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 7
- What it is: Pre-intervention emotional trust item 7 (responsive vs indifferent).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 7'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 8
- What it is: Pre-intervention emotional trust item 8 (open-minded vs judgmental).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 8'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 9
- What it is: Pre-intervention emotional trust item 9 (patient vs impatient).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 9'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI Security Post
- What it is: Post-intervention analytical trust item measuring perceived security provided by AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Security Post'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Suspicious Post
- What it is: Post-intervention analytical trust item measuring suspicion about AI intentions/actions/outputs.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Suspicious Post'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Confident Post
- What it is: Post-intervention analytical trust item measuring confidence in AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Confident Post'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Dishonest Post
- What it is: Post-intervention analytical trust item measuring perceived dishonesty of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Dishonest Post', 'AI Honest Post'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Trust Post
- What it is: Post-intervention analytical trust item measuring willingness to trust AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Trust Post', 'AI Trust Post 2'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Reliable Post
- What it is: Post-intervention analytical trust item measuring perceived reliability of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Reliable Post'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Trustworthy Post
- What it is: Post-intervention analytical trust item measuring perceived trustworthiness of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Trustworthy Post'], then applying standard Likert mapping (-2 to +2).
- Data type: int64

#### AI Wary Post
- What it is: Post-intervention analytical trust item measuring wariness toward AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Wary Post', 'AI Wary/Deceptive Post', 'AI Weary Post'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Deceptive Post
- What it is: Post-intervention analytical trust item measuring perceived deceptiveness of AI systems.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Deceptive Post', 'AI Wary/Deceptive Post'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI Harm Post
- What it is: Post-intervention analytical trust item measuring expected harmfulness of AI actions.
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI Harm Post', 'AI Harm Post 1', 'AI Harm Post 2'], then applying reverse Likert mapping (+2 to -2).
- Data type: int64

#### AI systems are 1 Post
- What it is: Post-intervention emotional trust item 1 (empathetic vs apathetic).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 1 Post', 'AI systems are Post 1'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 2 Post
- What it is: Post-intervention emotional trust item 2 (sensitive vs insensitive).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 2 Post', 'AI systems are Post 2'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 3 Post
- What it is: Post-intervention emotional trust item 3 (personal vs impersonal).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 3 Post', 'AI systems are Post 3'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 4 Post
- What it is: Post-intervention emotional trust item 4 (caring vs ignoring).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 4 Post', 'AI systems are Post 4'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 5 Post
- What it is: Post-intervention emotional trust item 5 (altruistic vs self-serving).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 5 Post', 'AI systems are Post 5'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 6 Post
- What it is: Post-intervention emotional trust item 6 (cordial vs rude).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 6 Post', 'AI systems are Post 6'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 7 Post
- What it is: Post-intervention emotional trust item 7 (responsive vs indifferent).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 7 Post', 'AI systems are Post 7'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 8 Post
- What it is: Post-intervention emotional trust item 8 (open-minded vs judgmental).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 8 Post', 'AI systems are Post 8'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

#### AI systems are 9 Post
- What it is: Post-intervention emotional trust item 9 (patient vs impatient).
- Source/Calculation: Calculated in metrics.py by coalescing first non-null value from candidate columns ['AI systems are 9 Post', 'AI systems are Post 9'], then mapping pair labels to polarity scores (+1/-1).
- Data type: int64

## Derived in combine.py: Matching/enrichment fields

### Dtype: float64

#### Submission id (Demographics)
- What it is: Fallback demographics copy of Submission id created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### PID (Demographics)
- What it is: Fallback demographics copy of PID created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Status (Demographics)
- What it is: Fallback demographics copy of Status created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Started at (Demographics)
- What it is: Fallback demographics copy of Started at created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Completed at (Demographics)
- What it is: Fallback demographics copy of Completed at created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Reviewed at (Demographics)
- What it is: Fallback demographics copy of Reviewed at created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Archived at (Demographics)
- What it is: Fallback demographics copy of Archived at created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Time taken (Demographics)
- What it is: Fallback demographics copy of Time taken created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Completion code (Demographics)
- What it is: Fallback demographics copy of Completion code created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Total approvals (Demographics)
- What it is: Fallback demographics copy of Total approvals created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Gender (Demographics)
- What it is: Fallback demographics copy of Gender created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Ethnicity (Demographics)
- What it is: Fallback demographics copy of Ethnicity created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Age (Demographics)
- What it is: Fallback demographics copy of Age created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Sex (Demographics)
- What it is: Fallback demographics copy of Sex created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Ethnicity simplified (Demographics)
- What it is: Fallback demographics copy of Ethnicity simplified created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Country of birth (Demographics)
- What it is: Fallback demographics copy of Country of birth created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Country of residence (Demographics)
- What it is: Fallback demographics copy of Country of residence created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Nationality (Demographics)
- What it is: Fallback demographics copy of Nationality created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Language (Demographics)
- What it is: Fallback demographics copy of Language created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Student status (Demographics)
- What it is: Fallback demographics copy of Student status created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Employment status (Demographics)
- What it is: Fallback demographics copy of Employment status created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

#### Condition (Demographics)
- What it is: Fallback demographics copy of Condition created during rematching in combine.py.
- Source/Calculation: Calculated in combine.py as a rematched fallback demographics field populated by matching participant keys (PID/email) after the primary merge.
- Data type: float64

### Dtype: int64

#### Is White
- What it is: Binary indicator for whether Ethnicity simplified equals White.
- Source/Calculation: Calculated in combine.py as Ethnicity simplified.isin(['White']).astype(int).
- Data type: int64

### Dtype: str

#### Explanation Comment
- What it is: Participant explanation/comment text carried forward from explanation prompt fields.
- Source/Calculation: Calculated in combine.py by taking first non-null text across explanation source columns via bfill(axis=1).
- Data type: str

#### Responses
- What it is: Nested list payload of response events matched from Responses.csv for each participant.
- Source/Calculation: Calculated in combine.py by participant matching and collecting structured rows from raw Responses.csv.
- Data type: str

#### Submissions
- What it is: Nested list payload of submission edit events matched from Submissions.csv for each participant.
- Source/Calculation: Calculated in combine.py by participant matching and collecting structured rows from raw Submissions.csv.
- Data type: str

## From Combined.csv raw merge: Demographics/admin fields

### Dtype: float64

#### Time taken
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: float64

### Dtype: int64

#### Total approvals
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: int64

#### Age
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: int64

### Dtype: str

#### Submission id
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### PID
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Status
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Started at
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Completed at
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Reviewed at
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Archived at
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Completion code
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Gender
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Ethnicity
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Sex
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Ethnicity simplified
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Country of birth
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Country of residence
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Nationality
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Language
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Student status
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Employment status
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Condition
- What it is: Raw merged field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

## From Combined.csv raw merge: Questionnaire/study fields

### Dtype: float64

#### Name
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: float64

#### Age.1
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: float64

### Dtype: int64

#### ID
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: int64

### Dtype: str

#### Start time
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Completion time
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Email
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Last modified time
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Education
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Knowledge
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Harm Post 1
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Wary/Deceptive Post
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Harm Post 2
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 1
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 2
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 3
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 4
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 5
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 6
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 7
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 8
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI systems are Post 9
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Interaction Feeling
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Need Model Understanding
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Job Screening Feeling
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### Age.2
- What it is: Raw questionnaire metadata/response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Trust Post 2
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str

#### AI Definition Feeling
- What it is: Raw questionnaire AI trust/emotion response field retained from Combined.csv.
- Source/Calculation: Not recalculated in metrics.py; carried from Combined.csv, which is produced in combine.py by merging raw questionnaire and demographics files and applying column renames/cleanup.
- Data type: str
