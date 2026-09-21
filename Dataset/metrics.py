from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent
COMBINED_PATH = DATASET_DIR / "Combined.csv"
METRICS_PATH = DATASET_DIR / "Metrics.csv"

Combined = pd.read_csv(COMBINED_PATH)
ORIGINAL_COMBINED_COLUMNS = Combined.columns.tolist()

if "Status" in Combined.columns:
      Combined["Status"] = "Approved"

print(Combined.columns)

"""
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
       'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
       'Consent 4', 'Prolific or Email', 'Education', 'AI Knowledge', 'Age',
       'AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm',
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
       'AI Job Post', 'Condition', 'Submission id', 'Participant id', 'Status',
       'Custom study tncs accepted at', 'Started at', 'Completed at',
       'Reviewed at', 'Archived at', 'Time taken', 'Completion code',
       'Total approvals', 'Gender', 'Ethnicity', 'Age (Demographics)', 'Sex',
       'Ethnicity simplified', 'Country of birth', 'Country of residence',
       'Nationality', 'Language', 'Student status', 'Employment status',
       'Responses', 'Submissions'],
      dtype='str')
"""

AI_Analytical_Trust = [
      "AI Deceptive",
      "AI Dishonest",
      "AI Suspicious",
      "AI Wary",
      "AI Harm",
      "AI Confident",
      "AI Security",
      "AI Trustworthy",
      "AI Reliable",
      "AI Trust",
]

AI_Analytical_Trust_Post = [
      "AI Deceptive Post",
      "AI Dishonest Post",
      "AI Suspicious Post",
      "AI Wary Post",
      "AI Harm Post",
      "AI Confident Post",
      "AI Security Post",
      "AI Trustworthy Post",
      "AI Reliable Post",
      "AI Trust Post",
]

AI_Emotional_Trust = [f"AI systems are {i}" for i in range(1, 10)]
AI_Emotional_Trust_Post = [f"AI systems are {i} Post" for i in range(1, 10)]


def _normalize_text(value: object) -> str:
      if pd.isna(value):
            return ""
      return str(value).strip().lower()


def _coalesce_columns(df: pd.DataFrame, target: str, candidates: list[str]) -> bool:
      available = [col for col in candidates if col in df.columns]
      if not available:
            df[target] = pd.NA
            return False

      df[target] = df[available].bfill(axis=1).iloc[:, 0]
      return True


analytical_column_candidates = {
      "AI Deceptive": ["AI Deceptive"],
      "AI Dishonest": ["AI Dishonest", "AI Honest"],
      "AI Suspicious": ["AI Suspicious"],
      "AI Wary": ["AI Wary", "AI Weary"],
      "AI Harm": ["AI Harm"],
      "AI Confident": ["AI Confident"],
      "AI Security": ["AI Security"],
      "AI Trustworthy": ["AI Trustworthy"],
      "AI Reliable": ["AI Reliable"],
      "AI Trust": ["AI Trust"],
      "AI Deceptive Post": ["AI Deceptive Post", "AI Wary/Deceptive Post"],
      "AI Dishonest Post": ["AI Dishonest Post", "AI Honest Post"],
      "AI Suspicious Post": ["AI Suspicious Post"],
      "AI Wary Post": ["AI Wary Post", "AI Wary/Deceptive Post", "AI Weary Post"],
      "AI Harm Post": ["AI Harm Post", "AI Harm Post 1", "AI Harm Post 2"],
      "AI Confident Post": ["AI Confident Post"],
      "AI Security Post": ["AI Security Post"],
      "AI Trustworthy Post": ["AI Trustworthy Post"],
      "AI Reliable Post": ["AI Reliable Post"],
      "AI Trust Post": ["AI Trust Post", "AI Trust Post 2"],
}

emotional_column_candidates = {
      **{f"AI systems are {i}": [f"AI systems are {i}"] for i in range(1, 10)},
}

missing_canonical_columns: list[str] = []
for target, candidates in {**analytical_column_candidates, **emotional_column_candidates}.items():
      if not _coalesce_columns(Combined, target, candidates):
            missing_canonical_columns.append(target)

if missing_canonical_columns:
      print(f"Warning: missing expected columns after schema normalization: {missing_canonical_columns}")


likert_map = {
      "strongly disagree": -2,
      "disagree": -1,
      "neutral": 0,
      "neither agree nor disagree": 0,
      "agree": 1,
      "strongly agree": 2,
}

reverse_likert_map = {
      "strongly disagree": 2,
      "disagree": 1,
      "neutral": 0,
      "neither agree nor disagree": 0,
      "agree": -1,
      "strongly agree": -2,
}

#analytical polarity
"""
'AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm',
       'AI Confident', 'AI Security', 'AI Trustworthy', 'AI Reliable',
       'AI Trust',

       AI-systems are deceptive
AI-systems behave in an dishonest manner
I am suspicious of  AI-system's intent, action, or, outputs
I am wary of AI-systems
AI-system’s actions will have a harmful or injurious outcome
I am confident in AI-systems
AI-systems provide security
AI-systems are trustworthy
AI-system are reliable
I can trust  AI-systems
       
"""


positive_analytical_cols = {
      "AI Confident",
      "AI Security",
      "AI Trustworthy",
      "AI Reliable",
      "AI Trust",
      "AI Confident Post",
      "AI Security Post",
      "AI Trustworthy Post",
      "AI Reliable Post",
      "AI Trust Post",
}

negative_analytical_cols = set(AI_Analytical_Trust + AI_Analytical_Trust_Post) - positive_analytical_cols


# Emotional polarity by item: positive term -> +1, opposite term -> -1.
emotional_polarity = {
      "AI systems are 1": {"empathetic": 1, "apathetic": -1},
      "AI systems are 2": {"sensitive": 1, "insensitive": -1},
      "AI systems are 3": {"personal": 1, "impersonal": -1},
      "AI systems are 4": {"caring": 1, "ignoring": -1},
      "AI systems are 5": {"altruistic": 1, "self-serving": -1},
      "AI systems are 6": {"cordial": 1, "rude": -1},
      "AI systems are 7": {"responsive": 1, "indifferent": -1},
      "AI systems are 8": {"open-minded": 1, "judgmental": -1},
      "AI systems are 9": {"patient": 1, "impatient": -1},
}

emotional_labels = {
      "AI systems are 1": {1: "Empathetic", -1: "Apathetic"},
      "AI systems are 2": {1: "Sensitive", -1: "Insensitive"},
      "AI systems are 3": {1: "Personal", -1: "Impersonal"},
      "AI systems are 4": {1: "Caring", -1: "Ignoring"},
      "AI systems are 5": {1: "Altruistic", -1: "Self-Serving"},
      "AI systems are 6": {1: "Cordial", -1: "Rude"},
      "AI systems are 7": {1: "Responsive", -1: "Indifferent"},
      "AI systems are 8": {1: "Open-Minded", -1: "Judgmental"},
      "AI systems are 9": {1: "Patient", -1: "Impatient"},
}


post_emotional_raw_columns = [
      column
      for i in range(1, 10)
      for column in (f"AI systems are Post {i}",)
      if column in Combined.columns
]


def _normalize_post_emotional_columns(df: pd.DataFrame) -> None:
      if not post_emotional_raw_columns:
            return

      raw_snapshot = df.loc[:, post_emotional_raw_columns].copy()

      for pre_col, polarity in emotional_polarity.items():
            item_number = pre_col.rsplit(" ", 1)[-1]
            raw_target = f"AI systems are Post {item_number}"
            post_target = f"{pre_col} Post"
            labels = emotional_labels[pre_col]

            def _resolve_label(row: pd.Series) -> object:
                  for raw_col in post_emotional_raw_columns:
                        score = polarity.get(_normalize_text(row[raw_col]))
                        if score is not None:
                              return labels[score]
                  return pd.NA

            normalized_values = raw_snapshot.apply(_resolve_label, axis=1)
            if raw_target in df.columns:
                  df[raw_target] = normalized_values
            df[post_target] = normalized_values


_normalize_post_emotional_columns(Combined)
Combined.loc[:, ORIGINAL_COMBINED_COLUMNS].to_csv(COMBINED_PATH, index=False)


for column in positive_analytical_cols:
      if column in Combined.columns:
            Combined[column] = Combined[column].map(lambda x: likert_map.get(_normalize_text(x), 0))

for column in negative_analytical_cols:
      if column in Combined.columns:
            Combined[column] = Combined[column].map(lambda x: reverse_likert_map.get(_normalize_text(x), 0))


for pre_col, polarity in emotional_polarity.items():
      if pre_col in Combined.columns:
            Combined[pre_col] = Combined[pre_col].map(lambda x: polarity.get(_normalize_text(x), 0))

for pre_col, polarity in emotional_polarity.items():
      post_col = f"{pre_col} Post"

      if not post_emotional_raw_columns:
            Combined[post_col] = 0
            continue

      def _map_post_emotional(row: pd.Series) -> int:
            for raw_col in post_emotional_raw_columns:
                  mapped = polarity.get(_normalize_text(row[raw_col]))
                  if mapped is not None:
                        return mapped
            return 0

      Combined[post_col] = Combined.apply(_map_post_emotional, axis=1)


Combined["Total Analytical Trust"] = Combined[AI_Analytical_Trust].sum(axis=1)
Combined["Total Analytical Trust Post"] = Combined[AI_Analytical_Trust_Post].sum(axis=1)
Combined["Total Emotional Trust"] = Combined[AI_Emotional_Trust].sum(axis=1)
Combined["Total Emotional Trust Post"] = Combined[AI_Emotional_Trust_Post].sum(axis=1)

Combined["Emotional Trust Difference"] = Combined["Total Emotional Trust"] - Combined["Total Emotional Trust Post"]
Combined["Analytical Trust Difference"] = Combined["Total Analytical Trust"] - Combined["Total Analytical Trust Post"]

Combined.to_csv(METRICS_PATH, index=False)
print(f"Saved trust totals and differences to {METRICS_PATH}")

