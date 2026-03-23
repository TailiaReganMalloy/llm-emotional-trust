import pandas as pd 

Combined = pd.read_csv("./data/Combined.csv")

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

AI_Analytical_Trust = ['AI Deceptive', 'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm', 'AI Confident', 'AI Security', 'AI Trustworthy', 'AI Reliable', 'AI Trust']

AI_Analytical_Trust_Post = ['AI Deceptive Post', 'AI Honest Post', 'AI Suspicious Post', 'AI Weary Post', 'AI Harm Post', 'AI Confident Post', 'AI Security Post', 'AI Trustworthy Post', 'AI Reliable Post', 'AI Trust Post']

AI_Emotional_Trust = ['AI systems are 1', 'AI systems are 2', 'AI systems are 3', 'AI systems are 4', 'AI systems are 5', 'AI systems are 6', 'AI systems are 7', 'AI systems are 8', 'AI systems are 9']

AI_Emotional_Trust_Post = ['AI systems are 1 Post', 'AI systems are 2 Post', 'AI systems are 3 Post', 'AI systems are 4 Post', 'AI systems are 5 Post', 'AI systems are 6 Post', 'AI systems are 7 Post', 'AI systems are 8 Post', 'AI systems are 9 Post']


def _normalize_text(value: object) -> str:
      if pd.isna(value):
            return ""
      return str(value).strip().lower()


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

for column in positive_analytical_cols:
      if column in Combined.columns:
            Combined[column] = Combined[column].map(lambda x: likert_map.get(_normalize_text(x), 0))

for column in negative_analytical_cols:
      if column in Combined.columns:
            Combined[column] = Combined[column].map(lambda x: reverse_likert_map.get(_normalize_text(x), 0))


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


for pre_col, polarity in emotional_polarity.items():
      post_col = f"{pre_col} Post"

      if pre_col in Combined.columns:
            Combined[pre_col] = Combined[pre_col].map(lambda x: polarity.get(_normalize_text(x), 0))

      if post_col in Combined.columns:
            Combined[post_col] = Combined[post_col].map(lambda x: polarity.get(_normalize_text(x), 0))


Combined["Total Analytical Trust"] = Combined[AI_Analytical_Trust].sum(axis=1)
Combined["Total Analytical Trust Post"] = Combined[AI_Analytical_Trust_Post].sum(axis=1)
Combined["Total Emotional Trust"] = Combined[AI_Emotional_Trust].sum(axis=1)
Combined["Total Emotional Trust Post"] = Combined[AI_Emotional_Trust_Post].sum(axis=1)

Combined["Emotional Trust Difference"] = Combined["Total Emotional Trust"] - Combined["Total Emotional Trust Post"]
Combined["Analytical Trust Difference"] = Combined["Total Analytical Trust"] - Combined["Total Analytical Trust Post"]

Combined.to_csv("./data/Metrics.csv", index=False)
print("Saved trust totals and differences to ./data/Metrics.csv")

