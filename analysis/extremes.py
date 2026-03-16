import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

combined = pd.read_pickle("./data/grouped.pkl")
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
       'AI Job Post', 'Condition', 'Interaction Name List',
       'Interaction Original List', 'Interaction Updated List',
       'Interaction Type List', 'Response Name List', 'Response Text List'],
      dtype='str')
"""

AI_systems_are_map = {
	"AI systems are 1": {"Apathetic":0, "Empathetic":1},
	"AI systems are 2": {"Insensitive":0, "Sensitive":1},
	"AI systems are 3": {"Impersonal":0, "Personal":1},
	"AI systems are 4": {"Ignoring":0, "Caring":1},
	"AI systems are 5": {"Self Serving":0, "Altruistic":1},
	"AI systems are 6": {"Rude":0, "Cordial":1},
	"AI systems are 7": {"Indifferent":0, "Responsive":1},
	"AI systems are 8": {"Judgemental":0, "Open-Minded":1},
	"AI systems are 9": {"Impatient":0, "Patient":1},
}

likert_list = ['AI Harm Post', 'AI Security Post', 'AI Suspicious Post',
       'AI Weary Post', 'AI Deceptive Post', 'AI Confident Post',
       'AI Honest Post', 'AI Trustworthy Post', 'AI Reliable Post',
       'AI Trust Post', 'AI systems are 1 Post', 'AI systems are 2 Post', 
	   'AI systems are 1 Post', 'AI systems are 2 Post',
       'AI systems are 3 Post', 'AI systems are 4 Post',
       'AI systems are 5 Post', 'AI systems are 6 Post',
       'AI systems are 7 Post', 'AI systems are 8 Post',
       'AI systems are 9 Post',]

likert_map = {
      "Strongly Disagree": -2,
      "Disagree": -1,
      "Agree": 1,
      "Strongly Agree": 2,
}


def clean_text(value: str) -> str:
      text = str(value).strip().lower().replace("\xa0", " ").replace("-", " ")
      text = text.replace("judgmental", "judgemental")
      return " ".join(text.split())


ai_value_map = {}
for pair_map in AI_systems_are_map.values():
      for key, val in pair_map.items():
            ai_value_map[clean_text(key)] = val

likert_value_map = {clean_text(key): val for key, val in likert_map.items()}


numerical = combined.copy()

ai_columns = [col for col in numerical.columns if col.startswith("AI systems are ")]
for col in ai_columns:
      numerical[col] = (
            numerical[col]
                  .astype(str)
                  .map(clean_text)
                  .map(ai_value_map)
      )

for col in sorted(set(likert_list)):
      if col in numerical.columns:
            numerical[col] = (
                  numerical[col]
                        .astype(str)
                        .map(clean_text)
                        .map(likert_value_map)
            )

numerical.to_pickle("./data/numerical.pkl")