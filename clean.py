import pandas as pd 
import matplotlib.pyplot as plt
from uuid import uuid4

Metrics = pd.read_csv("./data/Metrics.csv")

""" print(Metrics.columns)
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
       'Responses', 'Submissions', 'Total Analytical Trust',
       'Total Analytical Trust Post', 'Total Emotional Trust',
       'Total Emotional Trust Post', 'Emotional Trust Difference',
       'Analytical Trust Difference'],
      dtype='str')
"""

Cleaned = Metrics[Metrics['Time taken'] < 10000] # Removing extreme outliers 

time_taken = pd.to_numeric(Cleaned["Time taken"], errors="coerce").dropna()
mean_time_taken = time_taken.mean()
std_time_taken = time_taken.std()

lower_bound = mean_time_taken - (3 * std_time_taken)
upper_bound = mean_time_taken + (3 * std_time_taken)

time_taken_numeric = pd.to_numeric(Cleaned["Time taken"], errors="coerce")
Cleaned = Cleaned[time_taken_numeric.between(lower_bound, upper_bound, inclusive="both")]


participant_values = Cleaned["Prolific or Email"].astype(str)
participant_id_map = {value: uuid4().hex for value in participant_values.unique()}
Cleaned["ID"] = participant_values.map(participant_id_map)
Cleaned = Cleaned.drop(columns=["Prolific or Email"], errors="ignore")



"""print(Cleaned.columns)
Index(['ID', 'Start time', 'Completion time', 'Email', 'Name',
       'Last modified time', 'Consent 1', 'Consent 2', 'Consent 3',
       'Consent 4', 'Education', 'AI Knowledge', 'Age', 'AI Deceptive',
       'AI Honest', 'AI Suspicious', 'AI Weary', 'AI Harm', 'AI Confident',
       'AI Security', 'AI Trustworthy', 'AI Reliable', 'AI Trust',
       'AI systems are 1', 'AI systems are 2', 'AI systems are 3',
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
       'Responses', 'Submissions', 'Total Analytical Trust',
       'Total Analytical Trust Post', 'Total Emotional Trust',
       'Total Emotional Trust Post', 'Emotional Trust Difference',
       'Analytical Trust Difference'],
      dtype='str')
"""


Cleaned.to_csv("./data/Cleaned.csv", index=False)