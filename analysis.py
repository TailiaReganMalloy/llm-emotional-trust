import pandas as pd 

Combined = pd.read_csv("data/Combined.csv")
"""
Submission id
PID
Status
Started at
Completed at
Reviewed at
Archived at
Time taken
Completion code
Total approvals
Gender
Ethnicity
Age
Sex
Ethnicity simplified
Country of birth
Country of residence
Nationality
Language
Student status
Employment status
Condition
ID
Start time
Completion time
Email
Name
Last modified time
Education
AI Knowledge
Age.1
AI Deceptive
AI Dishonest
AI Suspicious
AI Wary
AI Harm
AI Confident
AI Security
AI Trustworthy
AI Reliable
AI Trust
AI systems are 1
AI systems are 2
AI systems are 3
AI systems are 4
AI systems are 5
AI systems are 6
AI systems are 7
AI systems are 8
AI systems are 9
AI Harm Post 1
AI Security Post
AI Suspicious Post
AI Wary/Deceptive Post
AI Harm Post 2
AI Confident Post
AI Dishonest Post
AI Trust Post
AI Reliable Post
AI Trustworthy Post
AI systems are Post 1
AI systems are Post 2
AI systems are Post 3
AI systems are Post 4
AI systems are Post 5
AI systems are Post 6
AI systems are Post 7
AI systems are Post 8
AI systems are Post 9
AI Interaction Feeling
Need Model Understanding
Job Screening Feeling
Age.2
AI Wary Post
AI Deceptive Post
AI Trust Post 2
AI Definition Feeling
Is White
Explanation Comment
Submission id (Demographics)
PID (Demographics)
Status (Demographics)
Started at (Demographics)
Completed at (Demographics)
Reviewed at (Demographics)
Archived at (Demographics)
Time taken (Demographics)
Completion code (Demographics)
Total approvals (Demographics)
Gender (Demographics)
Ethnicity (Demographics)
Age (Demographics)
Sex (Demographics)
Ethnicity simplified (Demographics)
Country of birth (Demographics)
Country of residence (Demographics)
Nationality (Demographics)
Language (Demographics)
Student status (Demographics)
Employment status (Demographics)
Condition (Demographics)
Responses
Submissions
"""

print(Combined['AI systems are Post 1'])