# LLM Emotional Trust

This repository contains merged survey + demographics data for an AI trust/emotion study.

Main dataset:
- data/Combined.csv

## Combined.csv column dictionary

Notes:
- Combined.csv currently has 103 columns.
- The header contains repeated names (for example Age appears multiple times).
- The index column below is the safest way to identify a specific field unambiguously.

| # | Column name | Description |
|---|---|---|
| 1 | Submission id | Prolific submission ID from the primary merged record. |
| 2 | PID | Prolific participant ID used for joins. |
| 3 | Status | Prolific submission status (for example APPROVED). |
| 4 | Started at | Prolific study start timestamp. |
| 5 | Completed at | Prolific study completion timestamp. |
| 6 | Reviewed at | Prolific review timestamp. |
| 7 | Archived at | Prolific archive timestamp (if archived). |
| 8 | Time taken | Time taken in Prolific export. |
| 9 | Completion code | Completion code used for payout tracking. |
| 10 | Total approvals | Participant total approval count on Prolific. |
| 11 | Gender | Participant gender from Prolific demographics export. |
| 12 | Ethnicity | Participant ethnicity from Prolific demographics export. |
| 13 | Age | Age from the first demographics block. |
| 14 | Sex | Participant sex from demographics export. |
| 15 | Ethnicity simplified | Simplified ethnicity category. |
| 16 | Country of birth | Country of birth. |
| 17 | Country of residence | Country of residence. |
| 18 | Nationality | Nationality. |
| 19 | Language | Participant language. |
| 20 | Student status | Participant student status. |
| 21 | Employment status | Participant employment status. |
| 22 | Condition | Study condition label (Interactive or Text). |
| 23 | ID | Questionnaire platform response ID. |
| 24 | Start time | Questionnaire start timestamp. |
| 25 | Completion time | Questionnaire completion timestamp. |
| 26 | Email | Questionnaire email field (often anonymous or blank). |
| 27 | Name | Questionnaire name field (often blank). |
| 28 | Last modified time | Questionnaire last-modified timestamp. |
| 29 | Education | Self-reported education level. |
| 30 | AI Knowledge | Self-reported AI knowledge level. |
| 31 | Age | Age from questionnaire block (duplicate header name). |
| 32 | AI Deceptive | Pre item: AI systems are deceptive. |
| 33 | AI Dishonest | Pre item: AI systems behave dishonestly. |
| 34 | AI Suspicious | Pre item: participant is suspicious of AI intent/actions/outputs. |
| 35 | AI Wary | Pre item: participant is wary of AI systems. |
| 36 | AI Harm | Pre item: AI actions have harmful/injurious outcomes. |
| 37 | AI Confident | Pre item: confidence in AI systems. |
| 38 | AI Security | Pre item: AI systems provide security. |
| 39 | AI Trustworthy | Pre item: AI systems are trustworthy. |
| 40 | AI Reliable | Pre item: AI systems are reliable. |
| 41 | AI Trust | Pre item: participant can trust AI systems. |
| 42 | AI systems are 1 | Pre emotional adjective item 1. |
| 43 | AI systems are 2 | Pre emotional adjective item 2. |
| 44 | AI systems are 3 | Pre emotional adjective item 3. |
| 45 | AI systems are 4 | Pre emotional adjective item 4. |
| 46 | AI systems are 5 | Pre emotional adjective item 5. |
| 47 | AI systems are 6 | Pre emotional adjective item 6. |
| 48 | AI systems are 7 | Pre emotional adjective item 7. |
| 49 | AI systems are 8 | Pre emotional adjective item 8. |
| 50 | AI systems are 9 | Pre emotional adjective item 9. |
| 51 | AI Harm Post 1 | Post trust item variant tied to harm wording. |
| 52 | AI Security Post | Post item: AI systems provide security. |
| 53 | AI Suspicious Post | Post item: suspiciousness toward AI. |
| 54 | AI Wary/Deceptive Post | Post item variant combining wary/deceptive wording in source form. |
| 55 | AI Harm Post 2 | Post trust item second harm variant. |
| 56 | AI Confident Post | Post item: confidence in AI systems. |
| 57 | AI Dishonest Post | Post item: AI systems behave dishonestly. |
| 58 | AI Trust Post | Post item: participant can trust AI systems. |
| 59 | AI Reliable Post | Post item: AI systems are reliable. |
| 60 | AI Trustworthy Post | Post item: AI systems are trustworthy. |
| 61 | AI systems are Post 1 | Post emotional adjective item 1. |
| 62 | AI systems are Post 2 | Post emotional adjective item 2. |
| 63 | AI systems are Post 3 | Post emotional adjective item 3. |
| 64 | AI systems are Post 4 | Post emotional adjective item 4. |
| 65 | AI systems are Post 5 | Post emotional adjective item 5. |
| 66 | AI systems are Post 6 | Post emotional adjective item 6. |
| 67 | AI systems are Post 7 | Post emotional adjective item 7. |
| 68 | AI systems are Post 8 | Post emotional adjective item 8. |
| 69 | AI systems are Post 9 | Post emotional adjective item 9. |
| 70 | AI Interaction Feeling | Feeling after interacting with the AI explanation/tool. |
| 71 | Need Model Understanding | Whether understanding the model is needed for trust. |
| 72 | Job Screening Feeling | Feeling about AI use in job screening. |
| 73 | Age | Third Age column from later questionnaire block (duplicate header name). |
| 74 | AI Wary Post | Post item: participant is wary of AI systems. |
| 75 | AI Deceptive Post | Post item: AI systems are deceptive. |
| 76 | AI Trust Post 2 | Alternate post trust item field (duplicate concept). |
| 77 | AI Definition Feeling | Feeling after reading AI definition text. |
| 78 | Is White | Derived binary indicator from ethnicity simplified (1 if White else 0). |
| 79 | Explanation Comment | Free-text comment after explanation interaction. |
| 80 | Submission id (Demographics) | Submission ID from secondary demographics lookup table. |
| 81 | PID (Demographics) | Participant ID from secondary demographics lookup table. |
| 82 | Status (Demographics) | Status from secondary demographics lookup table. |
| 83 | Started at (Demographics) | Start timestamp from secondary demographics lookup table. |
| 84 | Completed at (Demographics) | Completion timestamp from secondary demographics lookup table. |
| 85 | Reviewed at (Demographics) | Review timestamp from secondary demographics lookup table. |
| 86 | Archived at (Demographics) | Archive timestamp from secondary demographics lookup table. |
| 87 | Time taken (Demographics) | Time taken from secondary demographics lookup table. |
| 88 | Completion code (Demographics) | Completion code from secondary demographics lookup table. |
| 89 | Total approvals (Demographics) | Total approvals from secondary demographics lookup table. |
| 90 | Gender (Demographics) | Gender from secondary demographics lookup table. |
| 91 | Ethnicity (Demographics) | Ethnicity from secondary demographics lookup table. |
| 92 | Age (Demographics) | Age from secondary demographics lookup table. |
| 93 | Sex (Demographics) | Sex from secondary demographics lookup table. |
| 94 | Ethnicity simplified (Demographics) | Simplified ethnicity from secondary demographics lookup table. |
| 95 | Country of birth (Demographics) | Country of birth from secondary demographics lookup table. |
| 96 | Country of residence (Demographics) | Country of residence from secondary demographics lookup table. |
| 97 | Nationality (Demographics) | Nationality from secondary demographics lookup table. |
| 98 | Language (Demographics) | Language from secondary demographics lookup table. |
| 99 | Student status (Demographics) | Student status from secondary demographics lookup table. |
| 100 | Employment status (Demographics) | Employment status from secondary demographics lookup table. |
| 101 | Condition (Demographics) | Condition from secondary demographics lookup table. |
| 102 | Responses | List-like payload of matched response objects per participant. |
| 103 | Submissions | List-like payload of matched submission/edit objects per participant. |

## Simple seaborn plot example

The snippet below creates a simple bar plot of mean pre-study AI Trust by condition.

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load data
    df = pd.read_csv("data/Combined.csv")

    # Convert Likert labels to numeric scores for plotting
    likert_map = {
        "Strongly Disagree": 1,
        "Disagree": 2,
        "Neither Agree nor Disagree": 3,
        "Agree": 4,
        "Strongly Agree": 5,
    }

    # AI Trust is the pre item at header position #41
    df["AI Trust Score"] = df["AI Trust"].map(likert_map)

    # Keep only rows with valid trust scores and condition labels
    plot_df = df.dropna(subset=["AI Trust Score", "Condition"]).copy()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_df,
        x="Condition",
        y="AI Trust Score",
        estimator="mean",
        errorbar=("ci", 95),
        palette="Set2",
    )

    ax.set_title("Mean Pre-study AI Trust by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean AI Trust (1-5)")
    plt.tight_layout()
    plt.show()

Optional extension:
- If you want, I can also add a second example that compares pre vs post trust in one figure (long-format reshape + seaborn pointplot).
