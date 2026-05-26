# Exploratory Qualitative Analysis

This folder contains exploratory sentiment-based analyses of participant open responses and how they relate to:

- condition (Interactive vs Text)
- WEIRD vs NORMAL group
- trust changes (overall, analytical, emotional, analytical-minus-emotional)

## Run

From repository root:

```bash
/Users/tailia.malloy/Documents/Code/llm-emotional-trust/.venv/bin/python exploratory_qualitative_analysis/run_exploratory_qualitative_analysis.py
```

## Outputs

Generated under `exploratory_qualitative_analysis/results/`:

- `scored_participant_text.csv`: participant-level text and sentiment features
- `primary_tests.csv`: primary inferential tests
- `follow_up_tests.csv`: additional tests automatically triggered by significant primary findings
- `sentiment_by_group_condition.png`: grouped bar chart of sentiment by group and condition
- `sentiment_vs_trust_changes.png`: scatter panels of sentiment vs trust-change outcomes
- `report.md`: narrative summary and evaluation of findings
- `self_prompt_for_next_round.md`: automatically generated self-prompt for further analyses based on significance
- `sentiment.txt`: LaTeX-ready summary describing both sentiment figures and significance status (stars vs ns)

## Sentiment Scoring

The script uses VADER sentiment if `vaderSentiment` is installed.
If unavailable, it falls back to an internal lexicon-based polarity scorer.
