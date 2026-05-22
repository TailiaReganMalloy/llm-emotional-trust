# Exploratory Qualitative Sentiment Analysis

## Setup
- Sentiment backend: fallback_lexicon
- Participants analyzed: 504
- Text sources: AI Interaction Feeling, AI Definition Feeling, Explanation Comment
- Grouping: WEIRD vs NORMAL (Western country + English language)
- Outcomes: overall, analytical, emotional, and analytical-minus-emotional trust change

## Primary Test Counts
- Number of primary tests: 28
- Number of significant primary tests (p < .05): 7

## Follow-Up Test Counts
- Number of triggered follow-up tests: 16
- Number of significant follow-up tests (p < .05): 8

## Significant Findings
- primary | welch | definition_feeling_sentiment | group_weird_vs_normal | subset=all | p=0.0098
- primary | pearson | analytical_change | sentiment_mean_association | subset=all | p=0.0059
- primary | spearman | analytical_change | sentiment_mean_association | subset=all | p=0.0171
- primary | pearson | emotional_change | sentiment_mean_association | subset=all | p=0.0109
- primary | spearman | emotional_change | sentiment_mean_association | subset=all | p=0.0081
- primary | pearson | analytical_minus_emotional_change | sentiment_mean_association | subset=all | p=0.0001
- primary | spearman | analytical_minus_emotional_change | sentiment_mean_association | subset=all | p=0.0002
- follow_up | welch | definition_feeling_sentiment | group_weird_vs_normal | subset=condition=Text | p=0.0320
- follow_up | pearson | analytical_change | sentiment_mean_association | subset=condition=Text | p=0.0118
- follow_up | pearson | analytical_change | sentiment_mean_association | subset=group=NORMAL | p=0.0152
- follow_up | pearson | emotional_change | sentiment_mean_association | subset=condition=Text | p=0.0034
- follow_up | pearson | emotional_change | sentiment_mean_association | subset=group=WEIRD | p=0.0476
- follow_up | pearson | analytical_minus_emotional_change | sentiment_mean_association | subset=condition=Text | p=0.0001
- follow_up | pearson | analytical_minus_emotional_change | sentiment_mean_association | subset=group=WEIRD | p=0.0179
- follow_up | pearson | analytical_minus_emotional_change | sentiment_mean_association | subset=group=NORMAL | p=0.0016

## Evaluation
- This module is exploratory and not pre-registered; interpret p-values as hypothesis-generating.
- Sentiment is based on short open responses, so lexical noise and sparse text can attenuate effects.
- Effect sizes and directionality should be prioritized over binary significance in follow-up work.

## Self-Prompted Next Analyses
- Compare thematic language between WEIRD and NORMAL groups using keyword extraction and manual coding on high-impact terms, then test whether theme prevalence remains after conditioning on Condition.
- For significant sentiment-trust links (analytical_change, analytical_minus_emotional_change, emotional_change), run multivariable models adjusting for condition and group, then evaluate whether sentiment still predicts trust change.
- Check non-linear effects by adding quadratic sentiment terms and compare model fit with AIC or adjusted R-squared.
- Validate exploratory findings with false-discovery-rate correction and report which conclusions remain stable.

