# Sentiment Script Guide (sentiment1, sentiment2, sentiment3)

This document explains the three sentiment plotting scripts in this folder:

- `sentiment.py` -> writes `Figures/sentiment1.png`
- `sentiment2.py` -> writes `Figures/sentiment2.png`
- `sentiment3.py` -> writes `Figures/sentiment3.png`

All three scripts read `Dataset/Metrics.csv` and use the same participant-level preprocessing from `build_analysis_frame(...)` in `sentiment.py`.

## 1. How values are calculated

### Sentiment values

Each participant has up to 3 text responses:

- `AI Interaction Feeling`
- `AI Definition Feeling`
- `Explanation Comment`

For each text field:

1. Text is normalized with `str(value).strip()`.
2. Sentiment is scored with VADER compound polarity (`SentimentIntensityAnalyzer().polarity_scores(text)["compound"]`).
3. If text is empty, the score is `NaN`.

Participant-level sentiment is then:

- `sentiment_mean = mean([interaction_feeling_sentiment, definition_feeling_sentiment, explanation_comment_sentiment], skipna=True)`

So each `sentiment_mean` is roughly in `[-1, 1]`.

### Group labels

Participants are assigned to:

- `WEIRD` if country is in the western-country set and language starts with `english`
- otherwise `NORMAL` (displayed as `non-WEIRD` in plots)

### Trust-change outcomes used in the three scripts

Only these two outcomes are plotted now:

1. `overall_change`
2. `analytical_minus_emotional_change`

They are computed as:

- `analytical_pre = Total Analytical Trust / 10`
- `analytical_post = Total Analytical Trust Post / 10`
- `emotional_pre = Total Emotional Trust / 9`
- `emotional_post = Total Emotional Trust Post / 9`

Raw deltas:

- `analytical_change_raw = analytical_post - analytical_pre`
- `emotional_change_raw = emotional_post - emotional_pre`
- `overall_change_raw = (analytical_change_raw + emotional_change_raw) / 2`

Then z-scored:

- `analytical_change = zscore(analytical_change_raw)`
- `emotional_change = zscore(emotional_change_raw)`
- `overall_change = zscore(overall_change_raw)`

Finally:

- `analytical_minus_emotional_change = analytical_change - emotional_change`

### Binning used in sentiment1/sentiment2 panels

For plotting, points are grouped into sentiment bins of width `0.01`.
For each bin, the scripts compute mean and variance of x and y and plot error bars from variance.
Regression lines in `sentiment1.png` and `sentiment2.png` are fit to these binned means (not raw individual points).

## 2. What each image shows

## `Figures/sentiment1.png` (from `sentiment.py`)

Two panels:

1. Sentiment mean vs normalized overall trust change
2. Sentiment mean vs analytical-minus-emotional normalized change

Plot elements:

- all participants combined
- binned means with variance error bars
- one overall linear fit line per panel
- slope and p-value annotation per panel

Use this figure as the overall association view (no group split).

## `Figures/sentiment2.png` (from `sentiment2.py`)

Same two panels/outcomes as sentiment1, but with group split overlays:

- overall line
- WEIRD line
- non-WEIRD line

Plot elements:

- binned means with variance error bars
- three regression summaries (overall, WEIRD, non-WEIRD) per panel

Use this figure to visually compare group-specific trend directions/slopes.

## `Figures/sentiment3.png` (from `sentiment3.py`)

Two panels with explicit interaction tests, one per outcome:

1. Outcome = `overall_change`
2. Outcome = `analytical_minus_emotional_change`

For each panel, script fits a linear model:

- `outcome ~ sentiment_centered + group_bin + sentiment_centered:group_bin`
- where `group_bin = 1` for WEIRD, `0` for non-WEIRD

Panel annotation reports:

- interaction coefficient and p-value
- non-WEIRD simple slope and p-value
- WEIRD simple slope and p-value
- whether opposite-direction pattern is supported (`YES` requires non-WEIRD slope > 0, WEIRD slope < 0, and interaction p < .05)

This is the inferential figure for testing whether group differences in slope are statistically supported.

## 3. Current run summary (latest execution)

From the most recent run of `sentiment3.py`:

- `overall_change`: interaction `b = -0.5709`, `p = 0.1975`
- `analytical_minus_emotional_change`: interaction `b = -0.0189`, `p = 0.9729`

Interpretation:

- In this run, neither interaction term is significant at `alpha = 0.05`.
- So visual slope differences should be treated as descriptive unless further robustness checks support them.

## 4. How to run

From repository root:

```bash
source .venv/bin/activate
python Results/qualitative/sentiment.py
python Results/qualitative/sentiment2.py
python Results/qualitative/sentiment3.py
```
