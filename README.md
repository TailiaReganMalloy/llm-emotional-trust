# Analysis Outputs

This folder contains the condition-based pre/post analysis results produced by `analysis.py`.

## Files in this folder

- `condition_delta_stats.csv` — statistical test results for each pre/post survey item.
- `condition_delta_stats.png` — bar chart of test statistics for non-`AI systems are` items.
- `ai_systems_are_delta_stats.png` — bar chart of test statistics for the nine `AI systems are` bipolar items.

## How these files were made

From the project root:

```bash
python analysis.py
```

`analysis.py` does the following:

1. Loads the grouped dataset from `./data/grouped.pkl`.
2. Maps each pre item to its corresponding post item.
3. Normalizes response values:
   - Likert responses are mapped to numeric values (`Strongly Disagree=-2`, `Disagree=-1`, `Agree=1`, `Strongly Agree=2`).
   - `AI systems are` bipolar responses are mapped to 0/1 using the item-specific word pairs (e.g., `Apathetic=0`, `Empathetic=1`).
4. Computes per-participant change (`delta = post - pre`) for each item.
5. Compares deltas across conditions:
   - Two conditions: independent Student t-test (`scipy.stats.ttest_ind`).
   - More than two conditions: one-way ANOVA (`scipy.stats.f_oneway`).
6. Saves test outputs to `condition_delta_stats.csv`.
7. Creates and saves the two PNG visualizations in this folder.

## What the plots show

### 1) Condition effect across non-bipolar items

![Condition effect on pre/post change](./condition_delta_stats.png)

- Each bar is one survey item (excluding `AI systems are` items).
- Bar height = test statistic for condition differences in pre/post change.
- Labels above/below bars show p-values and direction (`Condition A > Condition B`), when available.
- This figure gives an item-level overview of where condition differences are strongest.

### 2) Condition effect across `AI systems are` items

![Condition effect on AI systems are items](./ai_systems_are_delta_stats.png)

- Each bar corresponds to one of the 9 bipolar `AI systems are` items.
- Bar height = test statistic for the condition difference in pre/post change.
- Labels show p-values and direction when applicable.
- The top and bottom annotations on each item show the positive/negative poles of that bipolar pair (e.g., `Empathetic` vs `Apathetic`).
- This figure isolates emotional/personality-style evaluations of AI systems.

## Interpretation notes

- Lower p-values indicate stronger evidence of a condition effect for that item.
- Test statistic sign and direction text indicate which condition had larger average change.
- These are condition-comparison statistics on change scores (not raw pre or post scores alone).
