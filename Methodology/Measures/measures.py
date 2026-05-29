"""Canonical trust-score calculations for methodology and downstream analyses.

This module mirrors the analytical and emotional trust change formulas used in
the analysis scripts (for example, Results/qualitative/ic2.py).
"""

from __future__ import annotations

import pandas as pd


ANALYTICAL_MAX = 10.0
EMOTIONAL_MAX = 9.0


def _to_numeric(series: pd.Series | None) -> pd.Series:
	"""Convert a pandas Series to numeric while preserving missing values."""
	if series is None:
		return pd.Series(dtype=float)
	return pd.to_numeric(series, errors="coerce")


def add_trust_scores(df: pd.DataFrame) -> pd.DataFrame:
	"""Add normalized pre/post trust scores and pre-post differences.

	Required source columns for direct calculation:
	- Total Analytical Trust
	- Total Analytical Trust Post
	- Total Emotional Trust
	- Total Emotional Trust Post

	If source totals are missing, this function falls back to:
	- Analytical Trust Difference
	- Emotional Trust Difference

	Returns a copy of ``df`` with new columns appended.
	"""
	work = df.copy()

	analytical_pre = _to_numeric(work.get("Total Analytical Trust")) / ANALYTICAL_MAX
	analytical_post = _to_numeric(work.get("Total Analytical Trust Post")) / ANALYTICAL_MAX
	emotional_pre = _to_numeric(work.get("Total Emotional Trust")) / EMOTIONAL_MAX
	emotional_post = _to_numeric(work.get("Total Emotional Trust Post")) / EMOTIONAL_MAX

	work["analytical_pre_01"] = analytical_pre
	work["analytical_post_01"] = analytical_post
	work["emotional_pre_01"] = emotional_pre
	work["emotional_post_01"] = emotional_post

	if {"Total Analytical Trust", "Total Analytical Trust Post"}.issubset(work.columns):
		work["analytical_change_01"] = analytical_post - analytical_pre
	else:
		work["analytical_change_01"] = _to_numeric(work.get("Analytical Trust Difference"))

	if {"Total Emotional Trust", "Total Emotional Trust Post"}.issubset(work.columns):
		work["emotional_change_01"] = emotional_post - emotional_pre
	else:
		work["emotional_change_01"] = _to_numeric(work.get("Emotional Trust Difference"))

	return work

