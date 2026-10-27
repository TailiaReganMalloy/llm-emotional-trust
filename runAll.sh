#!/usr/bin/env bash
set -euo pipefail

echo "Applying analysis exclusion criteria"
python Dataset/metrics.py
python Dataset/apply_exclusions.py
python Methodology/Participants/demographics.py

# Only the scripts that produce the figures kept in Figures/.
for script in Results/hypothesis{1..6}/hypothesis*_stats.py; do
	echo "Running $script"
	python "$script"
done

for script in Results/hypothesis{1..6}/hypothesis*_plot.py \
	Results/hypothesis2a/hypothesis2a.py \
	Results/qualitative/qualitative.py \
	Results/qualitative/qualitative1.py; do
	echo "Running $script"
	python "$script"
done
