#!/usr/bin/env bash
set -euo pipefail

echo "Applying analysis exclusion criteria"
python Dataset/metrics.py
python Dataset/apply_exclusions.py
python Methodology/Participants/demographics.py

find Results -type f -name "*.py" | sort | while IFS= read -r script; do
	echo "Running $script"
	python "$script"
done