#!/usr/bin/env bash
set -euo pipefail

find Results -type f -name "*.py" | sort | while IFS= read -r script; do
	echo "Running $script"
	python "$script"
done