from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


combined = pd.read_csv("./data/combined.csv")
combined = combined[combined["Time taken"] < 2000]

time_taken = pd.to_numeric(combined["Time taken"], errors="coerce")

initial_mean = float(time_taken.mean())
initial_sd = float(time_taken.std(ddof=1))

lower_bound = initial_mean - (2 * initial_sd)
upper_bound = initial_mean + (2 * initial_sd)

# Keep only rows with valid time values within ±2 SD of the original mean.
keep_mask = time_taken.notna() & time_taken.between(lower_bound, upper_bound)
filtered = combined.loc[keep_mask].copy()
filtered_time_taken = pd.to_numeric(filtered["Time taken"], errors="coerce")

filtered_mean = float(filtered_time_taken.mean())
filtered_sd = float(filtered_time_taken.std(ddof=1))

time_taken = pd.to_numeric(combined["Time taken"], errors="coerce")

plt.figure(figsize=(10, 6))
plt.hist(time_taken, bins=30, color="#F58518", edgecolor="white")
plt.title("Distribution of Time Taken")
plt.xlabel("Time Taken")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print(f"Original rows: {len(combined)}")
print(f"Filtered rows: {len(filtered)}")
print(f"Removed rows: {len(combined) - len(filtered)}")
print(f"Original Time taken mean: {initial_mean:.4f}")
print(f"Original Time taken SD: {initial_sd:.4f}")
print(f"Outlier bounds (±2 SD): [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"Filtered Time taken mean: {filtered_mean:.4f}")
print(f"Filtered Time taken SD: {filtered_sd:.4f}")

output_path = Path("./data/combined.csv")
filtered.to_csv(output_path, index=False)
print(f"Saved filtered dataset to: {output_path}")

