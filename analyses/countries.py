import pandas as pd

combined = pd.read_csv("./data/combined.csv")

if "Status" in combined.columns:
	combined = combined[combined["Status"].astype(str).str.upper() == "APPROVED"].copy()

if "Condition" not in combined.columns:
	raise ValueError("Expected 'Condition' column in dataset.")

if "Nationality" not in combined.columns:
	raise ValueError("Expected 'Nationality' column in dataset.")

TARGET_TOTAL_PER_CONDITION = 200
PRIORITY_NATIONALITIES = [
	"South Africa",
	"India",
	"Brazil",
	"United States",
	"United Kingdom",
]

normalized = combined.assign(
	Nationality=combined["Nationality"].fillna("Missing").astype(str).str.strip().replace("", "Missing")
)

counts = (
	normalized
	.groupby(["Condition", "Nationality"], dropna=False)
	.size()
	.reset_index(name="count")
)

conditions = counts["Condition"].dropna().unique().tolist()
if len(conditions) != 2:
	raise ValueError(f"Expected exactly 2 conditions, found {len(conditions)}: {conditions}")

if "Interactive" in conditions and "Static" in conditions:
	ordered_conditions = ["Interactive", "Static"]
else:
	ordered_conditions = sorted(conditions)

counts = counts[counts["Condition"].isin(ordered_conditions)].copy()

# Build one shared target nationality mix (for 160 participants).
# Priority countries are forced to the same target in both conditions.
pooled = counts.groupby("Nationality", as_index=False)["count"].sum()
pooled = pooled.rename(columns={"count": "pooled_count"})
pooled_total = int(pooled["pooled_count"].sum())

if pooled_total == 0:
	raise ValueError("No nationality counts found to compute balance targets.")

num_nationalities = len(pooled)
if num_nationalities == 0:
	raise ValueError("No nationalities found to compute balance targets.")

count_wide = counts.pivot_table(
	index="Nationality",
	columns="Condition",
	values="count",
	aggfunc="sum",
	fill_value=0,
)
for cond in ordered_conditions:
	if cond not in count_wide.columns:
		count_wide[cond] = 0

priority_targets: dict[str, int] = {}
for nationality in PRIORITY_NATIONALITIES:
	if nationality in count_wide.index:
		priority_targets[nationality] = int(count_wide.loc[nationality, ordered_conditions].max())

reserved_target_total = int(sum(priority_targets.values()))
remaining_total = TARGET_TOTAL_PER_CONDITION - reserved_target_total
if remaining_total < 0:
	raise ValueError(
		"Priority nationality targets exceed target total per condition. "
		f"Reserved={reserved_target_total}, target={TARGET_TOTAL_PER_CONDITION}."
	)

non_priority = pooled[~pooled["Nationality"].isin(priority_targets.keys())].copy()
non_priority_total = int(non_priority["pooled_count"].sum())

if len(non_priority) == 0 and remaining_total > 0:
	raise ValueError("No non-priority nationalities available for remaining target allocation.")

non_priority["target_floor"] = 0
non_priority["fraction"] = 0.0
if remaining_total > 0:
	non_priority["target_raw"] = non_priority["pooled_count"] / non_priority_total * remaining_total
	non_priority["target_floor"] = non_priority["target_raw"].astype(int)
	non_priority["fraction"] = non_priority["target_raw"] - non_priority["target_floor"]

	remainder = remaining_total - int(non_priority["target_floor"].sum())
	non_priority = non_priority.sort_values(["fraction", "Nationality"], ascending=[False, True]).reset_index(drop=True)
	non_priority["add_one"] = 0
	if remainder > 0:
		non_priority.loc[: remainder - 1, "add_one"] = 1
	non_priority["desired_target_count"] = non_priority["target_floor"] + non_priority["add_one"]
else:
	non_priority["desired_target_count"] = 0

priority_df = pd.DataFrame(
	{"Nationality": list(priority_targets.keys()), "desired_target_count": list(priority_targets.values())}
)

targets = pd.concat(
	[priority_df, non_priority[["Nationality", "desired_target_count"]]],
	ignore_index=True,
)

# Ensure every nationality appears for both conditions (missing counts become 0).
all_pairs = pd.MultiIndex.from_product(
	[ordered_conditions, targets["Nationality"].tolist()],
	names=["Condition", "Nationality"],
).to_frame(index=False)

counts = all_pairs.merge(counts, on=["Condition", "Nationality"], how="left")
counts["count"] = counts["count"].fillna(0).astype(int)
counts = counts.merge(targets, on="Nationality", how="left")

# Allocate additions so each condition reaches exactly TARGET_TOTAL_PER_CONDITION.
counts["deficit"] = (counts["desired_target_count"] - counts["count"]).clip(lower=0)
counts["is_priority"] = counts["Nationality"].isin(PRIORITY_NATIONALITIES)
counts["balance count"] = 0

for condition in ordered_conditions:
	mask = counts["Condition"] == condition
	current_total = int(counts.loc[mask, "count"].sum())
	need_total = max(TARGET_TOTAL_PER_CONDITION - current_total, 0)
	if need_total == 0:
		continue

	cond_rows = counts.loc[mask].copy()
	add = pd.Series(0, index=cond_rows.index, dtype=int)

	priority_rows = cond_rows[cond_rows["is_priority"]]
	mandatory_priority = priority_rows["deficit"].astype(int)
	mandatory_sum = int(mandatory_priority.sum())

	if mandatory_sum > need_total:
		# Rare fallback: scale priority additions if they exceed available slots.
		raw_priority = mandatory_priority / mandatory_sum * need_total
		floor_priority = raw_priority.astype(int)
		remainder_priority = int(need_total - floor_priority.sum())
		order_priority = (raw_priority - floor_priority).sort_values(ascending=False).index
		add.loc[floor_priority.index] = floor_priority
		if remainder_priority > 0:
			add.loc[order_priority[:remainder_priority]] += 1
		counts.loc[mask, "balance count"] = add.astype(int)
		continue

	add.loc[mandatory_priority.index] = mandatory_priority
	remaining_need = need_total - mandatory_sum
	if remaining_need == 0:
		counts.loc[mask, "balance count"] = add.astype(int)
		continue

	non_priority_rows = cond_rows[~cond_rows["is_priority"]]
	non_priority_deficits = non_priority_rows["deficit"]
	non_priority_deficit_total = float(non_priority_deficits.sum())

	if non_priority_deficit_total > 0:
		raw = non_priority_deficits / non_priority_deficit_total * remaining_need
		floor_vals = raw.astype(int)
		remainder = int(remaining_need - floor_vals.sum())
		order = (raw - floor_vals).sort_values(ascending=False).index
		add.loc[floor_vals.index] += floor_vals
		if remainder > 0:
			add.loc[order[:remainder]] += 1
	else:
		# If no deficits remain, spread remaining slots evenly across non-priority rows.
		if len(non_priority_rows) > 0:
			base_add = remaining_need // len(non_priority_rows)
			remainder = remaining_need % len(non_priority_rows)
			add.loc[non_priority_rows.index] += base_add
			if remainder > 0:
				ordered_idx = non_priority_rows.sort_values("Nationality").index
				add.loc[ordered_idx[:remainder]] += 1

	counts.loc[mask, "balance count"] = add.astype(int)

counts["target_count"] = counts["count"] + counts["balance count"]
counts = counts.drop(columns=["desired_target_count", "deficit", "is_priority"])

counts = counts.sort_values(["Condition", "target_count", "Nationality"], ascending=[True, False, True])

print("Nationality counts by condition:")
print(counts.to_string(index=False))

totals = counts.groupby("Condition", as_index=False)[["count", "balance count"]].sum()
totals["target_total"] = totals["count"] + totals["balance count"]
print("\nCondition totals:")
print(totals.to_string(index=False))

print("Target count sum: ", counts['target_count'].sum())

counts.to_csv("Balanced Nationalities.csv")

