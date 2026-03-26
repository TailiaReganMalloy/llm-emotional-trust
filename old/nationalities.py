import pandas as pd
from scipy.stats import chi2_contingency


Cleaned = pd.read_csv("./data/Cleaned.csv")

Cleaned = Cleaned[Cleaned['Nationality'] != 'CONSENT_REVOKED']

Cleaned.to_csv("./data/Cleaned.csv")

required_columns = ["Condition", "Nationality"]
missing_columns = [col for col in required_columns if col not in Cleaned.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Keep rows with both condition and nationality present.
balancing_df = Cleaned[required_columns].dropna().copy()
balancing_df["Condition"] = balancing_df["Condition"].astype(str).str.strip()
balancing_df["Nationality"] = balancing_df["Nationality"].astype(str).str.strip()

# Drop empty-string categories.
balancing_df = balancing_df[
    (balancing_df["Condition"] != "") & (balancing_df["Nationality"] != "")
]

contingency = pd.crosstab(balancing_df["Nationality"], balancing_df["Condition"])

chi2, p_value, dof, expected = chi2_contingency(contingency)

n = contingency.to_numpy().sum()
min_dim = min(contingency.shape) - 1
cramers_v = (chi2 / (n * min_dim)) ** 0.5 if min_dim > 0 and n > 0 else float("nan")

alpha = 0.05

print("--- Nationality Balance Across Conditions ---")
print(f"Rows included in test: {len(balancing_df)}")
print(f"Unique nationalities: {contingency.shape[0]}")
print(f"Condition groups: {list(contingency.columns)}")
print("\nContingency table (Nationality x Condition):")
print(contingency)

print("\nChi-square test of independence")
print(f"chi2 statistic: {chi2:.4f}")
print(f"degrees of freedom: {dof}")
print(f"p-value: {p_value:.6f}")
print(f"Cramer's V: {cramers_v:.4f}")

if p_value < alpha:
    print("Decision: Reject H0 at alpha=0.05 (nationality distribution differs by condition).")
else:
    print("Decision: Fail to reject H0 at alpha=0.05 (no evidence nationality distribution differs by condition).")
