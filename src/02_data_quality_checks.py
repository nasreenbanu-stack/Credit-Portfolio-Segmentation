
from pathlib import Path
import pandas as pd

# Locate project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loans_granting_model_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded successfully!")

# 1. Check issue date range
print("\n--- ISSUE DATE ---")
issue_dates = pd.to_datetime(
    df["issue_d"],
    format="%b-%Y",
    errors="coerce"
)

print("Earliest date:", issue_dates.min())
print("Latest date:", issue_dates.max())
print("Invalid dates:", issue_dates.isna().sum())

# 2. Investigate DTI
print("\n--- DTI INVESTIGATION ---")
print(df["dti_n"].describe())

print("DTI equal to 999:", (df["dti_n"] == 999).sum())
print("DTI greater than 100:", (df["dti_n"] > 100).sum())

# 3. Investigate revenue
print("\n--- REVENUE INVESTIGATION ---")
print(df["revenue"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
))

print("Revenue above 1 million:", (df["revenue"] > 1000000).sum())

# 4. Investigate experience_c
print("\n--- EXPERIENCE INVESTIGATION ---")
print(df["experience_c"].value_counts(dropna=False))

# 5. Investigate loan amount
print("\n--- LOAN AMOUNT INVESTIGATION ---")
print(df["loan_amnt"].describe())

# 6. Investigate FICO score
print("\n--- FICO INVESTIGATION ---")
print(df["fico_n"].describe())

# 7. Investigate categorical columns
categorical_columns = [
    "emp_length",
    "purpose",
    "home_ownership_n"
]

print("\n--- CATEGORICAL VALUE COUNTS ---")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

print("\nData quality investigation completed!")
