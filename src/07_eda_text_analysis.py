
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "loans_cleaned.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n--- DTI PERCENTILES ---")
print(df["dti_n"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\n--- TOP 10 LOAN PURPOSES ---")
print(df["purpose"].value_counts().head(10))

print("\n--- HOME OWNERSHIP DISTRIBUTION ---")
print(df["home_ownership_n"].value_counts())

print("\n--- DEFAULT LABEL DISTRIBUTION ---")
print(df["Default"].value_counts())

print("\n--- DEFAULT LABEL PERCENTAGES ---")
print(df["Default"].value_counts(normalize=True) * 100)
