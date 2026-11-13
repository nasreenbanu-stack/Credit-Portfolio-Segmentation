
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loans_granting_model_dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded successfully!")

# 1. Exact count of DTI = 999
print("\n--- DTI = 999 ---")

dti_999 = df.loc[df["dti_n"] == 999]

print("Number of records:", len(dti_999))
print(dti_999[["id", "revenue", "loan_amnt", "fico_n", "Default"]].head(10))

# 2. DTI greater than 100, excluding 999
print("\n--- DTI > 100 EXCLUDING 999 ---")

high_dti = df.loc[
    (df["dti_n"] > 100) & (df["dti_n"] != 999)
]

print("Number of records:", len(high_dti))
print("Minimum DTI:", high_dti["dti_n"].min())
print("Maximum DTI:", high_dti["dti_n"].max())

# 3. Distribution of unusual DTI values
print("\n--- DTI VALUE COUNTS ---")

print(
    df.loc[df["dti_n"] > 100, "dti_n"]
    .value_counts()
    .head(15)
)

# 4. Compare DTI with revenue and loan amount
print("\n--- DTI SUMMARY ---")

print(
    df.loc[df["dti_n"] > 100, [
        "dti_n", "revenue", "loan_amnt"
    ]].describe()
)

print("\nDTI investigation completed!")
