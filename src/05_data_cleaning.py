
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "loans_granting_model_dataset.csv"
)

CLEAN_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "loans_cleaned.csv"
)

# --------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# --------------------------------------------------

print("Loading original dataset...")

df = pd.read_csv(RAW_PATH, low_memory=False)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 3. CHECK DUPLICATES
# --------------------------------------------------

print("\n--- DUPLICATE CHECK ---")

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df["id"].duplicated().sum())

# --------------------------------------------------
# 4. HANDLE DTI SPECIAL VALUE
# --------------------------------------------------

print("\n--- DTI CLEANING ---")

print("DTI = 999 before cleaning:", (df["dti_n"] == 999).sum())

# Treat 999 as a provisional special missing-value code
df["dti_n"] = df["dti_n"].replace(999, np.nan)

print("DTI = 999 after cleaning:", (df["dti_n"] == 999).sum())
print("Missing DTI values:", df["dti_n"].isna().sum())

# --------------------------------------------------
# 5. CONVERT ISSUE DATE
# --------------------------------------------------

print("\n--- DATE CONVERSION ---")

df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%Y",
    errors="coerce"
)

print("Issue date data type:", df["issue_d"].dtype)
print("Invalid issue dates:", df["issue_d"].isna().sum())

# --------------------------------------------------
# 6. CHECK MISSING VALUES
# --------------------------------------------------

print("\n--- MISSING VALUES AFTER CLEANING ---")

print(df.isnull().sum())

# --------------------------------------------------
# 7. SAVE CLEANED DATASET
# --------------------------------------------------

CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEAN_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved location:", CLEAN_PATH)

# --------------------------------------------------
# 8. FINAL VERIFICATION
# --------------------------------------------------

print("\n--- FINAL DATASET CHECK ---")

print("Final shape:", df.shape)
print("Remaining duplicate rows:", df.duplicated().sum())
print("Remaining duplicate IDs:", df["id"].duplicated().sum())

print("\nData cleaning completed!")
