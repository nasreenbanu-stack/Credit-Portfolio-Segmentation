
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loans_granting_model_dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Default data type:")
print(df["Default"].dtype)

print("\nUnique Default values:")
print(df["Default"].unique())

print("\nDefault value counts:")
print(df["Default"].value_counts(dropna=False))

print("\nDTI values around 999:")
print(df.loc[df["dti_n"] >= 990, "dti_n"].value_counts().head(20))
