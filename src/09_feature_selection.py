
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "processed" / "loans_cleaned.csv"
OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "borrower_features_selected.csv"
)

print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Original shape:", df.shape)

# Features selected for borrower clustering
clustering_features = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "emp_length",
    "purpose",
    "home_ownership_n",
]

# Keep Default separately for later analysis
selected_columns = clustering_features + ["Default"]

# Create selected dataset
selected_df = df[selected_columns].copy()

print("\nSelected features:")
print(clustering_features)

print("\nSelected dataset shape:", selected_df.shape)

print("\nMissing values:")
print(selected_df.isnull().sum())

print("\nData types:")
print(selected_df.dtypes)

print("\nCategorical feature counts:")

for column in [
    "emp_length",
    "purpose",
    "home_ownership_n",
]:
    print(f"\n{column}:")
    print(selected_df[column].value_counts().head(15))

# Save selected features
selected_df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature selection completed!")
print("Saved file:", OUTPUT_PATH)
