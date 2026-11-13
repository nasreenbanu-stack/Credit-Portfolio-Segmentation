
from pathlib import Path
import pandas as pd

# 1. Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Locate the original dataset
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loans_granting_model_dataset.csv"

# 3. Load the dataset
print("Loading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\nDataset loaded successfully!")

# 4. Understand dataset dimensions
print("\n--- DATASET SHAPE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 5. Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 6. Check data types and missing values
print("\n--- DATASET INFORMATION ---")
df.info()

# 7. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 8. Check duplicate records
print("\n--- DUPLICATE CHECK ---")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df["id"].duplicated().sum())

# 9. Understand default distribution
print("\n--- DEFAULT DISTRIBUTION ---")
print(df["Default"].value_counts())

print("\nDefault percentages:")
print(df["Default"].value_counts(normalize=True) * 100)

# 10. Numerical summary
print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())

# 11. Save a dataset summary report
summary = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Missing_Values": df.isnull().sum().values,
    "Unique_Values": df.nunique().values
})

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    REPORT_DIR / "dataset_summary.csv",
    index=False
)

print("\nDataset summary saved successfully!")
print("Report location:", REPORT_DIR / "dataset_summary.csv")
