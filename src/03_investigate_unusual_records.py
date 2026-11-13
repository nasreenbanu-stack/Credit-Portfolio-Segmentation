
from pathlib import Path
import pandas as pd

# Load dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loans_granting_model_dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded successfully!")

# 1. Inspect DTI values greater than 100
print("\n--- RECORDS WITH DTI > 100 ---")

high_dti = df.loc[df["dti_n"] > 100, [
    "id", "revenue", "dti_n", "loan_amnt",
    "fico_n", "purpose", "Default"
]]

print("Number of records:", len(high_dti))
print(high_dti.head(10).to_string(index=False))

# 2. Inspect revenue above 1 million
print("\n--- REVENUE ABOVE 1 MILLION ---")

high_revenue = df.loc[df["revenue"] > 1000000, [
    "id", "revenue", "dti_n", "loan_amnt",
    "fico_n", "purpose", "Default"
]]

print("Number of records:", len(high_revenue))
print(high_revenue.head(10).to_string(index=False))

# 3. Inspect experience_c = 0
print("\n--- EXPERIENCE = 0 RECORDS ---")

zero_experience = df.loc[df["experience_c"] == 0]

print("Number of records:", len(zero_experience))
print(zero_experience.head(10).to_string(index=False))

# 4. Inspect DTI distribution around suspicious values
print("\n--- FREQUENCY OF HIGHEST DTI VALUES ---")
print(df["dti_n"].nlargest(20).value_counts().sort_index())

# 5. Check revenue percentiles
print("\n--- REVENUE PERCENTILES ---")
print(
    df["revenue"].quantile(
        [0.90, 0.95, 0.99, 0.995, 0.999, 1.00]
    )
)

print("\nInvestigation completed!")
