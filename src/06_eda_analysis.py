
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "loans_cleaned.csv"
)

PLOT_DIR = PROJECT_DIR / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load cleaned dataset
print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 3. Numerical feature analysis
numerical_columns = [
    "revenue",
    "loan_amnt",
    "fico_n",
    "dti_n"
]

print("\n--- NUMERICAL SUMMARY ---")
print(df[numerical_columns].describe())

# 4. Set plot style
sns.set_theme(style="whitegrid")

# 5. Plot numerical distributions
for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=50,
        kde=False
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of borrowers")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{column}_distribution.png",
        dpi=150
    )

    plt.close()

    print(f"Saved {column} distribution plot")

# 6. Loan purpose distribution
plt.figure(figsize=(10, 6))

purpose_order = df["purpose"].value_counts().index

sns.countplot(
    data=df,
    y="purpose",
    order=purpose_order
)

plt.title("Loan Purpose Distribution")
plt.xlabel("Number of loans")
plt.ylabel("Loan Purpose")
plt.tight_layout()

plt.savefig(PLOT_DIR / "loan_purpose_distribution.png", dpi=150)
plt.close()

print("Saved loan purpose plot")

# 7. Home ownership distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="home_ownership_n",
    order=df["home_ownership_n"].value_counts().index
)

plt.title("Home Ownership Distribution")
plt.xlabel("Home Ownership")
plt.ylabel("Number of loans")
plt.tight_layout()

plt.savefig(PLOT_DIR / "home_ownership_distribution.png", dpi=150)
plt.close()

print("Saved home ownership plot")

# 8. Default distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Default",
    order=[0, 1]
)

plt.title("Default Distribution")
plt.xlabel("Default label")
plt.ylabel("Number of loans")
plt.tight_layout()

plt.savefig(PLOT_DIR / "default_distribution.png", dpi=150)
plt.close()

print("Saved default distribution plot")

print("\nEDA analysis completed successfully!")
print("All plots saved in:", PLOT_DIR)
