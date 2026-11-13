
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_DIR / "data" / "raw"
    / "loans_granting_model_dataset.csv"
)

ASSIGNMENT_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "final_kmeans_assignments.csv"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load final cluster assignments
# --------------------------------------------------

print("Loading final cluster assignments...")

assignments = pd.read_csv(ASSIGNMENT_PATH)

print("Assignment shape:", assignments.shape)

# Verify row order
expected_indices = range(len(assignments))

if not assignments["row_index"].equals(
    pd.Series(expected_indices, name="row_index")
):
    raise ValueError("Assignment row indices are not in expected order.")


# --------------------------------------------------
# 3. Load original borrower information
# --------------------------------------------------

print("\nLoading borrower information...")

columns = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "emp_length",
    "purpose",
    "home_ownership_n"
]

borrowers = pd.read_csv(
    RAW_DATA_PATH,
    usecols=columns
)

if len(borrowers) != len(assignments):
    raise ValueError("Borrower count and assignment count do not match.")

# Apply the same provisional DTI treatment used during cleaning
borrowers["dti_n"] = borrowers["dti_n"].replace(999, pd.NA)

# Attach cluster labels using the verified row order
borrowers["Cluster"] = assignments["Cluster"].values

print("Borrower data shape:", borrowers.shape)


# --------------------------------------------------
# 4. Numerical cluster profiling
# --------------------------------------------------

print("\nGenerating numerical cluster profiles...")

numerical_columns = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n"
]

numerical_profile = (
    borrowers.groupby("Cluster")[numerical_columns]
    .agg(["median", "mean"])
)

# Flatten column names
numerical_profile.columns = [
    f"{feature}_{statistic}"
    for feature, statistic in numerical_profile.columns
]

numerical_profile = numerical_profile.reset_index()

numerical_path = (
    REPORT_DIR / "final_cluster_numerical_profile.csv"
)

numerical_profile.to_csv(numerical_path, index=False)

print("\nNUMERICAL CLUSTER PROFILE")
print(numerical_profile.round(2).to_string(index=False))

print("\nSaved:", numerical_path)


# --------------------------------------------------
# 5. Categorical cluster profiling
# --------------------------------------------------

print("\nGenerating categorical cluster profiles...")

categorical_columns = [
    "emp_length",
    "purpose",
    "home_ownership_n"
]

categorical_results = []

for cluster in sorted(borrowers["Cluster"].unique()):

    cluster_data = borrowers[
        borrowers["Cluster"] == cluster
    ]

    for column in categorical_columns:

        counts = (
            cluster_data[column]
            .fillna("Missing")
            .value_counts(dropna=False)
        )

        percentages = counts / len(cluster_data) * 100

        for category, count in counts.items():

            categorical_results.append({
                "Cluster": cluster,
                "Feature": column,
                "Category": category,
                "Borrower_Count": count,
                "Percentage": percentages[category]
            })


categorical_profile = pd.DataFrame(categorical_results)

categorical_path = (
    REPORT_DIR / "final_cluster_categorical_profile.csv"
)

categorical_profile.to_csv(categorical_path, index=False)

print("\nTOP CATEGORIES IN EACH CLUSTER")

top_categories = (
    categorical_profile
    .sort_values(
        ["Cluster", "Feature", "Percentage"],
        ascending=[True, True, False]
    )
    .groupby(["Cluster", "Feature"])
    .head(1)
)

print(top_categories.round(2).to_string(index=False))

print("\nSaved:", categorical_path)


# --------------------------------------------------
# 6. Final verification
# --------------------------------------------------

print("\nCLUSTER COUNTS")

print(
    borrowers["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nFINAL CLUSTER PROFILING COMPLETED SUCCESSFULLY!")
