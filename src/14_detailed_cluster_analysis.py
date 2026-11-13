
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "loans_cleaned.csv"
)

INDEX_PATH = (
    PROJECT_DIR / "outputs" / "reports"
    / "kmeans_sample_indices.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"

print("Loading borrower data...")

df = pd.read_csv(DATA_PATH, low_memory=False)

indices = pd.read_csv(INDEX_PATH)["row_index"].to_numpy()

sample_df = df.iloc[indices].copy().reset_index(drop=True)

categorical_features = [
    "emp_length",
    "purpose",
    "home_ownership_n",
]

for k in [2, 3]:

    print(f"\n{'=' * 50}")
    print(f"DETAILED ANALYSIS FOR K = {k}")
    print("=" * 50)

    assignment_path = (
        OUTPUT_DIR / f"kmeans_assignments_k{k}.csv"
    )

    assignments = pd.read_csv(assignment_path)

    # Confirm row alignment
    if not assignments["id"].equals(sample_df["id"]):
        raise ValueError("Sample row alignment does not match!")

    sample_df["Cluster"] = assignments["Cluster"]

    # Categorical percentages
    for column in categorical_features:

        print(f"\n--- {column} (%) ---")

        percentages = pd.crosstab(
            sample_df["Cluster"],
            sample_df[column],
            normalize="index"
        ) * 100

        print(percentages.round(2))

        percentages.to_csv(
            OUTPUT_DIR / f"kmeans_k{k}_{column}_percentages.csv"
        )

    # Examine Default label separately
    print("\n--- DEFAULT LABEL DISTRIBUTION ---")

    label_counts = pd.crosstab(
        sample_df["Cluster"],
        sample_df["Default"]
    )

    label_percentages = pd.crosstab(
        sample_df["Cluster"],
        sample_df["Default"],
        normalize="index"
    ) * 100

    print("\nCounts:")
    print(label_counts)

    print("\nPercentages:")
    print(label_percentages.round(2))

    label_percentages.to_csv(
        OUTPUT_DIR / f"kmeans_k{k}_label_percentages.csv"
    )

print("\nDetailed cluster analysis completed!")
