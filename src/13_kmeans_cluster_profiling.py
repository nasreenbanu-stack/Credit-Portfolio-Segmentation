
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.cluster import KMeans

PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

INDEX_PATH = (
    PROJECT_DIR / "outputs" / "reports"
    / "kmeans_sample_indices.csv"
)

CLEAN_DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "loans_cleaned.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed clustering matrix...")

X = load_npz(MATRIX_PATH)

indices = pd.read_csv(INDEX_PATH)["row_index"].to_numpy()

X_sample = X[indices]

print("Sample shape:", X_sample.shape)

# Load original borrower information for profiling
df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)

sample_df = df.iloc[indices].copy().reset_index(drop=True)

numeric_profile_features = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
]

categorical_profile_features = [
    "emp_length",
    "purpose",
    "home_ownership_n",
]

all_profiles = []

for k in [2, 3]:

    print(f"\nTraining K-Means with K = {k}...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_sample)

    result = sample_df.copy()
    result["Cluster"] = labels

    # Cluster sizes
    print(f"\n--- CLUSTER DISTRIBUTION: K = {k} ---")

    distribution = (
        result["Cluster"]
        .value_counts()
        .sort_index()
        .to_frame("borrower_count")
    )

    distribution["percentage"] = (
        distribution["borrower_count"]
        / len(result) * 100
    )

    print(distribution)

    # Numerical profile
    print(f"\n--- NUMERICAL PROFILE: K = {k} ---")

    numerical_profile = (
        result.groupby("Cluster")[numeric_profile_features]
        .median()
        .round(2)
    )

    print(numerical_profile)

    # Categorical profile
    print(f"\n--- CATEGORICAL PROFILE: K = {k} ---")

    for column in categorical_profile_features:

        print(f"\n{column}:")

        category_profile = (
            result.groupby("Cluster")[column]
            .agg(lambda x: x.mode().iloc[0])
        )

        print(category_profile)

    # Save assignments
    assignment_path = (
        OUTPUT_DIR / f"kmeans_assignments_k{k}.csv"
    )

    result[["id", "Cluster"]].to_csv(
        assignment_path,
        index=False,
    )

    # Save numerical profile
    profile_path = (
        OUTPUT_DIR / f"kmeans_numerical_profile_k{k}.csv"
    )

    numerical_profile.to_csv(profile_path)

    print("\nSaved:", assignment_path)
    print("Saved:", profile_path)

print("\nK-Means cluster profiling completed!")
