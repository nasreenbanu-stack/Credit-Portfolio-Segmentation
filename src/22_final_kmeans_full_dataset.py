
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

RAW_DATA_PATH = (
    PROJECT_DIR / "data" / "raw"
    / "loans_granting_model_dataset.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the preprocessed clustering matrix
# --------------------------------------------------

print("Loading preprocessed clustering matrix...")

X = load_npz(MATRIX_PATH)

print("Matrix shape:", X.shape)
print("Matrix format:", X.format)


# --------------------------------------------------
# 3. Load borrower IDs separately
# --------------------------------------------------

print("\nLoading borrower IDs...")

borrower_ids = pd.read_csv(
    RAW_DATA_PATH,
    usecols=["id"]
)

if len(borrower_ids) != X.shape[0]:
    raise ValueError(
        "Borrower ID count does not match matrix row count."
    )

print("Borrower records:", len(borrower_ids))


# --------------------------------------------------
# 4. Train final K-Means model
# --------------------------------------------------

print("\nTraining final K-Means model...")
print("Number of clusters: 3")
print("Total records:", X.shape[0])

start_time = time.time()

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42,
    algorithm="lloyd"
)

cluster_labels = kmeans.fit_predict(X)

end_time = time.time()

print("\nK-Means training completed!")
print("Execution time:", round(end_time - start_time, 2), "seconds")
print("Inertia:", round(kmeans.inertia_, 2))
print("Iterations:", kmeans.n_iter_)


# --------------------------------------------------
# 5. Save final trained model
# --------------------------------------------------

model_path = MODEL_DIR / "kmeans_final_k3.joblib"

joblib.dump(kmeans, model_path)

print("\nFinal model saved:", model_path)


# --------------------------------------------------
# 6. Save cluster assignments
# --------------------------------------------------

assignments = pd.DataFrame({
    "row_index": np.arange(X.shape[0]),
    "id": borrower_ids["id"].values,
    "Cluster": cluster_labels
})

assignment_path = (
    PROCESSED_DIR / "final_kmeans_assignments.csv"
)

assignments.to_csv(assignment_path, index=False)

print("Cluster assignments saved:", assignment_path)


# --------------------------------------------------
# 7. Calculate final cluster sizes
# --------------------------------------------------

cluster_counts = (
    assignments["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_summary = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Borrower_Count": cluster_counts.values,
    "Percentage": (
        cluster_counts.values / len(assignments) * 100
    )
})

summary_path = (
    REPORT_DIR / "final_kmeans_cluster_sizes.csv"
)

cluster_summary.to_csv(summary_path, index=False)

print("\nFINAL CLUSTER DISTRIBUTION")
print(cluster_summary.round(2).to_string(index=False))

print("\nCluster size report saved:", summary_path)

print("\nFULL-DATASET K-MEANS COMPLETED SUCCESSFULLY!")
