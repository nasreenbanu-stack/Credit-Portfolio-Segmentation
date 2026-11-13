
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

INDEX_PATH = (
    PROJECT_DIR / "outputs" / "reports"
    / "kmeans_sample_indices.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 5000
SILHOUETTE_SAMPLE_SIZE = 2000

print("Loading processed clustering matrix...")

X = load_npz(MATRIX_PATH)

print("Full dataset shape:", X.shape)

# Load the previously selected 50,000 sample indices
base_indices = pd.read_csv(INDEX_PATH)["row_index"].to_numpy()

# Select 5,000 borrowers from the existing sample
rng = np.random.default_rng(RANDOM_STATE)

selected_positions = rng.choice(
    len(base_indices),
    size=SAMPLE_SIZE,
    replace=False,
)

selected_positions.sort()

sample_indices = base_indices[selected_positions]

# Extract the same 5,000 borrowers
X_sample = X[sample_indices].toarray()

print("\nHierarchical sample created!")
print("Sample shape:", X_sample.shape)

# Save sample indices
sample_index_path = (
    OUTPUT_DIR / "hierarchical_sample_indices.csv"
)

pd.DataFrame({
    "row_index": sample_indices
}).to_csv(sample_index_path, index=False)

# Select the same 2,000 observations for silhouette evaluation
silhouette_indices = rng.choice(
    SAMPLE_SIZE,
    size=SILHOUETTE_SAMPLE_SIZE,
    replace=False,
)

X_silhouette = X_sample[silhouette_indices]

results = []

for k in [2, 3]:

    print("\n" + "=" * 45)
    print(f"CLUSTERING WITH K = {k}")
    print("=" * 45)

    # -------------------------------
    # 1. Hierarchical Clustering
    # -------------------------------

    print("\nRunning Hierarchical Clustering...")

    start_time = time.perf_counter()

    hierarchical_model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
        metric="euclidean",
    )

    hierarchical_labels = hierarchical_model.fit_predict(
        X_sample
    )

    hierarchical_time = time.perf_counter() - start_time

    hierarchical_silhouette = silhouette_score(
        X_silhouette,
        hierarchical_labels[silhouette_indices],
    )

    hierarchical_distribution = pd.Series(
        hierarchical_labels
    ).value_counts().sort_index()

    print("Hierarchical cluster distribution:")
    print(hierarchical_distribution)

    print(
        "Hierarchical silhouette:",
        round(hierarchical_silhouette, 4),
    )

    print(
        "Hierarchical time:",
        round(hierarchical_time, 2),
        "seconds",
    )

    # Save hierarchical assignments
    pd.DataFrame({
        "row_index": sample_indices,
        "Cluster": hierarchical_labels,
    }).to_csv(
        OUTPUT_DIR / f"hierarchical_assignments_k{k}.csv",
        index=False,
    )

    results.append({
        "algorithm": "Hierarchical",
        "n_clusters": k,
        "silhouette_score": hierarchical_silhouette,
        "training_time_seconds": hierarchical_time,
    })

    # -------------------------------
    # 2. K-Means on same sample
    # -------------------------------

    print("\nRunning K-Means on the same sample...")

    start_time = time.perf_counter()

    kmeans_model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    kmeans_labels = kmeans_model.fit_predict(X_sample)

    kmeans_time = time.perf_counter() - start_time

    kmeans_silhouette = silhouette_score(
        X_silhouette,
        kmeans_labels[silhouette_indices],
    )

    print(
        "K-Means silhouette:",
        round(kmeans_silhouette, 4),
    )

    print(
        "K-Means time:",
        round(kmeans_time, 2),
        "seconds",
    )

    results.append({
        "algorithm": "K-Means",
        "n_clusters": k,
        "silhouette_score": kmeans_silhouette,
        "training_time_seconds": kmeans_time,
    })

# Save comparison results
results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR / "hierarchical_kmeans_comparison.csv"
)

results_df.to_csv(results_path, index=False)

print("\n--- ALGORITHM COMPARISON ---")
print(results_df.round(4).to_string(index=False))

print("\nSaved comparison:", results_path)
print("Saved sample indices:", sample_index_path)

print("\nHierarchical clustering experiments completed!")
