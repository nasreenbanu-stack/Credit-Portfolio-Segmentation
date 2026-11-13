
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

INDEX_PATH = (
    PROJECT_DIR / "outputs" / "reports"
    / "hierarchical_sample_indices.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed clustering matrix...")

X = load_npz(MATRIX_PATH)

indices = pd.read_csv(INDEX_PATH)["row_index"].to_numpy()

X_sample = X[indices].toarray()

print("DBSCAN sample shape:", X_sample.shape)

# Experiment with a small range of parameters
eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples_values = [5, 10]

results = []

for eps in eps_values:

    for min_samples in min_samples_values:

        print("\n" + "=" * 40)
        print(f"DBSCAN eps={eps}, min_samples={min_samples}")
        print("=" * 40)

        start_time = time.perf_counter()

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1,
        )

        labels = model.fit_predict(X_sample)

        elapsed_time = time.perf_counter() - start_time

        unique_labels = set(labels)

        # -1 represents noise
        n_clusters = len(unique_labels - {-1})
        noise_count = np.sum(labels == -1)
        noise_percentage = noise_count / len(labels) * 100

        print("Number of clusters:", n_clusters)
        print("Noise points:", noise_count)
        print("Noise percentage:", round(noise_percentage, 2))
        print("Time taken:", round(elapsed_time, 2), "seconds")

        # Calculate silhouette only when valid
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]

        if (
            n_clusters >= 2
            and len(np.unique(valid_labels)) >= 2
        ):

            silhouette = silhouette_score(
                X_sample[valid_mask],
                valid_labels,
            )

            print("Silhouette score (excluding noise):",
                  round(silhouette, 4))

        else:
            silhouette = np.nan
            print("Silhouette score: Not applicable")

        results.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_percentage": noise_percentage,
            "silhouette_score": silhouette,
            "training_time_seconds": elapsed_time,
        })

        # Save labels for each parameter combination
        assignment_df = pd.DataFrame({
            "row_index": indices,
            "Cluster": labels,
        })

        assignment_df.to_csv(
            OUTPUT_DIR
            / f"dbscan_eps_{eps}_min_{min_samples}.csv",
            index=False,
        )

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "dbscan_experiment_results.csv"

results_df.to_csv(results_path, index=False)

print("\n--- DBSCAN EXPERIMENT SUMMARY ---")
print(results_df.round(4).to_string(index=False))

print("\nResults saved:", results_path)
print("\nDBSCAN experiments completed!")
