
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed clustering matrix...")

X = load_npz(MATRIX_PATH)

print("Full dataset shape:", X.shape)

# Reproducible random sample
SAMPLE_SIZE = 50000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)

sample_indices = rng.choice(
    X.shape[0],
    size=SAMPLE_SIZE,
    replace=False,
)

X_sample = X[sample_indices]

print("\nSample created successfully!")
print("Sample shape:", X_sample.shape)

# Save sample indices for reproducibility
indices_path = OUTPUT_DIR / "kmeans_sample_indices.csv"

pd.Series(sample_indices, name="row_index").to_csv(
    indices_path,
    index=False,
)

# Experiment with different cluster counts
cluster_range = range(2, 9)

results = []

print("\nStarting K-Means experiments...")

for k in cluster_range:

    print(f"\nTraining K-Means with k = {k}")

    start_time = time.time()

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )

    labels = model.fit_predict(X_sample)

    # Calculate Silhouette Score on a subset
    score = silhouette_score(
        X_sample,
        labels,
        sample_size=5000,
        random_state=RANDOM_STATE,
    )

    elapsed_time = time.time() - start_time

    results.append({
        "n_clusters": k,
        "inertia": model.inertia_,
        "silhouette_score": score,
        "training_time_seconds": elapsed_time,
    })

    print("Inertia:", round(model.inertia_, 2))
    print("Silhouette Score:", round(score, 4))
    print("Time taken:", round(elapsed_time, 2), "seconds")

# Save results
results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "kmeans_experiment_results.csv"

results_df.to_csv(results_path, index=False)

print("\n--- K-MEANS EXPERIMENT SUMMARY ---")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print(results_path)

print("\nSample indices saved to:")
print(indices_path)

print("\nK-Means experiments completed!")
