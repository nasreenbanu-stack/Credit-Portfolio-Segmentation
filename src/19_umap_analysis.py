
from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from umap import UMAP


PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

INDEX_PATH = (
    PROJECT_DIR / "outputs" / "reports"
    / "hierarchical_sample_indices.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "plots"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed clustering matrix...")

X = load_npz(MATRIX_PATH)

indices = pd.read_csv(INDEX_PATH)["row_index"].to_numpy()

X_sample = X[indices].toarray()

print("Sample shape:", X_sample.shape)

# Generate K-Means cluster labels
print("\nGenerating K-Means labels...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

labels = kmeans.fit_predict(X_sample)

# Apply UMAP
print("\nApplying UMAP...")

start_time = time.perf_counter()

umap_model = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)

X_umap = umap_model.fit_transform(X_sample)

elapsed_time = time.perf_counter() - start_time

print("UMAP completed!")
print("Time taken:", round(elapsed_time, 2), "seconds")

# Save coordinates and cluster labels
umap_df = pd.DataFrame({
    "UMAP_1": X_umap[:, 0],
    "UMAP_2": X_umap[:, 1],
    "Cluster": labels,
})

umap_path = REPORT_DIR / "umap_coordinates.csv"
umap_df.to_csv(umap_path, index=False)

# Plot clusters
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=labels,
    s=8,
    alpha=0.5,
)

plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.title("K-Means Clusters Visualized Using UMAP")

plt.colorbar(scatter, label="Cluster")

plt.savefig(
    OUTPUT_DIR / "umap_kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print(umap_path)
print(OUTPUT_DIR / "umap_kmeans_clusters.png")

print("\nUMAP analysis completed!")
