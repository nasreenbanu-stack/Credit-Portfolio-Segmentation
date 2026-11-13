
from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


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

# Fit K-Means on the same sample
print("\nGenerating K-Means labels...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

labels = kmeans.fit_predict(X_sample)

# Apply t-SNE
print("\nApplying t-SNE...")

start_time = time.perf_counter()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42,
)

X_tsne = tsne.fit_transform(X_sample)

elapsed_time = time.perf_counter() - start_time

print("t-SNE completed!")
print("Time taken:", round(elapsed_time, 2), "seconds")

# Save coordinates and cluster labels
tsne_df = pd.DataFrame({
    "TSNE_1": X_tsne[:, 0],
    "TSNE_2": X_tsne[:, 1],
    "Cluster": labels,
})

tsne_path = REPORT_DIR / "tsne_coordinates.csv"
tsne_df.to_csv(tsne_path, index=False)

# Plot
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=labels,
    s=8,
    alpha=0.5,
)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("K-Means Clusters Visualized Using t-SNE")

plt.colorbar(scatter, label="Cluster")

plt.savefig(
    OUTPUT_DIR / "tsne_kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print(tsne_path)
print(OUTPUT_DIR / "tsne_kmeans_clusters.png")

print("\nt-SNE analysis completed!")
