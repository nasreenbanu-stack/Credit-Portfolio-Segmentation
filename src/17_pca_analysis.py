
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
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

OUTPUT_DIR = PROJECT_DIR / "outputs" / "plots"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed clustering matrix...")

X = load_npz(MATRIX_PATH)

indices = pd.read_csv(INDEX_PATH)["row_index"].to_numpy()

X_sample = X[indices]

print("Sample shape:", X_sample.shape)

# PCA-like dimensionality reduction for sparse data
print("\nApplying Truncated SVD...")

svd = TruncatedSVD(
    n_components=20,
    random_state=42,
)

X_reduced = svd.fit_transform(X_sample)

explained_variance = svd.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

variance_df = pd.DataFrame({
    "component": range(1, 21),
    "explained_variance_ratio": explained_variance,
    "cumulative_variance": cumulative_variance,
})

print("\n--- EXPLAINED VARIANCE ---")
print(variance_df.round(4).to_string(index=False))

variance_path = REPORT_DIR / "pca_explained_variance.csv"
variance_df.to_csv(variance_path, index=False)

# Plot explained variance
plt.figure(figsize=(9, 5))

plt.plot(
    range(1, 21),
    cumulative_variance,
    marker="o",
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "pca_explained_variance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# Use first two components for visualization
X_2d = X_reduced[:, :2]

# K-Means labels using original processed features
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

labels = kmeans.fit_predict(X_sample)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=labels,
    s=8,
    alpha=0.5,
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters Visualized Using PCA")

plt.colorbar(scatter, label="Cluster")

plt.savefig(
    OUTPUT_DIR / "pca_kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print(variance_path)
print(OUTPUT_DIR / "pca_explained_variance.png")
print(OUTPUT_DIR / "pca_kmeans_clusters.png")

print("\nPCA analysis completed!")
