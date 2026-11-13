
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load K-Means and Hierarchical results
kh_path = REPORT_DIR / "hierarchical_kmeans_comparison.csv"
kh_df = pd.read_csv(kh_path)

# Load DBSCAN results
dbscan_path = REPORT_DIR / "dbscan_experiment_results.csv"
dbscan_df = pd.read_csv(dbscan_path)

comparison_rows = []

# Add K-Means and Hierarchical results
for _, row in kh_df.iterrows():

    comparison_rows.append({
        "algorithm": row["algorithm"],
        "configuration": f'K={int(row["n_clusters"])}',
        "n_clusters": int(row["n_clusters"]),
        "silhouette_score": row["silhouette_score"],
        "noise_percentage": None,
        "training_time_seconds": row["training_time_seconds"],
        "sample_size": 5000
    })

# Add DBSCAN results
for _, row in dbscan_df.iterrows():

    comparison_rows.append({
        "algorithm": "DBSCAN",
        "configuration": (
            f'eps={row["eps"]}, '
            f'min_samples={int(row["min_samples"])}'
        ),
        "n_clusters": int(row["n_clusters"]),
        "silhouette_score": row["silhouette_score"],
        "noise_percentage": row["noise_percentage"],
        "training_time_seconds": row["training_time_seconds"],
        "sample_size": 5000
    })

# Create combined DataFrame
comparison_df = pd.DataFrame(comparison_rows)

# Sort by algorithm and silhouette score
comparison_df = comparison_df.sort_values(
    by=["algorithm", "silhouette_score"],
    ascending=[True, False],
    na_position="last"
)

# Save report
output_path = REPORT_DIR / "clustering_algorithm_comparison.csv"

comparison_df.to_csv(output_path, index=False)

print("\n===== CLUSTERING ALGORITHM COMPARISON =====")

print(
    comparison_df.round(4).to_string(index=False)
)

print("\nSaved comparison report:")
print(output_path)

print("\nClustering algorithm comparison completed!")
