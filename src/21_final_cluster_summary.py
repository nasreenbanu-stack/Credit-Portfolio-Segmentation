
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

ASSIGNMENT_PATH = REPORT_DIR / "kmeans_assignments_k3.csv"
LABEL_PATH = REPORT_DIR / "kmeans_k3_label_percentages.csv"

assignments = pd.read_csv(ASSIGNMENT_PATH)

print("Assignment columns:")
print(assignments.columns.tolist())

print("\nCluster counts:")
cluster_counts = assignments["Cluster"].value_counts().sort_index()

summary = pd.DataFrame({
    "cluster": cluster_counts.index,
    "borrower_count": cluster_counts.values,
    "percentage": (
        cluster_counts.values / len(assignments) * 100
    )
})

print(summary.round(2).to_string(index=False))

# Save cluster size summary
output_path = REPORT_DIR / "final_cluster_size_summary.csv"

summary.to_csv(output_path, index=False)

print("\nSaved:", output_path)
