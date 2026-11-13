
from pathlib import Path
import pandas as pd

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Required project files
required_files = [
    "data/processed/loans_cleaned.csv",
    "data/processed/borrower_features_selected.csv",
    "data/processed/borrower_clustering_matrix.npz",
    "data/processed/default_labels.csv",
    "data/processed/final_kmeans_assignments.csv",
    "models/preprocessor.joblib",
    "models/kmeans_final_k3.joblib",
    "outputs/reports/dataset_summary.csv",
    "outputs/reports/kmeans_experiment_results.csv",
    "outputs/reports/clustering_algorithm_comparison.csv",
    "outputs/reports/final_kmeans_cluster_sizes.csv",
    "outputs/reports/final_cluster_numerical_profile.csv",
    "outputs/reports/final_cluster_categorical_profile.csv",
    "outputs/reports/final_default_label_analysis.csv",
]

# Required visualizations
required_plots = [
    "final_cluster_size_distribution.png",
    "final_revenue_by_cluster.png",
    "final_loan_amount_by_cluster.png",
    "final_fico_by_cluster.png",
    "final_dti_by_cluster.png",
    "final_default_labels_by_cluster.png",
    "final_cluster_comparison_heatmap.png",
]

print("=" * 65)
print("CREDIT PORTFOLIO PROJECT - OUTPUT VERIFICATION")
print("=" * 65)

# Verify project files
print("\n1. CHECKING REQUIRED FILES")

missing_files = []

for file_name in required_files:
    file_path = PROJECT_DIR / file_name

    if file_path.exists():
        print(f"[OK] {file_name}")
    else:
        print(f"[MISSING] {file_name}")
        missing_files.append(file_name)

# Verify visualizations
print("\n2. CHECKING FINAL VISUALIZATIONS")

missing_plots = []

for plot_name in required_plots:
    plot_path = PROJECT_DIR / "outputs" / "plots" / plot_name

    if plot_path.exists():
        print(f"[OK] {plot_name}")
    else:
        print(f"[MISSING] {plot_name}")
        missing_plots.append(plot_name)

# Verify final assignments
print("\n3. VERIFYING FINAL CLUSTER ASSIGNMENTS")

assignment_path = (
    PROJECT_DIR / "data" / "processed"
    / "final_kmeans_assignments.csv"
)

if assignment_path.exists():
    assignments = pd.read_csv(assignment_path)

    print("Total assigned borrowers:", len(assignments))
    print("Unique clusters:", sorted(assignments["Cluster"].unique()))
    print("\nCluster counts:")
    print(assignments["Cluster"].value_counts().sort_index())

    if len(assignments) == 1347681 and assignments["Cluster"].nunique() == 3:
        print("[OK] Full dataset has three cluster assignments.")
    else:
        print("[WARNING] Check assignment count or cluster count.")

# Final status
print("\n4. FINAL VERIFICATION")

if not missing_files and not missing_plots:
    print("SUCCESS: All required files and visualizations are present.")
else:
    print("WARNING: Some project outputs are missing.")

    if missing_files:
        print("\nMissing files:", missing_files)

    if missing_plots:
        print("\nMissing plots:", missing_plots)

print("\nPROJECT OUTPUT VERIFICATION COMPLETED.")
