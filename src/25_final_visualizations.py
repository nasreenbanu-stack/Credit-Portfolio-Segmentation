
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_DIR / "data" / "raw"
    / "loans_granting_model_dataset.csv"
)

ASSIGNMENT_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "final_kmeans_assignments.csv"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
PLOT_DIR = PROJECT_DIR / "outputs" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading borrower data...")

borrowers = pd.read_csv(
    RAW_DATA_PATH,
    usecols=[
        "revenue",
        "dti_n",
        "loan_amnt",
        "fico_n",
        "Default"
    ]
)

assignments = pd.read_csv(ASSIGNMENT_PATH)

if len(borrowers) != len(assignments):
    raise ValueError("Borrower and assignment counts do not match.")

borrowers["Cluster"] = assignments["Cluster"].values

borrowers["dti_n"] = borrowers["dti_n"].replace(999, float("nan"))

print("Data loaded successfully!")
print("Total records:", len(borrowers))


# --------------------------------------------------
# 3. Plot 1: Cluster size distribution
# --------------------------------------------------

print("\nGenerating cluster size distribution...")

cluster_counts = borrowers["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values
)

plt.title("Borrower Distribution Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Borrowers")

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_cluster_size_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 4. Plot 2: Revenue distribution
# --------------------------------------------------

print("Generating revenue distribution...")

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=borrowers,
    x="Cluster",
    y="revenue"
)

plt.title("Revenue Distribution Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_revenue_by_cluster.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 5. Plot 3: Loan amount distribution
# --------------------------------------------------

print("Generating loan amount distribution...")

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=borrowers,
    x="Cluster",
    y="loan_amnt"
)

plt.title("Loan Amount Distribution Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Loan Amount")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_loan_amount_by_cluster.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 6. Plot 4: FICO score distribution
# --------------------------------------------------

print("Generating FICO score distribution...")

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=borrowers,
    x="Cluster",
    y="fico_n"
)

plt.title("FICO Score Distribution Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("FICO Score")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_fico_by_cluster.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 7. Plot 5: DTI distribution
# --------------------------------------------------

print("Generating DTI distribution...")

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=borrowers,
    x="Cluster",
    y="dti_n"
)

plt.title("Debt-to-Income Ratio Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("DTI")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_dti_by_cluster.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 8. Plot 6: Historical Default label distribution
# --------------------------------------------------

print("Generating historical label distribution...")

label_percentages = (
    pd.crosstab(
        borrowers["Cluster"],
        borrowers["Default"],
        normalize="index"
    ) * 100
)

label_percentages = label_percentages.reindex(
    columns=[0, 1],
    fill_value=0
)

ax = label_percentages.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Historical Default Label Distribution by Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Percentage of Borrowers")
ax.set_xticklabels(
    [f"Cluster {c}" for c in label_percentages.index],
    rotation=0
)

ax.legend(title="Default Label")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_default_labels_by_cluster.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 9. Plot 7: Combined borrower segment comparison
# --------------------------------------------------

print("Generating combined cluster comparison...")

median_profile = (
    borrowers.groupby("Cluster")[
        ["revenue", "dti_n", "loan_amnt", "fico_n"]
    ]
    .median()
)

# Normalize each feature for visual comparison.
# This is only for plotting, not for model training.

normalized_profile = (
    median_profile - median_profile.min()
) / (
    median_profile.max() - median_profile.min()
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    normalized_profile,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Normalized Median Financial Profile by Cluster")
plt.xlabel("Financial Features")
plt.ylabel("Cluster")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "final_cluster_comparison_heatmap.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. Final verification
# --------------------------------------------------

print("\nFINAL VISUALIZATION FILES")

for file_path in sorted(PLOT_DIR.glob("final_*.png")):
    print(file_path.name)

print("\nALL FINAL VISUALIZATIONS GENERATED SUCCESSFULLY!")
