
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Define project paths
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

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load cluster assignments
# --------------------------------------------------

print("Loading final cluster assignments...")

assignments = pd.read_csv(ASSIGNMENT_PATH)

print("Assignment shape:", assignments.shape)


# --------------------------------------------------
# 3. Load original Default labels
# --------------------------------------------------

print("\nLoading original Default labels...")

labels = pd.read_csv(
    RAW_DATA_PATH,
    usecols=["id", "Default"]
)

print("Label data shape:", labels.shape)


# --------------------------------------------------
# 4. Verify borrower ID alignment
# --------------------------------------------------

print("\nVerifying borrower ID alignment...")

if not assignments["id"].equals(labels["id"]):
    raise ValueError(
        "Borrower IDs do not match in the same row order."
    )

print("Borrower ID alignment verified!")


# --------------------------------------------------
# 5. Attach Default labels to clusters
# --------------------------------------------------

analysis_data = assignments[
    ["id", "Cluster"]
].copy()

analysis_data["Default"] = labels["Default"].values

print("\nCombined analysis shape:", analysis_data.shape)

print("\nDefault label counts in complete dataset:")
print(analysis_data["Default"].value_counts().sort_index())


# --------------------------------------------------
# 6. Calculate label counts by cluster
# --------------------------------------------------

print("\nCalculating Default label counts...")

label_counts = pd.crosstab(
    analysis_data["Cluster"],
    analysis_data["Default"]
)

label_counts = label_counts.reindex(
    columns=[0, 1],
    fill_value=0
)

print("\nDEFAULT LABEL COUNTS BY CLUSTER")
print(label_counts)


# --------------------------------------------------
# 7. Calculate label percentages
# --------------------------------------------------

label_percentages = (
    pd.crosstab(
        analysis_data["Cluster"],
        analysis_data["Default"],
        normalize="index"
    ) * 100
)

label_percentages = label_percentages.reindex(
    columns=[0, 1],
    fill_value=0
)

print("\nDEFAULT LABEL PERCENTAGES BY CLUSTER")
print(label_percentages.round(2))


# --------------------------------------------------
# 8. Create detailed report
# --------------------------------------------------

report = label_counts.stack().reset_index()

report.columns = [
    "Cluster",
    "Default_Label",
    "Borrower_Count"
]

report["Percentage"] = report.apply(
    lambda row: (
        row["Borrower_Count"]
        / len(
            analysis_data[
                analysis_data["Cluster"] == row["Cluster"]
            ]
        )
    ) * 100,
    axis=1
)

report_path = (
    REPORT_DIR / "final_default_label_analysis.csv"
)

report.to_csv(report_path, index=False)

print("\nDetailed report saved:", report_path)


# --------------------------------------------------
# 9. Plot Default label percentages
# --------------------------------------------------

print("\nGenerating Default label comparison plot...")

ax = label_percentages.plot(
    kind="bar",
    figsize=(9, 6),
    width=0.7
)

ax.set_title(
    "Historical Default Label Distribution by Cluster"
)

ax.set_xlabel("K-Means Cluster")
ax.set_ylabel("Percentage of Borrowers")

ax.set_xticklabels(
    [f"Cluster {cluster}" for cluster in label_percentages.index],
    rotation=0
)

ax.legend(title="Default Label")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plot_path = (
    PLOT_DIR / "final_default_label_distribution.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()

print("Plot saved:", plot_path)


# --------------------------------------------------
# 10. Final verification
# --------------------------------------------------

print("\nFINAL VERIFICATION")

print("Total borrowers:", len(analysis_data))
print("Total cluster groups:", analysis_data["Cluster"].nunique())
print("Total Default labels:", analysis_data["Default"].nunique())

print("\nFINAL HISTORICAL LABEL ANALYSIS COMPLETED!")
