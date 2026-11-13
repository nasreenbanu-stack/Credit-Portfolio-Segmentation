
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Credit Portfolio Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Credit Portfolio Segmentation")
st.caption(
    "Unsupervised borrower segmentation and historical "
    "credit-label pattern analysis"
)


# ==========================================================
# 2. PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "loans_granting_model_dataset.csv"
)

ASSIGNMENT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "final_kmeans_assignments.csv"
)

NUMERICAL_PROFILE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "final_cluster_numerical_profile.csv"
)

CATEGORICAL_PROFILE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "final_cluster_categorical_profile.csv"
)

DEFAULT_ANALYSIS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "final_default_label_analysis.csv"
)


# ==========================================================
# 3. LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    borrowers = pd.read_csv(
        DATA_PATH,
        usecols=[
            "id",
            "revenue",
            "dti_n",
            "loan_amnt",
            "fico_n",
            "Default"
        ]
    )

    assignments = pd.read_csv(ASSIGNMENT_PATH)

    # Verify row count
    if len(borrowers) != len(assignments):
        raise ValueError(
            "Borrower data and cluster assignments do not match."
        )

    # Verify borrower IDs
    if not borrowers["id"].equals(assignments["id"]):
        raise ValueError(
            "Borrower IDs do not match cluster assignments."
        )

    # Attach cluster labels
    borrowers["Cluster"] = assignments["Cluster"]

    # Replace suspicious DTI placeholder
    borrowers["dti_n"] = borrowers["dti_n"].replace(
        999, float("nan")
    )

    numerical_profile = pd.read_csv(
        NUMERICAL_PROFILE_PATH
    )

    categorical_profile = pd.read_csv(
        CATEGORICAL_PROFILE_PATH
    )

    default_analysis = pd.read_csv(
        DEFAULT_ANALYSIS_PATH
    )

    return (
        borrowers,
        numerical_profile,
        categorical_profile,
        default_analysis
    )


try:

    (
        borrowers,
        numerical_profile,
        categorical_profile,
        default_analysis
    ) = load_data()

    st.success("Project data loaded successfully!")

except Exception as error:

    st.error(f"Unable to load project data: {error}")
    st.stop()


# ==========================================================
# 4. SIDEBAR FILTERS
# ==========================================================

st.sidebar.title("Dashboard Filters")

cluster_options = sorted(
    borrowers["Cluster"].unique().tolist()
)

selected_clusters = st.sidebar.multiselect(
    "Select Borrower Clusters",
    options=cluster_options,
    default=cluster_options
)

if not selected_clusters:

    st.warning("Please select at least one cluster.")
    st.stop()

filtered_data = borrowers[
    borrowers["Cluster"].isin(selected_clusters)
].copy()


# ==========================================================
# 5. DATASET OVERVIEW
# ==========================================================

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Borrowers",
        f"{len(filtered_data):,}"
    )

with col2:

    st.metric(
        "Selected Clusters",
        len(selected_clusters)
    )

with col3:

    st.metric(
        "Original Dataset Size",
        f"{len(borrowers):,}"
    )

with col4:

    st.metric(
        "Clustering Features",
        "7"
    )

st.divider()


# ==========================================================
# 6. CLUSTER SIZE DISTRIBUTION
# ==========================================================

st.header("2. Borrower Cluster Distribution")

cluster_counts = (
    filtered_data["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "Cluster",
    "Borrower Count"
]

cluster_counts["Cluster"] = (
    "Cluster " + cluster_counts["Cluster"].astype(str)
)

fig_cluster = px.bar(
    cluster_counts,
    x="Cluster",
    y="Borrower Count",
    color="Cluster",
    text="Borrower Count",
    title="Number of Borrowers in Each Cluster"
)

fig_cluster.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)

fig_cluster.update_layout(
    showlegend=False,
    yaxis_title="Number of Borrowers",
    xaxis_title="Borrower Segment"
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

st.divider()


# ==========================================================
# 7. NUMERICAL FEATURE ANALYSIS
# ==========================================================

st.header("3. Financial Characteristics by Cluster")

feature_options = {
    "Revenue": "revenue",
    "Debt-to-Income Ratio": "dti_n",
    "Loan Amount": "loan_amnt",
    "FICO Score": "fico_n"
}

selected_feature_name = st.selectbox(
    "Select Financial Feature",
    list(feature_options.keys())
)

selected_feature = feature_options[
    selected_feature_name
]

plot_data = filtered_data.dropna(
    subset=[selected_feature]
)

fig_box = px.box(
    plot_data,
    x="Cluster",
    y=selected_feature,
    color="Cluster",
    points=False,
    title=f"{selected_feature_name} Across Clusters"
)

fig_box.update_layout(
    xaxis_title="Borrower Cluster",
    yaxis_title=selected_feature_name,
    showlegend=False
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)


# ==========================================================
# 8. FINANCIAL PROFILE COMPARISON
# ==========================================================

st.header("4. Cluster Financial Profile")

profile_data = numerical_profile[
    numerical_profile["Cluster"].isin(selected_clusters)
].copy()

st.dataframe(
    profile_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Values represent numerical summaries of borrower "
    "characteristics within each cluster."
)

st.divider()


# ==========================================================
# 9. CATEGORICAL FEATURE ANALYSIS
# ==========================================================

st.header("5. Borrower Categorical Characteristics")

categorical_feature = st.selectbox(
    "Select Categorical Feature",
    [
        "emp_length",
        "purpose",
        "home_ownership_n"
    ]
)

category_data = categorical_profile[
    categorical_profile["Cluster"].isin(selected_clusters)
    & (
        categorical_profile["Feature"]
        == categorical_feature
    )
].copy()

if not category_data.empty:

    fig_category = px.bar(
        category_data,
        x="Category",
        y="Percentage",
        color="Cluster",
        barmode="group",
        title=f"{categorical_feature} Distribution by Cluster"
    )

    fig_category.update_layout(
        xaxis_title="Category",
        yaxis_title="Percentage (%)"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

else:

    st.info(
        "No categorical profile data is available "
        "for this feature."
    )

st.divider()


# ==========================================================
# 10. HISTORICAL DEFAULT LABEL ANALYSIS
# ==========================================================

st.header("6. Historical Default Label Analysis")

st.info(
    "Default is excluded from clustering. It is examined "
    "separately after clustering to compare historical "
    "label distributions. The meaning of labels 0 and 1 "
    "has not yet been independently verified."
)

label_data = filtered_data.copy()

label_counts = (
    pd.crosstab(
        label_data["Cluster"],
        label_data["Default"],
        normalize="index"
    ) * 100
)

label_counts = label_counts.reindex(
    columns=[0, 1],
    fill_value=0
)

label_chart_data = (
    label_counts
    .reset_index()
    .melt(
        id_vars="Cluster",
        var_name="Default Label",
        value_name="Percentage"
    )
)

label_chart_data["Default Label"] = (
    "Default = "
    + label_chart_data["Default Label"].astype(str)
)

fig_default = px.bar(
    label_chart_data,
    x="Cluster",
    y="Percentage",
    color="Default Label",
    barmode="stack",
    text_auto=".2f",
    title="Historical Default Label Distribution by Cluster"
)

fig_default.update_layout(
    xaxis_title="Borrower Cluster",
    yaxis_title="Percentage of Borrowers"
)

st.plotly_chart(
    fig_default,
    use_container_width=True
)


# ==========================================================
# 11. DEFAULT LABEL SUMMARY TABLE
# ==========================================================

st.subheader("Historical Label Summary")

default_summary = default_analysis[
    default_analysis["Cluster"].isin(selected_clusters)
].copy()

st.dataframe(
    default_summary,
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# 12. BORROWER DATA PREVIEW
# ==========================================================

st.header("7. Borrower Data Preview")

st.caption(
    "A limited preview of borrower characteristics and "
    "their assigned cluster."
)

preview_columns = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "Default",
    "Cluster"
]

st.dataframe(
    filtered_data[preview_columns].head(100),
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# 13. DOWNLOAD FILTERED DATA
# ==========================================================

st.subheader("Download Cluster Data")

csv_data = filtered_data[
    preview_columns
].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Borrower Data",
    data=csv_data,
    file_name="filtered_borrower_clusters.csv",
    mime="text/csv"
)


# ==========================================================
# 14. PROJECT INFORMATION
# ==========================================================

st.divider()

st.header("About This Project")

st.write(
    """
    This project applies unsupervised machine learning to
    identify borrower segments using financial and
    credit-related characteristics.

    **Clustering algorithms evaluated:**
    - K-Means
    - Hierarchical Clustering
    - DBSCAN

    **Dimensionality reduction techniques evaluated:**
    - PCA-related sparse reduction using TruncatedSVD
    - t-SNE
    - UMAP

    **Final selected model:**
    - K-Means with 3 clusters

    Historical Default labels are used only for
    post-clustering analysis, not as clustering inputs.
    """
)

st.caption(
    "Credit Portfolio Segmentation and Risk Pattern Analysis"
)

st.success("Dashboard loaded successfully!")
