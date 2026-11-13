Credit Portfolio Segmentation and Risk Pattern Analysis

Project Overview

This project focuses on analyzing borrower characteristics and identifying meaningful segments in a credit portfolio using unsupervised machine learning techniques.

The objective is to discover borrower groups based on financial, credit and loan-related characteristics and examine historical patterns across the resulting clusters.

The project also includes dimensionality reduction, clustering algorithm comparison and an interactive Streamlit dashboard.

Project Objectives

Understand and explore a large loan-granting dataset.

Perform data cleaning and preprocessing.

Identify meaningful borrower segments using clustering.

Compare K-Means, Hierarchical Clustering and DBSCAN.

Apply dimensionality reduction techniques for visualization.

Analyze historical Default label distributions after clustering.

Build an interactive Streamlit dashboard.

Dataset Description

The dataset contains approximately 1.35 million loan records and includes features such as:

Revenue

Debt-to-income ratio (DTI)

Loan amount

FICO score

Employment length

Loan purpose

Home ownership

Important: The Default column was excluded from clustering features. It was used only after clustering to examine historical label distributions.

Borrower-level datasets are not included in this public repository because of data privacy considerations.

Project Workflow

1. Data Understanding and Quality Assessment

Dataset dimensions and data types

Missing-value analysis

Duplicate checks

Distribution and unusual-value investigation

2. Data Cleaning

Investigated unusual DTI values.

Provisionally replaced DTI values of 999 with missing values.

Prepared the dataset for analysis.

3. Exploratory Data Analysis

Revenue distribution

Loan amount distribution

FICO score distribution

DTI distribution

Loan purpose and home ownership analysis

4. Feature Selection and Preprocessing

Selected clustering features:

revenue

dti_n

loan_amnt

fico_n

emp_length

purpose

home_ownership_n

Preprocessing included:

Missing-value imputation

Log transformation of revenue

Robust scaling of numerical features

One-hot encoding of categorical features

5. Clustering Algorithms

Three algorithms were evaluated:

K-Means

Tested different cluster counts.

Compared silhouette scores.

Selected three clusters for final segmentation and interpretability.

Hierarchical Clustering

Evaluated using sampled observations.

Compared its clustering performance with K-Means.

DBSCAN

Experimented with different epsilon and minimum-sample settings.

Evaluated cluster formation and noise percentages.

6. Dimensionality Reduction

The following techniques were explored:

TruncatedSVD – dimensionality reduction for the sparse preprocessed feature matrix.

t-SNE – visualization of local cluster structures.

UMAP – visualization of cluster patterns in lower-dimensional space.

7. Final Cluster Analysis

The final K-Means model was fitted using three clusters.

Cluster profiles were examined using:

Revenue

DTI

Loan amount

FICO score

Employment length

Loan purpose

Home ownership

Historical Default label distributions were analyzed separately after clustering.

8. Interactive Dashboard

A Streamlit dashboard was developed to explore:

Cluster distribution

Financial characteristics

Cluster profiles

Historical label distributions

Interactive visualizations

Borrower-level previews using locally available data

Key Findings

The final K-Means model identified three borrower segments with distinct financial characteristics.

Cluster

Median Revenue

Median Loan Amount

Median FICO

Cluster 0

50,000

9,000

682

Cluster 1

70,000

12,000

747

Cluster 2

98,000

21,000

692

Additional observations:

Cluster 0 has a relatively lower median revenue and loan amount.

Cluster 1 has the highest median FICO score.

Cluster 2 has the highest median revenue and loan amount.

Debt consolidation is a prominent loan purpose across all three clusters.

These clusters represent patterns in the dataset and should not be interpreted as validated credit-risk grades.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

SciPy

Matplotlib

Seaborn

Plotly

Streamlit

TruncatedSVD

t-SNE

UMAP

Git and GitHub

Project Structure

Credit-Portfolio-Segmentation/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── 01_data_understanding.py
│   ├── 02_data_quality_checks.py
│   ├── ...
│   └── 26_project_output_verification.py
│
├── .gitignore
├── README.md
└── requirements.txt

How to Run the Project

1. Clone the repository

git clone https://github.com/nasreenbanu-stack/Credit-Portfolio-Segmentation.git
cd Credit-Portfolio-Segmentation

2. Create a virtual environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Add the dataset locally

Place the authorized dataset in the expected location:

data/raw/loans_granting_model_dataset.csv

The dataset is not included in this repository.

5. Execute the analysis scripts

Run the scripts in numerical order from the src directory, following the data preparation, preprocessing and clustering workflow.

The scripts generate the processed data, models and reports required by subsequent stages.

6. Launch the dashboard

streamlit run dashboard/app.py

The dashboard requires the locally generated data, models and reports. These files are intentionally excluded from the public repository.

Limitations and Future Improvements

The semantic definition of the Default label has not been independently verified.

DTI values of 999 were provisionally treated as missing; their original meaning requires further investigation.

Hierarchical Clustering, DBSCAN and dimensionality-reduction experiments used samples where computationally appropriate.

Cluster assignments depend on feature selection and preprocessing decisions.

Clusters are exploratory segments, not approved lending or credit-risk decisions.

Future improvements:

Validate the meaning of the target label.

Evaluate cluster stability.

Investigate additional clustering methods.

Improve dashboard deployment.

Add model explainability and further segment comparisons.

Author

Nasreen Banu

GitHub: https://github.com/nasreenbanu-stack