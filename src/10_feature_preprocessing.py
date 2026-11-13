
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import save_npz, issparse, csr_matrix

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    FunctionTransformer,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_features_selected.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading selected borrower features...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Keep Default separate
default_values = df["Default"].copy()

# Only clustering features
numerical_features = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
]

categorical_features = [
    "emp_length",
    "purpose",
    "home_ownership_n",
]

feature_columns = numerical_features + categorical_features
X = df[feature_columns]

# Revenue: impute, log transform, scale
revenue_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "log_transform",
            FunctionTransformer(
                np.log1p,
                feature_names_out="one-to-one",
            ),
        ),
        ("scaler", RobustScaler()),
    ]
)

# Other numerical features: impute and scale
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
)

# Categorical features: impute and one-hot encode
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

# Combine all preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("revenue", revenue_pipeline, ["revenue"]),
        (
            "numeric",
            numeric_pipeline,
            ["dti_n", "loan_amnt", "fico_n"],
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features,
        ),
    ],
    sparse_threshold=1.0,
)

print("\nApplying corrected preprocessing...")

X_processed = preprocessor.fit_transform(X)

# Check missing values
if issparse(X_processed):
    missing_count = np.isnan(X_processed.data).sum()
else:
    missing_count = np.isnan(X_processed).sum()

print("\nPreprocessing completed!")
print("Processed matrix shape:", X_processed.shape)
print("Matrix type:", type(X_processed))
print("Missing values remaining:", missing_count)

# Save sparse matrix
matrix_path = OUTPUT_DIR / "borrower_clustering_matrix.npz"

if not issparse(X_processed):
    X_processed = csr_matrix(X_processed)

save_npz(matrix_path, X_processed)

# Save Default separately
default_path = OUTPUT_DIR / "default_labels.csv"
default_values.to_csv(default_path, index=False)

# Save feature names
feature_names = preprocessor.get_feature_names_out()

feature_path = OUTPUT_DIR / "encoded_feature_names.csv"

pd.Series(feature_names).to_csv(
    feature_path,
    index=False,
    header=["feature_name"],
)

# Save preprocessing pipeline
pipeline_path = PROJECT_DIR / "models" / "preprocessor.joblib"
pipeline_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(preprocessor, pipeline_path)

print("\nCorrected files saved:")
print(matrix_path)
print(default_path)
print(feature_path)
print(pipeline_path)
