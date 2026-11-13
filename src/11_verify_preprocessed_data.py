
from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz

PROJECT_DIR = Path(__file__).resolve().parent.parent

MATRIX_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "borrower_clustering_matrix.npz"
)

FEATURE_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "encoded_feature_names.csv"
)

DEFAULT_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "default_labels.csv"
)

print("Loading processed data...")

X = load_npz(MATRIX_PATH)
feature_names = pd.read_csv(FEATURE_PATH)
default = pd.read_csv(DEFAULT_PATH)

print("\n--- MATRIX INFORMATION ---")
print("Shape:", X.shape)
print("Format:", X.getformat())
print("Non-zero values:", X.nnz)

print("\n--- FEATURE NAMES ---")
print(feature_names.to_string(index=False))

print("\n--- DEFAULT LABELS ---")
print(default["Default"].value_counts())

print("\n--- ROW COUNT CHECK ---")
print("Matrix rows:", X.shape[0])
print("Default rows:", len(default))

print("\nVerification completed!")
