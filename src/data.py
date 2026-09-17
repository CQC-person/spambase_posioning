"""Data loading, parsing, and splitting utilities for UCI Spambase dataset."""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_spambase_feature_names(names_path: str = "data/spambase.names") -> List[str]:
    """Parse the 57 continuous feature names from spambase.names."""
    feature_names = []
    with open(names_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|") or line.startswith("1, 0"):
                continue
            if ":" in line:
                name = line.split(":")[0].strip()
                feature_names.append(name)
    if len(feature_names) != 57:
        raise ValueError(f"Expected 57 feature names, found {len(feature_names)}")
    return feature_names


def load_spambase(
    data_path: str = "data/spambase.data",
    names_path: str = "data/spambase.names"
) -> pd.DataFrame:
    """Load the UCI Spambase dataset into a pandas DataFrame with descriptive column names."""
    feature_names = parse_spambase_feature_names(names_path)
    columns = feature_names + ["is_spam"]
    df = pd.read_csv(data_path, header=None, names=columns)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Compute dataset health checks: shape, missing values, duplicates, class distribution."""
    n_rows, n_cols = df.shape
    n_features = n_cols - 1
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    class_counts = df["is_spam"].value_counts().to_dict()
    class_proportions = df["is_spam"].value_counts(normalize=True).to_dict()
    
    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "n_features": n_features,
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "class_distribution": {
            "non_spam_0": class_counts.get(0, 0),
            "spam_1": class_counts.get(1, 0)
        },
        "class_proportions": {
            "non_spam_0": class_proportions.get(0, 0.0),
            "spam_1": class_proportions.get(1, 0.0)
        }
    }


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified 80/20 train-test split.
    
    The test set must remain clean, isolated, and unchanged.
    """
    X = df.drop(columns=["is_spam"]).copy()
    y = df["is_spam"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
