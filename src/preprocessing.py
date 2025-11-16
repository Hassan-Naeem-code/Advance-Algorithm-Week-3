"""Preprocessing utilities: build a ColumnTransformer-based preprocessor.

This module creates a preprocessor that imputes numerics and one-hot encodes categoricals.
It is intentionally conservative (median imputation) and does not scale features because
Gradient Boosting trees do not require feature scaling.
"""
from typing import List
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer adapted to X.

    - Numeric: median imputation
    - Categorical: most_frequent imputation + one-hot encoding

    Returns a fitted ColumnTransformer (unfitted, ready to be used in a Pipeline).
    """
    numeric_cols: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols: List[str] = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    return preprocessor
