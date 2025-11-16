"""Train a Gradient Boosting model and evaluate it.

This module exposes a single function `train_and_evaluate` that accepts train/test
dataframes and a preprocessor (ColumnTransformer). It performs a small CV sanity-check
on the training data, fits the final model, computes metrics on the hold-out set,
saves plots to `figures/`, and returns a dictionary of key results.
"""
import os
from typing import Dict
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils import save_confusion_matrix, save_roc_curve, save_feature_importances


def train_and_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, preprocessor, output_dir: str = "figures") -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    # Pipeline with preprocessor + GBM classifier.
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    # Cross-validate on training set for sanity checks
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(clone(pipeline), X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

    # Fit final pipeline on full training data
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function
        try:
            y_proba = pipeline.decision_function(X_test)
        except Exception:
            y_proba = None

    # Compute metrics
    results = {}
    results["accuracy"] = float(accuracy_score(y_test, y_pred))
    results["precision"] = float(precision_score(y_test, y_pred))
    results["recall"] = float(recall_score(y_test, y_pred))
    results["f1"] = float(f1_score(y_test, y_pred))
    if y_proba is not None:
        results["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        results["roc_auc"] = None

    # Save confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix(y_test, y_pred, labels=[0, 1], out_path=cm_path)

    # Save ROC curve
    if y_proba is not None:
        roc_path = os.path.join(output_dir, "roc_curve.png")
        save_roc_curve(y_test, y_proba, out_path=roc_path)

    # Feature importances — expand preprocessor feature names if available
    try:
        pre = pipeline.named_steps["preprocessor"]
        # get_feature_names_out accepts input_features in recent sklearn; supply X_train.columns if needed
        try:
            feature_names = pre.get_feature_names_out(X_train.columns)
        except Exception:
            feature_names = pre.get_feature_names_out()

        importances = pipeline.named_steps["clf"].feature_importances_
        fi_path = os.path.join(output_dir, "feature_importances.png")
        save_feature_importances(feature_names, importances, out_path=fi_path, top_n=20)
        # Create a small table of top features
        idx = np.argsort(importances)[::-1]
        top_features = [(feature_names[i], float(importances[i])) for i in idx[:20]]
        results["top_features"] = top_features
    except Exception:
        results["top_features"] = None

    # Add CV summaries
    cv_summary = {k: float(v.mean()) for k, v in cv_results.items() if k.startswith("test_")}
    results["cv_summary"] = cv_summary

    return results
