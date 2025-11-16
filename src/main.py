"""Entry point: load data, run EDA, build preprocessor, train GBM, and save outputs.

Run: `python src/main.py`
"""
import os
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor
from model_gbm import train_and_evaluate


def do_eda(X: pd.DataFrame, y: pd.Series, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    logging.info("Data shape: %s", X.shape)
    logging.info("Target distribution:\n%s", y.value_counts(normalize=True))

    # A simple histogram for the first 2 numeric features to show distributions
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    sample_cols = numeric_cols[:2]
    if sample_cols:
        fig, ax = plt.subplots(len(sample_cols), 1, figsize=(6, 3 * len(sample_cols)))
        if len(sample_cols) == 1:
            ax = [ax]
        for a, col in zip(ax, sample_cols):
            X[col].hist(ax=a, bins=30)
            a.set_title(col)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "basic_feature_hists.png"), dpi=150)
        plt.close(fig)

    # Correlation heatmap for numeric features (small subset to keep it compact)
    subset = numeric_cols[:12]
    if subset:
        corr = X[subset].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_xticks(range(len(subset)))
        ax.set_yticks(range(len(subset)))
        ax.set_xticklabels(subset, rotation=90)
        ax.set_yticklabels(subset)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "correlation_subset.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(message)s"
        ),
    )

    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=[data.target.name])
    y = data.frame[data.target.name]

    # EDA
    do_eda(X, y, args.figures_dir)

    # Preprocessing
    preprocessor = build_preprocessor(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Train and evaluate
    results = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        output_dir=args.figures_dir,
    )

    logging.info("--- Hold-out Test Results ---")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        logging.info("%s: %s", k, results.get(k))

    logging.info("Top features (name, importance):")
    for name, imp in (results.get("top_features") or [])[:10]:
        logging.info(" - %s: %.4f", name, imp)

    logging.info("Cross-validation summary (training folds):")
    for k, v in (results.get("cv_summary") or {}).items():
        logging.info("%s: %.4f", k, v)


if __name__ == "__main__":
    main()
