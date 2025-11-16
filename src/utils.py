"""Utility plotting helpers for evaluation figures."""
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_confusion_matrix(y_true, y_pred, labels, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_feature_importances(
    feature_names: Sequence[str],
    importances: Sequence[float],
    out_path: str,
    top_n: int = 20,
):
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = np.array(importances)[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(names))))
    ax.barh(range(len(names))[::-1], vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (top {})".format(top_n))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
