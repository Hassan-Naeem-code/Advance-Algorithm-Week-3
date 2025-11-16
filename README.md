# Week 3 — Disease Diagnosis with Gradient Boosting

This repository implements a scikit-learn Gradient Boosting workflow for disease diagnosis using the built-in `sklearn.datasets.load_breast_cancer()` dataset. It includes a short EDA, a fit-for-purpose preprocessing pipeline, model training with justified hyperparameters, cross-validation checks, evaluation on a hold-out test set, and interpretation via feature importances.

See `src/main.py` for the main entrypoint.

Setup

1. Create and activate a virtual environment (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the script:

```bash
python src/main.py
```

Repository layout

- `src/` — Python scripts (entrypoint + helpers)
- `figures/` — saved plots (ROC, confusion matrix, feature importances)
- `requirements.txt` — pinned dependencies
- `.gitignore` — recommended ignores

Notes & decisions

- Dataset: `sklearn.datasets.load_breast_cancer()` (no external download required).
- Train/test split: stratified 80/20 with `random_state=42`.
- Preprocessing: `ColumnTransformer` with median imputation for numerics and one-hot encoding for categoricals (if present). No scaling since GBMs do not require it.
- Model: `GradientBoostingClassifier` with modest regularization: `n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, early stopping via `n_iter_no_change=10` and `validation_fraction=0.1`.

Results & deliverables

- The script prints evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) and saves plots into `figures/` for your PPT.

APA citation for scikit-learn (example):
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
