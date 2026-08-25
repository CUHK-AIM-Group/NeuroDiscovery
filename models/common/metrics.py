"""Task metrics with stable behavior for small smoke-test datasets."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        score = np.asarray(y_score)
        if score.ndim == 2:
            score = score[:, -1]
        out["auroc"] = float(roc_auc_score(y_true, score))
        out["auprc"] = float(average_precision_score(y_true, score))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson_r"] = float(pearsonr(y_true, y_pred).statistic)
    else:
        out["pearson_r"] = 0.0
    return out
