"""Reusable cross-validation and persistence utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from .artifacts import RunArtifacts
from .data import TabularData, make_splits
from .metrics import classification_metrics, regression_metrics


EstimatorFactory = Callable[[], Any]


def cross_validate_estimator(
    data: TabularData,
    estimator_factory: EstimatorFactory,
    task: str,
    output_dir: str | Path,
    config: dict[str, Any],
    n_splits: int = 5,
    seed: int = 123,
) -> dict[str, float]:
    splits = make_splits(data.y, task, n_splits=n_splits, seed=seed, groups=data.groups)
    pred = np.empty(len(data.y), dtype=float if task == "regression" else data.y.dtype)
    score = np.full(len(data.y), np.nan, dtype=float)
    fold_id = np.full(len(data.y), -1, dtype=int)
    fitted = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        estimator = estimator_factory()
        estimator.fit(data.X[train_idx], data.y[train_idx])
        pred[test_idx] = estimator.predict(data.X[test_idx])
        if task == "classification":
            if hasattr(estimator, "predict_proba"):
                values = estimator.predict_proba(data.X[test_idx])
                score[test_idx] = values[:, -1] if values.ndim == 2 else values
            elif hasattr(estimator, "decision_function"):
                values = np.asarray(estimator.decision_function(data.X[test_idx]))
                score[test_idx] = values[:, -1] if values.ndim == 2 else values
        fold_id[test_idx] = fold
        fitted.append(estimator)
    metrics = (
        classification_metrics(data.y, pred, score if np.isfinite(score).all() else None)
        if task == "classification"
        else regression_metrics(data.y, pred)
    )
    artifacts = RunArtifacts(output_dir, config)
    artifacts.write_config()
    artifacts.write_metrics(metrics)
    rows = {
        "subject_id": data.subject_ids,
        "fold": fold_id,
        "target": data.y,
        "prediction": pred,
    }
    if np.isfinite(score).any():
        rows["score"] = score
    artifacts.write_predictions(rows)
    artifacts.write_folds({"subject_id": data.subject_ids, "fold": fold_id})
    joblib.dump(deepcopy(fitted), Path(output_dir) / "checkpoint.joblib")
    artifacts.write_manifest({"n_folds": len(splits)})
    return metrics
