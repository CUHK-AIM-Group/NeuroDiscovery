"""Estimator registry for tabular NeuroClaw baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Cohen's d requires at least two observations per group")
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _xgboost(model: str, task: str, seed: int, params: dict[str, Any]):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is optional. Install the 'tabular' NeuroClaw model dependencies."
        ) from exc
    defaults = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 1,
    }
    defaults.update(params)
    if task == "classification":
        return XGBClassifier(eval_metric="logloss", **defaults)
    return XGBRegressor(objective="reg:squarederror", **defaults)


def make_estimator(
    model: str,
    task: str,
    seed: int = 123,
    **params: Any,
):
    """Build a leakage-safe sklearn-compatible estimator."""
    name = model.lower().replace("-", "_")
    if name in {"xgboost", "xgb"}:
        return _xgboost(name, task, seed, params)
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if task == "classification":
        if name in {"logistic", "logistic_regression"}:
            estimator = LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 2000)),
                class_weight=params.get("class_weight", "balanced"),
                random_state=seed,
            )
        elif name in {"ridge", "ridge_classifier"}:
            estimator = RidgeClassifier(alpha=float(params.get("alpha", 1.0)))
        elif name in {"elastic_net", "elastic_logistic"}:
            estimator = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 3000)),
                class_weight=params.get("class_weight", "balanced"),
                random_state=seed,
            )
        elif name in {"svm", "svc"}:
            estimator = SVC(
                C=float(params.get("C", 1.0)),
                kernel=params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale"),
                class_weight=params.get("class_weight", "balanced"),
                probability=True,
                random_state=seed,
            )
        elif name in {"random_forest", "rf"}:
            estimator = RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                max_features=params.get("max_features", "sqrt"),
                class_weight=params.get("class_weight", "balanced_subsample"),
                n_jobs=int(params.get("n_jobs", 1)),
                random_state=seed,
            )
        elif name in {"extra_trees", "extremely_randomized_trees"}:
            estimator = ExtraTreesClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                max_features=params.get("max_features", "sqrt"),
                class_weight=params.get("class_weight", "balanced"),
                n_jobs=int(params.get("n_jobs", 1)),
                random_state=seed,
            )
        elif name in {"hist_gradient_boosting", "hist_gb"}:
            estimator = HistGradientBoostingClassifier(
                learning_rate=float(params.get("learning_rate", 0.05)),
                max_iter=int(params.get("max_iter", 200)),
                max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                l2_regularization=float(params.get("l2_regularization", 0.1)),
                class_weight=params.get("class_weight", "balanced"),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown classification model: {model}")
    else:
        if name in {"ols", "linear", "linear_regression"}:
            estimator = LinearRegression()
        elif name == "ridge":
            estimator = Ridge(alpha=float(params.get("alpha", 1.0)))
        elif name in {"elastic_net", "elasticnet"}:
            estimator = ElasticNet(
                alpha=float(params.get("alpha", 0.01)),
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                max_iter=int(params.get("max_iter", 5000)),
                random_state=seed,
            )
        elif name in {"svm", "svr"}:
            estimator = SVR(
                C=float(params.get("C", 1.0)),
                kernel=params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale"),
                epsilon=float(params.get("epsilon", 0.1)),
            )
        elif name in {"random_forest", "rf"}:
            estimator = RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                max_features=params.get("max_features", 1.0),
                n_jobs=int(params.get("n_jobs", 1)),
                random_state=seed,
            )
        elif name in {"extra_trees", "extremely_randomized_trees"}:
            estimator = ExtraTreesRegressor(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                max_features=params.get("max_features", 1.0),
                n_jobs=int(params.get("n_jobs", 1)),
                random_state=seed,
            )
        elif name in {"hist_gradient_boosting", "hist_gb"}:
            estimator = HistGradientBoostingRegressor(
                learning_rate=float(params.get("learning_rate", 0.05)),
                max_iter=int(params.get("max_iter", 200)),
                max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                l2_regularization=float(params.get("l2_regularization", 0.1)),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown regression model: {model}")
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    if task == "regression" and params.get("scale_target", False):
        return TransformedTargetRegressor(
            regressor=pipeline,
            transformer=StandardScaler(),
        )
    return pipeline
