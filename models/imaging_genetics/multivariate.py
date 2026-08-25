"""PLS and CCA with fold-local scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class MultivariateResult:
    x_scores: np.ndarray
    y_scores: np.ndarray
    correlations: np.ndarray
    model: object


def _fit(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    n_components: int,
    max_iter: int,
) -> MultivariateResult:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    Xs, Ys = x_scaler.fit_transform(X), y_scaler.fit_transform(Y)
    limit = min(Xs.shape[1], Ys.shape[1], len(Xs) - 1)
    n_components = max(1, min(n_components, limit))
    if method == "cca":
        estimator = CCA(n_components=n_components, max_iter=max_iter)
    else:
        estimator = PLSRegression(n_components=n_components, max_iter=max_iter)
    x_scores, y_scores = estimator.fit_transform(Xs, Ys)
    correlations = np.array(
        [
            np.corrcoef(x_scores[:, index], y_scores[:, index])[0, 1]
            for index in range(n_components)
        ]
    )
    return MultivariateResult(
        x_scores,
        y_scores,
        correlations,
        {"model": estimator, "x_scaler": x_scaler, "y_scaler": y_scaler},
    )


def fit_pls(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 2,
    max_iter: int = 500,
) -> MultivariateResult:
    return _fit(X, Y, "pls", n_components, max_iter)


def fit_cca(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 2,
    max_iter: int = 500,
) -> MultivariateResult:
    return _fit(X, Y, "cca", n_components, max_iter)
