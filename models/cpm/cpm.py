"""Canonical CPM estimator with fold-local edge selection."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
from sklearn.linear_model import LinearRegression, LogisticRegression


class CPM:
    """Finn-style CPM for regression or binary classification."""

    def __init__(self, task: str = "regression", p_threshold: float = 0.01):
        self.task = task
        self.p_threshold = p_threshold

    @staticmethod
    def _vectorize(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            return X
        if X.ndim != 3 or X.shape[1] != X.shape[2]:
            raise ValueError("CPM expects [samples, ROI, ROI] or pre-vectorized features")
        upper = np.triu_indices(X.shape[1], k=1)
        return X[:, upper[0], upper[1]]

    @staticmethod
    def _edge_correlations(
        features: np.ndarray,
        y: np.ndarray,
        *,
        chunk_size: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute edge-wise Pearson statistics without a Python loop per edge."""
        features = np.asarray(features)
        y = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(y):
            raise ValueError("features and y must have aligned sample dimensions")
        n_samples, n_features = features.shape
        correlations = np.zeros(n_features, dtype=np.float64)
        p_values = np.ones(n_features, dtype=np.float64)
        if n_samples < 3 or np.var(y) == 0:
            return correlations, p_values

        y_sum = float(np.sum(y, dtype=np.float64))
        y_centered_ss = float(np.sum(y * y, dtype=np.float64) - y_sum**2 / n_samples)
        for start in range(0, n_features, chunk_size):
            stop = min(start + chunk_size, n_features)
            block = np.asarray(features[:, start:stop], dtype=np.float64)
            x_sum = np.sum(block, axis=0, dtype=np.float64)
            x_centered_ss = (
                np.sum(block * block, axis=0, dtype=np.float64)
                - x_sum * x_sum / n_samples
            )
            covariance = np.sum(block * y[:, None], axis=0, dtype=np.float64)
            covariance -= x_sum * y_sum / n_samples
            denominator = np.sqrt(np.maximum(x_centered_ss * y_centered_ss, 0.0))
            valid = denominator > 0
            r = np.zeros(stop - start, dtype=np.float64)
            r[valid] = covariance[valid] / denominator[valid]
            r = np.clip(r, -1.0, 1.0)
            correlations[start:stop] = r

            finite = valid & (np.abs(r) < 1.0)
            statistic = np.zeros_like(r)
            statistic[finite] = np.abs(r[finite]) * np.sqrt(
                (n_samples - 2) / np.maximum(1.0 - r[finite] ** 2, np.finfo(float).tiny)
            )
            p = np.ones_like(r)
            p[finite] = 2.0 * student_t.sf(statistic[finite], df=n_samples - 2)
            p[valid & (np.abs(r) >= 1.0)] = 0.0
            p_values[start:stop] = p
        return correlations, p_values

    def fit(self, X: np.ndarray, y: np.ndarray):
        features = self._vectorize(X)
        y = np.asarray(y, dtype=float)
        correlations, p_values = self._edge_correlations(features, y)
        self.positive_mask_ = (p_values < self.p_threshold) & (correlations > 0)
        self.negative_mask_ = (p_values < self.p_threshold) & (correlations < 0)
        if not (self.positive_mask_.any() or self.negative_mask_.any()):
            order = np.argsort(np.abs(correlations))[-min(10, len(correlations)) :]
            self.positive_mask_[order[correlations[order] >= 0]] = True
            self.negative_mask_[order[correlations[order] < 0]] = True
        strengths = self._strengths(features)
        self.model_ = (
            LogisticRegression(max_iter=2000, class_weight="balanced")
            if self.task == "classification"
            else LinearRegression()
        )
        self.model_.fit(strengths, y)
        return self

    def _strengths(self, features: np.ndarray) -> np.ndarray:
        positive = (
            features[:, self.positive_mask_].sum(axis=1)
            if self.positive_mask_.any()
            else np.zeros(len(features))
        )
        negative = (
            features[:, self.negative_mask_].sum(axis=1)
            if self.negative_mask_.any()
            else np.zeros(len(features))
        )
        return np.column_stack([positive, negative])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._strengths(self._vectorize(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("Regression CPM does not expose predict_proba")
        return self.model_.predict_proba(self._strengths(self._vectorize(X)))
