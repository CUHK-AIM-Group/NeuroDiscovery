"""Diagnostics and off-policy value estimators."""

from __future__ import annotations

import numpy as np


def standardized_mean_difference(
    X: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    weights = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)
    values = []
    for group in (0, 1):
        mask = treatment == group
        group_weights = weights[mask]
        group_weights = group_weights / group_weights.sum()
        mean = np.sum(X[mask] * group_weights[:, None], axis=0)
        variance = np.sum(
            group_weights[:, None] * (X[mask] - mean[None, :]) ** 2, axis=0
        )
        values.append((mean, variance))
    pooled = np.sqrt((values[0][1] + values[1][1]) / 2)
    return np.divide(
        values[1][0] - values[0][0],
        pooled,
        out=np.zeros(X.shape[1]),
        where=pooled > 0,
    )


def ipw_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> float:
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=float)
    p = np.clip(np.asarray(propensity, dtype=float), 0.01, 0.99)
    return float(np.mean(t * y / p - (1 - t) * y / (1 - p)))


def policy_value(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    policy: np.ndarray,
) -> float:
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    p = np.clip(np.asarray(propensity, dtype=float), 0.01, 0.99)
    policy = np.asarray(policy, dtype=int)
    probability = np.where(t == 1, p, 1 - p)
    match = t == policy
    denominator = np.sum(match / probability)
    return float(np.sum(match * y / probability) / denominator) if denominator else 0.0
