"""Lightweight GWAS/PRS helpers for already-loaded genotype matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _residualize(values: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if covariates is None or np.asarray(covariates).size == 0:
        return values - values.mean(axis=0, keepdims=True)
    covariates = np.asarray(covariates, dtype=float)
    design = np.column_stack([np.ones(len(covariates)), covariates])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def association_scan(
    genotype: np.ndarray,
    phenotype: np.ndarray,
    variant_ids: list[str] | None = None,
    covariates: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run covariate-adjusted additive single-variant linear associations."""
    G = np.asarray(genotype, dtype=float)
    y = np.asarray(phenotype, dtype=float).reshape(-1, 1)
    if G.ndim != 2 or len(G) != len(y):
        raise ValueError("Genotype must be [samples, variants] and align with phenotype")
    G_res = _residualize(G, covariates)
    y_res = _residualize(y, covariates).ravel()
    denominator = np.sum(G_res**2, axis=0)
    beta = np.divide(
        G_res.T @ y_res,
        denominator,
        out=np.zeros(G.shape[1]),
        where=denominator > 0,
    )
    residual = y_res[:, None] - G_res * beta[None, :]
    dof = max(1, len(y_res) - (0 if covariates is None else covariates.shape[1]) - 2)
    sigma2 = np.sum(residual**2, axis=0) / dof
    se = np.sqrt(
        np.divide(sigma2, denominator, out=np.full_like(sigma2, np.inf), where=denominator > 0)
    )
    statistic = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_value = 2 * student_t.sf(np.abs(statistic), df=dof)
    ids = variant_ids or [f"variant_{index + 1}" for index in range(G.shape[1])]
    return (
        pd.DataFrame(
            {
                "variant_id": ids,
                "beta": beta,
                "standard_error": se,
                "t": statistic,
                "p_value": p_value,
            }
        )
        .sort_values("p_value")
        .reset_index(drop=True)
    )


def association_scan_lmm(
    genotype: np.ndarray,
    phenotype: np.ndarray,
    kinship: np.ndarray,
    variant_ids: list[str] | None = None,
    covariates: np.ndarray | None = None,
    log_delta_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """EMMAX-style LMM scan using a supplied subject kinship matrix.

    The variance ratio is estimated once under the null model, then every
    variant is tested after whitening by the fitted covariance.
    """
    G = np.asarray(genotype, dtype=float)
    y = np.asarray(phenotype, dtype=float)
    K = np.asarray(kinship, dtype=float)
    if K.shape != (len(y), len(y)) or G.shape[0] != len(y):
        raise ValueError("Genotype, phenotype, and kinship dimensions do not align")
    K = (K + K.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.clip(eigenvalues, 0, None)
    fixed = np.ones((len(y), 1))
    if covariates is not None:
        fixed = np.column_stack([fixed, np.asarray(covariates, dtype=float)])
    transformed_y = eigenvectors.T @ y
    transformed_fixed = eigenvectors.T @ fixed
    grid = (
        np.asarray(log_delta_grid, dtype=float)
        if log_delta_grid is not None
        else np.linspace(-5, 5, 81)
    )
    best = None
    for log_delta in grid:
        variance = eigenvalues + np.exp(log_delta)
        whitening = 1 / np.sqrt(np.maximum(variance, 1e-12))
        yw = transformed_y * whitening
        Xw = transformed_fixed * whitening[:, None]
        residual = yw - Xw @ np.linalg.lstsq(Xw, yw, rcond=None)[0]
        dof = max(1, len(y) - Xw.shape[1])
        sigma2 = np.sum(residual**2) / dof
        log_likelihood = -0.5 * (
            np.sum(np.log(variance)) + dof * np.log(max(sigma2, 1e-12))
        )
        if best is None or log_likelihood > best[0]:
            best = (log_likelihood, variance)
    assert best is not None
    whitening = 1 / np.sqrt(np.maximum(best[1], 1e-12))
    yw = transformed_y * whitening
    Cw = transformed_fixed * whitening[:, None]
    Gw = (eigenvectors.T @ np.where(np.isnan(G), np.nanmean(G, axis=0), G)) * whitening[:, None]
    projector = np.eye(len(y)) - Cw @ np.linalg.pinv(Cw)
    y_res = projector @ yw
    G_res = projector @ Gw
    denominator = np.sum(G_res**2, axis=0)
    beta = np.divide(
        G_res.T @ y_res,
        denominator,
        out=np.zeros(G.shape[1]),
        where=denominator > 0,
    )
    residual = y_res[:, None] - G_res * beta[None, :]
    dof = max(1, len(y) - Cw.shape[1] - 1)
    sigma2 = np.sum(residual**2, axis=0) / dof
    se = np.sqrt(
        np.divide(sigma2, denominator, out=np.full_like(sigma2, np.inf), where=denominator > 0)
    )
    statistic = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_value = 2 * student_t.sf(np.abs(statistic), dof)
    ids = variant_ids or [f"variant_{index + 1}" for index in range(G.shape[1])]
    return (
        pd.DataFrame(
            {
                "variant_id": ids,
                "beta": beta,
                "standard_error": se,
                "t": statistic,
                "p_value": p_value,
                "delta": float(np.mean(best[1] - eigenvalues)),
            }
        )
        .sort_values("p_value")
        .reset_index(drop=True)
    )


def polygenic_score(genotype: np.ndarray, weights: np.ndarray) -> np.ndarray:
    genotype = np.asarray(genotype, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if genotype.shape[1] != len(weights):
        raise ValueError("One PRS weight is required per genotype column")
    centered = np.where(np.isnan(genotype), np.nanmean(genotype, axis=0), genotype)
    return centered @ weights


def pathway_scores(
    genotype: np.ndarray,
    weights: np.ndarray,
    pathway_labels: list[str],
) -> pd.DataFrame:
    genotype = np.asarray(genotype, dtype=float)
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(pathway_labels)
    if genotype.shape[1] != len(weights) or len(weights) != len(labels):
        raise ValueError("Genotype columns, weights, and pathway labels must align")
    values = {}
    for pathway in sorted(set(labels)):
        mask = labels == pathway
        values[str(pathway)] = polygenic_score(genotype[:, mask], weights[mask])
    return pd.DataFrame(values)
