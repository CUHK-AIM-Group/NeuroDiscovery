"""Shared statistical endpoint definitions for the formal Case Study 2 run."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def pathway_outcome_family_labels(
    results: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply BH across imaging mediators within each pathway-outcome family."""

    required = {"exposure", "outcome", "sobel_p", "a_path_p", "b_path_p"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Case 2 results are missing family-FDR fields: {missing}")
    q_values = np.full(len(results), 1.0, dtype=float)
    positions = pd.Series(np.arange(len(results)), index=results.index)
    for _, index_labels in results.groupby(
        ["exposure", "outcome"], sort=False
    ).groups.items():
        index = positions.loc[list(index_labels)].to_numpy(dtype=int)
        p_values = pd.to_numeric(
            results.iloc[index]["sobel_p"], errors="coerce"
        ).to_numpy(float)
        # Every registered mediator remains in its prespecified family. A
        # non-estimable test therefore contributes p=1 rather than shrinking
        # the multiplicity burden after outcomes have been opened.
        registered_p = np.where(np.isfinite(p_values), p_values, 1.0)
        q_values[index] = multipletests(registered_p, method="fdr_bh")[1]
    labels = (
        (q_values < float(alpha))
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
    )
    return labels, q_values


def global_chain_labels(
    results: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply BH across the complete registered Case Study 2 candidate universe."""

    p_values = pd.to_numeric(results["sobel_p"], errors="coerce").to_numpy(float)
    # The global family is the complete registered universe. Failed or
    # non-estimable candidates count as p=1 and cannot make correction easier.
    registered_p = np.where(np.isfinite(p_values), p_values, 1.0)
    q_values = multipletests(registered_p, method="fdr_bh")[1]
    labels = (
        (q_values < float(alpha))
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
    )
    return labels, q_values


# Last Updated At: 2026-08-16 02:01 HKT
