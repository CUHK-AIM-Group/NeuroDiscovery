"""Network-map similarity and neuromodulation target ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity_map(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    if candidates.shape[1] != len(reference):
        raise ValueError("Reference and candidate maps must have the same feature count")
    reference_norm = np.linalg.norm(reference)
    candidate_norm = np.linalg.norm(candidates, axis=1)
    denominator = np.maximum(reference_norm * candidate_norm, 1e-12)
    return (candidates @ reference) / denominator


def rank_targets(
    reference: np.ndarray,
    target_maps: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    scores = cosine_similarity_map(reference, target_maps)
    return (
        pd.DataFrame({"target": target_names, "similarity": scores})
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )


def permutation_pvalue(
    reference: np.ndarray,
    candidate: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 123,
) -> float:
    rng = np.random.default_rng(seed)
    observed = float(cosine_similarity_map(reference, candidate)[0])
    null = np.array(
        [
            cosine_similarity_map(rng.permutation(reference), candidate)[0]
            for _ in range(n_permutations)
        ]
    )
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1))
