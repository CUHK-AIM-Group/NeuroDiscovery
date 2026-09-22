"""Outcome-blind SearchPolicy compiler for Case Study 2.

The executable universe is pathway PRS x imaging marker x future clinical
outcome. Experimental coefficients, P values, FDR values, and KG ranks are
never included in the public registry or consulted by this compiler.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.scripts.case_study_search_policy import (
    PolicyAnchor,
    PolicyRule,
    SearchPolicy,
    policy_from_payload as _policy_from_payload,
    policy_to_payload,
    validate_exact_policy,
)

SCHEMA_VERSION = "case2-search-policy-v1"
COMPILER_VERSION = "case2-policy-compiler-v1-pathway-marker-outcome-max"
ID_FIELDS = ("exposure", "modality", "marker", "outcome")
PUBLIC_COLUMNS = (
    "candidate_id",
    "exposure",
    "pathway_id",
    "pathway_name",
    "pathway_source",
    "threshold_label",
    "gene_count",
    "modality",
    "marker",
    "outcome",
)
RULE_FIELDS = frozenset(PUBLIC_COLUMNS)


def _text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def candidate_id_from_fields(
    exposure: Any,
    modality: Any,
    marker: Any,
    outcome: Any,
) -> str:
    parts = [
        str(value or "").strip() for value in (exposure, modality, marker, outcome)
    ]
    if not all(parts) or any("|" in part for part in parts):
        raise ValueError(
            "Case 2 candidate ID fields must be non-empty and cannot contain '|'"
        )
    return "|".join(parts)


def build_public_registry(candidates: pd.DataFrame) -> pd.DataFrame:
    required = set(ID_FIELDS) | {
        "pathway_id",
        "pathway_name",
        "threshold_label",
        "gene_count",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Case 2 candidate registry is missing columns: {missing}")
    registry = pd.DataFrame(index=candidates.index)
    for column in PUBLIC_COLUMNS:
        if column == "candidate_id":
            registry[column] = [
                candidate_id_from_fields(*values)
                for values in candidates.loc[:, ID_FIELDS].itertuples(
                    index=False, name=None
                )
            ]
        elif column == "pathway_source" and column not in candidates:
            registry[column] = "NeuroOracle curated"
        elif column in candidates:
            registry[column] = _text_series(candidates[column])
        else:
            registry[column] = ""
    if registry["candidate_id"].duplicated().any():
        duplicate = registry.loc[
            registry["candidate_id"].duplicated(), "candidate_id"
        ].iloc[0]
        raise ValueError(f"Case 2 candidate_id must be unique: {duplicate}")
    return registry.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def policy_from_payload(payload: Mapping[str, Any]) -> SearchPolicy:
    return _policy_from_payload(
        payload,
        expected_schema=SCHEMA_VERSION,
        rule_fields=RULE_FIELDS,
    )


def validate_policy(policy: SearchPolicy, candidates: pd.DataFrame) -> None:
    registry = build_public_registry(candidates)
    validate_exact_policy(
        policy,
        expected_schema=SCHEMA_VERSION,
        candidate_ids=set(registry["candidate_id"]),
        rule_fields=RULE_FIELDS,
    )
    for rule in policy.rules:
        for field, value in rule.when.items():
            if value and value not in set(registry[field].astype(str)):
                raise ValueError(f"policy rule has unknown {field}={value!r}")
    if policy.quotas:
        raise ValueError("Case 2 SearchPolicy does not support quota fields")


def _mask(registry: pd.DataFrame, field: str, value: str) -> np.ndarray:
    return registry[field].astype(str).to_numpy() == str(value)


def compile_policy_scores(
    candidates: pd.DataFrame,
    policy: SearchPolicy,
) -> np.ndarray:
    registry = build_public_registry(candidates)
    validate_policy(policy, registry)
    scores = np.zeros(len(registry), dtype=float)
    id_to_index = {
        candidate_id: index
        for index, candidate_id in enumerate(registry["candidate_id"].astype(str))
    }

    for rank, anchor in enumerate(policy.anchors):
        index = id_to_index[anchor.candidate_id]
        row = registry.iloc[index]
        rank_discount = 1.0 / math.sqrt(rank + 1.0)
        weight = float(anchor.score) * rank_discount
        pathway = _mask(registry, "pathway_id", row["pathway_id"])
        exposure = _mask(registry, "exposure", row["exposure"])
        modality = _mask(registry, "modality", row["modality"])
        marker = _mask(registry, "marker", row["marker"])
        outcome = _mask(registry, "outcome", row["outcome"])
        expanded = np.maximum.reduce(
            (
                0.10 * weight * pathway,
                0.08 * weight * marker,
                0.04 * weight * outcome,
                0.03 * weight * modality,
                0.65 * weight * (pathway & marker),
                0.45 * weight * (marker & outcome),
                0.30 * weight * (pathway & outcome),
                0.18 * weight * (modality & outcome),
                0.72 * weight * (exposure & marker),
            )
        )
        expanded[index] = 3.0 + 0.5 * float(anchor.score) + 0.5 * rank_discount
        scores = np.maximum(scores, expanded)

    for rule in policy.rules:
        matched = np.ones(len(registry), dtype=bool)
        for field, expected in rule.when.items():
            matched &= _mask(registry, field, expected)
        scores[matched] += float(rule.weight)
    return scores


def compile_policy_order(
    candidates: pd.DataFrame,
    policy: SearchPolicy,
) -> np.ndarray:
    registry = build_public_registry(candidates)
    scores = compile_policy_scores(registry, policy)
    candidate_ids = registry["candidate_id"].astype(str).to_numpy()
    return np.lexsort((candidate_ids, -scores)).astype(np.int64)


__all__ = [
    "COMPILER_VERSION",
    "PolicyAnchor",
    "PolicyRule",
    "SCHEMA_VERSION",
    "SearchPolicy",
    "build_public_registry",
    "candidate_id_from_fields",
    "compile_policy_order",
    "compile_policy_scores",
    "policy_from_payload",
    "policy_to_payload",
    "validate_policy",
]


# Last Updated At: 2026-08-01 10:20 HKT
