"""Case-study-neutral audit for accidental generator-policy collapse."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from core.scripts.case_study_search_policy import SearchPolicy


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def audit_policy_independence(
    registry: pd.DataFrame,
    policies: Iterable[SearchPolicy],
    *,
    compile_order: Callable[[pd.DataFrame, SearchPolicy], np.ndarray],
    compiler_version: str,
    trial: int | None = None,
    top_ks: tuple[int, ...] = (10, 50, 100),
    collapse_threshold: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policies = list(policies)
    trials_by_method: dict[str, set[int]] = {}
    for policy in policies:
        trials_by_method.setdefault(policy.method, set()).add(policy.trial)
    if len(trials_by_method) < 2:
        raise ValueError("policy-independence audit requires at least two methods")
    common = set.intersection(*trials_by_method.values())
    selected_trial = min(common) if trial is None and common else trial
    if selected_trial is None:
        raise ValueError("adapted methods have no common trial")
    selected = {
        policy.method: policy
        for policy in policies
        if policy.trial == int(selected_trial)
    }
    if len(selected) < 2:
        raise ValueError(f"fewer than two methods have trial {selected_trial}")

    candidate_ids = registry["candidate_id"].astype(str).to_numpy()
    effective_ks = sorted(
        {min(int(value), len(registry)) for value in top_ks if int(value) > 0}
    )
    orders = {
        method: compile_order(registry, policy) for method, policy in selected.items()
    }
    rows: list[dict[str, Any]] = []
    collapsed_pairs: list[dict[str, Any]] = []
    for left, right in combinations(sorted(selected), 2):
        left_sequence = tuple(anchor.candidate_id for anchor in selected[left].anchors)
        right_sequence = tuple(
            anchor.candidate_id for anchor in selected[right].anchors
        )
        row: dict[str, Any] = {
            "trial": int(selected_trial),
            "method_a": left,
            "method_b": right,
            "anchors_a": len(left_sequence),
            "anchors_b": len(right_sequence),
            "anchor_jaccard": _jaccard(set(left_sequence), set(right_sequence)),
            "identical_anchor_sequence": left_sequence == right_sequence,
        }
        monitored: list[float] = []
        for k in effective_ks:
            left_prefix = set(candidate_ids[orders[left][:k]])
            right_prefix = set(candidate_ids[orders[right][:k]])
            row[f"top_{k}_jaccard"] = _jaccard(left_prefix, right_prefix)
            row[f"top_{k}_overlap"] = len(left_prefix & right_prefix)
            if k < len(registry):
                monitored.append(float(row[f"top_{k}_jaccard"]))
        collapsed = bool(
            row["identical_anchor_sequence"]
            or (monitored and all(value >= collapse_threshold for value in monitored))
        )
        row["collapsed"] = collapsed
        rows.append(row)
        if collapsed:
            collapsed_pairs.append({"method_a": left, "method_b": right})
    summary = {
        "compiler_version": compiler_version,
        "trial": int(selected_trial),
        "candidate_count": len(registry),
        "methods": sorted(selected),
        "top_ks": effective_ks,
        "collapse_threshold": collapse_threshold,
        "collapsed_pairs": collapsed_pairs,
        "passed": not collapsed_pairs,
    }
    return pd.DataFrame(rows), summary


# Last Updated At: 2026-08-01 10:20 HKT
