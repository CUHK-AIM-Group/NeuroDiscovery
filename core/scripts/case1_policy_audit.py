"""Detect accidental collapse between independently adapted CS1 generators."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from core.scripts.case1_search_policy import (
        COMPILER_VERSION,
        SearchPolicy,
        build_public_registry,
        compile_policy_order,
    )
except ModuleNotFoundError:
    from case1_search_policy import (
        COMPILER_VERSION,
        SearchPolicy,
        build_public_registry,
        compile_policy_order,
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _common_trial(policies: Iterable[SearchPolicy]) -> int:
    trials_by_method: dict[str, set[int]] = {}
    for policy in policies:
        trials_by_method.setdefault(policy.method, set()).add(policy.trial)
    if len(trials_by_method) < 2:
        raise ValueError("policy-independence audit requires at least two methods")
    common = set.intersection(*trials_by_method.values())
    if not common:
        raise ValueError("adapted methods have no common trial for independence audit")
    return min(common)


def audit_policy_independence(
    candidates: pd.DataFrame,
    policies: Iterable[SearchPolicy],
    *,
    trial: int | None = None,
    top_ks: tuple[int, ...] = (100, 1000, 10000),
    collapse_threshold: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare native anchors and compiled prefixes without consulting outcomes."""

    policies = list(policies)
    selected_trial = _common_trial(policies) if trial is None else int(trial)
    selected = {
        policy.method: policy for policy in policies if policy.trial == selected_trial
    }
    methods = sorted(selected)
    if len(methods) < 2:
        raise ValueError(f"fewer than two methods have policy trial {selected_trial}")

    registry = build_public_registry(candidates)
    candidate_ids = registry["candidate_id"].to_numpy(dtype=str)
    effective_ks = tuple(
        sorted({min(int(k), len(registry)) for k in top_ks if int(k) > 0})
    )
    orders = {
        method: compile_policy_order(registry, selected[method]) for method in methods
    }
    rows: list[dict[str, Any]] = []
    collapsed_pairs: list[dict[str, Any]] = []
    for left, right in combinations(methods, 2):
        left_anchor_sequence = tuple(
            anchor.candidate_id for anchor in selected[left].anchors
        )
        right_anchor_sequence = tuple(
            anchor.candidate_id for anchor in selected[right].anchors
        )
        left_anchors = set(left_anchor_sequence)
        right_anchors = set(right_anchor_sequence)
        row: dict[str, Any] = {
            "trial": selected_trial,
            "method_a": left,
            "method_b": right,
            "anchors_a": len(left_anchor_sequence),
            "anchors_b": len(right_anchor_sequence),
            "anchor_jaccard": _jaccard(left_anchors, right_anchors),
            "identical_anchor_sequence": (
                left_anchor_sequence == right_anchor_sequence
            ),
        }
        monitored_jaccards: list[float] = []
        for k in effective_ks:
            left_prefix = set(candidate_ids[orders[left][:k]])
            right_prefix = set(candidate_ids[orders[right][:k]])
            overlap = len(left_prefix & right_prefix)
            row[f"top_{k}_overlap"] = overlap
            row[f"top_{k}_overlap_fraction"] = float(overlap / k)
            row[f"top_{k}_jaccard"] = _jaccard(left_prefix, right_prefix)
            if k < len(registry) and k >= min(1000, len(registry) // 10):
                monitored_jaccards.append(row[f"top_{k}_jaccard"])
        rank_collapse = bool(
            monitored_jaccards
            and all(value >= collapse_threshold for value in monitored_jaccards)
        )
        collapsed = bool(row["identical_anchor_sequence"] or rank_collapse)
        row["collapsed"] = collapsed
        rows.append(row)
        if collapsed:
            collapsed_pairs.append(
                {
                    "method_a": left,
                    "method_b": right,
                    "identical_anchor_sequence": row["identical_anchor_sequence"],
                    "reason": (
                        "identical native anchor sequence"
                        if row["identical_anchor_sequence"]
                        else f"compiled prefix Jaccard >= {collapse_threshold:.2f}"
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    summary = {
        "compiler_version": COMPILER_VERSION,
        "trial": selected_trial,
        "candidate_count": len(registry),
        "methods": methods,
        "top_ks": list(effective_ks),
        "collapse_threshold": collapse_threshold,
        "collapsed_pairs": collapsed_pairs,
        "passed": not collapsed_pairs,
        "interpretation": (
            "This audit detects implementation collapse, not legitimately similar "
            "scientific performance."
        ),
    }
    return frame, summary


def write_policy_independence_audit(
    out_dir: Path,
    candidates: pd.DataFrame,
    policies: Iterable[SearchPolicy],
    *,
    fail_on_collapse: bool = True,
) -> dict[str, Any]:
    policies = list(policies)
    methods = sorted({policy.method for policy in policies})
    if len(methods) < 2:
        frame = pd.DataFrame()
        summary = {
            "compiler_version": COMPILER_VERSION,
            "methods": methods,
            "passed": True,
            "not_applicable": True,
            "collapsed_pairs": [],
            "interpretation": (
                "At least two methods are required to audit implementation collapse."
            ),
        }
    else:
        frame, summary = audit_policy_independence(candidates, policies)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "case1_policy_independence_audit.csv", index=False)
    (out_dir / "case1_policy_independence_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if fail_on_collapse and not summary["passed"]:
        pairs = ", ".join(
            f"{row['method_a']} vs {row['method_b']}"
            for row in summary["collapsed_pairs"]
        )
        raise ValueError(
            f"adapted search policies collapsed to indistinguishable outputs: {pairs}"
        )
    return summary
