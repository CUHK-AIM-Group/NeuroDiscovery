"""Case-study-neutral SearchPolicy data contract.

Candidate registries and score compilers remain case-specific. This module only
defines the portable policy payload and validates exact executable anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class PolicyAnchor:
    candidate_id: str
    score: float = 1.0
    rationale: str = ""
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyRule:
    weight: float
    when: Mapping[str, str]
    rationale: str = ""


@dataclass(frozen=True)
class SearchPolicy:
    method: str
    trial: int
    schema_version: str
    anchors: tuple[PolicyAnchor, ...] = ()
    rules: tuple[PolicyRule, ...] = ()
    quotas: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _text(value: Any) -> str:
    return str(value or "").strip()


def policy_to_payload(policy: SearchPolicy) -> dict[str, Any]:
    return {
        "schema_version": policy.schema_version,
        "method": policy.method,
        "trial": policy.trial,
        "anchors": [
            {
                "candidate_id": anchor.candidate_id,
                "score": anchor.score,
                "rationale": anchor.rationale,
                "evidence_ids": list(anchor.evidence_ids),
            }
            for anchor in policy.anchors
        ],
        "rules": [
            {
                "weight": rule.weight,
                "when": dict(rule.when),
                "rationale": rule.rationale,
            }
            for rule in policy.rules
        ],
        "quotas": dict(policy.quotas),
        "metadata": dict(policy.metadata),
    }


def policy_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
    rule_fields: frozenset[str],
) -> SearchPolicy:
    schema_version = _text(payload.get("schema_version") or expected_schema)
    if schema_version != expected_schema:
        raise ValueError(f"unsupported search-policy schema: {schema_version}")
    method = _text(payload.get("method"))
    if not method:
        raise ValueError("search policy requires a method")

    anchors: list[PolicyAnchor] = []
    for raw in payload.get("anchors") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("every policy anchor must be an object")
        candidate_id = _text(raw.get("candidate_id"))
        if not candidate_id:
            raise ValueError("policy anchor candidate_id cannot be empty")
        anchors.append(
            PolicyAnchor(
                candidate_id=candidate_id,
                score=float(raw.get("score", 1.0)),
                rationale=_text(raw.get("rationale")),
                evidence_ids=tuple(
                    _text(value)
                    for value in (raw.get("evidence_ids") or [])
                    if _text(value)
                ),
            )
        )

    rules: list[PolicyRule] = []
    for raw in payload.get("rules") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("every policy rule must be an object")
        when = raw.get("when") or {}
        if not isinstance(when, Mapping) or not when:
            raise ValueError("policy rule 'when' must be a non-empty object")
        unknown = sorted(set(map(str, when)) - rule_fields)
        if unknown:
            raise ValueError(f"policy rule uses non-public fields: {unknown}")
        rules.append(
            PolicyRule(
                weight=float(raw.get("weight", 0.0)),
                when={str(key): _text(value) for key, value in when.items()},
                rationale=_text(raw.get("rationale")),
            )
        )

    quotas = payload.get("quotas") or {}
    if not isinstance(quotas, Mapping):
        raise ValueError("search policy quotas must be an object")
    return SearchPolicy(
        method=method,
        trial=int(payload.get("trial", 0)),
        schema_version=schema_version,
        anchors=tuple(anchors),
        rules=tuple(rules),
        quotas=dict(quotas),
        metadata=dict(payload.get("metadata") or {}),
    )


def validate_exact_policy(
    policy: SearchPolicy,
    *,
    expected_schema: str,
    candidate_ids: set[str],
    rule_fields: frozenset[str],
) -> None:
    if policy.schema_version != expected_schema:
        raise ValueError(f"unsupported search-policy schema: {policy.schema_version}")
    if not policy.method.strip() or policy.trial < 0:
        raise ValueError("search policy requires a method and non-negative trial")
    anchor_ids = [anchor.candidate_id for anchor in policy.anchors]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("policy anchor candidate_id values must be unique")
    unknown = sorted(set(anchor_ids) - candidate_ids)
    if unknown:
        raise ValueError(
            f"policy references unknown candidate_id values: {unknown[:5]}"
        )
    for anchor in policy.anchors:
        if not math.isfinite(anchor.score) or not 0.0 <= anchor.score <= 1.0:
            raise ValueError("policy anchor score must be finite and in [0, 1]")
    for rule in policy.rules:
        if not math.isfinite(rule.weight) or not -1.0 <= rule.weight <= 1.0:
            raise ValueError("policy rule weight must be finite and in [-1, 1]")
        unknown_fields = sorted(set(rule.when) - rule_fields)
        if unknown_fields:
            raise ValueError(f"policy rule uses non-public fields: {unknown_fields}")


# Last Updated At: 2026-08-01 10:20 HKT
