"""Outcome-blind relation-aware ranking for frozen-KG hindcasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from neurooracle.src.case_study_scope import claim_case_study_ids_from_dict


RELATION_COMPONENT_RAW_FIELDS = (
    "kg_relation_global_node_support_raw",
    "kg_relation_global_pair_support_raw",
    "kg_relation_scoped_node_support_raw",
    "kg_relation_scoped_pair_support_raw",
)
RELATION_COMPONENT_FIELDS = tuple(
    field.removesuffix("_raw") for field in RELATION_COMPONENT_RAW_FIELDS
)


@dataclass(frozen=True)
class FrozenRelationAwarePolicy:
    """Shared static policy selected without reading hindcasting outcomes."""

    name: str = "relation_scoped_pair_static"
    global_node_weight: float = 0.04
    global_pair_weight: float = 0.06
    scoped_node_weight: float = 0.36
    scoped_pair_weight: float = 0.54
    tie_break_weight: float = 1e-9
    tie_break_seed: int = 20260810

    def validate(self) -> None:
        weights = self.component_weights()
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("relation-aware weights must be finite and non-negative")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("relation-aware component weights must sum to one")
        if self.tie_break_weight < 0 or not math.isfinite(self.tie_break_weight):
            raise ValueError("tie-break weight must be finite and non-negative")

    def component_weights(self) -> dict[str, float]:
        return {
            RELATION_COMPONENT_FIELDS[0]: float(self.global_node_weight),
            RELATION_COMPONENT_FIELDS[1]: float(self.global_pair_weight),
            RELATION_COMPONENT_FIELDS[2]: float(self.scoped_node_weight),
            RELATION_COMPONENT_FIELDS[3]: float(self.scoped_pair_weight),
        }

    def policy_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_FROZEN_RELATION_POLICY = FrozenRelationAwarePolicy()


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=float)
    low = float(np.nanmin(values[finite]))
    high = float(np.nanmax(values[finite]))
    out = values.copy()
    out[~finite] = low
    if math.isclose(low, high):
        return np.zeros(len(values), dtype=float)
    return (out - low) / (high - low)


def _stable_tie(candidate_id: str, seed: int) -> float:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}|{candidate_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    return value / float(2**64 - 1)


def score_relation_component_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: FrozenRelationAwarePolicy = DEFAULT_FROZEN_RELATION_POLICY,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compose relation-aware scores from public, frozen-KG components only."""

    policy.validate()
    normalized: dict[str, np.ndarray] = {}
    for raw_field, field in zip(
        RELATION_COMPONENT_RAW_FIELDS,
        RELATION_COMPONENT_FIELDS,
        strict=True,
    ):
        values = np.asarray([float(row.get(raw_field, math.nan)) for row in rows])
        if not np.isfinite(values).all():
            raise ValueError(f"missing or non-finite relation component: {raw_field}")
        normalized[field] = _minmax(values)

    scores = np.zeros(len(rows), dtype=float)
    for field, weight in policy.component_weights().items():
        scores += float(weight) * normalized[field]
    if policy.tie_break_weight:
        ties = np.asarray(
            [
                _stable_tie(str(row.get("candidate_id") or ""), policy.tie_break_seed)
                for row in rows
            ],
            dtype=float,
        )
        scores += policy.tie_break_weight * ties
        scores /= 1.0 + policy.tie_break_weight
    return scores, normalized


def _edge_supported(data: Mapping[str, Any]) -> bool:
    relation = str(data.get("relation_type") or data.get("relation") or "").casefold()
    if relation == "about" or "contradict" in relation:
        return False
    state = str(data.get("evidence_state") or data.get("status") or "").casefold()
    metadata = data.get("metadata")
    return "contradict" not in state and not (
        isinstance(metadata, Mapping) and bool(metadata.get("negated"))
    )


def _edge_in_scope(data: Mapping[str, Any], case_study_id: str) -> bool:
    return case_study_id in set(claim_case_study_ids_from_dict(data))


class FrozenGraphRelationScorer:
    """Cache local graph support needed to score one frozen Case Study pool."""

    def __init__(self, graph: Any, case_study_id: str):
        self.graph = graph
        self.case_study_id = str(case_study_id)
        self._neighbor_cache: dict[tuple[str, bool], frozenset[str]] = {}

    def _edge_payloads(self, source_id: str, target_id: str) -> tuple[Mapping[str, Any], ...]:
        data = self.graph.get_edge_data(source_id, target_id)
        if not isinstance(data, Mapping):
            return ()
        if "relation_type" in data or "relation" in data:
            return (data,)
        return tuple(value for value in data.values() if isinstance(value, Mapping))

    def _neighbors(self, node_id: str, *, scoped: bool) -> frozenset[str]:
        key = (str(node_id), bool(scoped))
        cached = self._neighbor_cache.get(key)
        if cached is not None:
            return cached
        if node_id not in self.graph:
            self._neighbor_cache[key] = frozenset()
            return self._neighbor_cache[key]
        neighbors: set[str] = set()
        for source_id, target_id, data in self.graph.out_edges(node_id, data=True):
            if _edge_supported(data) and (
                not scoped or _edge_in_scope(data, self.case_study_id)
            ):
                neighbors.add(str(target_id))
        for source_id, target_id, data in self.graph.in_edges(node_id, data=True):
            if _edge_supported(data) and (
                not scoped or _edge_in_scope(data, self.case_study_id)
            ):
                neighbors.add(str(source_id))
        result = frozenset(neighbors)
        self._neighbor_cache[key] = result
        return result

    def _direct_support(self, left_id: str, right_id: str, *, scoped: bool) -> float:
        best_forward = 0.0
        best_reverse = 0.0
        for data in self._edge_payloads(left_id, right_id):
            if _edge_supported(data) and (
                not scoped or _edge_in_scope(data, self.case_study_id)
            ):
                best_forward = max(best_forward, float(data.get("confidence", 1.0)))
        for data in self._edge_payloads(right_id, left_id):
            if _edge_supported(data) and (
                not scoped or _edge_in_scope(data, self.case_study_id)
            ):
                best_reverse = max(best_reverse, float(data.get("confidence", 1.0)))
        return max(best_forward, 0.85 * best_reverse)

    def _pair_support(self, left_id: str, right_id: str, *, scoped: bool) -> float:
        left_neighbors = self._neighbors(left_id, scoped=scoped)
        right_neighbors = self._neighbors(right_id, scoped=scoped)
        shared = len(left_neighbors & right_neighbors)
        return self._direct_support(left_id, right_id, scoped=scoped) + min(
            1.0,
            math.log1p(shared) / 4.0,
        )

    @staticmethod
    def _structure(
        hypothesis: Mapping[str, Any],
        *,
        use_path_pairs: bool,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        metadata = hypothesis.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        input_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in metadata.get("input_entity_ids") or ()
                if str(value)
            )
        )
        target_id = str(hypothesis.get("target_id") or "")
        if len(input_ids) >= 2 and target_id:
            nodes = tuple(dict.fromkeys((*input_ids, target_id)))
            return nodes, tuple((node_id, target_id) for node_id in input_ids)

        if use_path_pairs:
            pairs = tuple(
                (str(link.get("from_id") or ""), str(link.get("to_id") or ""))
                for link in hypothesis.get("path") or ()
                if link.get("from_id") and link.get("to_id")
            )
            nodes = tuple(dict.fromkeys(node_id for pair in pairs for node_id in pair))
            if pairs:
                return nodes, pairs

        source_id = str(hypothesis.get("source_id") or "")
        nodes = tuple(node_id for node_id in dict.fromkeys((source_id, target_id)) if node_id)
        pairs = ((source_id, target_id),) if source_id and target_id else ()
        return nodes, pairs

    def raw_components(
        self,
        hypothesis: Mapping[str, Any],
        *,
        use_path_pairs: bool = False,
    ) -> dict[str, float]:
        nodes, pairs = self._structure(hypothesis, use_path_pairs=use_path_pairs)

        def node_support(scoped: bool) -> float:
            values = [math.log1p(len(self._neighbors(node_id, scoped=scoped))) for node_id in nodes]
            return float(np.mean(values)) if values else 0.0

        def pair_support(scoped: bool) -> float:
            values = [
                self._pair_support(left_id, right_id, scoped=scoped)
                for left_id, right_id in pairs
            ]
            return max(values, default=0.0)

        return {
            RELATION_COMPONENT_RAW_FIELDS[0]: node_support(False),
            RELATION_COMPONENT_RAW_FIELDS[1]: pair_support(False),
            RELATION_COMPONENT_RAW_FIELDS[2]: node_support(True),
            RELATION_COMPONENT_RAW_FIELDS[3]: pair_support(True),
        }


def annotate_relation_aware_payload(
    payload: Mapping[str, Any],
    *,
    graph: Any,
    case_study_id: str,
    use_path_pairs: bool = False,
    policy: FrozenRelationAwarePolicy = DEFAULT_FROZEN_RELATION_POLICY,
) -> dict[str, Any]:
    """Attach frozen support components and order a hypothesis payload."""

    hypotheses = [dict(row) for row in payload.get("hypotheses") or ()]
    scorer = FrozenGraphRelationScorer(graph, case_study_id)
    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        components = scorer.raw_components(
            hypothesis,
            use_path_pairs=use_path_pairs,
        )
        rows.append({"candidate_id": str(hypothesis.get("id") or ""), **components})
    scores, normalized = score_relation_component_rows(rows, policy=policy)

    for index, hypothesis in enumerate(hypotheses):
        metadata = dict(hypothesis.get("metadata") or {})
        metadata.update(rows[index])
        for field in RELATION_COMPONENT_FIELDS:
            metadata[field] = float(normalized[field][index])
        metadata["relation_aware_static_score"] = float(scores[index])
        metadata["relation_aware_policy_id"] = policy.policy_id()
        metadata["relation_aware_uses_future_outcomes"] = False
        hypothesis["metadata"] = metadata

    order = sorted(
        range(len(hypotheses)),
        key=lambda index: (-float(scores[index]), str(hypotheses[index].get("id") or "")),
    )
    hypotheses = [hypotheses[index] for index in order]
    for rank, hypothesis in enumerate(hypotheses, start=1):
        metadata = dict(hypothesis.get("metadata") or {})
        metadata["relation_aware_rank"] = rank
        hypothesis["metadata"] = metadata

    metadata = dict(payload.get("metadata") or {})
    metadata["relation_aware_static_policy"] = {
        **asdict(policy),
        "policy_id": policy.policy_id(),
        "uses_future_outcomes": False,
        "case_study_id": case_study_id,
        "use_path_pairs": bool(use_path_pairs),
        "candidate_count": len(hypotheses),
    }
    return {**dict(payload), "n_hypotheses": len(hypotheses), "hypotheses": hypotheses, "metadata": metadata}


__all__ = [
    "DEFAULT_FROZEN_RELATION_POLICY",
    "FrozenGraphRelationScorer",
    "FrozenRelationAwarePolicy",
    "RELATION_COMPONENT_FIELDS",
    "RELATION_COMPONENT_RAW_FIELDS",
    "annotate_relation_aware_payload",
    "score_relation_component_rows",
]
