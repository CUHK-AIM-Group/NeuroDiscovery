"""Opt-in, read-only experiment view with fail-closed UMLS mapping admission.

The existing triple loader is intentionally unchanged. This view preserves its
filters and order, then admits reviewed UMLS mappings only. It does not certify
all remaining biological claims, merge entities, or rewrite the source graph.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from neurooracle.scripts.streaming_graph_json import iter_concepts, iter_edges
from .triple_loader import Triple, _DROP_RELATIONS, _node_is_infra, _primary_domain


POLICY_VERSION = "umls-experiment-admission.v1"
UMLS_MAPPING_SOURCE = "UMLS_2026AA_normalized_exact_atomic_mention_alignment"
ACCEPTED_REVIEW_STATUSES = frozenset({"auto_accepted_exact"})


def is_umls_mapping(edge: dict) -> bool:
    md = edge.get("metadata") or {}
    return (
        edge.get("source") == UMLS_MAPPING_SOURCE
        or str(md.get("audit_ref") or "").startswith("mappings/")
        or (edge.get("relation_type") == "maps_to" and str(edge.get("source_id") or "").startswith("CLM_ATOM:"))
    )


def mapping_admission_reason(edge: dict) -> str | None:
    """None admits; an explicit unknown/rejected review state never passes."""
    md = edge.get("metadata") or {}
    if not is_umls_mapping(edge) and "review_status" not in md:
        return None
    status = md.get("review_status")
    if status in ACCEPTED_REVIEW_STATUSES:
        return None
    if status == "needs_review":
        return "needs_review"
    if not status:
        return "missing_review_status"
    return "unrecognized_review_status"


def legacy_drop_reason(edge: dict, domains: dict, infra: set, min_confidence: float = 0.2) -> str | None:
    """Same ordered predicates as triple_loader.load_triples_from_kg; regression-bound."""
    s, t, r = (edge.get(key) for key in ("source_id", "target_id", "relation_type"))
    if not (s and t and r):
        return "incomplete"
    if s == t:
        return "self_loop"
    if r in _DROP_RELATIONS:
        return "dropped_relation"
    if s in infra or t in infra:
        return "infra"
    if (edge.get("confidence") or 0.0) < min_confidence:
        return "low_confidence"
    if s not in domains or t not in domains:
        return "dangling"
    return None


class AdmissionView:
    def __init__(self, min_confidence: float = 0.2):
        self.min_confidence = min_confidence
        self.domains: dict[str, str] = {}
        self.infra: set[str] = set()
        self.legacy: list[Triple] = []
        self.admitted: list[Triple] = []
        self.legacy_drops: Counter = Counter()
        self.admission_drops: Counter = Counter()
        self.all_mapping_statuses: Counter = Counter()
        self.admitted_mapping_statuses: Counter = Counter()
        self.retained_relations: Counter = Counter()
        self.excluded_relations: Counter = Counter()

    def add_node(self, node_id: str, record: dict) -> None:
        if node_id in self.domains:
            raise ValueError(f"duplicate node {node_id}")
        self.domains[node_id] = _primary_domain(record)
        if _node_is_infra(record):
            self.infra.add(node_id)

    def add_edge(self, record: dict) -> tuple[str, str | None]:
        md = record.get("metadata") or {}
        if is_umls_mapping(record):
            self.all_mapping_statuses[str(md.get("review_status") or "<missing>")] += 1
        reason = legacy_drop_reason(record, self.domains, self.infra, self.min_confidence)
        if reason:
            self.legacy_drops[reason] += 1
            return "legacy_excluded", reason
        triple = Triple(record["source_id"], record["relation_type"], record["target_id"])
        self.legacy.append(triple)
        reason = mapping_admission_reason(record)
        if reason:
            self.admission_drops[reason] += 1
            self.excluded_relations[triple.relation_type] += 1
            return "admission_excluded", reason
        self.admitted.append(triple)
        self.retained_relations[triple.relation_type] += 1
        if is_umls_mapping(record):
            self.admitted_mapping_statuses[str(md.get("review_status") or "<missing>")] += 1
        return "admitted", None


def load_admitted_triples_from_kg(kg_path: str | Path, min_confidence: float = 0.2) -> tuple[list[Triple], dict[str, str]]:
    """Use the strict mapping view explicitly; no legacy default is changed."""
    kg_path = Path(kg_path)
    view = AdmissionView(min_confidence)
    for node_id, record in iter_concepts(kg_path):
        view.add_node(node_id, record)
    for edge in iter_edges(kg_path):
        view.add_edge(edge)
    active = {entity for triple in view.admitted for entity in (triple.source_id, triple.target_id)}
    return view.admitted, {entity: view.domains[entity] for entity in active}
