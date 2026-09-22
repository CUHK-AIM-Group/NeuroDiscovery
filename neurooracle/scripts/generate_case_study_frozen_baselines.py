"""Generate frozen-knowledge hindcasting baselines for all formal Case Studies.

SciAgents is represented by seeded graph-path reasoning over the frozen KG.
OpenScholar-RAG is represented by the same path synthesis restricted to claims
assigned to the active Case Study before the freeze year. Neither method sees
NeuroDiscovery scores, future claims, KGE scores, or closed-loop feedback.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import ijson

from neurooracle.scripts.run_case_study_hindcasting import (
    DEFAULT_WINDOWS,
    Window,
    parse_window,
)
from neurooracle.src.atoms import DOMAIN_TO_ATOMS, Atom
from neurooracle.src.case_studies import (
    CaseStudy,
    case_study_by_name,
    list_case_study_names,
)
from neurooracle.src.case_study_relation_contracts import (
    CASE_STUDY_PAIR_RULES,
    case_study_endpoint_atom_routes,
    case_study_endpoint_names_allowed,
    case_study_pair_allowed,
    endpoint_matches_atom,
)
from neurooracle.src.case_study_scope import claim_case_study_ids_from_dict
from neurooracle.src.claim_semantics import (
    SemanticEndpoint,
    entity_name_tokens,
    semantic_claim_pair,
)
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
    sha256_file,
)
from neurooracle.src.experiment_source_bundle import verify_source_bundle


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("sciagents", "openscholar_rag")
TREE_RELATIONS = frozenset({"is_a", "part_of", "about"})
SEMANTIC_PROJECTION_VERSION = "claim_endpoint_semantics.v1"
ENDPOINT_CANONICAL_QUALITY_POLICY = "frozen_node_canonical_quality.v1"
NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT = 0.20
NEURODISCOVERY_ANCHOR_FANOUT_HEAD = 128
NEURODISCOVERY_ANCHOR_FANOUT_DEPTH = 800
NEURODISCOVERY_EXPLORATION_FRACTION = 0.35
NEURODISCOVERY_EXPLORATION_PER_BUCKET = 4
NEURODISCOVERY_EXPLORATION_RANK_BIN_SIZE = 16
NEURODISCOVERY_STRATIFIED_FANOUT_ANCHORS = 16
NEURODISCOVERY_STRATIFIED_FANOUT_ROWS_PER_ANCHOR = 16
NEURODISCOVERY_EVIDENCE_TAIL_FRACTION = 0.25
NEURODISCOVERY_EVIDENCE_PROTECTED_HEAD = 48
NEURODISCOVERY_FEEDBACK_ANCHOR_LIMIT = 32
NEURODISCOVERY_KGE_POLICY = "frozen_complex_pair_prior.v2"
NEURODISCOVERY_KGE_WEIGHT = 0.15
NEURODISCOVERY_KGE_ANCHOR_LIMIT = 256
NEURODISCOVERY_KGE_ANCHOR_PROTECTED_HEAD = 128
NEURODISCOVERY_KGE_NEIGHBOR_LIMIT = 512
NEURODISCOVERY_KGE_TAIL_ANCHOR_LIMIT = 2048
NEURODISCOVERY_KGE_TAIL_NEIGHBOR_LIMIT = 8
NEURODISCOVERY_KGE_NEIGHBOR_BAND = 16
NEURODISCOVERY_KGE_STRATIFIED_FRACTION = 0.20
NEURODISCOVERY_KGE_EXECUTABLE_FRACTION = 0.10
NEURODISCOVERY_ROUND_SHARD_STRIDE = 104729

_AUTHORITATIVE_SOURCE_MARKERS = (
    "cognitive atlas",
    "cognitiveatlas",
    "disgenet",
    "experiment_infra",
    "geneset",
    "hgnc",
    "hpo",
    "mesh",
    "neuroclaw-gm",
    "neuroclaw-im",
    "neuronames",
    "umls",
)
_SYNTHETIC_SOURCE_MARKERS = (
    "claim_concept",
    "extracted_claim",
    "generated_anchor",
    "local_anchor",
    "replay_anchor",
    "synthetic",
)
_STABLE_ENDPOINT_PREFIXES = (
    "ATLAS:",
    "CA:",
    "CUI:",
    "DATASET:",
    "GENE:",
    "GM:",
    "HPO:",
    "IF:",
    "IM:",
    "MESH:",
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename minimal so deeply nested Windows experiment
    # paths do not cross MAX_PATH solely because of the atomic-write suffix.
    temporary = path.parent / ".tmp"
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime_source_bundle(
    manifest_path: Path | None, *, verify_references: bool = True
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    manifest_path = manifest_path.resolve()
    archive_root = (manifest_path.parent / "files").resolve()
    executing_from_archive = Path(__file__).resolve().is_relative_to(archive_root)
    return verify_source_bundle(
        manifest_path,
        require_live_source=not executing_from_archive,
        verify_references=verify_references,
    )


def _validate_reusable_payload(
    payload: Mapping[str, Any],
    *,
    method: str,
    case_study_id: str,
    freeze_year: int,
    target_count: int,
    seed: int,
    source_bundle: Mapping[str, Any] | None,
    output: Path,
) -> None:
    metadata = payload.get("metadata") or {}
    expected = {
        "method": method,
        "case_study_id": case_study_id,
        "freeze_year": freeze_year,
        "requested": target_count,
        "seed": seed,
        "source_bundle": source_bundle,
    }
    observed = {key: metadata.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"baseline checkpoint identity mismatch at {output}; rerun with --force"
        )
    hypotheses = payload.get("hypotheses") or ()
    if len(hypotheses) != target_count:
        raise ValueError(
            f"baseline checkpoint has {len(hypotheses)} hypotheses, "
            f"expected {target_count}: {output}"
        )


@dataclass(frozen=True)
class FrozenEdge:
    source_id: str
    target_id: str
    relation: str
    confidence: float
    claim_id: str = ""
    raw_text: str = ""
    source_paper: dict[str, Any] | None = None
    year: int | None = None

    def other(self, node_id: str) -> str:
        return self.target_id if node_id == self.source_id else self.source_id


@dataclass
class FrozenGraphIndex:
    concepts: dict[str, dict[str, Any]]
    names: dict[str, str]
    atoms: dict[str, frozenset[str]]
    adjacency: dict[str, list[FrozenEdge]]
    direct_pairs: set[tuple[str, str]]

    @classmethod
    def load(cls, graph_path: Path) -> "FrozenGraphIndex":
        concepts: dict[str, dict[str, Any]] = {}
        names: dict[str, str] = {}
        atoms: dict[str, frozenset[str]] = {}
        with graph_path.open("rb") as handle:
            for node_id, node in ijson.kvitems(handle, "concepts"):
                node_id = str(node_id)
                if "claim" in (node.get("domain_tags") or []):
                    continue
                concepts[node_id] = node
                node_atoms = _node_atoms(node)
                if not node_atoms:
                    continue
                atoms[node_id] = node_atoms
                names[node_id] = str(node.get("preferred_name") or node_id)

        adjacency: dict[str, list[FrozenEdge]] = defaultdict(list)
        direct_pairs: set[tuple[str, str]] = set()
        with graph_path.open("rb") as handle:
            for row in ijson.items(handle, "edges.item"):
                source = str(row.get("source_id") or row.get("source") or "")
                target = str(row.get("target_id") or row.get("target") or "")
                relation = str(row.get("relation_type") or "related_to")
                if (
                    not source
                    or not target
                    or source == target
                    or relation in TREE_RELATIONS
                    or source not in atoms
                    or target not in atoms
                ):
                    continue
                metadata = row.get("metadata") or {}
                if (
                    metadata.get("claim_id")
                    or row.get("claim_id")
                    or str(row.get("source") or "").lower().startswith("claim:")
                ):
                    continue
                paper = row.get("source_paper") or metadata.get("source_paper") or {}
                edge = FrozenEdge(
                    source_id=source,
                    target_id=target,
                    relation=relation,
                    confidence=_unit_float(row.get("confidence"), default=0.5),
                    claim_id=str(metadata.get("claim_id") or row.get("claim_id") or ""),
                    raw_text=str(row.get("raw_text") or metadata.get("raw_text") or ""),
                    source_paper=paper if isinstance(paper, dict) else {},
                    year=_year(paper.get("year") if isinstance(paper, dict) else None),
                )
                adjacency[source].append(edge)
                adjacency[target].append(edge)
                direct_pairs.add(_pair(source, target))
        return cls(concepts, names, atoms, dict(adjacency), direct_pairs)


@dataclass(frozen=True)
class MultiInputCandidate:
    score: float
    output_id: str
    input_bindings: tuple[tuple[str, str], ...]
    edges: tuple[FrozenEdge, ...]
    paper_keys: tuple[str, ...]
    generation_mode: str = "connected_evidence_graph"


@dataclass(frozen=True)
class EndpointEvidence:
    """Frozen, outcome-blind evidence supporting one candidate endpoint."""

    node_id: str
    score: float
    best_edge: FrozenEdge
    paper_keys: tuple[str, ...]
    canonical_quality: float = 0.0


def _node_atoms(node: dict[str, Any]) -> frozenset[str]:
    metadata = node.get("metadata") or {}
    values = {
        str(value)
        for value in (metadata.get("atom_types") or node.get("atom_types") or [])
        if str(value)
    }
    for domain in node.get("domain_tags") or []:
        values.update(atom.value for atom in DOMAIN_TO_ATOMS.get(str(domain), ()))
    return frozenset(values)


def _pair(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def _year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unit_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(1.0, max(0.0, number))


def _stable_unit(*values: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _has_metadata_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_metadata_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_metadata_value(item) for item in value)
    return True


def _endpoint_canonical_quality(index: FrozenGraphIndex, node_id: str) -> float:
    """Score endpoint identity quality using only frozen-node metadata.

    The score does not resolve or merge entities. It modestly favours identifiers
    backed by controlled vocabularies, aliases, and external identifiers over
    minimally described replay anchors when evidence support is otherwise close.
    """

    cache = getattr(index, "_endpoint_canonical_quality_cache", None)
    if cache is None:
        cache = {}
        setattr(index, "_endpoint_canonical_quality_cache", cache)
    if node_id in cache:
        return cache[node_id]

    node = index.concepts.get(node_id) or {}

    def node_value(field: str, default: Any = None) -> Any:
        if isinstance(node, Mapping):
            return node.get(field, default)
        return getattr(node, field, default)

    metadata = node_value("metadata", {}) or {}
    metadata_get = (
        metadata.get
        if isinstance(metadata, Mapping)
        else lambda _key, default=None: default
    )
    source_vocab = " ".join(
        str(value).strip().lower()
        for value in (
            node_value("source_vocab"),
            node_value("source_vocabulary"),
            metadata_get("source_vocab"),
            metadata_get("source_vocabulary"),
        )
        if value
    )
    aliases = node_value("aliases") or metadata_get("aliases")
    external_ids = node_value("external_ids") or metadata_get("external_ids")
    atom_types = metadata_get("atom_types") or node_value("atom_types")
    domains = node_value("domain_tags") or metadata_get("domain_tags")
    preferred_name = str(node_value("preferred_name") or "").strip()

    quality = 0.20
    if node_id.upper().startswith(_STABLE_ENDPOINT_PREFIXES):
        quality += 0.15
    if any(marker in source_vocab for marker in _AUTHORITATIVE_SOURCE_MARKERS):
        quality += 0.15
    if _has_metadata_value(external_ids):
        quality += 0.20
    if _has_metadata_value(aliases):
        quality += 0.10
    if _has_metadata_value(atom_types):
        quality += 0.10
    if _has_metadata_value(domains):
        quality += 0.05
    if preferred_name and preferred_name != node_id:
        quality += 0.05
    if any(marker in source_vocab for marker in _SYNTHETIC_SOURCE_MARKERS):
        quality -= 0.25
    if node_id.startswith("CLM_CONCEPT:") and not (
        _has_metadata_value(external_ids) or _has_metadata_value(aliases)
    ):
        quality -= 0.10

    quality = min(1.0, max(0.0, quality))
    cache[node_id] = quality
    return quality


def _apply_endpoint_quality_prior(
    evidence_score: float,
    canonical_quality: float,
    weight: float,
) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("canonical_quality_weight must be in [0, 1]")
    # Multiplicative shrinkage keeps evidence strength primary and prevents a
    # metadata-rich catalog-only node from leapfrogging retrieved evidence.
    return evidence_score * (1.0 - weight * (1.0 - canonical_quality))


def load_semantic_claim_adjacencies(
    claims_path: Path,
    index: FrozenGraphIndex,
    case_study_ids: Iterable[str],
) -> tuple[
    dict[str, list[FrozenEdge]],
    dict[str, dict[str, list[FrozenEdge]]],
    dict[str, int],
]:
    selected = set(case_study_ids)
    all_claims: dict[str, list[FrozenEdge]] = defaultdict(list)
    scoped: dict[str, dict[str, list[FrozenEdge]]] = {
        case_id: defaultdict(list) for case_id in selected
    }
    audit: dict[str, int] = defaultdict(int)
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            claim = json.loads(line)
            if claim.get("negated"):
                audit["negated_claims_excluded"] += 1
                continue
            memberships = selected & set(claim_case_study_ids_from_dict(claim))
            projected = semantic_claim_pair(claim, index.concepts)
            if projected is None:
                audit["claims_without_semantic_pair"] += 1
                continue
            subject, obj = projected
            source = subject.entity_id
            target = obj.entity_id
            subject_atoms = frozenset(subject.atoms)
            object_atoms = frozenset(obj.atoms)
            if not subject_atoms or not object_atoms or source == target:
                audit["claims_without_task_atoms"] += 1
                continue
            index.names.setdefault(source, subject.name)
            index.names.setdefault(target, obj.name)
            index.atoms[source] = index.atoms.get(source, frozenset()) | subject_atoms
            index.atoms[target] = index.atoms.get(target, frozenset()) | object_atoms
            index.direct_pairs.add(_pair(source, target))
            if not subject.uses_canonical_id:
                audit["claim_local_subjects"] += 1
            if not obj.uses_canonical_id:
                audit["claim_local_objects"] += 1
            paper = claim.get("source_paper") or {}
            edge = FrozenEdge(
                source_id=source,
                target_id=target,
                relation=str(claim.get("predicate") or "related_to"),
                confidence=_unit_float(claim.get("confidence"), default=0.5),
                claim_id=str(claim.get("id") or ""),
                raw_text=str(claim.get("raw_text") or ""),
                source_paper=paper if isinstance(paper, dict) else {},
                year=_year(
                    (paper if isinstance(paper, dict) else {}).get("year")
                    or claim.get("year")
                ),
            )
            all_claims[source].append(edge)
            all_claims[target].append(edge)
            audit["semantic_claim_edges"] += 1
            for case_id in memberships:
                if not case_study_pair_allowed(
                    claim,
                    index.concepts,
                    projected,
                    case_id,
                    include_support=True,
                ):
                    audit["scoped_claim_contract_rejected"] += 1
                    audit[f"scoped_claim_contract_rejected::{case_id}"] += 1
                    continue
                scoped[case_id][source].append(edge)
                scoped[case_id][target].append(edge)
                audit["scoped_claim_memberships"] += 1
                audit[f"scoped_claim_contract_accepted::{case_id}"] += 1
    audit["semantic_claim_nodes"] = len(all_claims)
    return (
        dict(all_claims),
        {case_id: dict(adjacency) for case_id, adjacency in scoped.items()},
        dict(audit),
    )


def load_case_study_claim_adjacency(
    claims_path: Path,
    index: FrozenGraphIndex,
    case_study_ids: Iterable[str],
) -> dict[str, dict[str, list[FrozenEdge]]]:
    """Compatibility wrapper returning only the scoped literature graph."""

    _, scoped, _ = load_semantic_claim_adjacencies(
        claims_path,
        index,
        case_study_ids,
    )
    return scoped


def _merge_adjacencies(
    *parts: dict[str, list[FrozenEdge]],
) -> dict[str, list[FrozenEdge]]:
    merged: dict[str, list[FrozenEdge]] = defaultdict(list)
    for adjacency in parts:
        for node_id, edges in adjacency.items():
            merged[node_id].extend(edges)
    return dict(merged)


def _endpoint(index: FrozenGraphIndex, node_id: str) -> SemanticEndpoint:
    return SemanticEndpoint(
        entity_id=node_id,
        name=index.names.get(node_id, node_id),
        canonical_id=node_id,
        atoms=tuple(sorted(index.atoms.get(node_id, frozenset()))),
        uses_canonical_id=node_id in index.concepts,
        name_score=1.0,
        role_compatible=True,
    )


def _edge_key(edge: FrozenEdge) -> tuple[str, ...]:
    paper = edge.source_paper or {}
    return (
        *_pair(edge.source_id, edge.target_id),
        edge.relation,
        edge.claim_id,
        str(paper.get("pmid") or ""),
        str(paper.get("doi") or ""),
        str(edge.year or ""),
    )


def _paper_key(edge: FrozenEdge) -> str:
    paper = edge.source_paper or {}
    return str(
        paper.get("pmid")
        or paper.get("doi")
        or paper.get("title")
        or edge.claim_id
        or ""
    )


def _unique_edges(adjacency: dict[str, list[FrozenEdge]]) -> list[FrozenEdge]:
    unique: dict[tuple[str, ...], FrozenEdge] = {}
    for edges in adjacency.values():
        for edge in edges:
            unique.setdefault(_edge_key(edge), edge)
    return list(unique.values())


def _endpoint_evidence_pool(
    *,
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    atoms: tuple[Atom, ...],
    freeze_year: int,
    limit: int,
    include_catalog: bool = True,
    case_study_id: str | None = None,
    counterpart: SemanticEndpoint | None = None,
    source_side: bool = False,
    context_adjacency: dict[str, list[FrozenEdge]] | None = None,
    canonical_quality_weight: float = 0.0,
    evidence_tail_fraction: float = 0.0,
    seed: int = 0,
    pool_label: str = "",
    required_node_ids: tuple[str, ...] = (),
    ranked_evidence_out: list[EndpointEvidence] | None = None,
) -> list[EndpointEvidence]:
    """Rank canonical endpoints using only evidence available at the freeze year."""

    if not 0.0 <= canonical_quality_weight <= 1.0:
        raise ValueError("canonical_quality_weight must be in [0, 1]")
    if not 0.0 <= evidence_tail_fraction <= 1.0:
        raise ValueError("evidence_tail_fraction must be in [0, 1]")

    def name_allowed(endpoint: SemanticEndpoint) -> bool:
        if case_study_id is None or counterpart is None:
            return True
        return case_study_endpoint_names_allowed(
            case_study_id,
            endpoint if source_side else counterpart,
            counterpart if source_side else endpoint,
        )

    def summarize(
        incident: Iterable[FrozenEdge],
    ) -> tuple[float, FrozenEdge | None, tuple[str, ...]]:
        unique = list({_edge_key(edge): edge for edge in incident}.values())
        if not unique:
            return 0.0, None, ()
        unique.sort(
            key=lambda edge: (
                -edge.confidence,
                -(edge.year or 0),
                edge.claim_id,
                edge.source_id,
                edge.target_id,
            )
        )
        years = [edge.year for edge in unique if edge.year is not None]
        recency = (
            max(0.0, min(1.0, max(years) / float(freeze_year)))
            if years
            else 0.0
        )
        papers = tuple(
            dict.fromkeys(_paper_key(edge) for edge in unique if _paper_key(edge))
        )
        support = min(1.0, math.log1p(len(unique)) / math.log(33.0))
        paper_support = min(1.0, math.log1p(len(papers)) / math.log(17.0))
        evidence = sum(edge.confidence for edge in unique[:8]) / min(8, len(unique))
        score = 0.45 * evidence + 0.25 * recency + 0.20 * support + 0.10 * paper_support
        return score, unique[0], papers

    rows: list[EndpointEvidence] = []
    context_adjacency = context_adjacency or {}
    candidate_ids = set(adjacency) | set(context_adjacency)
    for node_id in candidate_ids:
        scoped_incident = adjacency.get(node_id, ())
        context_incident = context_adjacency.get(node_id, ())
        # Claim-local identities are executable only when frozen evidence already
        # contains them. Catalog-only local identities remain ineligible below.
        if node_id.startswith("CLAIM_ENTITY:") and not (
            scoped_incident or context_incident
        ):
            continue
        has_frozen_evidence = bool(scoped_incident or context_incident)
        is_frozen_claim_local = (
            node_id.startswith("CLAIM_ENTITY:")
            and has_frozen_evidence
            and node_id in index.atoms
            and node_id in index.names
        )
        if (
            node_id not in index.concepts
            and not is_frozen_claim_local
        ) or not has_frozen_evidence:
            continue
        endpoint = _endpoint(index, node_id)
        if not any(
            endpoint_matches_atom(endpoint, atom, index.concepts)
            for atom in atoms
        ) or not name_allowed(endpoint):
            continue
        scoped_score, scoped_best, scoped_papers = summarize(scoped_incident)
        context_score, context_best, context_papers = summarize(context_incident)
        if scoped_best is not None:
            # Task-specific evidence remains primary; global frozen context is a
            # small corroborating term, never a replacement for scoped support.
            score = min(1.0, 0.80 * scoped_score + 0.20 * context_score)
            best_edge = scoped_best
        else:
            # Broad-KG endpoints are allowed to enter the retrieval frontier, but
            # at a discount so a generic high-degree concept cannot outrank a
            # comparably supported task-scoped endpoint solely by popularity.
            score = 0.60 * context_score
            best_edge = context_best
        if best_edge is None:
            continue
        canonical_quality = _endpoint_canonical_quality(index, node_id)
        score = _apply_endpoint_quality_prior(
            score,
            canonical_quality,
            canonical_quality_weight,
        )
        papers = tuple(dict.fromkeys((*scoped_papers, *context_papers)))
        rows.append(
            EndpointEvidence(
                node_id=node_id,
                score=score,
                best_edge=best_edge,
                paper_keys=papers,
                canonical_quality=canonical_quality,
            )
        )
    ranked_evidence = sorted(rows, key=lambda row: (-row.score, row.node_id))
    if ranked_evidence_out is not None:
        ranked_evidence_out.extend(ranked_evidence)
    selected_evidence = _select_endpoint_evidence_head_tail(
        ranked_evidence,
        limit=limit,
        tail_fraction=evidence_tail_fraction,
        seed=seed,
        case_study_id=case_study_id or "",
        pool_label=pool_label,
    )
    required_order = tuple(dict.fromkeys(
        node_id for node_id in required_node_ids if node_id
    ))[:NEURODISCOVERY_FEEDBACK_ANCHOR_LIMIT]
    if required_order:
        evidence_by_id = {row.node_id: row for row in ranked_evidence}
        required_evidence = [
            evidence_by_id[node_id]
            for node_id in required_order
            if node_id in evidence_by_id
        ]
        required_evidence_ids = {row.node_id for row in required_evidence}
        selected_evidence = [
            *required_evidence,
            *(
                row
                for row in selected_evidence
                if row.node_id not in required_evidence_ids
            ),
        ][:limit]
    catalog_rows: list[EndpointEvidence] = []
    required_catalog_set: set[str] = set()
    if include_catalog:
        cached = getattr(index, "_catalog_nodes_by_atom", None)
        if cached is None:
            catalog: dict[str, list[str]] = defaultdict(list)
            for node_id, node_atoms in index.atoms.items():
                if node_id not in index.concepts:
                    continue
                for atom_name in node_atoms:
                    catalog[atom_name].append(node_id)
            cached = {
                atom_name: tuple(sorted(node_ids))
                for atom_name, node_ids in catalog.items()
            }
            setattr(index, "_catalog_nodes_by_atom", cached)
        # A low-ranked evidence endpoint must remain eligible for seeded long-tail
        # exploration. Only the evidence prefix that survives ``limit`` blocks a
        # duplicate catalog row; otherwise large global contexts permanently hide
        # valid but less-studied entities from every run.
        existing = {row.node_id for row in selected_evidence}
        catalog_ids = list({
            node_id
            for atom in atoms
            for node_id in cached.get(atom.value, ())
            if not node_id.startswith("CLAIM_ENTITY:")
        })
        catalog_ids.sort(
            key=lambda node_id: (
                _stable_unit(
                    "catalog-endpoint-order",
                    seed,
                    case_study_id or "",
                    pool_label,
                    node_id,
                ),
                node_id,
            )
        )
        required_catalog_ids = [
            node_id
            for node_id in required_order
            if node_id not in existing
        ]
        required_catalog_set = set(required_catalog_ids)
        for node_id in itertools.chain(required_catalog_ids, catalog_ids):
            if node_id in existing:
                continue
            endpoint = _endpoint(index, node_id)
            if not any(
                endpoint_matches_atom(endpoint, atom, index.concepts)
                for atom in atoms
            ) or not name_allowed(endpoint):
                continue
            # Catalog-only nodes are legal task entities but carry a deliberately
            # weak prior so retrieved evidence always ranks ahead of them.
            score = 0.05 + 0.01 * _stable_unit("catalog-endpoint", node_id)
            canonical_quality = _endpoint_canonical_quality(index, node_id)
            score = _apply_endpoint_quality_prior(
                score,
                canonical_quality,
                canonical_quality_weight,
            )
            catalog_rows.append(
                EndpointEvidence(
                    node_id=node_id,
                    score=score,
                    best_edge=FrozenEdge(
                        source_id=node_id,
                        target_id=node_id,
                        relation="catalog_member",
                        confidence=0.15,
                    ),
                    paper_keys=(),
                    canonical_quality=canonical_quality,
                )
            )
            existing.add(node_id)
            if len(catalog_rows) >= limit:
                break
    catalog_rows.sort(
        key=lambda row: (
            row.node_id not in required_catalog_set,
            -row.score,
            row.node_id,
        )
    )
    return [*selected_evidence, *catalog_rows[:limit]]


def _select_endpoint_evidence_head_tail(
    ranked: list[EndpointEvidence],
    *,
    limit: int,
    tail_fraction: float,
    seed: int,
    case_study_id: str,
    pool_label: str,
) -> list[EndpointEvidence]:
    """Keep a trusted head plus a complementary rank-stratified evidence tail."""

    if limit <= 0 or not ranked:
        return []
    if tail_fraction <= 0.0 or len(ranked) <= limit:
        return ranked[:limit]

    tail_count = min(
        max(0, limit - NEURODISCOVERY_EVIDENCE_PROTECTED_HEAD),
        max(1, int(round(limit * tail_fraction))),
        max(0, len(ranked) - 1),
    )
    head_count = max(1, limit - tail_count)
    head = ranked[:head_count]
    sampling_pool = ranked[head_count:]
    if not sampling_pool or tail_count <= 0:
        return ranked[:limit]

    # Equal-rank bands prevent a very large evidence pool from spending every
    # replicate on the same moderately high-scoring tail.  Seed/round changes
    # rotate the selected member inside each band, so repeated closed-loop
    # rounds cover complementary frozen endpoints without consulting outcomes.
    sampled: list[EndpointEvidence] = []
    band_count = min(tail_count, len(sampling_pool))
    for band_index in range(band_count):
        start = band_index * len(sampling_pool) // band_count
        stop = (band_index + 1) * len(sampling_pool) // band_count
        band = sampling_pool[start:stop]
        if not band:
            continue
        base_offset = int(
            _stable_unit(
                "endpoint-evidence-rank-band",
                case_study_id,
                pool_label,
                band_index,
            )
            * len(band)
        )
        offset = (base_offset + int(seed)) % len(band)
        sampled.append(band[offset])
    protected_count = min(NEURODISCOVERY_EVIDENCE_PROTECTED_HEAD, len(head))
    protected = head[:protected_count]
    remaining_head = head[protected_count:]
    remaining_total = len(remaining_head) + len(sampled)
    if not remaining_total:
        return protected

    merged = list(protected)
    head_index = 0
    tail_index = 0
    for position in range(remaining_total):
        expected_tail = round((position + 1) * len(sampled) / remaining_total)
        if expected_tail > tail_index and tail_index < len(sampled):
            merged.append(sampled[tail_index])
            tail_index += 1
        elif head_index < len(remaining_head):
            merged.append(remaining_head[head_index])
            head_index += 1
        elif tail_index < len(sampled):
            merged.append(sampled[tail_index])
            tail_index += 1
    return merged[:limit]


def _neurodiscovery_endpoint_pair_rows(
    sources: list[EndpointEvidence],
    targets: list[EndpointEvidence],
    *,
    pool_cap: int,
    seed: int,
    case_study_id: str,
    priority_source_ids: tuple[str, ...] = (),
    priority_target_ids: tuple[str, ...] = (),
) -> Iterable[tuple[EndpointEvidence, EndpointEvidence]]:
    """Cover feedback and strong-anchor fan-outs before long-tail traversal."""

    if not sources or not targets:
        return iter(())
    target_size = len(targets)

    def coprime_stride(label: str) -> int:
        if target_size <= 1:
            return 1
        candidate = 1 + int(
            _stable_unit(label, seed, case_study_id) * (target_size - 1)
        )
        while math.gcd(candidate, target_size) != 1:
            candidate = candidate % target_size + 1
        return candidate

    source_stride = coprime_stride("frontier-source-stride")
    offset_stride = coprime_stride("frontier-offset-stride")
    shift = int(
        _stable_unit("frontier-target-shift", seed, case_study_id) * target_size
    ) % target_size
    attempt_cap = min(
        len(sources) * target_size,
        max(pool_cap * 4, len(sources)),
    )
    diagonal = (
        (
            sources[step % len(sources)],
            targets[
                (
                    (step % len(sources)) * source_stride
                    + (step // len(sources)) * offset_stride
                    + shift
                )
                % target_size
            ],
        )
        for step in range(attempt_cap)
    )

    source_head = sources[: min(NEURODISCOVERY_ANCHOR_FANOUT_HEAD, len(sources))]
    target_head = targets[: min(NEURODISCOVERY_ANCHOR_FANOUT_HEAD, len(targets))]
    source_depth = sources[: min(NEURODISCOVERY_ANCHOR_FANOUT_DEPTH, len(sources))]
    target_depth = targets[: min(NEURODISCOVERY_ANCHOR_FANOUT_DEPTH, len(targets))]
    source_by_id = {row.node_id: row for row in sources}
    target_by_id = {row.node_id: row for row in targets}
    priority_sources = [
        source_by_id[node_id]
        for node_id in dict.fromkeys(priority_source_ids)
        if node_id in source_by_id
    ][:NEURODISCOVERY_FEEDBACK_ANCHOR_LIMIT]
    priority_targets = [
        target_by_id[node_id]
        for node_id in dict.fromkeys(priority_target_ids)
        if node_id in target_by_id
    ][:NEURODISCOVERY_FEEDBACK_ANCHOR_LIMIT]
    feedback_source_fanout = (
        (source, target) for source in priority_sources for target in targets
    )
    feedback_target_fanout = (
        (source, target) for target in priority_targets for source in sources
    )
    source_anchor_fanout = (
        (source, target) for source in source_head for target in target_depth
    )
    target_anchor_fanout = (
        (source, target) for target in target_head for source in source_depth
    )
    return itertools.chain(
        feedback_source_fanout,
        feedback_target_fanout,
        source_anchor_fanout,
        target_anchor_fanout,
        diagonal,
    )


def _kge_legal_triples(
    scorer: Any,
    case: CaseStudy,
    index: FrozenGraphIndex,
    source_id: str,
    target_id: str,
) -> list[tuple[str, str, str]]:
    """Build task-valid directed triples for one frozen endpoint pair."""

    if scorer is None:
        return []
    has_entity = getattr(scorer, "has_entity", None)
    if callable(has_entity) and not (
        has_entity(source_id) and has_entity(target_id)
    ):
        return []
    source = _endpoint(index, source_id)
    target = _endpoint(index, target_id)
    triples: list[tuple[str, str, str]] = []
    for rule in CASE_STUDY_PAIR_RULES.get(case.name, ()):
        forward = any(
            endpoint_matches_atom(source, atom, index.concepts)
            for atom in rule.left
        ) and any(
            endpoint_matches_atom(target, atom, index.concepts)
            for atom in rule.right
        )
        reverse = (
            not rule.directed
            and any(
                endpoint_matches_atom(source, atom, index.concepts)
                for atom in rule.right
            )
            and any(
                endpoint_matches_atom(target, atom, index.concepts)
                for atom in rule.left
            )
        )
        if forward:
            triples.extend(
                (source_id, predicate, target_id)
                for predicate in rule.predicates
            )
        if reverse:
            triples.extend(
                (target_id, predicate, source_id)
                for predicate in rule.predicates
            )
    return list(dict.fromkeys(triples))


def _kge_pair_score(
    scorer: Any,
    case: CaseStudy,
    index: FrozenGraphIndex,
    source_id: str,
    target_id: str,
) -> float | None:
    triples = _kge_legal_triples(scorer, case, index, source_id, target_id)
    if not triples:
        return None
    return max(scorer.score_batch(triples), default=0.5)


def _batch_kge_pair_scores(
    scorer: Any,
    case: CaseStudy,
    index: FrozenGraphIndex,
    pairs: Iterable[tuple[str, str]],
    *,
    batch_size: int = 65_536,
) -> dict[tuple[str, str], float]:
    """Score unique task-valid endpoint pairs in a few GPU batches."""

    if scorer is None:
        return {}
    triples: list[tuple[str, str, str]] = []
    owners: list[tuple[str, str]] = []
    for pair in dict.fromkeys(pairs):
        legal = _kge_legal_triples(
            scorer,
            case,
            index,
            pair[0],
            pair[1],
        )
        triples.extend(legal)
        owners.extend([pair] * len(legal))
    values: list[float] = []
    for start in range(0, len(triples), max(1, batch_size)):
        values.extend(
            scorer.score_batch(triples[start : start + max(1, batch_size)])
        )
    result: dict[tuple[str, str], float] = {}
    for pair, value in zip(owners, values, strict=False):
        result[pair] = max(result.get(pair, 0.0), float(value))
    return result


def _blend_kge_score(base_score: float, kge_score: float | None, weight: float) -> float:
    if kge_score is None or weight <= 0.0:
        return base_score
    return (1.0 - weight) * base_score + weight * kge_score


def _context_endpoint_evidence(
    *,
    index: FrozenGraphIndex,
    adjacency: Mapping[str, Iterable[FrozenEdge]],
    node_id: str,
    freeze_year: int,
    canonical_quality_weight: float,
) -> EndpointEvidence | None:
    """Summarize one catalog endpoint from frozen semantic context only."""

    incident = list(
        {
            _edge_key(edge): edge
            for edge in adjacency.get(node_id, ())
        }.values()
    )
    if not incident:
        canonical_quality = _endpoint_canonical_quality(index, node_id)
        score = _apply_endpoint_quality_prior(
            0.05 + 0.01 * _stable_unit("catalog-endpoint", node_id),
            canonical_quality,
            canonical_quality_weight,
        )
        return EndpointEvidence(
            node_id=node_id,
            score=score,
            best_edge=FrozenEdge(
                source_id=node_id,
                target_id=node_id,
                relation="catalog_member",
                confidence=0.15,
            ),
            paper_keys=(),
            canonical_quality=canonical_quality,
        )
    incident.sort(
        key=lambda edge: (
            -edge.confidence,
            -(edge.year or 0),
            edge.claim_id,
            edge.source_id,
            edge.target_id,
        )
    )
    years = [edge.year for edge in incident if edge.year is not None]
    recency = (
        max(0.0, min(1.0, max(years) / float(freeze_year)))
        if years
        else 0.0
    )
    paper_keys = tuple(
        dict.fromkeys(_paper_key(edge) for edge in incident if _paper_key(edge))
    )
    support = min(1.0, math.log1p(len(incident)) / math.log(33.0))
    paper_support = min(1.0, math.log1p(len(paper_keys)) / math.log(17.0))
    confidence = sum(edge.confidence for edge in incident[:8]) / min(8, len(incident))
    context_score = (
        0.45 * confidence
        + 0.25 * recency
        + 0.20 * support
        + 0.10 * paper_support
    )
    canonical_quality = _endpoint_canonical_quality(index, node_id)
    score = _apply_endpoint_quality_prior(
        0.60 * context_score,
        canonical_quality,
        canonical_quality_weight,
    )
    return EndpointEvidence(
        node_id=node_id,
        score=score,
        best_edge=incident[0],
        paper_keys=paper_keys,
        canonical_quality=canonical_quality,
    )


def _kge_catalog_ids(
    *,
    scorer: Any,
    index: FrozenGraphIndex,
    atoms: Iterable[Atom],
    context_adjacency: Mapping[str, Iterable[FrozenEdge]],
) -> list[str]:
    cached = getattr(index, "_catalog_nodes_by_atom", {})
    return sorted(
        {
            node_id
            for atom in atoms
            for node_id in cached.get(atom.value, ())
            if node_id in index.concepts
            and not node_id.startswith("CLAIM_ENTITY:")
            and scorer.has_entity(node_id)
        }
    )


def _rank_stratified_kge_anchor_ids(
    rows: Iterable[EndpointEvidence],
    *,
    limit: int,
    case_study_id: str,
    anchor_side: str,
    replicate_index: int,
    generation_round: int,
) -> list[str]:
    """Keep strong KGE anchors and sample complementary frozen-rank bands."""

    node_ids = list(dict.fromkeys(row.node_id for row in rows))
    if limit <= 0 or not node_ids:
        return []
    if len(node_ids) <= limit:
        return node_ids

    protected_count = min(
        limit,
        NEURODISCOVERY_KGE_ANCHOR_PROTECTED_HEAD,
    )
    protected = node_ids[:protected_count]
    remaining = node_ids[protected_count:]
    sample_count = min(limit - protected_count, len(remaining))
    if sample_count <= 0:
        return protected

    sampled: list[str] = []
    for band_index in range(sample_count):
        start = band_index * len(remaining) // sample_count
        stop = (band_index + 1) * len(remaining) // sample_count
        band = remaining[start:stop]
        if not band:
            continue
        base_offset = int(
            _stable_unit(
                "kge-anchor-band-base",
                case_study_id,
                anchor_side,
                band_index,
            )
            * len(band)
        )
        offset = _round_rotated_band_offset(
            base_offset=base_offset,
            band_size=len(band),
            replicate_index=replicate_index,
            generation_round=generation_round,
        )
        sampled.append(band[offset])
    return [*protected, *sampled][:limit]


def _rank_stratified_kge_rows(
    rows: Iterable[tuple[float, str, str]],
    *,
    anchor_side: str,
    seed: int,
    case_study_id: str,
    replicate_index: int,
    generation_round: int,
) -> list[tuple[float, str, str]]:
    """Keep strong neighbours plus complementary replicate/round shards."""

    grouped: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    anchor_index = 1 if anchor_side == "source" else 2
    for row in rows:
        grouped[row[anchor_index]].append(row)
    selected: list[tuple[float, str, str]] = []
    for anchor_id in sorted(grouped):
        ranked = sorted(
            grouped[anchor_id],
            key=lambda row: (-row[0], row[1], row[2]),
        )
        selected.extend(ranked[:4])
        for start in range(0, len(ranked), NEURODISCOVERY_KGE_NEIGHBOR_BAND):
            band = ranked[start : start + NEURODISCOVERY_KGE_NEIGHBOR_BAND]
            if not band:
                continue
            base_offset = int(
                _stable_unit(
                    "kge-neighbour-band-base",
                    case_study_id,
                    anchor_side,
                    anchor_id,
                    start,
                )
                * len(band)
            )
            offset = _round_rotated_band_offset(
                base_offset=base_offset,
                band_size=len(band),
                replicate_index=replicate_index,
                generation_round=generation_round,
            )
            selected.append(band[offset])
    return selected


def _round_rotated_band_offset(
    *,
    base_offset: int,
    band_size: int,
    replicate_index: int,
    generation_round: int,
) -> int:
    """Rotate every closed-loop round through a new member of a rank band."""

    if band_size <= 1:
        return 0
    stride = NEURODISCOVERY_ROUND_SHARD_STRIDE % band_size
    if stride == 0:
        stride = 1
    while math.gcd(stride, band_size) != 1:
        stride = (stride + 1) % band_size or 1
    return (
        int(base_offset)
        + int(replicate_index)
        + int(generation_round) * stride
    ) % band_size


def _kge_retrieval_quota_rows(
    high_score_rows: Iterable[tuple[float, str, str]],
    stratified_rows: Iterable[tuple[float, str, str]],
    *,
    top_k: int,
) -> tuple[list[tuple[float, str, str]], list[tuple[float, str, str]]]:
    """Keep a bounded high-score head and a coverage-oriented KGE tail."""

    if top_k <= 0:
        return [], []

    def dedupe(rows: Iterable[tuple[float, str, str]]) -> list[tuple[float, str, str]]:
        result: list[tuple[float, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (row[1], row[2])
            if key in seen:
                continue
            seen.add(key)
            result.append(row)
        return result

    high = sorted(
        dedupe(high_score_rows),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    stratified = dedupe(stratified_rows)
    stratified_target = min(
        len(stratified),
        max(1, int(round(top_k * NEURODISCOVERY_KGE_STRATIFIED_FRACTION))),
    )
    high_target = max(0, top_k - stratified_target)
    selected_high = high[:high_target]
    selected_high_keys = {(row[1], row[2]) for row in selected_high}
    stratified = [
        row for row in stratified if (row[1], row[2]) not in selected_high_keys
    ]
    # Two linear passes prioritize rows that add both endpoints, then one new
    # endpoint. This avoids a costly greedy scan over every frozen KGE neighbour.
    covered_sources: set[str] = set()
    covered_targets: set[str] = set()
    coverage_order: list[tuple[float, str, str]] = []
    remaining = stratified
    for minimum_new in (2, 1, 0):
        next_remaining: list[tuple[float, str, str]] = []
        for row in remaining:
            new_count = int(row[1] not in covered_sources) + int(
                row[2] not in covered_targets
            )
            if new_count < minimum_new:
                next_remaining.append(row)
                continue
            coverage_order.append(row)
            covered_sources.add(row[1])
            covered_targets.add(row[2])
        remaining = next_remaining

    stratified_target = min(len(coverage_order), stratified_target)
    selected_stratified = coverage_order[:stratified_target]
    if len(selected_high) + len(selected_stratified) < top_k:
        used = {
            (row[1], row[2]) for row in (*selected_high, *selected_stratified)
        }
        for row in (*high[high_target:], *coverage_order[stratified_target:]):
            if (row[1], row[2]) in used:
                continue
            selected_high.append(row)
            used.add((row[1], row[2]))
            if len(selected_high) + len(selected_stratified) >= top_k:
                break
    return selected_high, selected_stratified


def _kge_guided_endpoint_pair_rows(
    *,
    scorer: Any,
    case: CaseStudy,
    index: FrozenGraphIndex,
    sources: list[EndpointEvidence],
    targets: list[EndpointEvidence],
    source_tail_rows: list[EndpointEvidence],
    target_tail_rows: list[EndpointEvidence],
    freeze_year: int,
    canonical_quality_weight: float,
    seed: int,
    replicate_index: int,
    generation_round: int,
    top_k: int,
) -> list[tuple[EndpointEvidence, EndpointEvidence, float]]:
    """Retrieve one-sided broad frozen-KG pairs before candidate truncation."""

    if scorer is None or not hasattr(scorer, "top_pair_scores") or top_k <= 0:
        return []
    predicates = sorted(
        {
            predicate
            for rule in CASE_STUDY_PAIR_RULES.get(case.name, ())
            for predicate in rule.predicates
        }
    )
    if not predicates:
        return []
    context_adjacency = getattr(index, "_global_context_adjacency", {})
    routes = _generation_endpoint_atom_routes(case)
    source_atoms = tuple(dict.fromkeys(source for source, _ in routes))
    target_atoms = tuple(dict.fromkeys(target for _, target in routes))
    source_by_id = {row.node_id: row for row in sources}
    target_by_id = {row.node_id: row for row in targets}
    source_anchors = _rank_stratified_kge_anchor_ids(
        sources,
        limit=NEURODISCOVERY_KGE_ANCHOR_LIMIT,
        case_study_id=case.name,
        anchor_side="source",
        replicate_index=replicate_index,
        generation_round=generation_round,
    )
    target_anchors = _rank_stratified_kge_anchor_ids(
        targets,
        limit=NEURODISCOVERY_KGE_ANCHOR_LIMIT,
        case_study_id=case.name,
        anchor_side="target",
        replicate_index=replicate_index,
        generation_round=generation_round,
    )
    full_sources = _kge_catalog_ids(
        scorer=scorer,
        index=index,
        atoms=source_atoms,
        context_adjacency=context_adjacency,
    )
    full_targets = _kge_catalog_ids(
        scorer=scorer,
        index=index,
        atoms=target_atoms,
        context_adjacency=context_adjacency,
    )
    tail_source_anchors = _rank_stratified_kge_anchor_ids(
        source_tail_rows,
        limit=NEURODISCOVERY_KGE_TAIL_ANCHOR_LIMIT,
        case_study_id=case.name,
        anchor_side="tail_source",
        replicate_index=replicate_index,
        generation_round=generation_round,
    )
    tail_target_anchors = _rank_stratified_kge_anchor_ids(
        target_tail_rows,
        limit=NEURODISCOVERY_KGE_TAIL_ANCHOR_LIMIT,
        case_study_id=case.name,
        anchor_side="tail_target",
        replicate_index=replicate_index,
        generation_round=generation_round,
    )
    source_fanout = scorer.top_pair_scores(
        source_anchors,
        predicates,
        full_targets,
        top_k=max(
            1,
            len(source_anchors) * NEURODISCOVERY_KGE_NEIGHBOR_LIMIT
            + len(full_targets),
        ),
        per_source_k=NEURODISCOVERY_KGE_NEIGHBOR_LIMIT,
        per_target_k=1,
    )
    target_fanin = scorer.top_pair_scores(
        full_sources,
        predicates,
        target_anchors,
        top_k=max(
            1,
            len(target_anchors) * NEURODISCOVERY_KGE_NEIGHBOR_LIMIT
            + len(full_sources),
        ),
        per_source_k=1,
        per_target_k=NEURODISCOVERY_KGE_NEIGHBOR_LIMIT,
    )
    # The deep retrieval above gives a small trusted head many alternatives.
    # A second shallow pass lets a much wider frozen-evidence tail expose only
    # its nearest neighbours.  This recovers strong long-tail analogies without
    # evaluating or retaining the full endpoint Cartesian product.
    tail_source_fanout = scorer.top_pair_scores(
        tail_source_anchors,
        predicates,
        [row.node_id for row in targets],
        top_k=max(
            1,
            len(tail_source_anchors) * NEURODISCOVERY_KGE_TAIL_NEIGHBOR_LIMIT,
        ),
        per_source_k=NEURODISCOVERY_KGE_TAIL_NEIGHBOR_LIMIT,
    )
    tail_target_fanin = scorer.top_pair_scores(
        [row.node_id for row in sources],
        predicates,
        tail_target_anchors,
        top_k=max(
            1,
            len(tail_target_anchors) * NEURODISCOVERY_KGE_TAIL_NEIGHBOR_LIMIT,
        ),
        per_target_k=NEURODISCOVERY_KGE_TAIL_NEIGHBOR_LIMIT,
    )
    stratified = [
        *_rank_stratified_kge_rows(
            source_fanout,
            anchor_side="source",
            seed=seed,
            case_study_id=case.name,
            replicate_index=replicate_index,
            generation_round=generation_round,
        ),
        *_rank_stratified_kge_rows(
            target_fanin,
            anchor_side="target",
            seed=seed,
            case_study_id=case.name,
            replicate_index=replicate_index,
            generation_round=generation_round,
        ),
        *_rank_stratified_kge_rows(
            tail_source_fanout,
            anchor_side="source",
            seed=seed,
            case_study_id=case.name,
            replicate_index=replicate_index,
            generation_round=generation_round,
        ),
        *_rank_stratified_kge_rows(
            tail_target_fanin,
            anchor_side="target",
            seed=seed,
            case_study_id=case.name,
            replicate_index=replicate_index,
            generation_round=generation_round,
        ),
    ]
    high, stratified = _kge_retrieval_quota_rows(
        [
            *source_fanout,
            *target_fanin,
            *tail_source_fanout,
            *tail_target_fanin,
        ],
        stratified,
        top_k=top_k,
    )
    candidate_ids: list[tuple[str, str]] = []
    stratified_ids: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()

    def append_legal(
        rows: Iterable[tuple[float, str, str]],
        *,
        stratified_quota: bool,
    ) -> None:
        for _score, source_id, target_id in rows:
            if len(candidate_ids) >= top_k:
                break
            source = source_by_id.get(source_id)
            if source is None:
                source = _context_endpoint_evidence(
                    index=index,
                    adjacency=context_adjacency,
                    node_id=source_id,
                    freeze_year=freeze_year,
                    canonical_quality_weight=canonical_quality_weight,
                )
                if source is not None:
                    source_by_id[source_id] = source
            target = target_by_id.get(target_id)
            if target is None:
                target = _context_endpoint_evidence(
                    index=index,
                    adjacency=context_adjacency,
                    node_id=target_id,
                    freeze_year=freeze_year,
                    canonical_quality_weight=canonical_quality_weight,
                )
                if target is not None:
                    target_by_id[target_id] = target
            if source is None or target is None:
                continue
            key = (source_id, target_id)
            if (
                key in seen
                or not case_study_endpoint_names_allowed(
                    case.name,
                    _endpoint(index, source_id),
                    _endpoint(index, target_id),
                )
                or not _kge_legal_triples(
                    scorer,
                    case,
                    index,
                    source_id,
                    target_id,
                )
            ):
                continue
            seen.add(key)
            candidate_ids.append(key)
            if stratified_quota:
                stratified_ids.add(key)

    append_legal(high, stratified_quota=False)
    append_legal(stratified, stratified_quota=True)
    if len(candidate_ids) < top_k:
        append_legal(
            [
                *source_fanout,
                *target_fanin,
                *tail_source_fanout,
                *tail_target_fanin,
                *stratified,
            ],
            stratified_quota=False,
        )
    exact_scores = _batch_kge_pair_scores(
        scorer,
        case,
        index,
        candidate_ids,
    )
    high_ids = sorted(
        (pair for pair in candidate_ids if pair not in stratified_ids),
        key=lambda pair: (-exact_scores.get(pair, 0.5), pair[0], pair[1]),
    )
    stratified_ids_ordered = [
        pair for pair in candidate_ids if pair in stratified_ids
    ]
    ordered_ids = [*high_ids, *stratified_ids_ordered]
    return [
        (source_by_id[source_id], target_by_id[target_id], exact_scores[pair])
        for pair in ordered_ids[:top_k]
        for source_id, target_id in (pair,)
        if pair in exact_scores
    ]


def _endpoint_rank_band(rank: int) -> int:
    return max(0, rank) // NEURODISCOVERY_EXPLORATION_RANK_BIN_SIZE


def _ordered_neurodiscovery_exploration_buckets(
    buckets: Iterable[tuple[str, int, int]],
    *,
    seed: int,
    case_study_id: str,
) -> list[tuple[str, int, int]]:
    """Round-robin rank bands so every strong endpoint receives exploration."""

    grouped: dict[tuple[str, int], list[tuple[str, int, int]]] = defaultdict(list)
    for bucket in buckets:
        grouped[(bucket[0], bucket[1])].append(bucket)
    groups = sorted(
        grouped,
        key=lambda group: (
            _stable_unit("frontier-exploration-group", seed, case_study_id, *group),
            group,
        ),
    )
    ordered_groups: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
    for group in groups:
        ordered_groups[group] = sorted(
            grouped[group],
            key=lambda bucket: (
                _stable_unit(
                    "frontier-exploration-band",
                    seed,
                    case_study_id,
                    *bucket,
                ),
                bucket[2],
            ),
        )

    ordered: list[tuple[str, int, int]] = []
    max_bands = max((len(rows) for rows in ordered_groups.values()), default=0)
    for band_index in range(max_bands):
        for group in groups:
            rows = ordered_groups[group]
            if band_index < len(rows):
                ordered.append(rows[band_index])
    return ordered


def _stratified_anchor_fanout_rows(
    buckets: Mapping[tuple[str, int, int], Iterable[tuple]],
    *,
    anchor_side: str,
    max_anchors: int = NEURODISCOVERY_STRATIFIED_FANOUT_ANCHORS,
    rows_per_anchor: int = NEURODISCOVERY_STRATIFIED_FANOUT_ROWS_PER_ANCHOR,
) -> list[tuple]:
    """Expose complementary rank bands for a small trusted endpoint head.

    Marginal endpoint coverage can pair every endpoint with the wrong counterpart.
    This bounded two-dimensional fan-out gives each strong anchor one candidate
    from several opposite-side evidence bands, using only frozen ranks.
    """

    if max_anchors <= 0 or rows_per_anchor <= 0:
        return []
    grouped: dict[int, dict[int, list[tuple]]] = defaultdict(dict)
    for (side, anchor_rank, band_rank), rows in buckets.items():
        if side != anchor_side or anchor_rank >= max_anchors:
            continue
        candidate_rows = [
            entry[4]
            if len(entry) == 5
            and isinstance(entry[4], tuple)
            and len(entry[4]) >= 6
            else entry
            for entry in rows
        ]
        grouped[anchor_rank][band_rank] = sorted(
            candidate_rows,
            key=lambda row: (-float(row[0]), str(row[1]), str(row[2])),
        )

    selected: list[tuple] = []
    for anchor_rank in sorted(grouped):
        bands = grouped[anchor_rank]
        ordered_bands = sorted(bands)
        if not ordered_bands:
            continue
        for position in range(min(rows_per_anchor, len(ordered_bands))):
            band_index = position * len(ordered_bands) // min(
                rows_per_anchor,
                len(ordered_bands),
            )
            selected.append(bands[ordered_bands[band_index]][0])
    return selected


def _dedupe_endpoint_candidate_rows(rows: Iterable[tuple]) -> list[tuple]:
    unique: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = _pair(str(row[1]), str(row[2]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _coverage_first_endpoint_rows(
    rows: Iterable[tuple],
    target_count: int,
) -> list[tuple]:
    """Select frozen candidates that expose the widest directed endpoint set."""

    remaining = _dedupe_endpoint_candidate_rows(rows)
    selected: list[tuple] = []
    seen_sources: set[str] = set()
    seen_targets: set[str] = set()
    while remaining and len(selected) < target_count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                int(str(remaining[index][1]) not in seen_sources)
                + int(str(remaining[index][2]) not in seen_targets),
                float(remaining[index][0]),
                str(remaining[index][1]),
                str(remaining[index][2]),
            ),
        )
        row = remaining.pop(best_index)
        selected.append(row)
        seen_sources.add(str(row[1]))
        seen_targets.add(str(row[2]))
    return selected


def _interleave_neurodiscovery_exploration(
    exploitation: list[tuple],
    exploration: list[tuple],
    *,
    target_count: int,
    pool_cap: int,
    kge_rows: Iterable[tuple] = (),
    stratified_fanout_rows: Iterable[tuple] = (),
) -> list[tuple]:
    """Reserve rank-stratified and frozen-KGE rows in the executable prefix."""

    prefix_size = min(target_count, pool_cap)
    if prefix_size <= 0:
        return []

    unique_kge = _dedupe_endpoint_candidate_rows(kge_rows)
    desired_reserved = min(
        prefix_size,
        int(round(prefix_size * NEURODISCOVERY_EXPLORATION_FRACTION)),
    )
    desired_kge = min(
        len(unique_kge),
        desired_reserved,
        max(
            1 if unique_kge else 0,
            int(round(prefix_size * NEURODISCOVERY_KGE_EXECUTABLE_FRACTION)),
        ),
    )
    kge_high_target = max(0, desired_kge - int(round(desired_kge * 0.20)))
    selected_kge = sorted(
        unique_kge,
        key=lambda row: (-float(row[0]), str(row[1]), str(row[2])),
    )[:kge_high_target]
    selected_kge_keys = {
        _pair(str(row[1]), str(row[2])) for row in selected_kge
    }
    selected_kge.extend(
        _coverage_first_endpoint_rows(
            [
                row
                for row in unique_kge
                if _pair(str(row[1]), str(row[2])) not in selected_kge_keys
            ],
            desired_kge - len(selected_kge),
        )
    )
    selected_kge_keys = {
        _pair(str(row[1]), str(row[2])) for row in selected_kge
    }

    unique_fanout = [
        row
        for row in _dedupe_endpoint_candidate_rows(stratified_fanout_rows)
        if _pair(str(row[1]), str(row[2])) not in selected_kge_keys
    ]
    desired_fanout = min(
        len(unique_fanout),
        max(
            0,
            min(
                desired_reserved - len(selected_kge),
                int(round(prefix_size * 0.10)),
            ),
        ),
    )
    selected_fanout = _coverage_first_endpoint_rows(
        unique_fanout,
        desired_fanout,
    )
    selected_fanout_keys = {
        _pair(str(row[1]), str(row[2])) for row in selected_fanout
    }

    unique_exploration = [
        row
        for row in _dedupe_endpoint_candidate_rows(exploration)
        if _pair(str(row[1]), str(row[2])) not in selected_kge_keys
        and _pair(str(row[1]), str(row[2])) not in selected_fanout_keys
    ]
    desired_exploration = min(
        len(unique_exploration),
        max(0, desired_reserved - len(selected_kge) - len(selected_fanout)),
    )
    selected_exploration = _coverage_first_endpoint_rows(
        unique_exploration,
        desired_exploration,
    )
    non_kge_rows = [*selected_fanout, *selected_exploration]
    reserved_rows: list[tuple] = []
    if selected_kge and non_kge_rows:
        total_reserved = len(selected_kge) + len(non_kge_rows)
        kge_positions = {
            min(
                total_reserved - 1,
                round((index + 1) * (total_reserved + 1) / (len(selected_kge) + 1))
                - 1,
            )
            for index in range(len(selected_kge))
        }
        kge_index = 0
        exploration_index = 0
        for position in range(total_reserved):
            if position in kge_positions and kge_index < len(selected_kge):
                reserved_rows.append(selected_kge[kge_index])
                kge_index += 1
            elif exploration_index < len(non_kge_rows):
                reserved_rows.append(non_kge_rows[exploration_index])
                exploration_index += 1
            elif kge_index < len(selected_kge):
                reserved_rows.append(selected_kge[kge_index])
                kge_index += 1
    else:
        reserved_rows = [*non_kge_rows, *selected_kge]

    reserved_keys = {
        _pair(str(row[1]), str(row[2])) for row in reserved_rows
    }
    selected_exploitation = [
        row
        for row in exploitation
        if _pair(str(row[1]), str(row[2])) not in reserved_keys
    ][: max(0, prefix_size - len(reserved_rows))]

    if len(selected_exploitation) + len(reserved_rows) < prefix_size:
        used = {
            _pair(str(row[1]), str(row[2]))
            for row in (*selected_exploitation, *reserved_rows)
        }
        for row in (*exploitation, *unique_kge, *unique_fanout, *unique_exploration):
            key = _pair(str(row[1]), str(row[2]))
            if key in used:
                continue
            selected_exploitation.append(row)
            used.add(key)
            if len(selected_exploitation) + len(reserved_rows) >= prefix_size:
                break

    prefix: list[tuple] = []
    if reserved_rows and selected_exploitation:
        exploration_positions = {
            min(
                prefix_size - 1,
                round(
                    (index + 1)
                    * (prefix_size + 1)
                    / (len(reserved_rows) + 1)
                )
                - 1,
            )
            for index in range(len(reserved_rows))
        }
        for position in range(prefix_size - 1, -1, -1):
            if len(exploration_positions) >= len(reserved_rows):
                break
            exploration_positions.add(position)
        exploit_index = 0
        explore_index = 0
        for position in range(prefix_size):
            if (
                position in exploration_positions
                and explore_index < len(reserved_rows)
            ):
                prefix.append(reserved_rows[explore_index])
                explore_index += 1
            elif exploit_index < len(selected_exploitation):
                prefix.append(selected_exploitation[exploit_index])
                exploit_index += 1
            elif explore_index < len(reserved_rows):
                prefix.append(reserved_rows[explore_index])
                explore_index += 1
    else:
        prefix = [*selected_exploitation, *reserved_rows]

    used = {_pair(str(row[1]), str(row[2])) for row in prefix}
    remainder: list[tuple] = []
    for row in (*exploitation, *unique_kge, *unique_fanout, *unique_exploration):
        key = _pair(str(row[1]), str(row[2]))
        if key in used:
            continue
        remainder.append(row)
        used.add(key)
        if len(prefix) + len(remainder) >= pool_cap:
            break
    return [*prefix, *remainder][:pool_cap]


def _candidate_pool_node_limit(target_count: int) -> int:
    return max(96, min(800, int(math.ceil(math.sqrt(max(1, target_count * 64)))) * 3))


def _shared_neighbor_score(
    adjacency: dict[str, list[FrozenEdge]],
    left_id: str,
    right_id: str,
    cache: dict[str, frozenset[str]],
) -> float:
    def neighbours(node_id: str) -> frozenset[str]:
        if node_id not in cache:
            cache[node_id] = frozenset(
                edge.other(node_id) for edge in adjacency.get(node_id, ())
            )
        return cache[node_id]

    shared = len(neighbours(left_id) & neighbours(right_id))
    return min(1.0, math.log1p(shared) / math.log(9.0))


def _shared_paper_score(left: EndpointEvidence, right: EndpointEvidence) -> float:
    left_papers = set(left.paper_keys)
    right_papers = set(right.paper_keys)
    if not left_papers or not right_papers:
        return 0.0
    shared = len(left_papers & right_papers)
    if not shared:
        return 0.0
    return min(1.0, math.log1p(shared) / math.log(5.0))


_CONTEXT_STOP_TOKENS = frozenset({
    "associated",
    "baseline",
    "brain",
    "clinical",
    "effect",
    "feature",
    "future",
    "imaging",
    "marker",
    "model",
    "outcome",
    "patient",
    "predict",
    "risk",
    "score",
    "study",
})


def _semantic_context_score(
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    left_id: str,
    right_id: str,
    cache: dict[str, tuple[frozenset[str], frozenset[str]]],
) -> float:
    """Estimate frozen disease/phenotype context agreement for two endpoints."""

    def context(node_id: str) -> tuple[frozenset[str], frozenset[str]]:
        if node_id in cache:
            return cache[node_id]
        direct = frozenset(
            entity_name_tokens(index.names.get(node_id, node_id))
            - _CONTEXT_STOP_TOKENS
        )
        incident = sorted(
            adjacency.get(node_id, ()),
            key=lambda edge: (-edge.confidence, -(edge.year or 0), edge.claim_id),
        )
        expanded = set(direct)
        for edge in incident[:24]:
            expanded.update(
                entity_name_tokens(index.names.get(edge.other(node_id), edge.other(node_id)))
                - _CONTEXT_STOP_TOKENS
            )
        result = (direct, frozenset(expanded))
        cache[node_id] = result
        return result

    left_direct, left_context = context(left_id)
    right_direct, right_context = context(right_id)

    def containment(container: frozenset[str], query: frozenset[str]) -> float:
        if not query:
            return 0.0
        return len(container & query) / len(query)

    direct_alignment = max(
        containment(left_context, right_direct),
        containment(right_context, left_direct),
    )
    shared_context = (
        len(left_context & right_context) / min(len(left_context), len(right_context))
        if left_context and right_context
        else 0.0
    )
    return min(1.0, 0.80 * direct_alignment + 0.20 * shared_context)


def _ranked_endpoint_compositions(
    *,
    method: str,
    case: CaseStudy,
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    target_count: int,
    seed: int,
    endpoint_canonical_quality_weight: float | None = None,
    feedback_anchor_pairs: tuple[tuple[str, str], ...] = (),
    feedback_anchor_fraction: float = 0.0,
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
    replicate_index: int = 0,
    generation_round: int = 0,
) -> list[tuple[float, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]]]:
    """Compose legal endpoint pairs when a strict two-hop bridge is unavailable.

    OpenScholar-style systems can still propose a hypothesis after retrieving
    separate papers about each endpoint.  This fallback keeps that behaviour
    explicit, frozen, and auditable instead of padding the fixed budget with
    generation failures.
    """

    if case.task is None or len(case.task.inputs) != 1:
        return []
    coverage_seed = (
        int(replicate_index)
        + NEURODISCOVERY_ROUND_SHARD_STRIDE * int(generation_round)
        if method == "neurodiscovery"
        else seed
    )
    limit = _candidate_pool_node_limit(target_count)
    context_adjacency = (
        getattr(index, "_global_context_adjacency", None)
        if method == "neurodiscovery"
        else None
    )
    canonical_quality_weight = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
        if endpoint_canonical_quality_weight is None and method == "neurodiscovery"
        else float(endpoint_canonical_quality_weight or 0.0)
    )
    evidence_tail_fraction = (
        NEURODISCOVERY_EVIDENCE_TAIL_FRACTION
        if method == "neurodiscovery"
        else 0.0
    )
    routes = _generation_endpoint_atom_routes(case)
    source_atoms = tuple(dict.fromkeys(source for source, _ in routes))
    target_atoms = tuple(dict.fromkeys(target for _, target in routes))
    ranked_source_evidence: list[EndpointEvidence] = []
    ranked_target_evidence: list[EndpointEvidence] = []
    sources = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=source_atoms,
        freeze_year=freeze_year,
        limit=limit,
        case_study_id=case.name,
        context_adjacency=context_adjacency,
        canonical_quality_weight=canonical_quality_weight,
        evidence_tail_fraction=evidence_tail_fraction,
        seed=coverage_seed,
        pool_label="source",
        required_node_ids=tuple(
            source_id for source_id, _ in feedback_anchor_pairs if source_id
        ),
        ranked_evidence_out=ranked_source_evidence,
    )
    targets = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=target_atoms,
        freeze_year=freeze_year,
        limit=limit,
        case_study_id=case.name,
        context_adjacency=context_adjacency,
        canonical_quality_weight=canonical_quality_weight,
        evidence_tail_fraction=evidence_tail_fraction,
        seed=coverage_seed,
        pool_label="target",
        required_node_ids=tuple(
            target_id for _, target_id in feedback_anchor_pairs if target_id
        ),
        ranked_evidence_out=ranked_target_evidence,
    )
    if not sources or not targets:
        return []

    pool_cap = max(5000, target_count * 16)
    scoring_adjacency = context_adjacency or adjacency
    neighbor_cache: dict[str, frozenset[str]] = {}
    semantic_context_cache: dict[
        str, tuple[frozenset[str], frozenset[str]]
    ] = {}
    endpoint_coverage_rows: dict[tuple[str, str], tuple] = {}
    kge_exploration_rows: list[tuple] = []
    kge_scores: dict[tuple[str, str], float] = {}
    candidates: dict[
        tuple[str, str],
        tuple[float, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]],
    ] = {}
    feedback_candidates: dict[
        tuple[str, str],
        tuple[float, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]],
    ] = {}
    candidate_heap: list[
        tuple[
            float,
            str,
            str,
            int,
            tuple[float, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]],
        ]
    ] = []
    exploration_heaps: dict[
        tuple[str, int, int],
        list[
            tuple[
                float,
                str,
                str,
                int,
                tuple[float, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]],
            ]
        ],
    ] = defaultdict(list)
    source_ranks = {row.node_id: rank for rank, row in enumerate(sources)}
    target_ranks = {row.node_id: rank for rank, row in enumerate(targets)}
    # Strong-anchor fan-outs cover plausible hypotheses with one deeply ranked
    # endpoint. Seeded diagonal traversal then exposes a different long tail on
    # each run without materialising the full Cartesian product.
    if method == "neurodiscovery":
        kge_guided = _kge_guided_endpoint_pair_rows(
            scorer=kge_scorer,
            case=case,
            index=index,
            sources=sources,
            targets=targets,
            source_tail_rows=ranked_source_evidence,
            target_tail_rows=ranked_target_evidence,
            freeze_year=freeze_year,
            canonical_quality_weight=canonical_quality_weight,
            seed=coverage_seed,
            replicate_index=replicate_index,
            generation_round=generation_round,
            top_k=min(pool_cap, max(5000, target_count * 4)),
        )
        kge_scores = {
            (source.node_id, target.node_id): score
            for source, target, score in kge_guided
        }
        for source, target, _score in kge_guided:
            source_ranks.setdefault(source.node_id, len(source_ranks))
            target_ranks.setdefault(target.node_id, len(target_ranks))
        pair_rows = itertools.chain(
            ((source, target) for source, target, _ in kge_guided),
            _neurodiscovery_endpoint_pair_rows(
                sources,
                targets,
                pool_cap=pool_cap,
                seed=coverage_seed,
                case_study_id=case.name,
                priority_source_ids=tuple(
                    source_id for source_id, _ in feedback_anchor_pairs if source_id
                ),
                priority_target_ids=tuple(
                    target_id for _, target_id in feedback_anchor_pairs if target_id
                ),
            ),
        )
    else:
        pair_rows = (
            (source, targets[(source_index + offset) % len(targets)])
            for offset in range(len(targets))
            for source_index, source in enumerate(sources)
        )
    seen_candidate_pairs: set[tuple[str, str]] = set()
    for source, target in pair_rows:
            if source.node_id == target.node_id:
                continue
            key = _pair(source.node_id, target.node_id)
            if key in index.direct_pairs or key in seen_candidate_pairs:
                continue
            source_endpoint = _endpoint(index, source.node_id)
            target_endpoint = _endpoint(index, target.node_id)
            if not _endpoint_route_allowed(
                index,
                source_endpoint,
                target_endpoint,
                routes,
            ):
                continue
            if not case_study_endpoint_names_allowed(
                case.name,
                source_endpoint,
                target_endpoint,
            ):
                continue
            seen_candidate_pairs.add(key)
            structural = _shared_neighbor_score(
                scoring_adjacency,
                source.node_id,
                target.node_id,
                neighbor_cache,
            )
            jitter = _stable_unit(
                "frontier",
                method,
                seed,
                case.name,
                source.node_id,
                target.node_id,
            )
            endpoint_support = (source.score + target.score) / 2.0
            if method == "sciagents":
                score = 0.55 * endpoint_support + 0.35 * structural + 0.10 * jitter
            elif method == "neurodiscovery":
                paper_context = _shared_paper_score(source, target)
                semantic_context = _semantic_context_score(
                    index,
                    scoring_adjacency,
                    source.node_id,
                    target.node_id,
                    semantic_context_cache,
                )
                score = (
                    0.30 * endpoint_support
                    + 0.20 * structural
                    + 0.20 * paper_context
                    + 0.25 * semantic_context
                    + 0.05 * jitter
                )
                score = _blend_kge_score(
                    score,
                    kge_scores.get((source.node_id, target.node_id)),
                    kge_weight,
                )
            else:
                score = 0.72 * endpoint_support + 0.18 * structural + 0.10 * jitter
            papers = tuple(dict.fromkeys((*source.paper_keys, *target.paper_keys)))
            row = (
                score,
                source.node_id,
                target.node_id,
                source.best_edge,
                target.best_edge,
                papers,
            )
            if (source.node_id, target.node_id) in kge_scores:
                kge_exploration_rows.append(row)
            for endpoint_key in (
                ("source", source.node_id),
                ("target", target.node_id),
            ):
                current = endpoint_coverage_rows.get(endpoint_key)
                if current is None or (row[0], row[1], row[2]) > (
                    current[0],
                    current[1],
                    current[2],
                ):
                    endpoint_coverage_rows[endpoint_key] = row
            if any(
                (source.node_id == anchor_source)
                != (target.node_id == anchor_target)
                for anchor_source, anchor_target in feedback_anchor_pairs
            ):
                feedback_candidates[key] = row
            if method == "neurodiscovery":
                heap_row = (score, source.node_id, target.node_id, len(candidate_heap), row)
                if len(candidate_heap) < pool_cap:
                    heapq.heappush(candidate_heap, heap_row)
                elif score > candidate_heap[0][0]:
                    heapq.heapreplace(candidate_heap, heap_row)
                source_rank = source_ranks[source.node_id]
                target_rank = target_ranks[target.node_id]
                buckets: list[tuple[str, int, int]] = []
                if source_rank < NEURODISCOVERY_ANCHOR_FANOUT_HEAD:
                    buckets.append(
                        ("source", source_rank, _endpoint_rank_band(target_rank))
                    )
                if target_rank < NEURODISCOVERY_ANCHOR_FANOUT_HEAD:
                    buckets.append(
                        ("target", target_rank, _endpoint_rank_band(source_rank))
                    )
                for bucket in buckets:
                    bucket_heap = exploration_heaps[bucket]
                    bucket_row = (
                        score,
                        source.node_id,
                        target.node_id,
                        len(bucket_heap),
                        row,
                    )
                    if len(bucket_heap) < NEURODISCOVERY_EXPLORATION_PER_BUCKET:
                        heapq.heappush(bucket_heap, bucket_row)
                    elif score > bucket_heap[0][0]:
                        heapq.heapreplace(bucket_heap, bucket_row)
            else:
                candidates[key] = row
                if len(candidates) >= pool_cap:
                    break
    if method == "neurodiscovery":
        exploitation = sorted(
            (heap_row[4] for heap_row in candidate_heap),
            key=lambda row: (-row[0], row[1], row[2]),
        )
        ordered_buckets = _ordered_neurodiscovery_exploration_buckets(
            exploration_heaps,
            seed=coverage_seed,
            case_study_id=case.name,
        )
        stratified_fanout = [
            *_stratified_anchor_fanout_rows(
                exploration_heaps,
                anchor_side="source",
            ),
            *_stratified_anchor_fanout_rows(
                exploration_heaps,
                anchor_side="target",
            ),
        ]
        exploration: list[tuple] = [
            *endpoint_coverage_rows.values(),
            *kge_exploration_rows,
        ]
        bucket_rows = {
            key: sorted(
                (heap_row[4] for heap_row in exploration_heaps[key]),
                key=lambda row: (-row[0], row[1], row[2]),
            )
            for key in ordered_buckets
        }
        for row_index in range(NEURODISCOVERY_EXPLORATION_PER_BUCKET):
            for key in ordered_buckets:
                rows = bucket_rows[key]
                if row_index < len(rows):
                    exploration.append(rows[row_index])
        ranked = _interleave_neurodiscovery_exploration(
            exploitation,
            exploration,
            target_count=target_count,
            pool_cap=pool_cap,
            kge_rows=kge_exploration_rows,
            stratified_fanout_rows=stratified_fanout,
        )
        if feedback_candidates:
            feedback_rows = sorted(
                feedback_candidates.values(),
                key=lambda row: (-row[0], row[1], row[2]),
            )
            feedback_pool_target = min(
                len(feedback_rows),
                max(1, int(round(pool_cap * feedback_anchor_fraction))),
            )
            feedback_rows = feedback_rows[:feedback_pool_target]
            feedback_keys = {
                _pair(str(row[1]), str(row[2])) for row in feedback_rows
            }
            return _dedupe_endpoint_candidate_rows(
                [
                    *feedback_rows,
                    *(
                        row
                        for row in ranked
                        if _pair(str(row[1]), str(row[2])) not in feedback_keys
                    ),
                ]
            )[:pool_cap]
        return ranked
    else:
        values = list(candidates.values())
    return sorted(values, key=lambda row: (-row[0], row[1], row[2]))


def _ranked_chain_compositions(
    *,
    method: str,
    case: CaseStudy,
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    target_count: int,
    seed: int,
    endpoint_canonical_quality_weight: float | None = None,
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
) -> list[tuple[float, str, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]]]:
    """Expand a registered three-node chain without consulting future labels."""

    if case.chain is None or len(case.chain.mediators) != 1:
        return []
    limit = _candidate_pool_node_limit(target_count)
    context_adjacency = (
        getattr(index, "_global_context_adjacency", None)
        if method == "neurodiscovery"
        else None
    )
    canonical_quality_weight = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
        if endpoint_canonical_quality_weight is None and method == "neurodiscovery"
        else float(endpoint_canonical_quality_weight or 0.0)
    )
    evidence_tail_fraction = (
        NEURODISCOVERY_EVIDENCE_TAIL_FRACTION
        if method == "neurodiscovery"
        else 0.0
    )
    sources = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(case.chain.source,),
        freeze_year=freeze_year,
        limit=limit,
        context_adjacency=context_adjacency,
        canonical_quality_weight=canonical_quality_weight,
        evidence_tail_fraction=evidence_tail_fraction,
        seed=seed,
        pool_label="chain_source",
    )
    mediators = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(case.chain.mediators[0],),
        freeze_year=freeze_year,
        limit=limit,
        context_adjacency=context_adjacency,
        canonical_quality_weight=canonical_quality_weight,
        evidence_tail_fraction=evidence_tail_fraction,
        seed=seed,
        pool_label="chain_mediator",
    )
    targets = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(case.chain.target,),
        freeze_year=freeze_year,
        limit=limit,
        context_adjacency=context_adjacency,
        canonical_quality_weight=canonical_quality_weight,
        evidence_tail_fraction=evidence_tail_fraction,
        seed=seed,
        pool_label="chain_target",
    )
    if not sources or not mediators or not targets:
        return []

    pool_cap = max(5000, target_count * 16)
    scoring_adjacency = context_adjacency or adjacency
    neighbor_cache: dict[str, frozenset[str]] = {}
    semantic_context_cache: dict[
        str, tuple[frozenset[str], frozenset[str]]
    ] = {}
    candidates: dict[
        tuple[str, str, str],
        tuple[float, str, str, str, FrozenEdge, FrozenEdge, tuple[str, ...]],
    ] = {}
    max_offsets = max(12, int(math.ceil(pool_cap / max(1, len(sources)))))
    for offset in range(max_offsets):
        for source_index, source in enumerate(sources):
            mediator = mediators[(source_index * 7 + offset) % len(mediators)]
            target = targets[(source_index * 11 + offset * 3) % len(targets)]
            node_ids = (source.node_id, mediator.node_id, target.node_id)
            if len(set(node_ids)) < 3 or _pair(source.node_id, target.node_id) in index.direct_pairs:
                continue
            if node_ids in candidates:
                continue
            if not case_study_endpoint_names_allowed(
                case.name,
                _endpoint(index, source.node_id),
                _endpoint(index, target.node_id),
            ):
                continue
            left_structure = _shared_neighbor_score(
                scoring_adjacency,
                source.node_id,
                mediator.node_id,
                neighbor_cache,
            )
            right_structure = _shared_neighbor_score(
                scoring_adjacency,
                mediator.node_id,
                target.node_id,
                neighbor_cache,
            )
            endpoint_support = (source.score + mediator.score + target.score) / 3.0
            structural = (left_structure + right_structure) / 2.0
            jitter = _stable_unit("chain-frontier", method, seed, case.name, *node_ids)
            if method == "neurodiscovery":
                left_context = _semantic_context_score(
                    index,
                    scoring_adjacency,
                    source.node_id,
                    mediator.node_id,
                    semantic_context_cache,
                )
                right_context = _semantic_context_score(
                    index,
                    scoring_adjacency,
                    mediator.node_id,
                    target.node_id,
                    semantic_context_cache,
                )
                semantic_context = (left_context + right_context) / 2.0
                score = (
                    0.55 * endpoint_support
                    + 0.20 * structural
                    + 0.20 * semantic_context
                    + 0.05 * jitter
                )
            else:
                score = 0.68 * endpoint_support + 0.22 * structural + 0.10 * jitter
            papers = tuple(
                dict.fromkeys(
                    (*source.paper_keys, *mediator.paper_keys, *target.paper_keys)
                )
            )
            candidates[node_ids] = (
                score,
                source.node_id,
                mediator.node_id,
                target.node_id,
                source.best_edge,
                target.best_edge,
                papers,
            )
            if len(candidates) >= pool_cap:
                break
        if len(candidates) >= pool_cap:
            break
    values = list(candidates.values())
    if method == "neurodiscovery" and kge_scorer is not None and kge_weight > 0.0:
        pair_scores = _batch_kge_pair_scores(
            kge_scorer,
            case,
            index,
            (
                pair
                for row in values
                for pair in ((row[1], row[2]), (row[2], row[3]))
            ),
        )
        values = [
            (
                _blend_kge_score(
                    row[0],
                    (
                        sum(available) / len(available)
                        if (
                            available := [
                                pair_scores[pair]
                                for pair in ((row[1], row[2]), (row[2], row[3]))
                                if pair in pair_scores
                            ]
                        )
                        else None
                    ),
                    kge_weight,
                ),
                *row[1:],
            )
            for row in values
        ]
    return sorted(values, key=lambda row: (-row[0], row[1], row[2], row[3]))


def _ranked_multi_input_compositions(
    *,
    method: str,
    case: CaseStudy,
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    target_count: int,
    seed: int,
    endpoint_canonical_quality_weight: float | None = None,
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
    replicate_index: int = 0,
    generation_round: int = 0,
) -> list[MultiInputCandidate]:
    """Compose distinct task inputs around an independently grounded output."""

    if case.task is None or len(case.task.inputs) < 2:
        return []
    limit = _candidate_pool_node_limit(target_count)
    context_adjacency = (
        getattr(index, "_global_context_adjacency", None)
        if method == "neurodiscovery"
        else None
    )
    canonical_quality_weight = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
        if endpoint_canonical_quality_weight is None and method == "neurodiscovery"
        else float(endpoint_canonical_quality_weight or 0.0)
    )
    evidence_tail_fraction = (
        NEURODISCOVERY_EVIDENCE_TAIL_FRACTION
        if method == "neurodiscovery"
        else 0.0
    )
    input_order = tuple(sorted(case.task.inputs, key=lambda atom: atom.value))
    input_pools = {
        atom: _endpoint_evidence_pool(
            index=index,
            adjacency=adjacency,
            atoms=(atom,),
            freeze_year=freeze_year,
            limit=limit,
            context_adjacency=context_adjacency,
            canonical_quality_weight=canonical_quality_weight,
            evidence_tail_fraction=evidence_tail_fraction,
            seed=seed,
            pool_label=f"multi_input:{atom.value}",
        )
        for atom in input_order
    }
    outputs = _endpoint_evidence_pool(
        index=index,
        adjacency=adjacency,
        atoms=(case.task.output,),
        freeze_year=freeze_year,
        limit=limit,
        context_adjacency=context_adjacency,
        canonical_quality_weight=canonical_quality_weight,
        evidence_tail_fraction=evidence_tail_fraction,
        seed=seed,
        pool_label="multi_output",
    )
    if not outputs or any(not pool for pool in input_pools.values()):
        return []

    pool_cap = max(5000, target_count * 16)
    scoring_adjacency = context_adjacency or adjacency
    neighbor_cache: dict[str, frozenset[str]] = {}
    semantic_context_cache: dict[
        str, tuple[frozenset[str], frozenset[str]]
    ] = {}
    candidates: dict[tuple[tuple[str, ...], str], MultiInputCandidate] = {}
    attempts = pool_cap * 8
    for step in range(attempts):
        output = outputs[
            int(_stable_unit("multi-output", seed, case.name, step) * len(outputs))
            % len(outputs)
        ]
        chosen: list[tuple[Atom, EndpointEvidence]] = []
        for atom_index, atom in enumerate(input_order):
            pool = input_pools[atom]
            row = pool[
                int(
                    _stable_unit(
                        "multi-input",
                        seed,
                        case.name,
                        step,
                        atom_index,
                        atom.value,
                    )
                    * len(pool)
                )
                % len(pool)
            ]
            chosen.append((atom, row))
        input_ids = tuple(row.node_id for _, row in chosen)
        if len(set((*input_ids, output.node_id))) != len(input_ids) + 1:
            continue
        key = (input_ids, output.node_id)
        if key in candidates:
            continue
        if not all(
            case_study_endpoint_names_allowed(
                case.name,
                _endpoint(index, row.node_id),
                _endpoint(index, output.node_id),
            )
            for _, row in chosen
        ):
            continue
        structures = [
            _shared_neighbor_score(
                scoring_adjacency,
                row.node_id,
                output.node_id,
                neighbor_cache,
            )
            for _, row in chosen
        ]
        endpoint_support = (
            sum(row.score for _, row in chosen) + output.score
        ) / (len(chosen) + 1)
        structural = sum(structures) / len(structures)
        jitter = _stable_unit(
            "multi-frontier",
            method,
            seed,
            case.name,
            *input_ids,
            output.node_id,
        )
        if method == "neurodiscovery":
            contexts = [
                _semantic_context_score(
                    index,
                    scoring_adjacency,
                    row.node_id,
                    output.node_id,
                    semantic_context_cache,
                )
                for _, row in chosen
            ]
            semantic_context = sum(contexts) / len(contexts)
            score = (
                0.55 * endpoint_support
                + 0.20 * structural
                + 0.20 * semantic_context
                + 0.05 * jitter
            )
        else:
            score = 0.68 * endpoint_support + 0.22 * structural + 0.10 * jitter
        papers = tuple(
            dict.fromkeys(
                paper
                for _, row in chosen
                for paper in row.paper_keys
            )
        )
        papers = tuple(dict.fromkeys((*papers, *output.paper_keys)))
        synthetic_edges = tuple(
            FrozenEdge(
                source_id=row.node_id,
                target_id=output.node_id,
                relation="predicts",
                confidence=min(1.0, (row.score + output.score) / 2.0),
            )
            for _, row in chosen
        )
        candidates[key] = MultiInputCandidate(
            score=score,
            output_id=output.node_id,
            input_bindings=tuple(
                (atom.value, row.node_id) for atom, row in chosen
            ),
            edges=synthetic_edges,
            paper_keys=papers,
            generation_mode="evidence_frontier_composition",
        )
        if len(candidates) >= pool_cap:
            break
    values = list(candidates.values())
    if method == "neurodiscovery" and kge_scorer is not None and kge_weight > 0.0:
        pair_scores = _batch_kge_pair_scores(
            kge_scorer,
            case,
            index,
            (
                (node_id, candidate.output_id)
                for candidate in values
                for _, node_id in candidate.input_bindings
            ),
        )
        values = [
            MultiInputCandidate(
                score=_blend_kge_score(
                    candidate.score,
                    (
                        sum(available) / len(available)
                        if (
                            available := [
                                pair_scores[pair]
                                for _, node_id in candidate.input_bindings
                                for pair in ((node_id, candidate.output_id),)
                                if pair in pair_scores
                            ]
                        )
                        else None
                    ),
                    kge_weight,
                ),
                output_id=candidate.output_id,
                input_bindings=candidate.input_bindings,
                edges=candidate.edges,
                paper_keys=candidate.paper_keys,
                generation_mode=candidate.generation_mode,
            )
            for candidate in values
        ]
    return sorted(
        values,
        key=lambda candidate: (
            -candidate.score,
            candidate.output_id,
            candidate.input_bindings,
        ),
    )


def _proposed_link(
    index: FrozenGraphIndex,
    source_id: str,
    target_id: str,
    relation: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "from_id": source_id,
        "from_name": index.names.get(source_id, source_id),
        "to_id": target_id,
        "to_name": index.names.get(target_id, target_id),
        "relation_type": relation,
        "confidence": confidence,
        "claim_id": "",
        "raw_text": "",
        "evidence": {
            "status": "proposed_relation",
            "endpoint_evidence_only": True,
        },
        "source_paper": {},
    }


def _atom_route(case: CaseStudy) -> tuple[set[str], str, set[str] | None, str]:
    if case.chain is not None:
        mediator_atoms = {atom.value for atom in case.chain.mediators}
        return {case.chain.source.value}, case.chain.target.value, mediator_atoms, case.chain.signature
    if case.task is None:
        raise ValueError(f"{case.name} has neither task nor chain")
    return (
        {atom.value for atom in case.task.inputs},
        case.task.output.value,
        None,
        case.task.signature,
    )


def _generation_endpoint_atom_routes(
    case: CaseStudy,
) -> tuple[tuple[Atom, Atom], ...]:
    """Resolve legal directed endpoint routes for one generated hypothesis."""

    if case.chain is not None:
        return ((case.chain.source, case.chain.target),)
    if case.task is None:
        raise ValueError(f"{case.name} has neither task nor chain")
    contract_routes = case_study_endpoint_atom_routes(case.name, case.task.inputs)
    if contract_routes:
        return contract_routes
    return tuple(
        (source, case.task.output)
        for source in sorted(case.task.inputs, key=lambda atom: atom.value)
    )


def _endpoint_route_allowed(
    index: FrozenGraphIndex,
    source: SemanticEndpoint,
    target: SemanticEndpoint,
    routes: tuple[tuple[Atom, Atom], ...],
) -> bool:
    return any(
        endpoint_matches_atom(source, source_atom, index.concepts)
        and endpoint_matches_atom(target, target_atom, index.concepts)
        for source_atom, target_atom in routes
    )


def _candidate_score(
    method: str,
    source: str,
    mediator: str,
    target: str,
    left: FrozenEdge,
    right: FrozenEdge,
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    seed: int,
) -> float:
    evidence = (left.confidence + right.confidence) / 2.0
    jitter = _stable_unit(method, seed, source, mediator, target)
    if method == "sciagents":
        degree = len(adjacency.get(mediator, ()))
        specificity = 1.0 / (1.0 + math.log1p(max(0, degree - 1)))
        grounding = (bool(left.claim_id) + bool(right.claim_id)) / 2.0
        return 0.45 * evidence + 0.25 * grounding + 0.20 * specificity + 0.10 * jitter
    years = [year for year in (left.year, right.year) if year is not None]
    recency = 0.0 if not years else min(1.0, max(years) / float(freeze_year))
    papers = {
        str((edge.source_paper or {}).get("pmid") or (edge.source_paper or {}).get("doi"))
        for edge in (left, right)
        if (edge.source_paper or {}).get("pmid") or (edge.source_paper or {}).get("doi")
    }
    return 0.55 * evidence + 0.25 * recency + 0.15 * min(1.0, len(papers) / 2.0) + 0.05 * jitter


def _ranked_candidates(
    *,
    method: str,
    case: CaseStudy,
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    target_count: int,
    seed: int,
) -> list[tuple[float, str, str, str, FrozenEdge, FrozenEdge]]:
    _, _, mediator_atoms, _ = _atom_route(case)
    routes = _generation_endpoint_atom_routes(case)
    input_roles = tuple(dict.fromkeys(source for source, _ in routes))
    output_roles = tuple(dict.fromkeys(target for _, target in routes))
    mediator_roles = tuple(Atom(value) for value in sorted(mediator_atoms or ()))
    pool_cap = max(5000, target_count * 24)
    candidates: dict[tuple[str, str], tuple[float, str, str, str, FrozenEdge, FrozenEdge]] = {}
    mediators = sorted(
        adjacency,
        key=lambda node_id: (-len(adjacency[node_id]), _stable_unit(method, seed, node_id), node_id),
    )
    processed = 0
    for mediator in mediators:
        mediator_endpoint = _endpoint(index, mediator)
        if mediator_roles and not any(
            endpoint_matches_atom(mediator_endpoint, atom, index.concepts)
            for atom in mediator_roles
        ):
            continue
        edges = adjacency.get(mediator, ())
        sources = [
            edge
            for edge in edges
            if any(
                endpoint_matches_atom(
                    _endpoint(index, edge.other(mediator)),
                    atom,
                    index.concepts,
                )
                for atom in input_roles
            )
        ]
        targets = [
            edge
            for edge in edges
            if any(
                endpoint_matches_atom(
                    _endpoint(index, edge.other(mediator)),
                    output_role,
                    index.concepts,
                )
                for output_role in output_roles
            )
        ]
        if not sources or not targets:
            continue
        processed += 1
        sources = sorted(sources, key=lambda edge: (-edge.confidence, _stable_unit(seed, mediator, edge.other(mediator))))[:48]
        targets = sorted(targets, key=lambda edge: (-edge.confidence, _stable_unit(seed, edge.other(mediator), mediator)))[:48]
        local: list[tuple[float, str, str, str, FrozenEdge, FrozenEdge]] = []
        for left in sources:
            source = left.other(mediator)
            for right in targets:
                target = right.other(mediator)
                if source == target or _pair(source, target) in index.direct_pairs:
                    continue
                if not _endpoint_route_allowed(
                    index,
                    _endpoint(index, source),
                    _endpoint(index, target),
                    routes,
                ):
                    continue
                if not case_study_endpoint_names_allowed(
                    case.name,
                    _endpoint(index, source),
                    _endpoint(index, target),
                ):
                    continue
                score = _candidate_score(method, source, mediator, target, left, right, adjacency, freeze_year, seed)
                local.append((score, source, mediator, target, left, right))
        local.sort(key=lambda row: (-row[0], row[1], row[3]))
        for row in local[:256]:
            key = (row[1], row[3])
            if key not in candidates or row[0] > candidates[key][0]:
                candidates[key] = row
        if len(candidates) >= pool_cap and processed >= max(100, target_count // 5):
            break
    return sorted(candidates.values(), key=lambda row: (-row[0], row[1], row[3]))


def _multi_input_candidate_score(
    *,
    method: str,
    case: CaseStudy,
    output_id: str,
    bindings: tuple[tuple[str, str], ...],
    edges: tuple[FrozenEdge, ...],
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    seed: int,
) -> float:
    evidence = sum(edge.confidence for edge in edges) / len(edges)
    grounding = sum(bool(edge.claim_id) for edge in edges) / len(edges)
    years = [edge.year for edge in edges if edge.year is not None]
    recency = 0.0 if not years else min(1.0, max(years) / float(freeze_year))
    papers = {_paper_key(edge) for edge in edges if _paper_key(edge)}
    paper_diversity = min(1.0, len(papers) / max(2, len(edges)))
    compactness = min(1.0, len(bindings) / len(edges))
    bound_ids = {node_id for _, node_id in bindings} | {output_id}
    internal_ids = {
        node_id
        for edge in edges
        for node_id in (edge.source_id, edge.target_id)
        if node_id not in bound_ids
    }
    specificity_values = [
        1.0 / (1.0 + math.log1p(max(0, len(adjacency.get(node_id, ())) - 1)))
        for node_id in internal_ids
    ]
    specificity = (
        sum(specificity_values) / len(specificity_values)
        if specificity_values
        else 1.0
    )
    jitter = _stable_unit(
        method,
        seed,
        case.name,
        output_id,
        *(node_id for _, node_id in bindings),
    )
    if method == "sciagents":
        return (
            0.40 * evidence
            + 0.20 * grounding
            + 0.15 * specificity
            + 0.15 * compactness
            + 0.10 * jitter
        )
    return (
        0.45 * evidence
        + 0.20 * recency
        + 0.20 * paper_diversity
        + 0.10 * compactness
        + 0.05 * jitter
    )


def _ranked_multi_input_candidates(
    *,
    method: str,
    case: CaseStudy,
    index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]],
    freeze_year: int,
    target_count: int,
    seed: int,
) -> list[MultiInputCandidate]:
    if case.task is None or len(case.task.inputs) < 2:
        return []

    input_order = tuple(sorted(case.task.inputs, key=lambda atom: atom.value))
    edges = _unique_edges(adjacency)
    incident: dict[str, list[int]] = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        incident[edge.source_id].append(edge_index)
        incident[edge.target_id].append(edge_index)

    output_ids = [
        node_id
        for node_id in incident
        if endpoint_matches_atom(
            _endpoint(index, node_id),
            case.task.output,
            index.concepts,
        )
    ]
    output_ids.sort(key=lambda node_id: (-len(incident[node_id]), node_id))
    max_outputs = max(250, min(5000, target_count * 25))
    max_edges = len(input_order) + 1
    beam_width = max(48, min(160, target_count * 2))
    candidates: dict[tuple[tuple[str, ...], str], MultiInputCandidate] = {}

    for output_id in output_ids[:max_outputs]:
        output = _endpoint(index, output_id)
        frontier = [(frozenset({output_id}), tuple(), tuple())]
        for depth in range(1, max_edges + 1):
            next_states: dict[tuple, tuple] = {}
            for node_ids, edge_ids, bindings in frontier:
                used_edges = set(edge_ids)
                available = {
                    edge_index
                    for node_id in node_ids
                    for edge_index in incident.get(node_id, ())
                    if edge_index not in used_edges
                }
                available_rows = sorted(
                    available,
                    key=lambda edge_index: (
                        -edges[edge_index].confidence,
                        edges[edge_index].claim_id,
                        edge_index,
                    ),
                )[:48]
                for edge_index in available_rows:
                    edge = edges[edge_index]
                    if edge.source_id in node_ids and edge.target_id not in node_ids:
                        new_id = edge.target_id
                    elif edge.target_id in node_ids and edge.source_id not in node_ids:
                        new_id = edge.source_id
                    else:
                        continue
                    new_endpoint = _endpoint(index, new_id)
                    binding_map = {
                        Atom(atom_name): node_id
                        for atom_name, node_id in bindings
                    }
                    unmatched = [
                        atom
                        for atom in input_order
                        if atom not in binding_map
                        and endpoint_matches_atom(new_endpoint, atom, index.concepts)
                    ]
                    for matched_atom in (None, *unmatched):
                        updated = dict(binding_map)
                        if matched_atom is not None:
                            if new_id in updated.values():
                                continue
                            updated[matched_atom] = new_id
                        updated_bindings = tuple(
                            (atom.value, updated[atom])
                            for atom in input_order
                            if atom in updated
                        )
                        updated_edges = tuple(sorted((*edge_ids, edge_index)))
                        updated_nodes = frozenset((*node_ids, new_id))
                        selected_edges = tuple(edges[index] for index in updated_edges)
                        paper_keys = tuple(
                            _paper_key(selected_edge)
                            for selected_edge in selected_edges
                            if _paper_key(selected_edge)
                        )
                        if len(updated) == len(input_order):
                            input_ids = tuple(updated[atom] for atom in input_order)
                            if len(set(input_ids)) != len(input_ids):
                                continue
                            inputs = tuple(_endpoint(index, node_id) for node_id in input_ids)
                            if not all(
                                case_study_endpoint_names_allowed(case.name, endpoint, output)
                                for endpoint in inputs
                            ):
                                continue
                            # Complete-path tasks must be supported by independent
                            # literature rather than a single paper restated as a graph.
                            if len(set(paper_keys)) < 2:
                                continue
                            complete_bindings = tuple(
                                (atom.value, updated[atom]) for atom in input_order
                            )
                            score = _multi_input_candidate_score(
                                method=method,
                                case=case,
                                output_id=output_id,
                                bindings=complete_bindings,
                                edges=selected_edges,
                                adjacency=adjacency,
                                freeze_year=freeze_year,
                                seed=seed,
                            )
                            key = (tuple(sorted(input_ids)), output_id)
                            candidate = MultiInputCandidate(
                                score=score,
                                output_id=output_id,
                                input_bindings=complete_bindings,
                                edges=selected_edges,
                                paper_keys=tuple(dict.fromkeys(paper_keys)),
                            )
                            current = candidates.get(key)
                            if current is None or candidate.score > current.score:
                                candidates[key] = candidate
                            continue
                        if depth >= max_edges:
                            continue
                        evidence = sum(
                            selected_edge.confidence for selected_edge in selected_edges
                        ) / len(selected_edges)
                        state_quality = (
                            len(updated),
                            len(set(paper_keys)),
                            evidence,
                            -len(updated_edges),
                        )
                        signature = (
                            tuple(sorted(updated_nodes)),
                            updated_edges,
                            updated_bindings,
                        )
                        current = next_states.get(signature)
                        if current is None or state_quality > current[0]:
                            next_states[signature] = (
                                state_quality,
                                (updated_nodes, updated_edges, updated_bindings),
                            )
            frontier = [
                value[1]
                for _, value in sorted(
                    next_states.items(),
                    key=lambda item: (item[1][0], item[0]),
                    reverse=True,
                )[:beam_width]
            ]
            if not frontier:
                break

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.score,
            candidate.output_id,
            candidate.input_bindings,
        ),
    )


def _diverse_multi_input_top(
    ranked: list[MultiInputCandidate],
    target_count: int,
    *,
    coverage_fraction: float = 0.0,
) -> list[MultiInputCandidate]:
    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError("coverage_fraction must be in [0, 1]")
    selected: list[MultiInputCandidate] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    output_counts: dict[str, int] = defaultdict(int)
    covered_endpoints: set[tuple[str, str]] = set()
    coverage_target = min(
        target_count,
        int(round(target_count * coverage_fraction)),
    )
    remaining = list(ranked)
    while remaining and len(selected) < coverage_target:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(
                    {
                        ("output", remaining[index].output_id),
                        *(
                            (atom_name, node_id)
                            for atom_name, node_id
                            in remaining[index].input_bindings
                        ),
                    }
                    - covered_endpoints
                ),
                remaining[index].score,
                remaining[index].output_id,
                remaining[index].input_bindings,
            ),
        )
        candidate = remaining.pop(best_index)
        key = (
            tuple(node_id for _, node_id in candidate.input_bindings),
            candidate.output_id,
        )
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
        output_counts[candidate.output_id] += 1
        covered_endpoints.add(("output", candidate.output_id))
        covered_endpoints.update(candidate.input_bindings)
    if len(selected) >= target_count:
        return selected
    for output_limit in (4, 12, 10**9):
        for candidate in ranked:
            key = (
                tuple(node_id for _, node_id in candidate.input_bindings),
                candidate.output_id,
            )
            if key in seen or output_counts[candidate.output_id] >= output_limit:
                continue
            seen.add(key)
            output_counts[candidate.output_id] += 1
            selected.append(candidate)
            if len(selected) >= target_count:
                return selected
    return selected


def _diverse_top(ranked: list[tuple], target_count: int) -> list[tuple]:
    selected: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    source_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    for source_limit, target_limit in ((4, 12), (12, 32), (10**9, 10**9)):
        for row in ranked:
            key = (row[1], row[3])
            if key in seen or source_counts[row[1]] >= source_limit or target_counts[row[3]] >= target_limit:
                continue
            seen.add(key)
            source_counts[row[1]] += 1
            target_counts[row[3]] += 1
            selected.append(row)
            if len(selected) >= target_count:
                return selected
    return selected


def _diverse_endpoint_top(ranked: list[tuple], target_count: int) -> list[tuple]:
    """Select direct endpoint compositions without one endpoint monopolizing a batch."""

    selected: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    source_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    for source_limit, target_limit in ((3, 8), (8, 20), (10**9, 10**9)):
        for row in ranked:
            key = (row[1], row[2])
            if (
                key in seen
                or source_counts[row[1]] >= source_limit
                or target_counts[row[2]] >= target_limit
            ):
                continue
            seen.add(key)
            source_counts[row[1]] += 1
            target_counts[row[2]] += 1
            selected.append(row)
            if len(selected) >= target_count:
                return selected
    return selected


def _is_feedback_endpoint_variant(
    row: tuple,
    feedback_anchor_pairs: tuple[tuple[str, str], ...],
) -> bool:
    source_id = str(row[1])
    target_id = str(row[2])
    return any(
        (source_id == anchor_source) != (target_id == anchor_target)
        for anchor_source, anchor_target in feedback_anchor_pairs
    )


def _select_endpoint_frontier(
    ranked: list[tuple],
    target_count: int,
    *,
    feedback_anchor_pairs: tuple[tuple[str, str], ...] = (),
    feedback_anchor_fraction: float = 0.0,
    preserve_ranked_order: bool = False,
) -> list[tuple]:
    """Reserve bounded feedback mutations without discarding an audited ranking."""

    if target_count <= 0:
        return []
    if not 0.0 <= feedback_anchor_fraction <= 1.0:
        raise ValueError("feedback_anchor_fraction must be in [0, 1]")
    selector = (
        lambda rows, count: _dedupe_endpoint_candidate_rows(rows)[:count]
        if preserve_ranked_order
        else _diverse_endpoint_top(rows, count)
    )
    if not feedback_anchor_pairs or feedback_anchor_fraction <= 0.0:
        return selector(ranked, target_count)

    unique = _dedupe_endpoint_candidate_rows(ranked)
    mutations = [
        row
        for row in unique
        if _is_feedback_endpoint_variant(row, feedback_anchor_pairs)
    ]
    exploration = [
        row
        for row in unique
        if not _is_feedback_endpoint_variant(row, feedback_anchor_pairs)
    ]
    mutation_target = min(
        len(mutations),
        max(1, int(round(target_count * feedback_anchor_fraction))),
    )
    selected_mutations = selector(mutations, mutation_target)
    selected_exploration = selector(
        exploration,
        max(0, target_count - mutation_target),
    )
    if len(selected_mutations) + len(selected_exploration) < target_count:
        used = {
            _pair(str(row[1]), str(row[2]))
            for row in (*selected_mutations, *selected_exploration)
        }
        for row in unique:
            key = _pair(str(row[1]), str(row[2]))
            if key in used:
                continue
            selected_exploration.append(row)
            used.add(key)
            if len(selected_mutations) + len(selected_exploration) >= target_count:
                break

    total = min(target_count, len(selected_mutations) + len(selected_exploration))
    if not selected_mutations or not selected_exploration:
        return [*selected_exploration, *selected_mutations][:target_count]
    mutation_positions = {
        min(
            total - 1,
            round((index + 1) * (total + 1) / (len(selected_mutations) + 1)) - 1,
        )
        for index in range(len(selected_mutations))
    }
    for position in range(total - 1, -1, -1):
        if len(mutation_positions) >= len(selected_mutations):
            break
        mutation_positions.add(position)
    merged: list[tuple] = []
    mutation_index = 0
    exploration_index = 0
    for position in range(total):
        if position in mutation_positions and mutation_index < len(selected_mutations):
            merged.append(selected_mutations[mutation_index])
            mutation_index += 1
        elif exploration_index < len(selected_exploration):
            merged.append(selected_exploration[exploration_index])
            exploration_index += 1
        elif mutation_index < len(selected_mutations):
            merged.append(selected_mutations[mutation_index])
            mutation_index += 1
    return merged[:target_count]


def _link(index: FrozenGraphIndex, source: str, target: str, edge: FrozenEdge) -> dict[str, Any]:
    return {
        "from_id": source,
        "from_name": index.names.get(source, source),
        "to_id": target,
        "to_name": index.names.get(target, target),
        "relation_type": edge.relation,
        "confidence": edge.confidence,
        "claim_id": edge.claim_id,
        "raw_text": edge.raw_text,
        "evidence": {},
        "source_paper": edge.source_paper or {},
    }


def _interleave_evidence_frontier(
    hypotheses: list[dict[str, Any]],
    *,
    target_count: int,
    evidence_frontier_fraction: float,
) -> list[dict[str, Any]]:
    """Spread endpoint compositions through a connected-path candidate batch."""

    if evidence_frontier_fraction <= 0.0 or not hypotheses:
        return hypotheses[:target_count]

    def is_frontier(row: dict[str, Any]) -> bool:
        metadata = row.get("metadata") or {}
        return (
            metadata.get("generation_mode") == "evidence_frontier_composition"
            or row.get("hypothesis_type")
            in {"evidence_frontier_chain", "retrieval_composed_pair"}
        )

    connected = [row for row in hypotheses if not is_frontier(row)]
    frontier = [row for row in hypotheses if is_frontier(row)]
    desired_frontier = min(
        len(frontier),
        max(1, int(round(target_count * evidence_frontier_fraction))),
    )
    desired_connected = min(len(connected), target_count - desired_frontier)
    if desired_connected + desired_frontier < target_count:
        desired_frontier = min(
            len(frontier), target_count - desired_connected
        )
    if desired_connected + desired_frontier < target_count:
        desired_connected = min(
            len(connected), target_count - desired_frontier
        )

    connected = connected[:desired_connected]
    frontier = frontier[:desired_frontier]
    total = len(connected) + len(frontier)
    if not frontier or not connected:
        return [*connected, *frontier][:target_count]

    frontier_positions = {
        min(
            total - 1,
            round((index + 1) * (total + 1) / (len(frontier) + 1)) - 1,
        )
        for index in range(len(frontier))
    }
    for position in range(total - 1, -1, -1):
        if len(frontier_positions) >= len(frontier):
            break
        frontier_positions.add(position)

    merged: list[dict[str, Any]] = []
    connected_index = 0
    frontier_index = 0
    for position in range(total):
        if position in frontier_positions and frontier_index < len(frontier):
            merged.append(frontier[frontier_index])
            frontier_index += 1
        elif connected_index < len(connected):
            merged.append(connected[connected_index])
            connected_index += 1
        elif frontier_index < len(frontier):
            merged.append(frontier[frontier_index])
            frontier_index += 1
    return merged[:target_count]


def generate_case_hypotheses(
    *, method: str, case: CaseStudy, index: FrozenGraphIndex,
    adjacency: dict[str, list[FrozenEdge]], freeze_year: int,
    target_count: int, seed: int,
    evidence_frontier_fraction: float = 0.0,
    endpoint_canonical_quality_weight: float | None = None,
    feedback_anchor_pairs: tuple[tuple[str, str], ...] = (),
    feedback_anchor_fraction: float = 0.0,
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
    kge_checkpoint: str | None = None,
    replicate_index: int = 0,
    generation_round: int = 0,
) -> dict[str, Any]:
    if not 0.0 <= evidence_frontier_fraction <= 1.0:
        raise ValueError("evidence_frontier_fraction must be in [0, 1]")
    if not 0.0 <= feedback_anchor_fraction <= 1.0:
        raise ValueError("feedback_anchor_fraction must be in [0, 1]")
    _, _, _, signature = _atom_route(case)
    hypotheses: list[dict[str, Any]] = []
    connected_target = max(
        0,
        target_count - int(round(target_count * evidence_frontier_fraction)),
    )
    is_multi_input = case.task is not None and len(case.task.inputs) > 1
    if is_multi_input:
        ranked_multi = _ranked_multi_input_candidates(
            method=method,
            case=case,
            index=index,
            adjacency=adjacency,
            freeze_year=freeze_year,
            target_count=target_count,
            seed=seed,
        )
        selected_multi = (
            _diverse_multi_input_top(
                ranked_multi,
                connected_target,
                coverage_fraction=(
                    NEURODISCOVERY_EXPLORATION_FRACTION
                    if method == "neurodiscovery"
                    else 0.0
                ),
            )
            if connected_target
            else []
        )
        frontier_multi: list[MultiInputCandidate] = []
        if len(selected_multi) < target_count:
            seen_multi = {
                (
                    tuple(node_id for _, node_id in candidate.input_bindings),
                    candidate.output_id,
                )
                for candidate in selected_multi
            }
            ranked_frontier_multi = _ranked_multi_input_compositions(
                method=method,
                case=case,
                index=index,
                adjacency=adjacency,
                freeze_year=freeze_year,
                target_count=target_count,
                seed=seed,
                endpoint_canonical_quality_weight=(
                    endpoint_canonical_quality_weight
                ),
                kge_scorer=kge_scorer,
                kge_weight=kge_weight,
                replicate_index=replicate_index,
                generation_round=generation_round,
            )
            frontier_multi = _diverse_multi_input_top(
                [
                    candidate
                    for candidate in ranked_frontier_multi
                    if (
                        tuple(node_id for _, node_id in candidate.input_bindings),
                        candidate.output_id,
                    )
                    not in seen_multi
                ],
                target_count - len(selected_multi),
                coverage_fraction=(
                    NEURODISCOVERY_EXPLORATION_FRACTION
                    if method == "neurodiscovery"
                    else 0.0
                ),
            )
            selected_multi.extend(frontier_multi)
        for rank, candidate in enumerate(selected_multi, 1):
            input_ids = [node_id for _, node_id in candidate.input_bindings]
            input_names = [index.names.get(node_id, node_id) for node_id in input_ids]
            source = input_ids[0]
            supporting_claims = list(dict.fromkeys(
                edge.claim_id for edge in candidate.edges if edge.claim_id
            ))
            hypotheses.append({
                "id": f"{method.upper()}:{case.name}:{freeze_year}:S{seed:02d}:{rank:04d}",
                "hypothesis_type": "multi_input_evidence_graph",
                "source_id": source,
                "source_name": index.names.get(source, source),
                "target_id": candidate.output_id,
                "target_name": index.names.get(candidate.output_id, candidate.output_id),
                "path": [
                    _link(index, edge.source_id, edge.target_id, edge)
                    for edge in candidate.edges
                ],
                "confidence_score": candidate.score,
                "novelty_score": 1.0,
                "evidence_score": sum(edge.confidence for edge in candidate.edges) / len(candidate.edges),
                "testability_score": 0.5,
                "composite_score": candidate.score,
                "supporting_claims": supporting_claims,
                "explanation": (
                    f"Joint frozen-evidence hypothesis: {'; '.join(input_names)} may "
                    f"collectively relate to {index.names.get(candidate.output_id, candidate.output_id)}."
                ),
                "testability_reason": (
                    "Distinct entities cover every registered input atom; each endpoint "
                    "is grounded in frozen evidence without access to future outcomes."
                ),
                "metadata": {
                    "case_study_id": case.name,
                    "task_name": case.task.name,
                    "task_signature": signature,
                    "task_kind": "task",
                    "generator_method": method,
                    "freeze_year": freeze_year,
                    "knowledge_policy": "frozen_historical_kg" if method == "sciagents" else "frozen_case_study_literature",
                    "input_atom_order": [atom for atom, _ in candidate.input_bindings],
                    "input_entity_ids": input_ids,
                    "input_entity_names": input_names,
                    "cross_paper": len(set(candidate.paper_keys)) >= 2,
                    "source_paper_keys": list(candidate.paper_keys),
                    "generation_mode": candidate.generation_mode,
                },
            })
        ranked_count = len(ranked_multi) + len(frontier_multi)
        selected_count = len(selected_multi)
    else:
        ranked = _ranked_candidates(
            method=method, case=case, index=index, adjacency=adjacency,
            freeze_year=freeze_year, target_count=target_count, seed=seed,
        )
        selected = (
            _diverse_top(ranked, connected_target)
            if connected_target
            else []
        )
        for rank, (score, source, mediator, target, left, right) in enumerate(selected, 1):
            hypotheses.append({
                "id": f"{method.upper()}:{case.name}:{freeze_year}:S{seed:02d}:{rank:04d}",
                "hypothesis_type": "bridge",
                "source_id": source,
                "source_name": index.names.get(source, source),
                "target_id": target,
                "target_name": index.names.get(target, target),
                "path": [_link(index, source, mediator, left), _link(index, mediator, target, right)],
                "confidence_score": score,
                "novelty_score": 1.0,
                "evidence_score": (left.confidence + right.confidence) / 2.0,
                "testability_score": 0.5,
                "composite_score": score,
                "supporting_claims": [edge.claim_id for edge in (left, right) if edge.claim_id],
                "explanation": f"{index.names.get(source, source)} may relate to {index.names.get(target, target)} through {index.names.get(mediator, mediator)}.",
                "testability_reason": "Task-compatible relation grounded only in frozen evidence.",
                "metadata": {
                    "case_study_id": case.name,
                    "task_name": case.task.name if case.task is not None else case.chain.name,
                    "task_signature": signature,
                    "task_kind": "chain" if case.chain is not None else "task",
                    "generator_method": method,
                    "freeze_year": freeze_year,
                    "knowledge_policy": "frozen_historical_kg" if method == "sciagents" else "frozen_case_study_literature",
                    "generation_mode": "connected_evidence_bridge",
                },
            })
        frontier_count = 0
        if len(hypotheses) < target_count and case.chain is not None:
            seen_paths = {
                (
                    str(hypothesis.get("source_id") or ""),
                    str(((hypothesis.get("path") or [{}])[0]).get("to_id") or ""),
                    str(hypothesis.get("target_id") or ""),
                )
                for hypothesis in hypotheses
            }
            frontier = _ranked_chain_compositions(
                method=method,
                case=case,
                index=index,
                adjacency=adjacency,
                freeze_year=freeze_year,
                target_count=target_count,
                seed=seed,
                endpoint_canonical_quality_weight=(
                    endpoint_canonical_quality_weight
                ),
                kge_scorer=kge_scorer,
                kge_weight=kge_weight,
            )
            for score, source, mediator, target, left_support, right_support, papers in frontier:
                if (source, mediator, target) in seen_paths:
                    continue
                rank = len(hypotheses) + 1
                confidence = (left_support.confidence + right_support.confidence) / 2.0
                hypotheses.append({
                    "id": f"{method.upper()}:{case.name}:{freeze_year}:S{seed:02d}:{rank:04d}",
                    "hypothesis_type": "evidence_frontier_chain",
                    "source_id": source,
                    "source_name": index.names.get(source, source),
                    "target_id": target,
                    "target_name": index.names.get(target, target),
                    "path": [
                        _proposed_link(index, source, mediator, "associated_with", confidence),
                        _proposed_link(index, mediator, target, "predicts", confidence),
                    ],
                    "confidence_score": score,
                    "novelty_score": 1.0,
                    "evidence_score": confidence,
                    "testability_score": 0.5,
                    "composite_score": score,
                    "supporting_claims": list(dict.fromkeys(
                        claim_id
                        for claim_id in (left_support.claim_id, right_support.claim_id)
                        if claim_id
                    )),
                    "explanation": (
                        f"Frozen evidence independently prioritizes {index.names.get(source, source)}, "
                        f"{index.names.get(mediator, mediator)}, and {index.names.get(target, target)} "
                        "as a candidate mediation chain."
                    ),
                    "testability_reason": (
                        "The registered three-node atom chain is complete and every endpoint is "
                        "grounded before the freeze year."
                    ),
                    "metadata": {
                        "case_study_id": case.name,
                        "chain_name": case.chain.name,
                        "task_signature": signature,
                        "task_kind": "chain",
                        "generator_method": method,
                        "freeze_year": freeze_year,
                        "knowledge_policy": "frozen_historical_kg" if method == "sciagents" else "frozen_case_study_literature",
                        "generation_mode": "evidence_frontier_composition",
                        "mediator_ids": [mediator],
                        "mediator_names": [index.names.get(mediator, mediator)],
                        "source_paper_keys": list(papers),
                    },
                })
                seen_paths.add((source, mediator, target))
                frontier_count += 1
                if len(hypotheses) >= target_count:
                    break
        elif len(hypotheses) < target_count and case.task is not None:
            seen_pairs = {
                _pair(
                    str(hypothesis.get("source_id") or ""),
                    str(hypothesis.get("target_id") or ""),
                )
                for hypothesis in hypotheses
                if hypothesis.get("source_id") and hypothesis.get("target_id")
            }
            frontier = _ranked_endpoint_compositions(
                method=method,
                case=case,
                index=index,
                adjacency=adjacency,
                freeze_year=freeze_year,
                target_count=target_count,
                seed=seed,
                endpoint_canonical_quality_weight=(
                    endpoint_canonical_quality_weight
                ),
                feedback_anchor_pairs=feedback_anchor_pairs,
                feedback_anchor_fraction=feedback_anchor_fraction,
                kge_scorer=kge_scorer,
                kge_weight=kge_weight,
                replicate_index=replicate_index,
                generation_round=generation_round,
            )
            frontier = _select_endpoint_frontier(
                [
                    row
                    for row in frontier
                    if _pair(row[1], row[2]) not in seen_pairs
                ],
                target_count - len(hypotheses),
                feedback_anchor_pairs=feedback_anchor_pairs,
                feedback_anchor_fraction=feedback_anchor_fraction,
                preserve_ranked_order=method == "neurodiscovery",
            )
            for score, source, target, source_support, target_support, papers in frontier:
                if _pair(source, target) in seen_pairs:
                    continue
                rank = len(hypotheses) + 1
                evidence = (source_support.confidence + target_support.confidence) / 2.0
                hypotheses.append({
                    "id": f"{method.upper()}:{case.name}:{freeze_year}:S{seed:02d}:{rank:04d}",
                    "hypothesis_type": "retrieval_composed_pair",
                    "source_id": source,
                    "source_name": index.names.get(source, source),
                    "target_id": target,
                    "target_name": index.names.get(target, target),
                    "path": [],
                    "confidence_score": score,
                    "novelty_score": 1.0,
                    "evidence_score": evidence,
                    "testability_score": 0.5,
                    "composite_score": score,
                    "supporting_claims": list(dict.fromkeys(
                        claim_id
                        for claim_id in (source_support.claim_id, target_support.claim_id)
                        if claim_id
                    )),
                    "explanation": (
                        f"Frozen evidence independently prioritizes {index.names.get(source, source)} "
                        f"and {index.names.get(target, target)} for a task-compatible test."
                    ),
                    "testability_reason": (
                        "Both endpoints are canonical, task-compatible, and grounded in the frozen "
                        "retrieval corpus; the proposed endpoint relation is historically absent."
                    ),
                    "metadata": {
                        "case_study_id": case.name,
                        "task_name": case.task.name,
                        "task_signature": signature,
                        "task_kind": "task",
                        "generator_method": method,
                        "freeze_year": freeze_year,
                        "knowledge_policy": "frozen_historical_kg" if method == "sciagents" else "frozen_case_study_literature",
                        "generation_mode": "evidence_frontier_composition",
                        "source_paper_keys": list(papers),
                        "source_endpoint_claim_ids": [
                            source_support.claim_id
                        ] if source_support.claim_id else [],
                        "target_endpoint_claim_ids": [
                            target_support.claim_id
                        ] if target_support.claim_id else [],
                        "source_endpoint_paper_keys": [
                            paper_key
                            for paper_key in (_paper_key(source_support),)
                            if paper_key
                        ],
                        "target_endpoint_paper_keys": [
                            paper_key
                            for paper_key in (_paper_key(target_support),)
                            if paper_key
                        ],
                        "source_endpoint_support_confidence": (
                            source_support.confidence
                        ),
                        "target_endpoint_support_confidence": (
                            target_support.confidence
                        ),
                        "kge_pair_score": _kge_pair_score(
                            kge_scorer,
                            case,
                            index,
                            source,
                            target,
                        ),
                    },
                })
                seen_pairs.add(_pair(source, target))
                frontier_count += 1
                if len(hypotheses) >= target_count:
                    break
        ranked_count = len(ranked) + frontier_count
        selected_count = len(hypotheses)
    hypotheses = _interleave_evidence_frontier(
        hypotheses,
        target_count=target_count,
        evidence_frontier_fraction=evidence_frontier_fraction,
    )
    selected_count = len(hypotheses)
    for rank in range(len(hypotheses) + 1, target_count + 1):
        hypotheses.append({
            "id": f"{method.upper()}:{case.name}:{freeze_year}:S{seed:02d}:INVALID:{rank:04d}",
            "hypothesis_type": "generation_failure",
            "source_id": "", "source_name": "", "target_id": "", "target_name": "",
            "path": [], "confidence_score": 0.0, "novelty_score": 0.0,
            "evidence_score": 0.0, "testability_score": 0.0, "composite_score": 0.0,
            "supporting_claims": [],
            "explanation": "No valid frozen-knowledge candidate was available for this fixed-budget slot.",
            "metadata": {"case_study_id": case.name, "generator_method": method, "freeze_year": freeze_year, "generation_failure": True},
        })
    return {
        "metadata": {
            "method": method,
            "case_study_id": case.name,
            "freeze_year": freeze_year,
            "requested": target_count,
            "valid": selected_count,
            "candidate_pool": ranked_count,
            "seed": seed,
            "replicate_index": replicate_index,
            "generation_round": generation_round,
            "endpoint_canonical_quality_policy": (
                ENDPOINT_CANONICAL_QUALITY_POLICY
                if method == "neurodiscovery"
                else "disabled"
            ),
            "endpoint_canonical_quality_weight": (
                (
                    NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
                    if endpoint_canonical_quality_weight is None
                    else float(endpoint_canonical_quality_weight)
                )
                if method == "neurodiscovery" else 0.0
            ),
            "evidence_frontier_fraction": evidence_frontier_fraction,
            "kge_policy": (
                NEURODISCOVERY_KGE_POLICY
                if method == "neurodiscovery" and kge_scorer is not None
                else "disabled"
            ),
            "kge_weight": (
                float(kge_weight)
                if method == "neurodiscovery" and kge_scorer is not None
                else 0.0
            ),
            "kge_checkpoint": (
                kge_checkpoint
                if method == "neurodiscovery" and kge_scorer is not None
                else None
            ),
            "kge_catalog_expansion": (
                {
                    "anchor_limit": NEURODISCOVERY_KGE_ANCHOR_LIMIT,
                    "anchor_protected_head": (
                        NEURODISCOVERY_KGE_ANCHOR_PROTECTED_HEAD
                    ),
                    "neighbor_limit": NEURODISCOVERY_KGE_NEIGHBOR_LIMIT,
                    "tail_anchor_limit": (
                        NEURODISCOVERY_KGE_TAIL_ANCHOR_LIMIT
                    ),
                    "tail_neighbor_limit": (
                        NEURODISCOVERY_KGE_TAIL_NEIGHBOR_LIMIT
                    ),
                    "neighbor_band": NEURODISCOVERY_KGE_NEIGHBOR_BAND,
                    "stratified_fraction": (
                        NEURODISCOVERY_KGE_STRATIFIED_FRACTION
                    ),
                    "shard_policy": "rank_banded_coprime_round_rotation",
                    "round_shard_stride": NEURODISCOVERY_ROUND_SHARD_STRIDE,
                }
                if method == "neurodiscovery" and kge_scorer is not None
                else None
            ),
            "evidence_frontier_candidates": sum(
                (row.get("metadata") or {}).get("generation_mode")
                == "evidence_frontier_composition"
                or row.get("hypothesis_type")
                in {"evidence_frontier_chain", "retrieval_composed_pair"}
                for row in hypotheses
            ),
        },
        "hypotheses": hypotheses,
    }


def select_cases(names: list[str] | None) -> tuple[CaseStudy, ...]:
    selected = names or list(list_case_study_names())
    return tuple(case_study_by_name(name) for name in selected)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_bundle = _runtime_source_bundle(
        getattr(args, "source_bundle_manifest", None),
        verify_references=not bool(
            getattr(args, "skip_source_bundle_reference_rehash", False)
        ),
    )
    eligibility = (
        load_locked_hindcasting_eligibility(args.eligibility_manifest)
        if args.eligibility_manifest is not None
        else None
    )
    requested_case_ids = (
        args.case_study_ids
        or (
            list(eligibility.primary_case_study_ids)
            if eligibility is not None
            else None
        )
    )
    cases = select_cases(requested_case_ids)
    selected_primary_windows = (
        eligibility.selected_windows(
            case_study_ids=(case.name for case in cases),
            windows=args.windows,
        )
        if eligibility is not None
        else None
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    semantic_audits: dict[str, dict[str, int]] = {}
    for window in args.windows:
        eligible_cases = tuple(
            case
            for case in cases
            if selected_primary_windows is None
            or (
                case.name,
                window.freeze_year,
                window.future_start_year,
                window.future_end_year,
            )
            in selected_primary_windows
        )
        if not eligible_cases:
            continue
        snapshot = args.snapshot_root / f"kg_{window.freeze_year}"
        graph_path = snapshot / "knowledge_graph.json"
        claims_path = snapshot / "extracted_claims.jsonl"
        if not graph_path.is_file() or not claims_path.is_file():
            raise FileNotFoundError(f"missing snapshot: {snapshot}")
        print(f"[index] KG_{window.freeze_year}", flush=True)
        index = FrozenGraphIndex.load(graph_path)
        claim_adjacency, literature, semantic_audit = load_semantic_claim_adjacencies(
            claims_path,
            index,
            (case.name for case in eligible_cases),
        )
        semantic_audits[str(window.freeze_year)] = semantic_audit
        sciagents_adjacency = _merge_adjacencies(index.adjacency, claim_adjacency)
        for seed in args.seeds:
            for method in args.methods:
                for case in eligible_cases:
                    run_dir = args.output_root / method / f"seed_{seed:02d}" / case.name / window.label
                    output = run_dir / "hypotheses_raw.json"
                    if output.is_file() and not args.force:
                        payload = json.loads(output.read_text(encoding="utf-8"))
                        _validate_reusable_payload(
                            payload,
                            method=method,
                            case_study_id=case.name,
                            freeze_year=window.freeze_year,
                            target_count=args.target_per_case_study,
                            seed=seed,
                            source_bundle=source_bundle,
                            output=output,
                        )
                    else:
                        adjacency = (
                            sciagents_adjacency
                            if method == "sciagents"
                            else literature.get(case.name, {})
                        )
                        payload = generate_case_hypotheses(
                            method=method, case=case, index=index, adjacency=adjacency,
                            freeze_year=window.freeze_year, target_count=args.target_per_case_study,
                            seed=seed,
                            replicate_index=seed,
                        )
                        payload.setdefault("metadata", {})["source_bundle"] = source_bundle
                        _atomic_write_json(output, payload)
                    rows.append({
                        "method": method,
                        "seed": seed,
                        "case_study_id": case.name,
                        "freeze_year": window.freeze_year,
                        "future_start_year": window.future_start_year,
                        "future_end_year": window.future_end_year,
                        "hypotheses_path": str(output.resolve()),
                        "hypotheses_bytes": output.stat().st_size,
                        "hypotheses_sha256": sha256_file(output),
                        **(payload.get("metadata") or {}),
                    })
                    print(f"[{method}] seed={seed} {case.name} KG_{window.freeze_year}: {payload['metadata']['valid']}/{args.target_per_case_study}", flush=True)
    source_bundle_at_completion = _runtime_source_bundle(
        getattr(args, "source_bundle_manifest", None),
        verify_references=not bool(
            getattr(args, "skip_source_bundle_reference_rehash", False)
        ),
    )
    if source_bundle_at_completion != source_bundle:
        raise ValueError("source bundle changed during frozen-baseline generation")
    manifest = {
        "schema_version": "case-study-frozen-baselines.v2",
        "semantic_projection": SEMANTIC_PROJECTION_VERSION,
        "snapshot_root": str(args.snapshot_root.resolve()),
        "methods": list(args.methods), "seeds": list(args.seeds),
        "case_studies": sorted({str(row["case_study_id"]) for row in rows}),
        "windows": [
            {
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
            }
            for freeze, start, end in sorted(
                {
                    (
                        int(row["freeze_year"]),
                        int(row["future_start_year"]),
                        int(row["future_end_year"]),
                    )
                    for row in rows
                }
            )
        ],
        "target_per_case_study": args.target_per_case_study,
        "source_bundle": source_bundle_at_completion,
        "semantic_audit_by_freeze_year": semantic_audits,
        "eligibility": (
            {
                "manifest_path": str(eligibility.manifest_path),
                "manifest_sha256": sha256_file(eligibility.manifest_path),
                "matrix_path": str(eligibility.matrix_path),
                "matrix_sha256": eligibility.matrix_sha256,
                "analysis_tier": "primary",
                "selected_case_windows": len(selected_primary_windows or ()),
            }
            if eligibility is not None
            else None
        ),
        "runs": rows,
    }
    _atomic_write_json(args.output_root / "generation_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=(
            ROOT
            / "neurooracle/data/experiments/hindcasting/"
            "snapshots_full_v2_endpoint_v3"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "neurooracle/data/experiments/hindcasting/frozen_baselines_current")
    parser.add_argument("--case-study-ids", nargs="*", choices=list_case_study_names(), default=None)
    parser.add_argument(
        "--eligibility-manifest",
        type=Path,
        default=None,
        help=(
            "Optional immutable method-blind eligibility lock; only its primary "
            "Case Study/window rows are generated."
        ),
    )
    parser.add_argument(
        "--source-bundle-manifest",
        type=Path,
        default=None,
        help=(
            "Optional immutable source bundle; live/archive sources and every "
            "referenced snapshot are verified before and after generation."
        ),
    )
    parser.add_argument(
        "--skip-source-bundle-reference-rehash",
        action="store_true",
        help="Use only after an outer formal runner has deep-verified references.",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--target-per-case-study", type=int, default=1200)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.target_per_case_study < 1:
        parser.error("--target-per-case-study must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(json.dumps({"output_root": str(args.output_root), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()


# Updated: 2026-08-12 19:18 HKT - balance and deduplicate canonical rank-stratified exploration.
# Updated: 2026-08-13 08:56:00 HKT - add frozen rank-stratified two-dimensional endpoint fan-out.
# Updated: 2026-08-13 09:05:00 HKT - unwrap exploration heap records before endpoint fan-out selection.
# Updated: 2026-08-12 19:29 HKT - add frozen-evidence head/tail endpoint coverage for NeuroDiscovery.
# Updated: 2026-08-12 20:34 HKT - preserve source/target provenance for feedback-conditioned endpoint expansion.
# Updated: 2026-08-12 21:54 HKT - reserve NeuroDiscovery exploration for directed endpoint and multi-input coverage.
# Updated: 2026-08-12 22:28 HKT - add a frozen ComplEx pair prior before endpoint candidate truncation.
# Updated: 2026-08-12 23:51 HKT - retrieve rank-stratified KGE neighbours across the full frozen endpoint catalog.
# Updated: 2026-08-13 01:38 HKT - shard frozen-catalog KGE retrieval by replicate and closed-loop round.
# Updated: 2026-08-13 02:02 HKT - preserve a bounded KGE long-tail quota through candidate truncation.
# Updated: 2026-08-13 02:14 HKT - make endpoint coverage complementary across replicate-round coordinates.
# Updated: 2026-08-13 03:01 HKT - align claim-local generation with frozen endpoint reachability v2.
# Updated: 2026-08-13 03:23 HKT - add complementary KGE anchors and bidirectional catalog coverage.
# Updated: 2026-08-13 04:16 HKT - rotate frozen-evidence tails across equal rank bands for outcome-blind long-tail coverage.
# Updated: 2026-08-13 05:04 HKT - expose complete frozen-evidence ranks to shallow long-tail KGE retrieval.
# Updated: 2026-08-13 05:19 HKT - reserve a bounded frozen-KGE quota inside the executable exploration prefix.
# Updated: 2026-08-13 05:53:47 HKT - admit frozen-supported claim-local endpoints to the executable evidence pool.
# Updated: 2026-08-13 06:39:12 HKT - prevent closed-loop rank-band shard repetition with coprime round rotation.
# Updated: 2026-08-13 09:25:00 HKT - preserve NeuroDiscovery's audited ranked frontier through final truncation.
