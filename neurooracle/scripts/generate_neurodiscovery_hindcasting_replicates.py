"""Generate NeuroDiscovery hindcasting rankings while reusing each frozen KG."""

from __future__ import annotations

import argparse
from collections import ChainMap, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

from neurooracle.scripts.run_case_study_hindcasting import (
    DEFAULT_WINDOWS,
    _enforce_fixed_budget,
    _hypothesis_payload_semantic_key,
    _is_complete_fixed_budget_payload,
    parse_window,
)
from neurooracle.scripts.generate_case_study_frozen_baselines import (
    ENDPOINT_CANONICAL_QUALITY_POLICY,
    NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT,
    NEURODISCOVERY_KGE_WEIGHT,
    FrozenEdge,
    FrozenGraphIndex,
    _endpoint_canonical_quality,
    _endpoint,
    generate_case_hypotheses,
)
from neurooracle.src.case_studies import (
    CaseStudy,
    case_study_by_name,
    list_case_study_names,
)
from neurooracle.src.hypothesis_cli import cmd_batch, load_graph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.feedback_state import FeedbackState, SUPPORTED
from neurooracle.src.claim_semantics import concept_atom_roles, semantic_claim_pair
from neurooracle.src.case_study_relation_contracts import (
    case_study_endpoint_names_allowed,
)
from neurooracle.src.hindcasting_static_policy import annotate_relation_aware_payload
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


def _atomic_json_temporary_path(path: Path) -> Path:
    """Keep the sibling temporary name short for deep Windows run trees."""

    return path.parent / f".{os.getpid():x}.tmp"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_json_temporary_path(path)
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@contextmanager
def _exclusive_lock(
    path: Path,
    *,
    timeout_seconds: float = 120.0,
    stale_after_seconds: float = 600.0,
):
    """Serialize cross-process manifest merges with a crash-recoverable lockfile."""

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                is_stale = time.time() - path.stat().st_mtime > stale_after_seconds
            except FileNotFoundError:
                continue
            if is_stale:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for manifest lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _manifest_shard_id(args: argparse.Namespace, cases: tuple[CaseStudy, ...]) -> str:
    payload = {
        "seeds": sorted(set(map(int, args.seeds))),
        "case_studies": sorted(case.name for case in cases),
        "windows": sorted(
            (
                window.freeze_year,
                window.future_start_year,
                window.future_end_year,
            )
            for window in args.windows
        ),
        "target_per_case_study": int(args.target_per_case_study),
        "generation_pool_size": int(args.generation_pool_size),
        "seed_diversity_fraction": float(args.seed_diversity_fraction),
        "candidate_pool_mode": str(args.candidate_pool_mode),
        "task_scope_fraction": float(args.task_scope_fraction),
        "evidence_frontier_fraction": float(args.evidence_frontier_fraction),
        "endpoint_canonical_quality_weight": float(
            args.endpoint_canonical_quality_weight
        ),
        "protect_general_top_k": int(args.protect_general_top_k),
        "static_score_family": str(args.static_score_family),
        "eligibility_manifest": (
            str(args.eligibility_manifest.resolve())
            if args.eligibility_manifest is not None
            else None
        ),
        "eligibility_manifest_sha256": (
            sha256_file(args.eligibility_manifest.resolve())
            if args.eligibility_manifest is not None
            else None
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def merge_generation_manifests(output_root: Path) -> dict[str, Any]:
    """Atomically merge completed process shards into one canonical manifest."""

    with _exclusive_lock(output_root / ".generation_manifest.lock"):
        return _merge_generation_manifests_unlocked(output_root)


def _merge_generation_manifests_unlocked(output_root: Path) -> dict[str, Any]:
    shard_paths = sorted(output_root.glob("generation_manifest_shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No generation manifest shards under {output_root}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    keys: dict[tuple[Any, ...], dict[str, Any]] = {}
    for shard in shards:
        for row in shard.get("runs", []):
            key = (
                row.get("method"),
                int(row["seed"]),
                row["case_study_id"],
                int(row["freeze_year"]),
                int(row["future_start_year"]),
                int(row["future_end_year"]),
            )
            existing = keys.get(key)
            if existing is not None and existing != row:
                raise ValueError(f"Conflicting hindcasting manifest rows for {key}")
            keys[key] = row
    runs = [keys[key] for key in sorted(keys)]
    windows = {
        (
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
        )
        for row in runs
    }
    target_values = {int(shard["target_per_case_study"]) for shard in shards}
    pool_values = {int(shard["generation_pool_size"]) for shard in shards}
    diversity_values = {
        float(shard.get("seed_diversity_fraction", 0.0)) for shard in shards
    }
    pool_modes = {str(shard.get("candidate_pool_mode", "scoped")) for shard in shards}
    scope_fractions = {
        float(shard.get("task_scope_fraction", 0.0)) for shard in shards
    }
    frontier_fractions = {
        float(shard.get("evidence_frontier_fraction", 0.0)) for shard in shards
    }
    protected_values = {
        int(shard.get("protect_general_top_k", 0)) for shard in shards
    }
    score_families = {
        str(shard.get("static_score_family", "legacy")) for shard in shards
    }
    endpoint_quality_policies = {
        str(
            shard.get(
                "endpoint_canonical_quality_policy",
                ENDPOINT_CANONICAL_QUALITY_POLICY,
            )
        )
        for shard in shards
    }
    endpoint_quality_weights = {
        float(
            shard.get(
                "endpoint_canonical_quality_weight",
                NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT,
            )
        )
        for shard in shards
    }
    engine_reuse_scopes = {
        str(shard.get("engine_reuse_scope", "per_run")) for shard in shards
    }
    eligibility_manifests = {
        str(shard.get("eligibility_manifest") or "none") for shard in shards
    }
    eligibility_manifest_hashes = {
        str(shard.get("eligibility_manifest_sha256") or "none")
        for shard in shards
    }
    eligibility_matrix_hashes = {
        str(shard.get("eligibility_matrix_sha256") or "none")
        for shard in shards
    }
    snapshot_root_values = [
        str(shard.get("snapshot_root") or "unrecorded") for shard in shards
    ]
    recorded_snapshot_roots = {
        value for value in snapshot_root_values if value != "unrecorded"
    }
    python_hash_seed_values = [
        str(shard.get("python_hash_seed", "unrecorded")) for shard in shards
    ]
    recorded_python_hash_seeds = {
        value for value in python_hash_seed_values if value != "unrecorded"
    }
    if (
        len(target_values) != 1
        or len(pool_values) != 1
        or len(diversity_values) != 1
        or len(pool_modes) != 1
        or len(scope_fractions) != 1
        or len(frontier_fractions) != 1
        or len(protected_values) != 1
        or len(score_families) != 1
        or len(endpoint_quality_policies) != 1
        or len(endpoint_quality_weights) != 1
        or len(engine_reuse_scopes) != 1
        or len(eligibility_manifests) != 1
        or len(eligibility_manifest_hashes) != 1
        or len(eligibility_matrix_hashes) != 1
        or len(recorded_snapshot_roots) > 1
        or (recorded_snapshot_roots and "unrecorded" in snapshot_root_values)
        or len(recorded_python_hash_seeds) > 1
    ):
        raise ValueError("Cannot merge shards with different generation settings")
    manifest = {
        "schema_version": "neurodiscovery-hindcasting-replicates.v2",
        "method": "neurodiscovery",
        "snapshot_root": (
            next(iter(recorded_snapshot_roots))
            if recorded_snapshot_roots
            else None
        ),
        "snapshot_root_recording_complete": all(
            value != "unrecorded" for value in snapshot_root_values
        ),
        "seeds": sorted({int(row["seed"]) for row in runs}),
        "case_studies": sorted({str(row["case_study_id"]) for row in runs}),
        "windows": [
            {
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
            }
            for freeze, start, end in sorted(windows)
        ],
        "target_per_case_study": target_values.pop(),
        "generation_pool_size": pool_values.pop(),
        "seed_diversity_fraction": diversity_values.pop(),
        "candidate_pool_mode": pool_modes.pop(),
        "task_scope_fraction": scope_fractions.pop(),
        "evidence_frontier_fraction": frontier_fractions.pop(),
        "protect_general_top_k": protected_values.pop(),
        "static_score_family": score_families.pop(),
        "endpoint_canonical_quality_policy": endpoint_quality_policies.pop(),
        "endpoint_canonical_quality_weight": endpoint_quality_weights.pop(),
        "engine_reuse_scope": engine_reuse_scopes.pop(),
        "eligibility_manifest": (
            None
            if (value := eligibility_manifests.pop()) == "none"
            else value
        ),
        "eligibility_manifest_sha256": (
            None
            if (value := eligibility_manifest_hashes.pop()) == "none"
            else value
        ),
        "eligibility_matrix_sha256": (
            None
            if (value := eligibility_matrix_hashes.pop()) == "none"
            else value
        ),
        "python_hash_seed": (
            next(iter(recorded_python_hash_seeds))
            if recorded_python_hash_seeds
            else "unrecorded"
        ),
        "python_hash_seed_recording_complete": all(
            value != "unrecorded" for value in python_hash_seed_values
        ),
        "shards": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in shard_paths
        ],
        "runs": runs,
    }
    _atomic_write_json(output_root / "generation_manifest.json", manifest)
    return manifest


def _tag(engine: HypothesisEngine, output: Path, case: CaseStudy) -> None:
    """Tag rows without discarding generator-level audit metadata."""

    payload = json.loads(output.read_text(encoding="utf-8"))
    hypotheses = payload.get("hypotheses") or []
    for hypothesis in hypotheses:
        metadata = dict(hypothesis.get("metadata") or {})
        metadata["case_study_id"] = case.name
        hypothesis["metadata"] = metadata
    payload["n_hypotheses"] = len(hypotheses)
    _atomic_write_json(output, payload)


def _apply_candidate_canonical_quality(
    concepts: dict[str, Any],
    payload: dict[str, Any],
    *,
    weight: float,
) -> dict[str, Any]:
    """Apply an outcome-blind frozen entity-quality prior to ranked candidates."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("endpoint canonical quality weight must be in [0, 1]")
    quality_index = FrozenGraphIndex(
        concepts=concepts,
        names={},
        atoms={},
        adjacency={},
        direct_pairs=set(),
    )
    rows = [dict(row) for row in payload.get("hypotheses") or []]
    for original_rank, row in enumerate(rows, start=1):
        metadata = dict(row.get("metadata") or {})
        previous_audit = dict(metadata.get("endpoint_canonical_quality") or {})
        node_ids = tuple(dict.fromkeys(
            str(node_id)
            for node_id in (
                row.get("source_id"),
                *(metadata.get("input_entity_ids") or ()),
                *((metadata.get("path_node_ids") or ())[1:-1]),
                *(metadata.get("mediator_ids") or ()),
                row.get("target_id"),
            )
            if str(node_id or "")
        ))
        qualities = tuple(
            _endpoint_canonical_quality(quality_index, node_id)
            for node_id in node_ids
        )
        endpoint_quality = (
            sum(qualities) / len(qualities) if qualities else 0.0
        )
        original_score = float(
            previous_audit.get(
                "original_composite_score",
                row.get("composite_score") or 0.0,
            )
        )
        base_rank = int(previous_audit.get("original_rank", original_rank))
        adjusted_score = original_score * (
            1.0 - weight * (1.0 - endpoint_quality)
        )
        row["composite_score"] = adjusted_score
        metadata["endpoint_canonical_quality"] = {
            "policy": ENDPOINT_CANONICAL_QUALITY_POLICY,
            "weight": weight,
            "node_ids": list(node_ids),
            "node_qualities": list(qualities),
            "mean_quality": endpoint_quality,
            "original_composite_score": original_score,
            "adjusted_composite_score": adjusted_score,
            "original_rank": base_rank,
            "application": "candidate_composite_score",
            "uses_future_outcomes": False,
        }
        row["metadata"] = metadata
    rows.sort(
        key=lambda row: (
            (
                -float(row.get("composite_score") or 0.0)
                if weight > 0.0
                else int(
                    ((row.get("metadata") or {}).get("endpoint_canonical_quality") or {})
                    .get("original_rank", 10**12)
                )
            ),
            int(
                ((row.get("metadata") or {}).get("endpoint_canonical_quality") or {})
                .get("original_rank", 10**12)
            ) if weight > 0.0 else 0,
            str(row.get("id") or ""),
        )
    )
    payload["hypotheses"] = rows
    payload["n_hypotheses"] = len(rows)
    metadata = dict(payload.get("metadata") or {})
    metadata["endpoint_canonical_quality"] = {
        "policy": (
            ENDPOINT_CANONICAL_QUALITY_POLICY if weight > 0.0 else "disabled"
        ),
        "weight": weight,
        "uses_future_outcomes": False,
        "candidate_count": len(rows),
    }
    payload["metadata"] = metadata
    return payload


def _optional_year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frontier_catalog(engine: HypothesisEngine) -> tuple[
    dict[str, str],
    dict[str, frozenset[str]],
    dict[str, tuple[str, ...]],
]:
    """Build one reusable atom catalog for all scoped engines on a frozen KG."""

    cache_name = "_neurodiscovery_frontier_catalog_v1"
    cached = getattr(engine.kg, cache_name, None)
    if cached is not None:
        return cached

    names: dict[str, str] = {}
    atoms: dict[str, frozenset[str]] = {}
    nodes_by_atom: dict[str, list[str]] = defaultdict(list)
    for node_id, node in engine._index.items():
        if "claim" in set(node.domain_tags or ()):
            continue
        node_atoms = frozenset(atom.value for atom in concept_atom_roles(node))
        if not node_atoms:
            continue
        names[node_id] = str(node.preferred_name or node_id)
        atoms[node_id] = node_atoms
        for atom_name in node_atoms:
            nodes_by_atom[atom_name].append(node_id)
    catalog = (
        names,
        atoms,
        {
            atom_name: tuple(sorted(node_ids))
            for atom_name, node_ids in nodes_by_atom.items()
        },
    )
    setattr(engine.kg, cache_name, catalog)
    return catalog


def _global_frontier_context(
    engine: HypothesisEngine,
) -> tuple[
    dict[str, list[FrozenEdge]],
    dict[str, str],
    dict[str, frozenset[str]],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    """Build an outcome-blind semantic context from the frozen snapshot.

    Candidate legality remains Case-Study scoped.  This graph is used only to
    retrieve and compare endpoints that have related *historical* contexts in
    the same temporal snapshot.  It deliberately excludes claim nodes,
    provenance/tree relations, negated claims, and duplicate canonical claim
    edges whose collision-safe semantic projection is added separately below.
    """

    cache_name = "_neurodiscovery_global_frontier_context_v1"
    cached = getattr(engine.kg, cache_name, None)
    if cached is not None:
        return cached

    adjacency: dict[str, list[FrozenEdge]] = defaultdict(list)
    names: dict[str, str] = {}
    atoms: dict[str, frozenset[str]] = {}
    concepts: dict[str, dict[str, Any]] = {}
    audit: dict[str, int] = defaultdict(int)
    seen_edges: set[tuple[str, ...]] = set()

    def register_endpoint(endpoint: Any) -> None:
        if endpoint.uses_canonical_id and endpoint.entity_id in engine._index:
            canonical_node = engine._index[endpoint.entity_id]
            endpoint_name = str(
                canonical_node.preferred_name or endpoint.entity_id
            )
        else:
            endpoint_name = endpoint.name
        names.setdefault(endpoint.entity_id, endpoint_name)
        endpoint_atoms = frozenset(endpoint.atoms)
        atoms[endpoint.entity_id] = atoms.get(endpoint.entity_id, frozenset()) | endpoint_atoms
        if endpoint.entity_id not in engine._index:
            concepts.setdefault(
                endpoint.entity_id,
                {
                    "preferred_name": endpoint_name,
                    "domain_tags": [],
                    "metadata": {
                        "atom_types": sorted(endpoint_atoms),
                        "historical_semantic_endpoint": True,
                    },
                },
            )

    def add_edge(edge: FrozenEdge, *, source_kind: str) -> None:
        paper = edge.source_paper or {}
        key = (
            *tuple(sorted((edge.source_id, edge.target_id))),
            edge.relation,
            edge.claim_id,
            str(paper.get("pmid") or ""),
            str(paper.get("doi") or ""),
            str(edge.year or ""),
        )
        if key in seen_edges:
            audit["duplicate_context_edges"] += 1
            return
        seen_edges.add(key)
        adjacency[edge.source_id].append(edge)
        adjacency[edge.target_id].append(edge)
        audit[f"{source_kind}_context_edges"] += 1

    # Use semantic endpoint projection so canonical-ID collisions cannot pollute
    # the context graph.  Every claim in this engine already belongs to the
    # temporal snapshot loaded for the active freeze year.
    for claim_id, node in engine._index.items():
        if "claim" not in set(node.domain_tags or ()):
            continue
        metadata = node.metadata or {}
        if metadata.get("negated"):
            audit["negated_claims_excluded"] += 1
            continue
        projected = semantic_claim_pair(metadata, engine._index)
        if projected is None:
            audit["claims_without_semantic_pair"] += 1
            continue
        subject, obj = projected
        register_endpoint(subject)
        register_endpoint(obj)
        source_paper = metadata.get("source_paper") or {}
        if not isinstance(source_paper, dict):
            source_paper = {"reference": str(source_paper)}
        add_edge(
            FrozenEdge(
                source_id=subject.entity_id,
                target_id=obj.entity_id,
                relation=str(metadata.get("predicate") or "is_associated_with"),
                confidence=float(metadata.get("confidence") or 0.5),
                claim_id=str(claim_id),
                raw_text=str(metadata.get("raw_text") or ""),
                source_paper=dict(source_paper),
                year=_optional_year(source_paper.get("year")),
            ),
            source_kind="claim",
        )

    # Add non-claim curated semantic edges.  Claim-backed graph edges are
    # skipped because the collision-safe projection above is authoritative.
    for source_id, target_id, data in engine.G.edges(data=True):
        relation = str(data.get("relation_type") or "related_to")
        metadata = data.get("metadata") or {}
        if relation in {"about", "is_a", "part_of"}:
            continue
        if metadata.get("claim_id") or str(data.get("source") or "").lower().startswith("claim:"):
            continue
        source_node = engine._index.get(source_id)
        target_node = engine._index.get(target_id)
        if source_node is None or target_node is None:
            continue
        if "claim" in set(source_node.domain_tags or ()) or "claim" in set(target_node.domain_tags or ()):
            continue
        source_atoms = frozenset(atom.value for atom in concept_atom_roles(source_node))
        target_atoms = frozenset(atom.value for atom in concept_atom_roles(target_node))
        if not source_atoms or not target_atoms:
            continue
        names.setdefault(source_id, str(source_node.preferred_name or source_id))
        names.setdefault(target_id, str(target_node.preferred_name or target_id))
        atoms[source_id] = atoms.get(source_id, frozenset()) | source_atoms
        atoms[target_id] = atoms.get(target_id, frozenset()) | target_atoms
        paper = data.get("source_paper") or metadata.get("source_paper") or {}
        if not isinstance(paper, dict):
            paper = {"reference": str(paper)}
        add_edge(
            FrozenEdge(
                source_id=str(source_id),
                target_id=str(target_id),
                relation=relation,
                confidence=float(data.get("confidence") or 0.5),
                raw_text=str(data.get("raw_text") or metadata.get("raw_text") or ""),
                source_paper=dict(paper),
                year=_optional_year(paper.get("year")),
            ),
            source_kind="curated",
        )

    audit["context_nodes"] = len(adjacency)
    audit["context_edges"] = len(seen_edges)
    result = (dict(adjacency), names, atoms, concepts, dict(audit))
    setattr(engine.kg, cache_name, result)
    return result


def _frontier_index_for_case(
    engine: HypothesisEngine,
    case: CaseStudy,
) -> tuple[FrozenGraphIndex, dict[str, list[FrozenEdge]], int]:
    """Adapt scoped semantic claims to the frozen baseline frontier interface."""

    cache_name = f"_neurodiscovery_frontier_index_{case.name}"
    cached = getattr(engine, cache_name, None)
    if cached is not None:
        return cached

    records = engine._semantic_claim_records_for_scope(case.name)
    names, atoms, nodes_by_atom = _frontier_catalog(engine)
    (
        context_adjacency,
        context_names,
        context_atoms,
        context_concepts,
        context_audit,
    ) = _global_frontier_context(engine)
    local_names: dict[str, str] = {}
    local_atoms: dict[str, frozenset[str]] = {}
    local_concepts: dict[str, dict[str, Any]] = {}
    adjacency: dict[str, list[FrozenEdge]] = defaultdict(list)
    for record in records:
        subject = record["subject"]
        obj = record["object"]
        for endpoint in (subject, obj):
            # A canonical identity must also carry its canonical display name.
            # Claim wording is retained only for collision-safe local entities;
            # otherwise the payload can misleadingly show a specific readout or
            # phenotype while evaluation matches the broader canonical concept.
            if endpoint.uses_canonical_id and endpoint.entity_id in engine._index:
                canonical_node = engine._index[endpoint.entity_id]
                endpoint_name = str(
                    canonical_node.preferred_name or endpoint.entity_id
                )
            else:
                endpoint_name = endpoint.name
            local_names.setdefault(endpoint.entity_id, endpoint_name)
            local_atoms.setdefault(endpoint.entity_id, frozenset(endpoint.atoms))
            if endpoint.entity_id not in engine._index:
                local_concepts.setdefault(
                    endpoint.entity_id,
                    {
                        "preferred_name": endpoint_name,
                        "domain_tags": [],
                        "metadata": {"atom_types": list(endpoint.atoms)},
                    },
                )
        source_paper = record.get("source_paper") or {}
        edge = FrozenEdge(
            source_id=subject.entity_id,
            target_id=obj.entity_id,
            relation=str(record.get("predicate") or "is_associated_with"),
            confidence=float(record.get("confidence") or 0.5),
            claim_id=str(record.get("claim_id") or ""),
            raw_text=str(record.get("raw_text") or ""),
            source_paper=dict(source_paper),
            year=_optional_year(source_paper.get("year")),
        )
        adjacency[subject.entity_id].append(edge)
        adjacency[obj.entity_id].append(edge)

    # Rebuild the historical-pair exclusion set from the same collision-safe
    # frozen context used for retrieval.  This avoids re-proposing a relation
    # that already exists outside the active Case Study scope.
    historical_pairs = {
        tuple(sorted((edge.source_id, edge.target_id)))
        for incident in context_adjacency.values()
        for edge in incident
        if edge.source_id != edge.target_id
    }
    index = FrozenGraphIndex(
        concepts=ChainMap(local_concepts, context_concepts, engine._index),
        names=ChainMap(local_names, context_names, names),
        atoms=ChainMap(local_atoms, context_atoms, atoms),
        adjacency={},
        direct_pairs=historical_pairs,
    )
    # The canonical catalog is KG-wide and immutable; local claim entities are
    # reached through scoped evidence, not silently promoted to ontology entries.
    setattr(index, "_catalog_nodes_by_atom", nodes_by_atom)
    setattr(index, "_global_context_adjacency", context_adjacency)
    setattr(index, "_global_context_audit", context_audit)
    setattr(index, "_global_context_policy", "frozen_snapshot_semantic_context.v1")
    setattr(index, "_historical_pair_count", len(historical_pairs))
    result = (index, dict(adjacency), len(records))
    setattr(engine, cache_name, result)
    return result


_FRONTIER_RELATIONS = {
    "case1_transdiagnostic": "distinguishes",
    "biomarker_discovery": "is_biomarker_of",
    "disease_subtyping": "distinguishes",
    "progression_prediction": "predicts",
    "imaging_genetics": "is_associated_with",
    "differential_diagnosis": "distinguishes",
    "connectome_behavior": "correlates_with",
    "brain_age": "predicts",
}


def _make_frontier_candidate_executable(
    hypothesis: dict[str, Any],
    *,
    case: CaseStudy,
    generation_round: int,
) -> dict[str, Any]:
    """Represent a novel endpoint pair as an explicit proposed edge."""

    candidate = dict(hypothesis)
    metadata = dict(candidate.get("metadata") or {})
    metadata.update(
        {
            "case_study_id": case.name,
            "generator_method": "neurodiscovery",
            "candidate_pool_branch": "task_scoped_frontier",
            "dynamic_generation_round": generation_round,
            "frontier_uses_future_outcomes": False,
        }
    )
    if not (candidate.get("path") or []):
        confidence = min(
            1.0,
            max(0.05, float(candidate.get("confidence_score") or 0.5)),
        )
        candidate["path"] = [
            {
                "from_id": str(candidate.get("source_id") or ""),
                "from_name": str(candidate.get("source_name") or ""),
                "to_id": str(candidate.get("target_id") or ""),
                "to_name": str(candidate.get("target_name") or ""),
                "relation_type": _FRONTIER_RELATIONS.get(case.name, "predicts"),
                "confidence": confidence,
                "claim_id": "",
                "raw_text": "",
                "evidence": {
                    "status": "proposed_relation",
                    "endpoint_evidence_only": True,
                },
                "source_paper": {},
            }
        ]
        metadata["proposed_endpoint_relation"] = True
    metadata["path_node_ids"] = list(
        _hypothesis_payload_semantic_key(candidate)[1]
    )
    candidate["metadata"] = metadata
    return candidate


def _frontier_feedback_mutation_affinity(
    candidate: dict[str, Any],
    feedback_state: FeedbackState | None,
) -> tuple[float, tuple[str, ...]]:
    """Match candidates that preserve either endpoint of a supported result."""

    if feedback_state is None or not feedback_state.records:
        return 0.0, ()
    source_id = str(candidate.get("source_id") or "")
    target_id = str(candidate.get("target_id") or "")
    affinity = 0.0
    anchors: list[str] = []
    for record in feedback_state.records:
        if record.status != SUPPORTED:
            continue
        source_match = bool(record.source_id and source_id == record.source_id)
        target_match = bool(record.target_id and target_id == record.target_id)
        if source_match == target_match:
            continue
        affinity = max(affinity, min(1.0, max(0.0, float(record.weight))))
        if record.hypothesis_id:
            anchors.append(record.hypothesis_id)
    return affinity, tuple(dict.fromkeys(anchors))


def _feedback_conditioned_endpoint_mutations(
    raw_candidates: list[dict[str, Any]],
    *,
    case: CaseStudy,
    feedback_state: FeedbackState | None,
    generation_round: int,
    max_candidates: int,
    frontier_index: FrozenGraphIndex | None = None,
) -> list[dict[str, Any]]:
    """Expand either supported endpoint over frozen, legally typed evidence."""

    if (
        max_candidates <= 0
        or case.task is None
        or len(case.task.inputs) != 1
        or feedback_state is None
    ):
        return []

    supported_sources: dict[str, tuple[float, list[str], set[str]]] = {}
    supported_targets: dict[str, tuple[float, list[str], set[str]]] = {}
    for record in feedback_state.records:
        if record.status != SUPPORTED:
            continue
        if record.source_id:
            weight, anchors, prior_targets = supported_sources.setdefault(
                record.source_id,
                (0.0, [], set()),
            )
            supported_sources[record.source_id] = (
                max(weight, min(1.0, max(0.0, float(record.weight)))),
                [*anchors, *([record.hypothesis_id] if record.hypothesis_id else [])],
                {*prior_targets, *([record.target_id] if record.target_id else [])},
            )
        if record.target_id:
            weight, anchors, prior_sources = supported_targets.setdefault(
                record.target_id,
                (0.0, [], set()),
            )
            supported_targets[record.target_id] = (
                max(weight, min(1.0, max(0.0, float(record.weight)))),
                [*anchors, *([record.hypothesis_id] if record.hypothesis_id else [])],
                {*prior_sources, *([record.source_id] if record.source_id else [])},
            )
    if not supported_sources and not supported_targets:
        return []

    names: dict[str, str] = {}
    source_templates: dict[str, dict[str, Any]] = {}
    target_templates: dict[str, dict[str, Any]] = {}
    for raw in raw_candidates:
        source_id = str(raw.get("source_id") or "")
        target_id = str(raw.get("target_id") or "")
        if source_id:
            names.setdefault(source_id, str(raw.get("source_name") or source_id))
            source_templates.setdefault(source_id, raw)
        if target_id:
            names.setdefault(target_id, str(raw.get("target_name") or target_id))
            target_templates.setdefault(target_id, raw)
    if frontier_index is not None:
        names.update(frontier_index.names)

    mutations: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    input_preserving_specs = (
        (
            source_id,
            target_id,
            template,
            weight,
            anchor_ids,
            "preserve_task_input",
        )
        for target_id, template in target_templates.items()
        for source_id, (weight, anchor_ids, prior_targets) in supported_sources.items()
        if target_id not in prior_targets
    )
    output_preserving_specs = (
        (
            source_id,
            target_id,
            template,
            weight,
            anchor_ids,
            "preserve_task_output",
        )
        for source_id, template in source_templates.items()
        for target_id, (weight, anchor_ids, prior_sources) in supported_targets.items()
        if source_id not in prior_sources
    )
    mutation_specs = itertools.chain(
        input_preserving_specs,
        output_preserving_specs,
    )
    for (
        source_id,
        target_id,
        template,
        weight,
        anchor_ids,
        mutation_direction,
    ) in mutation_specs:
        pair = tuple(sorted((source_id, target_id)))
        if (
            source_id == target_id
            or pair in seen_pairs
            or (
                frontier_index is not None
                and pair in frontier_index.direct_pairs
            )
        ):
            continue
        if frontier_index is not None and not case_study_endpoint_names_allowed(
            case.name,
            _endpoint(frontier_index, source_id),
            _endpoint(frontier_index, target_id),
        ):
            continue

        candidate = dict(template)
        metadata = dict(candidate.get("metadata") or {})
        preserve_input = mutation_direction == "preserve_task_input"
        source_claim_ids = (
            []
            if preserve_input
            else list(metadata.get("source_endpoint_claim_ids") or [])
        )
        source_paper_keys = (
            []
            if preserve_input
            else list(metadata.get("source_endpoint_paper_keys") or [])
        )
        target_claim_ids = (
            list(metadata.get("target_endpoint_claim_ids") or [])
            if preserve_input
            else []
        )
        target_paper_keys = (
            list(metadata.get("target_endpoint_paper_keys") or [])
            if preserve_input
            else []
        )
        endpoint_support = float(
            metadata.get(
                "target_endpoint_support_confidence"
                if preserve_input
                else "source_endpoint_support_confidence"
            )
            or candidate.get("evidence_score")
            or 0.5
        )
        confidence = min(
            1.0,
            0.55 * float(candidate.get("confidence_score") or 0.5)
            + 0.45 * weight,
        )
        candidate.update(
            {
                    "id": (
                        f"NEURODISCOVERY:{case.name}:FEEDBACK:"
                        f"{generation_round:02d}:{len(mutations) + 1:06d}"
                    ),
                    "hypothesis_type": "feedback_conditioned_endpoint_pair",
                    "source_id": source_id,
                    "source_name": names.get(source_id, source_id),
                    "target_id": target_id,
                    "target_name": names.get(target_id, target_id),
                    "path": [],
                    "confidence_score": confidence,
                    "evidence_score": min(1.0, max(0.0, endpoint_support)),
                    "composite_score": confidence,
                    "supporting_claims": list(
                        dict.fromkeys((*source_claim_ids, *target_claim_ids))
                    ),
                    "explanation": (
                        "An early supported endpoint motivates testing the "
                        f"task-compatible relation between {names.get(source_id, source_id)} "
                        f"and {names.get(target_id, target_id)}; the varied endpoint had "
                        "evidence in the frozen knowledge graph."
                    ),
                    "testability_reason": (
                        "The input endpoint was supported during the early feedback window; "
                        "the output endpoint and its type were fixed before the freeze date."
                    ),
            }
        )
        metadata.update(
            {
                    "generation_mode": "feedback_conditioned_endpoint_expansion",
                    "source_endpoint_claim_ids": source_claim_ids,
                    "source_endpoint_paper_keys": source_paper_keys,
                    "target_endpoint_claim_ids": target_claim_ids,
                    "target_endpoint_paper_keys": target_paper_keys,
                    "source_paper_keys": list(
                        dict.fromkeys((*source_paper_keys, *target_paper_keys))
                    ),
                    "feedback_mutation": True,
                    "feedback_mutation_direction": mutation_direction,
                    "feedback_mutation_affinity": weight,
                    "feedback_mutation_anchor_ids": list(dict.fromkeys(anchor_ids)),
                    "feedback_mutation_policy": (
                        "supported_bidirectional_endpoint_expansion.v3"
                    ),
                    "feedback_source_is_experiment_result": preserve_input,
                    "feedback_target_is_experiment_result": not preserve_input,
                    "feedback_source_uses_frozen_evidence": not preserve_input,
                    "feedback_target_uses_frozen_evidence": preserve_input,
                    "feedback_uses_terminal_outcomes": False,
                }
        )
        candidate["metadata"] = metadata
        mutations.append(
            _make_frontier_candidate_executable(
                candidate,
                case=case,
                generation_round=generation_round,
            )
        )
        seen_pairs.add(pair)
        if len(mutations) >= max_candidates:
            return mutations
    return mutations


def _interleave_frontier_feedback_mutations(
    mutations: list[tuple[dict[str, Any], float, tuple[str, ...]]],
    exploration: list[tuple[dict[str, Any], float, tuple[str, ...]]],
    *,
    target: int,
    mutation_fraction: float,
) -> list[tuple[dict[str, Any], float, tuple[str, ...]]]:
    """Reserve a bounded mutation quota and retain broad frontier exploration."""

    if target <= 0:
        return []
    mutation_target = min(
        len(mutations),
        max(0, int(round(target * mutation_fraction))),
    )
    selected_mutations = mutations[:mutation_target]
    selected_exploration = exploration[: max(0, target - mutation_target)]
    if len(selected_mutations) + len(selected_exploration) < target:
        selected_mutations.extend(
            mutations[
                mutation_target : mutation_target
                + target
                - len(selected_mutations)
                - len(selected_exploration)
            ]
        )
    total = min(target, len(selected_mutations) + len(selected_exploration))
    if not selected_mutations or not selected_exploration:
        return [*selected_exploration, *selected_mutations][:target]

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

    merged: list[tuple[dict[str, Any], float, tuple[str, ...]]] = []
    mutation_index = 0
    exploration_index = 0
    for position in range(total):
        if (
            position in mutation_positions
            and mutation_index < len(selected_mutations)
        ):
            merged.append(selected_mutations[mutation_index])
            mutation_index += 1
        elif exploration_index < len(selected_exploration):
            merged.append(selected_exploration[exploration_index])
            exploration_index += 1
        elif mutation_index < len(selected_mutations):
            merged.append(selected_mutations[mutation_index])
            mutation_index += 1
    return merged[:target]


def _merge_neurodiscovery_frontier_payload(
    payload: dict[str, Any],
    frontier_payload: dict[str, Any],
    *,
    case: CaseStudy,
    target: int,
    generation_round: int,
    excluded_semantic_keys: frozenset[tuple[str, tuple[str, ...]]],
    feedback_state: FeedbackState | None = None,
    feedback_mutation_fraction: float = 0.0,
    frontier_index: FrozenGraphIndex | None = None,
) -> dict[str, Any]:
    """Fill a scoped proposal batch with unique, executable frontier candidates."""

    single_endpoint_task = case.task is not None and len(case.task.inputs) == 1

    def identity(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
        path_nodes = _hypothesis_payload_semantic_key(row)[1]
        if single_endpoint_task:
            endpoints = tuple(
                sorted(
                    (
                        str(row.get("source_id") or ""),
                        str(row.get("target_id") or ""),
                    )
                )
            )
            return "endpoint_pair", endpoints
        return "path", path_nodes

    existing = [
        dict(row)
        for row in payload.get("hypotheses") or []
        if row.get("hypothesis_type") != "generation_failure"
        and not (row.get("metadata") or {}).get("generation_failure")
    ]
    seen_identities = {identity(row) for row in existing}
    if single_endpoint_task:
        excluded_identities = {
            ("endpoint_pair", tuple(sorted((nodes[0], nodes[-1]))))
            for _, nodes in excluded_semantic_keys
            if len(nodes) >= 2
        }
    else:
        excluded_identities = {
            ("path", nodes) for _, nodes in excluded_semantic_keys
        }
    rejected_duplicate = 0
    eligible_seen = set(seen_identities)
    mutations: list[tuple[dict[str, Any], float, tuple[str, ...]]] = []
    exploration: list[tuple[dict[str, Any], float, tuple[str, ...]]] = []
    raw_frontier = list(frontier_payload.get("hypotheses") or [])
    conditioned_mutations = _feedback_conditioned_endpoint_mutations(
        raw_frontier,
        case=case,
        feedback_state=feedback_state,
        generation_round=generation_round,
        max_candidates=max(target * 4, target),
        frontier_index=frontier_index,
    )
    for raw in [*conditioned_mutations, *raw_frontier]:
        if raw.get("hypothesis_type") == "generation_failure":
            continue
        candidate = _make_frontier_candidate_executable(
            raw,
            case=case,
            generation_round=generation_round,
        )
        candidate_identity = identity(candidate)
        if (
            candidate_identity in eligible_seen
            or candidate_identity in excluded_identities
        ):
            rejected_duplicate += 1
            continue
        eligible_seen.add(candidate_identity)
        affinity, anchors = _frontier_feedback_mutation_affinity(
            candidate,
            feedback_state,
        )
        row = (candidate, affinity, anchors)
        if affinity > 0.0:
            mutations.append(row)
        else:
            exploration.append(row)

    available = max(0, target - len(existing))
    selected = _interleave_frontier_feedback_mutations(
        mutations,
        exploration,
        target=available,
        mutation_fraction=feedback_mutation_fraction,
    )
    added = 0
    feedback_mutations_added = 0
    for candidate, affinity, anchors in selected:
        candidate["id"] = f"NEURODISCOVERY:{case.name}:FRONTIER:{len(existing) + 1:06d}"
        if affinity > 0.0:
            metadata = dict(candidate.get("metadata") or {})
            metadata.update(
                {
                    "feedback_mutation": True,
                    "feedback_mutation_affinity": affinity,
                    "feedback_mutation_anchor_ids": list(anchors),
                    "feedback_mutation_policy": (
                        "supported_single_endpoint_replacement.v1"
                    ),
                }
            )
            candidate["metadata"] = metadata
            feedback_mutations_added += 1
        existing.append(candidate)
        added += 1

    metadata = dict(payload.get("metadata") or {})
    metadata["neurodiscovery_evidence_frontier"] = {
        "enabled": True,
        "uses_future_outcomes": False,
        "requested_batch_size": target,
        "valid_before_frontier": len(existing) - added,
        "frontier_candidates_added": added,
        "frontier_duplicates_or_previous_rounds_rejected": rejected_duplicate,
        "feedback_records_available": (
            len(feedback_state.records) if feedback_state is not None else 0
        ),
        "feedback_mutation_candidates_available": len(mutations),
        "feedback_mutation_candidates_added": feedback_mutations_added,
        "feedback_conditioned_candidates_generated": len(
            conditioned_mutations
        ),
        "feedback_mutation_fraction": feedback_mutation_fraction,
        "valid_after_frontier": len(existing),
        "generation_round": generation_round,
    }
    return {
        **payload,
        "n_hypotheses": len(existing),
        "hypotheses": existing,
        "metadata": metadata,
    }


def _augment_scoped_frontier(
    engine: HypothesisEngine,
    output: Path,
    *,
    case: CaseStudy,
    freeze_year: int,
    target: int,
    seed: int,
    replicate_index: int,
    generation_round: int,
    excluded_semantic_keys: frozenset[tuple[str, tuple[str, ...]]],
    feedback_state: FeedbackState | None = None,
    feedback_mutation_fraction: float = 0.0,
    evidence_frontier_fraction: float = 0.0,
    endpoint_canonical_quality_weight: float = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
    ),
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
    kge_checkpoint: str | None = None,
) -> None:
    if not 0.0 <= evidence_frontier_fraction <= 1.0:
        raise ValueError("evidence_frontier_fraction must be in [0, 1]")
    payload = json.loads(output.read_text(encoding="utf-8"))
    valid_rows = [
        dict(row)
        for row in payload.get("hypotheses") or []
        if row.get("hypothesis_type") != "generation_failure"
        and not (row.get("metadata") or {}).get("generation_failure")
    ]
    valid_count = len(valid_rows)
    if valid_count >= target and evidence_frontier_fraction <= 0.0:
        metadata = dict(payload.get("metadata") or {})
        metadata["neurodiscovery_evidence_frontier"] = {
            "enabled": True,
            "uses_future_outcomes": False,
            "requested_batch_size": target,
            "valid_before_frontier": valid_count,
            "frontier_candidates_added": 0,
            "valid_after_frontier": valid_count,
            "generation_round": generation_round,
            "evidence_frontier_fraction": evidence_frontier_fraction,
            "endpoint_canonical_quality_policy": (
                ENDPOINT_CANONICAL_QUALITY_POLICY
            ),
            "endpoint_canonical_quality_weight": (
                endpoint_canonical_quality_weight
            ),
        }
        payload["metadata"] = metadata
        _atomic_write_json(output, payload)
        return

    index, adjacency, scoped_claim_count = _frontier_index_for_case(engine, case)
    depth_multiplier = (
        2
        if evidence_frontier_fraction > 0.0
        else min(8, max(2, generation_round + 2))
    )
    dynamic_scoped = (
        (payload.get("metadata") or {}).get("generation_mode")
        == "dynamic_scoped_evidence_frontier"
    )
    requested_frontier = int(round(target * evidence_frontier_fraction))
    if evidence_frontier_fraction > 0.0 and not dynamic_scoped:
        # Keep a deterministic connected-path quota and replace only the
        # requested tail with independently grounded endpoint compositions.
        keep_connected = max(0, target - requested_frontier)
        payload = {
            **payload,
            "n_hypotheses": min(len(valid_rows), keep_connected),
            "hypotheses": valid_rows[:keep_connected],
        }
    frontier_payload = generate_case_hypotheses(
        method="neurodiscovery",
        case=case,
        index=index,
        adjacency=adjacency,
        freeze_year=freeze_year,
        target_count=max(target, target * depth_multiplier),
        seed=seed,
        evidence_frontier_fraction=(
            evidence_frontier_fraction
            if dynamic_scoped
            else (1.0 if evidence_frontier_fraction > 0.0 else 0.0)
        ),
        # Apply the quality prior once after branch generation so connected and
        # composed candidates share the same auditable formula.
        endpoint_canonical_quality_weight=0.0,
        feedback_anchor_pairs=tuple(
            (record.source_id, record.target_id)
            for record in (feedback_state.records if feedback_state is not None else ())
            if record.status == SUPPORTED and record.source_id and record.target_id
        ),
        feedback_anchor_fraction=feedback_mutation_fraction,
        kge_scorer=kge_scorer,
        kge_weight=kge_weight,
        kge_checkpoint=kge_checkpoint,
        replicate_index=replicate_index,
        generation_round=generation_round,
    )
    merged = _merge_neurodiscovery_frontier_payload(
        payload,
        frontier_payload,
        case=case,
        target=target,
        generation_round=generation_round,
        excluded_semantic_keys=excluded_semantic_keys,
        feedback_state=feedback_state,
        feedback_mutation_fraction=feedback_mutation_fraction,
        frontier_index=index,
    )
    metadata = dict(merged.get("metadata") or {})
    metadata["neurodiscovery_evidence_frontier"]["scoped_claim_count"] = (
        scoped_claim_count
    )
    metadata["neurodiscovery_evidence_frontier"]["candidate_depth_multiplier"] = (
        depth_multiplier
    )
    metadata["neurodiscovery_evidence_frontier"]["evidence_frontier_fraction"] = (
        evidence_frontier_fraction
    )
    metadata["neurodiscovery_evidence_frontier"]["requested_frontier_candidates"] = (
        requested_frontier
    )
    context_audit = getattr(index, "_global_context_audit", {})
    metadata["neurodiscovery_evidence_frontier"]["context_policy"] = getattr(
        index,
        "_global_context_policy",
        "task_scoped_only",
    )
    metadata["neurodiscovery_evidence_frontier"]["global_context_audit"] = dict(
        context_audit
    )
    metadata["neurodiscovery_evidence_frontier"]["historical_pair_count"] = int(
        getattr(index, "_historical_pair_count", 0)
    )
    metadata["neurodiscovery_evidence_frontier"]["global_context_uses_future_outcomes"] = (
        False
    )
    metadata["neurodiscovery_evidence_frontier"]["endpoint_canonical_quality_policy"] = (
        ENDPOINT_CANONICAL_QUALITY_POLICY
    )
    metadata["neurodiscovery_evidence_frontier"]["endpoint_canonical_quality_weight"] = (
        endpoint_canonical_quality_weight
    )
    merged["metadata"] = metadata
    _atomic_write_json(output, merged)


def _valid_unique_candidates(
    payload: dict[str, Any],
    *,
    branch: str,
    seen: set[tuple[str, tuple[str, ...]]] | None = None,
) -> list[dict[str, Any]]:
    """Return valid semantic-unique candidates with auditable branch provenance."""

    seen = seen if seen is not None else set()
    unique: list[dict[str, Any]] = []
    for raw in payload.get("hypotheses") or []:
        if raw.get("hypothesis_type") == "generation_failure":
            continue
        if (raw.get("metadata") or {}).get("generation_failure") is True:
            continue
        key = _hypothesis_payload_semantic_key(raw)
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(raw)
        metadata = dict(candidate.get("metadata") or {})
        metadata["candidate_pool_branch"] = branch
        candidate["metadata"] = metadata
        unique.append(candidate)
    return unique


def _merge_hybrid_candidate_payloads(
    general_payload: dict[str, Any],
    scoped_payload: dict[str, Any],
    *,
    case_study_id: str,
    max_candidates: int,
    task_scope_fraction: float,
    protect_general_top_k: int,
) -> dict[str, Any]:
    """Merge broad KG search with a small task-evidence quota.

    General candidates keep priority and the protected prefix is unchanged.  A
    deterministic quota of task-scoped candidates is spread over the remaining
    ranks.  This preserves the broad open-source KG search while making the
    injected case-study evidence available to the closed-loop policy.
    """

    seen: set[tuple[str, tuple[str, ...]]] = set()
    general = _valid_unique_candidates(
        general_payload, branch="general_graph", seen=seen
    )
    scoped = _valid_unique_candidates(
        scoped_payload, branch="task_scoped", seen=seen
    )
    planned_size = min(max_candidates, len(general) + len(scoped))
    protected_count = min(protect_general_top_k, len(general), planned_size)

    if task_scope_fraction <= 0.0 or not scoped:
        scoped_target = 0
    elif task_scope_fraction >= 1.0:
        scoped_target = min(len(scoped), planned_size - protected_count)
    else:
        desired = round(planned_size * task_scope_fraction)
        scoped_target = min(len(scoped), desired, planned_size - protected_count)
    general_target = min(len(general), planned_size - scoped_target)
    if general_target + scoped_target < planned_size:
        scoped_target = min(len(scoped), planned_size - general_target)

    protected = general[: min(protected_count, general_target)]
    remaining_general = general[len(protected):general_target]
    selected_scoped = scoped[:scoped_target]

    tail_size = len(remaining_general) + len(selected_scoped)
    scope_positions: set[int] = set()
    if selected_scoped and tail_size:
        scope_positions = {
            min(tail_size - 1, round((index + 1) * (tail_size + 1) / (scoped_target + 1)) - 1)
            for index in range(scoped_target)
        }
        # Rounding can collide for very small tails; fill any missing positions.
        for position in range(tail_size - 1, -1, -1):
            if len(scope_positions) >= scoped_target:
                break
            scope_positions.add(position)

    merged = list(protected)
    general_index = 0
    scoped_index = 0
    for position in range(tail_size):
        use_scoped = position in scope_positions and scoped_index < len(selected_scoped)
        if use_scoped:
            merged.append(selected_scoped[scoped_index])
            scoped_index += 1
        elif general_index < len(remaining_general):
            merged.append(remaining_general[general_index])
            general_index += 1
        elif scoped_index < len(selected_scoped):
            merged.append(selected_scoped[scoped_index])
            scoped_index += 1
    merged = merged[:max_candidates]

    for rank, hypothesis in enumerate(merged, start=1):
        hypothesis["id"] = f"NEURODISCOVERY:{case_study_id}:{rank:06d}"
        metadata = dict(hypothesis.get("metadata") or {})
        metadata["case_study_id"] = case_study_id
        metadata["hybrid_rank"] = rank
        hypothesis["metadata"] = metadata

    metadata = dict(general_payload.get("metadata") or {})
    metadata["candidate_pool"] = {
        "mode": "hybrid_general_plus_task_scope",
        "general_unique": len(general),
        "scoped_unique_after_cross_pool_dedup": len(scoped),
        "scoped_included": len(selected_scoped),
        "task_scope_fraction_requested": task_scope_fraction,
        "protect_general_top_k": protected_count,
        "max_candidates": max_candidates,
        "ranking_policy": (
            "preserve the general prefix, then deterministically interleave a "
            "small task-scoped quota while preserving within-branch rank"
        ),
    }
    return {
        "n_hypotheses": len(merged),
        "hypotheses": merged,
        "metadata": metadata,
    }


def _scoped_frontier_seed_mode(
    *,
    task_scoped: bool,
    dynamic_generation: bool,
    evidence_frontier_fraction: float,
) -> str | None:
    if not task_scoped:
        return None
    if dynamic_generation:
        return "dynamic_scoped_evidence_frontier"
    if evidence_frontier_fraction >= 1.0:
        return "static_scoped_evidence_frontier"
    return None


def _generate_branch(
    *,
    graph: Any,
    case: CaseStudy,
    output: Path,
    seed: int,
    freeze_year: int,
    target: int,
    seed_diversity_fraction: float,
    task_scoped: bool,
    replicate_index: int = 0,
    feedback_state: FeedbackState | None = None,
    engine: HypothesisEngine | None = None,
    dynamic_generation: bool = False,
    generation_round: int = 0,
    excluded_semantic_keys: frozenset[tuple[str, tuple[str, ...]]] = frozenset(),
    path_variants_per_endpoint: int = 2,
    feedback_mutation_fraction: float = 0.35,
    max_paths_per_endpoint: int = 4,
    evidence_frontier_fraction: float = 0.0,
    endpoint_canonical_quality_weight: float = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
    ),
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
    kge_checkpoint: str | None = None,
) -> HypothesisEngine:
    random.seed(seed)
    np.random.seed(seed)
    if engine is None:
        engine = HypothesisEngine(graph)
        if task_scoped and case.task is not None:
            engine.set_task_claim_scope(case.name)
        for hook in case.pre_hooks:
            hook(engine, case)
    engine.feedback_state = feedback_state
    engine.configure_dynamic_generation(
        enabled=dynamic_generation,
        exploration_round=generation_round,
        excluded_paths=(key[1] for key in excluded_semantic_keys),
        path_variants_per_endpoint=path_variants_per_endpoint,
        feedback_mutation_fraction=feedback_mutation_fraction,
        max_paths_per_endpoint=max_paths_per_endpoint,
    )
    scoped_seed_mode = _scoped_frontier_seed_mode(
        task_scoped=task_scoped,
        dynamic_generation=dynamic_generation,
        evidence_frontier_fraction=evidence_frontier_fraction,
    )
    if scoped_seed_mode is not None:
        # Dynamic rounds and a static 100% endpoint-frontier branch do not retain
        # any candidates from cmd_batch. Seed the auditable frontier directly.
        _atomic_write_json(
            output,
            {
                "n_hypotheses": 0,
                "hypotheses": [],
                "metadata": {
                    "generation_mode": scoped_seed_mode,
                    "connected_path_generation_skipped": True,
                    "uses_future_outcomes": False,
                },
            },
        )
    else:
        batch = case.stage_params.batch
        cmd_batch(
            engine,
            str(output),
            max_hops=batch.max_hops,
            min_hops=batch.min_hops,
            metapath_min_domains=batch.metapath_min_domains,
            max_paths=batch.max_paths,
            max_seeds=batch.max_seeds,
            target_per_task=target,
            max_retries=batch.max_retries,
            retry_scale=batch.retry_scale,
            prefer_longer_paths=batch.prefer_longer_paths,
            task_filter=case.task.name if case.task is not None and case.chain is None else "",
            chain_filter=case.chain.name if case.chain is not None else "",
            as_json=False,
            random_seed=seed,
            seed_diversity_fraction=seed_diversity_fraction,
            print_top_n=0,
        )
    if task_scoped:
        _augment_scoped_frontier(
            engine,
            output,
            case=case,
            freeze_year=freeze_year,
            target=target,
            seed=seed,
            replicate_index=replicate_index,
            generation_round=generation_round,
            excluded_semantic_keys=excluded_semantic_keys,
            feedback_state=feedback_state,
            feedback_mutation_fraction=feedback_mutation_fraction,
            evidence_frontier_fraction=evidence_frontier_fraction,
            endpoint_canonical_quality_weight=(
                endpoint_canonical_quality_weight
            ),
            kge_scorer=kge_scorer,
            kge_weight=kge_weight,
            kge_checkpoint=kge_checkpoint,
        )
    _tag(engine, output, case)
    quality_adjusted = _apply_candidate_canonical_quality(
        engine._index,
        json.loads(output.read_text(encoding="utf-8")),
        weight=endpoint_canonical_quality_weight,
    )
    _atomic_write_json(output, quality_adjusted)
    return engine


def generate_one(
    *, graph: Any, case: CaseStudy, output: Path, seed: int,
    freeze_year: int, target: int, fixed_budget: int, force: bool,
    seed_diversity_fraction: float,
    candidate_pool_mode: str = "hybrid",
    task_scope_fraction: float = 0.15,
    protect_general_top_k: int = 100,
    static_score_family: str = "legacy",
    feedback_state: FeedbackState | None = None,
    branch_engines: dict[str, HypothesisEngine] | None = None,
    dynamic_generation: bool = False,
    replicate_index: int = 0,
    generation_round: int = 0,
    excluded_semantic_keys: frozenset[tuple[str, tuple[str, ...]]] = frozenset(),
    path_variants_per_endpoint: int = 2,
    feedback_mutation_fraction: float = 0.35,
    max_paths_per_endpoint: int = 4,
    evidence_frontier_fraction: float = 0.0,
    endpoint_canonical_quality_weight: float = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
    ),
    kge_scorer: Any = None,
    kge_weight: float = 0.0,
    kge_checkpoint: str | None = None,
) -> dict[str, Any]:
    if static_score_family not in {"legacy", "relation_aware"}:
        raise ValueError(f"unknown static score family: {static_score_family!r}")
    if output.is_file() and not force:
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if _is_complete_fixed_budget_payload(
            existing,
            case_study_id=case.name,
            target_per_case_study=fixed_budget,
        ):
            existing_family = (
                (existing.get("metadata") or {})
                .get("relation_aware_static_policy", {})
                .get("name")
            )
            if static_score_family == "legacy" or existing_family:
                return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    use_hybrid = candidate_pool_mode == "hybrid" and case.task is not None
    if use_hybrid:
        branch_dir = output.parent / "candidate_branches"
        branch_dir.mkdir(parents=True, exist_ok=True)
        general_output = branch_dir / "general_graph.json"
        scoped_output = branch_dir / "task_scoped.json"
        engine = _generate_branch(
            graph=graph,
            case=case,
            output=general_output,
            seed=seed,
            freeze_year=freeze_year,
            target=target,
            seed_diversity_fraction=seed_diversity_fraction,
            task_scoped=False,
            replicate_index=replicate_index,
            feedback_state=feedback_state,
            engine=(branch_engines or {}).get("general_graph"),
            dynamic_generation=dynamic_generation,
            generation_round=generation_round,
            excluded_semantic_keys=excluded_semantic_keys,
            path_variants_per_endpoint=path_variants_per_endpoint,
            feedback_mutation_fraction=feedback_mutation_fraction,
            max_paths_per_endpoint=max_paths_per_endpoint,
            evidence_frontier_fraction=evidence_frontier_fraction,
            endpoint_canonical_quality_weight=endpoint_canonical_quality_weight,
            kge_scorer=kge_scorer,
            kge_weight=kge_weight,
            kge_checkpoint=kge_checkpoint,
        )
        if branch_engines is not None:
            branch_engines["general_graph"] = engine
        scoped_engine = _generate_branch(
            graph=graph,
            case=case,
            output=scoped_output,
            seed=seed,
            freeze_year=freeze_year,
            target=target,
            seed_diversity_fraction=seed_diversity_fraction,
            task_scoped=True,
            replicate_index=replicate_index,
            feedback_state=feedback_state,
            engine=(branch_engines or {}).get("task_scoped"),
            dynamic_generation=dynamic_generation,
            generation_round=generation_round,
            excluded_semantic_keys=excluded_semantic_keys,
            path_variants_per_endpoint=path_variants_per_endpoint,
            feedback_mutation_fraction=feedback_mutation_fraction,
            max_paths_per_endpoint=max_paths_per_endpoint,
            evidence_frontier_fraction=evidence_frontier_fraction,
            endpoint_canonical_quality_weight=endpoint_canonical_quality_weight,
            kge_scorer=kge_scorer,
            kge_weight=kge_weight,
            kge_checkpoint=kge_checkpoint,
        )
        if branch_engines is not None:
            branch_engines["task_scoped"] = scoped_engine
        payload = _merge_hybrid_candidate_payloads(
            json.loads(general_output.read_text(encoding="utf-8")),
            json.loads(scoped_output.read_text(encoding="utf-8")),
            case_study_id=case.name,
            max_candidates=target,
            task_scope_fraction=task_scope_fraction,
            protect_general_top_k=protect_general_top_k,
        )
        _atomic_write_json(output, payload)
    else:
        branch_name = "general_graph" if candidate_pool_mode == "general" else "task_scoped"
        engine = _generate_branch(
            graph=graph,
            case=case,
            output=output,
            seed=seed,
            freeze_year=freeze_year,
            target=target,
            seed_diversity_fraction=seed_diversity_fraction,
            task_scoped=candidate_pool_mode != "general",
            replicate_index=replicate_index,
            feedback_state=feedback_state,
            engine=(branch_engines or {}).get(branch_name),
            dynamic_generation=dynamic_generation,
            generation_round=generation_round,
            excluded_semantic_keys=excluded_semantic_keys,
            path_variants_per_endpoint=path_variants_per_endpoint,
            feedback_mutation_fraction=feedback_mutation_fraction,
            max_paths_per_endpoint=max_paths_per_endpoint,
            evidence_frontier_fraction=evidence_frontier_fraction,
            endpoint_canonical_quality_weight=endpoint_canonical_quality_weight,
            kge_scorer=kge_scorer,
            kge_weight=kge_weight,
            kge_checkpoint=kge_checkpoint,
        )
        if branch_engines is not None:
            branch_engines[branch_name] = engine
    for hook in case.post_hooks:
        hook(engine, case, output)
    if static_score_family == "relation_aware":
        rescored = annotate_relation_aware_payload(
            json.loads(output.read_text(encoding="utf-8")),
            graph=engine.G,
            case_study_id=case.name,
            use_path_pairs=case.chain is not None,
        )
        _atomic_write_json(output, rescored)
    return _enforce_fixed_budget(
        json.loads(output.read_text(encoding="utf-8")),
        output_path=output,
        target_per_case_study=fixed_budget,
        case_study_id=case.name,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    eligibility = (
        load_locked_hindcasting_eligibility(args.eligibility_manifest)
        if args.eligibility_manifest is not None
        else None
    )
    case_ids = (
        args.case_study_ids
        or (
            list(eligibility.primary_case_study_ids)
            if eligibility is not None
            else list_case_study_names()
        )
    )
    cases = tuple(case_study_by_name(name) for name in case_ids)
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
        graph_path = args.snapshot_root / f"kg_{window.freeze_year}" / "knowledge_graph.json"
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        print(f"[load] NeuroDiscovery KG_{window.freeze_year}", flush=True)
        graph = load_graph(graph_path)
        for seed in args.seeds:
            # Reuse branch indexes within one replicate only. Mutable lazy caches
            # must never make one seed depend on a previously executed seed.
            branch_engines_by_case: dict[str, dict[str, HypothesisEngine]] = {}
            for case in eligible_cases:
                output = (
                    args.output_root / f"seed_{seed:02d}" / case.name
                    / window.label / "hypotheses_raw.json"
                )
                payload = generate_one(
                    graph=graph,
                    case=case,
                    output=output,
                    seed=seed,
                    freeze_year=window.freeze_year,
                    target=args.generation_pool_size,
                    fixed_budget=args.target_per_case_study,
                    force=args.force,
                    seed_diversity_fraction=args.seed_diversity_fraction,
                    candidate_pool_mode=args.candidate_pool_mode,
                    task_scope_fraction=args.task_scope_fraction,
                    evidence_frontier_fraction=args.evidence_frontier_fraction,
                    endpoint_canonical_quality_weight=(
                        args.endpoint_canonical_quality_weight
                    ),
                    protect_general_top_k=args.protect_general_top_k,
                    static_score_family=args.static_score_family,
                    branch_engines=branch_engines_by_case.setdefault(case.name, {}),
                    replicate_index=seed,
                )
                fixed = (payload.get("metadata") or {}).get("fixed_budget") or {}
                rows.append({
                    "method": "neurodiscovery",
                    "seed": seed,
                    "case_study_id": case.name,
                    "freeze_year": window.freeze_year,
                    "future_start_year": window.future_start_year,
                    "future_end_year": window.future_end_year,
                    "kg_path": str(graph_path),
                    "historical_claims_path": str(graph_path.parent / "extracted_claims.jsonl"),
                    "hypotheses_path": str(output),
                    "n_hypotheses": len(payload.get("hypotheses") or []),
                    "valid_before_padding": fixed.get("valid_before_padding"),
                })
                print(
                    f"[neurodiscovery] seed={seed} {case.name} KG_{window.freeze_year}: "
                    f"{len(payload.get('hypotheses') or [])}",
                    flush=True,
                )
            del branch_engines_by_case
        del graph
    shard_id = _manifest_shard_id(args, cases)
    manifest = {
        "schema_version": "neurodiscovery-hindcasting-replicates.v1",
        "shard_id": shard_id,
        "method": "neurodiscovery",
        "snapshot_root": str(args.snapshot_root),
        "seeds": list(args.seeds),
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
        "generation_pool_size": args.generation_pool_size,
        "seed_diversity_fraction": args.seed_diversity_fraction,
        "candidate_pool_mode": args.candidate_pool_mode,
        "task_scope_fraction": args.task_scope_fraction,
        "evidence_frontier_fraction": args.evidence_frontier_fraction,
        "endpoint_canonical_quality_policy": ENDPOINT_CANONICAL_QUALITY_POLICY,
        "endpoint_canonical_quality_weight": (
            args.endpoint_canonical_quality_weight
        ),
        "protect_general_top_k": args.protect_general_top_k,
        "static_score_family": args.static_score_family,
        "engine_reuse_scope": "freeze_year_seed_case_study_branch",
        "eligibility_manifest": (
            str(eligibility.manifest_path) if eligibility is not None else None
        ),
        "eligibility_manifest_sha256": (
            sha256_file(eligibility.manifest_path)
            if eligibility is not None
            else None
        ),
        "eligibility_matrix_sha256": (
            eligibility.matrix_sha256 if eligibility is not None else None
        ),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unrecorded"),
        "runs": rows,
    }
    _atomic_write_json(
        args.output_root / f"generation_manifest_shard_{shard_id}.json",
        manifest,
    )
    return merge_generation_manifests(args.output_root)


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
    parser.add_argument("--output-root", type=Path, default=ROOT / "neurooracle/data/experiments/hindcasting/neurodiscovery_replicates_current")
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
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--target-per-case-study", type=int, default=1000)
    parser.add_argument("--generation-pool-size", type=int, default=1200)
    parser.add_argument(
        "--seed-diversity-fraction",
        type=float,
        default=0.35,
        help=(
            "Fraction of task source anchors drawn from a stable, "
            "evidence-weighted stochastic tail; 0 preserves legacy ordering."
        ),
    )
    parser.add_argument(
        "--candidate-pool-mode",
        choices=("hybrid", "general", "scoped"),
        default="hybrid",
        help=(
            "hybrid preserves broad KG search and injects a small task-scoped "
            "candidate quota; general/scoped are ablations"
        ),
    )
    parser.add_argument("--task-scope-fraction", type=float, default=0.15)
    parser.add_argument(
        "--evidence-frontier-fraction",
        type=float,
        default=0.0,
        help=(
            "Global fraction of each task-scoped batch reserved for novel "
            "endpoint compositions grounded independently before the freeze year."
        ),
    )
    parser.add_argument(
        "--endpoint-canonical-quality-weight",
        type=float,
        default=NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT,
        help=(
            "Outcome-blind multiplicative prior favouring frozen endpoints with "
            "controlled-vocabulary and identifier metadata; 0 disables it."
        ),
    )
    parser.add_argument("--protect-general-top-k", type=int, default=100)
    parser.add_argument(
        "--static-score-family",
        choices=("legacy", "relation_aware"),
        default="legacy",
        help="Outcome-blind ranking applied after frozen-KG candidate generation.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing process shards without generating hypotheses.",
    )
    args = parser.parse_args()
    if args.generation_pool_size <= args.target_per_case_study:
        parser.error("--generation-pool-size must exceed --target-per-case-study")
    if not 0.0 <= args.seed_diversity_fraction <= 1.0:
        parser.error("--seed-diversity-fraction must be in [0, 1]")
    if not 0.0 <= args.task_scope_fraction <= 1.0:
        parser.error("--task-scope-fraction must be in [0, 1]")
    if not 0.0 <= args.evidence_frontier_fraction <= 1.0:
        parser.error("--evidence-frontier-fraction must be in [0, 1]")
    if not 0.0 <= args.endpoint_canonical_quality_weight <= 1.0:
        parser.error("--endpoint-canonical-quality-weight must be in [0, 1]")
    if args.protect_general_top_k < 0:
        parser.error("--protect-general-top-k must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    manifest = (
        merge_generation_manifests(args.output_root) if args.merge_only else run(args)
    )
    print(json.dumps({"output_root": str(args.output_root), "runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()


# Updated: 2026-08-12 15:14:00 HKT - bound mixed-frontier depth while preserving seeded round-to-round exploration.
# Updated: 2026-08-13 07:27:39 HKT - shorten atomic temporary names for deep Windows hindcasting output trees.
# Updated: 2026-08-12 19:55 HKT - project supported feedback into bounded endpoint-frontier mutations.
# Updated: 2026-08-12 20:34 HKT - expand supported task inputs over frozen output evidence in the dynamic loop.
# Updated: 2026-08-12 22:28 HKT - thread the frozen KGE pair prior through scoped dynamic generation.
# Updated: 2026-08-13 01:38 HKT - keep KGE exploration shards complementary across replicates and rounds.
# Updated: 2026-08-13 04:16 HKT - expand early supported results along either frozen endpoint instead of one source-only branch.
