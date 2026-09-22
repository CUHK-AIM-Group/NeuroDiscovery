from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

import numpy as np

from neurooracle.src.case_studies import case_study_by_name
from neurooracle.src.case_study_scope import claim_case_study_ids_from_dict
from neurooracle.src.case_study_relation_contracts import (
    case_study_pair_allowed,
    requires_complete_path,
)
from neurooracle.src.claim_semantics import (
    SEMANTIC_ENDPOINT_IDENTITY_VERSION,
    SemanticEndpoint,
    concept_atom_roles,
    semantic_claim_pair,
)

try:
    from .temporal_hindcasting_core import _claim_year, _role_for_node, load_kg_index
except ImportError:  # Direct script execution.
    from temporal_hindcasting_core import _claim_year, _role_for_node, load_kg_index


SEMANTIC_PROJECTION_VERSION = SEMANTIC_ENDPOINT_IDENTITY_VERSION
CASE_STUDY_RELATION_CONTRACT_VERSION = "case_study_relation_contracts.v3"
TEMPORAL_ENDPOINT_CONTRACT_VERSION = "frozen_endpoint_atom_availability.v3"
DISCOVERY_METRIC_CONTRACT_VERSION = "unique_future_discoveries.v1"
MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK = 10


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_paper(claim: dict[str, Any]) -> dict[str, Any]:
    paper = claim.get("source_paper")
    return paper if isinstance(paper, dict) else {}


def _claim_case_study_ids(claim: dict[str, Any]) -> frozenset[str]:
    """Return claim-level scopes so unrelated claims cannot borrow paper scope."""

    return frozenset(claim_case_study_ids_from_dict(claim))


def load_future_claim_records(
    claims_path: Path,
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    case_study_ids: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Load a compact, case-study-scoped future-claim index for matrix runs."""

    requested_case_study_ids = (
        frozenset(case_study_ids) if case_study_ids is not None else None
    )
    records: list[dict[str, Any]] = []
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            claim = json.loads(line)
            if claim.get("negated"):
                continue
            year = _claim_year(claim)
            if year is None or (min_year is not None and year < min_year):
                continue
            if max_year is not None and year > max_year:
                continue
            claim_scopes = _claim_case_study_ids(claim)
            if not claim_scopes:
                continue
            if requested_case_study_ids is not None and not (
                claim_scopes & requested_case_study_ids
            ):
                continue
            paper = _safe_paper(claim)
            metadata = claim.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            records.append(
                {
                    "id": claim.get("id"),
                    "year": year,
                    "case_study_ids": claim_scopes,
                    "subject_id": str(claim.get("subject_id") or ""),
                    "subject_name": claim.get("subject_name"),
                    "subject_type": claim.get("subject_type") or metadata.get("subject_type"),
                    "object_id": str(claim.get("object_id") or ""),
                    "object_name": claim.get("object_name"),
                    "object_type": claim.get("object_type") or metadata.get("object_type"),
                    "predicate": claim.get("predicate"),
                    "raw_text": claim.get("raw_text"),
                    "pmid": paper.get("pmid"),
                    "doi": paper.get("doi"),
                    "title": paper.get("title"),
                    "journal": paper.get("journal"),
                }
            )
    return records


def _node_roles(concepts: dict[str, dict[str, Any]], node_id: str) -> tuple[str, ...]:
    node = concepts.get(node_id)
    if not node:
        return ()
    role = _role_for_node(node)
    if role:
        return (role,)
    tags = node.get("domain_tags") or []
    return tuple(str(tag) for tag in tags[:2])


def _edge_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _hyp_path_edges(hyp: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for link in hyp.get("path") or []:
        src = str(link.get("from_id") or "")
        dst = str(link.get("to_id") or "")
        if src and dst and src != dst:
            out.append((src, dst))
    return out


def _hyp_endpoint_pair(hyp: dict[str, Any]) -> tuple[str, str] | None:
    src = str(hyp.get("source_id") or "")
    dst = str(hyp.get("target_id") or "")
    if src and dst and src != dst:
        return src, dst
    edges = _hyp_path_edges(hyp)
    if edges:
        return edges[0][0], edges[-1][1]
    return None


def _hyp_signature(concepts: dict[str, dict[str, Any]], hyp: dict[str, Any]) -> str:
    roles: list[str] = []
    endpoint = _hyp_endpoint_pair(hyp)
    if endpoint:
        roles.extend("/".join(_node_roles(concepts, endpoint[0]) or ("unknown",)).split("|"))
    for src, dst in _hyp_path_edges(hyp):
        if not roles:
            roles.extend(_node_roles(concepts, src) or ("unknown",))
        roles.extend(_node_roles(concepts, dst) or ("unknown",))
    clean: list[str] = []
    for role in roles:
        if role and (not clean or clean[-1] != role):
            clean.append(role)
    return " -> ".join(clean[:8]) if clean else "unknown"


def _validate_hindcasting_hypotheses(
    hypotheses: list[dict[str, Any]],
    case_study_id: str | None,
) -> None:
    """Reject experiment-only identifiers that future claims cannot contain."""

    if case_study_id != "case1_transdiagnostic":
        return
    invalid = [
        str(hypothesis.get("id") or "")
        for hypothesis in hypotheses
        if hypothesis.get("hypothesis_type") == "case1_candidate"
        or str(hypothesis.get("target_id") or "").startswith("CASE1:CANDIDATE:")
    ]
    if invalid:
        sample = ", ".join(invalid[:3])
        raise ValueError(
            "CS1 hindcasting received synthetic disease x ROI x feature candidates "
            f"that cannot be matched to future literature claims: {sample}. "
            "Use the frozen-KG transdiagnostic_clustering task generator."
        )


def _is_claim_backed_edge(edge: dict[str, Any]) -> bool:
    metadata = edge.get("metadata") or {}
    return bool(
        metadata.get("claim_id")
        or edge.get("claim_id")
        or str(edge.get("source") or "").lower().startswith("claim:")
    )


def _historical_pairs(
    edges: list[dict[str, Any]],
    *,
    claims_path: Path | None = None,
    concepts: dict[str, dict[str, Any]] | None = None,
    stats: Counter[str] | None = None,
    endpoint_atoms_out: dict[str, set[str]] | None = None,
) -> set[tuple[str, str]]:
    """Return trusted curated edges plus semantically projected claim pairs."""

    audit = stats if stats is not None else Counter()
    out: set[tuple[str, str]] = set()
    if endpoint_atoms_out is not None and concepts is not None:
        for node_id, node in concepts.items():
            roles = {atom.value for atom in concept_atom_roles(node)}
            if roles:
                endpoint_atoms_out.setdefault(str(node_id), set()).update(roles)
    for edge in edges:
        src = str(edge.get("source_id") or "")
        dst = str(edge.get("target_id") or "")
        rel = str(edge.get("relation_type") or "")
        if not src or not dst or src == dst or rel in {"is_a", "part_of", "about"}:
            continue
        if _is_claim_backed_edge(edge):
            audit["claim_backed_canonical_edges_excluded"] += 1
            continue
        out.add(_edge_pair(src, dst))
        audit["curated_pairs"] += 1

    if claims_path is None:
        audit["historical_unique_pairs"] = len(out)
        return out
    if concepts is None:
        raise ValueError("concepts is required when claims_path is provided")

    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            claim = json.loads(line)
            if claim.get("negated"):
                audit["negated_claims_excluded"] += 1
                continue
            projected = semantic_claim_pair(claim, concepts)
            if projected is None:
                audit["claims_without_semantic_pair"] += 1
                continue
            subject, obj = projected
            if endpoint_atoms_out is not None:
                endpoint_atoms_out.setdefault(subject.entity_id, set()).update(
                    subject.atoms
                )
                endpoint_atoms_out.setdefault(obj.entity_id, set()).update(obj.atoms)
            if not subject.uses_canonical_id:
                audit["claim_local_subjects"] += 1
            if not obj.uses_canonical_id:
                audit["claim_local_objects"] += 1
            out.add(_edge_pair(subject.entity_id, obj.entity_id))
            audit["semantic_claim_pairs"] += 1
    audit["historical_unique_pairs"] = len(out)
    return out


def _canonical_endpoint_atoms(
    concepts: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for node_id, node in concepts.items():
        roles = frozenset(atom.value for atom in concept_atom_roles(node))
        if roles:
            result[str(node_id)] = roles
    return result


def _projected_pair_with_frozen_atoms(
    projected: tuple[SemanticEndpoint, SemanticEndpoint],
    historical_endpoint_atoms: Mapping[str, Iterable[str]],
) -> tuple[SemanticEndpoint, SemanticEndpoint] | None:
    frozen: list[SemanticEndpoint] = []
    for endpoint in projected:
        atoms = tuple(sorted({str(atom) for atom in historical_endpoint_atoms.get(
            endpoint.entity_id, ()
        ) if str(atom)}))
        if not atoms:
            return None
        frozen.append(
            SemanticEndpoint(
                entity_id=endpoint.entity_id,
                name=endpoint.name,
                canonical_id=endpoint.canonical_id,
                atoms=atoms,
                uses_canonical_id=endpoint.uses_canonical_id,
                name_score=endpoint.name_score,
                role_compatible=endpoint.role_compatible,
            )
        )
    return frozen[0], frozen[1]


def _future_indexes(
    claims_path: Path,
    concepts: dict[str, dict[str, Any]],
    historical_pairs: set[tuple[str, str]],
    start_year: int,
    end_year: int,
    *,
    case_study_id: str | None = None,
    future_records: list[dict[str, Any]] | None = None,
    historical_endpoint_atoms: Mapping[str, Iterable[str]] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    novel_pair_year: dict[tuple[str, str], int] = {}
    all_pair_year: dict[tuple[str, str], int] = {}
    pair_claims: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    frozen_atoms = (
        {str(node_id): frozenset(map(str, atoms)) for node_id, atoms in historical_endpoint_atoms.items()}
        if historical_endpoint_atoms is not None
        else _canonical_endpoint_atoms(concepts)
    )
    stats = Counter()
    if future_records is None:
        source: Any = claims_path.open("r", encoding="utf-8")
    else:
        source = future_records
    try:
        for item in source:
            claim = json.loads(item) if isinstance(item, str) else item
            if claim.get("negated"):
                stats["negated_claims_excluded"] += 1
                continue
            year = claim.get("year") if future_records is not None else _claim_year(claim)
            if year is None or year < start_year or year > end_year:
                continue
            if case_study_id is not None:
                case_study_ids = (
                    claim.get("case_study_ids")
                    if future_records is not None
                    else _claim_case_study_ids(claim)
                )
                if case_study_id not in (case_study_ids or ()):
                    continue
            stats["future_claims_total"] += 1
            projected = semantic_claim_pair(claim, concepts)
            if projected is None:
                stats["semantic_projection_failed"] += 1
                continue
            if not case_study_pair_allowed(
                claim,
                concepts,
                projected,
                case_study_id,
            ):
                stats["case_study_relation_contract_rejected"] += 1
                continue
            subject, obj = projected
            if not subject.uses_canonical_id:
                stats["claim_local_subjects"] += 1
            if not obj.uses_canonical_id:
                stats["claim_local_objects"] += 1

            # A deterministic claim-local identity is executable when the same
            # identity was already present in a frozen relation. This preserves
            # temporal isolation while avoiding the systematic loss of valid
            # endpoints whose broad canonical concept was intentionally rejected.
            subject_available = bool(frozen_atoms.get(subject.entity_id))
            object_available = bool(frozen_atoms.get(obj.entity_id))
            if not subject_available:
                stats["future_claims_unavailable_subject_endpoint"] += 1
            elif not subject.uses_canonical_id:
                stats["frozen_claim_local_subjects_recovered"] += 1
            if not object_available:
                stats["future_claims_unavailable_object_endpoint"] += 1
            elif not obj.uses_canonical_id:
                stats["frozen_claim_local_objects_recovered"] += 1
            if not subject_available or not object_available:
                stats["future_claims_unavailable_endpoint"] += 1
                continue
            frozen_projected = _projected_pair_with_frozen_atoms(
                projected,
                frozen_atoms,
            )
            if frozen_projected is None or not case_study_pair_allowed(
                claim,
                concepts,
                frozen_projected,
                case_study_id,
            ):
                stats["future_claims_frozen_atom_contract_rejected"] += 1
                continue
            sid = subject.entity_id
            oid = obj.entity_id
            pair = _edge_pair(sid, oid)
            all_pair_year[pair] = min(all_pair_year.get(pair, year), year)
            paper = claim if future_records is not None else _safe_paper(claim)
            pair_claims[pair].append(
                {
                    "claim_id": claim.get("id"),
                    "pmid": paper.get("pmid"),
                    "doi": paper.get("doi"),
                    "title": paper.get("title"),
                    "journal": paper.get("journal"),
                    "year": year,
                    "predicate": claim.get("predicate"),
                    "subject_id": sid,
                    "subject_canonical_id": subject.canonical_id,
                    "subject_name": subject.name,
                    "object_id": oid,
                    "object_canonical_id": obj.canonical_id,
                    "object_name": obj.name,
                    "raw_text": claim.get("raw_text"),
                }
            )
            if pair in historical_pairs:
                stats["already_direct_in_frozen_kg"] += 1
                continue
            novel_pair_year[pair] = min(novel_pair_year.get(pair, year), year)
            stats["future_evaluable_claims"] += 1
    finally:
        if future_records is None:
            source.close()
    stats["future_unique_pairs"] = len(novel_pair_year)
    stats["future_all_unique_pairs"] = len(all_pair_year)
    return {
        "novel_pair_year": novel_pair_year,
        "all_pair_year": all_pair_year,
        "pair_claims": pair_claims,
    }, dict(stats)


def _benchmark_status(n_hypotheses: int, future_unique_pairs: int) -> tuple[str, str]:
    if n_hypotheses <= 0:
        return "non_executable", "no hypotheses were generated"
    if future_unique_pairs <= 0:
        return (
            "non_executable",
            "no future relation passed scope, semantic, relation, and frozen-endpoint contracts",
        )
    if future_unique_pairs < MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK:
        return (
            "sparse",
            f"only {future_unique_pairs} unique future relations passed all contracts",
        )
    return "executable", ""


def _score_hypothesis(
    hyp: dict[str, Any],
    future: dict[str, Any],
    freeze_year: int,
    *,
    case_study_id: str | None = None,
) -> dict[str, Any]:
    novel_pair_year: dict[tuple[str, str], int] = future["novel_pair_year"]
    all_pair_year: dict[tuple[str, str], int] = future["all_pair_year"]
    endpoint = _hyp_endpoint_pair(hyp)
    endpoint_pair = _edge_pair(*endpoint) if endpoint else None
    endpoint_year = novel_pair_year.get(endpoint_pair) if endpoint_pair else None
    path_edges = [_edge_pair(src, dst) for src, dst in _hyp_path_edges(hyp)]
    hit_edges = [pair for pair in path_edges if pair in all_pair_year]
    hit_years = [all_pair_year[pair] for pair in hit_edges]
    any_years = ([endpoint_year] if endpoint_year is not None else []) + hit_years
    support_candidates: list[tuple[str, tuple[str, str], int]] = []
    if endpoint_pair is not None and endpoint_year is not None:
        support_candidates.append(("endpoint", endpoint_pair, endpoint_year))
    support_candidates.extend(("path_edge", pair, all_pair_year[pair]) for pair in hit_edges)
    support_kind = None
    support_pair = None
    if support_candidates:
        support_kind, support_pair, _ = min(support_candidates, key=lambda item: item[2])

    all_path_edges_hit = bool(path_edges) and len(hit_edges) == len(path_edges)
    primary_kind = None
    primary_pair = None
    primary_year = None
    primary_discovery_key = ""
    primary_recovered_pairs: list[tuple[str, str]] = []
    if requires_complete_path(case_study_id):
        # Mechanism-shaped tasks require every proposed hop to receive future support.
        if len(path_edges) >= 2 and all_path_edges_hit:
            primary_pair = max(path_edges, key=lambda pair: all_pair_year[pair])
            primary_year = all_pair_year[primary_pair]
            primary_kind = "complete_path"
            primary_discovery_key = "complete_path:" + ";".join(
                "|".join(pair) for pair in sorted(set(path_edges))
            )
            primary_recovered_pairs = [
                pair for pair in sorted(set(path_edges)) if pair in novel_pair_year
            ]
    elif endpoint_pair is not None and endpoint_year is not None:
        primary_pair = endpoint_pair
        primary_year = endpoint_year
        primary_kind = "endpoint"
        primary_discovery_key = "endpoint:" + "|".join(endpoint_pair)
        primary_recovered_pairs = [endpoint_pair]
    elif all_path_edges_hit:
        # For other path-shaped hypotheses, an entirely future-supported path is
        # accepted, but a partial path remains a secondary diagnostic only.
        primary_pair = max(path_edges, key=lambda pair: all_pair_year[pair])
        primary_year = all_pair_year[primary_pair]
        primary_kind = "complete_path"
        primary_discovery_key = "complete_path:" + ";".join(
            "|".join(pair) for pair in sorted(set(path_edges))
        )
        primary_recovered_pairs = [
            pair for pair in sorted(set(path_edges)) if pair in novel_pair_year
        ]
    return {
        "endpoint_hit": endpoint_year is not None,
        "endpoint_year": endpoint_year,
        "endpoint_lead_time": (endpoint_year - freeze_year) if endpoint_year is not None else None,
        "endpoint_pair": "|".join(endpoint_pair) if endpoint_pair else "",
        "endpoint_discovery_key": (
            "endpoint:" + "|".join(endpoint_pair)
            if endpoint_pair is not None and endpoint_year is not None
            else ""
        ),
        "path_edges": len(path_edges),
        "path_edge_hits": len(hit_edges),
        "path_edge_hit_pairs": ";".join("|".join(pair) for pair in hit_edges),
        "path_edge_hit_rate": len(hit_edges) / len(path_edges) if path_edges else 0.0,
        "any_path_edge_hit": bool(hit_edges),
        "all_path_edges_hit": all_path_edges_hit,
        "any_future_hit": bool(any_years),
        "first_future_year": min(any_years) if any_years else None,
        "lead_time": (min(any_years) - freeze_year) if any_years else None,
        "support_kind": support_kind,
        "support_pair": "|".join(support_pair) if support_pair else "",
        "primary_hit": primary_year is not None,
        "primary_year": primary_year,
        "primary_lead_time": (
            primary_year - freeze_year if primary_year is not None else None
        ),
        "primary_support_kind": primary_kind,
        "primary_support_pair": "|".join(primary_pair) if primary_pair else "",
        "primary_discovery_key": primary_discovery_key,
        "primary_recovered_pairs": ";".join(
            "|".join(pair) for pair in primary_recovered_pairs
        ),
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    future_pair_total: int | None = None,
) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "primary_hits": 0,
            "primary_hit_rate": 0.0,
            "unique_primary_discoveries": 0,
            "unique_primary_discovery_rate": 0.0,
            "recovered_future_pairs": 0,
            "future_pair_recall": (
                0.0 if future_pair_total is not None and future_pair_total > 0 else None
            ),
            "endpoint_hits": 0,
            "endpoint_hit_rate": 0.0,
            "unique_endpoint_discoveries": 0,
            "any_path_edge_hits": 0,
            "any_path_edge_hit_rate": 0.0,
            "any_future_hits": 0,
            "any_future_hit_rate": 0.0,
            "mean_path_edge_hit_rate": 0.0,
        }
    lead_times = [
        r["primary_lead_time"]
        for r in rows
        if r["primary_lead_time"] is not None
    ]
    primary_keys = {
        str(row.get("primary_discovery_key") or "")
        for row in rows
        if row.get("primary_hit") and row.get("primary_discovery_key")
    }
    endpoint_keys = {
        str(row.get("endpoint_discovery_key") or "")
        for row in rows
        if row.get("endpoint_hit") and row.get("endpoint_discovery_key")
    }
    unique_primary = len(primary_keys)
    recovered_pairs = {
        pair
        for row in rows
        if row.get("primary_hit")
        for pair in str(row.get("primary_recovered_pairs") or "").split(";")
        if pair
    }
    return {
        "n": n,
        "primary_hits": sum(1 for r in rows if r["primary_hit"]),
        "primary_hit_rate": sum(1 for r in rows if r["primary_hit"]) / n,
        "unique_primary_discoveries": unique_primary,
        "unique_primary_discovery_rate": unique_primary / n,
        "recovered_future_pairs": len(recovered_pairs),
        "future_pair_recall": (
            len(recovered_pairs) / future_pair_total
            if future_pair_total is not None and future_pair_total > 0
            else None
        ),
        "endpoint_hits": sum(1 for r in rows if r["endpoint_hit"]),
        "endpoint_hit_rate": sum(1 for r in rows if r["endpoint_hit"]) / n,
        "unique_endpoint_discoveries": len(endpoint_keys),
        "any_path_edge_hits": sum(1 for r in rows if r["any_path_edge_hit"]),
        "any_path_edge_hit_rate": sum(1 for r in rows if r["any_path_edge_hit"]) / n,
        "all_path_edges_hits": sum(1 for r in rows if r["all_path_edges_hit"]),
        "any_future_hits": sum(1 for r in rows if r["any_future_hit"]),
        "any_future_hit_rate": sum(1 for r in rows if r["any_future_hit"]) / n,
        "mean_path_edge_hit_rate": mean(r["path_edge_hit_rate"] for r in rows),
        "mean_primary_lead_time": mean(lead_times) if lead_times else None,
    }


def _random_baseline(
    scored: list[dict[str, Any]],
    k: int,
    trials: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if k >= len(scored):
        return {
            "applicable": False,
            "reason": "random selection is identical to the full candidate pool at this K",
            "candidate_pool_size": len(scored),
        }, []
    n = len(scored)
    any_good = sum(1 for row in scored if row["any_future_hit"])
    np_rng = np.random.default_rng(rng.randrange(0, 2**63))

    def grouped_draws(
        *,
        hit_field: str,
        key_field: str,
    ) -> tuple[list[int], list[int]]:
        counts = Counter(
            str(row.get(key_field) or f"__row__:{index}")
            for index, row in enumerate(scored)
            if row.get(hit_field)
        )
        group_sizes = list(counts.values())
        no_hit = n - sum(group_sizes)
        colors = np.asarray([*group_sizes, no_hit], dtype=np.int64)
        draws = np_rng.multivariate_hypergeometric(colors, k, size=trials)
        if not group_sizes:
            return [0] * trials, [0] * trials
        hit_draws = draws[:, : len(group_sizes)]
        slot_hits = hit_draws.sum(axis=1).astype(int).tolist()
        unique_hits = (hit_draws > 0).sum(axis=1).astype(int).tolist()
        return slot_hits, unique_hits

    primary_hits, unique_primary = grouped_draws(
        hit_field="primary_hit",
        key_field="primary_discovery_key",
    )
    hits = np_rng.hypergeometric(any_good, n - any_good, k, size=trials).astype(int).tolist()
    endpoint_hits, unique_endpoint = grouped_draws(
        hit_field="endpoint_hit",
        key_field="endpoint_discovery_key",
    )
    trial_rows = [
        {
            "trial": trial + 1,
            "k": k,
            "primary_hits": primary_hits[trial],
            "unique_primary_discoveries": unique_primary[trial],
            "any_future_hits": hits[trial],
            "endpoint_hits": endpoint_hits[trial],
            "unique_endpoint_discoveries": unique_endpoint[trial],
        }
        for trial in range(trials)
    ]
    return {
        "applicable": True,
        "trials": trials,
        "mean_primary_hits": mean(primary_hits),
        "mean_unique_primary_discoveries": mean(unique_primary),
        "mean_any_future_hits": mean(hits),
        "mean_endpoint_hits": mean(endpoint_hits),
        "mean_unique_endpoint_discoveries": mean(unique_endpoint),
        "sd_primary_hits": stdev(primary_hits) if len(primary_hits) > 1 else 0.0,
        "variance_unique_primary_discoveries": (
            float(np.var(unique_primary, ddof=1)) if len(unique_primary) > 1 else 0.0
        ),
        "sd_any_future_hits": stdev(hits) if len(hits) > 1 else 0.0,
        "sd_endpoint_hits": stdev(endpoint_hits) if len(endpoint_hits) > 1 else 0.0,
        "ci95_primary_hits": _percentile_interval(primary_hits),
        "ci95_any_future_hits": _percentile_interval(hits),
        "ci95_endpoint_hits": _percentile_interval(endpoint_hits),
    }, trial_rows


def _percentile_interval(values: list[int]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    xs = sorted(values)
    return [_percentile(xs, 0.025), _percentile(xs, 0.975)]


def _percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _empirical_p_ge(values: list[int], observed: int) -> float | None:
    if not values:
        return None
    return (1 + sum(1 for value in values if value >= observed)) / (len(values) + 1)


def _support_claim(future: dict[str, Any], support_pair: str, first_year: int | None) -> dict[str, Any]:
    if not support_pair or first_year is None:
        return {}
    parts = support_pair.split("|")
    if len(parts) != 2:
        return {}
    pair = tuple(parts)
    claims = future["pair_claims"].get(pair) or []
    if not claims:
        return {}
    return sorted(claims, key=lambda claim: (claim.get("year") or 9999, str(claim.get("claim_id") or "")))[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    *,
    kg_path: Path,
    hypotheses_path: Path,
    future_claims_path: Path,
    output_dir: Path,
    freeze_year: int,
    future_start_year: int,
    future_end_year: int,
    top_ks: list[int],
    random_trials: int,
    seed: int,
    case_study_id: str | None = None,
    method: str = "neurodiscovery",
    kg_index: tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]] | None = None,
    historical_pairs: set[tuple[str, str]] | None = None,
    historical_claims_path: Path | None = None,
    historical_stats: dict[str, int] | None = None,
    historical_endpoint_atoms: Mapping[str, Iterable[str]] | None = None,
    future_records: list[dict[str, Any]] | None = None,
    future_index: dict[str, Any] | None = None,
    future_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    concepts, edges, names = kg_index if kg_index is not None else load_kg_index(kg_path)
    del names
    payload = _load_json(hypotheses_path)
    hypotheses = list(payload.get("hypotheses") or [])
    _validate_hindcasting_hypotheses(hypotheses, case_study_id)
    if historical_pairs is None:
        if historical_claims_path is None:
            candidate = kg_path.parent / "extracted_claims.jsonl"
            historical_claims_path = candidate if candidate.is_file() else None
        historical_counter: Counter[str] = Counter()
        computed_historical_endpoint_atoms: dict[str, set[str]] = {}
        historical_pairs = _historical_pairs(
            edges,
            claims_path=historical_claims_path,
            concepts=concepts if historical_claims_path is not None else None,
            stats=historical_counter,
            endpoint_atoms_out=computed_historical_endpoint_atoms,
        )
        historical_stats = dict(historical_counter)
        historical_endpoint_atoms = computed_historical_endpoint_atoms
    elif historical_stats is None:
        historical_stats = {"historical_unique_pairs": len(historical_pairs)}
    if historical_endpoint_atoms is None:
        historical_endpoint_atoms = _canonical_endpoint_atoms(concepts)
    if future_index is None:
        future, computed_future_stats = _future_indexes(
            future_claims_path,
            concepts,
            historical_pairs,
            future_start_year,
            future_end_year,
            case_study_id=case_study_id,
            future_records=future_records,
            historical_endpoint_atoms=historical_endpoint_atoms,
        )
        future_stats = computed_future_stats
    else:
        if future_stats is None:
            raise ValueError("future_stats is required with a precomputed future_index")
        future = future_index
    scored: list[dict[str, Any]] = []
    for idx, hyp in enumerate(hypotheses):
        hypothesis_case_study_id = (
            (hyp.get("metadata") or {}).get("case_study_id")
            or case_study_id
            or "unknown"
        )
        score = _score_hypothesis(
            hyp,
            future,
            freeze_year,
            case_study_id=hypothesis_case_study_id,
        )
        row = {
            "rank": idx + 1,
            "method": method,
            "id": hyp.get("id"),
            "hypothesis_type": hyp.get("hypothesis_type") or "unknown",
            "case_study_id": hypothesis_case_study_id,
            "task_name": (hyp.get("metadata") or {}).get("task_name") or (hyp.get("metadata") or {}).get("chain_name") or "unknown",
            "task_kind": (hyp.get("metadata") or {}).get("task_kind") or "unknown",
            "signature": _hyp_signature(concepts, hyp),
            "source_id": hyp.get("source_id"),
            "source_name": hyp.get("source_name"),
            "target_id": hyp.get("target_id"),
            "target_name": hyp.get("target_name"),
            "composite_score": hyp.get("composite_score"),
            **score,
        }
        scored.append(row)

    rng = random.Random(seed + freeze_year)
    topk: dict[str, Any] = {}
    random_trial_rows: list[dict[str, Any]] = []
    for k in top_ks:
        subset = scored[: min(k, len(scored))]
        random_summary, trial_rows = _random_baseline(scored, len(subset), random_trials, rng)
        random_primary_values = [int(row["primary_hits"]) for row in trial_rows]
        random_unique_primary_values = [
            int(row["unique_primary_discoveries"]) for row in trial_rows
        ]
        random_values = [int(row["any_future_hits"]) for row in trial_rows]
        random_endpoint_values = [int(row["endpoint_hits"]) for row in trial_rows]
        observed = _aggregate(
            subset,
            future_pair_total=int((future_stats or {}).get("future_unique_pairs") or 0),
        )
        random_summary["p_primary_hits_ge_observed"] = _empirical_p_ge(
            random_primary_values,
            int(observed["primary_hits"]),
        )
        random_summary["p_unique_primary_discoveries_ge_observed"] = _empirical_p_ge(
            random_unique_primary_values,
            int(observed["unique_primary_discoveries"]),
        )
        random_summary["p_any_future_hits_ge_observed"] = _empirical_p_ge(
            random_values,
            int(observed["any_future_hits"]),
        )
        random_summary["p_endpoint_hits_ge_observed"] = _empirical_p_ge(
            random_endpoint_values,
            int(observed["endpoint_hits"]),
        )
        random_trial_rows.extend(trial_rows)
        topk[str(k)] = {
            "requested_experiment_slots": int(k),
            "executed_ranked_prefix_slots": len(subset),
            "observed": observed,
            "random_same_hypothesis_pool": random_summary,
        }

    by_type = {
        key: _aggregate(rows)
        for key, rows in _group(scored, "hypothesis_type").items()
    }
    by_task = {
        key: _aggregate(rows)
        for key, rows in _group(scored, "task_name").items()
    }
    by_signature = {
        key: _aggregate(rows)
        for key, rows in sorted(
            _group(scored, "signature").items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:50]
    }

    recovered_examples: list[dict[str, Any]] = []
    for row in scored:
        if not row["primary_hit"]:
            continue
        claim = _support_claim(
            future,
            str(row.get("primary_support_pair") or ""),
            row.get("primary_year"),
        )
        recovered_examples.append({
            **row,
            "support_claim_id": claim.get("claim_id"),
            "support_pmid": claim.get("pmid"),
            "support_doi": claim.get("doi"),
            "support_title": claim.get("title"),
            "support_journal": claim.get("journal"),
            "support_year": claim.get("year"),
            "support_predicate": claim.get("predicate"),
            "support_subject_name": claim.get("subject_name"),
            "support_object_name": claim.get("object_name"),
            "support_raw_text": claim.get("raw_text"),
        })

    benchmark_status, benchmark_reason = _benchmark_status(
        len(scored),
        int((future_stats or {}).get("future_unique_pairs") or 0),
    )
    manifest = {
        "method": method,
        "case_study_id": case_study_id,
        "validation_protocol": "hindcasting",
        "benchmark_status": benchmark_status,
        "benchmark_reason": benchmark_reason,
        "semantic_projection": SEMANTIC_PROJECTION_VERSION,
        "case_study_relation_contract": CASE_STUDY_RELATION_CONTRACT_VERSION,
        "temporal_endpoint_contract": TEMPORAL_ENDPOINT_CONTRACT_VERSION,
        "discovery_metric_contract": DISCOVERY_METRIC_CONTRACT_VERSION,
        "kg_path": str(kg_path),
        "historical_claims_path": (
            str(historical_claims_path) if historical_claims_path is not None else None
        ),
        "hypotheses_path": str(hypotheses_path),
        "future_claims_path": str(future_claims_path),
        "output_dir": str(output_dir),
        "freeze_year": freeze_year,
        "future_start_year": future_start_year,
        "future_end_year": future_end_year,
        "n_hypotheses": len(scored),
        "ranking_pool_size": len(scored),
        "experiment_budget_semantics": (
            "Each registered experiment budget K executes only the first min(K, "
            "ranking_pool_size) ranked slots; any retained candidate tail is not "
            "counted at that K. Generation-failure padding occupies an experiment "
            "slot and receives zero credit."
        ),
        "historical_stats": historical_stats,
        "future_stats": future_stats,
        "topk": topk,
        "by_type": by_type,
        "by_task": by_task,
        "by_signature_top50": by_signature,
        "validated_sample": recovered_examples[:100],
        "note": (
            "Case-study hindcasting filters future evidence by claim_case_study_ids. "
            "Claim endpoints use a conservative semantic projection shared by frozen "
            "generation and future evaluation, followed by the registered case-study "
            "relation contract. Future support must map to endpoint identities available "
            "in the frozen graph; inductive claim-local entities are excluded. "
            "Primary support requires the exact endpoint or a complete path; registered "
            "mechanism-shaped tasks require their complete multi-step chain. A single "
            "path-edge hit is retained only as a secondary diagnostic. Primary hits count "
            "hypothesis slots for backward compatibility; unique_primary_discoveries and "
            "future_pair_recall are deduplicated and are the primary comparison metrics."
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _write_window_tables(output_dir, scored, random_trial_rows, recovered_examples, topk)
    write_report(output_dir, manifest)
    return manifest


def _write_window_tables(
    output_dir: Path,
    scored: list[dict[str, Any]],
    random_trial_rows: list[dict[str, Any]],
    recovered_examples: list[dict[str, Any]],
    topk: dict[str, Any],
) -> None:
    scored_fields = [
        "rank", "method", "id", "hypothesis_type", "case_study_id", "task_name", "task_kind", "signature",
        "source_id", "source_name", "target_id", "target_name", "composite_score",
        "endpoint_hit", "endpoint_year", "endpoint_lead_time", "endpoint_pair",
        "endpoint_discovery_key",
        "path_edges", "path_edge_hits", "path_edge_hit_pairs", "path_edge_hit_rate",
        "any_path_edge_hit", "all_path_edges_hit", "any_future_hit",
        "first_future_year", "lead_time", "support_kind", "support_pair",
        "primary_hit", "primary_year", "primary_lead_time",
        "primary_support_kind", "primary_support_pair", "primary_discovery_key",
        "primary_recovered_pairs",
    ]
    _write_csv(output_dir / "scored_hypotheses.csv", scored, scored_fields)
    _write_csv(
        output_dir / "random_trials.csv",
        random_trial_rows,
        [
            "trial", "k", "primary_hits", "unique_primary_discoveries",
            "any_future_hits", "endpoint_hits", "unique_endpoint_discoveries",
        ],
    )
    example_fields = scored_fields + [
        "support_claim_id", "support_pmid", "support_doi", "support_title",
        "support_journal", "support_year", "support_predicate",
        "support_subject_name", "support_object_name", "support_raw_text",
    ]
    _write_csv(output_dir / "recovered_examples.csv", recovered_examples[:200], example_fields)
    rows: list[dict[str, Any]] = []
    for k, row in topk.items():
        obs = row["observed"]
        rand = row["random_same_hypothesis_pool"]
        rows.append({
            "k": k,
            "n": obs["n"],
            "primary_hits": obs["primary_hits"],
            "primary_hit_rate": obs["primary_hit_rate"],
            "unique_primary_discoveries": obs["unique_primary_discoveries"],
            "unique_primary_discovery_rate": obs["unique_primary_discovery_rate"],
            "recovered_future_pairs": obs["recovered_future_pairs"],
            "future_pair_recall": obs["future_pair_recall"],
            "endpoint_hits": obs["endpoint_hits"],
            "any_future_hits": obs["any_future_hits"],
            "any_future_hit_rate": obs["any_future_hit_rate"],
            "mean_primary_lead_time": obs.get("mean_primary_lead_time"),
            "random_primary_hits_mean": rand.get("mean_primary_hits"),
            "random_primary_hits_sd": rand.get("sd_primary_hits"),
            "p_primary_hits_ge_observed": rand.get("p_primary_hits_ge_observed"),
            "random_unique_primary_discoveries_mean": rand.get(
                "mean_unique_primary_discoveries"
            ),
            "random_unique_primary_discoveries_variance": rand.get(
                "variance_unique_primary_discoveries"
            ),
            "p_unique_primary_discoveries_ge_observed": rand.get(
                "p_unique_primary_discoveries_ge_observed"
            ),
            "random_any_future_hits_mean": rand.get("mean_any_future_hits"),
            "random_any_future_hits_sd": rand.get("sd_any_future_hits"),
            "random_any_future_hits_ci95_low": (rand.get("ci95_any_future_hits") or [None, None])[0],
            "random_any_future_hits_ci95_high": (rand.get("ci95_any_future_hits") or [None, None])[1],
            "p_any_future_hits_ge_observed": rand.get("p_any_future_hits_ge_observed"),
            "random_endpoint_hits_mean": rand.get("mean_endpoint_hits"),
            "p_endpoint_hits_ge_observed": rand.get("p_endpoint_hits_ge_observed"),
        })
    _write_csv(
        output_dir / "metrics_by_k.csv",
        rows,
        [
            "k", "n", "primary_hits", "primary_hit_rate",
            "unique_primary_discoveries", "unique_primary_discovery_rate",
            "recovered_future_pairs", "future_pair_recall", "endpoint_hits",
            "any_future_hits", "any_future_hit_rate", "mean_primary_lead_time",
            "random_primary_hits_mean", "random_primary_hits_sd",
            "p_primary_hits_ge_observed", "random_unique_primary_discoveries_mean",
            "random_unique_primary_discoveries_variance",
            "p_unique_primary_discoveries_ge_observed", "random_any_future_hits_mean",
            "random_any_future_hits_sd", "random_any_future_hits_ci95_low",
            "random_any_future_hits_ci95_high", "p_any_future_hits_ge_observed",
            "random_endpoint_hits_mean", "p_endpoint_hits_ge_observed",
        ],
    )


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key) or "unknown")].append(row)
    return dict(out)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# Hindcasting: {manifest.get('case_study_id') or 'unscoped'}",
        "",
        "This evaluates arbitrary generated hypotheses against future claims without constraining the hypothesis space to gene-imaging-disease.",
        "",
        "## Setup",
        f"- Freeze year: {manifest['freeze_year']}",
        f"- Future window: {manifest['future_start_year']}-{manifest['future_end_year']}",
        f"- Hypotheses: {manifest['n_hypotheses']}",
        f"- Future unique evaluable pairs: {manifest['future_stats'].get('future_unique_pairs', 0)}",
        f"- Benchmark status: {manifest['benchmark_status']}",
        f"- Benchmark note: {manifest.get('benchmark_reason') or 'none'}",
        "",
        "## Top-K",
        "| K | Primary hits | Primary hit rate | Random primary hits | Endpoint hits | Any-edge diagnostic hits | Path-edge hit rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, row in manifest["topk"].items():
        obs = row["observed"]
        rand = row["random_same_hypothesis_pool"]
        lines.append(
            "| {k} | {primary_hits} | {primary_rate} | {rand_primary} | {endpoint_hits} | {any_hits} | {path_rate} |".format(
                k=k,
                primary_hits=obs["primary_hits"],
                primary_rate=_fmt(obs["primary_hit_rate"]),
                rand_primary=_fmt(rand.get("mean_primary_hits")),
                endpoint_hits=obs["endpoint_hits"],
                any_hits=obs["any_future_hits"],
                path_rate=_fmt(obs["mean_path_edge_hit_rate"]),
            )
        )
    lines.extend([
        "",
        "## By Hypothesis Type",
        "| Type | N | Primary hit rate | Endpoint hit rate | Any-edge diagnostic rate |",
        "|:---|---:|---:|---:|---:|",
    ])
    for key, row in sorted(manifest["by_type"].items(), key=lambda item: item[1]["n"], reverse=True):
        lines.append(
            f"| {key} | {row['n']} | {_fmt(row['primary_hit_rate'])} | {_fmt(row['endpoint_hit_rate'])} | {_fmt(row['any_future_hit_rate'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "- Future evidence is filtered by claim-level Case Study membership.",
        "- Primary hits require an exact endpoint or a fully supported path; registered mechanism-shaped tasks require their complete chain.",
        "- Any-edge hits are secondary diagnostics and never count as the primary result.",
        "- Random baseline samples from the same hypothesis file, so it tests ranking quality rather than generator-vs-random-space quality.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hindcasting for one formal case study.")
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--future-claims", type=Path, default=Path("neurooracle/data/full_v2/extracted_claims.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-year", type=int, required=True)
    parser.add_argument("--future-start-year", type=int, required=True)
    parser.add_argument("--future-end-year", type=int, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--random-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--case-study-id", required=True, help="Formal Case Study ID used to filter future papers.")
    parser.add_argument("--method", default="neurodiscovery")
    args = parser.parse_args()
    case_study_by_name(args.case_study_id)
    manifest = evaluate(
        kg_path=args.kg,
        hypotheses_path=args.hypotheses,
        future_claims_path=args.future_claims,
        output_dir=args.output_dir,
        freeze_year=args.freeze_year,
        future_start_year=args.future_start_year,
        future_end_year=args.future_end_year,
        top_ks=args.top_k,
        random_trials=args.random_trials,
        seed=args.seed,
        case_study_id=args.case_study_id,
        method=args.method,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n_hypotheses": manifest["n_hypotheses"]}, indent=2))


if __name__ == "__main__":
    main()


# Updated: 2026-08-11 22:36 HKT
# Updated: 2026-08-13 02:56 HKT - admit only claim-local endpoints already reachable at freeze time.
# Updated: 2026-08-13 06:18:55 HKT - require future truth to satisfy the task atom contract already visible at freeze time.
