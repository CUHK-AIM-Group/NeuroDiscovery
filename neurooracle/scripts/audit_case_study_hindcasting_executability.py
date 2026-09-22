"""Audit structural executability for every formal hindcasting benchmark.

This is a label-free preflight.  It checks whether a frozen Case Study graph can
form the registered atom contract and whether the future window contains enough
scope-, relation-, and endpoint-valid evidence to score it.  It does not rank
hypotheses or inspect validation outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from typing import Any, Iterable

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    validate_canonical_kg_release,
)
from neurooracle.scripts.case_study_hindcasting_eval import (
    CASE_STUDY_RELATION_CONTRACT_VERSION,
    DISCOVERY_METRIC_CONTRACT_VERSION,
    MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK,
    _future_indexes,
    load_future_claim_records,
)
from neurooracle.scripts.generate_case_study_frozen_baselines import (
    FrozenEdge,
    FrozenGraphIndex,
    load_semantic_claim_adjacencies,
)
from neurooracle.scripts.run_case_study_hindcasting import (
    DEFAULT_WINDOWS,
    Window,
    parse_window,
)
from neurooracle.src.atoms import Atom
from neurooracle.src.case_studies import (
    CASE_STUDIES,
    CaseStudy,
    case_study_by_name,
)
from neurooracle.src.case_study_relation_contracts import (
    case_study_pair_allowed,
    endpoint_matches_atom,
    requires_complete_path,
)
from neurooracle.src.claim_semantics import (
    SEMANTIC_ENDPOINT_IDENTITY_VERSION,
    SemanticEndpoint,
    semantic_claim_pair,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "neurooracle" / "data" / "full_v2"
DEFAULT_SNAPSHOT_ROOT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "snapshots_full_v2_endpoint_v3"
)
DEFAULT_OUTPUT = (
    REPO
    / "neurooracle"
    / "data"
    / "experiments"
    / "hindcasting"
    / "executability_audit_v2"
)


def _required_atoms(case: CaseStudy) -> tuple[Atom, ...]:
    if case.chain is not None:
        values = (case.chain.source, *case.chain.mediators, case.chain.target)
    elif case.task is not None:
        values = (*sorted(case.task.inputs, key=lambda atom: atom.value), case.task.output)
    else:
        return ()
    return tuple(dict.fromkeys(values))


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
    if edge.claim_id:
        return ("claim", edge.claim_id)
    paper = edge.source_paper or {}
    paper_key = str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "")
    return (
        "edge",
        *sorted((edge.source_id, edge.target_id)),
        edge.relation,
        paper_key,
    )


def _unique_edges(adjacency: dict[str, list[FrozenEdge]]) -> list[FrozenEdge]:
    unique: dict[tuple[str, ...], FrozenEdge] = {}
    for edges in adjacency.values():
        for edge in edges:
            unique.setdefault(_edge_key(edge), edge)
    return list(unique.values())


def _paper_key(edge: FrozenEdge) -> str:
    paper = edge.source_paper or {}
    return str(
        paper.get("pmid")
        or paper.get("doi")
        or paper.get("title")
        or edge.claim_id
        or ""
    )


def _maximum_distinct_atom_coverage(
    index: FrozenGraphIndex,
    node_ids: Iterable[str],
    required_atoms: tuple[Atom, ...],
) -> int:
    nodes = tuple(sorted(set(node_ids)))
    candidates = {
        atom: tuple(
            node_id
            for node_id in nodes
            if endpoint_matches_atom(_endpoint(index, node_id), atom, index.concepts)
        )
        for atom in required_atoms
    }
    ordered = tuple(sorted(required_atoms, key=lambda atom: (len(candidates[atom]), atom.value)))

    def visit(position: int, used: frozenset[str]) -> int:
        if position >= len(ordered):
            return 0
        atom = ordered[position]
        best = visit(position + 1, used)
        for node_id in candidates[atom]:
            if node_id in used:
                continue
            best = max(best, 1 + visit(position + 1, used | {node_id}))
        return best

    return visit(0, frozenset())


def component_summary(
    index: FrozenGraphIndex,
    edges: list[FrozenEdge],
    required_atoms: tuple[Atom, ...],
) -> dict[str, int]:
    neighbours: dict[str, set[str]] = defaultdict(set)
    incident: dict[str, list[FrozenEdge]] = defaultdict(list)
    for edge in edges:
        if not edge.source_id or not edge.target_id or edge.source_id == edge.target_id:
            continue
        neighbours[edge.source_id].add(edge.target_id)
        neighbours[edge.target_id].add(edge.source_id)
        incident[edge.source_id].append(edge)
        incident[edge.target_id].append(edge)

    seen: set[str] = set()
    components = 0
    complete_any = 0
    complete_cross_paper = 0
    max_covered_roles = 0
    max_component_papers = 0
    for start in sorted(neighbours):
        if start in seen:
            continue
        components += 1
        stack = [start]
        nodes: set[str] = set()
        component_edges: dict[tuple[str, ...], FrozenEdge] = {}
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            nodes.add(node_id)
            for edge in incident.get(node_id, ()):
                component_edges.setdefault(_edge_key(edge), edge)
            stack.extend(neighbours.get(node_id, ()) - seen)
        covered = _maximum_distinct_atom_coverage(index, nodes, required_atoms)
        papers = {_paper_key(edge) for edge in component_edges.values() if _paper_key(edge)}
        max_covered_roles = max(max_covered_roles, covered)
        max_component_papers = max(max_component_papers, len(papers))
        if covered == len(required_atoms):
            complete_any += 1
            if len(papers) >= 2:
                complete_cross_paper += 1

    return {
        "components": components,
        "complete_components": complete_any,
        "complete_cross_paper_components": complete_cross_paper,
        "max_covered_roles": max_covered_roles,
        "max_component_papers": max_component_papers,
    }


def _primary_edge_count(
    case: CaseStudy,
    index: FrozenGraphIndex,
    edges: list[FrozenEdge],
) -> int:
    count = 0
    for edge in edges:
        projected = (_endpoint(index, edge.source_id), _endpoint(index, edge.target_id))
        if case_study_pair_allowed(
            {"predicate": edge.relation},
            index.concepts,
            projected,
            case.name,
        ):
            count += 1
    return count


def _future_contract_edges(
    records: list[dict[str, Any]],
    case: CaseStudy,
    index: FrozenGraphIndex,
    window: Window,
    historical_endpoint_atoms: dict[str, frozenset[str]] | None = None,
) -> list[FrozenEdge]:
    edges: list[FrozenEdge] = []
    for claim in records:
        year = claim.get("year")
        if year is None or year < window.future_start_year or year > window.future_end_year:
            continue
        projected = semantic_claim_pair(claim, index.concepts)
        if projected is None:
            continue
        subject, obj = projected
        frozen_atoms = historical_endpoint_atoms or index.atoms
        if not frozen_atoms.get(subject.entity_id) or not frozen_atoms.get(obj.entity_id):
            continue
        projected = (
            SemanticEndpoint(
                **{
                    **subject.__dict__,
                    "atoms": tuple(sorted(frozen_atoms[subject.entity_id])),
                }
            ),
            SemanticEndpoint(
                **{
                    **obj.__dict__,
                    "atoms": tuple(sorted(frozen_atoms[obj.entity_id])),
                }
            ),
        )
        if not case_study_pair_allowed(
            claim,
            index.concepts,
            projected,
            case.name,
        ):
            continue
        paper = {
            "pmid": claim.get("pmid"),
            "doi": claim.get("doi"),
            "title": claim.get("title"),
            "year": year,
        }
        edges.append(
            FrozenEdge(
                source_id=subject.entity_id,
                target_id=obj.entity_id,
                relation=str(claim.get("predicate") or "related_to"),
                confidence=0.5,
                claim_id=str(claim.get("id") or ""),
                raw_text=str(claim.get("raw_text") or ""),
                source_paper=paper,
                year=int(year),
            )
        )
    return edges


def _status(
    *,
    historical_ready: bool,
    future_unique_pairs: int,
    complete_path: bool,
    future_complete_components: int,
) -> tuple[str, str]:
    if not historical_ready:
        return "non_executable", "no frozen evidence graph satisfies the task atom contract"
    if future_unique_pairs <= 0:
        return "non_executable", "no future relation passes scope, relation, and endpoint contracts"
    if complete_path and future_complete_components <= 0:
        return "non_executable", "future evidence cannot cover the complete multi-input path"
    if future_unique_pairs < MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK:
        return "sparse", f"only {future_unique_pairs} unique future relations are evaluable"
    return "executable", ""


def audit(
    *,
    input_dir: Path,
    snapshot_root: Path,
    output_dir: Path,
    windows: tuple[Window, ...],
    cases: tuple[CaseStudy, ...],
) -> dict[str, Any]:
    canonical_release = validate_canonical_kg_release(
        kg_path=input_dir / "knowledge_graph.json",
        claims_path=input_dir / "extracted_claims.jsonl",
        state_path=input_dir / "CURRENT_STATE.json",
        expected_sha256=CURRENT_CANONICAL_SHA256,
    )
    future_records = load_future_claim_records(
        input_dir / "extracted_claims.jsonl",
        min_year=min(window.future_start_year for window in windows),
        max_year=max(window.future_end_year for window in windows),
        case_study_ids={case.name for case in cases},
    )
    future_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in future_records:
        for case_id in record.get("case_study_ids") or ():
            if case_id in {case.name for case in cases}:
                future_by_case[case_id].append(record)

    rows: list[dict[str, Any]] = []
    semantic_audits: dict[str, dict[str, int]] = {}
    for window in windows:
        snapshot = snapshot_root / f"kg_{window.freeze_year}"
        graph_path = snapshot / "knowledge_graph.json"
        claims_path = snapshot / "extracted_claims.jsonl"
        print(f"[load] KG_{window.freeze_year}", flush=True)
        index = FrozenGraphIndex.load(graph_path)
        _, scoped, semantic_audit = load_semantic_claim_adjacencies(
            claims_path,
            index,
            (case.name for case in cases),
        )
        semantic_audits[str(window.freeze_year)] = semantic_audit

        for case in cases:
            required = _required_atoms(case)
            historical_edges = _unique_edges(scoped.get(case.name, {}))
            historical_components = component_summary(index, historical_edges, required)
            primary_claims = _primary_edge_count(case, index, historical_edges)
            historical_papers = {
                _paper_key(edge) for edge in historical_edges if _paper_key(edge)
            }
            complete_path = requires_complete_path(case.name)
            historical_ready = (
                historical_components["complete_cross_paper_components"] > 0
                if complete_path
                else primary_claims > 0
            )

            case_future = future_by_case.get(case.name, [])
            _, future_stats = _future_indexes(
                input_dir / "extracted_claims.jsonl",
                index.concepts,
                index.direct_pairs,
                window.future_start_year,
                window.future_end_year,
                case_study_id=case.name,
                future_records=case_future,
                historical_endpoint_atoms=index.atoms,
            )
            future_edges = _future_contract_edges(case_future, case, index, window)
            future_components = component_summary(index, future_edges, required)
            status, reason = _status(
                historical_ready=historical_ready,
                future_unique_pairs=int(future_stats.get("future_unique_pairs", 0)),
                complete_path=complete_path,
                future_complete_components=future_components["complete_components"],
            )
            row = {
                "case_study_id": case.name,
                "freeze_year": window.freeze_year,
                "future_start_year": window.future_start_year,
                "future_end_year": window.future_end_year,
                "required_atoms": ",".join(atom.value for atom in required),
                "complete_path_required": complete_path,
                "historical_contract_claims": len(historical_edges),
                "historical_primary_claims": primary_claims,
                "historical_papers": len(historical_papers),
                **{f"historical_{key}": value for key, value in historical_components.items()},
                "future_contract_claims": len(future_edges),
                "future_unique_pairs": int(future_stats.get("future_unique_pairs", 0)),
                "future_all_unique_pairs": int(future_stats.get("future_all_unique_pairs", 0)),
                "future_unavailable_endpoints": int(
                    future_stats.get("future_claims_unavailable_endpoint", 0)
                ),
                "future_relation_contract_rejected": int(
                    future_stats.get("case_study_relation_contract_rejected", 0)
                ),
                **{f"future_{key}": value for key, value in future_components.items()},
                "structural_status": status,
                "status_reason": reason,
            }
            rows.append(row)
            print(
                f"  {case.name}: {status}; historical={len(historical_edges)} "
                f"future_pairs={row['future_unique_pairs']}",
                flush=True,
            )

        del index, scoped
        gc.collect()

    status_counts = Counter(row["structural_status"] for row in rows)
    manifest = {
        "schema_version": "case-study-hindcasting-executability.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_release": canonical_release,
        "snapshot_root": str(snapshot_root),
        "windows": [window.__dict__ for window in windows],
        "case_studies": [case.name for case in cases],
        "future_records_cached": len(future_records),
        "status_counts": dict(status_counts),
        "evaluation_contract": {
            "semantic_endpoint_identity_version": SEMANTIC_ENDPOINT_IDENTITY_VERSION,
            "case_study_relation_contract_version": CASE_STUDY_RELATION_CONTRACT_VERSION,
            "discovery_metric_contract_version": DISCOVERY_METRIC_CONTRACT_VERSION,
            "minimum_future_pairs_for_stable_benchmark": (
                MIN_FUTURE_PAIRS_FOR_STABLE_BENCHMARK
            ),
            "complete_path_case_studies_require_future_complete_component": True,
        },
        "semantic_audit_by_freeze_year": semantic_audits,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "executability_audit.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "executability_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=parse_window,
        default=list(DEFAULT_WINDOWS),
    )
    parser.add_argument("--case-study-ids", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = (
        tuple(case_study_by_name(name) for name in args.case_study_ids)
        if args.case_study_ids
        else tuple(CASE_STUDIES)
    )
    manifest = audit(
        input_dir=args.input_dir.resolve(),
        snapshot_root=args.snapshot_root.resolve(),
        output_dir=args.output_dir.resolve(),
        windows=tuple(args.windows),
        cases=cases,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "rows": len(manifest["rows"]),
        "status_counts": manifest["status_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()


# Updated: 2026-08-13 06:27:05 HKT - require static future relations to use freeze-visible endpoint atoms.
