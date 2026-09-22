"""Fail-closed projection of a completed targeted manual Case-2 re-audit.

The targeted campaign reviews only claims that currently carry
``case2_pathway_mediation``.  This command preserves every paper, claim, and
non-Case-2 membership; it updates the Case-2 claim membership, recomputes the
paper-level union for all claims from affected papers, and synchronizes the
formal graph claim nodes, graph edge metadata, and extracted-claim store.

The default mode is a read-only plan.  ``--apply`` stages and fully validates
both canonical files, creates a recoverable backup, then replaces them and
regenerates ``CURRENT_STATE.json``/``README.md``.  Any mismatch fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scripts.canonical_kg_release import CURRENT_CANONICAL_SHA256
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    AUDIT_CONTRACT_VERSION,
    AUDIT_NAME,
    AUDIT_VERSION,
    claim_contract_fields,
)
from neurooracle.scripts.merge_case2_targeted_manual_results import (
    _canonical_hash,
    merge as merge_campaign,
)
from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    PaperIdentityUnion,
    RUBRIC_VERSION,
    compact_json,
    paper_aliases,
    strongest_paper_key,
)
from neurooracle.scripts.refresh_full_v2_state import build_state, render_readme
from neurooracle.scripts.streaming_graph_json import (
    is_claim_node,
    iter_concepts,
    iter_edges,
    rewrite_concepts,
    rewrite_edges,
)
from neurooracle.src.case_study_scope import (
    CASE_STUDY_IDS,
    apply_case_study_membership,
    claim_case_study_ids_from_dict,
    normalize_case_study_ids,
    paper_case_study_ids_from_dict,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO / "neurooracle" / "data" / "full_v2"
DEFAULT_CAMPAIGN = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_reaudit"
    / "case2_targeted_manual_20260810"
)
ARCHIVE_ROOT = REPO / "neurooracle" / "data" / "archive" / "kg_mutation_backups"
CANONICAL_RELEASE = REPO / "core" / "scripts" / "canonical_kg_release.py"
CASE2_ID = "case2_pathway_mediation"
TARGET_AUDIT_FIELD = "case2_targeted_manual_reaudit"
TARGET_AUDIT_SCHEMA = "case2_targeted_manual_reaudit_projection.v1"
CAMPAIGN_ID = "case2_targeted_manual_20260810"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered(values: object) -> list[str]:
    return normalize_case_study_ids(values, strict=True)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_verified_campaign(campaign: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-run all campaign validation and bind the stored merged artifact."""

    merged_path = campaign / "merged_manual_reaudit.json"
    summary_path = campaign / "summary.json"
    require(merged_path.is_file(), f"missing merged audit: {merged_path}")
    require(summary_path.is_file(), f"missing audit summary: {summary_path}")
    stored_merged = read_json(merged_path)
    stored_summary = read_json(summary_path)
    fresh_merged, fresh_summary = merge_campaign(campaign, partial=False)

    require(fresh_summary.get("validation_passed") is True, "campaign validation failed")
    require(fresh_summary.get("complete") is True, "campaign is incomplete")
    require(not fresh_summary.get("validation_errors"), "campaign has validation errors")
    require(not fresh_summary.get("missing_shards"), "campaign has missing shards")
    require(
        canonical_sha256(stored_merged) == canonical_sha256(fresh_merged),
        "stored merged audit differs from a fresh deterministic merge",
    )
    # ``formal_kg_modified`` is allowed to change only after a successful apply.
    comparable_summary = dict(stored_summary)
    for key in (
        "formal_kg_modified",
        "formal_kg_applied_at",
        "injection_report",
        "post_case2_statistics",
    ):
        comparable_summary.pop(key, None)
    fresh_comparable = dict(fresh_summary)
    fresh_comparable.pop("formal_kg_modified", None)
    require(
        canonical_sha256(comparable_summary) == canonical_sha256(fresh_comparable),
        "stored summary differs from a fresh deterministic merge",
    )
    require(stored_merged.get("case_study_id") == CASE2_ID, "wrong campaign scope")
    require(stored_merged.get("complete") is True, "merged campaign is incomplete")
    require(
        stored_merged.get("merged_decisions_sha256")
        == _canonical_hash(stored_merged.get("decisions") or []),
        "merged decision hash mismatch",
    )
    require(
        stored_summary.get("merged_decisions_sha256")
        == stored_merged.get("merged_decisions_sha256"),
        "summary/merged decision hashes disagree",
    )
    require(stored_summary.get("papers") == {"total": 912, "remove": 861, "retain": 51},
            "unexpected paper decision inventory")
    require(stored_summary.get("claims") == {"total": 939, "remove": 872, "retain": 67},
            "unexpected claim decision inventory")
    return stored_merged, stored_summary


def flatten_decisions(merged: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    claim_decisions: dict[str, dict[str, Any]] = {}
    paper_decisions: dict[str, dict[str, Any]] = {}
    for paper in merged.get("decisions") or []:
        paper_key = str(paper.get("paper_key") or "")
        require(paper_key and paper_key not in paper_decisions, f"duplicate paper decision: {paper_key}")
        paper_decisions[paper_key] = paper
        retained = 0
        for claim in paper.get("claim_decisions") or []:
            claim_id = str(claim.get("claim_id") or "")
            decision = str(claim.get("decision") or "")
            require(decision in {"retain", "remove"}, f"invalid claim decision: {claim_id}")
            require(claim_id and claim_id not in claim_decisions, f"duplicate claim decision: {claim_id}")
            claim_decisions[claim_id] = {
                "claim_id": claim_id,
                "claim_decision": decision,
                "claim_reason": str(claim.get("reason") or ""),
                "paper_key": paper_key,
                "paper_decision": str(paper.get("paper_decision") or ""),
                "reviewer_id": str(paper.get("reviewer_id") or "unknown"),
                "confidence": float(paper.get("confidence") or 0.0),
                "component_evidence": paper.get("component_evidence") or {},
                "rationale": str(paper.get("rationale") or ""),
                "audit_input_sha256": str(paper.get("audit_input_sha256") or ""),
                "paper_index": int(paper.get("paper_index")),
                "shard_index": int(paper.get("shard_index")),
            }
            retained += decision == "retain"
        paper_decision = str(paper.get("paper_decision") or "")
        require(paper_decision in {"retain", "remove"}, f"invalid paper decision: {paper_key}")
        require((retained > 0) == (paper_decision == "retain"),
                f"paper/claim decisions disagree: {paper_key}")
    require(len(paper_decisions) == 912, "paper decision inventory is not 912")
    require(len(claim_decisions) == 939, "claim decision inventory is not 939")
    return claim_decisions, paper_decisions


def verify_canonical_baseline(data_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    graph = data_dir / "knowledge_graph.json"
    extracted = data_dir / "extracted_claims.jsonl"
    state_path = data_dir / "CURRENT_STATE.json"
    for path in (graph, extracted, state_path, data_dir / "README.md"):
        require(path.is_file(), f"missing canonical artifact: {path}")
    state = read_json(state_path)
    require(state.get("status") == "canonical_current", "CURRENT_STATE is not canonical_current")
    require(state.get("taxonomy_version") == "case_study_membership.v2", "wrong taxonomy version")
    expected = {key: str(value).upper() for key, value in CURRENT_CANONICAL_SHA256.items()}
    actual = {
        "knowledge_graph": sha256_file(graph),
        "extracted_claims": sha256_file(extracted),
        "current_state": sha256_file(state_path),
    }
    require(actual == expected, f"canonical baseline hash mismatch: {actual} != {expected}")
    statistics = state.get("formal_kg_statistics") or {}
    case2 = (statistics.get("case_studies") or {}).get(CASE2_ID) or {}
    require(case2 == {"papers": 912, "claims": 939}, f"unexpected Case 2 baseline: {case2}")
    require((statistics.get("general") or {}).get("claims") == 299461,
            "unexpected formal claim inventory")
    return state, actual


def build_projection_plan(
    graph: Path,
    extracted: Path,
    claim_decisions: dict[str, dict[str, Any]],
    paper_decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Scan the formal graph and compute deterministic final memberships.

    The graph is authoritative for paper identity.  Some legacy extracted rows
    retain identifiers only in historical ``metadata``/``source`` fields even
    though their graph claim node has the canonical ``source_paper`` record.
    """

    union = PaperIdentityUnion()
    claim_roots: dict[str, str] = {}
    old_claim_labels: dict[str, tuple[str, ...]] = {}
    final_claim_labels: dict[str, tuple[str, ...]] = {}
    old_paper_labels: dict[str, tuple[str, ...]] = {}
    current_case2_claims: set[str] = set()
    rows = 0

    for node_id, node in iter_concepts(graph):
            if not is_claim_node(node_id, node):
                continue
            claim_id = node_id
            require(claim_id not in claim_roots, f"duplicate graph claim: {claim_id}")
            row = node.get("metadata")
            require(isinstance(row, dict), f"graph claim metadata missing: {claim_id}")
            nested = row.get("metadata")
            require(isinstance(nested, dict), f"claim metadata missing: {claim_id}")
            claim_labels = tuple(claim_case_study_ids_from_dict(row))
            paper_labels = tuple(paper_case_study_ids_from_dict(row))
            require(claim_labels == tuple(claim_case_study_ids_from_dict(nested)),
                    f"top/nested claim labels disagree: {claim_id}")
            require(paper_labels == tuple(paper_case_study_ids_from_dict(nested)),
                    f"top/nested paper labels disagree: {claim_id}")
            require(set(claim_labels).issubset(paper_labels),
                    f"claim labels are not a paper-label subset: {claim_id}")
            aliases = paper_aliases(row)
            require(bool(aliases), f"paper identity missing: {claim_id}")
            union.add(aliases)
            claim_roots[claim_id] = strongest_paper_key(row)
            old_claim_labels[claim_id] = claim_labels
            old_paper_labels[claim_id] = paper_labels
            if CASE2_ID in claim_labels:
                current_case2_claims.add(claim_id)
            decision = claim_decisions.get(claim_id)
            if decision is None:
                final_claim_labels[claim_id] = claim_labels
            elif decision["claim_decision"] == "retain":
                require(CASE2_ID in claim_labels, f"retained target lacked Case 2: {claim_id}")
                final_claim_labels[claim_id] = claim_labels
            else:
                require(CASE2_ID in claim_labels, f"removed target lacked Case 2: {claim_id}")
                final_claim_labels[claim_id] = tuple(label for label in claim_labels if label != CASE2_ID)
            rows += 1
            if rows % 50_000 == 0:
                log(f"planned {rows:,} graph claims")

    extracted_seen: set[str] = set()
    with extracted.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            require(claim_id in claim_roots and claim_id not in extracted_seen,
                    f"invalid/duplicate extracted claim at line {line_number}: {claim_id}")
            require(tuple(claim_case_study_ids_from_dict(row)) == old_claim_labels[claim_id],
                    f"graph/extracted claim labels disagree: {claim_id}")
            require(tuple(paper_case_study_ids_from_dict(row)) == old_paper_labels[claim_id],
                    f"graph/extracted paper labels disagree: {claim_id}")
            nested = row.get("metadata")
            require(isinstance(nested, dict), f"extracted claim metadata missing: {claim_id}")
            require(tuple(claim_case_study_ids_from_dict(nested)) == old_claim_labels[claim_id],
                    f"extracted top/nested claim labels disagree: {claim_id}")
            require(tuple(paper_case_study_ids_from_dict(nested)) == old_paper_labels[claim_id],
                    f"extracted top/nested paper labels disagree: {claim_id}")
            extracted_seen.add(claim_id)
    require(extracted_seen == set(claim_roots), "graph/extracted claim inventories differ")

    require(current_case2_claims == set(claim_decisions),
            "target audit inventory is not exactly the formal Case-2 claim inventory")

    paper_memberships: dict[str, set[str]] = defaultdict(set)
    paper_claim_counts: Counter[str] = Counter()
    for claim_id, initial_root in list(claim_roots.items()):
        root = union.find(initial_root)
        claim_roots[claim_id] = root
        paper_memberships[root].update(final_claim_labels[claim_id])
        paper_claim_counts[root] += 1

    final_paper_labels = {
        root: tuple(ordered(labels)) for root, labels in paper_memberships.items()
    }
    target_roots: dict[str, str] = {}
    for paper_key, paper in paper_decisions.items():
        root = union.find(paper_key)
        target_roots[paper_key] = root
        expected_has_case2 = paper["paper_decision"] == "retain"
        require((CASE2_ID in final_paper_labels[root]) == expected_has_case2,
                f"paper-level final membership disagrees with audit: {paper_key}")
    require(len(set(target_roots.values())) == 912,
            "target paper identities collapsed unexpectedly after identity union")

    current_case2_papers = {
        claim_roots[claim_id]
        for claim_id, labels in old_paper_labels.items()
        if CASE2_ID in labels
    }
    final_case2_papers = {
        root for root, labels in final_paper_labels.items() if CASE2_ID in labels
    }
    require(len(current_case2_papers) == 912, "formal Case-2 paper baseline is not 912")
    require(len(final_case2_papers) == 51, "projected Case-2 paper count is not 51")
    require(sum(CASE2_ID in labels for labels in final_claim_labels.values()) == 67,
            "projected Case-2 claim count is not 67")

    changed_claim_memberships = sum(
        old_claim_labels[claim_id] != final_claim_labels[claim_id]
        for claim_id in claim_roots
    )
    changed_paper_memberships = sum(
        old_paper_labels[claim_id] != final_paper_labels[claim_roots[claim_id]]
        for claim_id in claim_roots
    )
    return {
        "claim_roots": claim_roots,
        "old_claim_labels": old_claim_labels,
        "final_claim_labels": final_claim_labels,
        "old_paper_labels": old_paper_labels,
        "final_paper_labels": final_paper_labels,
        "paper_claim_counts": paper_claim_counts,
        "target_roots": target_roots,
        "rows": rows,
        "papers": len(final_paper_labels),
        "changed_claim_memberships": changed_claim_memberships,
        "changed_paper_memberships": changed_paper_memberships,
        "case2_before": {"papers": len(current_case2_papers), "claims": len(current_case2_claims)},
        "case2_after": {"papers": len(final_case2_papers), "claims": 67},
    }


def targeted_record(
    decision: dict[str, Any],
    *,
    merged: dict[str, Any],
    applied_at: str,
) -> dict[str, Any]:
    record = {
        "schema_version": TARGET_AUDIT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "case_study_id": CASE2_ID,
        "paper_index": decision["paper_index"],
        "shard_index": decision["shard_index"],
        "paper_key": decision["paper_key"],
        "paper_decision": decision["paper_decision"],
        "claim_decision": decision["claim_decision"],
        "reviewer_id": decision["reviewer_id"],
        "confidence": decision["confidence"],
        "audit_input_sha256": decision["audit_input_sha256"],
        "component_evidence": decision["component_evidence"],
        "rationale": decision["rationale"],
        "claim_reason": decision["claim_reason"],
        "merged_decisions_sha256": merged["merged_decisions_sha256"],
        "source_supplements_sha256": merged["source_supplements_sha256"],
        "applied_at": applied_at,
    }
    record["projection_decision_sha256"] = canonical_sha256(record)
    return record


def update_claim(
    claim: dict[str, Any],
    claim_id: str,
    *,
    plan: dict[str, Any],
    claim_decisions: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    applied_at: str,
    authoritative_target_source: bool,
) -> None:
    expected_old_claim = list(plan["old_claim_labels"][claim_id])
    expected_old_paper = list(plan["old_paper_labels"][claim_id])
    require(claim_case_study_ids_from_dict(claim) == expected_old_claim,
            f"claim changed after planning: {claim_id}")
    require(paper_case_study_ids_from_dict(claim) == expected_old_paper,
            f"paper membership changed after planning: {claim_id}")
    final_claim = list(plan["final_claim_labels"][claim_id])
    final_paper = list(plan["final_paper_labels"][plan["claim_roots"][claim_id]])
    apply_case_study_membership(
        claim,
        paper_case_study_ids=final_paper,
        claim_case_study_ids=final_claim,
    )
    metadata = claim.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        claim["metadata"] = metadata
    apply_case_study_membership(
        metadata,
        paper_case_study_ids=final_paper,
        claim_case_study_ids=final_claim,
    )
    metadata["case_study_membership_schema_version"] = "case_study_membership.v2"

    decision = claim_decisions.get(claim_id)
    if decision is None:
        return
    if authoritative_target_source:
        previous_audit = claim.get("scope_reaudit")
        require(isinstance(previous_audit, dict), f"target claim has no prior scope seal: {claim_id}")
        gates = dict(previous_audit.get("gates") or {})
        gates["case2_full_chain_verified"] = decision["claim_decision"] == "retain"
        contract = claim_contract_fields(
            paper_key=plan["claim_roots"][claim_id],
            payload=claim,
            labels=final_claim,
            gates=gates,
            rubric_version=RUBRIC_VERSION,
            case_study_ids=CASE_STUDY_IDS,
        )
        scope_audit = {
            "audit_name": AUDIT_NAME,
            "audit_version": AUDIT_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "review_stage": "targeted_manual_case2_adjudication",
            "reviewer_id": decision["reviewer_id"],
            "reasoning_effort": "manual_full_source_context",
            "reviewed_at": applied_at,
            "decision": "finalized",
            "confidence": decision["confidence"],
            "decision_basis": decision["rationale"],
            "gates": gates,
            "previous_claim_case_study_ids": expected_old_claim,
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            **contract,
        }
        provenance = targeted_record(decision, merged=merged, applied_at=applied_at)
        plan.setdefault("target_scope_audits", {})[claim_id] = scope_audit
        plan.setdefault("target_provenance", {})[claim_id] = provenance
    else:
        require(claim_id in plan.get("target_scope_audits", {}),
                f"authoritative graph target seal was not prepared: {claim_id}")
        scope_audit = plan["target_scope_audits"][claim_id]
        provenance = plan["target_provenance"][claim_id]
    claim["scope_reaudit"] = scope_audit
    claim[TARGET_AUDIT_FIELD] = provenance


def stage_projection(
    *,
    graph: Path,
    extracted: Path,
    stage_dir: Path,
    plan: dict[str, Any],
    claim_decisions: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    applied_at: str,
) -> tuple[Path, Path, dict[str, Any]]:
    graph_concepts = stage_dir / "knowledge_graph.concepts.tmp.json"
    graph_projected = stage_dir / "knowledge_graph.json"
    extracted_projected = stage_dir / "extracted_claims.jsonl"
    graph_seen: set[str] = set()
    target_graph_seen: set[str] = set()

    def transform_concept(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
        if not is_claim_node(node_id, node):
            return node
        require(node_id in plan["claim_roots"], f"graph claim absent from claim plan: {node_id}")
        require(node_id not in graph_seen, f"duplicate graph claim: {node_id}")
        claim = node.get("metadata")
        require(isinstance(claim, dict), f"graph claim metadata invalid: {node_id}")
        update_claim(
            claim,
            node_id,
            plan=plan,
            claim_decisions=claim_decisions,
            merged=merged,
            applied_at=applied_at,
            authoritative_target_source=True,
        )
        graph_seen.add(node_id)
        if node_id in claim_decisions:
            target_graph_seen.add(node_id)
        if len(graph_seen) % 50_000 == 0:
            log(f"rewrote {len(graph_seen):,} graph claim nodes")
        return node

    rewrite_concepts(graph, graph_concepts, transform_concept)
    require(graph_seen == set(plan["claim_roots"]), "graph/extracted claim inventories differ")
    require(target_graph_seen == set(claim_decisions), "graph misses targeted claims")

    edge_counts: Counter[str] = Counter()

    def transform_edge(edge: dict[str, Any]) -> dict[str, Any]:
        metadata = edge.get("metadata")
        if not isinstance(metadata, dict):
            return edge
        claim_id = str(metadata.get("claim_id") or "")
        if claim_id not in plan["claim_roots"]:
            return edge
        metadata["claim_case_study_ids"] = list(plan["final_claim_labels"][claim_id])
        metadata["paper_case_study_ids"] = list(
            plan["final_paper_labels"][plan["claim_roots"][claim_id]]
        )
        metadata["case_study_membership_schema_version"] = "case_study_membership.v2"
        edge_counts[claim_id] += 1
        return edge

    rewrite_edges(graph_concepts, graph_projected, transform_edge)
    graph_concepts.unlink()
    missing_target_edges = set(claim_decisions) - set(edge_counts)

    extracted_seen: set[str] = set()
    with extracted.open("r", encoding="utf-8") as source, extracted_projected.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            require(claim_id in plan["claim_roots"], f"unplanned extracted claim: {claim_id}")
            require(claim_id not in extracted_seen, f"duplicate extracted claim: {claim_id}")
            update_claim(
                row,
                claim_id,
                plan=plan,
                claim_decisions=claim_decisions,
                merged=merged,
                applied_at=applied_at,
                authoritative_target_source=False,
            )
            destination.write(compact_json(row) + "\n")
            extracted_seen.add(claim_id)
            if len(extracted_seen) % 50_000 == 0:
                log(f"rewrote {len(extracted_seen):,} extracted claims")
    require(extracted_seen == set(plan["claim_roots"]), "projected extracted inventory mismatch")
    return graph_projected, extracted_projected, {
        "graph_claims_rewritten": len(graph_seen),
        "graph_claim_edges_rewritten": sum(edge_counts.values()),
        "target_claim_edges_rewritten": sum(edge_counts[claim_id] for claim_id in claim_decisions),
        "target_claims_without_graph_edges": len(missing_target_edges),
        "extracted_claims_rewritten": len(extracted_seen),
    }


def validate_claim_payload(
    claim: dict[str, Any],
    claim_id: str,
    *,
    plan: dict[str, Any],
    claim_decisions: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    applied_at: str,
    validate_authoritative_contract: bool,
) -> str:
    final_claim = list(plan["final_claim_labels"][claim_id])
    final_paper = list(plan["final_paper_labels"][plan["claim_roots"][claim_id]])
    require(claim_case_study_ids_from_dict(claim) == final_claim,
            f"final claim membership mismatch: {claim_id}")
    require(paper_case_study_ids_from_dict(claim) == final_paper,
            f"final paper membership mismatch: {claim_id}")
    metadata = claim.get("metadata")
    require(isinstance(metadata, dict), f"final metadata missing: {claim_id}")
    require(claim_case_study_ids_from_dict(metadata) == final_claim,
            f"final nested claim membership mismatch: {claim_id}")
    require(paper_case_study_ids_from_dict(metadata) == final_paper,
            f"final nested paper membership mismatch: {claim_id}")
    decision = claim_decisions.get(claim_id)
    if decision is not None:
        audit = claim.get("scope_reaudit") or {}
        require(audit == plan.get("target_scope_audits", {}).get(claim_id),
                f"target scope audit differs from authoritative graph seal: {claim_id}")
        require(audit.get("review_stage") == "targeted_manual_case2_adjudication",
                f"target audit stage mismatch: {claim_id}")
        require(audit.get("audit_contract_version") == AUDIT_CONTRACT_VERSION,
                f"target audit contract mismatch: {claim_id}")
        gates = audit.get("gates") or {}
        require(gates.get("case2_full_chain_verified") is (CASE2_ID in final_claim),
                f"target Case-2 gate mismatch: {claim_id}")
        if validate_authoritative_contract:
            expected_contract = claim_contract_fields(
                paper_key=plan["claim_roots"][claim_id],
                payload=claim,
                labels=final_claim,
                gates=gates,
                rubric_version=RUBRIC_VERSION,
                case_study_ids=CASE_STUDY_IDS,
            )
            for key, value in expected_contract.items():
                require(str(audit.get(key) or "") == value,
                        f"target audit seal mismatch ({key}): {claim_id}")
        expected_target = plan.get("target_provenance", {}).get(claim_id)
        require(claim.get(TARGET_AUDIT_FIELD) == expected_target,
                f"target provenance mismatch: {claim_id}")
    digest_payload = {
        "claim_case_study_ids": final_claim,
        "paper_case_study_ids": final_paper,
        "scope_reaudit": claim.get("scope_reaudit"),
        TARGET_AUDIT_FIELD: claim.get(TARGET_AUDIT_FIELD),
    }
    return canonical_sha256(digest_payload)


def validate_projection(
    *,
    graph: Path,
    extracted: Path,
    plan: dict[str, Any],
    claim_decisions: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    applied_at: str,
) -> dict[str, Any]:
    graph_seen: set[str] = set()
    graph_digests: dict[str, str] = {}
    claim_counts: Counter[str] = Counter()
    for node_id, node in iter_concepts(graph):
        if not is_claim_node(node_id, node):
            continue
        require(node_id in plan["claim_roots"] and node_id not in graph_seen,
                f"invalid projected graph claim: {node_id}")
        claim = node.get("metadata")
        require(isinstance(claim, dict), f"invalid projected claim metadata: {node_id}")
        graph_digests[node_id] = validate_claim_payload(
            claim,
            node_id,
            plan=plan,
            claim_decisions=claim_decisions,
            merged=merged,
            applied_at=applied_at,
            validate_authoritative_contract=True,
        )
        claim_counts.update(plan["final_claim_labels"][node_id])
        graph_seen.add(node_id)
    require(graph_seen == set(plan["claim_roots"]), "projected graph inventory mismatch")

    edge_counts: Counter[str] = Counter()
    for edge in iter_edges(graph):
        metadata = edge.get("metadata")
        if not isinstance(metadata, dict):
            continue
        claim_id = str(metadata.get("claim_id") or "")
        if claim_id not in plan["claim_roots"]:
            continue
        require(metadata.get("claim_case_study_ids") == list(plan["final_claim_labels"][claim_id]),
                f"projected edge claim membership mismatch: {claim_id}")
        require(metadata.get("paper_case_study_ids") == list(
            plan["final_paper_labels"][plan["claim_roots"][claim_id]]
        ), f"projected edge paper membership mismatch: {claim_id}")
        edge_counts[claim_id] += 1
    target_claims_without_edges = set(claim_decisions) - set(edge_counts)

    extracted_seen: set[str] = set()
    with extracted.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            require(claim_id in plan["claim_roots"] and claim_id not in extracted_seen,
                    f"invalid projected extracted claim: {claim_id}")
            digest = validate_claim_payload(
                row,
                claim_id,
                plan=plan,
                claim_decisions=claim_decisions,
                merged=merged,
                applied_at=applied_at,
                validate_authoritative_contract=False,
            )
            require(graph_digests[claim_id] == digest,
                    f"graph/extracted routing provenance mismatch: {claim_id}")
            extracted_seen.add(claim_id)
    require(extracted_seen == graph_seen, "projected extracted inventory mismatch")
    require(claim_counts[CASE2_ID] == 67, "projected graph does not contain 67 Case-2 claims")
    return {
        "graph_claims": len(graph_seen),
        "extracted_claims": len(extracted_seen),
        "graph_claim_edges": sum(edge_counts.values()),
        "target_claim_edges": sum(edge_counts[claim_id] for claim_id in claim_decisions),
        "target_claims_without_graph_edges": len(target_claims_without_edges),
        "case2_claims": claim_counts[CASE2_ID],
        "case2_papers": sum(CASE2_ID in labels for labels in plan["final_paper_labels"].values()),
        "graph_extracted_provenance_equal": True,
    }


def backup_canonical(
    *, data_dir: Path, pre_hashes: dict[str, str], campaign_hash: str
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = ARCHIVE_ROOT / f"{CAMPAIGN_ID}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        data_dir / "knowledge_graph.json",
        data_dir / "extracted_claims.jsonl",
        data_dir / "CURRENT_STATE.json",
        data_dir / "README.md",
        CANONICAL_RELEASE,
    ]
    for source in sources:
        shutil.copy2(source, backup_dir / source.name)
    backup_hashes = {
        "knowledge_graph": sha256_file(backup_dir / "knowledge_graph.json"),
        "extracted_claims": sha256_file(backup_dir / "extracted_claims.jsonl"),
        "current_state": sha256_file(backup_dir / "CURRENT_STATE.json"),
    }
    require(backup_hashes == pre_hashes, "recoverable backup hash verification failed")
    manifest = {
        "schema_version": "kg_mutation_backup.v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": utc_now(),
        "authorization": "User explicitly requested formal KG injection in the active session.",
        "campaign_merged_sha256": campaign_hash,
        "canonical_pre_hashes": pre_hashes,
        "files": {
            source.name: {
                "bytes": (backup_dir / source.name).stat().st_size,
                "sha256": sha256_file(backup_dir / source.name),
            }
            for source in sources
        },
    }
    atomic_write_json(backup_dir / "backup_manifest.json", manifest)
    return backup_dir


def restore_backup(backup_dir: Path, data_dir: Path) -> None:
    for name in ("knowledge_graph.json", "extracted_claims.jsonl", "CURRENT_STATE.json", "README.md"):
        source = backup_dir / name
        if not source.exists():
            continue
        temporary = data_dir / f".{name}.rollback.{os.getpid()}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, data_dir / name)


def assert_expected_state(pre_state: dict[str, Any], post_state: dict[str, Any]) -> None:
    pre_stats = pre_state["formal_kg_statistics"]
    post_stats = post_state["formal_kg_statistics"]
    require(post_stats["general"] == pre_stats["general"], "general paper/claim totals changed")
    require(post_stats["case_studies"][CASE2_ID] == {"papers": 51, "claims": 67},
            "formal Case-2 statistics are not 51 papers / 67 claims")
    for case_study_id in CASE_STUDY_IDS:
        if case_study_id == CASE2_ID:
            continue
        require(post_stats["case_studies"][case_study_id]
                == pre_stats["case_studies"][case_study_id],
                f"non-Case-2 statistics changed: {case_study_id}")
    require(post_state["extracted_claim_store"] == pre_state["extracted_claim_store"],
            "extracted claim-store inventory/quality changed")
    quality_before = pre_stats["quality"]
    quality_after = post_stats["quality"]
    for key in (
        "claims_without_paper_identity",
        "claim_membership_not_subset_of_paper_membership",
        "claims_with_legacy_scope_fields",
    ):
        require(quality_after[key] == quality_before[key], f"quality metric changed: {key}")


def plan_report(
    *,
    campaign: Path,
    merged: dict[str, Any],
    pre_state: dict[str, Any],
    pre_hashes: dict[str, str],
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "case2_targeted_manual_reaudit_injection_plan.v1",
        "status": "validated_ready_to_apply",
        "generated_at": utc_now(),
        "campaign": str(campaign.resolve()),
        "campaign_merged_file_sha256": sha256_file(campaign / "merged_manual_reaudit.json"),
        "merged_decisions_sha256": merged["merged_decisions_sha256"],
        "source_supplements_sha256": merged["source_supplements_sha256"],
        "formal_pre_hashes": pre_hashes,
        "formal_general": pre_state["formal_kg_statistics"]["general"],
        "projection": {
            "claim_rows": plan["rows"],
            "identity_union_papers": plan["papers"],
            "changed_claim_memberships": plan["changed_claim_memberships"],
            "claim_rows_with_changed_paper_membership": plan["changed_paper_memberships"],
            "case2_before": plan["case2_before"],
            "case2_after": plan["case2_after"],
            "papers_or_claims_deleted": 0,
            "non_case2_claim_memberships_changed": 0,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    campaign = args.campaign.resolve()
    graph = data_dir / "knowledge_graph.json"
    extracted = data_dir / "extracted_claims.jsonl"
    state_path = data_dir / "CURRENT_STATE.json"
    readme_path = data_dir / "README.md"

    log("revalidating all targeted manual audit artifacts")
    merged, _summary = load_verified_campaign(campaign)
    claim_decisions, paper_decisions = flatten_decisions(merged)
    log("verifying canonical KG release fingerprints")
    pre_state, pre_hashes = verify_canonical_baseline(data_dir)
    log("building full paper-identity and membership projection plan")
    plan = build_projection_plan(graph, extracted, claim_decisions, paper_decisions)
    report = plan_report(
        campaign=campaign,
        merged=merged,
        pre_state=pre_state,
        pre_hashes=pre_hashes,
        plan=plan,
    )
    atomic_write_json(campaign / "injection_plan.json", report)
    if not args.apply:
        return report

    applied_at = utc_now()
    stage_dir = Path(tempfile.mkdtemp(prefix=".case2_targeted_stage_", dir=data_dir))
    backup_dir: Path | None = None
    swapped = False
    try:
        log(f"staging canonical projection in {stage_dir.name}")
        staged_graph, staged_extracted, stage_counts = stage_projection(
            graph=graph,
            extracted=extracted,
            stage_dir=stage_dir,
            plan=plan,
            claim_decisions=claim_decisions,
            merged=merged,
            applied_at=applied_at,
        )
        log("validating staged graph, edges, claim store, seals, and provenance")
        projection_validation = validate_projection(
            graph=staged_graph,
            extracted=staged_extracted,
            plan=plan,
            claim_decisions=claim_decisions,
            merged=merged,
            applied_at=applied_at,
        )
        log("running full canonical taxonomy validation on staged files")
        staged_state = build_state(stage_dir)
        assert_expected_state(pre_state, staged_state)

        log("creating and hash-verifying recoverable backup")
        backup_dir = backup_canonical(
            data_dir=data_dir,
            pre_hashes=pre_hashes,
            campaign_hash=sha256_file(campaign / "merged_manual_reaudit.json"),
        )
        log("atomically replacing canonical graph and extracted-claim store")
        os.replace(staged_graph, graph)
        swapped = True
        os.replace(staged_extracted, extracted)

        log("regenerating CURRENT_STATE.json and README.md from formal files")
        post_state = build_state(data_dir)
        assert_expected_state(pre_state, post_state)
        post_state["case_study_membership_reaudit"] = pre_state.get(
            "case_study_membership_reaudit"
        )
        post_state[TARGET_AUDIT_FIELD] = {
            "schema_version": TARGET_AUDIT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "case_study_id": CASE2_ID,
            "applied_at": applied_at,
            "papers_reviewed": 912,
            "claims_reviewed": 939,
            "papers_retained": 51,
            "claims_retained": 67,
            "papers_case2_membership_removed": 861,
            "claims_case2_membership_removed": 872,
            "papers_or_claims_deleted": 0,
            "merged_decisions_sha256": merged["merged_decisions_sha256"],
            "source_supplements_sha256": merged["source_supplements_sha256"],
            "backup_dir": str(backup_dir.resolve()),
        }
        atomic_write_json(state_path, post_state)
        atomic_write_text(readme_path, render_readme(post_state))

        post_hashes = {
            "knowledge_graph": sha256_file(graph),
            "extracted_claims": sha256_file(extracted),
            "current_state": sha256_file(state_path),
        }
        report.update(
            {
                "status": "applied_and_validated",
                "applied_at": applied_at,
                "backup_dir": str(backup_dir.resolve()),
                "stage_counts": stage_counts,
                "projection_validation": projection_validation,
                "formal_post_hashes": post_hashes,
                "formal_post_statistics": post_state["formal_kg_statistics"],
                "canonical_release_constants_update_required": True,
            }
        )
        atomic_write_json(campaign / "injection_report.json", report)
        atomic_write_json(backup_dir / "injection_report.json", report)
        summary = read_json(campaign / "summary.json")
        summary["formal_kg_modified"] = True
        summary["formal_kg_applied_at"] = applied_at
        summary["injection_report"] = str((campaign / "injection_report.json").resolve())
        summary["post_case2_statistics"] = {"papers": 51, "claims": 67}
        atomic_write_json(campaign / "summary.json", summary)
        return report
    except Exception:
        if swapped and backup_dir is not None:
            log("apply failed; restoring canonical files from verified backup")
            restore_backup(backup_dir, data_dir)
        raise
    finally:
        if stage_dir.exists() and not args.keep_stage:
            shutil.rmtree(stage_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--apply", action="store_true", help="mutate the canonical KG after all staging validations")
    parser.add_argument("--keep-stage", action="store_true", help="keep multi-GB staged files for debugging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
