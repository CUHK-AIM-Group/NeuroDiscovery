"""Stage, validate, and atomically apply a completed full-graph routing re-audit.

The command is guarded by exact source fingerprints and requires every formal
graph claim to have ``review_status=final_complete``.  Until then it only emits
an incompleteness report and cannot stage or mutate canonical files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_EXTRACTED,
    DEFAULT_GRAPH,
    DEFAULT_OUTPUT_DIR,
    RUBRIC_VERSION,
    compact_json,
    connect,
    file_fingerprint,
    get_meta,
)
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    AUDIT_CONTRACT_VERSION,
    AUDIT_NAME,
    AUDIT_VERSION,
    CONTRACT_FIELD_NAMES,
    claim_contract_fields,
    contract_context,
)
from neurooracle.scripts.refresh_full_v2_state import build_state, render_readme
from neurooracle.scripts.streaming_graph_json import (
    is_claim_node,
    iter_concepts,
    iter_edges,
    rewrite_concepts,
    rewrite_edges,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


REPO = Path(__file__).resolve().parents[2]
DEFAULT_STATE = REPO / "neurooracle" / "data" / "full_v2" / "CURRENT_STATE.json"
DEFAULT_README = REPO / "neurooracle" / "data" / "full_v2" / "README.md"
ARCHIVE_ROOT = REPO / "neurooracle" / "data" / "archive" / "kg_mutation_backups"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ordered(values: set[str]) -> list[str]:
    return [value for value in CASE_STUDY_IDS if value in values]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def readiness(connection: sqlite3.Connection) -> dict[str, Any]:
    total = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    counts = dict(
        connection.execute(
            "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
        ).fetchall()
    )
    final = counts.get("final_complete", 0)
    return {
        "claims": total,
        "final_complete": final,
        "remaining": total - final,
        "completion_percent": round(final * 100 / total, 6) if total else 100.0,
        "status_counts": counts,
        "ready": total > 0 and final == total,
    }


def verify_sources(
    connection: sqlite3.Connection, graph: Path, extracted: Path
) -> dict[str, Any]:
    recorded = get_meta(connection, "sources")
    if not isinstance(recorded, dict):
        raise ValueError("ledger source fingerprints are missing")
    current = {
        "graph": file_fingerprint(graph),
        "extracted": file_fingerprint(extracted),
        "abstract_caches": recorded.get("abstract_caches") or [],
    }
    if current["graph"] != recorded.get("graph") or current["extracted"] != recorded.get(
        "extracted"
    ):
        raise RuntimeError(
            "canonical KG files changed after audit inventory preparation; "
            "the ledger cannot be applied"
        )
    return current


def load_decisions(
    connection: sqlite3.Connection,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, str]]:
    decisions: dict[str, dict[str, Any]] = {}
    paper_memberships: dict[str, set[str]] = defaultdict(set)
    claim_papers: dict[str, str] = {}
    rows = connection.execute(
        """
        SELECT claim_id, paper_key, payload_json, current_claim_ids_json, review_json
        FROM claims WHERE review_status='final_complete' ORDER BY graph_ordinal
        """
    )
    for claim_id, paper_key, payload_json, previous_json, review_json in rows:
        review = json.loads(review_json)
        labels = list(review.get("claim_case_study_ids") or [])
        if labels != ordered(set(labels)):
            raise ValueError(f"noncanonical final labels for {claim_id}")
        expected_contract = claim_contract_fields(
            paper_key=str(paper_key),
            payload=json.loads(payload_json),
            labels=labels,
            gates=review.get("gates") or {},
            rubric_version=RUBRIC_VERSION,
            case_study_ids=CASE_STUDY_IDS,
        )
        if any(
            review.get(key) not in (None, "", expected_contract[key])
            for key in CONTRACT_FIELD_NAMES
        ):
            raise ValueError(f"immutable audit contract mismatch for {claim_id}")
        review.update(expected_contract)
        review["previous_claim_case_study_ids"] = json.loads(previous_json)
        decisions[claim_id] = review
        claim_papers[claim_id] = paper_key
        paper_memberships[paper_key].update(labels)
    return (
        decisions,
        {paper_key: ordered(labels) for paper_key, labels in paper_memberships.items()},
        claim_papers,
    )


def audit_record(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "review_stage": review.get("review_stage"),
        "reviewer_id": review.get("reviewer_id"),
        "reasoning_effort": review.get("reasoning_effort"),
        "reviewed_at": review.get("reviewed_at"),
        "decision": "finalized",
        "confidence": review.get("confidence"),
        "decision_basis": review.get("reason"),
        "gates": review.get("gates") or {},
        "previous_claim_case_study_ids": review.get(
            "previous_claim_case_study_ids"
        )
        or [],
        "audit_contract_version": review.get("audit_contract_version"),
        "rubric_sha256": review.get("rubric_sha256"),
        "case_study_registry_sha256": review.get(
            "case_study_registry_sha256"
        ),
        "claim_evidence_sha256": review.get("claim_evidence_sha256"),
        "audit_key": review.get("audit_key"),
        "decision_sha256": review.get("decision_sha256"),
    }


def apply_membership(
    claim: dict[str, Any],
    *,
    claim_ids: list[str],
    paper_ids: list[str],
    review: dict[str, Any],
) -> None:
    claim["claim_case_study_ids"] = claim_ids
    claim["paper_case_study_ids"] = paper_ids
    claim["scope_reaudit"] = audit_record(review)
    metadata = claim.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        claim["metadata"] = metadata
    metadata["claim_case_study_ids"] = claim_ids
    metadata["paper_case_study_ids"] = paper_ids
    metadata["case_study_membership_schema_version"] = "case_study_membership.v2"


def stage_projection(
    *,
    graph: Path,
    extracted: Path,
    output_dir: Path,
    decisions: dict[str, dict[str, Any]],
    paper_memberships: dict[str, list[str]],
    claim_papers: dict[str, str],
) -> dict[str, Any]:
    stage_dir = output_dir / "projection"
    stage_dir.mkdir(parents=True, exist_ok=True)
    concepts_temp = stage_dir / "knowledge_graph.concepts.tmp.json"
    graph_projected = stage_dir / "knowledge_graph.projected.json"
    extracted_projected = stage_dir / "extracted_claims.projected.jsonl"
    orphan_quarantine = output_dir / "quarantine" / "extracted_orphans.jsonl"
    orphan_quarantine.parent.mkdir(parents=True, exist_ok=True)
    for path in (concepts_temp, graph_projected, extracted_projected, orphan_quarantine):
        if path.exists():
            path.unlink()

    transformed_claims = 0

    def transform_concept(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
        nonlocal transformed_claims
        if not is_claim_node(node_id, node):
            return node
        review = decisions.get(node_id)
        if review is None:
            raise ValueError(f"graph claim lacks final decision: {node_id}")
        claim = node.get("metadata")
        if not isinstance(claim, dict):
            raise ValueError(f"graph claim metadata is invalid: {node_id}")
        labels = list(review["claim_case_study_ids"])
        paper_ids = paper_memberships[claim_papers[node_id]]
        apply_membership(claim, claim_ids=labels, paper_ids=paper_ids, review=review)
        transformed_claims += 1
        return node

    rewrite_concepts(graph, concepts_temp, transform_concept)

    transformed_claim_edges = 0

    def transform_edge(edge: dict[str, Any]) -> dict[str, Any]:
        nonlocal transformed_claim_edges
        metadata = edge.get("metadata")
        if not isinstance(metadata, dict):
            return edge
        claim_id = str(metadata.get("claim_id") or "")
        review = decisions.get(claim_id)
        if review is None:
            return edge
        metadata["claim_case_study_ids"] = list(review["claim_case_study_ids"])
        metadata["paper_case_study_ids"] = paper_memberships[claim_papers[claim_id]]
        metadata["case_study_membership_schema_version"] = "case_study_membership.v2"
        transformed_claim_edges += 1
        return edge

    rewrite_edges(concepts_temp, graph_projected, transform_edge)
    concepts_temp.unlink()

    extracted_seen: set[str] = set()
    orphan_rows = 0
    with extracted.open("r", encoding="utf-8") as source, extracted_projected.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination, orphan_quarantine.open("w", encoding="utf-8", newline="\n") as quarantine:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            review = decisions.get(claim_id)
            if review is None:
                quarantine.write(
                    compact_json(
                        {
                            "reason": "extracted_row_absent_from_formal_graph",
                            "source_line": line_number,
                            "row": row,
                        }
                    )
                    + "\n"
                )
                orphan_rows += 1
                continue
            if claim_id in extracted_seen:
                raise ValueError(f"duplicate extracted claim during projection: {claim_id}")
            extracted_seen.add(claim_id)
            apply_membership(
                row,
                claim_ids=list(review["claim_case_study_ids"]),
                paper_ids=paper_memberships[claim_papers[claim_id]],
                review=review,
            )
            destination.write(compact_json(row) + "\n")
    if extracted_seen != decisions.keys():
        missing = set(decisions) - extracted_seen
        raise ValueError(f"projected extracted store is missing claims: {sorted(missing)[:5]}")

    validation = validate_projection(
        graph_projected,
        extracted_projected,
        decisions,
        paper_memberships,
        claim_papers,
    )
    return {
        "graph_projected": str(graph_projected.resolve()),
        "extracted_projected": str(extracted_projected.resolve()),
        "orphan_quarantine": str(orphan_quarantine.resolve()),
        "claims_transformed": transformed_claims,
        "claim_edges_transformed": transformed_claim_edges,
        "orphan_extracted_rows_quarantined": orphan_rows,
        "validation": validation,
    }


def validate_projection(
    graph: Path,
    extracted: Path,
    decisions: dict[str, dict[str, Any]],
    paper_memberships: dict[str, list[str]],
    claim_papers: dict[str, str],
) -> dict[str, Any]:
    graph_claims: set[str] = set()
    claim_label_counts: Counter[str] = Counter()
    for node_id, node in iter_concepts(graph):
        if not is_claim_node(node_id, node):
            continue
        review = decisions.get(node_id)
        if review is None:
            raise ValueError(f"unexpected claim in projected graph: {node_id}")
        claim = node.get("metadata") or {}
        expected_claim = list(review["claim_case_study_ids"])
        expected_paper = paper_memberships[claim_papers[node_id]]
        if claim.get("claim_case_study_ids") != expected_claim:
            raise ValueError(f"projected graph claim membership mismatch: {node_id}")
        if claim.get("paper_case_study_ids") != expected_paper:
            raise ValueError(f"projected graph paper membership mismatch: {node_id}")
        scope_audit = claim.get("scope_reaudit") or {}
        if scope_audit.get("audit_version") != AUDIT_VERSION:
            raise ValueError(f"projected graph audit marker mismatch: {node_id}")
        if scope_audit.get("audit_contract_version") != AUDIT_CONTRACT_VERSION:
            raise ValueError(f"projected graph audit contract mismatch: {node_id}")
        if scope_audit.get("audit_key") != review.get("audit_key"):
            raise ValueError(f"projected graph audit key mismatch: {node_id}")
        if scope_audit.get("decision_sha256") != review.get("decision_sha256"):
            raise ValueError(f"projected graph decision hash mismatch: {node_id}")
        graph_claims.add(node_id)
        claim_label_counts.update(expected_claim)
    if graph_claims != decisions.keys():
        raise ValueError("projected graph claim inventory does not equal audit ledger")

    checked_edges = 0
    for edge in iter_edges(graph):
        metadata = edge.get("metadata") or {}
        claim_id = str(metadata.get("claim_id") or "")
        review = decisions.get(claim_id)
        if review is None:
            continue
        if metadata.get("claim_case_study_ids") != review["claim_case_study_ids"]:
            raise ValueError(f"projected edge claim membership mismatch: {claim_id}")
        if metadata.get("paper_case_study_ids") != paper_memberships[claim_papers[claim_id]]:
            raise ValueError(f"projected edge paper membership mismatch: {claim_id}")
        checked_edges += 1

    extracted_ids: set[str] = set()
    with extracted.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            if claim_id in extracted_ids or claim_id not in decisions:
                raise ValueError(f"invalid projected extracted claim ID: {claim_id}")
            extracted_ids.add(claim_id)
            if row.get("claim_case_study_ids") != decisions[claim_id][
                "claim_case_study_ids"
            ]:
                raise ValueError(f"projected extracted membership mismatch: {claim_id}")
            if row.get("paper_case_study_ids") != paper_memberships[
                claim_papers[claim_id]
            ]:
                raise ValueError(f"projected extracted paper membership mismatch: {claim_id}")
            scope_audit = row.get("scope_reaudit") or {}
            if scope_audit.get("audit_key") != decisions[claim_id].get("audit_key"):
                raise ValueError(f"projected extracted audit key mismatch: {claim_id}")
            if scope_audit.get("decision_sha256") != decisions[claim_id].get(
                "decision_sha256"
            ):
                raise ValueError(
                    f"projected extracted decision hash mismatch: {claim_id}"
                )
    if extracted_ids != decisions.keys():
        raise ValueError("projected extracted inventory does not equal audit ledger")

    paper_counts: Counter[str] = Counter()
    for labels in paper_memberships.values():
        paper_counts.update(labels)
    return {
        "claims": len(graph_claims),
        "claim_edges_checked": checked_edges,
        "extracted_rows": len(extracted_ids),
        "claim_counts": {
            case_study_id: claim_label_counts[case_study_id]
            for case_study_id in CASE_STUDY_IDS
        },
        "paper_counts": {
            case_study_id: paper_counts[case_study_id]
            for case_study_id in CASE_STUDY_IDS
        },
    }


def apply_projection(
    *,
    graph: Path,
    extracted: Path,
    state_path: Path,
    stage_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = ARCHIVE_ROOT / f"full_graph_case_study_reaudit_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in (graph, extracted, state_path, DEFAULT_README):
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    swapped = False
    try:
        swapped = True
        shutil.copyfile(Path(stage_report["graph_projected"]), graph)
        shutil.copyfile(Path(stage_report["extracted_projected"]), extracted)
        state = build_state(graph.parent)
        state["case_study_membership_reaudit"] = {
            "audit_name": AUDIT_NAME,
            "audit_version": AUDIT_VERSION,
            "rubric_version": RUBRIC_VERSION,
            **contract_context(
                rubric_version=RUBRIC_VERSION,
                case_study_ids=CASE_STUDY_IDS,
            ),
            "completed_at": utc_now(),
            "claims_reviewed": stage_report["validation"]["claims"],
            "orphan_extracted_rows_quarantined": stage_report[
                "orphan_extracted_rows_quarantined"
            ],
        }
        write_json(state_path, state)
        DEFAULT_README.write_text(render_readme(state), encoding="utf-8")
        return {
            "status": "applied_and_validated",
            "applied_at": utc_now(),
            "backup_dir": str(backup_dir.resolve()),
            "state": state,
        }
    except Exception:
        if swapped:
            for path in (graph, extracted, state_path, DEFAULT_README):
                backup = backup_dir / path.name
                if backup.exists():
                    shutil.copy2(backup, path)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = output_dir / "reaudit.sqlite"
    connection = connect(ledger)
    try:
        source_report = verify_sources(connection, args.graph.resolve(), args.extracted.resolve())
        ready = readiness(connection)
        report: dict[str, Any] = {
            "schema_version": "full_graph_case_study_reaudit_apply.v1",
            "created_at": utc_now(),
            "sources": source_report,
            "readiness": ready,
            "requested_stage": args.stage or args.apply,
            "requested_apply": args.apply,
        }
        if not ready["ready"]:
            report["status"] = "incomplete_refused"
            write_json(output_dir / "apply_readiness.json", report)
            return report
        decisions, paper_memberships, claim_papers = load_decisions(connection)
    finally:
        connection.close()

    if not (args.stage or args.apply):
        report["status"] = "ready_not_staged"
        write_json(output_dir / "apply_readiness.json", report)
        return report
    stage_report = stage_projection(
        graph=args.graph.resolve(),
        extracted=args.extracted.resolve(),
        output_dir=output_dir,
        decisions=decisions,
        paper_memberships=paper_memberships,
        claim_papers=claim_papers,
    )
    report["stage"] = stage_report
    report["status"] = "staged_and_validated"
    if args.apply:
        report["apply"] = apply_projection(
            graph=args.graph.resolve(),
            extracted=args.extracted.resolve(),
            state_path=args.state.resolve(),
            stage_report=stage_report,
            output_dir=output_dir,
        )
        report["status"] = "applied_and_validated"
    write_json(output_dir / "apply_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "failed_at": utc_now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output_dir.resolve() / "apply_failed.json", failure)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
