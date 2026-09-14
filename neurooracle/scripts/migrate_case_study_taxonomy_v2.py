"""Migrate the formal KG from the legacy Case 3 taxonomy to 17 peer scopes.

The command is dry-run by default.  ``--apply`` writes and validates temporary
canonical files, moves the exact v1 sources into a dated archive directory, and
then atomically installs both v2 files.  It never edits either formal source in
place.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.count_case_study_kg_stats import (
    PaperIdentityIndex,
    paper_aliases,
)
from neurooracle.scripts.streaming_graph_json import (
    is_claim_node,
    iter_concepts,
    rewrite_concepts,
)
from neurooracle.src.case_study_scope import (
    CASE_STUDY_IDS,
    TASK_CASE_STUDY_IDS,
    apply_case_study_membership,
    canonical_case_study_id,
    claim_case_study_ids_from_dict,
    paper_case_study_ids_from_dict,
)


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "neurooracle" / "data"
DEFAULT_GRAPH = DATA_ROOT / "full_v2" / "knowledge_graph.json"
DEFAULT_EXTRACTED = DATA_ROOT / "full_v2" / "extracted_claims.jsonl"
DEFAULT_REPORT = DATA_ROOT / "migration_reports" / "case_study_taxonomy_v2.json"
DEFAULT_ARCHIVE = DATA_ROOT / "archive" / "legacy_case3_taxonomy_20260801"
SCHEMA_VERSION = "case_study_membership.v2"


def _nested_metadata(claim: dict[str, Any]) -> dict[str, Any]:
    metadata = claim.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        claim["metadata"] = metadata
    return metadata


def _legacy_values(claim: dict[str, Any], key: str) -> list[str]:
    for holder in (claim, claim.get("metadata") or {}):
        if key not in holder:
            continue
        value = holder.get(key)
        values = value if isinstance(value, list) else [value]
        return [str(item or "").strip() for item in values if str(item or "").strip()]
    return []


def _has_legacy_case3_scope(claim: dict[str, Any]) -> bool:
    values = [value.lower() for value in _legacy_values(claim, "paper_scope")]
    return any(value in {"case3", "cs3", "case_3", "case-3", "case study 3"} for value in values)


def _unknown_legacy_tasks(claim: dict[str, Any]) -> set[str]:
    raw = _legacy_values(claim, "case3_tasks") or _legacy_values(claim, "case3_subtasks")
    return {
        value
        for value in raw
        if not canonical_case_study_id(value)
        and value.lower() not in {"case3", "hindcasting"}
    }


def _claim_from_node(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
    claim = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    claim.setdefault("id", node_id)
    return claim


def build_graph_membership_index(
    graph_path: Path,
) -> tuple[PaperIdentityIndex, dict[str, Any]]:
    papers = PaperIdentityIndex()
    claim_counts: Counter[str] = Counter()
    report: dict[str, Any] = {
        "concept_nodes_scanned": 0,
        "claim_nodes": 0,
        "legacy_case3_claims_without_promoted_scope": 0,
        "claims_with_transdiagnostic_legacy_task": 0,
        "claims_without_paper_identity": 0,
        "unknown_legacy_task_labels": Counter(),
    }
    for node_id, node in iter_concepts(graph_path):
        report["concept_nodes_scanned"] += 1
        if not is_claim_node(node_id, node):
            continue
        report["claim_nodes"] += 1
        claim = _claim_from_node(node_id, node)
        claim_ids = claim_case_study_ids_from_dict(claim)
        if "transdiagnostic_clustering" in _legacy_values(claim, "case3_tasks"):
            report["claims_with_transdiagnostic_legacy_task"] += 1
        if _has_legacy_case3_scope(claim) and not (
            set(claim_ids) & set(TASK_CASE_STUDY_IDS)
        ) and "case1_transdiagnostic" not in claim_ids:
            report["legacy_case3_claims_without_promoted_scope"] += 1
        for label in _unknown_legacy_tasks(claim):
            report["unknown_legacy_task_labels"][label] += 1
        for case_study_id in claim_ids:
            claim_counts[case_study_id] += 1
        aliases = paper_aliases(claim)
        if not aliases:
            aliases = [f"claim:{node_id}"]
            report["claims_without_paper_identity"] += 1
        papers.add(aliases, claim_ids)

    roots = papers.roots()
    paper_counts: Counter[str] = Counter()
    for memberships in roots.values():
        for case_study_id in memberships:
            paper_counts[case_study_id] += 1
    report["projected_general"] = {
        "papers": len(roots),
        "claims": report["claim_nodes"],
    }
    report["projected_case_studies"] = {
        case_study_id: {
            "papers": paper_counts[case_study_id],
            "claims": claim_counts[case_study_id],
        }
        for case_study_id in CASE_STUDY_IDS
    }
    report["projected_multi_label_papers"] = sum(
        len(ids) > 1 for ids in roots.values()
    )
    report["unknown_legacy_task_labels"] = dict(
        sorted(report["unknown_legacy_task_labels"].items())
    )
    return papers, report


def scan_extracted(extracted_path: Path) -> dict[str, Any]:
    rows = 0
    legacy_rows = 0
    rows_without_membership = 0
    invalid_rows = 0
    for number, line in enumerate(extracted_path.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            continue
        rows += 1
        try:
            claim = json.loads(line)
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        if any(
            key in claim or key in (claim.get("metadata") or {})
            for key in ("paper_scope", "case3_tasks", "case3_subtasks")
        ):
            legacy_rows += 1
        if not claim_case_study_ids_from_dict(claim):
            rows_without_membership += 1
    return {
        "rows": rows,
        "legacy_rows": legacy_rows,
        "rows_without_case_study_membership": rows_without_membership,
        "invalid_json_rows": invalid_rows,
    }


def _paper_ids(
    papers: PaperIdentityIndex,
    claim: dict[str, Any],
    fallback: str,
) -> list[str]:
    aliases = paper_aliases(claim) or [fallback]
    root = papers.find(aliases[0])
    memberships = papers.roots().get(root, set())
    return [case_study_id for case_study_id in CASE_STUDY_IDS if case_study_id in memberships]


def rewrite_graph(
    graph_path: Path,
    output_path: Path,
    papers: PaperIdentityIndex,
) -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, int]]:
    roots = papers.roots()
    routing: dict[str, tuple[list[str], list[str]]] = {}
    counters = {"claims_rewritten": 0, "concepts_rewritten": 0}

    def transform(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
        if not is_claim_node(node_id, node):
            return node
        claim = _claim_from_node(node_id, node)
        claim_ids = claim_case_study_ids_from_dict(claim)
        aliases = paper_aliases(claim) or [f"claim:{node_id}"]
        root = papers.find(aliases[0])
        paper_ids = [
            case_study_id
            for case_study_id in CASE_STUDY_IDS
            if case_study_id in roots.get(root, set())
        ]
        metadata = _nested_metadata(claim)
        apply_case_study_membership(
            claim,
            paper_case_study_ids=paper_ids,
            claim_case_study_ids=claim_ids,
        )
        apply_case_study_membership(
            metadata,
            paper_case_study_ids=paper_ids,
            claim_case_study_ids=claim_ids,
        )
        metadata["case_study_membership_schema_version"] = SCHEMA_VERSION
        node["metadata"] = claim
        routing[node_id] = (paper_ids, claim_ids)
        counters["claims_rewritten"] += 1
        return node

    counters["concepts_rewritten"] = rewrite_concepts(graph_path, output_path, transform)
    return routing, counters


def build_extracted_only_index(
    extracted_path: Path,
    graph_claim_ids: set[str],
) -> PaperIdentityIndex:
    papers = PaperIdentityIndex()
    with extracted_path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                claim = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid extracted JSONL line {number}: {error}") from error
            claim_id = str(claim.get("id") or "")
            if claim_id in graph_claim_ids:
                continue
            aliases = paper_aliases(claim) or [f"claim:{claim_id or number}"]
            papers.add(aliases, claim_case_study_ids_from_dict(claim))
    return papers


def rewrite_extracted(
    extracted_path: Path,
    output_path: Path,
    routing: dict[str, tuple[list[str], list[str]]],
) -> dict[str, int]:
    extracted_only = build_extracted_only_index(extracted_path, set(routing))
    extra_roots = extracted_only.roots()
    seen_graph_claims: set[str] = set()
    rows = 0
    extracted_only_rows = 0
    with extracted_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                claim = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid extracted JSONL line {number}: {error}") from error
            claim_id = str(claim.get("id") or "")
            if claim_id in routing:
                paper_ids, claim_ids = routing[claim_id]
                seen_graph_claims.add(claim_id)
            else:
                extracted_only_rows += 1
                claim_ids = claim_case_study_ids_from_dict(claim)
                aliases = paper_aliases(claim) or [f"claim:{claim_id or number}"]
                root = extracted_only.find(aliases[0])
                paper_ids = [
                    case_study_id
                    for case_study_id in CASE_STUDY_IDS
                    if case_study_id in extra_roots.get(root, set())
                ]
            metadata = _nested_metadata(claim)
            apply_case_study_membership(
                claim,
                paper_case_study_ids=paper_ids,
                claim_case_study_ids=claim_ids,
            )
            apply_case_study_membership(
                metadata,
                paper_case_study_ids=paper_ids,
                claim_case_study_ids=claim_ids,
            )
            metadata["case_study_membership_schema_version"] = SCHEMA_VERSION
            output.write(json.dumps(claim, ensure_ascii=False, separators=(",", ":")) + "\n")

    missing = set(routing) - seen_graph_claims
    if missing:
        raise ValueError(
            f"extracted claim store is missing {len(missing)} KG claims; "
            f"examples: {', '.join(sorted(missing)[:5])}"
        )
    return {
        "rows_rewritten": rows,
        "graph_claims_matched": len(seen_graph_claims),
        "extracted_only_rows": extracted_only_rows,
    }


def validate_rewritten_graph(graph_path: Path, expected_claims: int) -> dict[str, int]:
    concepts = 0
    claims = 0
    legacy_claims = 0
    invalid_subset = 0
    for node_id, node in iter_concepts(graph_path):
        concepts += 1
        if not is_claim_node(node_id, node):
            continue
        claims += 1
        claim = _claim_from_node(node_id, node)
        if any(
            key in claim or key in (claim.get("metadata") or {})
            for key in ("paper_scope", "case3_tasks", "case3_subtasks")
        ):
            legacy_claims += 1
        claim_ids = claim_case_study_ids_from_dict(claim)
        paper_ids = paper_case_study_ids_from_dict(claim)
        if not set(claim_ids).issubset(paper_ids):
            invalid_subset += 1
    if claims != expected_claims or legacy_claims or invalid_subset:
        raise ValueError(
            "rewritten graph validation failed: "
            f"claims={claims}/{expected_claims}, legacy={legacy_claims}, "
            f"invalid_subset={invalid_subset}"
        )
    return {
        "concept_nodes": concepts,
        "claim_nodes": claims,
        "legacy_claim_nodes": legacy_claims,
        "invalid_membership_subset": invalid_subset,
    }


def _assert_archive_target(archive_dir: Path) -> None:
    archive_root = (DATA_ROOT / "archive").resolve()
    resolved = archive_dir.resolve()
    if resolved == archive_root or not resolved.is_relative_to(archive_root):
        raise ValueError(f"archive directory must be a child of {archive_root}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"archive directory is not empty: {resolved}")


def apply_migration(
    graph_path: Path,
    extracted_path: Path,
    archive_dir: Path,
    papers: PaperIdentityIndex,
    expected_claims: int,
) -> dict[str, Any]:
    _assert_archive_target(archive_dir)
    graph_temp = graph_path.with_suffix(graph_path.suffix + ".taxonomy-v2.tmp")
    extracted_temp = extracted_path.with_suffix(extracted_path.suffix + ".taxonomy-v2.tmp")
    for temp in (graph_temp, extracted_temp):
        if temp.exists():
            temp.unlink()

    try:
        routing, graph_rewrite = rewrite_graph(graph_path, graph_temp, papers)
        graph_validation = validate_rewritten_graph(graph_temp, expected_claims)
        extracted_rewrite = rewrite_extracted(extracted_path, extracted_temp, routing)

        archive_dir.mkdir(parents=True, exist_ok=False)
        archived_graph = archive_dir / "knowledge_graph.case3_taxonomy_v1.json"
        archived_extracted = archive_dir / "extracted_claims.case3_taxonomy_v1.jsonl"
        os.replace(graph_path, archived_graph)
        os.replace(extracted_path, archived_extracted)
        try:
            os.replace(graph_temp, graph_path)
            os.replace(extracted_temp, extracted_path)
        except Exception:
            if not graph_path.exists() and archived_graph.exists():
                os.replace(archived_graph, graph_path)
            if not extracted_path.exists() and archived_extracted.exists():
                os.replace(archived_extracted, extracted_path)
            raise
    finally:
        for temp in (graph_temp, extracted_temp):
            if temp.exists():
                temp.unlink()

    return {
        "graph_rewrite": graph_rewrite,
        "graph_validation": graph_validation,
        "extracted_rewrite": extracted_rewrite,
        "archive_dir": str(archive_dir),
        "archived_graph": str(archived_graph),
        "archived_extracted": str(archived_extracted),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph.resolve()
    extracted_path = args.extracted.resolve()
    papers, graph_report = build_graph_membership_index(graph_path)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "source": {
            "graph": str(graph_path),
            "graph_bytes": graph_path.stat().st_size,
            "extracted": str(extracted_path),
            "extracted_bytes": extracted_path.stat().st_size,
        },
        "graph_projection": graph_report,
        "extracted_scan": scan_extracted(extracted_path),
    }
    if report["extracted_scan"]["invalid_json_rows"]:
        raise ValueError("extracted claim store contains invalid JSON rows")
    if graph_report["unknown_legacy_task_labels"]:
        raise ValueError(
            "unknown legacy task labels require manual mapping: "
            f"{graph_report['unknown_legacy_task_labels']}"
        )
    if graph_report["legacy_case3_claims_without_promoted_scope"]:
        raise ValueError(
            "legacy Case 3 claims without a promoted case-study scope: "
            f"{graph_report['legacy_case3_claims_without_promoted_scope']}"
        )
    if args.apply:
        report["application"] = apply_migration(
            graph_path,
            extracted_path,
            args.archive_dir.resolve(),
            papers,
            graph_report["claim_nodes"],
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
