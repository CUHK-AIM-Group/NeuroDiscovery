"""Prepare a durable, one-row-per-claim Case Study re-audit ledger.

The formal knowledge graph is the inventory source of truth.  The extracted
claim store and abstract cache are joined as supporting context, but they are
never allowed to silently add claims to the review universe.

This command is intentionally read-only with respect to ``full_v2``.  It writes
an SQLite ledger and a JSON manifest under a separate audit directory so the
semantic review can resume after interruption and the eventual KG mutation can
be gated on complete coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from neurooracle.scripts.count_case_study_kg_stats import clean_doi, clean_title
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    AUDIT_NAME,
    AUDIT_VERSION,
    claim_contract_fields,
    contract_context,
    embedded_contract_matches,
)
from neurooracle.scripts.streaming_graph_json import is_claim_node, iter_concepts
from neurooracle.src.case_study_scope import (
    CASE_STUDY_IDS,
    claim_case_study_ids_from_dict,
    paper_case_study_ids_from_dict,
)
from neurooracle.src.case_study_membership_policy import RUBRIC_VERSION


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_EXTRACTED = REPO / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
DEFAULT_ABSTRACT_CACHE = (
    REPO / "neurooracle" / "data" / "full_snapshot_v2" / "abstract_cache.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO / "neurooracle" / "data" / "case_study_reaudit" / "full_graph_v3"
)
SCHEMA_VERSION = "full_graph_case_study_reaudit.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def normalize_identifier(value: object) -> str:
    return str(value or "").strip().casefold()


def paper_aliases(claim: dict[str, Any]) -> list[str]:
    paper = claim.get("source_paper") or {}
    aliases: list[str] = []
    for prefix, value in (
        ("pmid", paper.get("pmid")),
        ("doi", clean_doi(paper.get("doi"))),
        ("pmcid", paper.get("pmcid")),
        ("arxiv", paper.get("arxiv_id")),
        ("openalex", paper.get("openalex_id")),
    ):
        normalized = normalize_identifier(value)
        if normalized:
            aliases.append(f"{prefix}:{normalized}")
    title = clean_title(paper.get("title"))
    year = str(paper.get("year") or paper.get("publication_year") or "")
    if title:
        aliases.append(f"title_year:{title}|{year}")
    metadata = claim.get("metadata") or {}
    fallback = normalize_identifier(claim.get("paper_id") or metadata.get("paper_id"))
    if fallback:
        aliases.append(f"paper_id:{fallback}")
    return list(dict.fromkeys(aliases))


def strongest_paper_key(claim: dict[str, Any]) -> str:
    aliases = paper_aliases(claim)
    if aliases:
        return aliases[0]
    # The formal graph currently has no identity-free claims, but keep a stable
    # fallback so preparation fails visibly rather than merging unknown papers.
    claim_id = str(claim.get("id") or "")
    return f"claim_fallback:{claim_id}"


class PaperIdentityUnion:
    """Union paper aliases using the same identity semantics as KG statistics."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, alias: str) -> str:
        self.parent.setdefault(alias, alias)
        root = alias
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[alias] != alias:
            next_alias = self.parent[alias]
            self.parent[alias] = root
            alias = next_alias
        return root

    def add(self, aliases: list[str]) -> str:
        if not aliases:
            raise ValueError("paper identity union requires at least one alias")
        root = self.find(aliases[0])
        for alias in aliases[1:]:
            other = self.find(alias)
            if other != root:
                self.parent[other] = root
        return root

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for alias in self.parent:
            result.setdefault(self.find(alias), []).append(alias)
        return result


def claim_payload(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
    claim = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    nested = claim.get("metadata") if isinstance(claim.get("metadata"), dict) else {}
    paper = claim.get("source_paper") if isinstance(claim.get("source_paper"), dict) else {}
    evidence = claim.get("evidence") if isinstance(claim.get("evidence"), dict) else {}
    payload = {
        "id": node_id,
        "subject_name": claim.get("subject_name") or "",
        "predicate": claim.get("predicate") or "",
        "object_name": claim.get("object_name") or "",
        "negated": bool(claim.get("negated", False)),
        "raw_text": claim.get("raw_text") or node.get("definition") or "",
        "subject_type": nested.get("subject_type") or "",
        "object_type": nested.get("object_type") or "",
        "conditions": nested.get("conditions") or [],
        "evidence": {
            key: evidence.get(key)
            for key in (
                "study_type",
                "methodology",
                "p_value",
                "effect_size",
                "effect_metric",
                "sample_size",
                "replicability",
                "direction",
            )
        },
        "source_paper": {
            key: paper.get(key)
            for key in (
                "pmid",
                "doi",
                "pmcid",
                "arxiv_id",
                "openalex_id",
                "title",
                "authors",
                "year",
                "journal",
            )
            if paper.get(key) not in (None, "")
        },
        "old_scope_reaudit": claim.get("scope_reaudit") or {},
    }
    return payload


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            paper_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            current_claim_ids_json TEXT NOT NULL,
            current_paper_ids_json TEXT NOT NULL,
            graph_ordinal INTEGER NOT NULL UNIQUE,
            extracted_row_count INTEGER NOT NULL DEFAULT 0,
            extracted_routing_conflict INTEGER NOT NULL DEFAULT 0,
            abstract_available INTEGER NOT NULL DEFAULT 0,
            review_status TEXT NOT NULL DEFAULT 'pending',
            review_json TEXT,
            reviewed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_claims_paper ON claims(paper_key);
        CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(review_status, graph_ordinal);
        CREATE TABLE IF NOT EXISTS papers (
            paper_key TEXT PRIMARY KEY,
            aliases_json TEXT NOT NULL,
            source_paper_json TEXT NOT NULL,
            claim_count INTEGER NOT NULL DEFAULT 0,
            abstract TEXT,
            abstract_source TEXT
        );
        CREATE TABLE IF NOT EXISTS extracted_orphans (
            row_number INTEGER PRIMARY KEY,
            claim_id TEXT NOT NULL,
            row_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS extracted_invalid_rows (
            row_number INTEGER PRIMARY KEY,
            raw_line TEXT NOT NULL,
            error TEXT NOT NULL
        );
        """
    )


def set_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value_json) VALUES (?, ?)",
        (key, compact_json(value)),
    )


def get_meta(connection: sqlite3.Connection, key: str) -> object | None:
    row = connection.execute("SELECT value_json FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def trusted_embedded_review(
    *, claim_id: str, paper_key: str, payload: dict[str, Any], labels: list[str]
) -> dict[str, Any] | None:
    """Recover an immutable prior decision only when its content contract matches."""
    audit = payload.get("old_scope_reaudit") or {}
    if not isinstance(audit, dict):
        return None
    if (
        audit.get("audit_name") != AUDIT_NAME
        or str(audit.get("audit_version") or "") != AUDIT_VERSION
        or audit.get("rubric_version") != RUBRIC_VERSION
        or audit.get("decision") != "finalized"
    ):
        return None
    canonical_labels = [value for value in CASE_STUDY_IDS if value in set(labels)]
    if labels != canonical_labels:
        return None
    gates = audit.get("gates") or {}
    if not isinstance(gates, dict):
        return None
    expected = claim_contract_fields(
        paper_key=paper_key,
        payload=payload,
        labels=labels,
        gates=gates,
        rubric_version=RUBRIC_VERSION,
        case_study_ids=CASE_STUDY_IDS,
    )
    if not embedded_contract_matches(audit, expected):
        return None
    return {
        "claim_id": claim_id,
        "claim_case_study_ids": labels,
        "confidence": float(audit.get("confidence") or 1.0),
        "reason": str(audit.get("decision_basis") or "Immutable embedded audit reuse."),
        "needs_secondary_review": False,
        "gates": {str(key): bool(value) for key, value in gates.items()},
        "rubric_version": RUBRIC_VERSION,
        "review_stage": "embedded_contract_reuse",
        "reviewer_id": "deterministic:embedded_audit_contract",
        "reasoning_effort": "content_addressed_reuse",
        "reviewed_at": str(audit.get("reviewed_at") or utc_now()),
        **expected,
    }


def prepare_graph(connection: sqlite3.Connection, graph_path: Path) -> dict[str, Any]:
    connection.execute("DELETE FROM claims")
    connection.execute("DELETE FROM papers")
    claim_count = 0
    current_labels: Counter[str] = Counter()
    old_audits: Counter[str] = Counter()
    old_mojibake_suspects = 0
    identity_fallbacks = 0
    paper_union = PaperIdentityUnion()
    trusted_embedded_reuses = 0

    for node_id, node in iter_concepts(graph_path):
        if not is_claim_node(node_id, node):
            continue
        claim = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        claim.setdefault("id", node_id)
        payload = claim_payload(node_id, node)
        aliases = paper_aliases(claim)
        paper_key = strongest_paper_key(claim)
        if paper_key.startswith("claim_fallback:"):
            identity_fallbacks += 1
            aliases = [paper_key]
        paper_union.add(aliases)
        claim_ids = claim_case_study_ids_from_dict(claim)
        paper_ids = paper_case_study_ids_from_dict(claim)
        for case_study_id in claim_ids:
            current_labels[case_study_id] += 1
        old_audit = payload.get("old_scope_reaudit") or {}
        reviewer = str(old_audit.get("reviewer_id") or "missing")
        old_audits[reviewer] += 1
        reason = str(old_audit.get("decision_basis") or "")
        if reason and ("�" in reason or re.search(r"[\u0400-\u04ff]{3,}", reason)):
            old_mojibake_suspects += 1

        connection.execute(
            """
            INSERT INTO claims(
                claim_id, paper_key, payload_json, current_claim_ids_json,
                current_paper_ids_json, graph_ordinal
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                paper_key,
                compact_json(payload),
                compact_json(claim_ids),
                compact_json(paper_ids),
                claim_count,
            ),
        )
        claim_count += 1
        if claim_count % 10_000 == 0:
            connection.commit()
    connection.commit()

    # Claims encountered before a bridging identifier can carry an earlier root
    # (for example DOI before a later PMID+DOI row).  Canonicalize every stored
    # key after the full alias union is known, then rebuild the paper table.
    updates: list[tuple[str, str]] = []
    paper_counts: Counter[str] = Counter()
    paper_payloads: dict[str, dict[str, Any]] = {}
    for claim_id, paper_key, payload_json in connection.execute(
        "SELECT claim_id, paper_key, payload_json FROM claims ORDER BY graph_ordinal"
    ):
        canonical_key = paper_union.find(paper_key)
        updates.append((canonical_key, claim_id))
        paper_counts[canonical_key] += 1
        paper_payloads[canonical_key] = json.loads(payload_json).get("source_paper") or {}
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE claims SET paper_key=? WHERE claim_id=?", updates
            )
            connection.commit()
            updates.clear()
    if updates:
        connection.executemany("UPDATE claims SET paper_key=? WHERE claim_id=?", updates)
        connection.commit()

    connection.execute("DELETE FROM papers")
    alias_groups = paper_union.groups()
    connection.executemany(
        """
        INSERT INTO papers(
            paper_key, aliases_json, source_paper_json, claim_count
        ) VALUES (?, ?, ?, ?)
        """,
        (
            (
                paper_key,
                compact_json(alias_groups.get(paper_key, [paper_key])),
                compact_json(paper_payloads.get(paper_key, {})),
                count,
            )
            for paper_key, count in paper_counts.items()
        ),
    )
    connection.commit()

    # A fresh ledger may reuse a prior formal decision only when the embedded
    # content-addressed contract proves that evidence, rubric, registry, and
    # decision are unchanged. Model/version changes alone never invalidate it.
    for claim_id, paper_key, payload_json, labels_json in connection.execute(
        """
        SELECT claim_id, paper_key, payload_json, current_claim_ids_json
        FROM claims ORDER BY graph_ordinal
        """
    ):
        payload = json.loads(payload_json)
        labels = json.loads(labels_json)
        review = trusted_embedded_review(
            claim_id=str(claim_id),
            paper_key=str(paper_key),
            payload=payload,
            labels=labels,
        )
        if review is None:
            continue
        connection.execute(
            """
            UPDATE claims SET review_status='final_complete', review_json=?, reviewed_at=?
            WHERE claim_id=? AND review_status='pending'
            """,
            (compact_json(review), review["reviewed_at"], claim_id),
        )
        trusted_embedded_reuses += 1
    connection.commit()
    return {
        "graph_claims": claim_count,
        "graph_papers_by_identity_union": connection.execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0],
        "identity_fallback_claims": identity_fallbacks,
        "current_claim_label_counts": {
            case_study_id: current_labels[case_study_id]
            for case_study_id in CASE_STUDY_IDS
        },
        "old_scope_reaudit_reviewer_counts": dict(sorted(old_audits.items())),
        "old_scope_reaudit_mojibake_suspects": old_mojibake_suspects,
        "trusted_embedded_final_reviews_reused": trusted_embedded_reuses,
    }


def scan_extracted(connection: sqlite3.Connection, extracted_path: Path) -> dict[str, Any]:
    connection.execute("DELETE FROM extracted_orphans")
    connection.execute("DELETE FROM extracted_invalid_rows")
    rows = 0
    invalid = 0
    orphans = 0
    matched = 0
    duplicates = 0
    routing_conflicts = 0
    seen_ids: Counter[str] = Counter()
    with extracted_path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                connection.execute(
                    "INSERT INTO extracted_invalid_rows VALUES (?, ?, ?)",
                    (row_number, line.rstrip("\n"), str(exc)),
                )
                continue
            claim_id = str(row.get("id") or "")
            seen_ids[claim_id] += 1
            graph_row = connection.execute(
                "SELECT current_claim_ids_json, current_paper_ids_json FROM claims WHERE claim_id=?",
                (claim_id,),
            ).fetchone()
            if graph_row is None:
                orphans += 1
                connection.execute(
                    "INSERT INTO extracted_orphans VALUES (?, ?, ?)",
                    (row_number, claim_id, compact_json(row)),
                )
                continue
            matched += 1
            extracted_claim_ids = claim_case_study_ids_from_dict(row)
            extracted_paper_ids = paper_case_study_ids_from_dict(row)
            conflict = int(
                extracted_claim_ids != json.loads(graph_row[0])
                or extracted_paper_ids != json.loads(graph_row[1])
            )
            routing_conflicts += conflict
            connection.execute(
                """
                UPDATE claims SET
                    extracted_row_count = extracted_row_count + 1,
                    extracted_routing_conflict = MAX(extracted_routing_conflict, ?)
                WHERE claim_id=?
                """,
                (conflict, claim_id),
            )
            if rows % 10_000 == 0:
                connection.commit()
    connection.commit()
    duplicates = sum(count - 1 for claim_id, count in seen_ids.items() if claim_id and count > 1)
    missing = connection.execute(
        "SELECT COUNT(*) FROM claims WHERE extracted_row_count=0"
    ).fetchone()[0]
    duplicate_graph_claims = connection.execute(
        "SELECT COUNT(*) FROM claims WHERE extracted_row_count>1"
    ).fetchone()[0]
    return {
        "rows": rows,
        "invalid_json_rows": invalid,
        "matched_rows": matched,
        "orphan_rows_not_in_graph": orphans,
        "duplicate_rows_beyond_first": duplicates,
        "graph_claims_with_duplicate_extracted_rows": duplicate_graph_claims,
        "graph_claims_missing_from_extracted": missing,
        "matched_rows_with_routing_conflict": routing_conflicts,
    }


def attach_abstracts(
    connection: sqlite3.Connection,
    cache_paths: Iterable[Path],
) -> dict[str, Any]:
    pmid_to_paper: dict[str, str] = {}
    for paper_key, aliases_json in connection.execute(
        "SELECT paper_key, aliases_json FROM papers"
    ):
        for alias in json.loads(aliases_json):
            if alias.startswith("pmid:"):
                pmid_to_paper[alias[5:]] = paper_key
                break
    records = 0
    matched_records = 0
    matched_papers: set[str] = set()
    used_paths: list[str] = []
    for cache_path in cache_paths:
        if not cache_path.is_file():
            continue
        used_paths.append(str(cache_path.resolve()))
        with cache_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pmid = normalize_identifier(row.get("pmid"))
                abstract = str(row.get("abstract") or "").strip()
                paper_key = pmid_to_paper.get(pmid)
                if not paper_key or not abstract:
                    continue
                matched_records += 1
                matched_papers.add(paper_key)
                connection.execute(
                    "UPDATE papers SET abstract=?, abstract_source=? WHERE paper_key=?",
                    (abstract, str(cache_path.resolve()), paper_key),
                )
        connection.commit()
    connection.execute(
        """
        UPDATE claims SET abstract_available = CASE WHEN EXISTS(
            SELECT 1 FROM papers p
            WHERE p.paper_key=claims.paper_key AND p.abstract IS NOT NULL
        ) THEN 1 ELSE 0 END
        """
    )
    connection.commit()
    return {
        "cache_paths_used": used_paths,
        "cache_records_scanned": records,
        "matching_cache_records": matched_records,
        "papers_with_abstract": len(matched_papers),
        "claims_with_abstract": connection.execute(
            "SELECT COUNT(*) FROM claims WHERE abstract_available=1"
        ).fetchone()[0],
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "reaudit.sqlite"
    manifest_path = output_dir / "manifest.json"
    graph_path = args.graph.resolve()
    extracted_path = args.extracted.resolve()
    requested_caches = list(getattr(args, "abstract_cache", []) or [])
    cache_paths = [
        path.resolve()
        for path in (requested_caches or [DEFAULT_ABSTRACT_CACHE])
    ]

    connection = connect(ledger_path)
    try:
        create_schema(connection)
        existing_final = connection.execute(
            "SELECT COUNT(*) FROM claims WHERE review_status='final_complete'"
        ).fetchone()[0]
        if args.rebuild and existing_final and not getattr(args, "discard_reviews", False):
            raise RuntimeError(
                f"refusing to discard {existing_final} finalized reviews; use a new "
                "output directory, or pass --discard-reviews only for an intentional "
                "new audit epoch"
            )
        sources = {
            "graph": file_fingerprint(graph_path),
            "extracted": file_fingerprint(extracted_path),
            "abstract_caches": [
                file_fingerprint(path) for path in cache_paths if path.is_file()
            ],
        }
        previous_sources = get_meta(connection, "sources")
        if previous_sources and previous_sources != sources and not args.rebuild:
            raise RuntimeError(
                "source files changed since ledger preparation; rerun with --rebuild "
                "after verifying the canonical files"
            )
        if args.rebuild or get_meta(connection, "prepared") is not True:
            graph_report = prepare_graph(connection, graph_path)
            extracted_report = scan_extracted(connection, extracted_path)
            abstract_report = attach_abstracts(connection, cache_paths)
            set_meta(connection, "sources", sources)
            set_meta(connection, "prepared", True)
            set_meta(connection, "schema_version", SCHEMA_VERSION)
            set_meta(connection, "rubric_version", RUBRIC_VERSION)
            set_meta(
                connection,
                "audit_contract",
                contract_context(
                    rubric_version=RUBRIC_VERSION,
                    case_study_ids=CASE_STUDY_IDS,
                ),
            )
            connection.commit()
        else:
            graph_report = get_meta(connection, "graph_report") or {}
            extracted_report = get_meta(connection, "extracted_report") or {}
            abstract_report = get_meta(connection, "abstract_report") or {}

        claim_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        paper_count = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        status_counts = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "created_at": utc_now(),
            "sources": sources,
            "ledger": str(ledger_path.resolve()),
            "inventory": {
                "claims": claim_count,
                "papers_by_strongest_identity": paper_count,
                "batch_size": args.batch_size,
                "total_batches": math.ceil(claim_count / args.batch_size),
                "review_status_counts": status_counts,
            },
            "graph_profile": graph_report,
            "extracted_store_profile": extracted_report,
            "abstract_profile": abstract_report,
            "mutation_policy": (
                "No full_v2 mutation is permitted until every graph claim has one "
                "validated semantic review and the apply command passes dry-run."
            ),
        }
        set_meta(connection, "graph_report", graph_report)
        set_meta(connection, "extracted_report", extracted_report)
        set_meta(connection, "abstract_report", abstract_report)
        connection.commit()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        connection.close()


def report(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = args.output_dir.resolve() / "reaudit.sqlite"
    connection = connect(ledger_path)
    try:
        total = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        status_counts = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        primary_reviewed = (
            status_counts.get("final_complete", 0)
            + status_counts.get("secondary_pending", 0)
        )
        finalized = status_counts.get("final_complete", 0)
        pending_primary = total - primary_reviewed
        pending_final = total - finalized
        return {
            "schema_version": get_meta(connection, "schema_version"),
            "rubric_version": get_meta(connection, "rubric_version"),
            "claims": total,
            "primary_reviewed": primary_reviewed,
            "pending_primary_review": pending_primary,
            "secondary_review_pending": status_counts.get("secondary_pending", 0),
            "finalized": finalized,
            "pending_final": pending_final,
            "primary_completion_percent": round(primary_reviewed * 100 / total, 6)
            if total
            else 100.0,
            "final_completion_percent": round(finalized * 100 / total, 6)
            if total
            else 100.0,
            "review_status_counts": status_counts,
            "papers": connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "claims_with_abstract": connection.execute(
                "SELECT COUNT(*) FROM claims WHERE abstract_available=1"
            ).fetchone()[0],
            "extracted_orphan_rows": connection.execute(
                "SELECT COUNT(*) FROM extracted_orphans"
            ).fetchone()[0],
            "extracted_invalid_rows": connection.execute(
                "SELECT COUNT(*) FROM extracted_invalid_rows"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    prepare_parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    prepare_parser.add_argument(
        "--abstract-cache", type=Path, action="append", default=[]
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--batch-size", type=int, default=100)
    prepare_parser.add_argument("--rebuild", action="store_true")
    prepare_parser.add_argument(
        "--discard-reviews",
        action="store_true",
        help="allow --rebuild to discard finalized ledger decisions for a deliberate new epoch",
    )

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "report":
        result = report(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
