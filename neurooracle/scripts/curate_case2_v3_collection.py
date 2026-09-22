#!/usr/bin/env python3
"""Curate the Case 2 v3 extraction queue without touching the formal KG.

The abstract cache remains append-only and is the recovery source.  This script
only rewrites the canonical ``collection_metadata.csv`` queue and emits a full
exclusion audit.  It removes journal-replaced preprints, two verified duplicate
publication records, and known multi-abstract proceedings compilations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING_DIR = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "case_study_staging"
    / "kg_v3_case2_pathway_mediation_20260810"
)

COMPOSITE_RECORD_IDS = {
    "PMCID:PMC4896262",
    "PMCID:PMC6152593",
    "PMCID:PMC3997469",
    "PMCID:PMC6152592",
    "PMCID:PMC4173965",
    "PMCID:PMC4513500",
    "PMCID:PMC3671359",
    "PMCID:PMC3433979",
}
VERIFIED_DUAL_PUBLICATION_GROUPS = (
    frozenset({"34782355", "34782351"}),
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value: object) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", html.unescape(str(value or "")).lower())
    )


def _tokens(value: object) -> set[str]:
    return set(_normalize(value).split())


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("pmid") or "").strip()


def _is_preprint(record: dict[str, Any]) -> bool:
    doi = str(record.get("doi") or "").lower()
    journal = str(record.get("journal") or "").lower()
    source_id = _record_id(record).lower()
    return (
        doi.startswith("10.1101/")
        or doi.startswith("10.21203/")
        or source_id.startswith("epmc:10.1101/")
        or any(
            marker in journal
            for marker in ("medrxiv", "biorxiv", "arxiv", "preprint", "research square")
        )
    )


def _year(record: dict[str, Any]) -> int:
    try:
        return int(record.get("year") or 0)
    except (TypeError, ValueError):
        return 0


def _canonical_rank(record: dict[str, Any]) -> tuple:
    source_id = _record_id(record)
    return (
        0 if _is_preprint(record) else 1,
        1 if record.get("doi") else 0,
        1 if source_id.isdigit() else 0,
        _year(record),
        tuple(-ord(char) for char in source_id),
    )


def _load_cache(path: Path) -> tuple[dict[str, dict], int]:
    latest: dict[str, dict] = {}
    physical_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            physical_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid cache JSON at line {line_number}: {exc}") from exc
            paper = record.get("paper") or {}
            source_id = str(record.get("pmid") or paper.get("pmid") or "").strip()
            if source_id:
                latest[source_id] = record
    return latest, physical_rows


def _abstract(cache: dict[str, dict], source_id: str) -> str:
    return str((cache.get(source_id) or {}).get("abstract") or "")


def _select_exclusions(
    rows: list[dict[str, Any]],
    cache: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    by_id = {_record_id(row): row for row in rows}
    exclusions: dict[str, dict[str, Any]] = {}

    for source_id in sorted(COMPOSITE_RECORD_IDS):
        # Missing here means a prior successful run already removed it.  The
        # persisted exclusion audit remains the recovery record.
        if source_id not in by_id:
            continue
        exclusions[source_id] = {
            "reason": "multi_abstract_proceedings_compilation",
            "duplicate_of": None,
        }

    title_tokens = [_tokens(row.get("title")) for row in rows]
    journal_token_index: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if _is_preprint(row):
            continue
        for token in title_tokens[index]:
            journal_token_index[token].append(index)

    for preprint_index, preprint in enumerate(rows):
        if not _is_preprint(preprint) or len(title_tokens[preprint_index]) < 5:
            continue
        candidate_overlap: Counter[int] = Counter()
        for token in title_tokens[preprint_index]:
            candidate_overlap.update(journal_token_index[token])
        matches: list[tuple[float, dict[str, Any]]] = []
        for journal_index, intersection in candidate_overlap.items():
            journal = rows[journal_index]
            if abs(_year(preprint) - _year(journal)) > 3:
                continue
            union_size = len(title_tokens[preprint_index] | title_tokens[journal_index])
            similarity = intersection / union_size if union_size else 0.0
            if similarity >= 0.90:
                matches.append((similarity, journal))
        if matches:
            _, canonical = max(
                matches,
                key=lambda item: (item[0], _canonical_rank(item[1])),
            )
            exclusions.setdefault(
                _record_id(preprint),
                {
                    "reason": "journal_replaced_preprint_title_match",
                    "duplicate_of": _record_id(canonical),
                },
            )

    # Catch renamed journal articles whose complete normalized abstract is
    # unchanged from the preprint (for example the FAP network paper).
    abstract_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized_abstract = _normalize(_abstract(cache, _record_id(row)))
        if len(normalized_abstract) >= 200:
            abstract_index[_sha256_text(normalized_abstract)].append(row)
    for group in abstract_index.values():
        preprints = [row for row in group if _is_preprint(row)]
        journals = [row for row in group if not _is_preprint(row)]
        for preprint in preprints:
            first_author = _normalize(str(preprint.get("authors") or "").split(",")[0])
            author_matches = [
                row
                for row in journals
                if _normalize(str(row.get("authors") or "").split(",")[0])
                == first_author
            ]
            if not author_matches:
                continue
            canonical = max(author_matches, key=_canonical_rank)
            exclusions.setdefault(
                _record_id(preprint),
                {
                    "reason": "journal_replaced_preprint_exact_abstract",
                    "duplicate_of": _record_id(canonical),
                },
            )

    # A duplicated OpenAlex thesis/work record with identical title, author,
    # year, venue, and abstract.  This general grouping is deliberately limited
    # to OpenAlex IDs so versioned journal reviews are preserved.
    openalex_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_id = _record_id(row)
        if not source_id.startswith("OA:"):
            continue
        openalex_groups[
            (
                _normalize(row.get("title")),
                _normalize(row.get("authors")),
                _year(row),
                _normalize(row.get("journal")),
                _sha256_text(_normalize(_abstract(cache, source_id))),
            )
        ].append(row)
    for group in openalex_groups.values():
        if len(group) < 2:
            continue
        canonical = max(group, key=_canonical_rank)
        for duplicate in group:
            if duplicate is canonical:
                continue
            exclusions.setdefault(
                _record_id(duplicate),
                {
                    "reason": "duplicate_openalex_work_record",
                    "duplicate_of": _record_id(canonical),
                },
            )

    # One verified dual-publication record contains exactly the same title,
    # author list, and abstract in two journal venues.  Other update/version
    # groups (notably Cochrane updates) are intentionally retained.
    for source_ids in VERIFIED_DUAL_PUBLICATION_GROUPS:
        group = [by_id[source_id] for source_id in source_ids if source_id in by_id]
        if len(group) < 2:
            continue
        canonical = max(group, key=_canonical_rank)
        for duplicate in group:
            if duplicate is canonical:
                continue
            exclusions.setdefault(
                _record_id(duplicate),
                {
                    "reason": "verified_duplicate_publication_record",
                    "duplicate_of": _record_id(canonical),
                },
            )

    return exclusions


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temp_path = path.with_suffix(path.suffix + ".curation.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _write_json_atomic(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".curation.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _update_existing_json(path: Path, curation: dict[str, Any]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    payload["post_collection_curation"] = curation
    payload["canonical_extraction_queue_count"] = curation["papers_after"]
    _write_json_atomic(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--apply", action="store_true", help="rewrite canonical queue")
    args = parser.parse_args()

    staging_dir = args.staging_dir.resolve()
    canonical_root = DEFAULT_STAGING_DIR.resolve().parent
    if staging_dir.parent != canonical_root:
        raise ValueError(f"refusing curation outside expected staging root: {staging_dir}")
    csv_path = staging_dir / "collection_metadata.csv"
    cache_path = staging_dir / "abstract_cache.jsonl"
    if not csv_path.exists() or not cache_path.exists():
        raise FileNotFoundError("collection_metadata.csv or abstract_cache.jsonl is missing")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len({_record_id(row) for row in rows}) != len(rows):
        raise ValueError("canonical queue already contains duplicate source IDs")

    cache, physical_cache_rows = _load_cache(cache_path)
    exclusions = _select_exclusions(rows, cache)
    kept = [row for row in rows if _record_id(row) not in exclusions]
    removed = [row for row in rows if _record_id(row) in exclusions]
    if len(rows) != len(kept) + len(removed):
        raise RuntimeError("curation accounting mismatch")

    reason_counts = Counter(exclusions[source_id]["reason"] for source_id in exclusions)
    abstract_lengths = [len(_abstract(cache, _record_id(row))) for row in kept]
    missing_abstracts = sum(length == 0 for length in abstract_lengths)
    duplicate_targets = {
        details["duplicate_of"]
        for details in exclusions.values()
        if details.get("duplicate_of")
    }
    unresolved_targets = sorted(duplicate_targets - {_record_id(row) for row in kept})
    if unresolved_targets:
        raise ValueError(f"duplicate canonical targets were removed: {unresolved_targets}")

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "case2_post_collection_curation.v1",
        "generated_at": generated_at,
        "dry_run": not args.apply,
        "formal_kg_mutation": False,
        "staging_dir": str(staging_dir),
        "papers_before": len(rows),
        "papers_removed": len(removed),
        "papers_after": len(kept),
        "duplicate_records_removed": sum(
            count
            for reason, count in reason_counts.items()
            if reason != "multi_abstract_proceedings_compilation"
        ),
        "multi_abstract_compilations_removed": reason_counts.get(
            "multi_abstract_proceedings_compilation", 0
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "abstract_cache": {
            "physical_rows_unchanged": physical_cache_rows,
            "unique_ids_last_write_wins": len(cache),
            "cache_rewritten": False,
            "note": "Append-only cache retained for recovery; collection_metadata.csv is the canonical extraction queue.",
        },
        "curated_queue": {
            "missing_abstracts": missing_abstracts,
            "minimum_abstract_characters": min(abstract_lengths) if abstract_lengths else 0,
            "median_abstract_characters": statistics.median(abstract_lengths) if abstract_lengths else 0,
            "maximum_abstract_characters": max(abstract_lengths) if abstract_lengths else 0,
            "by_source": dict(sorted(Counter(row.get("source") or "" for row in kept).items())),
            "by_query_family": dict(sorted(Counter(row.get("preset") or "" for row in kept).items())),
        },
    }

    audit_rows = []
    for row in removed:
        source_id = _record_id(row)
        abstract = _abstract(cache, source_id)
        audit_rows.append(
            {
                "schema_version": "case2_collection_curation_exclusion.v1",
                "excluded_at": generated_at,
                "source_id": source_id,
                **exclusions[source_id],
                "metadata": row,
                "abstract_length": len(abstract),
                "abstract_sha256": _sha256_text(abstract),
                "recoverable_from_abstract_cache": source_id in cache,
            }
        )

    print(_canonical_json(report))
    if not args.apply:
        return 0

    _write_csv_atomic(csv_path, kept, fieldnames)
    audit_path = staging_dir / "post_collection_curation_exclusions.jsonl"
    audit_temp = audit_path.with_suffix(audit_path.suffix + ".curation.tmp")
    with audit_temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in audit_rows:
            handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(audit_temp, audit_path)

    report["dry_run"] = False
    report["audit_path"] = str(audit_path)
    report_path = staging_dir / "POST_COLLECTION_CURATION_REPORT.json"
    _write_json_atomic(report_path, report)
    curation_summary = {
        key: value
        for key, value in report.items()
        if key
        in {
            "schema_version",
            "generated_at",
            "formal_kg_mutation",
            "papers_before",
            "papers_removed",
            "papers_after",
            "duplicate_records_removed",
            "multi_abstract_compilations_removed",
            "reason_counts",
            "audit_path",
        }
    }
    _update_existing_json(staging_dir / "collection_manifest.json", curation_summary)
    _update_existing_json(staging_dir / "CAMPAIGN_SUMMARY.json", curation_summary)

    qa_path = staging_dir / "QA_REPORT.json"
    if qa_path.exists():
        prior_qa = json.loads(qa_path.read_text(encoding="utf-8-sig"))
        qa = {
            "schema_version": "kg_v3_case2_collection_qa.v3",
            "generated_at": generated_at,
            "case_study_id": "case2_pathway_mediation",
            "status": "curated_abstract_ready_no_scope_assignment",
            "year_policy": prior_qa.get("year_policy"),
            "formal_case2_baseline": prior_qa.get("formal_case2_baseline"),
            "collection": {
                "original_target_unique_candidates": 20000,
                "pre_curation_candidates": len(rows),
                "canonical_extraction_queue": len(kept),
                "abstract_ready": len(kept) - missing_abstracts,
                "abstract_coverage": (
                    (len(kept) - missing_abstracts) / len(kept) if kept else 0.0
                ),
                "post_curation_duplicate_records": 0,
                "post_curation_multi_abstract_compilations": 0,
                **report["curated_queue"],
            },
            "post_collection_curation": curation_summary,
            "scope_assignment": {
                "performed": False,
                "note": "The curated queue must undergo full-abstract extraction, deterministic Case 2 component routing, and a valid current scope seal before formal KG ingestion.",
            },
            "claim_extraction": False,
            "kg_injection": False,
            "formal_kg_mutation": {"performed": False},
        }
        _write_json_atomic(qa_path, qa)

    print(f"curated queue written: {len(rows)} -> {len(kept)}; audit={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
