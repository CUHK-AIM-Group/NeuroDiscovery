"""Read-mostly academic literature service for the NeuroClaw MCP server.

The service deliberately exposes no formal KG mutation operation.  Its only
write operation starts a collection under the controlled academic MCP staging
root; the collector itself hashes the formal KG before and after the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

from neurooracle.src import academic_literature as literature
from neurooracle.src.case_studies import CASE_STUDIES
from neurooracle.src.paper_identity import (
    GlobalPaperIdentityIndex,
    paper_identity_aliases,
)


COLLECTION_MARKERS = (
    "manifest.json",
    "COLLECTION_COMPLETE.json",
    "collection_state.json",
    "abstracts_ready_for_extraction.jsonl",
)
CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


def _case_study_catalog(
    coverage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, object], ...]:
    """Merge the installed registry with formal-KG taxonomy entries.

    The MCP can be deployed alongside an older runtime while reading a newer
    formal KG snapshot.  Coverage-only IDs therefore remain available without
    making the server depend on an in-progress registry migration.
    """

    rows: dict[str, dict[str, object]] = {}
    for case in CASE_STUDIES:
        if case.name == "case3_hindcasting":
            continue
        rows[case.name] = {
            "number": len(rows) + 1,
            "id": case.name,
            "name": case.english_name,
            "chinese_name": case.chinese_name,
            "english_name": case.english_name,
        }
    for case_id in sorted(str(value).strip() for value in (coverage or {}) if value):
        if case_id == "case3_hindcasting" or case_id in rows:
            continue
        display_name = case_id.replace("_", " ").title()
        rows[case_id] = {
            "number": len(rows) + 1,
            "id": case_id,
            "name": display_name,
            "chinese_name": "",
            "english_name": display_name,
        }
    return tuple(rows.values())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, row


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                process_query_limited_information, False, int(pid)
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    return True


class AcademicService:
    """Backend shared by MCP tools and deterministic tests."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = (
            Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        ).resolve()
        self.data_root = self.repo_root / "neurooracle" / "data"
        self.formal_root = self.data_root / "full_v2"
        self.formal_claims = self.formal_root / "extracted_claims.jsonl"
        self.formal_files = (
            self.formal_root / "knowledge_graph.json",
            self.formal_claims,
            self.formal_root / "CURRENT_STATE.json",
        )
        self.staging_root = self.data_root / "case_study_staging"
        self.phase2_staging_root = self.data_root / "phase2_staging"
        self.mcp_staging_root = self.staging_root / "academic_mcp"
        self.identity_index_path = (
            self.data_root / "build_artifacts" / "paper_identity_dedup_v3.sqlite3"
        )
        self.collector_script = (
            self.repo_root
            / "neurooracle"
            / "scripts"
            / "collect_sparse_case_study_literature_v3.py"
        )

    @property
    def active_staging_roots(self) -> tuple[Path, ...]:
        return (self.staging_root, self.phase2_staging_root)

    def formal_snapshot(self, *, include_hash: bool = False) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for path in self.formal_files:
            if not path.is_file():
                files[str(path.resolve())] = {"exists": False}
                continue
            stat = path.stat()
            payload: dict[str, Any] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if include_hash:
                payload["sha256"] = _hash_file(path)
            files[str(path.resolve())] = payload
        return {"captured_at": _utc_now(), "files": files}

    def list_case_studies(self) -> dict[str, Any]:
        state = _read_json(self.formal_root / "CURRENT_STATE.json")
        statistics = state.get("formal_kg_statistics") or {}
        coverage = statistics.get("case_studies") or {}
        cases = []
        for item in _case_study_catalog(coverage):
            row = dict(item)
            row["coverage"] = dict(coverage.get(str(item["id"])) or {})
            cases.append(row)
        return {
            "ok": True,
            "taxonomy_version": state.get("taxonomy_version"),
            "case_study_count": len(cases),
            "hindcasting_is_case_study": False,
            "case_studies": cases,
            "membership_policy": literature.ASSIGNMENT_POLICY,
        }

    def get_kg_coverage(self, case_study_id: str = "") -> dict[str, Any]:
        state_path = self.formal_root / "CURRENT_STATE.json"
        state = _read_json(state_path)
        statistics = state.get("formal_kg_statistics") or {}
        cases = statistics.get("case_studies") or {}
        valid_ids = {str(item["id"]) for item in _case_study_catalog(cases)}
        requested = str(case_study_id or "").strip()
        if requested and requested not in valid_ids:
            raise ValueError(f"unknown Case Study ID: {requested}")
        selected = {requested: cases.get(requested, {})} if requested else cases
        return {
            "ok": True,
            "source": str(state_path.resolve()),
            "generated_at": state.get("generated_at"),
            "taxonomy_version": state.get("taxonomy_version"),
            "general": statistics.get("general") or {},
            "case_studies": selected,
            "quality": statistics.get("quality") or {},
            "counting_policy": statistics.get("counting_policy"),
        }

    def _synchronised_index(self) -> tuple[GlobalPaperIdentityIndex, dict[str, Any]]:
        index = GlobalPaperIdentityIndex(self.identity_index_path)
        try:
            stats = index.sync(
                formal_claim_store=self.formal_claims,
                staging_roots=self.active_staging_roots,
            )
        except Exception:
            index.close()
            raise
        return index, stats

    def check_paper_identity(
        self,
        *,
        paper_id: str = "",
        pmid: str = "",
        doi: str = "",
        pmcid: str = "",
        arxiv_id: str = "",
        openalex_id: str = "",
        title: str = "",
        year: int | None = None,
    ) -> dict[str, Any]:
        record = {
            "paper_id": str(paper_id or "").strip(),
            "pmid": str(pmid or "").strip(),
            "doi": str(doi or "").strip(),
            "pmcid": str(pmcid or "").strip(),
            "arxiv_id": str(arxiv_id or "").strip(),
            "openalex_id": str(openalex_id or "").strip(),
            "title": str(title or "").strip(),
            "year": year,
        }
        aliases = paper_identity_aliases(record)
        if not aliases:
            raise ValueError(
                "provide a stable identifier or both a normalized title and publication year"
            )
        index, stats = self._synchronised_index()
        try:
            match = index.match(record)
        finally:
            index.close()
        return {
            "ok": True,
            "duplicate": match is not None,
            "candidate_aliases": aliases,
            "match": (
                {
                    "alias": match.alias,
                    "origin": match.origin,
                    "source_path": match.source_path,
                }
                if match is not None
                else None
            ),
            "index_sync": stats,
            "deduplication_scope": "formal KG plus all active staging corpora",
        }

    def _validate_case_ids(self, case_study_ids: Iterable[str]) -> list[str]:
        state = _read_json(self.formal_root / "CURRENT_STATE.json")
        statistics = state.get("formal_kg_statistics") or {}
        coverage = statistics.get("case_studies") or {}
        valid = {str(item["id"]) for item in _case_study_catalog(coverage)}
        values = sorted({str(value).strip() for value in case_study_ids if str(value).strip()})
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"unknown Case Study IDs: {', '.join(unknown)}")
        return values

    def search_literature(
        self,
        *,
        query: str,
        case_study_ids: Iterable[str] = (),
        cursor: str = "*",
        page_size: int = 25,
        max_results: int = 10,
    ) -> dict[str, Any]:
        expression = str(query or "").strip()
        if not expression:
            raise ValueError("query cannot be empty")
        if len(expression) > 8_000:
            raise ValueError("query cannot exceed 8,000 characters")
        if not 1 <= int(page_size) <= 100:
            raise ValueError("page_size must be in [1, 100]")
        if not 1 <= int(max_results) <= 25:
            raise ValueError("max_results must be in [1, 25]")
        target_ids = self._validate_case_ids(case_study_ids)
        query_id = "mcp_" + hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]
        payload = literature.fetch_search_page(
            expression,
            cursor=str(cursor or "*"),
            page_size=int(page_size),
            result_type="lite",
        )
        raw_results = ((payload.get("resultList") or {}).get("result")) or []
        if not isinstance(raw_results, list):
            raw_results = []

        invalid = Counter()
        excluded_by_origin = Counter()
        excluded_samples: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        abstracts_requested = 0
        seen_aliases: set[str] = set()
        index, sync_stats = self._synchronised_index()
        try:
            for raw in raw_results:
                if len(records) >= int(max_results):
                    break
                if not isinstance(raw, dict):
                    invalid["non_object_result"] += 1
                    continue
                candidate, status = literature.normalize_search_candidate(
                    raw,
                    search_query_ids=(query_id,),
                    search_target_case_study_ids=target_ids,
                )
                if candidate is None:
                    invalid[status] += 1
                    continue
                aliases = paper_identity_aliases(candidate)
                if any(alias in seen_aliases for alias in aliases):
                    invalid["duplicate_within_response"] += 1
                    continue
                match = index.match(candidate)
                if match is not None:
                    excluded_by_origin[match.origin] += 1
                    if len(excluded_samples) < 5:
                        excluded_samples.append(
                            {
                                "paper_id": candidate.get("paper_id"),
                                "title": candidate.get("title"),
                                "matched_alias": match.alias,
                                "matched_origin": match.origin,
                            }
                        )
                    continue
                abstracts_requested += 1
                try:
                    core = literature.fetch_article_core(
                        str(candidate["europepmc_source"]),
                        str(candidate["europepmc_id"]),
                    )
                except Exception:
                    invalid["abstract_retrieval_failed"] += 1
                    continue
                record, status = literature.normalize_core_result(
                    core,
                    search_query_ids=(query_id,),
                    search_target_case_study_ids=target_ids,
                    expected_candidate=candidate,
                )
                if record is None:
                    invalid[status] += 1
                    continue
                index.add_overlay(
                    record,
                    origin="current_mcp_response",
                    source_path="in_memory",
                )
                seen_aliases.update(paper_identity_aliases(record))
                records.append(record)
        finally:
            index.close()

        return {
            "ok": True,
            "provider": "Europe PMC official REST API",
            "years": {"start": literature.YEAR_START, "end": literature.YEAR_END},
            "query": expression,
            "query_id": query_id,
            "cursor": str(cursor or "*"),
            "next_cursor": str(payload.get("nextCursorMark") or ""),
            "provider_hit_count": int(payload.get("hitCount") or 0),
            "raw_results_scanned": len(raw_results),
            "abstracts_requested_after_dedup": abstracts_requested,
            "returned_records": len(records),
            "excluded_by_origin": dict(sorted(excluded_by_origin.items())),
            "excluded_samples": excluded_samples,
            "invalid_results": dict(sorted(invalid.items())),
            "index_sync": sync_stats,
            "search_provenance_only": True,
            "case_study_membership_assigned": False,
            "claim_labels_must_be_nonexclusive": True,
            "records": records,
        }

    def _collection_directories(self) -> list[Path]:
        directories: list[Path] = []
        if self.staging_root.is_dir():
            for path in self.staging_root.iterdir():
                if not path.is_dir():
                    continue
                if path.name == self.mcp_staging_root.name:
                    directories.extend(child for child in path.iterdir() if child.is_dir())
                elif any((path / marker).exists() for marker in COLLECTION_MARKERS):
                    directories.append(path)
        return directories

    def _collection_id(self, path: Path) -> str:
        return path.resolve().relative_to(self.staging_root.resolve()).as_posix()

    def _resolve_collection(self, collection_id: str) -> Path:
        raw = str(collection_id or "").strip().replace("\\", "/")
        parsed = PurePosixPath(raw)
        if not raw or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError("collection_id must be a safe path relative to staging")
        path = (self.staging_root / Path(*parsed.parts)).resolve()
        try:
            path.relative_to(self.staging_root.resolve())
        except ValueError as exc:
            raise ValueError("collection_id escapes the staging root") from exc
        if not path.is_dir():
            raise FileNotFoundError(f"collection not found: {raw}")
        return path

    def _collection_summary(self, path: Path) -> dict[str, Any]:
        manifest = _read_json(path / "manifest.json")
        seal = _read_json(path / "COLLECTION_COMPLETE.json")
        state = _read_json(path / "collection_state.json")
        job = _read_json(path / "academic_mcp_job.json")
        status = str(seal.get("status") or manifest.get("status") or "")
        if not status:
            status = "running" if _pid_is_running(int(job.get("pid") or 0)) else "partial"
        papers = (
            seal.get("papers")
            or manifest.get("collected_unique_papers")
            or state.get("accepted_unique_papers")
            or 0
        )
        batches = seal.get("batches") or manifest.get("total_batches") or 0
        modified_ns = max(
            (item.stat().st_mtime_ns for item in path.iterdir() if item.is_file()),
            default=path.stat().st_mtime_ns,
        )
        return {
            "collection_id": self._collection_id(path),
            "status": status,
            "papers": int(papers or 0),
            "batches": int(batches or 0),
            "target_unique_papers": int(manifest.get("target_unique_papers") or 0),
            "sealed": bool(seal),
            "kg_injection": bool(seal.get("kg_injection") or manifest.get("kg_injection")),
            "formal_kg_mutated": bool(manifest.get("formal_kg_mutated")),
            "modified_ns": modified_ns,
        }

    def list_collections(self, *, offset: int = 0, limit: int = 25) -> dict[str, Any]:
        if int(offset) < 0:
            raise ValueError("offset must be non-negative")
        if not 1 <= int(limit) <= 100:
            raise ValueError("limit must be in [1, 100]")
        rows = [self._collection_summary(path) for path in self._collection_directories()]
        rows.sort(key=lambda row: (-int(row["modified_ns"]), str(row["collection_id"])))
        selected = rows[int(offset) : int(offset) + int(limit)]
        return {
            "ok": True,
            "total": len(rows),
            "offset": int(offset),
            "limit": int(limit),
            "has_more": int(offset) + len(selected) < len(rows),
            "collections": selected,
        }

    def get_collection_status(self, collection_id: str) -> dict[str, Any]:
        path = self._resolve_collection(collection_id)
        summary = self._collection_summary(path)
        manifest = _read_json(path / "manifest.json")
        seal = _read_json(path / "COLLECTION_COMPLETE.json")
        state = _read_json(path / "collection_state.json")
        job = _read_json(path / "academic_mcp_job.json")
        targets = []
        for target in manifest.get("targets") or []:
            if not isinstance(target, Mapping):
                continue
            targets.append(
                {
                    "display_number": target.get("display_number"),
                    "case_study_id": target.get("case_study_id"),
                    "formal_baseline_papers": target.get("formal_baseline_papers"),
                    "planned_primary_quota": target.get("planned_primary_quota"),
                    "collected_primary_papers": target.get("collected_primary_papers"),
                    "search_provenance_papers_nonexclusive": target.get(
                        "search_provenance_papers_nonexclusive"
                    ),
                }
            )
        files = {}
        for name in COLLECTION_MARKERS:
            file_path = path / name
            files[name] = {
                "exists": file_path.is_file(),
                "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
            }
        return {
            "ok": True,
            **summary,
            "path": str(path),
            "years": manifest.get("years") or {
                "start": literature.YEAR_START,
                "end": literature.YEAR_END,
            },
            "targets": targets,
            "claim_extraction_performed": bool(manifest.get("claim_extraction_performed")),
            "job": {
                **job,
                "observed_running": _pid_is_running(int(job.get("pid") or 0)),
            }
            if job
            else {},
            "state": state,
            "seal": seal,
            "files": files,
        }

    def get_batch(
        self,
        *,
        collection_id: str,
        batch_number: int,
        offset: int = 0,
        limit: int = 10,
        include_abstract: bool = True,
    ) -> dict[str, Any]:
        if int(batch_number) <= 0:
            raise ValueError("batch_number must be positive")
        if int(offset) < 0:
            raise ValueError("offset must be non-negative")
        if not 1 <= int(limit) <= 25:
            raise ValueError("limit must be in [1, 25]")
        path = self._resolve_collection(collection_id)
        batch_dir = path / "extraction_batches"
        batch_files = sorted(
            batch_dir.glob("batch_*.jsonl"),
            key=lambda item: int(re.search(r"(\d+)", item.stem).group(1)),
        )
        if int(batch_number) > len(batch_files):
            raise FileNotFoundError(
                f"batch {batch_number} not found; collection has {len(batch_files)} batches"
            )
        batch_path = batch_files[int(batch_number) - 1]
        rows: list[dict[str, Any]] = []
        total = 0
        for _, row in _iter_jsonl(batch_path):
            if total >= int(offset) and len(rows) < int(limit):
                value = dict(row)
                if not include_abstract:
                    value.pop("abstract", None)
                rows.append(value)
            total += 1
        return {
            "ok": True,
            "collection_id": self._collection_id(path),
            "batch_number": int(batch_number),
            "batch_file": batch_path.name,
            "total_records_in_batch": total,
            "offset": int(offset),
            "limit": int(limit),
            "has_more": int(offset) + len(rows) < total,
            "include_abstract": bool(include_abstract),
            "records": rows,
        }

    def _formal_snapshot_matches_manifest(self, manifest: Mapping[str, Any]) -> list[str]:
        expected = manifest.get("formal_kg_files") or {}
        if not isinstance(expected, Mapping):
            return ["manifest lacks formal_kg_files"]
        errors: list[str] = []
        expected_by_name = {
            Path(str(path)).name: value
            for path, value in expected.items()
            if isinstance(value, Mapping)
        }
        for path in self.formal_files:
            if not path.is_file():
                errors.append(f"formal file missing: {path.name}")
                continue
            target = expected.get(str(path.resolve())) or expected_by_name.get(path.name)
            if not isinstance(target, Mapping):
                errors.append(f"formal snapshot missing: {path.name}")
                continue
            stat = path.stat()
            if int(target.get("size_bytes") or -1) != stat.st_size:
                errors.append(f"formal file size changed: {path.name}")
            if int(target.get("mtime_ns") or -1) != stat.st_mtime_ns:
                errors.append(f"formal file timestamp changed: {path.name}")
        return errors

    def validate_collection(
        self, *, collection_id: str, mode: str = "seal"
    ) -> dict[str, Any]:
        validation_mode = str(mode or "seal").strip().lower()
        if validation_mode not in {"seal", "full"}:
            raise ValueError("mode must be 'seal' or 'full'")
        path = self._resolve_collection(collection_id)
        manifest = _read_json(path / "manifest.json")
        seal = _read_json(path / "COLLECTION_COMPLETE.json")
        queue = path / "abstracts_ready_for_extraction.jsonl"
        batch_dir = path / "extraction_batches"
        errors: list[str] = []
        warnings: list[str] = []

        if not manifest:
            errors.append("manifest.json is missing or empty")
        if not seal:
            errors.append("COLLECTION_COMPLETE.json is missing or empty")
        if not queue.is_file():
            errors.append("abstracts_ready_for_extraction.jsonl is missing")
        if str(manifest.get("status")) != "complete":
            errors.append("manifest status is not complete")
        if str(seal.get("status")) != "complete":
            errors.append("completion seal status is not complete")
        if bool(manifest.get("kg_injection")) or bool(seal.get("kg_injection")):
            errors.append("collection claims it was injected into the KG")
        if bool(manifest.get("formal_kg_mutated")):
            errors.append("collection claims the formal KG was mutated")
        years = manifest.get("years") or {}
        if years != {"start": literature.YEAR_START, "end": literature.YEAR_END}:
            errors.append("collection does not use the fixed 1980-2026 year window")

        observed_queue_hash = ""
        expected_queue_hash = str(
            seal.get("ready_queue_sha256")
            or (manifest.get("output_sha256") or {}).get("ready_queue")
            or ""
        )
        if queue.is_file():
            observed_queue_hash = _hash_file(queue)
            if not expected_queue_hash:
                errors.append("completion seal lacks a ready-queue SHA-256")
            elif observed_queue_hash != expected_queue_hash:
                errors.append("ready-queue SHA-256 does not match the seal")

        batch_files = list(batch_dir.glob("batch_*.jsonl")) if batch_dir.is_dir() else []
        expected_batches = int(seal.get("batches") or manifest.get("total_batches") or 0)
        if len(batch_files) != expected_batches:
            errors.append(
                f"batch count mismatch: observed {len(batch_files)}, expected {expected_batches}"
            )
        formal_errors = self._formal_snapshot_matches_manifest(manifest)
        if formal_errors:
            errors.append(
                "formal KG snapshot is stale; re-deduplicate this collection before extraction"
            )
            warnings.extend(formal_errors)

        row_count = 0
        batch_row_count = 0
        duplicate_aliases = 0
        invalid_rows: list[dict[str, Any]] = []
        if validation_mode == "full" and queue.is_file():
            aliases_seen: set[str] = set()
            for line_number, row in _iter_jsonl(queue):
                row_count += 1
                abstract = str(row.get("abstract") or "")
                row_errors = []
                if len(abstract) < literature.MIN_ABSTRACT_CHARACTERS:
                    row_errors.append("abstract below complete-abstract gate")
                if row.get("status") != "abstract_verified_ready_for_claim_extraction":
                    row_errors.append("invalid extraction-ready status")
                if row.get("search_provenance_only") is not True:
                    row_errors.append("search provenance is not explicitly non-membership")
                if bool(row.get("kg_injection")):
                    row_errors.append("row claims KG injection")
                aliases = paper_identity_aliases(row)
                collisions = [alias for alias in aliases if alias in aliases_seen]
                if collisions:
                    duplicate_aliases += len(collisions)
                    row_errors.append("duplicate paper identity within queue")
                aliases_seen.update(aliases)
                if row_errors and len(invalid_rows) < 20:
                    invalid_rows.append(
                        {
                            "line": line_number,
                            "paper_id": row.get("paper_id"),
                            "errors": row_errors,
                        }
                    )
            expected_papers = int(
                seal.get("papers") or manifest.get("collected_unique_papers") or 0
            )
            if row_count != expected_papers:
                errors.append(
                    f"ready-queue row mismatch: observed {row_count}, expected {expected_papers}"
                )
            if invalid_rows:
                errors.append("one or more ready-queue rows violate the frozen contract")
            if duplicate_aliases:
                errors.append("ready queue contains duplicate paper identities")
            for batch_path in batch_files:
                batch_row_count += sum(1 for _ in _iter_jsonl(batch_path))
            if batch_row_count != row_count:
                errors.append(
                    f"batch rows {batch_row_count} do not equal ready-queue rows {row_count}"
                )

        return {
            "ok": not errors,
            "valid_for_extraction": not errors,
            "collection_id": self._collection_id(path),
            "mode": validation_mode,
            "errors": errors,
            "warnings": warnings,
            "seal_present": bool(seal),
            "observed_ready_queue_sha256": observed_queue_hash,
            "expected_ready_queue_sha256": expected_queue_hash,
            "observed_batches": len(batch_files),
            "expected_batches": expected_batches,
            "validated_rows": row_count,
            "validated_batch_rows": batch_row_count,
            "duplicate_aliases": duplicate_aliases,
            "invalid_row_samples": invalid_rows,
            "formal_kg_mutated": False,
        }

    def start_sparse_collection(
        self,
        *,
        campaign_id: str,
        target_unique_papers: int,
        page_size: int = 1000,
    ) -> dict[str, Any]:
        slug = str(campaign_id or "").strip().lower()
        if not CAMPAIGN_RE.fullmatch(slug):
            raise ValueError(
                "campaign_id must be a 3-64 character lowercase slug using a-z, 0-9, '.', '_' or '-'"
            )
        if not 1 <= int(target_unique_papers) <= 1_000_000:
            raise ValueError("target_unique_papers must be in [1, 1,000,000]")
        if not 1 <= int(page_size) <= 1000:
            raise ValueError("page_size must be in [1, 1000]")
        if not self.collector_script.is_file():
            raise FileNotFoundError(f"collector script is missing: {self.collector_script}")

        self.mcp_staging_root.mkdir(parents=True, exist_ok=True)
        output_dir = (self.mcp_staging_root / slug).resolve()
        try:
            output_dir.relative_to(self.mcp_staging_root.resolve())
        except ValueError as exc:
            raise ValueError("campaign path escapes the controlled MCP staging root") from exc
        output_dir.mkdir(parents=True, exist_ok=True)
        seal = _read_json(output_dir / "COLLECTION_COMPLETE.json")
        if seal.get("status") == "complete":
            return {
                "ok": True,
                "started": False,
                "reason": "already_complete",
                "collection_id": self._collection_id(output_dir),
                "seal": seal,
                "formal_kg_mutated": False,
            }
        job_path = output_dir / "academic_mcp_job.json"
        previous_job = _read_json(job_path)
        previous_pid = int(previous_job.get("pid") or 0)
        if _pid_is_running(previous_pid):
            return {
                "ok": True,
                "started": False,
                "reason": "already_running",
                "collection_id": self._collection_id(output_dir),
                "pid": previous_pid,
                "formal_kg_mutated": False,
            }

        log_path = output_dir / "academic_mcp_collection.log"
        command = [
            sys.executable,
            str(self.collector_script),
            "--output-dir",
            str(output_dir),
            "--target-total",
            str(int(target_unique_papers)),
            "--batch-size",
            "100",
            "--page-size",
            str(int(page_size)),
            "--log-level",
            "INFO",
        ]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
                close_fds=True,
            )
        job = {
            "schema_version": "academic_mcp_collection_job.v1",
            "started_at": _utc_now(),
            "pid": process.pid,
            "campaign_id": slug,
            "collection_id": self._collection_id(output_dir),
            "target_unique_papers": int(target_unique_papers),
            "page_size": int(page_size),
            "batch_size": 100,
            "command": command,
            "log_path": str(log_path),
            "formal_kg_write_authorized": False,
        }
        _atomic_write_json(job_path, job)
        return {
            "ok": True,
            "started": True,
            "collection_id": self._collection_id(output_dir),
            "pid": process.pid,
            "job_path": str(job_path),
            "log_path": str(log_path),
            "years": {"start": literature.YEAR_START, "end": literature.YEAR_END},
            "pipeline_order": [
                "search lite metadata",
                "deduplicate against formal KG and active staging",
                "download complete provider abstract",
                "write 100-paper extraction batches",
                "seal only if formal KG stayed unchanged",
            ],
            "formal_kg_mutated": False,
        }


SERVICE = AcademicService()


__all__ = ["AcademicService", "SERVICE"]
