"""Auditable paper-identity deduplication for KG literature expansion.

The index is deliberately independent of Case Study membership.  It answers
only whether a paper identity is already represented in the formal KG, an
active staging queue, or the current collection campaign.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


logger = logging.getLogger(__name__)

IDENTITY_INDEX_SCHEMA_VERSION = "paper_identity_aliases.v2"
_SUPPORTED_STAGING_SUFFIXES = {".csv", ".jsonl"}
_SKIP_FILE_PARTS = (
    "dedup_audit",
    "search_errors",
    ".next.",
    ".tmp.",
)
_INJECTION_COMPLETE_MARKERS = (
    "INJECTION_COMPLETE.json",
    "INJECTION_COMPLETE",
)


def normalise_doi(value: object) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(" .;,)")


def normalise_title(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.replace("_", " ").split())


def _mapping_value(record: object, key: str) -> object:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _paper_payload(record: object) -> object:
    if not isinstance(record, Mapping):
        return record
    for key in ("source_paper", "paper"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            merged = dict(record)
            merged.update(nested)
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                for identity_key in (
                    "pmid", "doi", "pmcid", "arxiv_id", "openalex_id",
                    "paper_id", "title", "year", "publication_year",
                ):
                    if not merged.get(identity_key) and metadata.get(identity_key):
                        merged[identity_key] = metadata[identity_key]
            return merged
    return record


def _add_source_id_aliases(aliases: list[str], value: object) -> None:
    raw = str(value or "").strip()
    if not raw:
        return
    lowered = raw.lower()
    if lowered.startswith("pmid:"):
        raw = raw.split(":", 1)[1].strip()
        lowered = raw.lower()
    if raw.isdigit():
        aliases.append(f"pmid:{raw}")
    elif lowered.startswith("pmc") and lowered[3:].isdigit():
        aliases.append(f"pmcid:{lowered}")
    elif lowered.startswith(("oa:", "openalex:")):
        aliases.append(f"openalex:{lowered.split(':', 1)[1]}")
    elif lowered.startswith("https://openalex.org/"):
        aliases.append(f"openalex:{lowered.rsplit('/', 1)[-1]}")
    elif lowered.startswith("arxiv:"):
        aliases.append(f"arxiv:{lowered.split(':', 1)[1]}")
    elif lowered.startswith(("biorxiv:", "medrxiv:")):
        suffix = lowered.split(":", 1)[1]
        if suffix.startswith("10."):
            aliases.append(f"doi:{normalise_doi(suffix)}")
        else:
            aliases.append(f"paper_id:{lowered}")
    else:
        aliases.append(f"paper_id:{lowered}")


def paper_identity_aliases(record: object) -> list[str]:
    """Return all stable identities available for a paper-like record."""

    paper = _paper_payload(record)
    aliases: list[str] = []

    _add_source_id_aliases(aliases, _mapping_value(paper, "pmid"))
    for key in ("paper_id", "queue_id", "source_id", "cache_id"):
        _add_source_id_aliases(aliases, _mapping_value(paper, key))

    doi = normalise_doi(_mapping_value(paper, "doi"))
    if doi:
        aliases.append(f"doi:{doi}")

    pmcid = str(_mapping_value(paper, "pmcid") or "").strip().lower()
    if pmcid:
        aliases.append(f"pmcid:{pmcid}")

    arxiv_id = str(
        _mapping_value(paper, "arxiv_id")
        or _mapping_value(paper, "arxiv")
        or ""
    ).strip().lower()
    if arxiv_id:
        aliases.append(f"arxiv:{arxiv_id.removeprefix('arxiv:')}")

    openalex_id = str(
        _mapping_value(paper, "openalex_id")
        or _mapping_value(paper, "openalex")
        or ""
    ).strip().lower()
    if openalex_id:
        openalex_id = openalex_id.rsplit("/", 1)[-1].removeprefix("oa:")
        aliases.append(f"openalex:{openalex_id}")

    external_ids = _mapping_value(paper, "external_ids")
    if isinstance(external_ids, Mapping):
        for key, prefix in (
            ("pmid", "pmid"),
            ("pmcid", "pmcid"),
            ("doi", "doi"),
            ("arxiv", "arxiv"),
            ("openalex", "openalex"),
        ):
            value = external_ids.get(key) or external_ids.get(key.upper())
            if not value:
                continue
            normalized = normalise_doi(value) if key == "doi" else str(value).strip().lower()
            if normalized:
                aliases.append(f"{prefix}:{normalized}")

    title = normalise_title(_mapping_value(paper, "title"))
    year = _mapping_value(paper, "year") or _mapping_value(paper, "publication_year")
    year_text = str(year or "").strip()
    if title and re.fullmatch(r"\d{4}", year_text):
        aliases.append(f"title_year:{title}|{year_text}")

    return list(dict.fromkeys(alias for alias in aliases if alias.split(":", 1)[-1]))


def paper_identity_payload(record: object) -> dict[str, Any]:
    paper = _paper_payload(record)
    return {
        "pmid": str(_mapping_value(paper, "pmid") or ""),
        "doi": normalise_doi(_mapping_value(paper, "doi")),
        "pmcid": str(_mapping_value(paper, "pmcid") or ""),
        "arxiv_id": str(_mapping_value(paper, "arxiv_id") or ""),
        "openalex_id": str(_mapping_value(paper, "openalex_id") or ""),
        "title": str(_mapping_value(paper, "title") or ""),
        "year": _mapping_value(paper, "year") or _mapping_value(paper, "publication_year"),
        "aliases": paper_identity_aliases(record),
    }


@dataclass(frozen=True)
class PaperIdentityMatch:
    alias: str
    origin: str
    source_path: str


class GlobalPaperIdentityIndex:
    """Incremental SQLite identity index over formal KG and active staging."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_aliases (
                alias TEXT NOT NULL,
                source_path TEXT NOT NULL,
                origin TEXT NOT NULL,
                PRIMARY KEY (alias, source_path)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_alias ON paper_aliases(alias)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_source_path ON paper_aliases(source_path)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                source_path TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                records_scanned INTEGER NOT NULL,
                records_with_identity INTEGER NOT NULL,
                invalid_records INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        prior_version = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if prior_version is None or prior_version[0] != IDENTITY_INDEX_SCHEMA_VERSION:
            self.connection.execute("DELETE FROM paper_aliases")
            self.connection.execute("DELETE FROM indexed_files")
            self.connection.execute(
                "INSERT OR REPLACE INTO index_metadata(key, value) VALUES ('schema_version', ?)",
                (IDENTITY_INDEX_SCHEMA_VERSION,),
            )
        self.connection.commit()
        self.overlay: dict[str, PaperIdentityMatch] = {}
        self.sync_stats: dict[str, Any] = {}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "GlobalPaperIdentityIndex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def match(self, record: object) -> Optional[PaperIdentityMatch]:
        for alias in paper_identity_aliases(record):
            overlay = self.overlay.get(alias)
            if overlay is not None:
                return overlay
            row = self.connection.execute(
                """
                SELECT origin, source_path
                FROM paper_aliases
                WHERE alias = ?
                ORDER BY CASE origin
                    WHEN 'formal_kg' THEN 0
                    WHEN 'staging' THEN 1
                    ELSE 2
                END
                LIMIT 1
                """,
                (alias,),
            ).fetchone()
            if row is not None:
                return PaperIdentityMatch(alias=alias, origin=row[0], source_path=row[1])
        return None

    def add_overlay(self, record: object, *, origin: str, source_path: str) -> int:
        match = PaperIdentityMatch(alias="", origin=origin, source_path=source_path)
        added = 0
        for alias in paper_identity_aliases(record):
            if alias not in self.overlay:
                self.overlay[alias] = PaperIdentityMatch(
                    alias=alias,
                    origin=match.origin,
                    source_path=match.source_path,
                )
                added += 1
        return added

    def sync(
        self,
        *,
        formal_claim_store: Optional[Path],
        staging_roots: Iterable[Path],
    ) -> dict[str, Any]:
        staging_roots = tuple(Path(root).resolve() for root in staging_roots)
        sources: dict[str, tuple[str, Path]] = {}
        if formal_claim_store is not None:
            formal = Path(formal_claim_store).resolve()
            if formal.exists():
                sources[str(formal)] = ("formal_kg", formal)

        for root in staging_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in _SUPPORTED_STAGING_SUFFIXES:
                    continue
                if any(part in path.name.lower() for part in _SKIP_FILE_PARTS):
                    continue
                if _under_redundant_extraction_batch(path, root):
                    continue
                if _under_completed_injection(path, root):
                    continue
                sources[str(path)] = ("staging", path)

        expected = set(sources)
        managed_rows = self.connection.execute(
            "SELECT source_path FROM indexed_files WHERE origin IN ('formal_kg', 'staging')"
        ).fetchall()
        removed = [row[0] for row in managed_rows if row[0] not in expected]
        with self.connection:
            for source_path in removed:
                self.connection.execute(
                    "DELETE FROM paper_aliases WHERE source_path = ?", (source_path,)
                )
                self.connection.execute(
                    "DELETE FROM indexed_files WHERE source_path = ?", (source_path,)
                )

        updated_files = 0
        reused_files = 0
        records_scanned = 0
        records_with_identity = 0
        invalid_records = 0
        transiently_missing_files = 0
        for source_path, (origin, path) in sorted(sources.items()):
            try:
                stat = path.stat()
                prior = self.connection.execute(
                    "SELECT size_bytes, mtime_ns FROM indexed_files WHERE source_path = ?",
                    (source_path,),
                ).fetchone()
                if prior == (stat.st_size, stat.st_mtime_ns):
                    reused_files += 1
                    continue
                stats = self._reindex_file(path, origin)
            except FileNotFoundError:
                # Active workers publish JSONL checkpoints with atomic rename.
                # A path can therefore disappear after rglob()/is_file() but
                # before stat(), open(), or the final stat in _reindex_file().
                # Keep any prior aliases conservatively and retry next sync.
                transiently_missing_files += 1
                logger.info(
                    "paper identity index: skipped transiently missing file %s",
                    path,
                )
                continue
            updated_files += 1
            records_scanned += stats["records_scanned"]
            records_with_identity += stats["records_with_identity"]
            invalid_records += stats["invalid_records"]

        alias_rows = self.connection.execute(
            "SELECT COUNT(*) FROM paper_aliases"
        ).fetchone()[0]
        distinct_aliases = self.connection.execute(
            "SELECT COUNT(DISTINCT alias) FROM paper_aliases"
        ).fetchone()[0]
        self.sync_stats = {
            "index_path": str(self.path.resolve()),
            "formal_claim_store": str(Path(formal_claim_store).resolve())
            if formal_claim_store is not None
            else "",
            "staging_roots": [str(root) for root in staging_roots],
            "source_files": len(sources),
            "updated_files": updated_files,
            "reused_files": reused_files,
            "removed_files": len(removed),
            "records_scanned_this_sync": records_scanned,
            "records_with_identity_this_sync": records_with_identity,
            "invalid_records_this_sync": invalid_records,
            "transiently_missing_files": transiently_missing_files,
            "alias_rows": alias_rows,
            "distinct_aliases": distinct_aliases,
        }
        return dict(self.sync_stats)

    def _reindex_file(self, path: Path, origin: str) -> dict[str, int]:
        logger.info("paper identity index: scanning %s", path)
        records_scanned = 0
        records_with_identity = 0
        invalid_records = 0
        batch: list[tuple[str, str, str]] = []
        source_path = str(path.resolve())

        with self.connection:
            self.connection.execute(
                "DELETE FROM paper_aliases WHERE source_path = ?", (source_path,)
            )
            for record, valid in _iter_paper_records(path):
                records_scanned += 1
                if not valid:
                    invalid_records += 1
                    continue
                aliases = paper_identity_aliases(record)
                if not aliases:
                    continue
                records_with_identity += 1
                batch.extend((alias, source_path, origin) for alias in aliases)
                if len(batch) >= 10_000:
                    self.connection.executemany(
                        "INSERT OR IGNORE INTO paper_aliases(alias, source_path, origin) VALUES (?, ?, ?)",
                        batch,
                    )
                    batch.clear()
                if records_scanned % 50_000 == 0:
                    logger.info(
                        "  identity scan progress: %s records, %s with identity",
                        f"{records_scanned:,}",
                        f"{records_with_identity:,}",
                    )
            if batch:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO paper_aliases(alias, source_path, origin) VALUES (?, ?, ?)",
                    batch,
                )
            stat = path.stat()
            self.connection.execute(
                """
                INSERT OR REPLACE INTO indexed_files(
                    source_path, origin, size_bytes, mtime_ns,
                    records_scanned, records_with_identity, invalid_records, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_path,
                    origin,
                    stat.st_size,
                    stat.st_mtime_ns,
                    records_scanned,
                    records_with_identity,
                    invalid_records,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.info(
            "paper identity index: %s -> %s records, %s with identity, %s invalid",
            path.name,
            f"{records_scanned:,}",
            f"{records_with_identity:,}",
            f"{invalid_records:,}",
        )
        return {
            "records_scanned": records_scanned,
            "records_with_identity": records_with_identity,
            "invalid_records": invalid_records,
        }


class CandidatePaperDeduplicator:
    """Apply the global identity gate and buffer an exclusion audit."""

    def __init__(self, index: GlobalPaperIdentityIndex, audit_path: Path):
        self.index = index
        self.audit_path = Path(audit_path)
        self._audit_rows: list[dict[str, Any]] = []
        self.excluded = 0
        self.accepted = 0

    def is_seen(
        self,
        record: object,
        *,
        source: str,
        preset: str,
        query_index: Optional[int] = None,
        stage: str = "post_search_pre_abstract",
    ) -> bool:
        match = self.index.match(record)
        if match is None:
            return False
        self.excluded += 1
        self._audit_rows.append(
            {
                "schema_version": "paper_dedup_audit.v1",
                "excluded_at": datetime.now(timezone.utc).isoformat(),
                "candidate": paper_identity_payload(record),
                "source": source,
                "preset": preset,
                "query_index": query_index,
                "dedup_stage": stage,
                "matched_alias": match.alias,
                "matched_origin": match.origin,
                "matched_source_path": match.source_path,
                "exclusion_reason": f"paper identity already present in {match.origin}",
            }
        )
        return True

    def accept(
        self,
        record: object,
        *,
        source: str,
        preset: str,
        query_index: Optional[int] = None,
        collection_path: str = "",
    ) -> bool:
        if self.is_seen(
            record,
            source=source,
            preset=preset,
            query_index=query_index,
            stage="post_search_pre_extraction",
        ):
            return False
        self.index.add_overlay(
            record,
            origin="current_collection",
            source_path=collection_path,
        )
        self.accepted += 1
        return True

    def flush(self) -> int:
        if not self._audit_rows:
            return 0
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(self._audit_rows)
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._audit_rows.clear()
        return len(rows)

    def summary(self) -> dict[str, Any]:
        return {
            "accepted_current_campaign": self.accepted,
            "excluded_current_campaign": self.excluded,
            "audit_path": str(self.audit_path.resolve()),
            "index": dict(self.index.sync_stats),
        }


def _under_completed_injection(path: Path, root: Path) -> bool:
    parent = path.parent
    while True:
        if any((parent / marker).exists() for marker in _INJECTION_COMPLETE_MARKERS):
            return True
        if parent == root or parent.parent == parent:
            return False
        parent = parent.parent


def _under_redundant_extraction_batch(path: Path, root: Path) -> bool:
    """Skip batch copies when their canonical ready queue is already indexed."""

    try:
        relative_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return False
    if "extraction_batches" not in {part.casefold() for part in relative_parts}:
        return False
    parent = path.parent
    while True:
        if (parent / "abstracts_ready_for_extraction.jsonl").is_file():
            return True
        if parent == root or parent.parent == parent:
            return False
        parent = parent.parent


def _iter_paper_records(path: Path):
    if path.suffix.lower() == ".csv":
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    yield row, True
        except (OSError, csv.Error, UnicodeError):
            yield {}, False
        return

    try:
        with open(path, "rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line), True
                except (json.JSONDecodeError, UnicodeDecodeError):
                    yield {}, False
    except OSError:
        yield {}, False


__all__ = [
    "CandidatePaperDeduplicator",
    "GlobalPaperIdentityIndex",
    "PaperIdentityMatch",
    "normalise_doi",
    "normalise_title",
    "paper_identity_aliases",
    "paper_identity_payload",
]
