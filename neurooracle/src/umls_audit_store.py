"""Read-only access to lossless UMLS detail records outside a compact graph."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


ATOM_METADATA_FIELDS = (
    "source_mention_id", "mapping_status", "mapping_count", "evidence_span",
)
CUI_METADATA_FIELDS = ("semantic_type_names",)
EDGE_METADATA_FIELDS = ("method", "review_status", "semantic_compatibility")
TABLES = frozenset({"atoms", "cuis", "mappings"})


def compact_record(record: dict, table: str, record_id: str) -> dict:
    """Keep runtime fields and a reference; the complete record stays in SQLite."""
    fields = {
        "atoms": ATOM_METADATA_FIELDS,
        "cuis": CUI_METADATA_FIELDS,
        "mappings": EDGE_METADATA_FIELDS,
    }[table]
    metadata = record.get("metadata") or {}
    result = dict(record)
    result["metadata"] = {key: metadata[key] for key in fields if key in metadata}
    result["metadata"]["audit_ref"] = f"{table}/{record_id}"
    return result


class UmlsAuditStore:
    """Indexed lookups without loading the graph or permitting database writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "UmlsAuditStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def fetch(self, reference: str) -> dict:
        table, separator, record_id = reference.partition("/")
        if table not in TABLES or not separator or not record_id:
            raise ValueError(f"invalid UMLS detail reference: {reference!r}")
        row = self.connection.execute(
            f"SELECT payload_json FROM {table} WHERE record_id=?", (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(reference)
        return json.loads(row[0])

    def hydrate(self, record: dict) -> dict:
        reference = (record.get("metadata") or {}).get("audit_ref")
        if not reference:
            return record
        original = self.fetch(reference)
        table, record_id = reference.split("/", 1)
        if compact_record(original, table, record_id) != record:
            raise ValueError("core record does not match its UMLS detail record")
        return original

    def atoms_for_mention(self, mention_id: str, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        rows = self.connection.execute(
            "SELECT record_id,mapping_status,in_core FROM atoms "
            "WHERE source_mention_id=? ORDER BY ordinal LIMIT ?",
            (mention_id, limit),
        )
        return [dict(zip(("atom_id", "mapping_status", "in_core"), row)) for row in rows]

    def mappings_for_atom(self, atom_id: str, include_needs_review: bool = False) -> list[dict]:
        clause = "" if include_needs_review else " AND review_status='auto_accepted_exact'"
        rows = self.connection.execute(
            "SELECT payload_json FROM mappings WHERE source_id=?" + clause + " ORDER BY ordinal",
            (atom_id,),
        )
        return [json.loads(row[0]) for row in rows]
