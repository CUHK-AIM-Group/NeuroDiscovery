"""Bounded read-only source loading; exclusions happen before decompression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from .validation import ValidationError


def load_sources(path: Path, *, protected_works: set[str], exposed_works: set[str],
                 limit: int = 60, after_job_id: str = "") -> list[dict]:
    if type(limit) is not int or not 1 <= limit <= 60:
        raise ValidationError("source reading cap is 60")
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        output = []
        candidates = connection.execute(
            "SELECT job_id,work_key FROM inputs WHERE status='READY' AND fold='corpus' "
            "AND job_id>? ORDER BY job_id", (after_job_id,),
        )
        for job_id, work_key in candidates:
            if work_key in protected_works or work_key in exposed_works:
                continue
            row = connection.execute(
                "SELECT source_sha256,input_sha256,input_zlib,utf8_bytes FROM inputs "
                "WHERE job_id=? AND work_key=? AND status='READY' AND fold='corpus'",
                (job_id, work_key),
            ).fetchone()
            if row is None:
                raise ValidationError("source binding changed")
            corpus_sha, input_sha, compressed, expected_bytes = row
            try:
                decoded = zlib.decompress(compressed)
                if len(decoded) != expected_bytes or hashlib.sha256(decoded).hexdigest() != input_sha:
                    raise ValidationError("compiled input binding mismatch")
                payload = json.loads(decoded.decode("utf-8"))
                text = payload["abstract"]
                if not isinstance(text, str) or not text.strip():
                    raise ValidationError("empty abstract")
            except (zlib.error, UnicodeError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValidationError("source decoding failed; never match compressed bytes") from exc
            output.append({
                "job_id": job_id, "work_key": work_key, "fold": "corpus",
                "corpus_document_sha256": corpus_sha, "compiled_input_sha256": input_sha,
                "text": text, "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_family": None, "source_family_review_required": True,
                "reading_scope": "abstract_only", "labels": None,
            })
            if len(output) >= limit:
                break
        return output
    finally:
        connection.close()
