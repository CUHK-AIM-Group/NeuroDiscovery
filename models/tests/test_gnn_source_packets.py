import hashlib
import json
import sqlite3
import zlib

import pytest

from models.kg_link_prediction.source_packets import load_sources
from models.kg_link_prediction.validation import ValidationError


def database(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE inputs(job_id TEXT PRIMARY KEY, work_key TEXT, status TEXT, fold TEXT, source_sha256 TEXT, input_sha256 TEXT, input_zlib BLOB, utf8_bytes INT)")
        for index in range(65):
            payload = json.dumps({"abstract": "Brain &amp; memory; source " + str(index)}).encode()
            connection.execute("INSERT INTO inputs VALUES (?,?,?,?,?,?,?,?)", (
                f"job-{index:03}", f"work-{index}", "READY", "heldout" if index == 0 else "corpus",
                "historical-document-hash", hashlib.sha256(payload).hexdigest(), zlib.compress(payload), len(payload),
            ))
        connection.execute("UPDATE inputs SET input_zlib=? WHERE job_id IN ('job-000','job-001','job-002')", (b"bad",))


def test_excludes_heldout_protected_and_exposed_before_decoding(tmp_path):
    path = tmp_path / "sources.sqlite"
    database(path)
    before = path.read_bytes()
    rows = load_sources(path, protected_works={"work-1"}, exposed_works={"work-2"})
    assert len(rows) == 60
    assert rows[0]["work_key"] == "work-3"
    assert rows[0]["text"].startswith("Brain &amp;")
    assert all(row["source_family"] is None and row["labels"] is None for row in rows)
    assert path.read_bytes() == before


def test_bad_compression_never_silently_becomes_text(tmp_path):
    path = tmp_path / "sources.sqlite"
    database(path)
    with pytest.raises(ValidationError, match="decoding"):
        load_sources(path, protected_works=set(), exposed_works=set())


def test_limit_must_stay_within_reading_budget(tmp_path):
    with pytest.raises(ValidationError, match="60"):
        load_sources(tmp_path / "missing.sqlite", protected_works=set(), exposed_works=set(), limit=61)


def test_input_hash_mismatch_is_not_normalized_away(tmp_path):
    path = tmp_path / "sources.sqlite"
    database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE inputs SET input_sha256='wrong' WHERE job_id='job-003'")
    with pytest.raises(ValidationError, match="binding"):
        load_sources(path, protected_works={"work-1"}, exposed_works={"work-2"})
