from __future__ import annotations

import json

from neurooracle.src.paper_identity import (
    CandidatePaperDeduplicator,
    GlobalPaperIdentityIndex,
    paper_identity_aliases,
)
from neurooracle.src.schema import PaperRef


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_aliases_cover_primary_ids_and_title_year():
    aliases = paper_identity_aliases({
        "pmid": "12345678",
        "doi": "https://doi.org/10.1000/Example",
        "pmcid": "PMC123",
        "arxiv_id": "2601.12345",
        "openalex_id": "https://openalex.org/W123",
        "title": "A Genetic → Imaging Study",
        "year": 2024,
    })

    assert "pmid:12345678" in aliases
    assert "doi:10.1000/example" in aliases
    assert "pmcid:pmc123" in aliases
    assert "arxiv:2601.12345" in aliases
    assert "openalex:w123" in aliases
    assert any(alias.startswith("title_year:a genetic imaging study|2024") for alias in aliases)

    historical_aliases = paper_identity_aliases({
        "source_paper": {"pmid": "", "title": "", "year": None},
        "metadata": {"pmid": "87654321", "title": "Metadata paper", "year": 2019},
    })
    assert "pmid:87654321" in historical_aliases
    assert "title_year:metadata paper|2019" in historical_aliases


def test_global_index_matches_formal_and_active_staging_but_skips_completed(tmp_path):
    formal = tmp_path / "formal" / "extracted_claims.jsonl"
    staging = tmp_path / "staging"
    _write_jsonl(formal, [{
        "source_paper": {
            "pmid": "11111111",
            "doi": "10.1000/formal",
            "title": "Formal paper",
            "year": 2020,
        }
    }])
    _write_jsonl(staging / "active" / "queue.jsonl", [{
        "pmid": "22222222",
        "doi": "10.1000/staged",
        "title": "Staged paper",
        "year": 2021,
    }])
    completed = staging / "completed"
    _write_jsonl(completed / "queue.jsonl", [{
        "pmid": "33333333",
        "title": "Completed paper",
        "year": 2022,
    }])
    (completed / "INJECTION_COMPLETE.json").write_text("{}\n", encoding="utf-8")

    with GlobalPaperIdentityIndex(tmp_path / "index.sqlite3") as index:
        stats = index.sync(formal_claim_store=formal, staging_roots=[staging])

        formal_match = index.match({"doi": "10.1000/formal"})
        staged_match = index.match({"title": "Staged paper", "year": 2021})
        completed_match = index.match({"pmid": "33333333"})

        assert formal_match is not None and formal_match.origin == "formal_kg"
        assert staged_match is not None and staged_match.origin == "staging"
        assert completed_match is None
        assert stats["source_files"] == 2

        reused = index.sync(formal_claim_store=formal, staging_roots=[staging])
        assert reused["updated_files"] == 0
        assert reused["reused_files"] == 2


def test_global_index_skips_atomic_publish_temporary_files(tmp_path):
    staging = tmp_path / "staging"
    _write_jsonl(staging / "queue.jsonl", [{
        "pmid": "22222222",
        "title": "Canonical staged paper",
        "year": 2021,
    }])
    _write_jsonl(staging / "checkpoint.next.jsonl", [{
        "pmid": "99999999",
        "title": "Transient publish file",
        "year": 2021,
    }])

    with GlobalPaperIdentityIndex(tmp_path / "index.sqlite3") as index:
        stats = index.sync(formal_claim_store=None, staging_roots=[staging])

        assert index.match({"pmid": "22222222"}) is not None
        assert index.match({"pmid": "99999999"}) is None
        assert stats["source_files"] == 1


def test_global_index_retries_file_that_disappears_during_sync(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    volatile = staging / "volatile.jsonl"
    row = {
        "pmid": "77777777",
        "title": "Atomically replaced paper",
        "year": 2024,
    }
    _write_jsonl(volatile, [row])

    with GlobalPaperIdentityIndex(tmp_path / "index.sqlite3") as index:
        original_reindex = index._reindex_file
        disappear_once = True

        def transient_reindex(path, origin):
            nonlocal disappear_once
            if path == volatile.resolve() and disappear_once:
                disappear_once = False
                path.unlink()
                raise FileNotFoundError(path)
            return original_reindex(path, origin)

        monkeypatch.setattr(index, "_reindex_file", transient_reindex)
        first = index.sync(formal_claim_store=None, staging_roots=[staging])
        assert first["transiently_missing_files"] == 1
        assert index.match(row) is None

        _write_jsonl(volatile, [row])
        second = index.sync(formal_claim_store=None, staging_roots=[staging])
        assert second["transiently_missing_files"] == 0
        assert index.match(row) is not None


def test_candidate_gate_is_nonexclusive_of_source_and_writes_audit(tmp_path):
    formal = tmp_path / "formal.jsonl"
    _write_jsonl(formal, [{
        "source_paper": {
            "pmid": "44444444",
            "title": "Known paper",
            "year": 2023,
        }
    }])

    with GlobalPaperIdentityIndex(tmp_path / "index.sqlite3") as index:
        index.sync(formal_claim_store=formal, staging_roots=[])
        gate = CandidatePaperDeduplicator(index, tmp_path / "audit.jsonl")

        assert not gate.accept(
            PaperRef(pmid="44444444", title="Known paper", year=2023),
            source="openalex",
            preset="case2_pathway_mediation",
        )
        assert gate.accept(
            PaperRef(pmid="55555555", title="New paper", year=2024),
            source="pubmed",
            preset="case2_pathway_mediation",
            collection_path="collection_metadata.csv",
        )
        assert not gate.accept(
            PaperRef(doi="", title="New paper", year=2024),
            source="europepmc",
            preset="case2_pathway_mediation",
        )
        assert gate.flush() == 2

    rows = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert rows[0]["matched_origin"] == "formal_kg"
    assert rows[1]["matched_origin"] == "current_collection"
    assert all(row["preset"] == "case2_pathway_mediation" for row in rows)
