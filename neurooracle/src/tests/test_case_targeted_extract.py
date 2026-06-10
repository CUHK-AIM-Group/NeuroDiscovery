from __future__ import annotations

import csv

import pytest

from neurooracle.src import case_targeted_extract as cte
from neurooracle.src.schema import PaperRef


def test_supplemental_classic_preset_builds_history_queries():
    queries = cte.build_case_targeted_queries(
        "case2_supplemental_classic",
        year_start=2010,
        year_end=2026,
    )
    phrases = cte.build_case_targeted_search_phrases("case2_supplemental_classic")

    assert len(queries) == 8
    assert len(phrases) == 8
    assert "review[Publication Type]" in queries[0]
    assert "2010:2026[pdat]" in queries[0]
    assert any("ADNI" in q for q in queries)
    assert any("UK Biobank" in phrase for phrase in phrases)


def test_case1_transdiagnostic_preset_builds_cross_disorder_queries():
    queries = cte.build_case_targeted_queries(
        "case1_transdiagnostic",
        year_start=2010,
        year_end=2026,
    )
    phrases = cte.build_case_targeted_search_phrases("case1_transdiagnostic")

    assert len(queries) >= 12
    assert len(phrases) >= 18
    assert any("transdiagnostic" in q for q in queries)
    assert any("schizophrenia" in q and "bipolar disorder" in q for q in queries)
    assert any("functional connectivity" in q for q in queries)
    assert any("RDoC" in q and "p-factor" in q for q in queries)
    assert any("ABCD" in q and "Transdiagnostic Connectome Project" in q for q in queries)
    assert any("ALFF" in q and "ReHo" in q for q in queries)
    assert not any("polygenic risk score" in q for q in queries)
    assert any("ENIGMA" in phrase for phrase in phrases)
    assert any("HiTOP" in phrase for phrase in phrases)
    assert any("Transdiagnostic Connectome Project" in phrase for phrase in phrases)


def test_old_case_targeted_preset_aliases_are_rejected():
    for old_preset in ("cs2_transdiagnostic", "cs3_supplemental_classic"):
        with pytest.raises(ValueError):
            cte.build_case_targeted_queries(
                old_preset,
                year_start=2020,
                year_end=2020,
            )
        with pytest.raises(ValueError):
            cte.build_case_targeted_search_phrases(old_preset)


def test_anysearch_reference_parsers_extract_primary_ids():
    text = (
        "PDF: https://alzres.biomedcentral.com/counter/pdf/"
        "10.1186/s13195-023-01256-z.pdf PMID: 37199999"
    )

    assert cte._extract_pmid(text) == "37199999"
    assert cte._extract_doi(text) == "10.1186/s13195-023-01256-z"


def test_openalex_title_resolution_requires_close_title(monkeypatch):
    paper = PaperRef(
        pmid="OA:W123",
        doi="10.1000/example",
        title="Polygenic risk score and cortical thickness in Alzheimer disease",
        year=2025,
    )

    def fake_search_openalex(query, *, year_start, year_end, max_results):
        assert year_start == 2010
        assert year_end == 2026
        assert max_results == 3
        return [("OA:W123", "Results showed an association.", paper)]

    monkeypatch.setattr(cte, "_search_openalex", fake_search_openalex)

    rec = cte._resolve_openalex_by_title(
        "Polygenic risk score and cortical thickness in Alzheimer's disease",
        year_start=2010,
        year_end=2026,
    )

    assert rec is not None
    assert rec[0] == "OA:W123"


def test_openalex_title_resolution_rejects_unrelated_title(monkeypatch):
    paper = PaperRef(
        pmid="OA:W123",
        doi="10.1000/example",
        title="Functional connectivity in depression",
        year=2025,
    )

    monkeypatch.setattr(
        cte,
        "_search_openalex",
        lambda *args, **kwargs: [("OA:W123", "Results showed an association.", paper)],
    )

    assert cte._resolve_openalex_by_title(
        "Polygenic risk score and cortical thickness in Alzheimer's disease",
        year_start=2010,
        year_end=2026,
    ) is None


def test_europepmc_result_normalises_primary_ids():
    rec = cte._normalise_europepmc_result({
        "pmid": "12345678",
        "pmcid": "PMC123",
        "doi": "https://doi.org/10.1000/Example",
        "title": "A brain imaging study",
        "authorString": "Ada Lovelace et al.",
        "pubYear": "2025",
        "journalTitle": "Neuro Journal",
        "abstractText": "Results showed a robust association.",
    })

    assert rec is not None
    cache_id, abstract, paper = rec
    assert cache_id == "12345678"
    assert abstract.startswith("Results")
    assert paper.pmid == "12345678"
    assert paper.doi == "10.1000/example"
    assert paper.year == 2025


def test_preprint_result_normalises_server_scoped_cache_id():
    rec = cte._normalise_preprint_result({
        "doi": "10.1101/2026.01.02.123456",
        "title": "Transdiagnostic MRI markers",
        "authors": "A Author; B Author",
        "date": "2026-01-02",
        "server": "medRxiv",
        "abstract": "This preprint studies MRI markers across psychiatric disorders.",
        "published": "10.1000/final",
    }, "medrxiv")

    assert rec is not None
    cache_id, abstract, paper = rec
    assert cache_id == "MEDRXIV:10.1101/2026.01.02.123456"
    assert paper.pmid == cache_id
    assert paper.doi == "10.1101/2026.01.02.123456"
    assert paper.year == 2026
    assert "published=10.1000/final" in paper.journal
    assert "psychiatric" in abstract


def test_collect_only_writes_cache_and_collection_metadata(tmp_path, monkeypatch):
    paper = PaperRef(
        pmid="ARXIV:2601.12345",
        doi="",
        title="Transdiagnostic brain imaging",
        authors="Example Author",
        year=2026,
        journal="arXiv",
    )

    def fake_select_arxiv(**kwargs):
        assert kwargs["preset"] == "case1_transdiagnostic"
        return [("A cached abstract.", paper)], 0, [{"query_index": 1, "hits": 1, "new_added": 1}]

    monkeypatch.setattr(cte, "_select_arxiv_papers", fake_select_arxiv)
    monkeypatch.setattr(cte, "load_graph", lambda *args, **kwargs: pytest.fail("collect-only loaded graph"))

    summary = cte.run_case_targeted_extraction(
        preset="case1_transdiagnostic",
        source="arxiv",
        target_papers=1,
        max_results_per_query=1,
        data_dir=tmp_path,
        collect_only=True,
    )

    assert summary["mode"] == "collect-only"
    assert summary["total_papers"] == 1
    assert (tmp_path / "abstract_cache.jsonl").exists()
    assert (tmp_path / "collection_metadata.csv").exists()

    with open(tmp_path / "collection_metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["pmid"] == "ARXIV:2601.12345"
    assert rows[0]["source"] == "arxiv"
    assert rows[0]["preset"] == "case1_transdiagnostic"
