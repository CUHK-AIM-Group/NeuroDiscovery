from __future__ import annotations

import csv

import pytest

from neurooracle.src import case_targeted_extract as cte
from neurooracle.src.schema import PaperRef


def test_supplemental_classic_preset_builds_history_queries():
    queries = cte.build_case_targeted_queries(
        "case2_supplemental_classic",
        year_start=1980,
        year_end=2026,
    )
    phrases = cte.build_case_targeted_search_phrases("case2_supplemental_classic")

    assert len(queries) == 8
    assert len(phrases) == 8
    assert "review[Publication Type]" in queries[0]
    assert "1980:2026[pdat]" in queries[0]
    assert any("ADNI" in q for q in queries)
    assert any("UK Biobank" in phrase for phrase in phrases)


def test_case1_transdiagnostic_preset_builds_cross_disorder_queries():
    queries = cte.build_case_targeted_queries(
        "case1_transdiagnostic",
        year_start=1980,
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


def test_case_study_search_window_is_frozen_to_1980_2026():
    with pytest.raises(ValueError, match="canonical publication window 1980-2026"):
        cte.build_case_targeted_queries(
            "case2_pathway_mediation",
            year_start=2010,
            year_end=2026,
        )


def test_case2_queries_target_complete_longitudinal_mediation_chain():
    queries = cte.build_case_targeted_queries(
        "case2_pathway_mediation",
        year_start=1980,
        year_end=2026,
    )
    phrases = cte.build_case_targeted_search_phrases("case2_pathway_mediation")

    assert len(queries) >= 15
    assert all("1980:2026[pdat]" in query for query in queries)
    assert any("mediation" in query and "longitudinal" in query for query in queries)
    assert any("Mendelian randomization" in query for query in queries)
    assert any("ADNI" in query and "BioFINDER" in query for query in queries)
    assert any("PPMI" in query for query in queries)
    assert any("longitudinal" in phrase and "mediation" in phrase for phrase in phrases)


def test_case2_high_recall_expansion_covers_diseases_cohorts_and_modalities():
    queries = cte.build_case_targeted_queries(
        "case2_high_recall_expansion",
        year_start=1980,
        year_end=2026,
    )
    phrases = cte.build_case_targeted_search_phrases("case2_high_recall_expansion")

    assert len(queries) >= 70
    assert len(phrases) >= 45
    assert len(queries) == len(set(queries))
    assert len(phrases) == len(set(phrases))
    assert all("1980:2026[pdat]" in query for query in queries)
    assert any("Alzheimer disease" in query and "longitudinal" in query for query in queries)
    assert any("multiple sclerosis" in query for query in queries)
    assert any("UK Biobank" in query for query in queries)
    assert any("diffusion MRI" in query for query in queries)
    assert any("Mendelian randomization" in query for query in queries)


def test_usable_abstract_rejects_short_and_placeholder_text():
    assert not cte._is_usable_abstract("peer reviewed")
    assert not cte._is_usable_abstract("No abstract is available for this article." * 10)
    assert cte._is_usable_abstract("A substantive neuroscience abstract. " * 10)


def test_anysearch_reference_parsers_extract_primary_ids():
    text = (
        "PDF: https://alzres.biomedcentral.com/counter/pdf/"
        "10.1186/s13195-023-01256-z.pdf PMID: 37199999"
    )

    assert cte._extract_pmid(text) == "37199999"
    assert cte._extract_doi(text) == "10.1186/s13195-023-01256-z"


def test_pubmed_search_uses_post_for_long_case_queries(monkeypatch):
    import requests

    from neurooracle.src import batch_extract

    calls = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"esearchresult": {"idlist": ["12345678"]}}

    def fake_post(url, *, data, timeout):
        calls.append((url, data, timeout))
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    assert batch_extract._search_pubmed("x" * 5_000, 1) == ["12345678"]
    assert calls[0][1]["term"] == "x" * 5_000
    assert calls[0][2] == 30


def test_pubmed_year_falls_back_to_medline_date():
    import xml.etree.ElementTree as ET

    from neurooracle.src import batch_extract

    article = ET.fromstring(
        "<PubmedArticle><JournalIssue><PubDate>"
        "<MedlineDate>2025 Oct-Dec</MedlineDate>"
        "</PubDate></JournalIssue></PubmedArticle>"
    )
    assert batch_extract._extract_pubmed_year(article) == 2025


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


def test_openalex_rate_limit_is_classified(monkeypatch):
    import requests

    sleeps = []

    class Response:
        status_code = 429
        headers = {}
        text = "rate limited"

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(cte.time, "sleep", lambda seconds: sleeps.append(seconds))

    records = cte._search_openalex(
        "genetics imaging longitudinal mediation",
        year_start=1980,
        year_end=2026,
        max_results=1,
    )

    assert records == []
    assert records.error == "http_429"
    assert sleeps == [15.0, 30.0, 60.0, 120.0]


def test_europepmc_search_uses_cursor_pagination(monkeypatch):
    import requests

    calls = []
    payloads = [
        {
            "nextCursorMark": "cursor-2",
            "resultList": {"result": [{
                "pmid": "101",
                "title": "First",
                "abstractText": "First substantive abstract. " * 12,
                "pubYear": "2020",
            }]},
        },
        {
            "nextCursorMark": "cursor-2",
            "resultList": {"result": [{
                "pmid": "102",
                "title": "Second",
                "abstractText": "Second substantive abstract. " * 12,
                "pubYear": "2021",
            }]},
        },
    ]

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        @staticmethod
        def raise_for_status():
            return None

        def json(self):
            return self._payload

    def fake_get(url, *, params, timeout):
        calls.append(dict(params))
        return Response(payloads.pop(0))

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(cte.time, "sleep", lambda _seconds: None)

    records = cte._search_europepmc(
        "genetics MRI longitudinal",
        year_start=1980,
        year_end=2026,
        max_results=2,
    )

    assert [record[0] for record in records] == ["101", "102"]
    assert [call["cursorMark"] for call in calls] == ["*", "cursor-2"]
    assert [call["pageSize"] for call in calls] == [2, 1]


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
        "abstractText": "Results showed a robust association. " * 10,
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
        "abstract": "This preprint studies MRI markers across psychiatric disorders. " * 8,
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
        formal_claim_store=None,
        staging_roots=[],
        identity_index_path=tmp_path / "identity.sqlite3",
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
    assert summary["year_start"] == 1980
    assert summary["year_end"] == 2026
    assert summary["kg_injection"] is False
    assert (tmp_path / "collection_manifest.json").exists()
