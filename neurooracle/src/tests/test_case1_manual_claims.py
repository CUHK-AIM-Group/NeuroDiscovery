from __future__ import annotations

from neurooracle.src.case1_manual_claims import (
    CURATED_CASE1_CLAIMS,
    _existing_claim_ids,
    _filter_existing_claim_ids,
    build_manual_case1_results,
    load_external_manual_claim_specs,
)
from neurooracle.src.schema import PaperRef


class _FakeCache:
    def __init__(self):
        self.records = {}
        for spec in CURATED_CASE1_CLAIMS:
            self.records[spec.pmid] = (
                spec.raw_sentence,
                PaperRef(
                    pmid=spec.pmid,
                    title=f"Manual Case Study 1 paper {spec.pmid}",
                    year=2026,
                    journal="Test Journal",
                ),
            )

    def get(self, pmid: str):
        return self.records.get(str(pmid))


def test_manual_case1_claim_table_covers_ten_seed_papers():
    pmids = {spec.pmid for spec in CURATED_CASE1_CLAIMS}

    assert pmids == {
        "32857118",
        "30988201",
        "33879764",
        "32539527",
        "29960671",
        "27137745",
        "26122586",
        "28461699",
        "26033243",
        "29377733",
    }
    assert len(CURATED_CASE1_CLAIMS) >= 30


def test_build_manual_case1_results_marks_claims_as_manual():
    results = build_manual_case1_results(_FakeCache(), pmids={"29960671"})

    assert len(results) == 1
    assert results[0].paper.pmid == "29960671"
    assert results[0].claims
    assert all(c.metadata["manual_curation"] is True for c in results[0].claims)
    assert all(c.metadata["curation_scope"] == "case1_transdiagnostic" for c in results[0].claims)


def test_filter_existing_manual_claim_ids_makes_rerun_noop():
    results = build_manual_case1_results(_FakeCache(), pmids={"29960671"})
    existing = {claim.id for result in results for claim in result.claims}

    assert _filter_existing_claim_ids(results, existing) == []


def test_existing_claim_ids_use_only_case1_prefix(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text('{"id":"CLM:CASE1MAN:29960671:001"}\n', encoding="utf-8")

    ids = _existing_claim_ids(path)

    assert "CLM:CASE1MAN:29960671:001" in ids


def test_load_external_manual_claim_specs(tmp_path):
    path = tmp_path / "manual_case1_claims.jsonl"
    path.write_text(
        '{"pmid":"1","subject":"cortical thickness","predicate":"distinguishes",'
        '"object":"schizophrenia","raw_sentence":"Cortical thickness differed.",'
        '"direction":"reduced","conditions":["adult"]}\n',
        encoding="utf-8",
    )

    specs = load_external_manual_claim_specs(path)

    assert len(specs) == 1
    assert specs[0].pmid == "1"
    assert specs[0].conditions == ("adult",)

