from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest

from neurooracle.src.kg_bibliography_repair import review_title, apply_title, quote_key, without_selected_title, verify_title_result
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, VERSION
from neurooracle.src.pubmed_title_evidence import complete_title_evidence


QUOTE = "The measured binding was not increased after treatment in the tested adult population."
TITLE = "Measured binding in adult subjects after a randomized intervention."


def fixture():
    witness = dict(response_sha256="a" * 64, response_file="public.xml",
                   url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=123")
    article = ET.fromstring(f'<PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>{TITLE}</ArticleTitle>'
        f'<Abstract><AbstractText>{QUOTE}</AbstractText></Abstract></Article></MedlineCitation>'
        '<PubmedData><ArticleIdList><ArticleId IdType="pubmed">123</ArticleId>'
        '<ArticleId IdType="doi">10.1234/test</ArticleId></ArticleIdList></PubmedData></PubmedArticle>')
    record = dict(pmid="123", title=TITLE.casefold().rstrip("."), years=["2020"], preprint=False,
                  doi=["10.1234/test"], pmcid=[], witness=witness)
    record["complete_title_evidence"] = complete_title_evidence(article, record, witness)
    registry = VerifiedPaperIdentities(dict(version=VERSION, records={"123": record}, observed_collisions={}))
    row = dict(id="CLM:test", metadata=dict(id="CLM:test", source_paper=dict(pmid="123", doi="10.1234/test",
        title="Measured binding", year=2020, authors="Original full names", journal="Example Journal"),
        raw_text=QUOTE, negated=True, evidence=dict(sample_size=42, conditions=["adult"]), metadata=dict(unknown_science=[1, 2])))
    return row, registry, {"123": article}, {"123": witness}


def test_exact_quote_repairs_only_title_without_mutating_input():
    row, registry, articles, witnesses = fixture()
    original = deepcopy(row)
    event, reason = review_title(row, registry, articles, witnesses)
    assert reason == "exact_identifiers_and_owning_complete_abstract_quote"
    assert event["value"] == TITLE
    out = apply_title(row, event)
    assert row == original
    assert without_selected_title(row) == without_selected_title(out)
    assert registry.resolve(out["metadata"])["status"] == "verified"
    assert "before" not in event and "source_paper" not in event and QUOTE not in str(event)


@pytest.mark.parametrize("change", ["negation", "short_quote", "paraphrase", "year", "synthetic", "outer_pmid", "alternate_title", "preprint", "arxiv", "authority_preprint"])
def test_unproven_or_conflicting_sources_are_not_repaired(change):
    row, registry, articles, witnesses = fixture()
    md = row["metadata"]
    if change == "negation": md["raw_text"] = QUOTE.replace("not ", "")
    if change == "short_quote": md["raw_text"] = "The measured binding"
    if change == "paraphrase": md["raw_text"] = "There was no increase in binding after treatment in adults enrolled in this study."
    if change == "year": md["source_paper"]["year"] = 2009
    if change == "synthetic": md["source_paper"]["pmid"] = "OA:W123"
    if change == "outer_pmid": md["pmid"] = "456"
    if change == "alternate_title": md["title"] = "Another title"
    if change == "preprint": md["source_paper"]["journal"] = "bioRxiv"
    if change == "arxiv": md["source_paper"]["arxiv_id"] = "2001.00123"
    if change == "authority_preprint":
        payload = registry.export_payload(); payload["records"]["123"]["preprint"] = True
        registry = VerifiedPaperIdentities(payload)
    assert review_title(row, registry, articles, witnesses)[0] is None


def test_doi_only_does_not_fill_missing_pmid():
    row, registry, articles, witnesses = fixture()
    row["metadata"]["source_paper"]["pmid"] = ""
    event, _ = review_title(row, registry, articles, witnesses)
    assert event
    assert apply_title(row, event)["metadata"]["source_paper"]["pmid"] == ""


def test_wrong_owning_xml_fails_closed():
    row, registry, articles, witnesses = fixture()
    articles["123"].find("./MedlineCitation/PMID").text = "456"
    with pytest.raises(ValueError): review_title(row, registry, articles, witnesses)


@pytest.mark.parametrize("change", ["claim", "field", "value", "result_sha"])
def test_tampered_plan_or_claim_rejected(change):
    row, registry, articles, witnesses = fixture()
    event, _ = review_title(row, registry, articles, witnesses)
    if change == "claim": row["metadata"]["evidence"]["sample_size"] = 99
    if change == "field": event["field"] = "metadata.raw_text"
    if change == "value": event["value"] = "Invented"
    if change == "result_sha": event["current_node_sha256"] = "0" * 64
    with pytest.raises(ValueError): apply_title(row, event)


def test_quote_key_keeps_negation_punctuation_and_ranges():
    assert quote_key("NOT increased; 1–2") != quote_key("increased 12")
    assert quote_key("first\nsecond") == "first second"


def test_candidate_witness_is_reproduced_without_old_record():
    row, registry, articles, witnesses = fixture()
    event, _ = review_title(row, registry, articles, witnesses)
    out = apply_title(row, event)
    verify_title_result(out, event, registry, articles, witnesses)
    out["metadata"]["negated"] = False
    with pytest.raises(ValueError): verify_title_result(out, event, registry, articles, witnesses)


def test_current_census_is_rebuilt_from_only_current_rows(tmp_path):
    import sqlite3
    from neurooracle.scripts.apply_kg_bibliography_titles import CurrentCensus
    from neurooracle.src.relation_evidence import relation_key as test_relation_key
    from neurooracle.src.kg_identity_pilot import digest
    from neurooracle.src.kg_paper_identity import bibliography
    class Terms:
        relation_key = staticmethod(test_relation_key)
    row, registry, articles, witnesses = fixture()
    row["metadata"].update(subject_id="A", subject_name="a", predicate="affects", object_id="B", object_name="b")
    event, _ = review_title(row, registry, articles, witnesses)
    out = apply_title(row, event)
    path = tmp_path / "current.sqlite"
    census = CurrentCensus(path, Terms(), {row["id"]})
    census.add(out["id"], out)
    summary = census.finish()
    assert summary["counts"]["claims"] == summary["counts"]["shared_claims"] == 1
    db = sqlite3.connect(path)
    assert db.execute("select node_sha,paper_sig from claims").fetchone() == (digest(out), digest(bibliography(out["metadata"]["source_paper"])))
    assert db.execute("select title from papers").fetchone()[0] == TITLE.casefold().rstrip(".")
    assert db.execute("select count(*) from papers").fetchone()[0] == 1
    db.close()
    with pytest.raises((ValueError, RuntimeError, AssertionError)): CurrentCensus(path, Terms(), set())
