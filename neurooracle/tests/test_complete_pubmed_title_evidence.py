from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest

from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.pubmed_title_evidence import complete_title_evidence, presentation_key, title_renderings
from neurooracle.tests.test_kg_paper_authorities import registry, source_record, claim

WITNESS = dict(response_sha256="b"*64, url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=1", response_file="witness.xml")


def xml(title, *, pmid="1", doi="10.1234/example", book=False):
    content = f'<PMID>{pmid}</PMID><ArticleTitle>{title}</ArticleTitle>'
    ids = f'<ArticleIdList><ArticleId IdType="pubmed">{pmid}</ArticleId><ArticleId IdType="doi">{doi}</ArticleId><ArticleId IdType="pmc">PMC{pmid}</ArticleId></ArticleIdList>'
    if book: return ET.fromstring(f'<PubmedBookArticle><BookDocument>{content}{ids}</BookDocument></PubmedBookArticle>')
    return ET.fromstring(f'<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>{title}</ArticleTitle></Article></MedlineCitation><PubmedData>{ids}<ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9999/other</ArticleId></ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle>')


def evidenced(summary="A (18)F study.", markup="A <sup>18</sup>F study.", **kwargs):
    p = registry(source_record(title=summary)).export_payload()
    r = p["records"]["1"]
    r["complete_title_evidence"] = complete_title_evidence(xml(markup, **kwargs), r, WITNESS)
    return VerifiedPaperIdentities(p)


@pytest.mark.parametrize("title", ["A (18)F study", "A 18F study", "A 18 F study"])
def test_actual_superscript_has_complete_supported_renderings(title):
    c = claim(pmid="1", title=title, year=2021); before = deepcopy(c)
    assert evidenced().resolve(c)["status"] == "verified"
    assert c == before


@pytest.mark.parametrize("title", ["A 19F study", "A F study", "A study", "A 18F study does not work", "A 18F", "A (18)F study in children"])
def test_words_numbers_negation_and_suffix_are_never_discarded(title):
    assert evidenced().resolve(claim(pmid="1", title=title))["status"] == "conflict"


def test_arbitrary_parentheses_not_erased_and_only_real_boundaries_get_spaces():
    r = evidenced("A (18)F study", "A (18)F study")
    assert r.resolve(claim(pmid="1", title="A 18F study"))["status"] == "conflict"
    assert evidenced().resolve(claim(pmid="1", title="A18Fstudy"))["status"] == "conflict"


def test_nested_markup_and_tail_preserved():
    keys = title_renderings('<ArticleTitle>Not <i>α<sub>2</sub></i>-related: complete tail.</ArticleTitle>')
    assert presentation_key("Not α(2)-related: complete tail") in keys
    assert presentation_key("Not α2-related: complete tail") in keys
    assert presentation_key("α2-related: complete tail") not in keys


@pytest.mark.parametrize("a,b", [("Alzheimer’s disease","Alzheimer's disease"), ("error‐processing","error-processing"),
    ("obsessive&amp;ndash;compulsive","obsessive-compulsive"), ("&amp;#x3b1;","α")])
def test_bounded_presentation_encoding(a,b):
    assert presentation_key(a) == presentation_key(b)


@pytest.mark.parametrize("a,b", [("α disease","a disease"), ("5–10","5-10"), ("A − B","A - B"),
    ("A′ B","A' B"), ("not increased","increased"), ("left hippocampus","hippocampus"), ("A—B","A-B")])
def test_semantic_punctuation_and_scope_do_not_collapse(a,b):
    assert presentation_key(a) != presentation_key(b)


def test_old_registry_stays_strict_without_new_authority_witness():
    r = registry(source_record(title="A (18)F study"))
    assert r.resolve(claim(pmid="1", title="A 18F study"))["status"] == "conflict"


@pytest.mark.parametrize("override", [{"pmid":"2"}, {"doi":"10.9999/different"}, {"arxiv_id":"2106.03052"}, {"year":2025}])
def test_title_witness_never_overrides_id_year_or_version_guards(override):
    c = claim(pmid="1", doi="10.1234/example", title="A 18F study")
    c["source_paper"].update(override)
    assert evidenced().resolve(c)["status"] == "conflict"


def test_title_only_still_unverified():
    assert evidenced().resolve(claim(title="A 18F study"))["status"] == "unverified"


@pytest.mark.parametrize("kwargs,match", [({"pmid":"2"},"PMID"), ({"doi":"10.9999/different"},"identifiers")])
def test_owning_xml_ids_required(kwargs,match):
    with pytest.raises(ValueError, match=match): evidenced(**kwargs)


def test_book_document_owns_its_title_and_pmids():
    r = evidenced(book=True)
    assert r.resolve(claim(pmid="1", title="A 18F study"))["status"] == "verified"
    assert r.export_payload()["records"]["1"]["complete_title_evidence"]["document_kind"] == "book_document"


@pytest.mark.parametrize("markup", ["A <img>18</img>F study", "A <inline-formula>18</inline-formula>F study", ""])
def test_unsupported_or_empty_title_is_held(markup):
    with pytest.raises(ValueError): evidenced(markup=markup)


def test_altered_witness_rejected_on_registry_reload():
    p = evidenced().export_payload()
    p["records"]["1"]["complete_title_evidence"]["title_xml"] = "<ArticleTitle>A different study</ArticleTitle>"
    with pytest.raises(ValueError, match="complete titles"): VerifiedPaperIdentities(p)


def test_missing_response_hash_rejected():
    p = evidenced().export_payload()
    p["records"]["1"]["complete_title_evidence"]["witness"]["response_sha256"] = ""
    with pytest.raises(ValueError, match="hash"): VerifiedPaperIdentities(p)


def test_references_cannot_supply_owning_doi():
    r = evidenced().export_payload()["records"]["1"]["complete_title_evidence"]
    assert r["doi"] == ["10.1234/example"]


def test_complete_witness_survives_actual_save_and_reload(tmp_path):
    from neurooracle.tests.test_claim_evidence_identity import graph
    from neurooracle.src.storage import save_graph, load_graph
    kg = graph(); kg.paper_identities = evidenced()
    path = save_graph(kg, tmp_path / "kg.json")
    loaded = load_graph(path)
    assert loaded.paper_identities.resolve(claim(pmid="1", title="A 18 F study"))["status"] == "verified"
    assert loaded.paper_identities.resolve(claim(pmid="1", title="A F study"))["status"] == "conflict"


def test_rendering_equivalent_source_dedup_preserves_different_observations():
    from neurooracle.tests.test_claim_evidence_identity import claim as scientific_claim
    from neurooracle.src.claim_evidence_identity import evidence_dedup_key
    a = scientific_claim(pmid="1").to_dict()
    a["source_paper"] = dict(pmid="1", title="A 18F study")
    b = deepcopy(a); b.update(id="CLM:second"); b["source_paper"] = dict(doi="10.1234/example", title="A (18)F study")
    p = evidenced()
    assert evidence_dedup_key(a, papers=p) == evidence_dedup_key(b, papers=p)
    for field,value in (("negated",True), ("raw_text","Different raw observation."), ("conditions",{"age":0})):
        assert evidence_dedup_key(a, papers=p) != evidence_dedup_key({**b,field:value}, papers=p)
