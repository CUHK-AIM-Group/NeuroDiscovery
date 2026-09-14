"""Bibliographic XML parsing only: mocked HTTP, no extraction/model/KG writes."""
from copy import deepcopy
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from neurooracle.src.ad_paper_fetcher import _parse_pubmed_xml_element
from neurooracle.src.batch_extract import _fetch_pubmed_details


TITLES = [
    ("A <i>specific</i> protein predicts outcomes, not remission.", "A specific protein predicts outcomes, not remission."),
    ("<i>Complete</i> title follows the first tag.", "Complete title follows the first tag."),
    ("Hybrid <sup>13</sup>N-ammonia PET/MRI in coronary artery disease.", "Hybrid 13N-ammonia PET/MRI in coronary artery disease."),
    ("<i><sup>18</sup>F</i> PET and <sub>β</sub>-amyloid: no benefit.", "18F PET and β-amyloid: no benefit."),
    ("Genetic targets highlight <i>ACE</i> as a candidate in Alzheimer's disease.", "Genetic targets highlight ACE as a candidate in Alzheimer's disease."),
    ("A &amp; B: preserved negation and punctuation.", "A & B: preserved negation and punctuation."),
    ("  Multiple\n whitespace   segments.  ", "Multiple whitespace segments."),
    ("", ""),
]


def xml(title):
    return f'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
    <ArticleTitle>{title}</ArticleTitle><Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
    <Abstract><AbstractText Label="RESULTS">No <i>benefit</i> was found in the second cohort.</AbstractText></Abstract>
    </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1234/example</ArticleId></ArticleIdList></PubmedData>
    </PubmedArticle></PubmedArticleSet>'''.encode("utf-8")


@pytest.mark.parametrize("title,expected", TITLES)
def test_element_parser_keeps_full_title_tail_and_nested_text(title, expected):
    element = ElementTree.fromstring(xml(title)).find("PubmedArticle")
    before = ElementTree.tostring(element)
    abstract, paper = _parse_pubmed_xml_element(element)
    assert paper.title == expected
    assert paper.pmid == "123" and paper.doi == "10.1234/example"
    assert abstract == "RESULTS: No benefit was found in the second cohort."
    assert ElementTree.tostring(element) == before


@pytest.mark.parametrize("title,expected", TITLES)
def test_batch_fetch_parser_keeps_full_title_without_real_http(monkeypatch, title, expected):
    import requests
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, content=xml(title), raise_for_status=lambda:None)
    monkeypatch.setattr(requests, "get", get)
    def forbidden(*args, **kwargs): raise AssertionError("Unexpected network or model path")
    monkeypatch.setattr(requests, "post", forbidden)
    records = _fetch_pubmed_details(["123"])
    assert len(records) == 1 and records[0][1].title == expected
    assert records[0][0] == "RESULTS: No benefit was found in the second cohort."
    assert len(calls) == 1 and calls[0][1]["params"]["id"] == "123"


def test_correct_title_reaches_cache_and_reload_without_refetch(monkeypatch):
    import requests
    stored = {}
    class Cache:
        def get_many(self, ids):
            return [deepcopy(stored[x]) for x in ids if x in stored], [x for x in ids if x not in stored]
        def put_many(self, rows):
            for pmid, abstract, paper in rows: stored[pmid] = (abstract, deepcopy(paper))
            return len(rows)
        def __len__(self): return len(stored)
    title, expected = TITLES[4]
    monkeypatch.setattr(requests, "get", lambda *a, **k:SimpleNamespace(status_code=200, content=xml(title), raise_for_status=lambda:None))
    cache = Cache(); first = _fetch_pubmed_details(["123"], cache)
    assert first[0][1].title == expected and stored["123"][1].title == expected
    def forbidden(*args, **kwargs): raise AssertionError("Cache hit attempted external call")
    monkeypatch.setattr(requests, "get", forbidden)
    assert _fetch_pubmed_details(["123"], cache)[0][1].title == expected


def test_element_parser_own_electronic_doi_fallback():
    element = ElementTree.fromstring(xml("Plain title.")).find("PubmedArticle")
    element.find("PubmedData").clear()
    location = ElementTree.SubElement(element.find("./MedlineCitation/Article"), "ELocationID", EIdType="doi", ValidYN="Y")
    location.text = "10.1234/owned"
    assert _parse_pubmed_xml_element(element)[1].doi == "10.1234/owned"
    location.set("ValidYN", "N")
    assert _parse_pubmed_xml_element(element)[1].doi == ""


@pytest.mark.parametrize("parser", ["element", "batch"])
def test_cited_article_doi_never_assigned_to_this_article(monkeypatch, parser):
    import requests
    root = ElementTree.fromstring(xml("Plain title.")); element = root.find("PubmedArticle")
    data = element.find("PubmedData"); data.clear()
    reference = ElementTree.SubElement(ElementTree.SubElement(data,"ReferenceList"), "Reference")
    item = ElementTree.SubElement(ElementTree.SubElement(reference,"ArticleIdList"),"ArticleId", IdType="doi")
    item.text = "10.1234/cited-paper"
    if parser == "element": paper = _parse_pubmed_xml_element(element)[1]
    else:
        monkeypatch.setattr(requests, "get", lambda *a,**k:SimpleNamespace(status_code=200, content=ElementTree.tostring(root), raise_for_status=lambda:None))
        paper = _fetch_pubmed_details(["123"])[0][1]
    assert paper.pmid == "123" and paper.doi == ""
