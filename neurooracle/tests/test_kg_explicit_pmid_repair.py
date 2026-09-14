from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import pytest

from neurooracle.src.kg_explicit_pmid_repair import (
    abstract_witness, reference_changes, apply_reference, masked_record, project_coverage)
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.metadata_field_audit import Coverage
from neurooracle.tests.test_kg_bibliography_repair import fixture as title_fixture, TITLE
from neurooracle.scripts.audit_kg_explicit_pmid_provenance import review_pmid, proposed_node


ABSTRACT = "The measured binding was not increased after treatment in the tested adult population. The complete study reported no improvement over control."
OTHER = "An independent bibliographic partner wrote this additional summary of the same published article."


def fixture(shape="missing"):
    row, registry, articles, witnesses = title_fixture()
    articles["123"].find("./MedlineCitation/Article/Abstract/AbstractText").text = ABSTRACT
    md = row["metadata"]
    md["source_paper"].update(pmid="" if shape == "missing" else "GENMEDSUPP:not-an-id", title=TITLE)
    md.update(pmid="123", paper_id="GENMEDSUPP:not-an-id", source="manual_abstract_review")
    md["metadata"].update(dataset="general_neuromed_manual_20260610", batch=1, input_idx=1, batch_index=1,
        source_snapshot_index=90, queue_id=md["paper_id"], script="curate_batch_1_manual.js")
    md["evidence"]["legacy_value"] = md["raw_text"]
    p = md["source_paper"]
    item = dict(pmid="123", queue_id=md["paper_id"], source_snapshot_index=90, abstract=ABSTRACT,
                **{k:p[k] for k in ("title", "year", "journal", "doi")})
    claim = dict(pmid="123", paper_id=md["paper_id"], raw_sentence=md["raw_text"], evidence=md["raw_text"],
                 evidence_text=md["raw_text"], metadata=dict(source_snapshot_index=90),
                 **{k:p[k] for k in ("title", "year", "journal", "doi")})
    class Archive:
        paths = {k:Path(v) for k,v in dict(script="curate_batch_1_manual.js", input="input.json", claims="claims.jsonl").items()}
        files = {str(v):dict(sha256="a"*64) for v in paths.values()}
        def load(self, batch):
            assert batch == 1
            return dict(inputs={1:item}, claims={1:[claim]}, paths=self.paths)
    return row, registry, articles, witnesses, Archive(), item, claim


def review(f):
    row, registry, articles, witnesses, archive, _, _ = f
    expected = dict(claim_sha256=digest(row), alternate_pmid=row["metadata"].get("pmid"))
    return review_pmid(row, expected, registry, articles, witnesses, archive)


@pytest.mark.parametrize("shape", ["missing", "queue"])
def test_explicit_provenance_moves_one_pmid_without_guessing_suffix(shape):
    f = fixture(shape); original = deepcopy(f[0]); event, reason = review(f)
    assert event and event["set_source_paper_pmid"] == "123"
    out = proposed_node(f[0], event)
    assert f[0] == original
    assert out["metadata"]["source_paper"]["pmid"] == "123" and "pmid" not in out["metadata"]
    assert masked_record("node", out, event) == masked_record("node", original, event)
    assert "raw_text" not in event and ABSTRACT not in str(event)
    assert f[1].resolve(out["metadata"])["status"] == "verified"


@pytest.mark.parametrize("change", ["explicit", "input_id", "claim_id", "queue", "locator", "script", "evidence", "year",
    "title", "journal", "doi", "another_pmid", "inner_id", "arxiv", "preprint", "full_abstract", "negation", "family"])
def test_provenance_mismatch_cannot_promote_a_claim(change):
    f = fixture(); row, _, _, _, _, item, claim = f; md = row["metadata"]
    if change == "explicit": md["pmid"] = "GENMEDSUPP:123"
    if change == "input_id": item["pmid"] = "456"
    if change == "claim_id": claim["pmid"] = "456"
    if change == "queue": item["queue_id"] = "different"
    if change == "locator": md["metadata"]["source_snapshot_index"] = 91
    if change == "script": md["metadata"]["script"] = "different.js"
    if change == "evidence": claim["evidence"] = "changed"
    if change in ("year", "title", "journal", "doi"): md["source_paper"][change] = "different"
    if change == "another_pmid": md["source_paper"]["pmid"] = "456"
    if change == "inner_id": md["metadata"]["pmid"] = "456"
    if change == "arxiv": md["source_paper"]["arxiv_id"] = "2001.00123"
    if change == "preprint":
        md["source_paper"]["journal"] = item["journal"] = claim["journal"] = "bioRxiv"
    if change == "full_abstract": item["abstract"] = ABSTRACT[:110]
    if change == "negation": item["abstract"] = ABSTRACT.replace("not ", "")
    if change == "family": md["metadata"]["dataset"] = "unreviewed"
    assert review(f)[0] is None


def test_wrong_owning_article_fails_closed():
    f = fixture(); f[2]["123"].find("./MedlineCitation/PMID").text = "456"
    with pytest.raises(ValueError): review(f)


def test_own_partner_abstract_exactly_explains_complete_import():
    f = fixture(); article = f[2]["123"]
    other = ET.SubElement(article.find("./MedlineCitation"), "OtherAbstract", Type="PIP", Language="eng")
    ET.SubElement(other, "AbstractText").text = OTHER
    f[-2]["abstract"] = ABSTRACT + " " + OTHER
    event, _ = review(f)
    assert event["abstract_witness"]["mode"] == "complete_primary_plus_own_other_abstract"
    assert event["abstract_witness"]["other_abstracts"] == [dict(type="PIP", language="eng")]


@pytest.mark.parametrize("bad", ["cited_other", "partial_other", "missing_primary", "unrelated_tail", "wrong_pmid"])
def test_other_abstract_does_not_broaden_identity_matching(bad):
    f = fixture(); article = f[2]["123"]
    parent = article.find("./MedlineCitation")
    if bad == "cited_other": parent = ET.SubElement(article, "References")
    other = ET.SubElement(parent, "OtherAbstract", Type="PIP")
    ET.SubElement(other, "AbstractText").text = OTHER
    text = ABSTRACT + " " + OTHER
    if bad == "partial_other": text = text[:-20]
    if bad == "missing_primary": text = OTHER * 2
    if bad == "unrelated_tail": text += " extra scientific claim"
    if bad == "wrong_pmid":
        with pytest.raises(ValueError): abstract_witness(article, text, "456")
    else: assert abstract_witness(article, text, "123") is None


@pytest.mark.parametrize("kind", ["doi", "queue", "about"])
def test_reference_normalization_keeps_extraction_provenance(kind):
    f = fixture("queue" if kind == "queue" else "missing"); row = f[0]; event, _ = review(f)
    source = {"doi":"claim:10.1234/test", "queue":"claim:GENMEDSUPP:not-an-id", "about":"claim_extraction"}[kind]
    edge = dict(source_id="A", target_id="B", relation_type="about" if kind == "about" else "affects", source=source,
        confidence=0.9, evidence_ref=TITLE, metadata=dict(claim_id=row["id"], negated=True))
    changes, reasons, holds = reference_changes(edge, row, event)
    assert not holds
    assert changes == ({} if kind == "about" else dict(source="claim:123"))


def test_unproven_edge_or_wrong_owner_is_not_normalized():
    f = fixture(); row = f[0]; event, _ = review(f)
    edge = dict(relation_type="affects", source="claim:10.1234/another", metadata=dict(claim_id=row["id"]))
    assert reference_changes(edge, row, event)[2]
    edge["metadata"]["claim_id"] = "CLM:another"
    with pytest.raises(ValueError): reference_changes(edge, row, event)


def test_only_exact_predecessor_title_is_allowed_on_edge():
    f = fixture(); row = f[0]; old = deepcopy(row); old["metadata"]["source_paper"]["title"] = "Old exact title"
    event = dict(source_node_sha256=digest(old))
    edge = dict(relation_type="about", source="claim_extraction", evidence_ref="Old exact title", metadata=dict(claim_id=row["id"]))
    changes, bases, holds = reference_changes(edge, row, title_event=event)
    assert changes == dict(evidence_ref=TITLE) and not holds
    edge["evidence_ref"] = "Old exact"
    assert reference_changes(edge, row, title_event=event)[2]


@pytest.mark.parametrize("change", ["pmid", "science", "edge_field", "edge_sha"])
def test_tampered_transform_or_mask_rejected(change):
    f = fixture(); event, _ = review(f); row = f[0]
    if change == "pmid":
        event["set_source_paper_pmid"] = "456"
        with pytest.raises((ValueError, RuntimeError, AssertionError)): proposed_node(row, event)
    if change == "science":
        out = proposed_node(row, event); out["metadata"]["negated"] = False
        assert masked_record("node", out, event) != masked_record("node", row, event)
    if change.startswith("edge"):
        edge = dict(source="old", source_id="A", metadata={})
        changed = dict(source="new") if change == "edge_sha" else dict(source_id="B")
        ev = dict(source_edge_sha256=digest(edge), current_edge_sha256="0"*64, set_fields=changed)
        with pytest.raises(ValueError): apply_reference(edge, ev)


def test_coverage_delta_drops_only_redundant_field_and_updates_nonempty():
    f = fixture(); event, _ = review(f); before = f[0]; after = proposed_node(before, event)
    baseline, minus, plus, actual = (Coverage() for _ in range(4))
    for cv, record in ((baseline,before),(minus,before),(plus,after),(actual,after)):
        cv.add("node/claim",record)
    assert project_coverage(baseline.rows(),minus,plus) == actual.rows()
    assert not any(r["field"] == "metadata.pmid" for r in actual.rows())
    assert next(r for r in actual.rows() if r["field"] == "metadata.source_paper.pmid")["nonempty"] == 1


def test_coverage_different_denominator_rejected():
    a, b = Coverage(), Coverage(); a.add("node/claim",dict(metadata={}))
    with pytest.raises(ValueError): project_coverage(a.rows(),a,b)
