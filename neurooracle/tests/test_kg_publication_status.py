from copy import deepcopy
import json
from xml.etree import ElementTree as ET

import pytest

from neurooracle.src.kg_publication_status import adjudicate_publication_status, make_publication_review, owning_status_projection
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.shared_relation_catalog import find_shared_relations
from neurooracle.tests.test_kg_paper_authorities import registry, source_record, claim
from neurooracle.tests.test_shared_relation_catalog import setup_files, fp


def projection(pmid, types, relation, target):
    return dict(pmid=pmid,publication_types=types,links=[dict(relation=relation,pmid=target)],
        witness=dict(response_sha256="a"*64,url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="+pmid))


def retraction(notice_type="Retraction Notice"):
    return make_publication_review(projection("1",["Journal Article","Retracted Publication"],"RetractionIn","9"),
        [projection("9",[notice_type],"RetractionOf","1")])


def correction(forward="RepublishedFrom",reverse="RepublishedIn"):
    return make_publication_review(projection("1",["Corrected and Republished Article"],forward,"9"),
        [projection("9",["Journal Article"],reverse,"1")])


def reviewed(review=None):
    p=registry(source_record(),source_record("2",doi="10.1234/other")).export_payload()
    p["publication_reviews"]={"1":review or retraction()}
    return VerifiedPaperIdentities(p)


@pytest.mark.parametrize("notice",["Retraction Notice","Retraction of Publication"])
def test_both_authoritative_notice_type_names(notice):
    assert retraction(notice)["status"]=="retracted"


@pytest.mark.parametrize("forward,reverse",[("RepublishedFrom","RepublishedIn"),
    ("CorrectedandRepublishedFrom","CorrectedandRepublishedIn"), ("RetractedandRepublishedFrom","RetractedandRepublishedIn")])
def test_corrected_version_is_not_its_retracted_or_uncorrected_origin(forward,reverse):
    assert correction(forward,reverse)["status"]=="corrected_republished"


@pytest.mark.parametrize("change",["no_backlink","wrong_target","wrong_link_type","notice_without_type","no_source_retraction_type","no_notice","self_link","unknown_id"])
def test_publication_status_needs_owning_bidirectional_evidence(change):
    r=retraction(); source, linked=deepcopy(r["source"]),deepcopy(r["linked"])
    if change=="no_backlink": linked[0]["links"]=[]
    if change=="wrong_target": linked[0]["links"][0]["pmid"]="2"
    if change=="wrong_link_type": linked[0]["links"][0]["relation"]="RepublishedFrom"
    if change=="notice_without_type": linked[0]["publication_types"]=["Published Erratum"]
    if change=="no_source_retraction_type": source["publication_types"]=["Journal Article"]
    if change=="no_notice": linked=[]
    if change=="self_link": source["links"][0]["pmid"]="1"
    if change=="unknown_id": source["links"][0]["pmid"]=""
    with pytest.raises(ValueError): adjudicate_publication_status(source,linked)


def test_plain_reprint_without_correction_publication_type_is_not_corrected():
    r=correction(); r["source"]["publication_types"]=["Journal Article"]
    with pytest.raises(ValueError): adjudicate_publication_status(r["source"],r["linked"])


@pytest.mark.parametrize("field,value",[("response_sha256",""),("url","https://untrusted.test/notice")])
def test_missing_or_wrong_witness_rejected(field,value):
    r=retraction(); r["source"]["witness"][field]=value
    with pytest.raises(ValueError): reviewed(r)


def test_unsupported_registry_status_cannot_silently_count_as_good():
    r=retraction(); r["status"]="good"
    with pytest.raises(ValueError,match="not reproduced"): reviewed(r)


def test_identity_negation_conditions_and_original_payload_stay_unchanged():
    c=claim(pmid="1",title="A complete scientific title"); original=deepcopy(c)
    r=reviewed()
    assert r.resolve(c)==registry().resolve(c)
    assert r.resolve(c)["status"]=="verified"
    assert r.publication_review("pmid:1")["status"]=="retracted" and c==original
    assert r.publication_review("pmid:2")["status"]=="not_reviewed"
    assert r.publication_review("doi:10.1234/example")["status"]=="not_reviewed"
    exported=r.publication_review("pmid:1"); exported["status"]="fake"
    assert r.publication_review("pmid:1")["status"]=="retracted"


def test_status_parser_uses_only_owning_metadata_not_cited_article():
    a=ET.fromstring('<PubmedArticle><MedlineCitation><PMID>1</PMID><Article><PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList></Article><CommentsCorrectionsList><CommentsCorrections RefType="CommentIn"><PMID>8</PMID></CommentsCorrections></CommentsCorrectionsList></MedlineCitation><PubmedData><ReferenceList><Reference><MedlineCitation><PMID>2</PMID><Article><PublicationTypeList><PublicationType>Retracted Publication</PublicationType></PublicationTypeList></Article></MedlineCitation></Reference></ReferenceList></PubmedData></PubmedArticle>')
    p=owning_status_projection(a,retraction()["source"]["witness"])
    assert p["pmid"]=="1" and p["publication_types"]==["Journal Article"] and p["links"]==[]


def query_files(tmp_path,review=None,validated=True):
    graph,catalog,receipt,campaign,control=setup_files(tmp_path)
    members=[dict(claim_id="CLM:1",paper_key="pmid:1",paper_status="verified",negated=True),
        dict(claim_id="CLM:2",paper_key="pmid:2",paper_status="verified",negated=False),
        dict(claim_id="CLM:3",paper_key="pmid:1",paper_status="verified",negated=False)]
    row=json.loads(catalog.read_text()); row.update(members=members,claim_count=3,verified_paper_count=2)
    catalog.write_text(json.dumps(row)+"\n")
    papers=tmp_path/"papers.json"; papers.write_text(json.dumps(reviewed(review).export_payload()))
    accepted=json.loads(receipt.read_text()); accepted.update(shared_relations=fp(catalog),paper_identities=fp(papers))
    accepted["checks"].update(paper_identity_witnesses_validated=True,publication_status_witnesses_validated=validated)
    receipt.write_text(json.dumps(accepted)); control.update(current_shared_relations=fp(catalog),current_paper_identities=fp(papers),current_acceptance=fp(receipt))
    campaign.write_text(json.dumps(control))
    return campaign,catalog


def test_default_query_excludes_retracted_source_only_from_threshold(tmp_path):
    campaign,catalog=query_files(tmp_path); before=catalog.read_bytes()
    assert find_shared_relations(campaign,minimum_papers=2)==[]
    row=find_shared_relations(campaign,minimum_papers=1)[0]
    assert row["verified_paper_count"]==2 and row["counted_paper_count"]==1
    assert row["excluded_retracted_paper_keys"]==["pmid:1"] and len(row["members"])==3
    assert {m["negated"] for m in row["members"]}=={True,False}
    assert catalog.read_bytes()==before


def test_explicit_inclusion_retains_auditable_all_source_view(tmp_path):
    campaign,_=query_files(tmp_path)
    row=find_shared_relations(campaign,minimum_papers=2,include_retracted=True)[0]
    assert row["counted_paper_count"]==2 and row["excluded_retracted_paper_keys"]==[]


def test_corrected_versions_not_blanket_excluded(tmp_path):
    campaign,_=query_files(tmp_path,correction())
    assert find_shared_relations(campaign,minimum_papers=2)[0]["counted_paper_count"]==2


def test_unvalidated_publication_registry_fails_closed(tmp_path):
    campaign,_=query_files(tmp_path,validated=False)
    with pytest.raises(ValueError,match="unvalidated publication"): find_shared_relations(campaign)


@pytest.mark.parametrize("value",[None,0,1,"true",[],{}])
def test_retraction_inclusion_must_be_explicit_boolean(tmp_path,value):
    campaign,_=query_files(tmp_path)
    with pytest.raises(ValueError,match="boolean"): find_shared_relations(campaign,include_retracted=value)


def test_unknown_and_legacy_registries_do_not_invent_healthy_status():
    r=registry()
    assert not r.has_publication_reviews
    assert r.publication_review("pmid:1")=={"status":"not_reviewed"}


def test_actual_save_reload_and_relation_iteration_retain_evidence(tmp_path):
    from neurooracle.tests.test_claim_evidence_identity import graph, claim as scientific_claim, ingest
    from neurooracle.src.storage import save_graph,load_graph
    kg=graph();kg.paper_identities=reviewed()
    a=scientific_claim(pmid="1"); ingest(kg,[a]); b=scientific_claim("CLM:second",pmid="2"); b.negated=True; ingest(kg,[b])
    path=save_graph(kg,tmp_path/"kg.json"); loaded=load_graph(path)
    assert loaded.paper_identities.publication_review("pmid:1")["status"]=="retracted"
    group=list(loaded.iter_relation_evidence())[0]
    assert group["claim_count"]==2 and group["verified_paper_count"]==2
    assert {m["negated"] for m in group["members"]}=={True,False}
