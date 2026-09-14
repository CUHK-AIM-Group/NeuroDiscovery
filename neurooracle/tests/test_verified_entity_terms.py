from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from neurooracle.src.verified_entity_terms import (VERSION,VerifiedEntityTerms,atomic_term_proof,
    approve_live_terms,endpoint_eligibility,target_contract,load_candidates)
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.umls_audit_store import compact_record
from neurooracle.src.umls_mention_mapping import normalize_term
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode,Edge
from neurooracle.src.storage import load_graph,save_graph,save_display_graph
from neurooracle.src.relation_evidence import group_claim_evidence


def proof_fixture(name="bipolar disorder",compound=True):
    text="anxiety and "+name if compound else name
    atom=dict(id="CLM_ATOM:a",preferred_name=name,semantic_types=["T048"],metadata={
        "mapping_status":"auto_accepted_exact","mapping_count":1,"source_text":text,
        "source_mention_id":"CLM_CONCEPT:parent","atomization_rule":"top_level_composite_split" if compound else "full_mention",
        "evidence_span":dict(field="preferred_name",start=len(text)-len(name),end=len(text),text=name)})
    edge=dict(source_id=atom["id"],target_id="CUI:test",relation_type="maps_to",metadata={
        "review_status":"auto_accepted_exact","semantic_compatibility":"compatible",
        "candidate_overflow":False,"ambiguous_best_cui_count":1,"matched_term":name,
        "lookup_variant":dict(rule="normalized_exact",source_field="preferred_name",normalized=normalize_term(name))})
    parent=dict(id="CLM_CONCEPT:parent",preferred_name=text)
    target=dict(id="CUI:test",preferred_name="Bipolar Disorder",semantic_types=["T048"],domain_tags=["disease"])
    return atom,edge,parent,target


def registry_fixture():
    atom,edge,parent,target=proof_fixture()
    proof,reason=atomic_term_proof(atom,[("a",edge)])
    assert reason is None
    ca=compact_record(atom,"atoms",atom["id"]); ce=compact_record(edge,"mappings","a")
    nodes={r["id"]:r for r in (ca,parent,target)}
    terms,_=approve_live_terms({proof["name"]:[proof]},nodes,{proof["mapping_ref"]:digest(ce)})
    return VerifiedEntityTerms(dict(version=VERSION,terms=list(terms.values()))),nodes,[ce]


def claim(name="bipolar disorder",**changes):
    return dict(id="CLM:one",subject_id="IM:x",subject_name="hippocampal volume",predicate="predicts",
        object_id="CUI:test",object_name=name,source_paper={"pmid":"1"},negated=False,
        evidence={"direction":"negative","p_value":0},**changes)


def test_compound_atom_proof_only_resolves_whole_other_endpoint():
    registry,_,_=registry_fixture(); term=next(iter(registry.entries.values()))
    assert endpoint_eligibility(claim(),"object",term) is None
    assert endpoint_eligibility(claim("anxiety and bipolar disorder"),"object",term)=="not_whole_endpoint"
    assert endpoint_eligibility(claim("left bipolar disorder"),"object",term)=="not_whole_endpoint"


@pytest.mark.parametrize("typ",["disease","OUTCOME","FINDING","CLINICAL_FINDING","psychiatric_disorder"])
def test_clinical_identity_preserves_different_study_roles(typ):
    registry,_,_=registry_fixture(); original=claim(object_type=typ); before=deepcopy(original)
    assert registry.term_for(original,"object") is not None
    assert original==before


@pytest.mark.parametrize("typ",["drug","gene_target","method","not-a-type"])
def test_known_conflict_or_unknown_type_is_not_name_inferred(typ):
    registry,_,_=registry_fixture()
    assert registry.term_for(claim(object_type=typ),"object") is None


def test_inner_outer_types_and_ids_are_both_checked():
    registry,_,_=registry_fixture()
    assert registry.term_for(claim(object_type="disease",metadata={"object_type":"drug"}),"object") is None
    assert registry.term_for(claim(object_type="disease",metadata={"object_type":"OUTCOME"}),"object")
    assert registry.term_for(claim(metadata={"object_id":"wrong"}),"object") is None


@pytest.mark.parametrize("tui,roles,safe",[("T048",["disease"],True),("T121",["drug"],True),
    ("T023",["imaging_marker"],True),("T033",["individual_data","outcome"],False),
    ("T201",["individual_data","outcome"],True)])
def test_tui_not_accumulated_domain_tags_controls_missing_type(tui,roles,safe):
    _,_,_,target=proof_fixture(); target.update(semantic_types=[tui],domain_tags=["drug","gene_target","disease"])
    contract=target_contract(target,"survival time")
    assert contract["target_roles"]==roles and contract["missing_type_safe"] is safe


@pytest.mark.parametrize("tui",["T028","T085","T114","T999"])
def test_molecular_or_unrecognized_target_held(tui):
    _,_,_,target=proof_fixture(); target["semantic_types"]=[tui]
    assert target_contract(target,"a specific named target") is None


def test_finding_not_promoted_to_disease_or_drug():
    _,_,_,target=proof_fixture(); target["semantic_types"]=["T121"]
    term=dict(name="test medication",**target_contract(target,"test medication"))
    assert endpoint_eligibility(dict(object_name="test medication",object_type="FINDING"),"object",term)
    target["semantic_types"]=["T033"]; term=dict(name="test finding",**target_contract(target,"test finding"))
    assert endpoint_eligibility(dict(object_name="test finding"),"object",term)=="missing_type_for_contextual_concept"


@pytest.mark.parametrize("name",["risk score","treatment response","survival change"])
def test_scope_dependent_concept_missing_type_held(name):
    _,_,_,target=proof_fixture(); target["semantic_types"]=["T201"]
    assert target_contract(target,name)["missing_type_safe"] is False


@pytest.mark.parametrize("change",["span","overflow","stripped","review","ambiguous"])
def test_atomic_proof_rejects_unverified_or_modified_lookup(change):
    atom,edge,_,_=proof_fixture()
    if change=="span": atom["metadata"]["evidence_span"]["start"]=0
    if change=="overflow": edge["metadata"]["candidate_overflow"]=True
    if change=="stripped": edge["metadata"]["lookup_variant"]["rule"]="leading_modifier_stripped_exact"
    if change=="review": edge["metadata"]["review_status"]="needs_review"
    if change=="ambiguous": edge["metadata"]["ambiguous_best_cui_count"]=2
    assert atomic_term_proof(atom,[("a",edge)])[0] is None


def test_global_term_ambiguity_includes_nonaccepted_other_atom():
    atom,edge,_,_=proof_fixture(); second=deepcopy(atom); second["id"]="CLM_ATOM:b"
    second["metadata"]["mapping_status"]="needs_review"
    other=deepcopy(edge); other.update(source_id=second["id"],target_id="CUI:other")
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE atoms(payload_json TEXT,in_core INT,ordinal INT)")
    db.execute("CREATE TABLE mappings(record_id TEXT,payload_json TEXT,source_id TEXT,ordinal INT)")
    for i,(a,e) in enumerate(((atom,edge),(second,other))):
        db.execute("INSERT INTO atoms VALUES (?,1,?)",(json.dumps(a),i))
        db.execute("INSERT INTO mappings VALUES (?,?,?,?)",(str(i),json.dumps(e),a["id"],i))
    terms,counts=load_candidates(db); db.close()
    assert terms=={} and counts["globally_ambiguous_surface"]==1


def test_canonical_group_retains_papers_negation_and_scopes():
    registry,_,_=registry_fixture(); term=deepcopy(next(iter(registry.entries.values())))
    term["name"]="manic depressive disorder"
    registry=VerifiedEntityTerms(dict(version=VERSION,terms=[*registry.entries.values(),term]))
    a=claim(); b=claim("manic depressive disorder"); b.update(id="CLM:two",source_paper={"pmid":"2"},negated=True)
    before=deepcopy([a,b]); rows=list(group_claim_evidence([a,b],identities=registry))
    assert len(rows)==1 and rows[0]["paper_count"]==2 and rows[0]["evidence_variant_count"]==2
    assert [a,b]==before
    b["object_id"]="unverified"
    assert len(list(group_claim_evidence([a,b],identities=registry)))==2


def test_label_hold_keeps_existing_name_group_consistent():
    registry,_,_=registry_fixture(); term=deepcopy(next(iter(registry.entries.values()))); term["canonicalize"]=False
    registry=VerifiedEntityTerms(dict(version=VERSION,terms=[term]))
    a=claim(object_type="disease"); b=claim(object_type="drug"); b["id"]="CLM:two"
    assert registry.relation_key(a)==registry.relation_key(b)
    assert registry.term_for(a,"object") and registry.term_for(b,"object") is None


def saved_fixture(tmp_path):
    registry,nodes,edges=registry_fixture(); path=tmp_path/"graph.json"; terms=tmp_path/"terms.json"
    raw=json.dumps(dict(version=VERSION,terms=list(registry.entries.values()))).encode()
    terms.write_bytes(raw)
    data=dict(metadata={"entity_identity":dict(version=VERSION,registry=terms.name,sha256=hashlib.sha256(raw).hexdigest())},concepts=nodes,edges=edges)
    path.write_text(json.dumps(data),encoding="utf-8")
    return path,terms,data


def test_load_save_reload_keeps_verified_ingestion_routing(tmp_path):
    from neurooracle.src.claim_ingestion import resolve_entity
    path,_,_=saved_fixture(tmp_path); kg=load_graph(path)
    assert resolve_entity(kg,"bipolar disorder","OUTCOME")=="CUI:test"
    kg.add_concept(ConceptNode(id="CLM:new",preferred_name="new evidence",metadata=claim()))
    assert kg.relation_identities is not None
    destination=tmp_path/"saved.json"; save_graph(kg,destination); reloaded=load_graph(destination)
    assert resolve_entity(reloaded,"bipolar disorder","")=="CUI:test"
    assert resolve_entity(reloaded,"bipolar disorder","gene_target")!="CUI:test"
    save_graph(reloaded,destination)
    assert len(list(tmp_path.glob("saved.json.entity_terms.*.json")))==1
    assert "relation_evidence" not in json.loads(destination.read_text())["metadata"]


@pytest.mark.parametrize("change",["registry","node","mapping"])
def test_load_rejects_tampered_identity_proofs(tmp_path,change):
    path,terms,data=saved_fixture(tmp_path)
    if change=="registry": terms.write_bytes(terms.read_bytes()+b" ")
    if change=="node": data["concepts"]["CUI:test"]["preferred_name"]="wrong identity"
    if change=="mapping": data["edges"][0]["target_id"]="wrong"
    if change!="registry": path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_graph(path)


@pytest.mark.parametrize("mutation",["api_node","api_mapping","private_node","private_mapping"])
def test_changed_proof_invalidates_registry_or_fails_export(tmp_path,mutation):
    path,_,_=saved_fixture(tmp_path); kg=load_graph(path)
    if mutation=="api_node": kg.add_concept(ConceptNode(id="CUI:test",preferred_name="Bipolar Disorder",aliases=["new"]))
    if mutation=="api_mapping": kg.add_edge(Edge("CLM_ATOM:a","CUI:test","maps_to"))
    if mutation=="private_node": kg.get_concept("CUI:test").preferred_name="changed"
    if mutation=="private_mapping": kg.G.edges["CLM_ATOM:a","CUI:test"]["confidence"]=0
    if mutation.startswith("api_"): assert kg.relation_identities is None
    else:
        with pytest.raises(ValueError): save_graph(kg,tmp_path/"bad.json")


def test_display_projection_does_not_advertise_missing_identity_bundle(tmp_path):
    path,_,_=saved_fixture(tmp_path); kg=load_graph(path); out=tmp_path/"display.json"
    save_display_graph(kg,out)
    assert "entity_identity" not in json.loads(out.read_text())["metadata"]
