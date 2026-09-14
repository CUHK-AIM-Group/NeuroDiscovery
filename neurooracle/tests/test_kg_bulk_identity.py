from copy import deepcopy
import pytest

from neurooracle.src.kg_bulk_identity import mapping_proof, validate_entities, endpoint_decision, change_claim, change_edge, digest
from neurooracle.src.kg_identity_pilot import nonidentity_claim


def records():
    name="bipolar disorder"
    atom=dict(id="CLM_ATOM:a",preferred_name=name,semantic_types=["T048"],metadata={
        "mapping_status":"auto_accepted_exact","mapping_count":1,"source_text":name,
        "source_mention_id":"CLM_CONCEPT:a","atomization_rule":"full_mention",
        "evidence_span":dict(field="preferred_name",start=0,end=len(name),text=name)})
    edge=dict(source_id=atom["id"],target_id="CUI:C0005586",metadata={
        "review_status":"auto_accepted_exact","semantic_compatibility":"compatible",
        "candidate_overflow":False,"ambiguous_best_cui_count":1,"matched_term":name,
        "lookup_variant":dict(rule="normalized_exact",source_field="preferred_name",normalized=name)})
    parent=dict(id="CLM_CONCEPT:a",preferred_name=name)
    target=dict(id=edge["target_id"],preferred_name=name,semantic_types=["T048"],domain_tags=["disease"])
    return atom,edge,parent,target


def test_full_surface_previously_accepted_proof():
    atom,edge,parent,target=records()
    proof,reason=mapping_proof(atom,[edge])
    assert reason is None
    assert validate_entities(proof,parent,target,atom) is None
    assert endpoint_decision(dict(object_id=parent["id"],object_name=parent["preferred_name"],
                                 metadata={"object_type":"disease"}),"object",proof) is None


@pytest.mark.parametrize("field,value",[("atomization_rule","top_level_composite_split"),("mapping_count",2),
    ("mapping_status","needs_review"),("source_text","left bipolar disorder")])
def test_partial_or_unreviewed_atoms_rejected(field,value):
    atom,edge,_,_=records(); atom["metadata"][field]=value
    assert mapping_proof(atom,[edge])[0] is None


@pytest.mark.parametrize("field,value",[("candidate_overflow",True),("semantic_compatibility","unknown"),
    ("review_status","needs_review"),("ambiguous_best_cui_count",2),("matched_term","disorder")])
def test_ambiguous_or_incompatible_mapping_rejected(field,value):
    atom,edge,_,_=records(); edge["metadata"][field]=value
    assert mapping_proof(atom,[edge])[0] is None


def test_alias_modifier_and_multiple_mappings_rejected():
    atom,edge,_,_=records()
    assert mapping_proof(atom,[edge,deepcopy(edge)])[0] is None
    edge["metadata"]["lookup_variant"]["rule"]="leading_modifier_stripped_exact"
    assert mapping_proof(atom,[edge])[0] is None


@pytest.mark.parametrize("name,typ,inner",[("bipolar disorder subtype","disease",{}),
    ("bipolar disorder","outcome",{}),("bipolar disorder","",{}),
    ("bipolar disorder","disease",{"object_id":"other"}),
    ("Bipolar disorder","disease",{})])
def test_names_roles_nested_ids_and_case_guarded(name,typ,inner):
    atom,edge,parent,target=records(); proof,_=mapping_proof(atom,[edge]); validate_entities(proof,parent,target,atom)
    assert endpoint_decision(dict(object_id=parent["id"],object_name=name,object_type=typ,metadata=inner),"object",proof)


def event_fixture():
    rec=dict(id="CLM:1",metadata=dict(id="CLM:1",subject_id="oldA",object_id="oldB",
        subject_name="A",object_name="B",predicate="predicts",negated=True,
        metadata={"subject_id":"oldA","population":{"age":0}},evidence={"p_value":0,"extra":False}))
    event=dict(claim_id=rec["id"],claim_sha256=digest(rec),changes=[
        dict(side="subject",old_id="oldA",target_id="newA"),dict(side="object",old_id="oldB",target_id="newB")])
    return rec,event


def test_both_endpoint_changes_preserve_every_other_field():
    rec,event=event_fixture(); before=deepcopy(rec); out=change_claim(rec,event)
    assert rec==before and out["metadata"]["subject_id"]=="newA" and out["metadata"]["object_id"]=="newB"
    assert nonidentity_claim(out)==nonidentity_claim(rec)
    assert out["metadata"]["metadata"]["subject_id"]=="newA"


def test_science_and_both_about_edges_retargeted_only_for_owner():
    _,event=event_fixture(); events={"CLM:1":event}
    edge=dict(source_id="oldA",target_id="oldB",relation_type="predicts",metadata={"claim_id":"CLM:1","negated":True})
    out=change_edge(edge,events)
    assert (out["source_id"],out["target_id"])==("newA","newB")
    for old,new in [("oldA","newA"),("oldB","newB")]:
        assert change_edge(dict(source_id="CLM:1",target_id=old,relation_type="about"),events)["target_id"]==new
    unrelated=deepcopy(edge); unrelated["metadata"]["claim_id"]="CLM:2"
    assert change_edge(unrelated,events) is unrelated


def test_stale_claim_and_edge_conflict_fail_closed():
    rec,event=event_fixture(); rec["metadata"]["negated"]=False
    with pytest.raises(ValueError): change_claim(rec,event)
    with pytest.raises(ValueError): change_edge(dict(source_id="wrong",target_id="oldB",relation_type="predicts",metadata={"claim_id":"CLM:1"}),{"CLM:1":event})


def test_no_self_loop_creation():
    rec,event=event_fixture(); event["changes"][1]["target_id"]="newA"
    with pytest.raises(ValueError): change_claim(rec,event)


@pytest.mark.parametrize("tui,roles",[("T033",["disease"]),("T060",["imaging_feature"]),
    ("T081",["imaging_feature"]),("T109",["drug"])])
def test_accumulated_domain_tags_are_not_type_proof(tui,roles):
    atom,edge,parent,target=records()
    atom["semantic_types"]=[tui]; target["semantic_types"]=[tui]; target["domain_tags"]=roles
    proof,_=mapping_proof(atom,[edge])
    assert validate_entities(proof,parent,target,atom)=="target_TUI_does_not_confirm_role"


def test_full_stream_two_endpoint_batch_and_independent_index_validation(tmp_path):
    from io import BytesIO
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
    first,event=event_fixture()
    first["metadata"]["source_paper"]={"pmid":"1"}; event["claim_sha256"]=digest(first)
    second=deepcopy(first); second["id"]="CLM:2"; second["metadata"]["id"]="CLM:2"
    second["metadata"].update(subject_id="newA",object_id="newB",source_paper={"pmid":"2"})
    second["metadata"]["metadata"]["subject_id"]="newA"
    nodes={k:dict(id=k,preferred_name=k,metadata={}) for k in ("oldA","oldB","newA","newB")}
    nodes.update({first["id"]:first,second["id"]:second})
    edges=[]
    for cid,sid,oid in [("CLM:1","oldA","oldB"),("CLM:2","newA","newB")]:
        edges.extend([dict(source_id=cid,target_id=n,relation_type="about") for n in (sid,oid)])
        edges.append(dict(source_id=sid,target_id=oid,relation_type="predicts",metadata={"claim_id":cid}))
    records=[("metadata","",{})]+[("node",k,v) for k,v in nodes.items()]+[("edge",str(i),v) for i,v in enumerate(edges,1)]
    stream=BytesIO(); catalog=tmp_path/"index.jsonl"
    expected=stream_patch(iter(records),stream,[event],{k:digest(nodes[k]) for k in ("newA","newB")},catalog,
                          claim_transform=change_claim,edge_transform=change_edge)
    out=json.loads(stream.getvalue())
    records=[("metadata","",out["metadata"])]+[("node",k,v) for k,v in out["concepts"].items()]+[("edge",str(i),v) for i,v in enumerate(out["edges"],1)]
    assert verify_catalog_and_graph(iter(records),expected,catalog)["shared_relation_index_complete"]
    assert expected["changed"]==dict(claim_nodes=1,edge_records=3)
    assert expected["catalog_stats"]["multi_paper_shared_groups"]==1
