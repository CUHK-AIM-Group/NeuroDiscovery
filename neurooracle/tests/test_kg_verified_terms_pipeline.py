from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from apply_kg_verified_terms import SourceReview,proof_hashes,rebind_issue
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from neurooracle.src.kg_structure_repairs import choose_repairs,repair_edge
from neurooracle.src.kg_bulk_identity import change_claim,change_edge
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.metadata_field_audit import Coverage,node_class
from neurooracle.tests.test_verified_entity_terms import registry_fixture,claim


def summarized(edge,ordinal):
    return dict(ordinal=ordinal,edge_sha256=digest(edge),source_id=edge["source_id"],target_id=edge["target_id"],
        relation_type=edge["relation_type"],claim_id=(edge.get("metadata") or {}).get("claim_id"),
        nonendpoint_sha256=digest({k:v for k,v in edge.items() if k not in {"source_id","target_id"}}))


def graph_fixture():
    registry,nodes,mappings=registry_fixture()
    old="CLM_CONCEPT:old"; unnamed="CLM_CONCEPT:unnamed_da39a3ee5e6b"
    for nid,name in (("IM:x","hippocampal volume"),("IM:alt","hippocampal volume"),(old,"bipolar disorder"),(unnamed,"unnamed")):
        nodes[nid]=dict(id=nid,preferred_name=name,metadata={})
    a=claim(); a["object_id"]=old
    b=claim(); b.update(id="CLM:two",source_paper={"pmid":"2"})
    for md in (a,b): nodes[md["id"]]=dict(id=md["id"],preferred_name=md["id"],metadata=md)
    edges=list(mappings)
    for md in (a,b):
        edges.extend([dict(source_id=md["id"],target_id=n,relation_type="about") for n in (md["subject_id"],md["object_id"])])
        edges.append(dict(source_id=md["subject_id"],target_id=md["object_id"],relation_type=md["predicate"],
            metadata={"claim_id":md["id"],"negated":False,"population":{"age":0}}))
    edges[2]["target_id"]=unnamed
    edges.extend([dict(source_id=b["id"],target_id="IM:alt",relation_type="about"),
                  {**deepcopy(edges[6]),"source_id":"IM:alt"}])
    reviews=[]
    for md in (a,b):
        owned=[]
        for i,e in enumerate(edges,1):
            owner=e["source_id"] if e["relation_type"]=="about" else (e.get("metadata") or {}).get("claim_id")
            if owner==md["id"]: owned.append(summarized(e,i))
        reviews.append(dict(claim_id=md["id"],claim_sha256=digest(nodes[md["id"]]),edges=owned,
            **{k:md[k] for k in ("subject_id","subject_name","predicate","object_id","object_name")}))
    ns={nid:dict(name=n["preferred_name"],record_sha256=digest(n)) for nid,n in nodes.items() if not nid.startswith("CLM:")}
    repairs=choose_repairs(reviews,ns)
    event=dict(claim_id=a["id"],claim_sha256=digest(nodes[a["id"]]),changes=[dict(side="object",old_id=old,target_id="CUI:test")])
    plan=dict(events=[event],structural_repairs=repairs,structure_review=reviews,structure_target_nodes=ns)
    records=[("metadata","",{})]+[("node",k,v) for k,v in nodes.items()]+[("edge",str(i),v) for i,v in enumerate(edges,1)]
    return registry,plan,records,nodes,edges


def test_structural_choices_require_exact_same_owner_payloads():
    _,plan,_,_,_=graph_fixture(); p=plan["structural_repairs"]
    assert p["retargeted_edges"]==1 and p["removed_duplicate_edges"]==2 and len(p["repaired_claim_ids"])==2


@pytest.mark.parametrize("change",["different_name","different_polarity","different_owner","different_population"])
def test_duplicate_path_not_deleted_when_scientific_scope_differs(change):
    _,plan,_,nodes,edges=graph_fixture(); review=deepcopy(plan["structure_review"][1])
    ns=deepcopy(plan["structure_target_nodes"])
    if change=="different_name": ns["IM:alt"]["name"]="left hippocampal volume"
    else:
        edge=deepcopy(edges[-1])
        edge["metadata"][{"different_polarity":"negated","different_owner":"claim_id","different_population":"population"}[change]]="different"
        review["edges"][-1]=summarized(edge,len(edges))
    assert choose_repairs([review],ns)["repaired_claim_ids"]==[]


def test_wrong_about_owner_is_not_repaired():
    _,plan,_,_,_=graph_fixture(); review=deepcopy(plan["structure_review"][0]); review["edges"][0]["source_id"]="CLM:other"
    assert choose_repairs([review],plan["structure_target_nodes"])["edits"]==[]


def test_stale_deleted_or_surviving_edge_hash_rejected():
    registry,plan,records,_,edges=graph_fixture()
    edit=next(e for e in plan["structural_repairs"]["edits"] if e["action"]=="remove_same_owner_duplicate")
    with pytest.raises(ValueError): repair_edge({**edges[edit["ordinal"]-1],"confidence":0},edit)
    modified=deepcopy(records)
    for kind,key,row in modified:
        if kind=="edge" and int(key)==edit["keep_ordinal"]: row["confidence"]=0
    review=SourceReview(plan,registry)
    with pytest.raises((ValueError,RuntimeError),match="survivor changed"): list(review.records(iter(modified)))
    review.db.close()


def test_full_stream_combined_repair_index_and_independent_validation(tmp_path):
    registry,plan,records,nodes,edges=graph_fixture(); review=SourceReview(plan,registry)
    handle=BytesIO(); catalog=tmp_path/"shared.jsonl"
    expected=stream_patch(review.records(iter(records)),handle,plan["events"],proof_hashes(registry,plan),catalog,
        claim_transform=change_claim,edge_transform=change_edge,relation_key_func=registry.relation_key,
        relation_version="kg.relation_evidence.v2",extra_graph_metadata={"entity_identity":{"version":"test"}})
    assert expected["counts"]["edges"]==len(edges)-2
    assert review.removed==2 and review.unique_changed==2
    comparison=review.comparison(); review.db.close()
    assert comparison["existing_raw_groups_newly_split"]==0
    assert comparison["before_raw"]["all_fine_grained_relation_groups"]==2
    assert comparison["after_canonical"]["multi_paper_shared_groups"]==1
    data=json.loads(handle.getvalue()); coverage=Coverage()
    actual=[("metadata","",data["metadata"])]+[("node",k,v) for k,v in data["concepts"].items()]+[("edge",str(i),v) for i,v in enumerate(data["edges"],1)]
    def observed():
        for kind,key,row in actual:
            if kind!="metadata": coverage.add("edge/all" if kind=="edge" else "node/"+node_class(key,row),row)
            yield kind,key,row
    result=verify_catalog_and_graph(observed(),expected,catalog,relation_key_func=registry.relation_key)
    assert result["shared_relation_index_complete"] and result["dangling_endpoints"]==0
    assert coverage.denominators["edge/all"]==len(edges)-2
    assert data["metadata"]["relation_evidence"]["version"]=="kg.relation_evidence.v2"
    assert data["concepts"]["CLM:one"]["metadata"]["evidence"]==nodes["CLM:one"]["metadata"]["evidence"]


def test_registry_mapping_hash_required_in_stream():
    registry,plan,records,_,_=graph_fixture(); records=[r for r in records if not (r[0]=="edge" and r[1]=="1")]
    records=[(kind,str(int(key)-1),row) if kind=="edge" else (kind,key,row) for kind,key,row in records]
    # Removing a proof is rejected, even when node records remain intact.
    plan=deepcopy(plan)
    for item in plan["structure_review"]:
        for e in item["edges"]: e["ordinal"]-=1
    plan["structural_repairs"]=choose_repairs(plan["structure_review"],plan["structure_target_nodes"])
    review=SourceReview(plan,registry)
    with pytest.raises((ValueError,RuntimeError),match="incomplete source witnesses"): list(review.records(iter(records)))
    review.db.close()


def test_issue_ordinals_rebound_without_touching_historical_coordinates():
    issue=dict(claim_id="CLM:hold",related_edge_ordinals=[2,7,10],owned_science_ordinals=[7],about_ordinals=[2,10],r12_related_edge_ordinals=[99])
    original=deepcopy(issue); result=rebind_issue(issue,{"sha256":"new"},[3,8])
    assert result["related_edge_ordinals"]==[2,6,8] and result["r12_related_edge_ordinals"]==[99] and issue==original
    with pytest.raises((ValueError,RuntimeError)): rebind_issue(issue,{},[7])
