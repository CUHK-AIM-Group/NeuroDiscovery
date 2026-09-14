from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sqlite3
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from apply_kg_literal_endpoint_repair import LiteralTransform
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from plan_kg_literal_endpoint_repair import expected_shared
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import literal_node,reviewed_claim,reviewed_edges,reverse_claim,apply_edge
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges


def fixture(science=True):
    row=claim(); node=literal_node(row["metadata"]["subject_name"])
    event,out=reviewed_claim(row,[dict(side="subject",old_id=row["metadata"]["subject_id"],target_id=node["id"],name=node["preferred_name"])])
    refs=edges(row) if science else edges(row)[:2]
    plan=dict(events=[event],edge_events=reviewed_edges(row["id"],row,out,refs),
        new_literals=[dict(id=node["id"],name=node["preferred_name"],node_sha256=digest(node))],existing_targets={})
    source=[("metadata",None,{}),("node","CUI:C1421437",dict(id="CUI:C1421437",preferred_name="VCP")),
        ("node","CUI:disease",dict(id="CUI:disease",preferred_name="anxiety")),("node",row["id"],row),
        *[("edge",ordinal,edge) for ordinal,edge in refs]]
    return source,plan


@pytest.mark.parametrize("science",[True,False])
def test_entire_stream_and_independent_catalog_validation(tmp_path,science):
    source,plan=fixture(science); transform=LiteralTransform(plan)
    candidate=list(transform.records(iter(source)))
    old_nodes=[r for kind,key,r in source if kind=="node"]
    nodes=[r for kind,key,r in candidate if kind=="node"]
    assert len(nodes)==len(old_nodes)+1 and nodes[:2]==old_nodes[:2]
    assert len([1 for kind,_,_ in candidate if kind=="edge"])==(3 if science else 2)
    out=BytesIO(); catalog=tmp_path/"catalog.jsonl"
    result=stream_patch(iter(candidate),out,[],{},catalog)
    serialized=json.loads(out.getvalue())
    reread=[("metadata",None,serialized["metadata"]),
        *[("node",key,value) for key,value in serialized["concepts"].items()],
        *[("edge",i,edge) for i,edge in enumerate(serialized["edges"],1)]]
    checks=verify_catalog_and_graph(iter(reread),result,catalog)
    assert checks["shared_relation_index_complete"]
    assert result["counts"]==dict(nodes=4,claims=1,edges=3 if science else 2)
    assert result["self_loops"]==0
    inverse={k:hashlib.sha256() for k in ("nodes","edges")}
    by_edge={e["ordinal"]:e for e in plan["edge_events"]}
    for kind,key,row in candidate:
        if key==plan["new_literals"][0]["id"]: continue
        if kind=="node" and key=="CLM:test": row=reverse_claim(row,plan["events"][0])
        if kind=="edge" and key in by_edge: row=apply_edge(row,by_edge[key],reverse=True)
        if kind!="metadata": inverse[kind+"s"].update(compact(row).encode()+b"\n")
    assert {k:v.hexdigest() for k,v in inverse.items()}=={k:v.hexdigest() for k,v in transform.digests.items()}


def test_literal_id_collision_blocks_stream():
    source,plan=fixture(); node=literal_node(plan["new_literals"][0]["name"])
    source.insert(1,("node",node["id"],node))
    with pytest.raises(Exception,match="collides"): list(LiteralTransform(plan).records(iter(source)))


def test_unreviewed_edge_is_not_silently_ignored():
    source,plan=fixture(); duplicate=deepcopy(source[-1]); duplicate=("edge",4,duplicate[2]); source.append(duplicate)
    with pytest.raises(Exception,match="reference closure"): list(LiteralTransform(plan).records(iter(source)))


def test_shared_membership_recomputed_after_identity_change():
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE claims(cid TEXT,relation_id TEXT)")
    db.executemany("INSERT INTO claims VALUES (?,?)",[("a","old1"),("b","old2"),("c","unaffected"),("d","unaffected")])
    changes={"a":dict(old_relation_id="old1",new_relation_id="new"),"b":dict(old_relation_id="old2",new_relation_id="new")}
    assert expected_shared(db,changes,[{"members":[{"claim_id":"c"},{"claim_id":"d"}]}])==["a","b","c","d"]
    db.close()


def test_shared_membership_can_shrink_without_deleting_claims():
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE claims(cid TEXT,relation_id TEXT)")
    db.executemany("INSERT INTO claims VALUES (?,?)",[("a","old"),("b","old")])
    assert expected_shared(db,{"a":dict(old_relation_id="old",new_relation_id="new")},
        [{"members":[{"claim_id":"a"},{"claim_id":"b"}]}])==[]
    assert db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]==2
    db.close()
