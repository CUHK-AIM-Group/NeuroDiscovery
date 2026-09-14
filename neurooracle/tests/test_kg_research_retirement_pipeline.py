from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_kg_research_statement_retirement import RetirementTransform,IndependentRetirement,project_issue_rows
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from neurooracle.src import kg_research_statement_retirement as scope
from neurooracle.tests.test_kg_research_retirement import research_claim


def fixture(monkeypatch):
    claims=[];refs=[];removed=[];proof={}
    for i,cid in enumerate(sorted(scope.REVIEWS)):
        row,owned,p=research_claim(monkeypatch,cid);proof.update(p)
        es=[(i*3+o,e) for o,e in owned];claims.append(row);refs.extend(es)
        removed.append(scope.reviewed_claim(row,es,p))
    entities={nid:dict(id=nid,preferred_name=nid,metadata={}) for row in claims for nid in (row['metadata']['subject_id'],row['metadata']['object_id'])}
    extra=dict(source_id=next(iter(entities)),target_id=list(entities)[1],relation_type='is_associated_with',metadata={})
    refs.append((13,extra))
    plan=dict(deleted_claims=removed,removed_edge_ordinals=list(range(1,13)),public_source_proof=proof)
    source=[('metadata',None,{}),*[('node',n['id'],n) for n in [*entities.values(),*claims]],*[('edge',o,e) for o,e in refs]]
    return source,plan


def test_full_kept_digest_actual_serialization_and_index_validation(monkeypatch,tmp_path):
    source,plan=fixture(monkeypatch);original=deepcopy(source)
    transform=RetirementTransform(plan);candidate=list(transform.records(iter(source)))
    assert source==original and [key for kind,key,_ in candidate if kind=='edge']==[1]
    inverse=IndependentRetirement(plan,{k:v.hexdigest() for k,v in transform.digests.items()})
    assert list(inverse.records(iter(candidate)))==candidate;inverse.verify()
    output=BytesIO();catalog=tmp_path/'catalog.jsonl';result=stream_patch(iter(candidate),output,[],{},catalog)
    parsed=json.loads(output.getvalue());again=[('metadata',None,parsed['metadata']),
        *[('node',key,row) for key,row in parsed['concepts'].items()],*[('edge',i,row) for i,row in enumerate(parsed['edges'],1)]]
    assert verify_catalog_and_graph(iter(again),result,catalog)['shared_relation_index_complete']
    assert result['counts']==dict(nodes=8,claims=0,edges=1)


def test_changed_deleted_claim_blocks(monkeypatch):
    source,plan=fixture(monkeypatch);next(r for k,i,r in source if i in scope.REVIEWS)['metadata']['negated']=True
    with pytest.raises(ValueError,match='deleted source claim'):list(RetirementTransform(plan).records(iter(source)))


def test_removed_edge_with_new_payload_blocks(monkeypatch):
    source,plan=fixture(monkeypatch);next(r for k,i,r in source if k=='edge' and i==1)['unexpected']=1
    with pytest.raises(ValueError,match='removed edge differs'):list(RetirementTransform(plan).records(iter(source)))


def test_exact_foreign_reference_blocks_source_transform(monkeypatch):
    source,plan=fixture(monkeypatch);source[1][2]['metadata']['foreign']=[next(iter(scope.REVIEWS))]
    with pytest.raises(Exception,match='exact reference'):list(RetirementTransform(plan).records(iter(source)))


def test_all_unrelated_kept_record_values_are_checked(monkeypatch):
    source,plan=fixture(monkeypatch);transform=RetirementTransform(plan);candidate=list(transform.records(iter(source)))
    expected={k:v.hexdigest() for k,v in transform.digests.items()};candidate[1][2]['metadata']['changed']=True
    inverse=IndependentRetirement(plan,expected);list(inverse.records(iter(candidate)))
    with pytest.raises(ValueError,match='kept source record digests'):inverse.verify()


def test_independent_candidate_reference_check(monkeypatch):
    source,plan=fixture(monkeypatch);transform=RetirementTransform(plan);candidate=list(transform.records(iter(source)))
    candidate[1][2]['metadata']['foreign']=next(iter(scope.REVIEWS))
    inverse=IndependentRetirement(plan,{k:v.hexdigest() for k,v in transform.digests.items()})
    with pytest.raises(Exception,match='exact reference'):list(inverse.records(iter(candidate)))


def test_only_removed_semantic_holds_drop_and_history_is_preserved():
    records=[dict(claim_id='gone',about_ordinals=[1,2]),dict(claim_id='kept',related_edge_ordinals=[3,7],r12_related_edge_ordinals=[10,20])]
    original=deepcopy(records);out=project_issue_rows(records,{'sha256':'new'},[1,2],{'gone'})
    assert records==original and len(out)==1
    assert out[0]['related_edge_ordinals']==[1,5] and out[0]['r12_related_edge_ordinals']==[10,20]


def test_unexpected_retired_ordinal_in_surviving_hold_blocks():
    with pytest.raises(ValueError,match='removed ordinal'):
        project_issue_rows([dict(claim_id='kept',about_ordinals=[2])],{},[2],set())
