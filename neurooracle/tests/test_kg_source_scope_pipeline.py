from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_kg_source_scope_resolution import ResolutionTransform,IndependentResolution,project_issue_rows
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from neurooracle.src import kg_source_scope_resolution as scope
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_source_scope_resolution import myelin,frailty,bind,reuse_fixture
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges


def fixture(monkeypatch,viral=True,reuse=False):
    q,qe,qp=myelin(monkeypatch);f,fe,fp=frailty(monkeypatch)
    v=claim('VCP');v['id']=scope.VIRUS
    v['metadata'].update(id=v['id'],object_name='uptake in all retinal layers followed by clearance within 1-3 hours',
        raw_text='After VCP was injected into the eye, it was taken up in all layers of the retina but was cleared within 1-3 hours of delivery.')
    vp=bind(monkeypatch,v,'29534078');proof={**qp,**fp,**vp}
    fe=[(o+2,e) for o,e in fe];ve=[(o+8,e) for o,e in edges(v)]
    for _,e in fe+ve:
        if 'negated' in e.get('metadata',{}):e['metadata']['negated']=False
    removed=scope.reviewed_myelin_removal(q,qe,proof);retired=scope.reviewed_frailty_retirements(f,fe,proof)
    node=scope.viral_node();witness=None
    if reuse:
        node,incidents,sources=reuse_fixture(monkeypatch);witness=scope.reviewed_viral_reuse(node,incidents,sources)
    ev,updated=scope.reviewed_viral_identity(v,proof,node['id'],witness)
    existing={nid:dict(id=nid,preferred_name=nid) for nid in {
        'CUI:C1421437','CUI:disease','CLM_CONCEPT:white_matter_hyperintensity_burden',
        'CLM_CONCEPT:frailty_09cfc9267861','CLM_CONCEPT:white_matter_hyperintensity_burden_e206a534fc08','CLM_CONCEPT:frailty'}}
    if reuse:existing[node['id']]=node
    plan=dict(public_source_proof=proof,deleted_claims=[removed],retired_branch_edges=retired,viral_target_witness=witness,
        removed_edge_ordinals=sorted([r['ordinal'] for r in removed['owned_edges']]+[r['ordinal'] for r in retired]),
        events=[ev] if viral else [],edge_events=scope.reviewed_edges(scope.VIRUS,v,updated,ve) if viral else [],
        new_literals=[dict(id=node['id'],name=node['preferred_name'],node_sha256=digest(node))] if viral and not reuse else [],
        existing_targets={nid:digest(n) for nid,n in existing.items()},frailty_claim_sha256=digest(f),
        frailty_current_canonical_edges=[dict(ordinal=o,edge_sha256=digest(e)) for o,e in fe[:3]])
    source=[('metadata',None,{}),*[('node',n['id'],n) for n in [*existing.values(),q,f,v]],
        *[('edge',o,e) for o,e in qe+fe+ve]]
    return source,plan


@pytest.mark.parametrize('viral,reuse',[(True,False),(False,False),(True,True)])
def test_atomic_stream_and_independent_kept_record_validation(monkeypatch,tmp_path,viral,reuse):
    source,plan=fixture(monkeypatch,viral,reuse);before=deepcopy(source)
    transform=ResolutionTransform(plan);candidate=list(transform.records(iter(source)))
    assert source==before and all(k!=scope.MYELIN for _,k,_ in candidate)
    assert [key for kind,key,_ in candidate if kind=='edge']==list(range(1,7))
    expected={k:v.hexdigest() for k,v in transform.digests.items()}
    independent=IndependentResolution(plan,expected)
    assert list(independent.records(iter(candidate)))==candidate
    independent.verify()
    output=BytesIO();catalog=tmp_path/'catalog.jsonl'
    result=stream_patch(iter(candidate),output,[],{},catalog)
    serialized=json.loads(output.getvalue())
    reread=[('metadata',None,serialized['metadata']),*[('node',k,r) for k,r in serialized['concepts'].items()],
        *[('edge',i,r) for i,r in enumerate(serialized['edges'],1)]]
    assert verify_catalog_and_graph(iter(reread),result,catalog)['shared_relation_index_complete']
    assert result['counts']==dict(nodes=9 if viral else 8,claims=2,edges=6)


def test_other_exact_reference_to_deleted_claim_blocks(monkeypatch):
    source,plan=fixture(monkeypatch);source[1][2]['metadata']={'nested':{'arbitrary_ref':scope.MYELIN}}
    plan['existing_targets'][source[1][1]]=digest(source[1][2])
    with pytest.raises(Exception,match='exact reference'):list(ResolutionTransform(plan).records(iter(source)))


def test_unapproved_removed_edge_payload_blocks(monkeypatch):
    source,plan=fixture(monkeypatch)
    next(r for kind,key,r in source if kind=='edge' and key==6)['source']='new source'
    with pytest.raises(Exception,match='removed edge changed'):list(ResolutionTransform(plan).records(iter(source)))


def test_changed_deleted_claim_is_not_removed(monkeypatch):
    source,plan=fixture(monkeypatch)
    next(r for kind,key,r in source if key==scope.MYELIN)['metadata']['negated']=True
    with pytest.raises(ValueError,match='hash differs'):list(ResolutionTransform(plan).records(iter(source)))


def test_changed_kept_claim_science_fails_inverse(monkeypatch):
    source,plan=fixture(monkeypatch);transform=ResolutionTransform(plan);candidate=list(transform.records(iter(source)))
    next(r for kind,key,r in candidate if key==scope.VIRUS)['metadata']['conditions']['age']='children'
    inverse=IndependentResolution(plan,{k:v.hexdigest() for k,v in transform.digests.items()})
    with pytest.raises(ValueError,match='output changed'):list(inverse.records(iter(candidate)))


def test_changed_unrelated_node_fails_full_kept_digest(monkeypatch):
    source,plan=fixture(monkeypatch)
    source.insert(1,('node','UNRELATED',dict(id='UNRELATED',metadata={})))
    transform=ResolutionTransform(plan);candidate=list(transform.records(iter(source)))
    next(r for kind,key,r in candidate if key=='UNRELATED')['metadata']['new']='unexpected'
    inverse=IndependentResolution(plan,{k:v.hexdigest() for k,v in transform.digests.items()})
    list(inverse.records(iter(candidate)))
    with pytest.raises(Exception,match='kept source record digests'):inverse.verify()


def test_new_viral_literal_collision_blocks(monkeypatch):
    source,plan=fixture(monkeypatch);n=scope.viral_node();source.insert(1,('node',n['id'],n))
    with pytest.raises(Exception,match='collides'):list(ResolutionTransform(plan).records(iter(source)))


def test_current_hold_ordinals_reindex_but_historical_ordinals_do_not():
    rows=[dict(claim_id='CLM:held',related_edge_ordinals=[3,7,9],owned_science_ordinals=[3],about_ordinals=[7,9],r12_related_edge_ordinals=[13,17,19])]
    before=deepcopy(rows);out=project_issue_rows(rows,{'sha256':'new'},[1,2,5])
    assert rows==before
    assert out[0]['related_edge_ordinals']==[1,4,6] and out[0]['about_ordinals']==[4,6]
    assert out[0]['r12_related_edge_ordinals']==[13,17,19]


def test_hold_cannot_keep_removed_edge_position():
    with pytest.raises(ValueError,match='removed ordinal'):
        project_issue_rows([dict(claim_id='CLM:held',about_ordinals=[2])],{},[2])
