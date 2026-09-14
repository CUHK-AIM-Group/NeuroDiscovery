from pathlib import Path
from io import BytesIO
import hashlib
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import apply_kg_literal_endpoint_repair as engine
import apply_kg_clinical_literal_reuse as adapter
from apply_kg_relation_identity import stream_patch
from build_umls_simplification_candidate import compact
from neurooracle.src import kg_clinical_literal_reuse as r
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import reviewed_edges,apply_edge
from neurooracle.tests.test_kg_clinical_literal_reuse import family,claim,changes,NAME


def test_configuration_does_not_replay_old_paths_or_mutate_at_import():
    assert engine.OUTPUT.name=='round37_relation_scope'
    conf=adapter.configuration();assert conf['OUTPUT'].name=='round39_literal_duplicates'
    assert conf['TEMP'].name.endswith('.clinical.tmp') and conf['change_claim'] is r.change_claim


@pytest.mark.parametrize('old',[None,'CUI:C1421437'])
def test_all_source_nodes_and_detail_anchors_retained_with_inverse_digest(tmp_path,monkeypatch,old):
    nodes,_=family();row=claim(old);event,out=r.reviewed_claim(row,changes(row));cid=row['id']
    refs=[(1,dict(source_id=cid,target_id=row['metadata']['subject_id'],relation_type='about',metadata={})),
          (2,dict(source_id=cid,target_id='CUI:disease',relation_type='about',metadata={}))]
    plan=dict(events=[event],edge_events=reviewed_edges(cid,row,out,refs),new_literals=[],
        existing_targets={nid:digest(n) for nid,n in nodes.items()})
    source=[('metadata',None,{}),*[('node',nid,n) for nid,n in nodes.items()],
        ('node','CUI:C1421437',dict(id='CUI:C1421437',preferred_name='VCP')),
        ('node','CUI:disease',dict(id='CUI:disease',preferred_name='schizophrenia')),('node',cid,row),
        *[('edge',n,e) for n,e in refs]]
    monkeypatch.setattr(engine,'reviewed_claim',r.reviewed_claim);monkeypatch.setattr(engine,'change_claim',r.change_claim)
    transform=engine.LiteralTransform(plan);candidate=list(transform.records(iter(source)))
    result=stream_patch(iter(candidate),BytesIO(),[],{},tmp_path/'catalog.jsonl')
    assert result['counts']==dict(nodes=5,claims=1,edges=2)
    assert {key:digest(n) for kind,key,n in candidate if kind=='node' and key in nodes}==plan['existing_targets']
    inverse={k:hashlib.sha256() for k in ('nodes','edges')};by_edge={e['ordinal']:e for e in plan['edge_events']}
    for kind,key,value in candidate:
        if kind=='metadata':continue
        if kind=='node' and key==cid:value=r.verified_reverse(value,event)
        if kind=='edge' and key in by_edge:value=apply_edge(value,by_edge[key],reverse=True)
        inverse[kind+'s'].update(compact(value).encode()+b'\n')
    assert {k:h.hexdigest() for k,h in inverse.items()}=={k:h.hexdigest() for k,h in transform.digests.items()}


def test_capabilities_published_before_adoption(tmp_path,monkeypatch):
    monkeypatch.setattr(adapter,'OUTPUT',tmp_path)
    monkeypatch.setattr(adapter,'_validate',lambda:adapter.j.atomic_json(tmp_path/'VALIDATED.json',dict(checks={})))
    adapter.validate();checks=adapter.j.read_json(tmp_path/'VALIDATED.json')['checks']
    assert all(checks[k] for k in ('verified_identity_proofs_complete','paper_identity_witnesses_validated',
        'publication_status_witnesses_validated','all_source_anchor_nodes_and_detail_rows_preserved'))

