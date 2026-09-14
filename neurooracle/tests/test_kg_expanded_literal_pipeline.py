from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_kg_expanded_literal_repair import LiteralTransform
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from neurooracle.src import kg_expanded_literal_repair as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_expanded_literal_repair import fixture
from neurooracle.tests.test_kg_literal_endpoint_repair import edges


def stream_fixture(science=True,double=False):
    row,gene,witness,review,change=fixture();genes={gene['id']:gene};witnesses={gene['id']:witness}
    reviews={repair.review_key(row['id'],'subject'):review};changes=[change]
    if double:
        other=dict(id='CUI:C1538994',preferred_name='MENT',semantic_types=['T028'],aliases=[],metadata={})
        genes[other['id']]=other;md=row['metadata']
        md.update(object_id=other['id'],object_name='emotional improvement',object_type='OUTCOME')
        md['metadata']['object_id']=other['id']
        witnesses[other['id']]=dict(node_id=other['id'],name='MENT',labels=['MENT'],semantic_types=['T028'],node_sha256=digest(other))
        reviews[repair.review_key(row['id'],'object')]=dict(review,side='object',name=md['object_name'],current_node_id=other['id'],
            source_gene_node_sha256=digest(other),word_interior_alias_hits=['ment'],outer_type='OUTCOME',declared_roles=['outcome'])
        changes.append(dict(side='object',old_id=other['id'],name=md['object_name'],target_id=repair.literal_node(md['object_name'])['id']))
    else:genes['CUI:disease']=dict(id='CUI:disease',preferred_name='anxiety',metadata={})
    for r in reviews.values():r['claim_sha256']=digest(row)
    event,current=repair.reviewed_claim(row,changes,witnesses,reviews)
    owned=edges(row) if science else edges(row)[:2]
    literals=[dict(id=ch['target_id'],name=ch['name'],node_sha256=digest(repair.literal_node(ch['name']))) for ch in changes]
    plan=dict(events=[event],edge_events=repair.reviewed_edges(row['id'],row,current,owned),
        gene_witnesses=witnesses,endpoint_reviews=reviews,new_literals=literals,existing_targets={k:digest(v) for k,v in genes.items()})
    source=[('metadata',None,{}),*[('node',k,v) for k,v in genes.items()],('node',row['id'],row),*[('edge',o,e) for o,e in owned]]
    return source,plan


def inverse(records,plan):
    ev=plan['events'][0];edges_by={e['ordinal']:e for e in plan['edge_events']};new={r['id']:r for r in plan['new_literals']}
    hashes={k:hashlib.sha256() for k in ('nodes','edges')}
    for kind,key,row in records:
        if kind=='metadata':continue
        if kind=='node' and key in new:
            if row!=repair.literal_node(new[key]['name']):raise ValueError('new literal differs')
            continue
        if kind=='node' and key==ev['claim_id']:row=repair.reverse_claim(row,ev)
        elif kind=='edge' and key in edges_by:row=repair.apply_edge(row,edges_by[key],reverse=True)
        hashes[kind+'s'].update(compact(row).encode()+b'\n')
    return {k:h.hexdigest() for k,h in hashes.items()}


@pytest.mark.parametrize('science',[True,False])
@pytest.mark.parametrize('double',[True,False])
def test_stream_full_inverse_all_science_preserved(tmp_path,science,double):
    source,plan=stream_fixture(science,double);before=deepcopy(source);transform=LiteralTransform(plan)
    out=BytesIO();catalog=tmp_path/'catalog.jsonl'
    result=stream_patch(transform.records(iter(source)),out,[],{},catalog)
    graph=json.loads(out.getvalue())
    current=[('metadata',None,graph['metadata']),*[('node',k,v) for k,v in graph['concepts'].items()],*[('edge',i,e) for i,e in enumerate(graph['edges'],1)]]
    assert source==before and inverse(current,plan)=={k:h.hexdigest() for k,h in transform.digests.items()}
    assert verify_catalog_and_graph(iter(current),result,catalog)['shared_relation_index_complete']
    assert result['counts']==dict(nodes=sum(k=='node' for k,_,_ in source)+len(plan['new_literals']),claims=1,edges=3 if science else 2)
    assert current[-1][2].get('metadata',{}).get('negated',True) is True


@pytest.mark.parametrize('kind',['extra_reference','existing_mutated','new_collision','scope','wrong_gene'])
def test_forward_current_scope_is_not_loosened(kind):
    source,plan=stream_fixture()
    if kind=='extra_reference':source.append(('edge',4,deepcopy(source[-1][2])))
    elif kind=='existing_mutated':source[1][2]['preferred_name']='changed'
    elif kind=='new_collision':
        n=repair.literal_node(plan['new_literals'][0]['name']);source.insert(1,('node',n['id'],n))
    elif kind=='scope':plan['endpoint_reviews']={}
    else:plan['gene_witnesses'][source[1][1]]['labels']=['different']
    with pytest.raises(ValueError):list(LiteralTransform(plan).records(iter(source)))


@pytest.mark.parametrize('kind',['science','edge_evidence','new_type'])
def test_inverse_rejects_unapproved_values(kind):
    source,plan=stream_fixture();transform=LiteralTransform(plan);current=list(transform.records(iter(source)))
    if kind=='science':next(row for key,cid,row in current if cid=='CLM:test')['metadata']['evidence']={'p_value':0.01}
    elif kind=='edge_evidence':current[-1][2]['evidence_ref']='different paper'
    else:next(row for key,cid,row in current if cid==plan['new_literals'][0]['id'])['semantic_types']=['T028']
    with pytest.raises(ValueError):inverse(current,plan)
