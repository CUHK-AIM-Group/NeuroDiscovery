from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_kg_scientific_definition_repair import LiteralTransform
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from neurooracle.src import kg_scientific_definition_repair as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_scientific_definition_repair import fixture
from neurooracle.tests.test_kg_literal_endpoint_repair import edges


def stream_fixture(cid,science=True):
    row,reviews,proof=fixture(cid);event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews)
    owned=edges(row);owned[-1][1]['metadata']['negated']=False
    if not science:owned=owned[:2]
    nodes={row['metadata'][s+'_id']:dict(id=row['metadata'][s+'_id'],preferred_name=row['metadata'][s+'_name'],metadata={}) for s in ('subject','object')}
    new={}
    for ch in event['changes']:
        if ch['new_name'] in repair.NEW_NAMES:
            node=repair.literal_node(ch['new_name']);new[node['id']]=dict(id=node['id'],name=node['preferred_name'],node_sha256=digest(node))
        else:nodes[ch['target_id']]=dict(id=ch['target_id'],preferred_name=ch['new_name'],metadata={})
    plan=dict(events=[event],edge_events=repair.reviewed_edges(cid,row,current,owned),public_source_proof=proof,claim_reviews=reviews,
        new_literals=list(new.values()),existing_targets={nid:digest(n) for nid,n in nodes.items()})
    source=[('metadata',None,{}),*[('node',nid,n) for nid,n in nodes.items()],('node',cid,row),*[('edge',o,e) for o,e in owned]]
    return source,plan


def inverse(records,plan):
    event=plan['events'][0];ed={e['ordinal']:e for e in plan['edge_events']};new={r['id']:r for r in plan['new_literals']};hashes={k:hashlib.sha256() for k in ('nodes','edges')}
    for kind,key,row in records:
        if kind=='metadata':continue
        if kind=='node' and key in new:
            if row!=repair.literal_node(new[key]['name']):raise ValueError('new node differs')
            continue
        if kind=='node' and key==event['claim_id']:row=repair.reverse_claim(row,event)
        elif kind=='edge' and key in ed:row=repair.apply_edge(row,ed[key],reverse=True)
        hashes[kind+'s'].update(compact(row).encode()+b'\n')
    return {k:h.hexdigest() for k,h in hashes.items()}


@pytest.mark.parametrize('cid',sorted(repair.SPECS))
@pytest.mark.parametrize('science',[True,False])
def test_full_scientific_source_inverse_and_real_catalog(tmp_path,cid,science):
    source,plan=stream_fixture(cid,science);before=deepcopy(source);t=LiteralTransform(plan);out=BytesIO();catalog=tmp_path/'catalog.jsonl'
    result=stream_patch(t.records(iter(source)),out,[],{},catalog);g=json.loads(out.getvalue())
    current=[('metadata',None,g['metadata']),*[('node',k,v) for k,v in g['concepts'].items()],*[('edge',i,e) for i,e in enumerate(g['edges'],1)]]
    assert source==before and inverse(current,plan)=={k:h.hexdigest() for k,h in t.digests.items()}
    assert result['counts']==dict(nodes=sum(k=='node' for k,_,r in source)+len(plan['new_literals']),claims=1,edges=3 if science else 2)
    assert verify_catalog_and_graph(iter(current),result,catalog)['shared_relation_index_complete']


@pytest.mark.parametrize('kind',['extra_reference','target_changed','public_proof','claim_review'])
def test_forward_rejects_stale_science_or_identity(kind):
    source,plan=stream_fixture(repair.ACC)
    if kind=='extra_reference':source.append(('edge',4,deepcopy(source[-1][2])))
    elif kind=='target_changed':source[1][2]['preferred_name']='new scope'
    elif kind=='public_proof':plan['public_source_proof']={}
    else:plan['claim_reviews']={}
    with pytest.raises(ValueError):list(LiteralTransform(plan).records(iter(source)))


@pytest.mark.parametrize('kind',['raw','audit','stats','edge_source','new_type'])
def test_independent_inverse_rejects_nonapproved_values(kind):
    source,plan=stream_fixture(repair.BACE);current=list(LiteralTransform(plan).records(iter(source)))
    claim=next(r for k,c,r in current if c==repair.BACE)
    if kind=='raw':claim['metadata']['raw_text']='changed original'
    elif kind=='audit':claim['metadata']['scope_reaudit']['decision']='revalidated'
    elif kind=='stats':claim['metadata']['evidence']['sample_size']=100
    elif kind=='edge_source':current[-1][2]['source']='another paper'
    else:next(r for k,c,r in current if c==plan['new_literals'][0]['id'])['semantic_types']=['T028']
    with pytest.raises(ValueError):inverse(current,plan)
