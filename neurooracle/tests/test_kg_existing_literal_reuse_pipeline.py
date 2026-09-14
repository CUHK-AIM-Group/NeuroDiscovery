from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_kg_existing_literal_reuse import LiteralTransform
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from neurooracle.src import kg_existing_literal_reuse as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_existing_literal_reuse import fixture
from neurooracle.tests.test_kg_literal_endpoint_repair import edges


def stream_fixture(science=True):
    row,gene,witness,review,ch,target=fixture()
    event,current=repair.reviewed_claim(row,[ch],{gene['id']:witness},{repair.review_key(row['id'],'subject'):review})
    owned=edges(row) if science else edges(row)[:2]
    disease=dict(id='CUI:disease',preferred_name='anxiety',metadata={});nodes={n['id']:n for n in (gene,target,disease)}
    plan=dict(events=[event],edge_events=repair.reviewed_edges(row['id'],row,current,owned),gene_witnesses={gene['id']:witness},
        endpoint_reviews={repair.review_key(row['id'],'subject'):review},new_literals=[],existing_targets={k:digest(n) for k,n in nodes.items()})
    source=[('metadata',None,{}),*[('node',k,n) for k,n in nodes.items()],('node',row['id'],row),*[('edge',o,e) for o,e in owned]]
    return source,plan


def inverse(records,plan):
    ev=plan['events'][0];ed={e['ordinal']:e for e in plan['edge_events']};hashes={k:hashlib.sha256() for k in ('nodes','edges')}
    for kind,key,row in records:
        if kind=='metadata':continue
        if kind=='node' and key==ev['claim_id']:row=repair.reverse_claim(row,ev)
        elif kind=='edge' and key in ed:row=repair.apply_edge(row,ed[key],reverse=True)
        hashes[kind+'s'].update(compact(row).encode()+b'\n')
    return {k:h.hexdigest() for k,h in hashes.items()}


@pytest.mark.parametrize('science',[True,False])
def test_reuse_full_inverse_all_records_no_new_node(tmp_path,science):
    source,plan=stream_fixture(science);before=deepcopy(source);t=LiteralTransform(plan);out=BytesIO();catalog=tmp_path/'catalog.jsonl'
    result=stream_patch(t.records(iter(source)),out,[],{},catalog);g=json.loads(out.getvalue())
    current=[('metadata',None,g['metadata']),*[('node',k,v) for k,v in g['concepts'].items()],*[('edge',i,e) for i,e in enumerate(g['edges'],1)]]
    assert source==before and inverse(current,plan)=={k:h.hexdigest() for k,h in t.digests.items()}
    assert result['counts']==dict(nodes=4,claims=1,edges=3 if science else 2)
    assert verify_catalog_and_graph(iter(current),result,catalog)['shared_relation_index_complete']


@pytest.mark.parametrize('kind',['extra_reference','target_mutated','scope','wrong_gene'])
def test_forward_exact_scope_rejects_tampering(kind):
    source,plan=stream_fixture()
    if kind=='extra_reference':source.append(('edge',4,deepcopy(source[-1][2])))
    elif kind=='target_mutated':source[2][2]['metadata']['curation_scope']='different'
    elif kind=='scope':plan['endpoint_reviews']={}
    else:next(iter(plan['gene_witnesses'].values()))['labels']=['different']
    with pytest.raises(ValueError):list(LiteralTransform(plan).records(iter(source)))


@pytest.mark.parametrize('kind',['science','source'])
def test_inverse_rejects_changed_evidence(kind):
    source,plan=stream_fixture();current=list(LiteralTransform(plan).records(iter(source)))
    if kind=='science':next(r for k,c,r in current if c=='CLM:test')['metadata']['raw_text']='new assertion'
    else:current[-1][2]['evidence_ref']='another source'
    with pytest.raises(ValueError):inverse(current,plan)
