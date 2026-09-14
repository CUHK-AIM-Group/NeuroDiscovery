from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from apply_kg_gene_boundary_repair import LiteralTransform
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import compact
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_gene_boundary_repair import literal_node,reviewed_claim,reviewed_edges,reverse_claim,apply_edge
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges


def fixture(science=True,reuse=False):
    row=claim('cortical surface area')
    gene=dict(id='CUI:C1414531',preferred_name='FANCE',aliases=['FACE'],semantic_types=['T028'])
    row['metadata']['subject_id']=gene['id']
    row['metadata']['metadata']['subject_id']=gene['id']
    witness=dict(node_id=gene['id'],node_sha256=digest(gene),name='FANCE',labels=['FANCE','FACE'],semantic_types=['T028'])
    witnesses={gene['id']:witness};node=literal_node(row['metadata']['subject_name'])
    event,out=reviewed_claim(row,[dict(side='subject',old_id=gene['id'],target_id=node['id'],name=node['preferred_name'])],witnesses)
    refs=edges(row) if science else edges(row)[:2]
    plan=dict(events=[event],edge_events=reviewed_edges(row['id'],row,out,refs),gene_witnesses=witnesses,
        new_literals=[] if reuse else [dict(id=node['id'],name=node['preferred_name'],node_sha256=digest(node))],
        existing_targets={gene['id']:digest(gene),**({node['id']:digest(node)} if reuse else {})})
    source=[('metadata',None,{}),('node',gene['id'],gene),
        ('node','CUI:disease',dict(id='CUI:disease',preferred_name='anxiety')),('node',row['id'],row)]
    if reuse:source.append(('node',node['id'],node))
    source.extend(('edge',ordinal,edge) for ordinal,edge in refs)
    return source,plan


@pytest.mark.parametrize('science',[True,False])
@pytest.mark.parametrize('reuse',[True,False])
def test_full_stream_and_independent_inverse_preserve_science(tmp_path,science,reuse):
    source,plan=fixture(science,reuse);before=deepcopy(source);transform=LiteralTransform(plan)
    candidate=list(transform.records(iter(source)))
    assert source==before
    original=next(row for kind,key,row in source if key=='CLM:test')
    changed=next(row for kind,key,row in candidate if key=='CLM:test')
    assert reverse_claim(changed,plan['events'][0])==original
    assert changed['metadata']['negated'] is True
    assert changed['metadata']['raw_text']==original['metadata']['raw_text']
    assert changed['metadata']['conditions']==original['metadata']['conditions']
    output=BytesIO();catalog=tmp_path/'catalog.jsonl'
    result=stream_patch(iter(candidate),output,[],{},catalog)
    serialized=json.loads(output.getvalue())
    reread=[('metadata',None,serialized['metadata']),
        *[('node',key,row) for key,row in serialized['concepts'].items()],
        *[('edge',i,row) for i,row in enumerate(serialized['edges'],1)]]
    assert verify_catalog_and_graph(iter(reread),result,catalog)['shared_relation_index_complete']
    assert result['counts']==dict(nodes=4,claims=1,edges=3 if science else 2)
    inverse={k:hashlib.sha256() for k in ('nodes','edges')}
    by_edge={e['ordinal']:e for e in plan['edge_events']}
    new_ids={r['id'] for r in plan['new_literals']}
    for kind,key,row in candidate:
        if key in new_ids:continue
        if key=='CLM:test':row=reverse_claim(row,plan['events'][0])
        if kind=='edge' and key in by_edge:row=apply_edge(row,by_edge[key],reverse=True)
        if kind!='metadata':inverse[kind+'s'].update(compact(row).encode()+b'\n')
    assert {k:v.hexdigest() for k,v in inverse.items()}=={k:v.hexdigest() for k,v in transform.digests.items()}


def test_actual_gene_witness_changes_block_build():
    source,plan=fixture();source[1][2]['aliases'].append('new alias')
    with pytest.raises(Exception,match='existing target changed'):
        list(LiteralTransform(plan).records(iter(source)))


def test_full_token_gene_mention_cannot_be_smuggled_in_plan():
    source,plan=fixture();row=next(row for kind,key,row in source if key=='CLM:test')
    row['metadata']['subject_name']='FANCE cortical surface area'
    node=literal_node(row['metadata']['subject_name'])
    with pytest.raises(ValueError,match='not_word_interior_only'):
        reviewed_claim(row,[dict(side='subject',old_id='CUI:C1414531',target_id=node['id'],name=node['preferred_name'])],plan['gene_witnesses'])


def test_literal_collision_blocks_instead_of_overwriting():
    source,plan=fixture();node=literal_node(plan['new_literals'][0]['name'])
    source.insert(1,('node',node['id'],node))
    with pytest.raises(Exception,match='collides'):
        list(LiteralTransform(plan).records(iter(source)))


def test_extra_owned_branch_blocks_instead_of_partial_fix():
    source,plan=fixture();source.append(('edge',4,deepcopy(source[-1][2])))
    with pytest.raises(Exception,match='reference closure'):
        list(LiteralTransform(plan).records(iter(source)))


def test_missing_existing_target_blocks_even_if_claim_can_change():
    source,plan=fixture();source=[r for r in source if r[1]!='CUI:C1414531']
    with pytest.raises(Exception,match='target witnesses incomplete'):
        list(LiteralTransform(plan).records(iter(source)))


def test_same_complete_name_is_one_node_across_independent_claims():
    a=literal_node('cortical surface area');b=literal_node(' cortical   surface area ')
    assert a==b
    assert a['id']!=literal_node('left cortical surface area')['id']
    assert a['id']!=literal_node('Cortical surface area')['id']
