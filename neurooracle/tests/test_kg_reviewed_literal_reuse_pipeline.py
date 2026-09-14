from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_kg_reviewed_literal_reuse import LiteralTransform,IndependentLiteralProof
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from plan_kg_reviewed_literal_reuse import adjusted_ordinal
from neurooracle.src import kg_reviewed_literal_reuse as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_reviewed_literal_reuse import fixture


def setup(science=True,reuse=True):
    cids=[repair.CHORIO,'CLM:585022a2f24a',repair.MOUSE,repair.PRENATAL]
    events=[];changes=[];retired=[];reviews={};refs=[];claims=[];nodes={};new={}
    genes={nid:dict(id=nid,preferred_name=name,aliases=aliases,semantic_types=['T028'],metadata={})
        for nid,name,aliases in [('CUI:C1414531','FANCE',['FACE']),('CUI:C1421437','VCP',['TERA'])]}
    witnesses={nid:dict(node_id=nid,name=r['preferred_name'],labels=[r['preferred_name'],*r['aliases']],
        semantic_types=['T028'],node_sha256=digest(r)) for nid,r in genes.items()}
    for cid in cids:
        row,ch,_,review,owned=fixture(cid)
        event,current=repair.reviewed_claim(row,ch,witnesses,review,dict(source_is_protein_not_human_gene=True))
        claims.append(row);reviews.update(review);events.append(event)
        if not science:owned=[r for r in owned if r[1]['relation_type']=='about']
        owned=[(len(refs)+i,r) for i,(_,r) in enumerate(owned,1)]
        refs.extend(owned);changes.extend(repair.reviewed_edges(cid,row,current,owned))
        _,extra=repair.split_owned(cid,row,owned);retired.extend(extra)
        side,name,nid=repair.SPECS[cid]
        if cid in {repair.MOUSE,repair.PRENATAL} or (cid==repair.CHORIO and not reuse):
            name=repair.MOUSE_NAME if cid==repair.MOUSE else name
            node=repair.literal_node(name);new[nid]=dict(id=nid,name=name,node_sha256=digest(node))
        else:nodes[nid]=repair.literal_node(name) if cid==repair.CHORIO else dict(id=nid,preferred_name=name,metadata={})
        for side in ('subject','object'):
            nid=row['metadata'][side+'_id']
            if nid not in genes:nodes.setdefault(nid,dict(id=nid,preferred_name=row['metadata'][side+'_name'],metadata={}))
    nodes[repair.OLD_COGNITIVE]=dict(id=repair.OLD_COGNITIVE,preferred_name='cognitive performance',metadata={})
    removed=sorted(e['ordinal'] for e in retired)
    for e in changes:e['current_ordinal']=adjusted_ordinal(e['ordinal'],removed)
    for e in retired:e['current_kept_ordinal']=adjusted_ordinal(e['kept_ordinal'],removed)
    plan=dict(events=events,edge_events=sorted(changes,key=lambda e:e['ordinal']),retired_branches=retired,
        removed_edge_ordinals=removed,new_literals=sorted(new.values(),key=lambda r:r['id']),
        existing_targets={nid:digest(row) for nid,row in {**genes,**nodes}.items()},gene_witnesses=witnesses,
        reviewed_claims=reviews,public_scope_proof=dict(source_is_protein_not_human_gene=True))
    source=[('metadata',None,{}),*[('node',nid,row) for nid,row in {**genes,**nodes}.items()],
        *[('node',row['id'],row) for row in claims],*[('edge',o,r) for o,r in refs]]
    return source,plan


def current_records(source,plan,tmp_path):
    transform=LiteralTransform(plan);out=BytesIO();catalog=tmp_path/'catalog.jsonl'
    result=stream_patch(transform.records(iter(source)),out,[],{},catalog)
    graph=json.loads(out.getvalue())
    records=[('metadata',None,graph['metadata']),*[('node',k,r) for k,r in graph['concepts'].items()],
        *[('edge',i,r) for i,r in enumerate(graph['edges'],1)]]
    assert verify_catalog_and_graph(iter(records),result,catalog)['shared_relation_index_complete']
    return records,transform,result


@pytest.mark.parametrize('science',[True,False])
@pytest.mark.parametrize('reuse',[True,False])
def test_full_current_inverse_handles_removed_ordinals_reuse_and_mouse(tmp_path,science,reuse):
    source,plan=setup(science,reuse);before=deepcopy(source)
    current,transform,result=current_records(source,plan,tmp_path)
    assert source==before and result['counts']['claims']==4
    assert result['counts']['edges']==len([r for r in source if r[0]=='edge'])-len(plan['retired_branches'])
    proof=IndependentLiteralProof(plan,{k:h.hexdigest() for k,h in transform.digests.items()})
    assert list(proof.records(iter(current)))==current
    proof.verify()
    assert any(e['ordinal']!=e['current_ordinal'] for e in plan['edge_events'])
    assert all(r['metadata']=={} for kind,key,r in current if kind=='node' and key in {n['id'] for n in plan['new_literals']})
    assert {r['relation_type'] for kind,key,r in current if kind=='edge'} <= {'about','is_associated_with'}


@pytest.mark.parametrize('part',['claim_science','unrelated_node','kept_about','changed_edge','mouse_taxon','drop_reference'])
def test_independent_scan_rejects_changed_kept_records(tmp_path,part):
    source,plan=setup();current,transform,_=current_records(source,plan,tmp_path)
    if part=='claim_science':next(r for kind,key,r in current if key==repair.CHORIO)['metadata']['raw_text']='different scientific claim'
    elif part=='unrelated_node':next(r for kind,key,r in current if key==repair.OLD_COGNITIVE)['preferred_name']='changed'
    elif part=='kept_about':next(r for kind,key,r in current if kind=='edge' and key==plan['retired_branches'][0]['current_kept_ordinal'])['confidence']=0.01
    elif part=='changed_edge':next(r for kind,key,r in current if kind=='edge' and key==plan['edge_events'][-1]['current_ordinal'])['metadata']['new_audit']='changed'
    elif part=='mouse_taxon':next(r for kind,key,r in current if key==repair.MOUSE_ID)['external_ids']['NCBI_Taxonomy']='9606'
    else:current=[r for r in current if not (r[0]=='edge' and r[1]==plan['edge_events'][-1]['current_ordinal'])]
    proof=IndependentLiteralProof(plan,{k:h.hexdigest() for k,h in transform.digests.items()})
    with pytest.raises(ValueError):
        list(proof.records(iter(current)));proof.verify()


@pytest.mark.parametrize('part',['new_collision','old_branch_payload','wrong_owner','unreviewed_claim','missing_existing'])
def test_forward_scope_is_exact(part):
    source,plan=setup()
    if part=='new_collision':
        n=repair.literal_node(repair.MOUSE_NAME);source.insert(1,('node',n['id'],n))
    elif part=='old_branch_payload':next(r for kind,key,r in source if kind=='edge' and key==plan['removed_edge_ordinals'][0])['evidence_ref']='additional evidence'
    elif part=='wrong_owner':next(r for kind,key,r in source if kind=='edge' and key==plan['edge_events'][0]['ordinal'])['metadata']['claim_id']='CLM:wrong'
    elif part=='unreviewed_claim':plan['reviewed_claims'].pop(repair.CHORIO)
    else:source=[r for r in source if r[1]!=repair.OLD_COGNITIVE]
    with pytest.raises(ValueError):list(LiteralTransform(plan).records(iter(source)))


def test_bad_current_old_index_mapping_rejected(tmp_path):
    source,plan=setup();current,transform,_=current_records(source,plan,tmp_path)
    plan=deepcopy(plan);plan['edge_events'][-1]['current_ordinal']+=1
    proof=IndependentLiteralProof(plan,{k:h.hexdigest() for k,h in transform.digests.items()})
    with pytest.raises(ValueError):
        list(proof.records(iter(current)));proof.verify()
