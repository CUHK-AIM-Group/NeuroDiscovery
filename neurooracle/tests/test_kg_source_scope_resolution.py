from copy import deepcopy
import pytest
from neurooracle.src import kg_source_scope_resolution as scope
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges


def bind(monkeypatch,row,pmid):
    row['metadata']['source_paper']={'pmid':pmid}
    row['metadata']['negated']=False
    monkeypatch.setitem(scope.CLAIM_HASHES,row['id'],digest(row))
    return {pmid:{'severity_association_reported_as_nonsignificant':True}}


def myelin(monkeypatch):
    row=claim('lateral prefrontal cortex qMRI myelin content');row['id']=scope.MYELIN
    md=row['metadata'];md.update(id=row['id'],object_name='major depressive disorder symptom severity',
        predicate='correlates_with',raw_text='Myelin in the LPFC was related to depression-relevant clinical measures.')
    proof=bind(monkeypatch,row,'28526817')
    return row,edges(row)[:2],proof


def test_myelin_removal_requires_specific_affirmative_mismatch(monkeypatch):
    row,owned,proof=myelin(monkeypatch);r=scope.reviewed_myelin_removal(row,owned,proof)
    assert r['claim_id']==scope.MYELIN and len(r['owned_edges'])==2
    assert r['no_null_result_claim_synthesized'] and r['original_source_evidence_preserved']
    assert 'metadata' not in r


@pytest.mark.parametrize('field,value',[('negated',True),('object_name','number of depressive episodes'),('predicate','causes')])
def test_changed_source_claim_is_never_deleted(monkeypatch,field,value):
    row,owned,proof=myelin(monkeypatch);row['metadata'][field]=value
    with pytest.raises(ValueError):scope.reviewed_myelin_removal(row,owned,proof)


def test_missing_fulltext_severity_proof_blocks(monkeypatch):
    row,owned,proof=myelin(monkeypatch);proof['28526817']={}
    with pytest.raises(ValueError,match='missing source'):scope.reviewed_myelin_removal(row,owned,proof)


def frailty(monkeypatch):
    row=claim('white-matter-hyperintensity burden');row['id']=scope.FRAILTY
    md=row['metadata'];md.update(id=row['id'],subject_id='CLM_CONCEPT:white_matter_hyperintensity_burden',
        object_id='CLM_CONCEPT:frailty_09cfc9267861',object_name='frailty')
    proof=bind(monkeypatch,row,'41297452');kept=edges(row)
    old={'CLM_CONCEPT:white_matter_hyperintensity_burden':'CLM_CONCEPT:white_matter_hyperintensity_burden_e206a534fc08',
        'CLM_CONCEPT:frailty_09cfc9267861':'CLM_CONCEPT:frailty'}
    extra=[]
    for ordinal,edge in kept:
        e=deepcopy(edge)
        for f in ('source_id','target_id'):e[f]=old.get(e[f],e[f])
        extra.append((ordinal+3,e))
    return row,kept+extra,proof


def test_frailty_retires_only_exact_alternative_payload(monkeypatch):
    row,owned,proof=frailty(monkeypatch);before=deepcopy((row,owned))
    r=scope.reviewed_frailty_retirements(row,owned,proof)
    assert [e['ordinal'] for e in r]==[4,5,6]
    assert {e['retained_ordinal'] for e in r}=={1,2,3}
    assert (row,owned)==before


@pytest.mark.parametrize('part',['different_evidence','other_owner','missing_branch'])
def test_mixed_frailty_branch_is_not_silently_erased(monkeypatch,part):
    row,owned,proof=frailty(monkeypatch)
    if part=='different_evidence':owned[-1][1]['evidence_ref']='Another scientific source'
    if part=='other_owner':owned[-1][1]['metadata']['claim_id']='CLM:other'
    if part=='missing_branch':owned.pop()
    with pytest.raises(ValueError):scope.reviewed_frailty_retirements(row,owned,proof)


def test_viral_expansion_preserves_quote_and_does_not_create_global_vcp_alias(monkeypatch):
    row=claim('VCP');row['id']=scope.VIRUS
    row['metadata'].update(id=row['id'],object_name='uptake in all retinal layers followed by clearance within 1-3 hours',
        raw_text='After VCP was injected into the eye, it was taken up in all layers of the retina but was cleared within 1-3 hours of delivery.')
    proof=bind(monkeypatch,row,'29534078');node=scope.viral_node()
    event,out=scope.reviewed_viral_identity(row,proof,node['id'])
    assert scope.reverse_claim(out,event)==row and out['metadata']['subject_name']=='VCP'
    assert node['preferred_name']==scope.VIRAL_NAME and not node['aliases'] and not node['external_ids']
    assert not node['metadata']


@pytest.mark.parametrize('removed',[[],[1],[2,3,7],[1,2,5,8,9]])
def test_ordinal_roundtrip_excludes_removed_positions(removed):
    for old in range(1,20):
        if old in removed:
            with pytest.raises(ValueError):scope.compacted_ordinal(old,removed)
        else:
            current=scope.compacted_ordinal(old,removed)
            assert scope.original_ordinal(current,removed)==old


def test_wrong_pubmed_sources_fail_closed():
    with pytest.raises(ValueError,match='own PMID'):scope.public_source_proof('<PubmedArticleSet/>','<article/>')


def reuse_fixture(monkeypatch):
    node=dict(id=scope.VIRAL_REUSE_ID,preferred_name=scope.VIRAL_NAME,semantic_types=[],aliases=[],external_ids={},metadata={})
    monkeypatch.setattr(scope,'VIRAL_REUSE_SHA',digest(node))
    incidents=[];sources={}
    for i,(cid,(sha,pmid,doi)) in enumerate(scope.VIRAL_INCIDENTS.items()):
        incidents.append(dict(claim_id=cid,claim_sha256=sha,pmid=pmid,doi=doi,side='subject',
            name=scope.VIRAL_NAME if i<2 else 'Vaccinia virus complement control protein',outer_type=None,inner_type=None))
        sources[pmid]=dict(title=scope.VIRAL_NAME+' study',doi=doi,abstract_sha256='test')
    return node,incidents,sources


def test_existing_full_protein_identity_can_be_reused_without_fuzzy_alias(monkeypatch):
    node,incidents,sources=reuse_fixture(monkeypatch)
    witness=scope.reviewed_viral_reuse(node,incidents,sources)
    assert scope.validate_viral_reuse_witness(witness)
    assert witness['node_id']==scope.VIRAL_REUSE_ID
    assert witness['existing_claim_scientific_conclusions_not_revalidated']


@pytest.mark.parametrize('problem',['missing_incident','wrong_name','unknown_source','wrong_role','node_changed'])
def test_unreviewed_existing_scope_is_not_reused(monkeypatch,problem):
    node,incidents,sources=reuse_fixture(monkeypatch)
    if problem=='missing_incident':incidents.pop()
    if problem=='wrong_name':incidents[0]['name']='valosin-containing protein'
    if problem=='unknown_source':incidents[0]['pmid']='unknown'
    if problem=='wrong_role':incidents[0]['outer_type']='IMAGING_MARKER'
    if problem=='node_changed':node['external_ids']={'HGNC':'VCP'}
    with pytest.raises(ValueError):scope.reviewed_viral_reuse(node,incidents,sources)
