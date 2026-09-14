from copy import deepcopy
import pytest
from neurooracle.src import kg_clinical_literal_reuse as r

NAME='voxel-mirrored homotopic connectivity alterations'


def family(name=NAME):
    f=r.FAMILIES[name]
    nodes={nid:dict(id=nid,preferred_name=name,domain_tags=['claim_concept'],metadata={'atom_types':['imaging_marker']},
        semantic_types=[],external_ids={},aliases=[],definition='',spatial_mapping=None) for nid in f['members']}
    incidents={nid:[dict(claim_id='CLM:'+str(i),name=name,roles=['imaging_marker'])] for i,nid in enumerate(nodes)}
    return nodes,incidents


def claim(old=None):
    f=r.FAMILIES[NAME]; old=old or next(n for n in f['members'] if n!=f['target'])
    return dict(id='CLM:test',metadata=dict(id='CLM:test',subject_id=old,subject_name=NAME,subject_type='IMAGING_MARKER',
        object_id='CUI:disease',object_name='schizophrenia',object_type='DISEASE',predicate='is_associated_with',
        raw_text='Original study evidence and conditions.',negated=True,evidence={'p_value':None},conditions={'age':'adults'},
        source_paper={'pmid':'38382862'},metadata={'subject_id':old,'subject_type':'IMAGING_MARKER'}))


def changes(row):
    return [dict(side='subject',old_id=row['metadata']['subject_id'],name=NAME,target_id=r.FAMILIES[NAME]['target'])]


@pytest.mark.parametrize('name',sorted(r.FAMILIES))
def test_whole_literal_family_without_new_external_identity(name):
    nodes,incidents=family(name);p=r.validate_family(name,nodes,incidents,[])
    assert p['source_anchor_nodes_and_detail_rows_preserved'] and len(p['member_hashes'])==2


@pytest.mark.parametrize('field,value',[('semantic_types',['T1']),('external_ids',{'x':'y'}),('aliases',['different']),
    ('definition','different definition'),('spatial_mapping',{'atlas':'x'}),('preferred_name',NAME.title())])
def test_additional_scope_never_silently_merged(field,value):
    nodes,incidents=family();nodes[next(iter(nodes))][field]=value
    with pytest.raises(ValueError):r.validate_family(NAME,nodes,incidents,[])


@pytest.mark.parametrize('role',[[],['outcome'],['gene_target'],['individual_data']])
def test_incident_type_must_be_explicit_not_majority(role):
    nodes,incidents=family();incidents[next(iter(incidents))][0]['roles']=role
    with pytest.raises(ValueError):r.validate_family(NAME,nodes,incidents,[])


def test_exact_alias_still_requires_separate_proof():
    nodes,incidents=family()
    with pytest.raises(ValueError):r.validate_family(NAME,nodes,incidents,[{'alias':NAME}])


def test_unknown_noncanonical_node_allowed_only_with_explicit_incident_type():
    nodes,incidents=family();other=next(n for n in nodes if n!=r.FAMILIES[NAME]['target'])
    nodes[other]['metadata']={}
    assert r.validate_family(NAME,nodes,incidents,[])['target_id']==r.FAMILIES[NAME]['target']
    nodes[r.FAMILIES[NAME]['target']]['metadata']={}
    with pytest.raises(ValueError):r.validate_family(NAME,nodes,incidents,[])


@pytest.mark.parametrize('old',[None,'CUI:C1421437','CUI:C1414531'])
def test_identity_reuse_and_full_inverse_preserve_science(old):
    row=claim(old);before=deepcopy(row);event,out=r.reviewed_claim(row,changes(row))
    assert row==before and r.verified_reverse(out,event)==row
    for field in ('raw_text','negated','conditions','evidence','source_paper'):
        assert out['metadata'][field]==row['metadata'][field]


@pytest.mark.parametrize('problem',['case','short_name','side_type','inner_id','inner_type','wrong_target','unknown_source','same_target'])
def test_unapproved_endpoint_change_rejected(problem):
    row=claim();ch=changes(row)
    if problem=='case':row['metadata']['subject_name']=NAME.title()
    if problem=='short_name':row['metadata']['subject_name']='homotopic connectivity'
    if problem=='side_type':row['metadata']['subject_type']='OUTCOME';row['metadata']['metadata']['subject_type']='OUTCOME'
    if problem=='inner_id':row['metadata']['metadata']['subject_id']='different'
    if problem=='inner_type':row['metadata']['metadata']['subject_type']='GENE_TARGET'
    if problem=='wrong_target':ch[0]['target_id']='different'
    if problem=='unknown_source':row=claim('CLM_CONCEPT:unknown');ch=changes(row)
    if problem=='same_target':row=claim(r.FAMILIES[NAME]['target']);ch=changes(row)
    with pytest.raises(ValueError):r.reviewed_claim(row,ch)


def test_stale_scientific_content_and_duplicate_side_rejected():
    row=claim();event,out=r.reviewed_claim(row,changes(row));out['metadata']['negated']=False
    with pytest.raises(ValueError):r.verified_reverse(out,event)
    with pytest.raises(ValueError):r.reviewed_claim(row,changes(row)*2)

