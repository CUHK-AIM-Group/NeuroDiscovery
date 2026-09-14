from copy import deepcopy
import pytest
from neurooracle.src.kg_gene_boundary_repair import endpoint_gate,word_interior_hits


def fixture(name='cortical surface area'):
    witness=dict(node_id='CUI:C1414531',node_sha256='a'*64,name='FANCE',semantic_types=['T028'],labels=['FANCE','FACE'])
    md=dict(subject_id=witness['node_id'],subject_name=name,subject_type='IMAGING_MARKER',metadata={})
    return md,witness


def test_full_measurement_word_interior_hit_passes_identity_gate():
    md,w=fixture(); assert endpoint_gate(md,'subject',w) is None


@pytest.mark.parametrize('name,labels,hits',[
    ('surface area',['FACE'],['face']),('number of episodes',['NUMB'],['numb']),
    ('NUMB mutation',['NUMB'],[]),('VCP',['VCP','TERA'],[]),
    ('VCP-related alterations',['VCP','TERA'],[]),('NUMB2',['NUMB'],['numb']),
    ('white matter',['MATT'],['matt']),('gene expression',['GENE'],[]),
])
def test_word_interior_is_not_a_whole_gene_symbol(name,labels,hits):
    assert word_interior_hits(name,labels)==hits


@pytest.mark.parametrize('field,value,reason',[
    ('subject_type',None,'not_explicit_imaging_type'),
    ('subject_type','OUTCOME','not_explicit_imaging_type'),
    ('subject_name','FANCE','not_word_interior_only'),
    ('subject_name','FANCE cortical surface area','not_word_interior_only'),
    ('subject_name','genetic influences on cortical surface area','molecular_or_genetic_context'),
    ('subject_name','frontotemporal alterations','not_word_interior_only'),
])
def test_hold_risky_scopes(field,value,reason):
    md,w=fixture();md[field]=value
    assert endpoint_gate(md,'subject',w)==reason


@pytest.mark.parametrize('nested,reason',[
    ({'subject_id':'CUI:OTHER'},'nested_id_conflict'),
    ({'subject_type':'OUTCOME'},'nested_type_conflict'),
])
def test_nested_conflicts(nested,reason):
    md,w=fixture();md['metadata']=nested
    assert endpoint_gate(md,'subject',w)==reason


def test_witness_needs_actual_gene_not_generic_biomarker():
    md,w=fixture();w['semantic_types']=['T033']
    assert endpoint_gate(md,'subject',w)=='not_confirmed_gene_type'


def test_inputs_not_changed():
    md,w=fixture();a,b=deepcopy(md),deepcopy(w)
    endpoint_gate(md,'subject',w)
    assert md==a and w==b
