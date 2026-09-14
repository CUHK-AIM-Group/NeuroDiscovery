from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from classify_kg_expanded_literal_candidates import candidate_gate


def record(name,role='imaging_marker'):
    return dict(claim_id='CLM:test',name=name,declared_roles=[role] if role else [],structural_holds=[],existing_name_or_alias_candidates=[])


@pytest.mark.parametrize('name,role',[('gray matter asymmetry','imaging_marker'),('emotional improvement','outcome'),
    ('frequency of hospitalization','individual_data'),('contour integration impairment','cognitive_task'),
    ('childhood attention-deficit hyperactivity disorder','disease'),('whole hippocampus volume','imaging_marker'),
    ('default mode network global clustering coefficient in theta band','imaging_marker')])
def test_finite_nonmolecular_complete_surfaces(name,role):
    assert candidate_gate(record(name,role),set()) is None


@pytest.mark.parametrize('name,role',[('CAMK2N2','gene_target'),('psychiatric risk genes','gene_target'),('DNA methylation GDF15','imaging_marker'),
    ('plasma BACE1 concentration','imaging_marker'),('vascular cell types','imaging_marker'),('amyloid-atrophy relationship','outcome'),
    ('genetically inferred major depression liability','outcome'),('structural network gray matter volume',None),('gray matter volume','drug')])
def test_molecular_or_unspecified_scope_is_not_assumed(name,role):
    assert candidate_gate(record(name,role),set())


@pytest.mark.parametrize('kind',['collision','closure','advanced','mixed'])
def test_fresh_current_and_existing_identity_reviews_cannot_be_bypassed(kind):
    r=record('gray matter volume');overlap=set()
    if kind=='collision':r['existing_name_or_alias_candidates']=['CLM_CONCEPT:existing']
    elif kind=='closure':r['structural_holds']=['ambiguous']
    elif kind=='advanced':overlap.add(r['claim_id'])
    else:r['declared_roles'].append('gene_target')
    assert candidate_gate(r,overlap)
