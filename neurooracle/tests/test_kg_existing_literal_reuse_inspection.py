from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_kg_existing_literal_reuse import candidate_node_gate,incidence_gate
from neurooracle.src.kg_identity_pilot import digest


def fixture():
    r=dict(claim_id='CLM:test',name='frontal gray matter volume',literal_candidate_gate='existing_full_name_or_case_alias_requires_reuse_review',
        existing_name_or_alias_candidates=['CLM_CONCEPT:target'],declared_roles=['imaging_marker'],structural_holds=[])
    n=dict(name=r['name'],semantic_types=[],external_ids={},aliases=[],definition_sha256=digest(''),spatial_mapping_sha256=digest(None),
        domain_tags=['biomarker'],metadata_keys=['curation_scope','atom_type'],reviewed_metadata={'curation_scope':'old_case_study','atom_type':'imaging_marker'})
    return r,n


def test_whole_name_reuse_does_not_modify_provenance_or_add_a_new_node():
    r,n=fixture();assert candidate_node_gate(r,n) is None
    assert incidence_gate(n,[dict(name=n['name'])],[]) is None


@pytest.mark.parametrize('field,value',[('semantic_types',['T028']),('aliases',['gray matter']),('external_ids',{'source':'one'}),
    ('definition_sha256',digest('regional scope')),('spatial_mapping_sha256',digest({})),('name','regional frontal gray matter volume'),
    ('metadata_keys',['population'])])
def test_unreviewed_node_scope_held(field,value):
    r,n=fixture();n[field]=value
    assert candidate_node_gate(r,n)


def test_broad_incident_and_nonclaim_relations_require_separate_review():
    r,n=fixture()
    assert incidence_gate(n,[dict(name='regional '+n['name'])],[])
    assert incidence_gate(n,[],[])
    assert incidence_gate(n,[dict(name=n['name'])],[{'relation_type':'same_as'}])


def test_molecular_role_not_hidden_by_matching_name():
    r,n=fixture();n['reviewed_metadata']['atom_type']='gene_target'
    assert candidate_node_gate(r,n)
