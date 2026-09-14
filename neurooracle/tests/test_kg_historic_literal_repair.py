from copy import deepcopy
import pytest

from neurooracle.src import kg_historic_literal_repair as repair
from neurooracle.src.kg_identity_pilot import digest, nonidentity_claim
from neurooracle.src.kg_literal_endpoint_repair import literal_node as imaging_literal_node
from neurooracle.tests.test_kg_literal_endpoint_repair import claim, edges


def fixture(name='brain structural alterations', declared='IMAGING_MARKER'):
    row = claim(name); md = row['metadata']
    gene = dict(id='CUI:C1421437', preferred_name='VCP', aliases=['TERA'], semantic_types=['T028'])
    md['subject_id'] = gene['id']; md['metadata']['subject_id'] = gene['id']
    md['subject_type'] = declared; md['metadata']['subject_type'] = declared
    witness = dict(node_id=gene['id'], node_sha256=digest(gene), name='VCP', labels=['VCP', 'TERA'], semantic_types=['T028'])
    review = dict(claim_id=row['id'], claim_sha256=digest(row), side='subject', name=name,
        current_node_id=gene['id'], source_gene_node_sha256=witness['node_sha256'],
        current_gate=None, word_interior_alias_hits=['tera'],
        identity_scope='complete_original_mention_not_single_VCP_or_FANCE_gene',
        outer_type=declared, inner_type=declared)
    return row, gene, witness, review


def event_for(row, witness, review, target=None):
    node = target or repair.literal_node(row['metadata']['subject_name'])
    change = dict(side='subject', old_id=witness['node_id'], name=node['preferred_name'], target_id=node['id'])
    return repair.reviewed_claim(row, [change], {witness['node_id']: witness}, {repair.review_key(row['id'], 'subject'): review})


@pytest.mark.parametrize('name,role', [
    ('brain structural alterations', 'IMAGING_MARKER'),
    ('AKT1-COMT Val158Met interaction', 'GENE_TARGET'),
    ('PTSD severity in combat veterans', 'OUTCOME'),
    ('gut microbiota alterations', None),
    ('left dorsolateral prefrontal cortex transcranial magnetic stimulation target', 'DRUG'),
])
def test_finite_identity_restoration_does_not_infer_a_scientific_role(name, role):
    row, gene, witness, review = fixture(name, role)
    before = deepcopy(row); event, current = event_for(row, witness, review)
    assert row == before and repair.reverse_claim(current, event) == row
    assert nonidentity_claim(current) == nonidentity_claim(row)
    node = repair.literal_node(name)
    assert node['domain_tags'] == ['claim_concept'] and node['metadata'] == {}
    assert node['semantic_types'] == [] and node['external_ids'] == {} and node['aliases'] == []
    assert current['metadata']['subject_type'] == role


@pytest.mark.parametrize('field,value,reason', [
    ('claim_sha256', 'f'*64, 'reviewed_claim_changed'),
    ('name', 'truncated alterations', 'complete_original_name_changed'),
    ('current_gate', 'ambiguous about endpoint', 'historic_review_still_held'),
    ('current_node_id', 'CUI:OTHER', 'wrong_reviewed_gene'),
    ('source_gene_node_sha256', 'a'*64, 'unbound_gene_witness'),
    ('word_interior_alias_hits', [], 'not_reviewed_word_interior_only'),
    ('identity_scope', 'probably similar', 'unreviewed_identity_scope'),
    ('outer_type', 'OUTCOME', 'reviewed_types_changed'),
])
def test_finite_review_cannot_be_broadened(field, value, reason):
    row, gene, witness, review = fixture(); review[field] = value
    assert repair.endpoint_gate(row, 'subject', witness, review) == reason


@pytest.mark.parametrize('name', ['VCP', 'VCP alterations', 'TERA interaction'])
def test_real_gene_token_is_never_a_word_interior_repair(name):
    row, gene, witness, review = fixture(name)
    assert repair.endpoint_gate(row, 'subject', witness, review) == 'not_reviewed_word_interior_only'


def test_allowlist_required_and_nested_conflicts_do_not_disappear():
    row, gene, witness, review = fixture()
    assert repair.endpoint_gate(row, 'subject', witness, {}) == 'not_in_finite_reviewed_scope'
    row['metadata']['metadata']['subject_id'] = 'CUI:OTHER'; review['claim_sha256'] = digest(row)
    assert repair.endpoint_gate(row, 'subject', witness, review) == 'nested_id_conflict'


def test_literal_identity_shared_by_exact_name_not_claim_or_paper():
    assert repair.literal_node(' structural   alterations ') == repair.literal_node('structural alterations')
    assert repair.literal_node('structural alterations') != repair.literal_node('Structural alterations')
    assert repair.literal_node('left structural alterations') != repair.literal_node('right structural alterations')
    with pytest.raises(ValueError, match='missing complete name'): repair.literal_node(' ')


def test_only_two_reviewed_empty_metadata_imaging_literals_are_reused():
    name = 'frontal cortical surface area'; node = imaging_literal_node(name)
    incidents = [dict(name=name, declared_roles=['imaging_marker'])]
    assert repair.reuse_gate(node, name, incidents) is None
    assert repair.reuse_gate(node, name, []) == 'existing_incidence_proof_missing'
    altered = deepcopy(node); altered['metadata']['paper_case_study_ids'] = ['case1']
    assert repair.reuse_gate(altered, name, incidents) == 'existing_literal_needs_separate_scope_review'
    assert repair.reuse_gate(node, name, [dict(name='left frontal cortical surface area', declared_roles=[])]) == 'existing_incident_scope_differs'


@pytest.mark.parametrize('target', [None, 'CLM_CONCEPT:invented', 'CUI:C1421437'])
def test_forged_target_rejected(target):
    row, gene, witness, review = fixture()
    change = dict(side='subject', name=row['metadata']['subject_name'], old_id=gene['id'], target_id=target)
    with pytest.raises(ValueError, match='unreviewed literal target'):
        repair.reviewed_claim(row, [change], {gene['id']: witness}, {repair.review_key(row['id'], 'subject'): review})
