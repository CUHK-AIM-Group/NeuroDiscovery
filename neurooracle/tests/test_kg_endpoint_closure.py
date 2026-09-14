from copy import deepcopy
from types import SimpleNamespace
import pytest

from neurooracle.src.kg_endpoint_closure import (consolidate_root_holds, verified_route,
    finite_endpoint_events, gene_literal_gate, repair_owned_closure)
from neurooracle.src.kg_root_repairs import apply_event


def record():
    return dict(id='CLM:a', metadata=dict(subject_id='A', object_id='B', predicate='correlates_with',
        subject_name='left hippocampal volume', object_name='symptom score', negated=True,
        source_paper={'pmid': '12345678'}, raw_text='Original negative observation',
        evidence={'sample_size': 53}, metadata={'subject_id': 'A', 'object_id': 'B'}))


def edges():
    return [(0, dict(source_id='CLM:a', target_id='A', relation_type='about')),
            (1, dict(source_id='CLM:a', target_id='B', relation_type='about')),
            (2, dict(source_id='A', target_id='B', relation_type='correlates_with', metadata={'claim_id': 'CLM:a', 'n': 53}))]


def test_disabled_canonical_label_does_not_override_complete_term_proof():
    term = {'target_id': 'CUI:verified', 'canonicalize': False}
    terms = SimpleNamespace(term_for=lambda md, side: term)
    assert verified_route(record()['metadata'], 'subject', terms) == term
    md = record()['metadata']; md['subject_id'] = term['target_id']
    assert verified_route(md, 'subject', terms) is None


def test_ineligible_term_is_never_forced():
    assert verified_route(record()['metadata'], 'subject', SimpleNamespace(term_for=lambda *_: None)) is None


def test_finite_repair_preserves_complete_scientific_payload():
    before = record()
    out, proposal, changed_edges = finite_endpoint_events(before, {'subject': 'C'}, edges())
    assert apply_event(before, proposal) == out
    assert out['metadata']['subject_name'] == before['metadata']['subject_name']
    assert out['metadata']['metadata']['subject_id'] == 'C'
    for field in ('evidence', 'source_paper', 'raw_text', 'negated', 'predicate'):
        assert out['metadata'][field] == before['metadata'][field]
    assert len(changed_edges) == 2
    assert [apply_event(dict(edges())[e['ordinal']], e) for e in changed_edges][1]['metadata']['n'] == 53


@pytest.mark.parametrize('fault', ['missing_about', 'duplicate_about', 'wrong_owner', 'wrong_science', 'multiple_science'])
def test_ambiguous_closure_stays_held(fault):
    rows = edges()
    if fault == 'missing_about': rows.pop(1)
    elif fault == 'duplicate_about': rows.append((3, deepcopy(rows[0][1])))
    elif fault == 'wrong_owner': rows[0][1]['metadata'] = {'claim_id': 'CLM:other'}
    elif fault == 'wrong_science': rows[2][1]['source_id'] = 'wrong'
    elif fault == 'multiple_science': rows.append((3, deepcopy(rows[2][1])))
    with pytest.raises(ValueError): finite_endpoint_events(record(), {'subject': 'C'}, rows)


def test_no_science_edge_is_invented():
    assert len(finite_endpoint_events(record(), {'subject': 'C'}, edges()[:2])[2]) == 1


def test_consolidation_retains_both_facts_not_scientific_resolution():
    base = dict(claim_id='CLM:a', side='object', claim_sha256='bound', scientifically_resolved=False)
    rows = [dict(base, reason='structural_endpoint_repaired_but_scientific_type_unresolved', complete_name='ALPS index'),
            dict(base, reason='structural_reference_fixed_scientific_entity_type_unresolved', current_node_id='literal')]
    result = consolidate_root_holds(rows)
    assert len(result) == 1
    assert result[0]['complete_name'] == 'ALPS index' and result[0]['current_node_id'] == 'literal'
    assert not result[0]['scientifically_resolved'] and len(result[0]['historical_queue_reasons']) == 2


def test_conflicting_issue_bindings_cannot_merge():
    rows = [dict(claim_id='CLM:a', side='object', claim_sha256='x', reason='structural_endpoint_repaired_but_scientific_type_unresolved'),
            dict(claim_id='CLM:a', side='object', claim_sha256='y', reason='structural_reference_fixed_scientific_entity_type_unresolved')]
    with pytest.raises(ValueError): consolidate_root_holds(rows)


def test_unknown_gene_witness_never_enables_literal_reuse():
    assert gene_literal_gate(record()['metadata'], 'subject', {}, {}, []) == 'wrong_gene_witness'


def test_old_object_duplicate_edges_removed_only_with_complete_equal_payload():
    r = record(); r['metadata']['original_object_id'] = 'OLD_B'
    rows = edges()
    for ordinal in (1, 2):
        e = deepcopy(rows[ordinal][1]); e['target_id'] = 'OLD_B'
        rows.append((ordinal + 3, e))
    retained, deleted = repair_owned_closure(r, rows)
    assert retained == edges() and len(deleted) == 2
    assert all(d['full_unknown_payload_preserved'] for d in deleted)


def test_old_duplicate_with_distinct_evidence_must_not_be_deleted():
    r = record(); r['metadata']['original_object_id'] = 'OLD_B'
    rows = edges(); extra = deepcopy(rows[2][1]); extra['target_id'] = 'OLD_B'
    extra['metadata']['n'] = 72; rows.append((4, extra))
    with pytest.raises(ValueError): repair_owned_closure(r, rows)


def test_unnamed_placeholder_only_when_missing_side_is_unique():
    r = record(); r['metadata'].update(original_subject_id='unnamed', original_object_id='unnamed')
    rows = edges(); rows[0][1]['metadata'] = {'claim_id': 'CLM:a', 'anchor_role': 'subject'}
    rows[1][1].update(target_id='unnamed', metadata={'claim_id': 'CLM:a', 'anchor_role': 'subject'})
    repaired, deleted = repair_owned_closure(r, rows)
    assert not deleted and repaired[1][1]['target_id'] == 'B'
    assert repaired[1][1]['metadata']['anchor_role'] == 'object'
    assert rows[1][1]['target_id'] == 'unnamed'


@pytest.mark.parametrize('fault', ['unknown_old_id', 'both_sides_unknown', 'polarity_conflict', 'owner_conflict'])
def test_closure_repairs_reject_unsupported_inferences(fault):
    r = record(); rows = edges()
    r['metadata'].update(original_subject_id='OLD', original_object_id='OLD')
    if fault == 'unknown_old_id': rows[1][1]['target_id'] = 'UNKNOWN'
    elif fault == 'both_sides_unknown':
        for i in (0, 1): rows[i][1].update(target_id='OLD', metadata={'anchor_role': 'subject'})
    elif fault == 'polarity_conflict': rows[2][1]['metadata']['negated'] = False
    else: rows[2][1]['metadata']['claim_id'] = 'CLM:other'
    with pytest.raises(ValueError): repair_owned_closure(r, rows)
