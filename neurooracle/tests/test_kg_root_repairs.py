from copy import deepcopy
import pytest
from neurooracle.src import kg_root_repairs as r


def claim(st='manual_curated_abstract'):
    return {'id': 'CLM:x', 'metadata': {'id': 'CLM:x', 'subject_id': 'S', 'object_id': 'O',
        'subject_name': 'left volume', 'object_name': 'condition', 'predicate': 'is_associated_with',
        'negated': True, 'confidence': .7, 'raw_text': 'Original null result.',
        'source_paper': {'pmid': '1'}, 'evidence': {'study_type': st, 'methodology': 'Original method',
            'sample_size': 40, 'unknown': False}, 'metadata': {'conditions': ['baseline']}}}


@pytest.mark.parametrize('value', sorted(r.WORKFLOW_STUDY_TYPES))
def test_workflow_is_removed_only_from_study_design(value):
    before = claim(value); after, reason = r.metadata_repair(before)
    assert reason == 'workflow_label_removed_from_science'
    restored = deepcopy(after); restored['metadata']['evidence']['study_type'] = value
    assert restored == before


@pytest.mark.parametrize('value', ['', None, False, 'human_abstract_study', 'manual_abstract_review_review_or_method',
    'review_abstract', 'fMRI cohort in mice', 'fMRI navigation in multidimensional abstract spaces', 'case Report'])
def test_unknown_missing_mixed_and_case_variants_are_not_guessed(value):
    before = claim(value)
    assert r.metadata_repair(before) == (before, None)


@pytest.mark.parametrize('value,expected', sorted(r.DESIGN_ALIASES.items()))
def test_exact_design_alias(value, expected):
    before = claim(value); after, reason = r.metadata_repair(before)
    assert after['metadata']['evidence']['study_type'] == expected
    assert reason == 'exact_design_alias'
    assert r.metadata_repair(after)[0] == after


@pytest.mark.parametrize('method', ['', None, 'MRI', 'original; fMRI', 'original method'])
def test_method_is_moved_losslessly_not_used_to_guess_design(method):
    before = claim('fMRI'); before['metadata']['evidence']['methodology'] = method
    after, reason = r.metadata_repair(before); e = after['metadata']['evidence']
    assert 'study_type' not in e and 'fMRI' in e['methodology'].split('; ')
    assert not method or method in e['methodology']
    assert e['sample_size'] == 40 and e['unknown'] is False
    assert r.metadata_repair(after)[0] == after


def test_nonstring_methodology_kept_for_review():
    before = claim('EEG'); before['metadata']['evidence']['methodology'] = ['fMRI']
    assert r.metadata_repair(before) == (before, 'hold_nonstring_methodology')


def test_finite_event_roundtrip_and_unknown_fields_preserved():
    before = claim(); after, _ = r.metadata_repair(before)
    proposal = r.event(before, after)
    assert r.apply_event(before, proposal) == after
    changed = deepcopy(before); changed['metadata']['evidence']['unknown'] = 0
    with pytest.raises(ValueError, match='stale'): r.apply_event(changed, proposal)


def test_missing_null_and_boolean_zero_are_not_equal():
    fields = r.changes({'a': None, 'b': False}, {'b': 0})
    assert len(fields) == 2
    assert r.apply_fields({'a': None, 'b': False}, fields) == {'b': 0}
    with pytest.raises(ValueError): r.apply_fields({'a': None, 'b': 0}, fields)


def test_overlapping_changes_rejected():
    fields = r.changes({'a': 1}, {'a': 2})
    with pytest.raises(ValueError): r.apply_fields({'a': 1}, fields + fields)


def test_endpoint_nested_reference_changes_only_ids():
    before = claim(); before['metadata']['metadata']['subject_id'] = 'S'
    after = r.endpoint_change(before, 'subject', 'S2')
    assert after['metadata']['subject_id'] == after['metadata']['metadata']['subject_id'] == 'S2'
    assert len(r.changes(before, after)) == 2
    with pytest.raises(ValueError): r.endpoint_change(before, 'subject', 'CLM:bad')
    with pytest.raises(ValueError): r.endpoint_change(before, 'subject', 'O')
    before['metadata']['metadata']['subject_id'] = 'wrong'
    with pytest.raises(ValueError): r.endpoint_change(before, 'subject', 'S2')


def edges(science=True):
    es = [(0, {'source_id': 'CLM:x', 'target_id': 'S', 'relation_type': 'about'}),
          (1, {'source_id': 'CLM:x', 'target_id': 'O', 'relation_type': 'about'})]
    if science: es.append((2, {'source_id': 'S', 'target_id': 'O', 'relation_type': 'is_associated_with',
                                'metadata': {'claim_id': 'CLM:x', 'negated': True}}))
    return es


@pytest.mark.parametrize('science', [True, False])
def test_endpoint_change_updates_owned_edges_without_inventing_missing_science(science):
    before = claim(); after = r.endpoint_change(before, 'subject', 'S2')
    proposals = r.owned_endpoint_changes(before, after, edges(science))
    assert len(proposals) == 1 + int(science)
    for p in proposals:
        changed = r.apply_event(dict(edges(science))[p['ordinal']], p)
        if p['ordinal'] == 2: assert changed['metadata']['negated'] is True


def test_mixed_or_incomplete_edge_closure_is_rejected():
    before = claim(); after = r.endpoint_change(before, 'subject', 'S2')
    with pytest.raises(ValueError): r.owned_endpoint_changes(before, after, edges()[:1])
    e = edges(); e[2][1]['source_id'] = 'WRONG'
    with pytest.raises(ValueError): r.owned_endpoint_changes(before, after, e)


def test_unclassified_full_mentions_are_shared_not_paper_salted_or_shortened():
    a = r.unclassified_literal('left X during task A')
    assert a == r.unclassified_literal('left X during task A')
    assert a['id'] != r.unclassified_literal('right X during task A')['id']
    assert a['domain_tags'] == ['external'] and a['preferred_name'] == 'left X during task A'


@pytest.mark.parametrize('field,value', [('negated', False), ('confidence', .8), ('raw_text', 'Other observation'),
                                        ('unknown', False), ('source_paper', {'pmid': '2'})])
def test_duplicate_guard_protects_more_than_candidate_fingerprint(field, value):
    a = claim(); b = deepcopy(a); b['metadata'][field] = value
    assert r.duplicate_node_payload(a) != r.duplicate_node_payload(b)


def test_substantive_audit_difference_blocks_duplicate_removal():
    a = claim(); b = deepcopy(a)
    a['metadata']['scope_reaudit'] = {'gates': {'verified': True}, 'reviewed_at': 'old'}
    b['metadata']['scope_reaudit'] = {'gates': {'verified': False}, 'reviewed_at': 'new'}
    assert r.audit_decision(a) != r.audit_decision(b)
