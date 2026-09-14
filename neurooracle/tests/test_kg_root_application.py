from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import pytest

from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_root_application import RootTransform, reverse_event, source_edge_ordinal, unique_index
from neurooracle.src.kg_root_repairs import event, unclassified_literal


@pytest.mark.parametrize('before,after', [({'a': 0}, {'a': False}), ({'a': None}, {}), ({}, {'a': None}),
    ({'x': {'a': 1, 'unknown': [False, 0, None]}}, {'x': {'unknown': [False, 0, None]}}),
    ({'x': {'a': 'fMRI', 'm': 'MRI'}}, {'x': {'m': 'MRI; fMRI'}})])
def test_inverse_round_trip_preserves_presence_and_unknown_values(before, after):
    proposal = event(before, after)
    assert reverse_event(after, proposal) == before


def test_inverse_rejects_unapproved_change_even_outside_patch():
    before, after = {'a': 1, 'unknown': 5}, {'a': 2, 'unknown': 5}
    proposal = event(before, after)
    with pytest.raises(ValueError, match='candidate record'):
        reverse_event({**after, 'unknown': 6}, proposal)


def test_unique_operation_required():
    with pytest.raises(ValueError, match='duplicate'):
        unique_index([{'id': 'x'}, {'id': 'x'}], 'id')


@pytest.mark.parametrize('removed,expected', [([], [1, 2, 3]), ([1], [2, 3, 4]),
    ([2, 3], [1, 4, 5]), ([1, 2, 4, 8], [3, 5, 6]), ([4, 7], [1, 2, 3])])
def test_edge_ordinal_mapping_is_exact(removed, expected):
    assert [source_edge_ordinal(i, removed) for i in range(1, 4)] == expected


def fixture():
    original = {'id': 'CLM:a', 'metadata': {'id': 'CLM:a', 'negated': True, 'evidence': {'study_type': 'manual_curated_abstract'}}}
    after = deepcopy(original); del after['metadata']['evidence']['study_type']
    deleted = {'id': 'CLM:b', 'metadata': {'id': 'CLM:b'}}
    edge = {'source_id': 'CLM:a', 'target_id': 'CLM:b'}
    node = unclassified_literal('complete unknown observation')
    patch = RootTransform([event(original, after, claim_id='CLM:a')], [],
        [dict(claim_id='CLM:b', record_sha256=digest(deleted))],
        [dict(ordinal=1, record_sha256=digest(edge))], [node])
    records = [('metadata', '', {}), ('node', 'CLM:a', original), ('node', 'CLM:b', deleted), ('edge', '1', edge)]
    return patch, records, original, after, node


def test_stream_applies_once_and_keeps_no_original_records():
    patch, records, original, after, node = fixture()
    out = list(patch.records(records))
    assert out == [('metadata', '', {}), ('node', 'CLM:a', after), ('node', node['id'], node)]
    assert patch.source_digests['nodes'].hexdigest() == hashlib.sha256(bytes.fromhex(digest(original))).hexdigest()
    assert not hasattr(patch, 'originals')


def test_stream_rejects_stale_deletion():
    patch, records, *_ = fixture()
    records[2][2]['metadata']['unexpected'] = True
    with pytest.raises(ValueError, match='deletion digest'):
        list(patch.records(records))


def test_stream_rejects_missing_approved_event():
    patch, records, *_ = fixture()
    with pytest.raises(ValueError, match='closure'):
        list(patch.records(records[0:1] + records[2:]))


def test_node_collision_fails_closed():
    patch, records, *_, node = fixture()
    with pytest.raises(ValueError, match='collides'):
        list(patch.records([*records[:1], ('node', node['id'], node), *records[1:]]))


def test_edit_and_delete_cannot_overlap():
    with pytest.raises(ValueError, match='overlap'):
        RootTransform([{'claim_id': 'CLM:a'}], [], [{'claim_id': 'CLM:a'}], [], [])


def test_infrastructure_cannot_be_removed_by_claim_contract():
    with pytest.raises(ValueError, match='non-claim'):
        RootTransform([], [], [{'claim_id': 'CUI:one'}], [], [])


def test_unknown_added_entity_class_rejected():
    with pytest.raises(ValueError, match='entity kind'):
        RootTransform([], [], [], [], [{'id': 'CUI:one'}])


def test_primary_article_role_requires_own_title_and_doi():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from apply_kg_root_repairs import review_for
    row = dict(id='CLM:a', metadata=dict(source_paper=dict(pmid='123', title='My review', doi='10.1/test')))
    docs = {'123': dict(source={'sha256': 'witness'}, document=dict(title='My review', dois=['10.1/test'],
        publication_types=['Review'], abstract='source'))}
    assert review_for(row, docs)['source_role'] == 'review_or_evidence_synthesis'
    changed = deepcopy(row); changed['metadata']['source_paper']['doi'] = '10.1/other'
    assert review_for(changed, docs) is None
    changed['metadata']['source_paper']['doi'] = '10.1/test'; changed['metadata']['source_paper']['title'] = 'Other'
    assert review_for(changed, docs) is None
    docs['123']['document']['publication_types'] = ['Journal Article']
    assert review_for(row, docs) is None
