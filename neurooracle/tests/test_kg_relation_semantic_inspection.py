from copy import deepcopy
import pytest
from neurooracle.scripts.inspect_kg_relation_semantic_batch import NAMES, SEMANTIC, selected_groups, incidence_rows


def groups():
    return [dict(name=n, candidate_ids=['a','b'], current_candidate_ids=['a','b']) for n in sorted(NAMES)]


def test_thirteen_finite_names():
    assert len(NAMES) == 13 and len(SEMANTIC) == 2


def test_select_complete():
    assert selected_groups(groups()) == groups()


def test_ignore_other_names():
    assert selected_groups(groups()+[dict(name='other')]) == groups()


def test_missing_group_rejected():
    with pytest.raises(Exception): selected_groups(groups()[:-1])


def test_duplicate_group_rejected():
    with pytest.raises(Exception): selected_groups(groups()+[groups()[0]])


def test_changed_candidate_set_rejected():
    gs=groups(); gs[0]['current_candidate_ids']=['a']
    with pytest.raises(Exception): selected_groups(gs)


def test_order_not_identity():
    gs=groups(); gs[0]['current_candidate_ids']=['b','a']
    assert selected_groups(gs) == gs


def test_nonclaim_ignored():
    assert incidence_rows('NODE:x', {}, {'a'}) == []


def test_both_sides_captured_without_name_gate():
    row=dict(id='CLM:x', metadata=dict(subject_id='a', object_id='a', subject_name='left regional volume', object_name='volume'))
    original=deepcopy(row); got=incidence_rows('CLM:x', row, {'a'})
    assert [r['side'] for r in got] == ['subject','object'] and got[0]['name'] != got[1]['name']
    assert row == original and got[0]['claim_sha256'] == got[1]['claim_sha256']


def test_unrelated_endpoints_ignored():
    assert incidence_rows('CLM:x', dict(metadata=dict(subject_id='a',object_id='b')), {'c'}) == []


def test_role_preserved_not_approved():
    got=incidence_rows('CLM:x', dict(metadata=dict(subject_id='a',subject_name='volume',subject_type='GENE',metadata=dict(subject_type='IMAGING_MARKER'))), {'a'})
    assert got[0]['outer_type']=='GENE' and got[0]['inner_type']=='IMAGING_MARKER'
