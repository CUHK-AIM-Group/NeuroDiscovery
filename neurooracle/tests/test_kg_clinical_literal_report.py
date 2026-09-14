from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_clinical_literal_reuse import audited_families


def plan():
    return dict(family_proofs=[dict(name='complete measurement',target_id='n1',incident_claim_ids=['a','b'],source_anchor_nodes_and_detail_rows_preserved=True)],
        events=[dict(claim_id='b',changes=[dict(name='complete measurement')]),dict(claim_id='c',changes=[dict(name='complete measurement')])])


def test_unchanged_and_changed_claims_counted_once_without_node_deletion():
    result=audited_families(plan())[0]
    assert result['reviewed_claim_ids']==['a','b','c'] and result['reviewed_claim_count']==3
    assert result['physical_nodes_deleted']==0 and result['not_an_exhaustive_same_name_scientific_equivalence']


def test_unrelated_family_not_folded_into_count():
    p=plan();p['events'].append(dict(claim_id='d',changes=[dict(name='different measurement')]))
    assert audited_families(p)[0]['reviewed_claim_count']==3


def test_source_anchor_retirement_cannot_be_misreported_as_this_batch():
    p=plan();p['family_proofs'][0]['source_anchor_nodes_and_detail_rows_preserved']=False
    with pytest.raises(Exception):audited_families(p)

