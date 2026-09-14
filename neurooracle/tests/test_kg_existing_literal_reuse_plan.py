from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_existing_literal_reuse import inspection_bridge,current_hash,rebind_review


def fixture():
    inspection=dict(full_source_sha_verified=True,reviewed_endpoints=382,graph={'sha256':'old'},pending_identity_plan={'sha256':'plan'})
    baseline=dict(current_graph={'sha256':'new'})
    accepted=dict(status='CURRENT_EXPANDED_LITERAL_REPAIR_APPLIED',graph=baseline['current_graph'],source_graph=inspection['graph'],provenance_plan=inspection['pending_identity_plan'],
        checks=dict(inverse_reproduces_all_source_node_and_edge_record_digests=True,no_record_deletion=True,original_quotes_negation_conditions_independent_sources_preserved=True))
    return inspection,baseline,accepted


def test_exact_accepted_R53_bridge():inspection_bridge(*fixture())


@pytest.mark.parametrize('kind',['source','plan','proof','count','deletion','science'])
def test_stale_or_semantically_changed_source_not_bridged(kind):
    inspection,baseline,accepted=fixture()
    if kind=='source':accepted['source_graph']={}
    elif kind=='plan':accepted['provenance_plan']={}
    elif kind=='proof':accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests']=False
    elif kind=='count':inspection['reviewed_endpoints']=381
    elif kind=='deletion':accepted['checks']['no_record_deletion']=False
    else:accepted['checks']['original_quotes_negation_conditions_independent_sources_preserved']=False
    with pytest.raises(ValueError):inspection_bridge(inspection,baseline,accepted)


def test_only_other_side_change_rebinds_current_hash():
    r=dict(claim_id='CLM:one',side='object',claim_sha256='old')
    ev={'CLM:one':dict(claim_sha256='old',current_node_sha256='current',changes=[dict(side='subject')])}
    assert rebind_review(r,ev)==dict(r,claim_sha256='current') and r['claim_sha256']=='old'
    assert current_hash('CLM:untouched','kept',ev)=='kept'
    with pytest.raises(ValueError):rebind_review(dict(r,side='subject'),ev)
    with pytest.raises(ValueError):rebind_review(dict(r,claim_sha256='different'),ev)
