from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_observational_semantics import inspection_bridge
from neurooracle.src.kg_observational_semantics import SPECS,REBOX


def fixture():
    review=dict(full_source_sha_verified=True,claims=17,selected=list(SPECS),graph={'sha256':'old'},pending_disjoint_plan={'sha256':'plan'})
    baseline=dict(current_graph={'sha256':'new'})
    accepted=dict(status='CURRENT_SCIENTIFIC_DEFINITION_REPAIR_APPLIED',graph=baseline['current_graph'],source_graph=review['graph'],provenance_plan=review['pending_disjoint_plan'],
        checks=dict(inverse_reproduces_all_source_node_and_edge_record_digests=True,no_record_deletion=True,all_original_statistics_and_historic_audit_records_preserved=True))
    return review,baseline,accepted,dict(events=[])


def test_scientific_predecessor_can_advance_only_disjoint_finite_records():inspection_bridge(*fixture())


@pytest.mark.parametrize('kind',['source','plan','proof','deletion','audit','overlap','selected'])
def test_unreviewed_advancement_does_not_authorize_observational_changes(kind):
    review,baseline,accepted,prior=fixture()
    if kind=='source':accepted['source_graph']={}
    elif kind=='plan':accepted['provenance_plan']={}
    elif kind=='proof':accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests']=False
    elif kind=='deletion':accepted['checks']['no_record_deletion']=False
    elif kind=='audit':accepted['checks']['all_original_statistics_and_historic_audit_records_preserved']=False
    elif kind=='overlap':prior['events']=[dict(claim_id=REBOX)]
    else:review['selected']=[]
    with pytest.raises(ValueError):inspection_bridge(review,baseline,accepted,prior)
