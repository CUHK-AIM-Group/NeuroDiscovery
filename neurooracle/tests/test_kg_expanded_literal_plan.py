from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_expanded_literal_repair import inspection_bridge,case_conflicts


def fixture():
    source=dict(full_source_sha_verified=True,reviewed_endpoints=5490,graph={'sha256':'old'},pending_identity_plan={'sha256':'plan'})
    baseline=dict(current_graph={'sha256':'new'})
    accepted=dict(status='CURRENT_REVIEWED_LITERAL_REUSE_APPLIED',graph=baseline['current_graph'],source_graph=source['graph'],
        provenance_plan=source['pending_identity_plan'],checks=dict(inverse_reproduces_all_kept_source_node_and_edge_record_digests=True,
        no_existing_concept_metadata_change=True,no_existing_concept_or_claim_deletion=True))
    return source,baseline,accepted


def test_current_R50_bridge_requires_full_kept_record_proof():
    inspection_bridge(*fixture())


@pytest.mark.parametrize('mutation',['source','plan','proof','count','concept'])
def test_arbitrary_current_advancement_is_rejected(mutation):
    source,baseline,accepted=fixture()
    if mutation=='source':accepted['source_graph']={'sha256':'wrong'}
    elif mutation=='plan':accepted['provenance_plan']={}
    elif mutation=='proof':accepted['checks']['inverse_reproduces_all_kept_source_node_and_edge_record_digests']=False
    elif mutation=='count':source['reviewed_endpoints']=5489
    else:accepted['checks']['no_existing_concept_or_claim_deletion']=False
    with pytest.raises(ValueError):inspection_bridge(source,baseline,accepted)


def test_proposed_case_variants_block_new_duplicate_nodes():
    assert case_conflicts([dict(name='gray matter volume'),dict(name='Gray matter volume'),dict(name='left gray matter volume')])=={'gray matter volume','Gray matter volume'}
    assert not case_conflicts([dict(name='gray   matter volume'),dict(name='gray matter volume')])
