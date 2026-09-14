from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_scientific_definition_repair import inspection_bridge,incidence_bridge
from neurooracle.src.kg_scientific_definition_repair import BACE,TARGETS,SEPARATE


def fixture():
    review=dict(full_source_sha_verified=True,claims=31,graph={'sha256':'old'},pending_disjoint_plan={'sha256':'plan'})
    baseline=dict(current_graph={'sha256':'new'})
    accepted=dict(status='CURRENT_EXISTING_LITERAL_REUSE_APPLIED',graph=baseline['current_graph'],source_graph=review['graph'],provenance_plan=review['pending_disjoint_plan'],
        checks=dict(inverse_reproduces_all_source_node_and_edge_record_digests=True,no_record_deletion=True,no_new_concept_nodes=True))
    return review,baseline,accepted,dict(events=[])


def test_disjoint_existing_reuse_advancement_is_rebound():inspection_bridge(*fixture())


@pytest.mark.parametrize('kind',['source','plan','proof','deletion','new','overlap'])
def test_unreviewed_advancement_cannot_authorize_scientific_changes(kind):
    review,baseline,accepted,prior=fixture()
    if kind=='source':accepted['source_graph']={}
    elif kind=='plan':accepted['provenance_plan']={}
    elif kind=='proof':accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests']=False
    elif kind=='deletion':accepted['checks']['no_record_deletion']=False
    elif kind=='new':accepted['checks']['no_new_concept_nodes']=False
    else:prior['events']=[dict(claim_id=BACE)]
    with pytest.raises(ValueError):inspection_bridge(review,baseline,accepted,prior)


def incidence_fixture():
    from copy import deepcopy
    old=[dict(node_id=TARGETS[SEPARATE],claim_id='CLM:existing',side='object',name=SEPARATE,claim_sha256='old',
        nonidentity_sha256='same',outer_type=None,inner_type='brain_structure')]
    row=dict(node_id=TARGETS[SEPARATE],claim_id='CLM:CASE1MAN:33668432:9546',side='subject',name=SEPARATE,
        claim_sha256='new',nonidentity_sha256='science',outer_type=None,inner_type='IMAGING_MARKER')
    event=dict(claim_id=row['claim_id'],claim_sha256='source',current_node_sha256='new',nonidentity_sha256='science',
        changes=[dict(side='subject',name=SEPARATE,old_id='CUI:gene',target_id=TARGETS[SEPARATE])])
    review=dict(claim_id=row['claim_id'],claim_sha256='source',side='subject',name=SEPARATE,target_id=TARGETS[SEPARATE],
        outer_type=None,inner_type='IMAGING_MARKER')
    return [deepcopy(row),*deepcopy(old)],old,dict(events=[event],endpoint_reviews={'review':review}),{TARGETS[SEPARATE]}


def test_r55_new_incidence_is_reproduced_without_dropping_old_incidence():
    actual,old,prior,targets=incidence_fixture()
    assert incidence_bridge(actual,old,prior,targets)==actual[:1]


@pytest.mark.parametrize('kind',['missing_old','changed_old','extra','hash','science','type','name','review','unreviewed_claim','duplicate'])
def test_incidence_advancement_requires_exact_closed_proof(kind):
    actual,old,prior,targets=incidence_fixture()
    if kind=='missing_old':actual.pop()
    elif kind=='changed_old':actual[-1]['name']='different'
    elif kind=='extra':actual.append(dict(actual[0],claim_id='CLM:extra'))
    elif kind=='hash':actual[0]['claim_sha256']='wrong'
    elif kind=='science':actual[0]['nonidentity_sha256']='wrong'
    elif kind=='type':actual[0]['inner_type']='PATHWAY'
    elif kind=='name':actual[0]['name']='mean bilateral hippocampal volume'
    elif kind=='review':prior['endpoint_reviews']['review']['claim_sha256']='wrong'
    elif kind=='unreviewed_claim':prior['events'][0]['claim_id']='CLM:extra'
    else:actual.append(actual[0])
    with pytest.raises(ValueError):incidence_bridge(actual,old,prior,targets)
