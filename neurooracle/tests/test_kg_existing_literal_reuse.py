from copy import deepcopy
import pytest
from neurooracle.src import kg_existing_literal_reuse as repair
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.src.schema import ConceptNode
from neurooracle.scripts.inspect_kg_remaining_scoped_literals import scope_projection
from neurooracle.tests.test_kg_expanded_literal_repair import fixture as source_fixture


def fixture():
    row,gene,witness,review,ch=source_fixture('frontal gray matter volume')
    target=ConceptNode(id='CLM_CONCEPT:existing',preferred_name=ch['name'],source_vocab='manual_claim_anchor',
        domain_tags=['biomarker'],metadata={'atom_type':'imaging_marker','curation_scope':'original_case'}).to_dict()
    review.update(literal_candidate_gate='existing_full_name_or_case_alias_requires_reuse_review',reuse_candidate_gate=None,
        existing_name_or_alias_candidates=[target['id']],target_id=target['id'],target_node_sha256=digest(target),target_witness=scope_projection(target))
    ch['target_id']=target['id']
    return row,gene,witness,review,ch,target


def event_for(row,witness,review,ch):
    return repair.reviewed_claim(row,[ch],{witness['node_id']:witness},{repair.review_key(row['id'],'subject'):review})


def test_reuse_preserves_original_science_and_existing_identity():
    row,gene,witness,review,ch,target=fixture();before=deepcopy((row,target))
    event,current=event_for(row,witness,review,ch)
    assert (row,target)==before and nonidentity_claim(current)==nonidentity_claim(row)
    assert repair.reverse_claim(current,event)==row and current['metadata']['subject_id']==target['id']


@pytest.mark.parametrize('kind',['source_hash','gene_hash','whole_alias','role','nested','multiple','wrong_candidate','primary_name','aliases','scope','target','target_hash','type','definition','molecular','no_review'])
def test_reuse_never_bypasses_scope_or_identity_proof(kind):
    row,gene,witness,review,ch,target=fixture()
    if kind=='source_hash':review['claim_sha256']='0'*64
    elif kind=='gene_hash':witness['node_sha256']='0'*64
    elif kind=='whole_alias':witness['labels'].append('gray')
    elif kind=='role':review['declared_roles']=['gene_target']
    elif kind=='nested':row['metadata']['metadata']['subject_id']='different';review['claim_sha256']=digest(row)
    elif kind=='multiple':review['existing_name_or_alias_candidates'].append('CLM_CONCEPT:second')
    elif kind=='wrong_candidate':review['existing_name_or_alias_candidates']=['CLM_CONCEPT:other']
    elif kind=='primary_name':review['target_witness']['name']='Frontal gray matter volume'
    elif kind=='aliases':review['target_witness']['aliases']=['regional volume']
    elif kind=='scope':review['reuse_candidate_gate']='existing_incident_full_name_differs'
    elif kind=='target':ch['target_id']='CLM_CONCEPT:paper_specific_new'
    elif kind=='target_hash':review['target_node_sha256']='0'*64
    elif kind=='type':review['target_witness']['semantic_types']=['T028']
    elif kind=='definition':review['target_witness']['definition_sha256']=digest('limited to one group')
    elif kind=='molecular':review['target_witness']['reviewed_metadata']['atom_type']='gene_target'
    else:review={}
    with pytest.raises((ValueError,KeyError)):event_for(row,witness,review,ch)


def test_new_nodes_not_authorized():
    with pytest.raises(ValueError,match='never creates'):repair.literal_node('frontal gray matter volume')
