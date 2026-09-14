from copy import deepcopy
import pytest
from core.web.claim_layer_extension_v3 import complete_identity
from core.web.claim_evidence import EvidenceUnavailable


class Authority:
    def resolve(self, claim):
        assert claim == {'source_paper':{'pmid':'10','title':'original full title'},'raw_text':'original result'}
        return {'paper_key':'pmid:10','status':'verified','reasons':[]}
    def publication_review(self, key):
        assert key == 'pmid:10'
        return {'status':'not_reviewed'}


def fixture():
    old=dict(claim_id='c',claim_sha256='bound-node',claim={'source_paper':{'pmid':'10','title':'original full title'},'raw_text':'original result'},
        source_identity={'paper_key':'pmid:10','status':'unverified','reasons':['no_authority_witness']},
        publication_review={'status':'not_reviewed'},source_review={'source_role':'review_or_evidence_synthesis'})
    entry=dict(claim_id='c',claim_sha256='bound-node',pmid='10',previous_source_identity=deepcopy(old['source_identity']),
        source_identity={'paper_key':'pmid:10','status':'verified','reasons':[]})
    return old,entry


def test_identity_completion_keeps_full_original_and_previous_identity():
    old,entry=fixture(); before=deepcopy(old)
    result=complete_identity(old,entry,Authority())
    assert old==before
    assert result['claim']==old['claim'] and result['source_review']==old['source_review']
    assert result['previous_source_identity']==old['source_identity']
    assert result['source_identity']['status']=='verified'


@pytest.mark.parametrize('case',['stale_claim','wrong_source','already_verified','already_adjudicated','forged_identity'])
def test_identity_completion_rejects_invalid_or_protected_inputs(case):
    old,entry=fixture()
    if case=='stale_claim': entry['claim_sha256']='stale'
    if case=='wrong_source': entry['pmid']='11'
    if case=='already_verified': old['source_identity']=deepcopy(entry['source_identity']);entry['previous_source_identity']=deepcopy(old['source_identity'])
    if case=='already_adjudicated': old['source_review']['proposition_support']='supports'
    if case=='forged_identity': entry['source_identity']['paper_key']='pmid:11'
    with pytest.raises(EvidenceUnavailable):complete_identity(old,entry,Authority())
