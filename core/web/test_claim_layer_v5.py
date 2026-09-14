from copy import deepcopy
import pytest
from core.web.claim_layer_v5 import project, EvidenceUnavailable
from core.web.test_claim_layer_v1 import release
from neurooracle.src.kg_identity_pilot import digest


def original_without_pmid(tmp_path):
    _,_,_,catalog,dossier,payload=release(tmp_path)
    observation=dossier['observations'][0]
    observation['claim']['source_paper']['pmid']=''
    sha=digest(observation['claim'])
    observation['claim_sha256']=sha
    observation['source_review']['claim_sha256']=sha
    payload['source_reviews'][observation['claim_id']]['claim_sha256']=sha
    return catalog,dossier,payload


def test_verified_doi_identity_can_support_without_fabricating_original_pmid(tmp_path):
    catalog,dossier,payload=original_without_pmid(tmp_path);before=deepcopy((catalog,dossier,payload))
    _,_,aliases,details=project(catalog,[dossier],payload)
    result=next(iter(details.values()))
    assert result['reviewed_supporting_article_count']==2
    obs=next(o for p in result['papers'] for o in p['observations'] if o['claim_id']=='CLM:0')
    assert obs['original_claim']['source_paper']['pmid']==''
    assert aliases[dossier['relation_id']]==result['shared_claim_id']
    assert (catalog,dossier,payload)==before


@pytest.mark.parametrize('fault',['wrong_review_paper','wrong_original_pmid','unverified_identity'])
def test_absent_pmid_route_does_not_accept_wrong_or_unverified_identity(tmp_path,fault):
    catalog,dossier,payload=original_without_pmid(tmp_path);o=dossier['observations'][0]
    if fault=='wrong_review_paper':payload['source_reviews'][o['claim_id']]['pmid']='999'
    elif fault=='wrong_original_pmid':o['claim']['source_paper']['pmid']='999'
    else:o['source_identity']['status']='unverified'
    with pytest.raises(EvidenceUnavailable):project(catalog,[dossier],payload)
