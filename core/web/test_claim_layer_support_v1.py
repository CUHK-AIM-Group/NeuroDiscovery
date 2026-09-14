from copy import deepcopy
import json
import pytest

from core.web.test_claim_layer_v1 import release
from core.web.claim_layer_support_v1 import ORIGIN, make_observation, load_support_records
from core.web.claim_layer_v4 import project, EvidenceUnavailable
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.scripts.whole_graph_sources_20260914 import own_records
from neurooracle.tests.test_claim_evidence_query import fingerprint


def fixture(tmp_path,pmid='444'):
    path,campaign,records,catalog,dossier,payload=release(tmp_path)
    text='Scoped exposure correlated with the measured outcome in this study.'
    xml=tmp_path/'own.xml'
    xml.write_text(f'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>Own {pmid}</ArticleTitle><Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal><Abstract><AbstractText>{text}</AbstractText></Abstract><PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="pubmed">{pmid}</ArticleId></ArticleIdList><ReferenceList><Reference><ArticleIdList><ArticleId IdType="pubmed">999</ArticleId></ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle></PubmedArticleSet>',encoding='utf8')
    _,doc,authority,state=next(own_records(xml.read_bytes(),fingerprint(xml)))
    registry=json.loads((tmp_path/'papers.json').read_text());registry['records'][pmid]=authority
    registry_path=tmp_path/'support_registry.json';registry_path.write_text(json.dumps(registry),encoding='utf8')
    documents=tmp_path/'own_documents.json';documents.write_text(json.dumps({'records':{pmid:doc}}),encoding='utf8')
    group=payload['groups'][0]
    action=dict(role='own_result',source_role='primary_research_article',support='supports',anchor=text,own_abstract_sha256=digest(doc['abstract']))
    decision=dict(adjudicator='current_task_host',owning_abstracts_read=[pmid],supplemental_sources={pmid:action},
        target_shared_claim_id=group['shared_claim_id'],original_claim_ids=sorted(group['original_claim_ids']),canonical_claim=group['claim'],
        review_id='unit-host-review',note='Study-bounded correlation; no causal or cohort-independence inference.',reading_provenance='Explicit unit fixture, never a scientific publication decision.')
    ledger=tmp_path/'support_decisions.json';ledger.write_text(json.dumps({'unit':decision}),encoding='utf8')
    observation=make_observation(group['shared_claim_id'],group['claim'],decision['original_claim_ids'],doc,action,decision['review_id'],decision['note'],decision['reading_provenance'],VerifiedPaperIdentities(registry))
    payload['original_relation_extension']={'supplemental_identity_registry':fingerprint(registry_path)}
    payload['source_support_inputs']=[fingerprint(p) for p in (xml,registry_path,documents,ledger)]
    payload['supplemental_source_records']={observation['claim_id']:dict(pmid=pmid,target_shared_claim_id=group['shared_claim_id'],anchor_original_claim_ids=decision['original_claim_ids'],owning_documents=fingerprint(documents),host_decision={'ledger':fingerprint(ledger),'key':'unit'},observation=observation)}
    group['reviewed_supporting_article_count']=2 if pmid in {'111','222'} else 3
    return catalog,dossier,payload,doc,decision,registry


def test_new_paper_without_original_graph_node_adds_honest_source_evidence(tmp_path):
    catalog,dossier,payload,*_=fixture(tmp_path)
    _,dossiers,aliases,details=project(catalog,[dossier],payload)
    result=next(iter(details.values()))
    assert result['reviewed_supporting_article_count']==3
    assert result['original_claim_ids']==['CLM:0','CLM:1','CLM:2']
    assert result['original_observation_count']==3 and result['observation_count']==4
    assert len(result['supplemental_observation_ids'])==1
    observations=[o for p in result['papers'] for o in p['observations']]
    new=next(o for o in observations if o['claim_id'].startswith('CLM:SUPPLEMENTAL:'))
    assert new['observation_origin']==ORIGIN and new['original_graph_observation'] is False
    assert 'source_derived_claim' in new and 'original_claim' not in new
    assert {o['claim_id']:o['original_claim'] for o in observations if 'original_claim' in o}=={o['claim_id']:o['claim'] for o in dossier['observations']}
    assert result['independence_established'] is False and result['consensus_inferred'] is False


def test_second_record_of_same_paper_does_not_increase_paper_count(tmp_path):
    catalog,dossier,payload,*_=fixture(tmp_path,pmid='111')
    _,_,_,details=project(catalog,[dossier],payload)
    assert next(iter(details.values()))['reviewed_supporting_article_count']==2


@pytest.mark.parametrize('fault',['made_up_anchor','notice','wrong_role','wrong_support','retracted','missing_provenance'])
def test_ineligible_source_cannot_become_supplemental_support(tmp_path,fault):
    _,_,payload,doc,decision,registry=fixture(tmp_path)
    action=deepcopy(decision['supplemental_sources']['444'])
    if fault=='made_up_anchor':action['anchor']='An invented result.'
    elif fault=='notice':doc['comments_corrections']=[{'ref_type':'RetractionIn','pmid':'555'}]
    elif fault=='wrong_role':action['role']='hypothesis_proposal'
    elif fault=='wrong_support':action['support']='context_only'
    elif fault=='retracted':doc['publication_types'].append('Retracted Publication')
    else:decision['reading_provenance']=''
    with pytest.raises(EvidenceUnavailable):
        make_observation(decision['target_shared_claim_id'],decision['canonical_claim'],decision['original_claim_ids'],doc,action,decision['review_id'],decision['note'],decision['reading_provenance'],VerifiedPaperIdentities(registry))


@pytest.mark.parametrize('fault',['mutated_document','unsealed_input','wrong_record','outside_scope','unknown_target','missing_record','reference_pmid'])
def test_sealed_support_cannot_be_forged_or_bound_to_another_claim(tmp_path,fault):
    catalog,dossier,payload,*_=fixture(tmp_path)
    cid,record=next(iter(payload['supplemental_source_records'].items()))
    if fault=='mutated_document':
        with (tmp_path/'own_documents.json').open('a') as stream:stream.write(' ')
    elif fault=='unsealed_input':payload['source_support_inputs'].pop()
    elif fault=='wrong_record':record['observation']['claim']['raw_text']='forged'
    elif fault=='outside_scope':record['anchor_original_claim_ids']=['CLM:other']
    elif fault=='unknown_target':record['target_shared_claim_id']='REL:other'
    elif fault=='missing_record':payload['supplemental_source_records']={}
    else:record['pmid']='999'
    with pytest.raises((EvidenceUnavailable,ValueError,KeyError)):
        project(catalog,[dossier],payload)


def test_existing_layer_without_supplements_is_unchanged(tmp_path):
    _,_,_,catalog,dossier,payload=release(tmp_path,versions=True)
    _,_,_,details=project(catalog,[dossier],payload)
    result=next(iter(details.values()))
    assert result['reviewed_supporting_article_count']==1 and result['publication_record_count']==2
    assert not result['supplemental_observation_ids']
