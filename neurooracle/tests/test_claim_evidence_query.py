from copy import deepcopy
import hashlib
import json
import sqlite3
import pytest
from neurooracle.src.claim_evidence_query import query_claim_evidence,paper_evidence
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.relation_evidence import relation_key,relation_id,summarize_relation
from neurooracle.src.relation_evidence_dossier import observation,summarize

def fingerprint(path):
    stat=path.stat()
    return dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def setup(tmp_path):
    def save(name,value,lines=False):
        path=tmp_path/name
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value) if lines else json.dumps(value,ensure_ascii=False,separators=(',',':'))
        path.write_text(text,encoding='utf8');return fingerprint(path)
    payload=dict(version='kg.paper_identity.v1',records={p:dict(pmid=p,title='own '+p,years=['2020'],doi=[],pmcid=[],
        witness={'response_sha256':'a'*64}) for p in ('111','222','333')})
    papers=VerifiedPaperIdentities(payload)
    records=[];reviews={}
    for i,p in enumerate(('111','222','111','333')):
        cid='CLM:'+str(i)
        md=dict(id=cid,subject_id='X',subject_name='exposure' if i<3 else 'separate exposure',predicate='correlates_with',
            object_id='Y',object_name='outcome',negated=False,raw_text='Own result '+cid,
            source_paper=dict(pmid=p,title='Own '+p,year=2020),evidence=dict(sample_size=0,p_value=None,unknown=False),
            population='Population '+p,conditions=['adjusted'],metadata={'qualifier':'retained'})
        record=dict(id=cid,metadata=md);records.append(record)
        role='own_result' if i==1 else 'background_assertion'
        reviews[cid]=dict(claim_sha256=digest(record),source_role='primary_research_article',observation_role=role,proposition_support='supports')
    shared=summarize_relation(relation_key(records[0]['metadata']),[r['metadata'] for r in records[:3]],papers=papers)
    dossier=summarize(shared,[observation(r,papers,reviews) for r in records[:3]])
    graph=save('graph.json',dict(metadata={},concepts={r['id']:r for r in records},edges=[]))
    dbpath=tmp_path/'census.sqlite'
    with sqlite3.connect(dbpath) as db:
        db.execute('CREATE TABLE claims(cid TEXT PRIMARY KEY,node_sha TEXT,relation_id TEXT,shared INTEGER)')
        db.executemany('INSERT INTO claims VALUES(?,?,?,?)',[(r['id'],digest(r),relation_id(relation_key(r['metadata'])),int(i<3)) for i,r in enumerate(records)])
    census=save('census.json',dict(graph=graph,database=fingerprint(dbpath)))
    catalog=save('catalog.jsonl',[shared],True);dossiers=save('dossiers.jsonl',[dossier],True)
    terms=save('terms.json',dict(version='kg.verified_entity_terms.v1',terms=[]))
    paperfp=save('papers.json',payload);rolefp=save('reviews.json',reviews)
    checks={k:True for k in ('shared_relation_index_complete','verified_identity_proofs_complete','paper_identity_witnesses_validated',
        'all_current_census_rows_independently_verified','shared_observation_dossiers_complete')}
    receipt=save('acceptance.json',dict(graph=graph,current_paper_census=census,shared_relations=catalog,
        entity_terms=terms,paper_identities=paperfp,evidence_dossiers=dossiers,source_role_reviews=rolefp,checks=checks))
    campaign=save('campaign.json',dict(status='COMPLETED',active_process=None,current_graph=graph,current_acceptance=receipt,
        current_paper_census=census,current_shared_relations=catalog,current_entity_terms=terms,current_paper_identities=paperfp,
        current_evidence_dossiers=dossiers,current_source_role_reviews=rolefp))
    return tmp_path/'campaign.json',records,dossier

def test_original_claim_ids_find_all_papers_and_deduplicate_observations(tmp_path):
    path,records,_=setup(tmp_path)
    one=query_claim_evidence(path,claim_id='CLM:0');two=query_claim_evidence(path,claim_id='CLM:1')
    assert one['shared_claim_id']==two['shared_claim_id']
    assert one['original_claim_ids']==['CLM:0','CLM:1','CLM:2']
    assert one['observation_count']==3 and one['verified_article_count']==2
    assert one['reviewed_supporting_article_count']==2 and one['own_result_article_count']==1
    assert one['independent_primary_study_count'] is None and not one['consensus_inferred']
    originals={o['claim_id']:o['original_claim'] for p in one['papers'] for o in p['observations']}
    assert originals=={r['id']:r['metadata'] for r in records[:3]}
    assert query_claim_evidence(path,relation_id=one['shared_claim_id'])['papers']==one['papers']

def test_single_source_claim_is_returned_instead_of_hiding_it(tmp_path):
    path,records,_=setup(tmp_path);out=query_claim_evidence(path,claim_id='CLM:3')
    assert out['article_count']==1 and out['observation_count']==1
    assert out['papers'][0]['observations'][0]['original_claim']==records[3]['metadata']

@pytest.mark.parametrize('lookup',[dict(claim_id='CLM:missing'),dict(relation_id='REL:missing')])
def test_unknown_ids_are_explicit(tmp_path,lookup):
    path,_,_=setup(tmp_path)
    with pytest.raises(KeyError):query_claim_evidence(path,**lookup)

@pytest.mark.parametrize('lookup',[{},dict(claim_id='CLM:0',relation_id='REL:0'),dict(claim_id='not-a-claim')])
def test_ambiguous_or_invalid_request_rejected(tmp_path,lookup):
    path,_,_=setup(tmp_path)
    with pytest.raises(ValueError):query_claim_evidence(path,**lookup)

@pytest.mark.parametrize('file',['graph.json','census.sqlite','dossiers.jsonl','acceptance.json'])
def test_stale_graph_census_dossier_and_acceptance_fail_closed(tmp_path,file):
    path,_,_=setup(tmp_path)
    with (tmp_path/file).open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError):query_claim_evidence(path,claim_id='CLM:0')

@pytest.mark.parametrize('case',['hypothesis','negated','null_result','retracted','unverified','unreviewed'])
def test_mentions_nulls_hypotheses_and_retractions_do_not_inflate_support(tmp_path,case):
    _,_,dossier=setup(tmp_path);d=deepcopy(dossier)
    for o in d['observations']:
        if case=='hypothesis':o['source_review'].update(proposition_support='hypothesis_only',observation_role='hypothesis_proposal')
        elif case=='negated':o['claim']['negated']=True
        elif case=='null_result':o['source_review']['proposition_support']='does_not_support'
        elif case=='retracted':o['publication_review']['status']='retracted'
        elif case=='unverified':o['source_identity']['status']='unverified'
        else:o['source_review']=None
    out=paper_evidence(d)
    assert out['reviewed_supporting_article_count']==0
    assert out['own_result_article_count']==0 and not out['independence_established']

def test_reviews_and_background_are_not_counted_as_primary_studies(tmp_path):
    _,_,d=setup(tmp_path)
    for o in d['observations']:o['source_review']['observation_role']='review_synthesis'
    result=paper_evidence(d)
    assert result['review_article_count']==2 and result['own_result_article_count']==0
    assert result['independent_primary_study_count']==0

def test_stale_source_review_cannot_transfer_to_modified_claim(tmp_path):
    _,_,d=setup(tmp_path);d['observations'][0]['source_review']['claim_sha256']='stale'
    with pytest.raises(ValueError):paper_evidence(d)

def test_verified_versions_are_one_paper_with_all_original_evidence(tmp_path):
    _,_,d=setup(tmp_path)
    for o in d['observations']:
        o['source_review'].update(verified_work_key='one_review',preferred_version_pmid='222',
            verified_version_pmids=['111','222'],version_identity_witness={'sha256':'a'*64},observation_role='review_synthesis')
    result=paper_evidence(d)
    assert result['article_count']==result['reviewed_supporting_article_count']==1
    assert result['publication_record_count']==2 and result['verified_versions_deduplicated']==1
    assert len(result['papers'][0]['publication_versions'])==2
    assert len(result['papers'][0]['observations'])==3
