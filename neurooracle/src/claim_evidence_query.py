"""Resolve an original claim ID to the complete current multi-paper evidence.

The current census routes IDs; accepted dossiers carry shared observations.
A singleton is read from the actual graph and matched to its census node seal.
No claim deletion, new literature retrieval, or inferred scientific agreement.
"""
from collections import defaultdict
from copy import deepcopy
import codecs
import json
import mmap
from pathlib import Path
import sqlite3

from .kg_identity_pilot import digest
from .kg_paper_identity import VerifiedPaperIdentities
from .relation_evidence import summarize_relation
from .relation_evidence_dossier import observation, summarize
from .shared_relation_catalog import check_file, current_shared_relations

def require(condition,message):
    if not condition:raise ValueError(message)

def paper_evidence(dossier):
    """Count articles once and separate reviewed assertion roles from results."""
    papers={}
    for item in dossier['observations']:
        md=item['claim'];identity=item['source_identity'];key=identity['paper_key']
        review=item.get('source_review') or {}
        require(not review or review['claim_sha256']==item['claim_sha256'],'stale observation review')
        role=review.get('observation_role','not_reviewed')
        support=review.get('proposition_support','not_reviewed')
        eligible=identity['status']=='verified' and item['publication_review']['status']!='retracted'
        qualified=eligible and support=='supports' and not md.get('negated') and role!='hypothesis_proposal'
        if key not in papers:
            papers[key]=dict(paper_key=key,source_identity=deepcopy(identity),
                bibliography=deepcopy(md['source_paper']),publication_review=deepcopy(item['publication_review']),
                verified=identity['status']=='verified',counted_toward_default=eligible,
                supports_reviewed_proposition=False,own_result=False,observations=[])
        p=papers[key]
        require(p['source_identity']==identity,'one source has inconsistent identity')
        p['supports_reviewed_proposition']|=qualified
        p['own_result']|=qualified and role=='own_result'
        p['observations'].append(dict(claim_id=item['claim_id'],raw_text=md.get('raw_text'),
            evidence=deepcopy(md.get('evidence')),negated=md.get('negated'),
            conditions=deepcopy(md.get('conditions')),population=deepcopy(md.get('population')),
            observation_role=role,proposition_support=support,source_review=deepcopy(item.get('source_review')),
            original_claim=deepcopy(md)))
    publication_count=len(papers)
    families={}
    for key,paper in papers.items():
        declarations=[]
        for obs in paper['observations']:
            review=obs.get('source_review') or {}
            if review.get('verified_work_key'):
                ids=review.get('verified_version_pmids',[]);preferred=review.get('preferred_version_pmid')
                witness=review.get('version_identity_witness') or {}
                require(str(paper['bibliography'].get('pmid')) in ids and preferred in ids
                    and len(witness.get('sha256',''))==64,'invalid reviewed version identity')
                declarations.append((review['verified_work_key'],tuple(ids),preferred,witness['sha256']))
        require(len(set(declarations))<=1,'conflicting reviewed article versions')
        if declarations:families[key]=declarations[0]
    grouped=defaultdict(list)
    for key,paper in papers.items():grouped[families[key][0] if key in families else key].append(paper)
    result=[]
    for work,versions in grouped.items():
        declarations={families[p['paper_key']] for p in versions if p['paper_key'] in families}
        require(len(declarations)<=1,'inconsistent work identity across sources')
        preferred=next(iter(declarations))[2] if declarations else None
        chosen=next((p for p in versions if str(p['bibliography'].get('pmid'))==preferred),versions[0])
        merged=deepcopy(chosen);merged['work_key']=work
        merged['publication_versions']=[{k:deepcopy(p[k]) for k in ('paper_key','bibliography','source_identity','publication_review')} for p in sorted(versions,key=lambda p:p['paper_key'])]
        merged['observations']=sorted([o for p in versions for o in p['observations']],key=lambda o:o['claim_id'])
        for flag in ('verified','counted_toward_default','supports_reviewed_proposition','own_result'):merged[flag]=any(p[flag] for p in versions)
        result.append(merged)
    result.sort(key=lambda p:(not p['supports_reviewed_proposition'],p['paper_key']))
    roles={o['observation_role'] for p in result for o in p['observations']}
    known_nonprimary={'background_assertion','review_synthesis','hypothesis_proposal'}
    return dict(papers=result,article_count=len(result),publication_record_count=publication_count,
        verified_versions_deduplicated=publication_count-len(result),
        verified_article_count=sum(p['verified'] for p in result),
        default_counted_article_count=sum(p['counted_toward_default'] for p in result),
        reviewed_supporting_article_count=sum(p['supports_reviewed_proposition'] for p in result),
        own_result_article_count=sum(p['own_result'] for p in result),
        background_article_count=sum(any(o['observation_role']=='background_assertion' for o in p['observations']) for p in result),
        review_article_count=sum(any(o['observation_role']=='review_synthesis' for o in p['observations']) for p in result),
        hypothesis_article_count=sum(any(o['proposition_support']=='hypothesis_only' for o in p['observations']) for p in result),
        independent_primary_study_count=0 if roles and roles<=known_nonprimary else None,
        independence_established=False,consensus_inferred=False)

def _claim_record(path,cid,seal):
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        start=mm.find(b',"concepts":{')
        require(start>=0,'unsupported graph layout for bounded claim lookup')
        needle=json.dumps(cid,ensure_ascii=False).encode('utf8')+b':'
        pos=mm.find(needle,start)
        require(pos>=0,'census claim missing from graph')
        offset=pos+len(needle);size=65536
        while size<=16*1024*1024:
            text=codecs.getincrementaldecoder('utf-8')().decode(mm[offset:offset+size],final=False)
            try:
                record,_=json.JSONDecoder().raw_decode(text.lstrip());break
            except json.JSONDecodeError:size*=2
        else:raise ValueError('claim exceeds bounded lookup size')
    require(record.get('id')==cid and digest(record)==seal,'actual claim does not match current census')
    return record

def query_claim_evidence(campaign_path, *, claim_id=None, relation_id=None):
    if bool(claim_id)==bool(relation_id):raise ValueError('provide exactly one claim ID or relation ID')
    if claim_id is not None and (not isinstance(claim_id,str) or not claim_id.startswith('CLM:')):
        raise ValueError('invalid original claim ID')
    if relation_id is not None and (not isinstance(relation_id,str) or not relation_id.startswith('REL:')):
        raise ValueError('invalid shared relation ID')
    path=Path(campaign_path);campaign=json.loads(path.read_text(encoding='utf8'))
    # This exhausts the catalog iterator so all snapshot checks execute.
    catalog={g['id']:g for g in current_shared_relations(path)}
    receipt=json.loads(check_file(campaign['current_acceptance'],full_hash=True).read_text(encoding='utf8'))
    census_fp=campaign['current_paper_census']
    require(receipt.get('current_paper_census')==census_fp
        and receipt['checks'].get('all_current_census_rows_independently_verified'),'unvalidated claim census')
    census=json.loads(check_file(census_fp,full_hash=True).read_text(encoding='utf8'))
    require(census['graph']==campaign['current_graph'],'census graph differs from current snapshot')
    database=check_file(census['database'])
    con=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
    try:
        if claim_id:
            item=con.execute('SELECT node_sha,relation_id,shared FROM claims WHERE cid=?',(claim_id,)).fetchone()
            if item is None:raise KeyError('claim not found: '+claim_id)
            relation_id=item[1]
        members=list(con.execute('SELECT cid,node_sha,shared FROM claims WHERE relation_id=? ORDER BY cid',(relation_id,)))
        if not members:raise KeyError('relation not found: '+relation_id)
    finally:con.close()
    expected={cid:seal for cid,seal,shared in members}
    if relation_id in catalog:
        group=catalog[relation_id]
        require({m['claim_id'] for m in group['members']}==set(expected) and all(shared for cid,seal,shared in members),
            'catalog and complete census membership differ')
        fp=campaign.get('current_evidence_dossiers')
        require(fp and receipt.get('evidence_dossiers')==fp and receipt['checks'].get('shared_observation_dossiers_complete'),'unvalidated evidence dossier')
        found=[]
        with check_file(fp,full_hash=True).open(encoding='utf8') as f:
            for line in f:
                row=json.loads(line)
                if row['relation_id']==relation_id:found.append(row)
        require(len(found)==1,'missing or duplicate current dossier')
        dossier=found[0]
        require(summarize(group,dossier['observations'])==dossier,'dossier summary differs')
        require({o['claim_id']:o['claim_sha256'] for o in dossier['observations']}==expected,'dossier census seals differ')
    else:
        require(len(members)==1 and not members[0][2],'shared claim absent from catalog')
        cid,seal,_=members[0]
        record=_claim_record(check_file(campaign['current_graph']),cid,seal)
        registry=VerifiedPaperIdentities(json.loads(check_file(campaign['current_paper_identities'],full_hash=True).read_text(encoding='utf8')))
        reviews={}
        if (fp:=campaign.get('current_source_role_reviews')):
            require(receipt.get('source_role_reviews')==fp,'unvalidated source reviews')
            reviews=json.loads(check_file(fp,full_hash=True).read_text(encoding='utf8'))
        # Use accepted current entity terms for the exact census relation ID.
        from .verified_entity_terms import VerifiedEntityTerms
        from .correlation_grouping import IndexTerms
        terms=IndexTerms(VerifiedEntityTerms(json.loads(check_file(campaign['current_entity_terms'],full_hash=True).read_text(encoding='utf8'))))
        group=summarize_relation(terms.relation_key(record['metadata']),[record['metadata']],identities=terms,papers=registry)
        require(group['id']==relation_id,'singleton relation differs from current census')
        dossier=summarize(group,[observation(record,registry,reviews)])
    result=dict(requested_claim_id=claim_id,shared_claim_id=relation_id,
        claim={k:group[k] for k in ('subject_id','subject_name','predicate','object_id','object_name')},
        original_claim_ids=sorted(expected),observation_count=len(expected),
        **paper_evidence(dossier),
        scope='complete current graph membership of this reviewed fine relation; not complete literature recall')
    for fp in (campaign['current_graph'],census['database'],campaign['current_acceptance'],census_fp):check_file(fp)
    for name in ('current_shared_relations','current_evidence_dossiers','current_source_role_reviews','current_entity_terms','current_paper_identities'):
        if campaign.get(name):check_file(campaign[name])
    require(json.loads(path.read_text(encoding='utf8'))==campaign,'current KG changed during claim lookup')
    return result
