"""Plan exact source-backed deletion, stale-branch retirement and viral identity."""
from collections import Counter,defaultdict
from pathlib import Path
import re
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_explicit_pmid_provenance import authorities
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_source_scope_resolution as scope
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import bibliography
from neurooracle.src.relation_evidence import relation_id,strongest_paper_key
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round44_source_resolution'
REVIEW=j.OUTPUT/'round42_source_scope'


def progress(phase,**values):
    r=dict(status='READ_ONLY_PLANNING',phase=phase,at=j.utc_now(),**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',r);print(compact(r),flush=True)


def expected_shared(db,events,deleted,prior):
    ids={m['claim_id'] for g in prior for m in g['members']}
    affected={e[k] for e in events for k in ('old_relation_id','new_relation_id')}
    for cid in deleted:
        r=db.execute('SELECT relation_id FROM claims WHERE cid=?',(cid,)).fetchone()
        require(r is not None,'deleted claim missing in census');affected.add(r[0])
    groups={rid:{r[0] for r in db.execute('SELECT cid FROM claims WHERE relation_id=?',(rid,))} for rid in affected}
    for members in groups.values():ids.difference_update(members);members.difference_update(deleted)
    for e in events:
        require(e['claim_id'] in groups[e['old_relation_id']],'changed member missing')
        groups[e['old_relation_id']].remove(e['claim_id']);groups[e['new_relation_id']].add(e['claim_id'])
    for members in groups.values():
        if len(members)>1:ids.update(members)
    return sorted(ids)


def main(replace_unapplied=False):
    require(not (OUTPUT/'PLAN.json').exists() or replace_unapplied,'plan exists; inspect/resume')
    require(not (OUTPUT/'BUILD_STATE.json').exists() and not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists(),'cannot revise a built or adopted plan')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'writer/retention boundary')
    require(Path(c['current_acceptance']['path']).parent.name=='round43_gene_boundary','expected accepted R43')
    require((j.OUTPUT/'round43_gene_boundary/REPORT_VALIDATION.json').exists(),'publish and verify R43 first')
    receipt=j.read_json(c['current_acceptance']['path']);census=j.read_json(c['current_paper_census']['path'])
    for fp in receipt['code']:require(j.fingerprint(fp['path'])==fp,'frozen current code differs')
    source_manifest=j.fingerprint(REVIEW/'PUBLIC_SOURCE_FETCH.json');sources=j.read_json(source_manifest['path'])
    for fp in (sources['code'],sources['abstract_response'],sources['fulltext_response']):require(j.fingerprint(fp['path'])==fp,'source witness changed')
    for cid,pmid in scope.PMIDS.items():require(sources['explicit_current_owners'][cid]==[pmid,scope.DOIS[pmid]],'source ownership differs')
    proof=scope.public_source_proof(Path(sources['abstract_response']['path']).read_text(encoding='utf8'),
        Path(sources['fulltext_response']['path']).read_text(encoding='utf8'))
    reuse_manifest_fp=j.fingerprint(OUTPUT/'VIRAL_REUSE_SOURCE_FETCH.json');reuse_manifest=j.read_json(reuse_manifest_fp['path'])
    require(reuse_manifest['graph']==c['current_graph'] and reuse_manifest['current_acceptance']==c['current_acceptance'],'viral reuse source boundary differs')
    for fp in (reuse_manifest['code'],reuse_manifest['response']):require(j.fingerprint(fp['path'])==fp,'viral reuse source changed')
    reuse_sources=scope.viral_reuse_source_proof(Path(reuse_manifest['response']['path']).read_text(encoding='utf8'))
    if replace_unapplied:
        old_plan=j.read_json(OUTPUT/'PLAN.json')
        require(old_plan['status']=='REVIEWED_NOT_APPLIED' and old_plan['graph']==c['current_graph'],'unapplied source advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']]);OUTPUT.mkdir(exist_ok=True)
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_source_scope_resolution.py')]
    selected=set(scope.CLAIM_HASHES);claims={};owned=defaultdict(list);found_nodes={};exact=[];counts=Counter();viral_matches={};reuse_incidents=[]
    generated=scope.viral_node()
    wanted_nodes={'CUI:C1421437','CLM_CONCEPT:white_matter_hyperintensity_burden','CLM_CONCEPT:frailty_09cfc9267861',
        'CLM_CONCEPT:white_matter_hyperintensity_burden_e206a534fc08','CLM_CONCEPT:frailty'}
    progress('FULL_CURRENT_SOURCE_AND_ALL_EXACT_DELETION_REFERENCES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in selected:
                    scope.source_claim(row,proof);claims[key]=row
                if key.startswith('CLM:'):
                    md=row['metadata'];inner=md.get('metadata') or {};paper=md.get('source_paper') or {}
                    for side in ('subject','object'):
                        if md.get(side+'_id')==scope.VIRAL_REUSE_ID:
                            reuse_incidents.append(dict(claim_id=key,claim_sha256=digest(row),side=side,name=md.get(side+'_name'),
                                outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type'),pmid=str(paper.get('pmid') or ''),doi=paper.get('doi') or ''))
                if key in wanted_nodes:found_nodes[key]=digest(row)
                if not key.startswith('CLM:'):
                    labels=[row.get('preferred_name'),*(row.get('aliases') or [])]
                    if any(isinstance(s,str) and ' '.join(s.split()).casefold()==scope.VIRAL_NAME for s in labels):viral_matches[key]=row
                    if key==generated['id']:require(row==generated,'viral ID collision')
            elif kind=='edge' and scope.edge_owner(row) in selected:owned[scope.edge_owner(row)].append((int(key),row))
            encoded=compact(row)
            if scope.MYELIN in encoded:
                hits=list(exact_references(row,{scope.MYELIN}))
                if hits:exact.append(dict(kind=kind,key=key,record_sha256=digest(row),references=hits))
            if counts[kind]%1000000==0:progress(kind,records=counts[kind])
        require(h.hexdigest()==c['current_graph']['sha256'],'source full SHA differs')
    require(set(claims)==selected and set(found_nodes)==wanted_nodes,'selected graph scope incomplete')
    removed=scope.reviewed_myelin_removal(claims[scope.MYELIN],owned[scope.MYELIN],proof)
    branch=scope.reviewed_frailty_retirements(claims[scope.FRAILTY],owned[scope.FRAILTY],proof)
    allowed={('node',scope.MYELIN),*[('edge',str(e['ordinal'])) for e in removed['owned_edges']]}
    require({(r['kind'],str(r['key'])) for r in exact}==allowed,'unexpected exact reference to selected deletion')
    reuse_witness=None;target_id=generated['id'];reviewed_reuse=False
    if set(viral_matches)=={scope.VIRAL_REUSE_ID}:
        reuse_witness=scope.reviewed_viral_reuse(viral_matches[scope.VIRAL_REUSE_ID],reuse_incidents,reuse_sources)
        target_id=scope.VIRAL_REUSE_ID;reviewed_reuse=True;found_nodes[target_id]=reuse_witness['node_sha256']
    if viral_matches and not reviewed_reuse and viral_matches!={generated['id']:generated}:
        viral_hold='existing_complete_protein_name_or_alias_requires_separate_identity_review';events=[];edges=[];new=[]
    else:
        viral_hold=None;event,out=scope.reviewed_viral_identity(claims[scope.VIRUS],proof,target_id,reuse_witness)
        terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
        event.update(old_relation_id=relation_id(terms.relation_key(claims[scope.VIRUS]['metadata'])),new_relation_id=relation_id(terms.relation_key(out['metadata'])))
        events=[event];edges=scope.reviewed_edges(scope.VIRUS,claims[scope.VIRUS],out,owned[scope.VIRUS])
        new=[] if viral_matches else [dict(id=generated['id'],name=scope.VIRAL_NAME,node_sha256=digest(generated))]
        if viral_matches and not reviewed_reuse:found_nodes[generated['id']]=digest(generated)
    require(len(removed['owned_edges'])==2 and len(branch)==3,'reviewed deletion delta differs')
    removed_ordinals=sorted([e['ordinal'] for e in removed['owned_edges']]+[e['ordinal'] for e in branch])
    require(len(set(removed_ordinals))==5 and not set(removed_ordinals)&{e['ordinal'] for e in edges},'deletion/change overlap')
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for cid,row in claims.items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(digest(row),),'current claim census differs')
    shared=expected_shared(db,events,{scope.MYELIN},rows(c['current_shared_relations']['path']))
    paper_sig,legacy=db.execute('SELECT paper_sig,legacy_key FROM claims WHERE cid=?',(scope.MYELIN,)).fetchone()
    require(db.execute('SELECT COUNT(*) FROM claims WHERE cid!=? AND paper_sig=?',(scope.MYELIN,paper_sig)).fetchone()[0]>0,'deletion would retire a bibliography; new review required')
    require(db.execute('SELECT COUNT(*) FROM claims WHERE cid!=? AND legacy_key=?',(scope.MYELIN,legacy)).fetchone()[0]>0,'deletion would retire a source key; new review required')
    require(db.execute('SELECT pmid FROM papers WHERE sig=?',(paper_sig,)).fetchone()==('28526817',),'own current PMID differs')
    db.close()
    papers,_=authorities(c);answer=papers.resolve(claims[scope.MYELIN]['metadata'])
    old_audit=j.read_json(receipt['identity_audit']['path']);expected_status=Counter(old_audit['claim_status']);expected_status[answer['status']]-=1
    reasons=Counter(old_audit['hold_reasons']);reasons.subtract(answer['reasons'])
    require(all(n>=0 for n in [*expected_status.values(),*reasons.values()]),'source status decrement differs')
    expected_census=dict(census['counts'],claims=census['counts']['claims']-1,shared_claims=len(shared))
    proposed_counts=dict(c['counts'],nodes=c['counts']['nodes']-1+len(new),claims=c['counts']['claims']-1,edges=c['counts']['edges']-5)
    detail=sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro',uri=True)
    deletion_dependencies=dict(atoms=detail.execute('SELECT COUNT(*) FROM atoms WHERE source_mention_id=?',(scope.MYELIN,)).fetchone()[0],
        mappings_source=detail.execute('SELECT COUNT(*) FROM mappings WHERE source_id=?',(scope.MYELIN,)).fetchone()[0],
        mappings_target=detail.execute('SELECT COUNT(*) FROM mappings WHERE target_id=?',(scope.MYELIN,)).fetchone()[0])
    detail.close();require(not any(deletion_dependencies.values()),'selected claim has offline detail dependencies')
    plan=dict(version=scope.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],source_manifest=source_manifest,public_sources=[sources['abstract_response'],sources['fulltext_response']],
        viral_reuse_source_manifest=reuse_manifest_fp,viral_reuse_public_source=reuse_manifest['response'],viral_target_witness=reuse_witness,
        public_source_proof=proof,source_full_sha_verified=True,source_census_full_sha_verified=True,code=code,
        deleted_claims=[removed],retired_branch_edges=branch,removed_edge_ordinals=removed_ordinals,exact_deletion_references=exact,
        deletion_detail_dependencies=deletion_dependencies,events=events,edge_events=sorted(edges,key=lambda e:e['ordinal']),new_literals=new,
        existing_targets=found_nodes,frailty_claim_sha256=digest(claims[scope.FRAILTY]),
        frailty_current_canonical_edges=[dict(ordinal=o,edge_sha256=digest(e)) for o,e in owned[scope.FRAILTY] if o not in removed_ordinals],
        viral_candidates=[dict(node_id=nid,node_sha256=digest(n),preferred_name=n.get('preferred_name')) for nid,n in viral_matches.items()],
        viral_hold=viral_hold,expected_shared_claim_ids=shared,expected_counts=proposed_counts,expected_census_counts=expected_census,
        reused_existing_viral_node=reviewed_reuse,revision=2 if replace_unapplied else 1,
        expected_paper_claim_status={k:v for k,v in expected_status.items() if v},expected_paper_hold_reasons={k:v for k,v in reasons.items() if v},
        expected_verified_key_changes=old_audit['verified_key_changes']-int(answer['status']=='verified' and answer['paper_key']!=strongest_paper_key(claims[scope.MYELIN]['metadata'])),
        relation_grouping=POLICY,changed_claims=len(events),changed_endpoints=len(events),changed_edges=len(edges),added_literal_nodes=len(new),
        deleted_claim_count=1,deleted_edges=5,record_preimages_saved=False,graph_backups=0,
        boundaries=['Remove one affirmative symptom-severity claim contradicted by its own fulltext result; no null-result or episode claim synthesized.',
            'Retire three exact obsolete owned branch payloads; keep canonical WMH/frailty relation and every concept/detail node.',
            'Viral VCP identity is source-specific; preserve claim name, raw quotation, other scientific fields and audits, no global acronym alias.',
            'Other scientific and gene endpoints remain held; no models, training, re-extraction or formal full_v2 writes.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',deleted_claims=1,retired_owned_edges=5,identity_claims=len(events),identity_edges=len(edges),new_nodes=len(new),viral_hold=viral_hold)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replace-unapplied',action='store_true')
    main(p.parse_args().replace_unapplied)
