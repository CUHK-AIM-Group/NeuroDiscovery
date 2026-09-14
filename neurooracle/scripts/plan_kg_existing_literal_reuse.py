"""R55 fresh R53-source planning for the finite R54 existing-name reuse set."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from inspect_kg_existing_literal_reuse import candidate_node_gate,incidence_gate
from inspect_kg_remaining_scoped_literals import scope_projection
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_existing_literal_reuse as repair
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round55_existing_literal_reuse'
REVIEW=j.OUTPUT/'round54_existing_literal_review'


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def inspection_bridge(inspection,baseline,accepted):
    require(inspection['full_source_sha_verified'] and inspection['reviewed_endpoints']==382,'R54 source incomplete')
    require(accepted['status']=='CURRENT_EXPANDED_LITERAL_REPAIR_APPLIED' and accepted['graph']==baseline['current_graph']
        and accepted['source_graph']==inspection['graph'] and accepted['provenance_plan']==inspection['pending_identity_plan'],
        'not the inspected R53 advancement')
    require(accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests']
        and accepted['checks']['no_record_deletion'] and accepted['checks']['original_quotes_negation_conditions_independent_sources_preserved'],
        'R53 all-source/scientific preservation bridge missing')


def current_hash(cid,old_hash,prior_events):
    event=prior_events.get(cid)
    if event:
        require(event['claim_sha256']==old_hash,'prior event is not inspected original claim')
        return event['current_node_sha256']
    return old_hash


def rebind_review(review,prior_events):
    event=prior_events.get(review['claim_id'])
    if event:require(review['side'] not in {ch['side'] for ch in event['changes']},'already changed endpoint cannot be reused from stale review')
    return dict(review,claim_sha256=current_hash(review['claim_id'],review['claim_sha256'],prior_events))


def source_inputs():
    fp=j.fingerprint(REVIEW/'SOURCE_INSPECTION.json');inspection=j.read_json(fp['path'])
    for item in [inspection['code'],inspection['source_classification'],inspection['pending_identity_plan'],*inspection['artifacts'].values()]:small_check(item)
    prior=j.read_json(inspection['pending_identity_plan']['path']);prior_events={e['claim_id']:e for e in prior['events']}
    targets={r['node_id']:r for r in rows(inspection['artifacts']['CURRENT_TARGET_WITNESSES.jsonl']['path'])}
    incidences=defaultdict(list)
    for r in rows(inspection['artifacts']['CURRENT_TARGET_INCIDENCES.jsonl']['path']):incidences[r['node_id']].append(r)
    other={r['node_id']:r['edges'] for r in rows(inspection['artifacts']['CURRENT_NONCLAIM_RELATIONS.jsonl']['path'])}
    reviews={}
    for r in rows(inspection['artifacts']['CURRENT_REUSE_CANDIDATES.jsonl']['path']):
        if r['reuse_candidate_gate'] is not None:continue
        target=targets[r['target_id']]
        require(candidate_node_gate(r,target) is None and incidence_gate(target,incidences[r['target_id']],other.get(r['target_id'],[])) is None,
            'R54 existing identity scope not reproduced')
        review=rebind_review(r,prior_events);review['target_witness']=target
        reviews[repair.review_key(r['claim_id'],r['side'])]=review
    require(len(reviews)==305,'finite whole-name reuse set differs')
    return fp,inspection,prior,reviews,targets,incidences


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'source boundary')
    receipt=j.read_json(c['current_acceptance']['path']);require((Path(c['current_acceptance']['path']).parent/'REPORT_VALIDATION.json').exists(),'publish current R53 first')
    inspection_fp,inspection,prior,reviews,old_targets,old_incidences=source_inputs();inspection_bridge(inspection,c,receipt)
    for fp in receipt['code']:small_check(fp)
    witnesses=prior['gene_witnesses'];prior_events={e['claim_id']:e for e in prior['events']};prior_edges={e['ordinal']:e for e in prior['edge_events']}
    held=rows(c['current_gene_holds']['path']);queue={(r['claim_id'],r['side']):r for r in held}
    require(len(queue)==len(held)==2511,'current R53 queue differs')
    protected={r['claim_id'] for f in ('current_issues','current_structure_holds') for r in rows(c[f]['path'])}
    protected.update(j.read_json(c['current_scope_findings']['path'])['current_claim_hashes'])
    selected={r['claim_id'] for r in reviews.values()};targets={r['target_id'] for r in reviews.values()}
    require(not selected&protected,'separate scientific hold overlaps')
    by_claim=defaultdict(list)
    for r in reviews.values():
        q=queue.get((r['claim_id'],r['side']))
        require(q and q['claim_sha256']==r['claim_sha256'] and q['current_node_id']==r['current_node_id'],'fresh queued endpoint differs')
        by_claim[r['claim_id']].append(r)
    names={name_key(old_targets[nid]['name']).casefold():nid for nid in targets}
    require(len(names)==len(targets),'duplicate target complete names')
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_existing_literal_reuse.py')]
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    claims={};seen_targets=set();seen_genes=set();matches=defaultdict(set);owned=defaultdict(list);incidences=defaultdict(list);other=defaultdict(list);counts=Counter()
    progress('FULL_CURRENT_SOURCE_EXACT_EXISTING_NAMES_ALL_INCIDENCES_GENES_AND_REFERENCES',endpoints=len(reviews),targets=len(targets))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in witnesses:
                    require(digest(row)==witnesses[key]['node_sha256'],'canonical source gene changed');seen_genes.add(key)
                if key in targets:
                    require(scope_projection(row)==old_targets[key],'existing target scope changed');seen_targets.add(key)
                if key in selected:
                    require(all(digest(row)==r['claim_sha256'] for r in by_claim[key]),'fresh selected claim hash differs');claims[key]=row
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata'];inner=md.get('metadata') or {}
                    for side in ('subject','object'):
                        if md.get(side+'_id') in targets:
                            incidences[md[side+'_id']].append(dict(node_id=md[side+'_id'],claim_id=key,side=side,name=md.get(side+'_name'),
                                claim_sha256=digest(row),nonidentity_sha256=digest(nonidentity_claim(row)),outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type')))
                else:
                    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
                    for label in names.keys()&labels:matches[label].add(key)
            elif kind=='edge':
                owner=repair.edge_owner(row)
                if owner in selected:owned[owner].append((int(key),row))
                if not owner:
                    for nid in {row.get('source_id'),row.get('target_id')}&targets:other[nid].append(dict(ordinal=int(key),edge_sha256=digest(row)))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full actual source SHA differs')
    require(set(claims)==selected and seen_targets==targets and seen_genes==set(witnesses),'selected scope incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    for label,nid in names.items():require(matches[label]=={nid},'new whole-name/alias/case collision requires review')
    for nid in targets:
        expected=[dict(r,claim_sha256=current_hash(r['claim_id'],r['claim_sha256'],prior_events)) for r in old_incidences[nid]]
        require(incidences[nid]==expected,'complete existing incidence source/semantics changed')
        require(incidence_gate(old_targets[nid],incidences[nid],other[nid]) is None,'fresh target scope gate differs')
    for cid,rs in by_claim.items():
        expected=rs[0]['owned_edges'];reproduced=[]
        require(all(r['owned_edges']==expected for r in rs),'inconsistent original closure')
        for ordinal,row in owned[cid]:
            original=repair.apply_edge(row,prior_edges[ordinal],reverse=True) if ordinal in prior_edges else row
            reproduced.append(dict(ordinal=ordinal,edge_sha256=digest(original)))
        require(reproduced==expected,'current owned references do not invert to inspected R50 scope')
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    events=[];edge_events=[];fixed=set()
    for cid,row in sorted(claims.items()):
        changes=[dict(side=r['side'],name=name_key(r['name']),old_id=r['current_node_id'],target_id=r['target_id']) for r in by_claim[cid]]
        event,current=repair.reviewed_claim(row,changes,witnesses,reviews)
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(current['metadata'])))
        events.append(event);edge_events.extend(repair.reviewed_edges(cid,row,current,owned[cid]));fixed.update((cid,ch['side']) for ch in changes)
    require(len(fixed)==305 and len(events)==304 and len(targets)==220,'finite exact reuse counts differ')
    remaining=[r for k,r in sorted(queue.items()) if k not in fixed]
    from report_kg_expanded_literal_repair import HISTORIC_THREE
    require(len(set(queue)&HISTORIC_THREE)==3 and not fixed&HISTORIC_THREE,'historic hippocampal definitions remain held')
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for e in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(e['claim_id'],)).fetchone()==(e['claim_sha256'],e['old_relation_id']),'source census differs')
        shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']))
    finally:db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    write_rows(OUTPUT/'CURRENT_EXISTING_TARGET_INCIDENCES.jsonl',[r for nid in sorted(targets) for r in incidences[nid]])
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],source_review=inspection_fp,prior_identity_plan=inspection['pending_identity_plan'],
        source_full_sha_verified=True,source_census_full_sha_verified=True,code=code,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),
        endpoint_reviews=reviews,new_literals=[],existing_targets={**{nid:old_targets[nid]['node_sha256'] for nid in targets},**{g:w['node_sha256'] for g,w in witnesses.items()}},
        gene_witnesses=witnesses,expected_shared_claim_ids=shared,relation_grouping=POLICY,changed_claims=len(events),changed_endpoints=len(fixed),
        changed_edges=len(edge_events),added_literal_nodes=0,reused_existing_nodes=len(targets),expanded_review_endpoints=len(queue),old_watchlist_endpoints=len(queue),
        held_endpoints=len(remaining),original_watchlist_449_remaining_before=3,original_watchlist_449_repaired=0,original_watchlist_449_remaining_after=3,
        protected_scientific_claims=sorted(protected),remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),
        existing_incidence_evidence=j.fingerprint(OUTPUT/'CURRENT_EXISTING_TARGET_INCIDENCES.jsonl'),
        record_preimages_saved=False,graph_backups=0,no_claim_or_edge_deleted=True,no_metadata_fields_added=True,
        boundaries=['Reuse only one exact case-sensitive full-primary-name node with the complete existing incidence scope unchanged.',
            'R53 other-side changes rebound by accepted all-record inversion and fresh current hashes, source genes and owned references.',
            'No new nodes, global concept merges, discarded audit/metadata, original-source edits or new scientific assertions.',
            'Historic three hippocampal definitions, scientific holds and all ambiguous or molecular cases remain held.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',claims=len(events),endpoints=len(fixed),reused_nodes=len(targets),new_nodes=0,remaining=len(remaining))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='FAILED',at=j.utc_now(),error=repr(error),graph_modified=False));raise
