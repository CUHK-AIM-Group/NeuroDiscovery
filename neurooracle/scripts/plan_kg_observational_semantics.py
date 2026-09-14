"""R59 finite observational wording and existing evidence context, no new graph records."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import re
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from inspect_kg_claim_deletion import exact_references
from inspect_kg_remaining_scoped_literals import scope_projection,owned_projection
from inspect_kg_semantic_hold_sources import projection
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_observational_semantics as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round59_observational_semantics'
REVIEW=j.OUTPUT/'round58_observational_semantics_review'


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def source_inputs():
    fp=j.fingerprint(REVIEW/'SOURCE_INSPECTION.json');inspection=j.read_json(fp['path'])
    for item in [inspection['code'],inspection['pending_disjoint_plan'],inspection['public_sources'],inspection['public_response'],
        inspection['original_projections'],*inspection['artifacts'].values()]:small_check(item)
    reviews={r['claim_id']:r for r in rows(inspection['artifacts']['CURRENT_CLAIM_PROJECTIONS.jsonl']['path'])}
    require(set(repair.SPECS)<=set(reviews),'three observations not reviewed')
    proof=repair.source_proof(Path(inspection['public_response']['path']).read_text(encoding='utf8'))
    for cid in repair.SPECS:
        own=inspection['public_proofs'][cid]
        require(proof[own['pmid']]['abstract_sha256']==own['abstract_sha256'] and proof[own['pmid']]['title']==own['title'],'own abstract differs')
    return fp,inspection,reviews,proof


def inspection_bridge(inspection,baseline,accepted,prior):
    require(inspection['full_source_sha_verified'] and inspection['claims']==17 and set(inspection['selected'])==set(repair.SPECS),'R58 incomplete')
    require(accepted['status']=='CURRENT_SCIENTIFIC_DEFINITION_REPAIR_APPLIED' and accepted['graph']==baseline['current_graph']
        and accepted['source_graph']==inspection['graph'] and accepted['provenance_plan']==inspection['pending_disjoint_plan'],'not reviewed R57 advancement')
    require(accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests'] and accepted['checks']['no_record_deletion']
        and accepted['checks']['all_original_statistics_and_historic_audit_records_preserved'],'R57 bridge incomplete')
    require(not set(repair.SPECS)&{e['claim_id'] for e in prior['events']},'observation source changed by R57')


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'idle current source required')
    receipt=j.read_json(c['current_acceptance']['path']);require((Path(c['current_acceptance']['path']).parent/'REPORT_VALIDATION.json').exists(),'publish R57 first')
    inspection_fp,inspection,reviews,proof=source_inputs();prior=j.read_json(inspection['pending_disjoint_plan']['path'])
    inspection_bridge(inspection,c,receipt,prior)
    for fp in receipt['code']:small_check(fp)
    selected=set(repair.SPECS);witnesses={r['node_id']:r for r in rows(inspection['artifacts']['CURRENT_TARGET_WITNESSES.jsonl']['path'])}
    original_ids={reviews[cid]['science'][s+'_id'] for cid in selected for s in ('subject','object')}
    require(original_ids==set(witnesses),'six unchanged endpoint witnesses required')
    old_issues=rows(c['current_issues']['path']);require(len(old_issues)==17 and selected<={r['claim_id'] for r in old_issues},'old issue scope differs')
    queue=rows(c['current_gene_holds']['path']);scope=j.read_json(c['current_scope_findings']['path'])
    require(len(queue)==2204 and not selected&{r['claim_id'] for r in queue} and not selected&set(scope['current_claim_hashes'])
        and not rows(c['current_structure_holds']['path']),'unreviewed issue overlap')
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_observational_semantics.py')]
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    claims={};owned=defaultdict(list);seen_nodes=set();exact=[];counts=Counter();needle=re.compile('|'.join(re.escape(cid) for cid in sorted(selected)))
    progress('FULL_CURRENT_THREE_OBSERVATIONS_AND_ALL_EXACT_REFERENCES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in original_ids:
                    require(scope_projection(row)==witnesses[key],'unchanged endpoint node differs');seen_nodes.add(key)
                if key in selected:
                    require(projection(row)==reviews[key],'current source science differs');claims[key]=row
                if key.startswith('CLM:'):counts['claims']+=1
            elif kind=='edge':
                owner=repair.edge_owner(row)
                if owner in selected:owned[owner].append((int(key),row))
            if needle.search(compact(row)):
                refs=list(exact_references(row,selected))
                if refs:exact.append(dict(kind=kind,key=key,record_sha256=digest(row),references=refs))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full actual source SHA differs')
    require(set(claims)==selected and seen_nodes==original_ids,'source closure incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    require([owned_projection(o,r) for cid in sorted(owned) for o,r in owned[cid]]==rows(inspection['artifacts']['CURRENT_OWNED_EDGE_PROJECTIONS.jsonl']['path']),
        'full current owned edge source differs')
    require(exact==rows(inspection['artifacts']['CURRENT_EXACT_CLAIM_REFERENCES.jsonl']['path']),'all exact source references differ')
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])));events=[];edge_events=[]
    for cid in sorted(selected):
        row=claims[cid];event,current=repair.reviewed_claim(row,[],proof,reviews)
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(current['metadata'])))
        events.append(event);edge_events.extend(repair.reviewed_edges(cid,row,current,owned[cid]))
    remaining=repair.remaining_issues(old_issues,dict(events=events));require(len(remaining)==14,'remaining semantic queue differs')
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'source census full SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for event in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(event['claim_id'],)).fetchone()==(event['claim_sha256'],event['old_relation_id']),'source claim census differs')
        shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']))
    finally:db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',queue)
    write_rows(OUTPUT/'REMAINING_SEMANTIC_ISSUES.jsonl',remaining)
    write_rows(OUTPUT/'CURRENT_EXISTING_TARGET_INCIDENCES.jsonl',[])
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],detail_store=c['current_detail_store'],
        source_review=inspection_fp,prior_identity_plan=inspection['pending_disjoint_plan'],code=code,public_manifests=[inspection['public_sources']],
        public_abstracts=inspection['public_response'],public_source_proof=proof,claim_reviews={cid:reviews[cid] for cid in selected},
        source_full_sha_verified=True,source_census_full_sha_verified=True,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),
        new_literals=[],existing_targets={nid:witnesses[nid]['node_sha256'] for nid in original_ids},reused_target_ids=[],
        expected_shared_claim_ids=shared,relation_grouping=POLICY,changed_claims=3,changed_endpoints=0,changed_edges=len(edge_events),
        added_literal_nodes=0,reused_existing_nodes=0,changed_direction_values=3,changed_predicates=3,changed_study_type_values=3,changed_methodology_values=3,
        expanded_review_endpoints=len(queue),old_watchlist_endpoints=len(queue),held_endpoints=len(queue),repaired_queued_endpoints=0,
        original_watchlist_449_remaining_before=1,original_watchlist_449_repaired=0,original_watchlist_449_remaining_after=1,
        old_semantic_holds=17,remaining_semantic_holds=14,resolved_semantic_holds=3,
        remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),remaining_semantic_issues=j.fingerprint(OUTPUT/'REMAINING_SEMANTIC_ISSUES.jsonl'),
        existing_incidence_evidence=j.fingerprint(OUTPUT/'CURRENT_EXISTING_TARGET_INCIDENCES.jsonl'),
        record_preimages_saved=False,graph_backups=0,no_claim_or_edge_deleted=True,no_metadata_fields_added=True,
        boundaries=['Three exact own-source observational corrections, no endpoint or node change and no record deletion.',
            'Only predicate, preferred display name and existing evidence.direction/study_type/methodology values change.',
            'Original quotations, legacy claim text, statistical values, bibliography, confidence, history and old audits remain unchanged.',
            'Within-group improvement is not between-arm superiority; the continuation phase is not randomized merely because the acute phase was.',
            'The other fourteen qualified semantic issues, six scientific findings and 2204 endpoint candidates stay open.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']]);j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',claims=3,edges=len(edge_events),new_nodes=0,remaining_semantic_holds=14)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='FAILED',at=j.utc_now(),error=repr(error),graph_modified=False));raise
