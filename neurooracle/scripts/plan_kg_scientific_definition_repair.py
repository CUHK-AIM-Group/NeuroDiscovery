"""R57 finite source-proved corrections with current existing-name and scope guards."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from inspect_kg_remaining_scoped_literals import scope_projection,owned_projection
from inspect_kg_semantic_hold_sources import projection
from inspect_kg_scientific_definition_scope import field_hashes
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_scientific_definition_repair as repair
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round57_scientific_definition_repair'
REVIEW=j.OUTPUT/'round56_scientific_definition_review'
TELOMERE={'CLM:case3_topup_20260730_manual_b0020_039_01','CLM:case3_topup_20260730_manual_b0020_039_02'}


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def source_inputs():
    fp=j.fingerprint(REVIEW/'SOURCE_INSPECTION.json');inspection=j.read_json(fp['path'])
    for item in [inspection['code'],inspection['pending_disjoint_plan'],inspection['public_sources'],inspection['duplicate_differences'],*inspection['artifacts'].values()]:small_check(item)
    reviews={r['claim_id']:r for r in rows(inspection['artifacts']['CURRENT_CLAIM_PROJECTIONS.jsonl']['path'])}
    require(set(repair.SPECS)<=set(reviews) and TELOMERE<=set(reviews),'review incomplete')
    paths=[j.OUTPUT/'round49_remaining_scoped_literals/PUBLIC_SOURCE_FETCH.json',j.OUTPUT/'round51_scientific_scope_review/PUBLIC_SOURCE_FETCH.json',REVIEW/'PUBLIC_SOURCE_FETCH.json']
    manifests=[j.fingerprint(p) for p in paths];payloads=[j.read_json(p) for p in paths]
    abstracts=payloads[0]['response'];fulltexts={}
    for manifest in payloads:
        for r in manifest['fulltexts']:
            if r['status']=='PUBLIC_FULLTEXT_FETCHED' and r['pmid'] in repair.FULL_FRAGMENTS:
                require(r['pmid'] not in fulltexts,'duplicate owning fulltext');fulltexts[r['pmid']]=r['response']
    require(set(fulltexts)==set(repair.FULL_FRAGMENTS),'six own fulltext sources required')
    for item in [abstracts,*fulltexts.values()]:small_check(item)
    proof=repair.source_proof(Path(abstracts['path']).read_text(encoding='utf8'),{pmid:Path(f['path']).read_text(encoding='utf8') for pmid,f in fulltexts.items()})
    return fp,inspection,reviews,manifests,abstracts,fulltexts,proof


def inspection_bridge(inspection,baseline,accepted,prior):
    require(inspection['full_source_sha_verified'] and inspection['claims']==31,'R56 incomplete')
    require(accepted['status']=='CURRENT_EXISTING_LITERAL_REUSE_APPLIED' and accepted['graph']==baseline['current_graph']
        and accepted['source_graph']==inspection['graph'] and accepted['provenance_plan']==inspection['pending_disjoint_plan'],'not reviewed R55 advancement')
    require(accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests'] and accepted['checks']['no_record_deletion']
        and accepted['checks']['no_new_concept_nodes'],'R55 bridge incomplete')
    require(not set(repair.SPECS)&{e['claim_id'] for e in prior['events']},'scientific source changed by R55')


def incidence_bridge(actual,original,prior,targets):
    """Retain old incidences and account for R55's one newly routed claim."""
    key=lambda r:(r['node_id'],r['claim_id'],r['side'])
    old={key(r):r for r in original};current={key(r):r for r in actual}
    require(len(old)==len(original) and len(current)==len(actual),'duplicate target incidence')
    require(all(current.get(k)==r for k,r in old.items()),'original measurement incidence changed')
    additions={}
    for event in prior['events']:
        for ch in event['changes']:
            require(ch['old_id'] not in targets,'R55 removed an existing measurement incidence')
            if ch['target_id'] not in targets:continue
            require(event['claim_id']=='CLM:CASE1MAN:33668432:9546' and ch['side']=='subject'
                and ch['target_id']==repair.TARGETS[repair.SEPARATE] and ch['name']==repair.SEPARATE,
                'unreviewed new R55 measurement incidence')
            reviews=[r for r in prior['endpoint_reviews'].values() if r['claim_id']==event['claim_id'] and r['side']==ch['side']]
            require(len(reviews)==1,'prior added incidence review missing')
            r=reviews[0]
            require(r['claim_sha256']==event['claim_sha256'] and r['target_id']==ch['target_id'] and r['name']==ch['name'],
                'prior added incidence review differs')
            row=dict(node_id=ch['target_id'],claim_id=event['claim_id'],side=ch['side'],name=ch['name'],
                claim_sha256=event['current_node_sha256'],nonidentity_sha256=event['nonidentity_sha256'],
                outer_type=r['outer_type'],inner_type=r['inner_type'])
            require(key(row) not in old and key(row) not in additions,'added incidence duplicate or already present')
            additions[key(row)]=row
    require(len(additions)==1 and current=={**old,**additions},'current measurement incidence closure differs')
    return list(additions.values())


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'idle source required')
    receipt=j.read_json(c['current_acceptance']['path']);require((Path(c['current_acceptance']['path']).parent/'REPORT_VALIDATION.json').exists(),'publish R55 first')
    inspection_fp,inspection,reviews,manifests,abstracts,fulltexts,proof=source_inputs()
    prior=j.read_json(inspection['pending_disjoint_plan']['path']);inspection_bridge(inspection,c,receipt,prior)
    for fp in receipt['code']:small_check(fp)
    targets={repair.TARGETS[n] for n in (repair.MEAN,repair.SEPARATE)}
    witnesses={r['node_id']:r for r in rows(inspection['artifacts']['CURRENT_TARGET_WITNESSES.jsonl']['path'])}
    old_owned=rows(inspection['artifacts']['CURRENT_OWNED_EDGE_PROJECTIONS.jsonl']['path'])
    old_incidences=[r for r in rows(inspection['artifacts']['CURRENT_TARGET_INCIDENCES.jsonl']['path']) if r['node_id'] in targets]
    require(not {r['claim_id'] for r in old_incidences}&{e['claim_id'] for e in prior['events']},'existing measurement incidence advanced')
    original_ids={old_id for pmid,pred,sides in repair.SPECS.values() for side,old_name,old_id,new_name in sides}
    original_ids.update(reviews[repair.ACC]['science'][s+'_id'] for s in ('subject','object'))
    selected=set(repair.SPECS);observed_claims=selected|TELOMERE
    require(original_ids|targets<=set(witnesses),'source node witnesses missing')
    names={name_key(name).casefold():nid for name,nid in repair.TARGETS.items()};new={repair.TARGETS[name]:repair.literal_node(name) for name in repair.NEW_NAMES}
    genes={r['claim_id'] for r in rows(c['current_issues']['path'])};require(not selected&genes,'old semantic hold needs separate review')
    current_scope=j.read_json(c['current_scope_findings']['path'])
    require(selected&set(current_scope['current_claim_hashes'])=={repair.BACE,repair.ACC},'unexpected current scientific hold overlap')
    census=j.read_json(c['current_paper_census']['path']);code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_scientific_definition_repair.py')]
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    claims={};owned=defaultdict(list);seen_nodes=set();incidences=[];matches=defaultdict(set);counts=Counter()
    progress('FULL_CURRENT_SOURCE_EIGHT_CLAIMS_TARGET_IDENTITIES_AND_ALL_OWNED_REFERENCES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                require(key not in new,'new complete identity collision')
                if key in original_ids|targets:
                    require(scope_projection(row)==witnesses[key],'existing target/source node changed');seen_nodes.add(key)
                if key in observed_claims:
                    require(projection(row)==reviews[key],'source science or audit projection changed');claims[key]=row
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata'];inner=md.get('metadata') or {}
                    for side in ('subject','object'):
                        if md.get(side+'_id') in targets:
                            incidences.append(dict(node_id=md[side+'_id'],claim_id=key,side=side,name=md.get(side+'_name'),claim_sha256=digest(row),
                                nonidentity_sha256=digest(nonidentity_claim(row)),outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type')))
                else:
                    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
                    for label in names.keys()&labels:matches[label].add(key)
            elif kind=='edge':
                owner=repair.edge_owner(row)
                if owner in selected:owned[owner].append((int(key),row))
                if not owner:require(not {row.get('source_id'),row.get('target_id')}&targets,'unreviewed nonclaim measurement scope')
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full actual source SHA differs')
    require(set(claims)==observed_claims and seen_nodes==original_ids|targets,'source closure incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    prior_incidence_additions=incidence_bridge(incidences,old_incidences,prior,targets)
    for name,nid in repair.TARGETS.items():require(matches[name.casefold()]==({nid} if nid in targets else set()),'new name/alias candidate requires separate review: '+name)
    for nid in targets:
        node=witnesses[nid]
        require(not node['semantic_types'] and not node['external_ids'] and not node['aliases']
            and node['definition_sha256']==digest('') and node['spatial_mapping_sha256']==digest(None),'unreviewed existing measurement scope')
        require(all(r['name']==node['name'] for r in incidences if r['node_id']==nid),'existing measurement full-name mixture')
    actual=[owned_projection(o,r) for cid in sorted(owned) for o,r in owned[cid]]
    require(actual==[r for r in old_owned if r['claim_id'] in selected],'full current owned edge source differs')
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    events=[];edge_events=[]
    for cid in sorted(selected):
        row=claims[cid];event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews)
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(current['metadata'])))
        events.append(event);edge_events.extend(repair.reviewed_edges(cid,row,current,owned[cid]))
    queue=rows(c['current_gene_holds']['path']);keys={(r['claim_id'],r['side']) for r in queue}
    fixed={(repair.ENIGMA,'subject'),(repair.FES,'subject')}
    require(len(keys)==len(queue)==2206 and fixed<=keys,'gene queue scope differs')
    require({(e['claim_id'],ch['side']) for e in events for ch in e['changes']}&keys==fixed,'unexpected additional gene hold repair')
    remaining=[r for r in queue if (r['claim_id'],r['side']) not in fixed]
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'source census full SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for event in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(event['claim_id'],)).fetchone()==(event['claim_sha256'],event['old_relation_id']),'source claim census differs')
        shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']))
    finally:db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    write_rows(OUTPUT/'CURRENT_EXISTING_TARGET_INCIDENCES.jsonl',sorted(incidences,key=lambda r:r['node_id']))
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],detail_store=c['current_detail_store'],
        source_review=inspection_fp,prior_identity_plan=inspection['pending_disjoint_plan'],code=code,public_manifests=manifests,public_abstracts=abstracts,public_fulltexts=fulltexts,
        public_source_proof=proof,claim_reviews={cid:reviews[cid] for cid in selected},unmodified_sample_size_reviews={cid:reviews[cid] for cid in TELOMERE},
        source_full_sha_verified=True,source_census_full_sha_verified=True,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),
        new_literals=[dict(id=nid,name=node['preferred_name'],node_sha256=digest(node)) for nid,node in sorted(new.items())],
        existing_targets={nid:witnesses[nid]['node_sha256'] for nid in original_ids|targets},reused_target_ids=sorted(targets),
        prior_target_incidence_additions=prior_incidence_additions,
        expected_shared_claim_ids=shared,relation_grouping=POLICY,changed_claims=len(events),changed_endpoints=sum(len(e['changes']) for e in events),changed_edges=len(edge_events),
        added_literal_nodes=len(new),reused_existing_nodes=len(targets),changed_direction_values=3,changed_predicates=1,changed_declared_type_values=1,
        expanded_review_endpoints=len(queue),old_watchlist_endpoints=len(queue),held_endpoints=len(remaining),repaired_queued_endpoints=2,
        original_watchlist_449_remaining_before=3,original_watchlist_449_repaired=2,original_watchlist_449_remaining_after=1,
        remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),existing_incidence_evidence=j.fingerprint(OUTPUT/'CURRENT_EXISTING_TARGET_INCIDENCES.jsonl'),
        record_preimages_saved=False,graph_backups=0,no_claim_or_edge_deleted=True,no_metadata_fields_added=True,
        boundaries=['Eight finite own-source definition corrections, no new claims or scientific edges, no global concept merge.',
            'All raw_text, scope_evidence_spans, statistical values, source bibliography, conditions and old audit records remain byte-identical.',
            'Only names/IDs, one PATHWAY-to-measurement role, one association predicate and three explicitly supported directions change.',
            'Mean, sum and separately reported hemispheres remain distinct. ENIGMA compound object is NOT repaired or silently dropped.',
            'DPD records share one relation identity but retain their distinct audit records and count only one source paper.',
            'Historic 15038994 aggregation and telomere sample denominator remain unknown; do not guess replacements.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',claims=len(events),endpoints=plan['changed_endpoints'],edges=len(edge_events),new_nodes=len(new),reused_nodes=len(targets),historic_remaining=1,expanded_remaining=len(remaining))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='FAILED',at=j.utc_now(),error=repr(error),graph_modified=False));raise
