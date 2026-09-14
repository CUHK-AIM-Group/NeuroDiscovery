"""R53 fresh current-source plan for the finite R52 non-molecular literal set."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from classify_kg_expanded_literal_candidates import candidate_gate
from inspect_kg_historic_literal_candidates import node_projection
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from plan_kg_reviewed_literal_reuse import adjusted_ordinal
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_expanded_literal_repair as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round53_expanded_literal_repair'
REVIEW=j.OUTPUT/'round52_expanded_literal_review'


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def inspection_bridge(review,baseline,accepted):
    require(review['full_source_sha_verified'] and review['reviewed_endpoints']==5490,'R52 review incomplete')
    require(accepted['status']=='CURRENT_REVIEWED_LITERAL_REUSE_APPLIED' and accepted['graph']==baseline['current_graph']
        and accepted['source_graph']==review['graph'] and accepted['provenance_plan']==review['pending_identity_plan'],'not the inspected R50 advancement')
    require(accepted['checks']['inverse_reproduces_all_kept_source_node_and_edge_record_digests']
        and accepted['checks']['no_existing_concept_metadata_change'] and accepted['checks']['no_existing_concept_or_claim_deletion'],'R50 identity bridge missing')


def case_conflicts(reviews):
    groups=defaultdict(set)
    for r in reviews:groups[name_key(r['name']).casefold()].add(name_key(r['name']))
    return {name for key,values in groups.items() if len(values)>1 for name in values}


def classification_inputs():
    classification_fp=j.fingerprint(REVIEW/'CANDIDATE_CLASSIFICATION.json');classification=j.read_json(classification_fp['path'])
    require(classification['status']=='CLASSIFIED_NOT_PLANNED_OR_APPLIED' and classification['candidate_endpoints']==2979,'classification differs')
    for fp in (classification['source_inspection'],classification['code'],classification['classified'],classification['candidate_names_file']):small_check(fp)
    inspection=j.read_json(classification['source_inspection']['path'])
    for fp in [inspection['code'],*inspection['artifacts'].values()]:small_check(fp)
    reviews={repair.review_key(r['claim_id'],r['side']):r for r in rows(classification['classified']['path'])}
    require(len(reviews)==5490,'finite expanded review differs')
    for r in reviews.values():
        require(candidate_gate(r,set(inspection['pending_changed_claim_overlap']))==r['literal_candidate_gate'],'classification not reproducible')
    return classification_fp,classification,inspection,reviews


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'source boundary')
    receipt=j.read_json(c['current_acceptance']['path']);require((Path(c['current_acceptance']['path']).parent/'REPORT_VALIDATION.json').exists(),'publish current R50 first')
    class_fp,classification,inspection,reviews=classification_inputs();inspection_bridge(inspection,c,receipt)
    for fp in receipt['code']:small_check(fp)
    prior=j.read_json(receipt['provenance_plan']['path']);small_check(receipt['provenance_plan'])
    witnesses={r['node_id']:r for r in rows(inspection['artifacts']['GENE_WITNESSES.jsonl']['path'])}
    held=rows(c['current_gene_holds']['path']);queue={(r['claim_id'],r['side']):r for r in held}
    require(len(queue)==len(held)==5490,'current queue differs')
    protected={r['claim_id'] for f in ('current_issues','current_structure_holds') for r in rows(c[f]['path'])}
    protected.update(j.read_json(c['current_scope_findings']['path'])['current_claim_hashes'])
    proposals={k:r for k,r in reviews.items() if r['literal_candidate_gate'] is None and r['claim_id'] not in protected}
    require(not {r['claim_id'] for r in proposals.values()}&set(inspection['pending_changed_claim_overlap']),'changed R50 source not excluded')
    for r in proposals.values():
        q=queue.get((r['claim_id'],r['side']))
        require(q and q['claim_sha256']==r['claim_sha256'] and q['current_node_id']==r['current_node_id'],'current proposal source differs')
    conflicts=case_conflicts(proposals.values())
    selected={r['claim_id'] for r in proposals.values()};by_claim=defaultdict(list)
    for r in proposals.values():by_claim[r['claim_id']].append(r)
    names={name_key(r['name']).casefold() for r in proposals.values()}
    generated={repair.literal_node(r['name'])['id'] for r in proposals.values()}
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_expanded_literal_repair.py')]
    census=j.read_json(c['current_paper_census']['path']);j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    claims={};nodes={};matches=defaultdict(set);owned=defaultdict(list);found_genes=set();existing_generated=set();counts=Counter()
    progress('FULL_CURRENT_SOURCE_2979_CANDIDATES_NAMES_ALIASES_CANONICAL_GENES_AND_OWNED_REFERENCES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in witnesses:
                    require(digest(row)==witnesses[key]['node_sha256'],'source gene changed');found_genes.add(key)
                if key in selected:
                    require(all(digest(row)==r['claim_sha256'] for r in by_claim[key]),'reviewed current claim changed');claims[key]=row
                if key in generated:existing_generated.add(key)
                if key.startswith('CLM:'):counts['claims']+=1
                else:
                    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
                    for label in names&labels:matches[label].add(key);nodes[key]=node_projection(row)
            elif kind=='edge' and repair.edge_owner(row) in selected:owned[repair.edge_owner(row)].append((int(key),row))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full actual source SHA differs')
    require(set(claims)==selected and found_genes==set(witnesses),'source selection incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    for cid,rs in by_claim.items():
        expected=[dict(ordinal=adjusted_ordinal(e['ordinal'],prior['removed_edge_ordinals']),edge_sha256=e['edge_sha256']) for e in rs[0]['owned_edges']]
        require(all(r['owned_edges']==rs[0]['owned_edges'] for r in rs),'duplicated claim review has inconsistent edge evidence')
        require([dict(ordinal=o,edge_sha256=digest(r)) for o,r in owned[cid]]==expected,'R52 owned hashes/fresh current ordinals differ')
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    events=[];edge_events=[];fixed=set();new={};declines={(r['claim_id'],r['side']):r['literal_candidate_gate'] for r in reviews.values() if r['literal_candidate_gate']}
    for r in reviews.values():
        if r['claim_id'] in protected:declines[r['claim_id'],r['side']]='separate_scientific_scope_hold'
    for cid,row in sorted(claims.items()):
        changes=[];local_new={}
        for r in by_claim[cid]:
            name=name_key(r['name']);side=r['side'];nid=repair.literal_node(name)['id']
            if name in conflicts:declines[cid,side]='within_batch_case_variants_require_separate_review';continue
            if matches.get(name.casefold()):declines[cid,side]='fresh_existing_name_or_alias_requires_separate_reuse';continue
            require(nid not in existing_generated,'unexpected hashed-identity collision')
            require(repair.endpoint_gate(row,side,witnesses[r['current_node_id']],r) is None,'finite review gate changed')
            target=repair.literal_node(name);local_new[nid]=dict(id=nid,name=name,node_sha256=digest(target))
            changes.append(dict(side=side,name=name,old_id=r['current_node_id'],target_id=nid))
        if not changes:continue
        event,current=repair.reviewed_claim(row,changes,witnesses,reviews);edges=repair.reviewed_edges(cid,row,current,owned[cid])
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(current['metadata'])))
        events.append(event);edge_events.extend(edges);new.update(local_new);fixed.update((cid,ch['side']) for ch in changes)
    require(events and len(fixed)<=classification['candidate_endpoints'] and fixed<=set(queue),'scope expansion')
    historical=rows(j.OUTPUT/'round42_source_scope/ENDPOINT_REVIEW.jsonl');hist_keys={(r['claim_id'],r['side']) for r in historical}
    current_hist=set(queue)&hist_keys
    require(len(hist_keys)==449 and len(current_hist)==3 and not fixed&current_hist,'historic three scopes must stay held')
    remaining=[dict(r,latest_review_reason=declines[k]) if k in declines else r for k,r in sorted(queue.items()) if k not in fixed]
    require(len(remaining)+len(fixed)==len(queue),'queue partition differs')
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for e in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(e['claim_id'],)).fetchone()==(e['claim_sha256'],e['old_relation_id']),'current census claim differs')
        shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']))
    finally:db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    write_rows(OUTPUT/'CURRENT_COMPLETE_NAME_COLLISIONS.jsonl',list(nodes.values()))
    write_rows(OUTPUT/'CURRENT_DECLINES.jsonl',[dict(claim_id=cid,side=side,reason=reason) for (cid,side),reason in sorted(declines.items())])
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],source_review=classification['source_inspection'],classification=class_fp,
        prior_identity_plan=receipt['provenance_plan'],source_full_sha_verified=True,source_census_full_sha_verified=True,
        historical_projection_rebound_by_fresh_current_claim_gene_node_incidence_and_full_source_hashes=True,
        code=code,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),endpoint_reviews=reviews,
        new_literals=sorted(new.values(),key=lambda r:r['id']),existing_targets={g:w['node_sha256'] for g,w in witnesses.items()},
        gene_witnesses=witnesses,expected_shared_claim_ids=shared,relation_grouping=POLICY,
        changed_claims=len(events),changed_endpoints=len(fixed),changed_edges=len(edge_events),added_literal_nodes=len(new),reused_existing_nodes=0,
        old_watchlist_endpoints=len(queue),expanded_review_endpoints=len(queue),held_endpoints=len(remaining),
        original_watchlist_449_remaining_before=3,original_watchlist_449_repaired=0,original_watchlist_449_remaining_after=3,
        preflight_candidates=len(proposals),preflight_declines=dict(Counter(declines.values())),protected_scientific_claims=sorted(protected),
        remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),
        existing_name_collision_evidence=j.fingerprint(OUTPUT/'CURRENT_COMPLETE_NAME_COLLISIONS.jsonl'),decline_evidence=j.fingerprint(OUTPUT/'CURRENT_DECLINES.jsonl'),
        record_preimages_saved=False,graph_backups=0,no_claim_or_edge_deleted=True,no_metadata_fields_added=True,
        boundaries=['Finite current-source non-molecular full mentions, with original declared roles and word-interior-only gene proof.',
            'No existing same-name, alias, case or generated-ID ambiguity is bypassed by creating another node.',
            'Molecular, genetic, drug, unspecified-role and mixed-reference cases remain held.',
            'All scientific fields, quotes, values, conditions and old audit records unchanged.',
            'Generic literal concepts do not assert an ontology type; names and IDs are not salted per paper.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',claims=len(events),endpoints=len(fixed),new_nodes=len(new),historic_remaining=3,expanded_remaining=len(remaining),declines=plan['preflight_declines'])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='FAILED',at=j.utc_now(),error=repr(error),graph_modified=False));raise
