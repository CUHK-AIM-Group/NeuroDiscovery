"""R50 current-source plan for finite same-name reuse and neutral old branches."""
from bisect import bisect_left
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
from inspect_kg_historic_literal_candidates import node_projection
from inspect_kg_remaining_scoped_literals import scope_projection,owned_projection
from inspect_kg_semantic_hold_sources import projection
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src import kg_reviewed_literal_reuse as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round50_literal_scope_reuse'
REVIEW=j.OUTPUT/'round49_remaining_scoped_literals'


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def inspection_bridge(inspection,baseline,accepted,prior_plan):
    require(accepted['status']=='CURRENT_HISTORIC_LITERAL_REPAIR_APPLIED' and accepted['graph']==baseline['current_graph']
        and accepted['source_graph']==inspection['graph'],'not the reviewed R48 advancement')
    require(accepted['provenance_plan']==inspection['pending_disjoint_plan']
        and accepted['checks']['inverse_reproduces_all_source_node_and_edge_record_digests'],'R48 unchanged-source bridge missing')
    require(not set(repair.SPECS)&{r['claim_id'] for r in prior_plan['events']},'reviewed claim was changed by R48')
    require(inspection['full_source_sha_verified'] and inspection['full_census_sha_verified'],'R49 incomplete')


def adjusted_ordinal(ordinal,removed):
    require(ordinal not in removed,'retired ordinal has no current identity')
    return ordinal-bisect_left(removed,ordinal)


def original_ordinal(ordinal,removed):
    original=ordinal
    for retired in removed:
        if retired<=original:original+=1
        else:break
    return original


def protein_identity_candidates(row):
    text=' '.join(str(v) for v in (row.get('external_ids') or {}).values())
    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
    species_labels={'mouse valosin-containing protein','valosin-containing protein (mus musculus)',
        'vcp protein, mouse','valosin-containing protein, mouse','tera_mouse'}
    return bool(re.search(r'(?<![A-Za-z0-9])Q01853(?![A-Za-z0-9])',str(row.get('id'))+' '+text) or labels & species_labels)


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'source boundary')
    receipt=j.read_json(c['current_acceptance']['path']);prior=j.read_json(receipt['provenance_plan']['path'])
    inspection_fp=j.fingerprint(REVIEW/'SOURCE_INSPECTION.json');inspection=j.read_json(inspection_fp['path'])
    inspection_bridge(inspection,c,receipt,prior)
    public_fp=j.fingerprint(REVIEW/'PUBLIC_SOURCE_FETCH.json');public=j.read_json(public_fp['path'])
    require(public['source_inspection']==inspection_fp and public['protein_witness']['status']=='PUBLIC_PROTEIN_WITNESS_FETCHED','public proof not bound')
    for fp in [inspection['code'],*inspection['artifacts'].values(),public['code'],public['response'],public['protein_witness']['response'],*receipt['code']]:small_check(fp)
    source_proof=repair.public_scope_proof(Path(public['response']['path']).read_text(encoding='utf8'),j.read_json(public['protein_witness']['response']['path']))
    reviews={r['claim_id']:r for r in rows(inspection['artifacts']['CURRENT_CLAIM_PROJECTIONS.jsonl']['path'])}
    require(set(repair.SPECS)<=set(reviews),'finite claims not inspected')
    old_nodes={r['node_id']:r for r in rows(inspection['artifacts']['CURRENT_NODE_SCOPE_WITNESSES.jsonl']['path'])}
    old_names={r['node_id']:r for r in rows(inspection['artifacts']['CURRENT_FULL_NAME_CANDIDATES.jsonl']['path'])}
    old_incidents=rows(inspection['artifacts']['CURRENT_EXISTING_INCIDENCES.jsonl']['path'])
    old_owned=rows(inspection['artifacts']['CURRENT_OWNED_EDGE_PROJECTIONS.jsonl']['path'])
    genes={r['node_id']:r for r in rows(j.OUTPUT/'round47_historic_mentions/GENE_WITNESSES.jsonl')}
    queue=rows(c['current_gene_holds']['path']);queue_keys={(r['claim_id'],r['side']) for r in queue}
    require(len(queue)==len(queue_keys)==5509,'current gene queue differs')
    selected=set(repair.SPECS);target_ids={r[2] for r in repair.SPECS.values()}
    names={name_key(r[1]).casefold() for cid,r in repair.SPECS.items() if cid!=repair.MOUSE}
    names.add(repair.MOUSE_NAME.casefold())
    old_review_ids=set(inspection['detail_dependencies'])
    claims={};nodes={};matches=defaultdict(set);owned=defaultdict(list);incidents=[];seen_old=set();seen_genes=set();collisions=[];counts=Counter()
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_reviewed_literal_reuse.py')]
    census=j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    progress('FULL_CURRENT_SOURCE_21_CLAIMS_NAMES_PROTEIN_ACCESSION_AND_OWNED_BRANCHES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in genes:
                    require(digest(row)==genes[key]['node_sha256'],'gene witness changed');seen_genes.add(key)
                if key in selected:
                    require(projection(row)==reviews[key],'current selected claim/source fields changed');claims[key]=row
                if key in target_ids:nodes[key]=row
                if key=='CUI:C1613884':
                    require(scope_projection(row)==old_names[key],'unscoped VCP protein candidate changed')
                if key in old_review_ids:
                    require(scope_projection(row)==old_nodes[key],'existing source-anchor scope changed');seen_old.add(key)
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata']
                    for side in ('subject','object'):
                        if md.get(side+'_id') in old_review_ids:
                            incidents.append(dict(node_id=md[side+'_id'],claim_id=key,side=side,claim_sha256=digest(row),name=md.get(side+'_name')))
                else:
                    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
                    for label in names & labels:matches[label].add(key)
                    if protein_identity_candidates(row):collisions.append(scope_projection(row))
            elif kind=='edge' and repair.edge_owner(row) in selected:owned[repair.edge_owner(row)].append((int(key),row))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'current full source SHA differs')
    require(set(claims)==selected and seen_genes==set(genes) and seen_old==old_review_ids,'current source closure incomplete')
    require(incidents==old_incidents,'existing source-anchor incidences changed')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source census differs')
    write_rows(OUTPUT/'PROTEIN_IDENTITY_COLLISIONS.jsonl',collisions)
    require(not collisions,'mouse protein identity already represented; review reuse explicitly')
    detail=sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        import json
        item=detail.execute('SELECT payload_json FROM cuis WHERE record_id=?',('CUI:C1613884',)).fetchone()
        require(item is not None,'unscoped VCP protein detail witness missing')
        protein_cui=json.loads(item[0])
        require(protein_cui.get('preferred_name')=='VCP' and not protein_cui.get('aliases')
            and protein_cui.get('external_ids')=={'UMLS_CUI':'C1613884'},'new species-specific detail evidence requires review')
        unscoped_protein_detail=dict(record_id='CUI:C1613884',record_sha256=digest(protein_cui),
            name=protein_cui['preferred_name'],semantic_types=protein_cui['semantic_types'],external_ids=protein_cui['external_ids'],
            own_taxon_or_UniProt_accession_not_provided=True,not_merged_without_species_proof=True)
    finally:detail.close()
    current_owned=[owned_projection(o,r) for cid in sorted(owned) for o,r in owned[cid]]
    require(sorted(current_owned,key=lambda r:r['ordinal'])==[r for r in old_owned if r['claim_id'] in selected],'fresh current owned-edge scope changed')
    expected_candidates=defaultdict(set)
    for nid,row in old_names.items():
        for label in [row['name'],*(row.get('aliases') or [])]:expected_candidates[name_key(label).casefold()].add(nid)
    expected_candidates['chorio-scleral interface thickness'].add(repair.literal_node('chorio-scleral interface thickness')['id'])
    for name in names-{repair.MOUSE_NAME.casefold()}:
        require(matches[name]==expected_candidates[name], 'additional existing full-name/alias candidate needs review: '+name)
    new={};reused={}
    for cid,(side,name,nid) in repair.SPECS.items():
        if nid in nodes:
            node=nodes[nid];preferred=node.get('preferred_name')
            expected='Chronic mild traumatic brain injury in Iraq and Afghanistan veterans' if nid==repair.CHRONIC else name
            require(name_key(preferred)==expected,'existing full name differs')
            if cid==repair.CHORIO:require(node==repair.literal_node(name),'R48 complete chorio literal differs')
            else:
                require(nid in old_nodes and scope_projection(node)==old_nodes[nid],'existing full source-node proof missing')
                require(not node.get('semantic_types') and not node.get('external_ids'),'external identity was not approved for reuse')
            reused[nid]=digest(node)
        else:
            target_name=repair.MOUSE_NAME if cid==repair.MOUSE else name
            node=repair.literal_node(target_name)
            require(node['id']==nid and not matches[target_name.casefold()],'unreviewed new-node identity/collision')
            new[nid]=dict(id=nid,name=target_name,node_sha256=digest(node))
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    events=[];edge_events=[];retired=[]
    for cid,row in sorted(claims.items()):
        side,name,nid=repair.SPECS[cid]
        event,out=repair.reviewed_claim(row,[dict(side=side,name=name,old_id=row['metadata'][side+'_id'],target_id=nid)],genes,reviews,source_proof)
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(out['metadata'])))
        events.append(event);edge_events.extend(repair.reviewed_edges(cid,row,out,owned[cid]))
        _,removed=repair.split_owned(cid,row,owned[cid]);retired.extend(removed)
    removed=sorted(r['ordinal'] for r in retired);require(len(removed)==len(set(removed))==3,'three exact old about branches required')
    for e in edge_events:e['current_ordinal']=adjusted_ordinal(e['ordinal'],removed)
    for r in retired:r['current_kept_ordinal']=adjusted_ordinal(r['kept_ordinal'],removed)
    fixed={(cid,repair.SPECS[cid][0]) for cid in selected if cid not in repair.BROAD_REASSIGN}
    require(len(fixed)==19 and fixed<=queue_keys,'historic-only endpoint correction scope differs')
    remaining=[r for r in queue if (r['claim_id'],r['side']) not in fixed]
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for event in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(event['claim_id'],)).fetchone()==(event['claim_sha256'],event['old_relation_id']),'current claim census differs')
        shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']))
    finally:db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    protected={r['claim_id'] for field in ('current_issues','current_structure_holds') for r in rows(c[field]['path'])}
    require(not selected&protected,'unreviewed semantic/structure hold overlap')
    scope=j.read_json(c['current_scope_findings']['path'])
    require(selected & set(scope['current_claim_hashes'])=={repair.PRENATAL},'unreviewed scientific-scope overlap')
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],source_review=inspection_fp,public_manifest=public_fp,public_source=public['response'],unscoped_existing_VCP_protein=unscoped_protein_detail,
        protein_source=public['protein_witness']['response'],public_scope_proof=source_proof,source_full_sha_verified=True,source_census_full_sha_verified=True,
        prior_disjoint_identity_plan=receipt['provenance_plan'],code=code,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),
        retired_branches=sorted(retired,key=lambda r:r['ordinal']),removed_edge_ordinals=removed,reviewed_claims={cid:reviews[cid] for cid in sorted(selected)},
        gene_witnesses=genes,new_literals=sorted(new.values(),key=lambda r:r['id']),existing_targets={**reused,**{g:w['node_sha256'] for g,w in genes.items()},repair.REGIONAL:old_nodes[repair.REGIONAL]['node_sha256']},
        expected_shared_claim_ids=shared,relation_grouping=POLICY,changed_claims=len(events),changed_endpoints=len(events),changed_edges=len(edge_events),
        added_literal_nodes=len(new),reused_existing_nodes=len(reused),retired_obsolete_about_edges=3,
        original_watchlist_remaining_before=22,original_watchlist_repaired=19,original_watchlist_remaining_after=3,
        expanded_review_endpoints=len(queue),held_endpoints=len(remaining),nonhistoric_complete_name_reassignments=2,
        remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),protein_collision_evidence=j.fingerprint(OUTPUT/'PROTEIN_IDENTITY_COLLISIONS.jsonl'),
        partial_scientific_scope_identity_only=[repair.PRENATAL],all_scientific_scope_holds_remain_open=True,
        expected_counts=dict(c['counts'],nodes=c['counts']['nodes']+len(new),edges=c['counts']['edges']-3),
        no_existing_concept_or_detail_record_modified=True,no_claim_deleted=True,no_new_scientific_assertion=True,no_metadata_fields_added=True,
        record_preimages_saved=False,graph_backups=0)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',claims=len(events),edges=len(edge_events),retired_about=3,new_nodes=len(new),reused_nodes=len(reused),historic_remaining=3,expanded_remaining=len(remaining))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='FAILED',at=j.utc_now(),error=repr(error),graph_modified=False));raise
