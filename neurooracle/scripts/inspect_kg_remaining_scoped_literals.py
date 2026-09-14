"""R49 read-only exact references, node scope and current remaining claim evidence."""
from collections import Counter, defaultdict
from pathlib import Path
import os
import re
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from inspect_kg_historic_literal_candidates import node_projection
from inspect_kg_semantic_hold_sources import projection
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner
from neurooracle.src.relation_evidence import name_key

OUTPUT = j.OUTPUT/'round49_remaining_scoped_literals'
R47 = j.OUTPUT/'round47_historic_mentions'
R48 = j.OUTPUT/'round48_historic_literal_repair'
META_FIELDS = ('anchor_role','atom_type','atom_types','curation_scope','staging_source','claim_case_study_ids','paper_case_study_ids')
INTENDED_NAMES = {'plasma BACE1 concentration', 'mouse valosin-containing protein',
    'prenatal tobacco exposure after pregnancy knowledge', 'VCP'}


def scope_projection(row):
    md = row.get('metadata') or {}
    return dict(**node_projection(row), definition_sha256=digest(row.get('definition')),
        spatial_mapping_sha256=digest(row.get('spatial_mapping')),
        reviewed_metadata={k:md[k] for k in META_FIELDS if k in md})


def owned_projection(ordinal, row):
    return dict(ordinal=ordinal, edge_sha256=digest(row), claim_id=edge_owner(row),
        source_id=row['source_id'], target_id=row['target_id'], relation_type=row['relation_type'],
        source=row.get('source'), confidence=row.get('confidence'), evidence_ref=row.get('evidence_ref'),
        metadata_keys=sorted((row.get('metadata') or {}).keys()), metadata_sha256=digest(row.get('metadata') or {}),
        nonendpoint_sha256=digest({k:v for k,v in row.items() if k not in {'source_id','target_id'}}),
        complete_record_not_saved=True)


def detail_references(connection, ids):
    counts={nid:dict(atoms=0,mappings_source=0,mappings_target=0) for nid in ids}
    if not ids:return counts
    markers=','.join('?' for _ in ids)
    for table,column,label in [('atoms','source_mention_id','atoms'),('mappings','source_id','mappings_source'),('mappings','target_id','mappings_target')]:
        for nid,count in connection.execute(f'SELECT {column},COUNT(*) FROM {table} WHERE {column} IN ({markers}) GROUP BY {column}',sorted(ids)):
            counts[nid][label]=count
    return counts


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),phase=phase,at=j.utc_now(),**values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'inspection exists; inspect bindings')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');p=j.read_json(R48/'PLAN.json')
    require(c['status']=='MANUAL_ACTIVE' and c['active_process']['kind']=='historic_literal_repair','expected active R48 with unchanged R46 accepted source')
    require(c['current_graph']==p['graph'] and c['current_acceptance']==p['source_acceptance'],'source advanced')
    plan_fp=j.fingerprint(R48/'PLAN.json');r47_fp=j.fingerprint(R47/'SOURCE_INSPECTION.json');r47=j.read_json(r47_fp['path'])
    for fp in [r47['code'],*r47['artifacts'].values()]:small_check(fp)
    declined=rows(R48/'HISTORIC_DECLINES.jsonl');remaining={r['claim_id'] for r in declined}
    require(len(declined)==len(remaining)==22,'next historic remainder differs')
    original_nodes={r['node_id']:r for r in rows(R47/'EXISTING_NODE_WITNESSES.jsonl') if not r['node_id'].startswith('CLM_CONCEPT:imaging_mention_')}
    old_incidents=[r for r in rows(R47/'EXISTING_INCIDENTS.jsonl') if r['node_id'] in original_nodes]
    scopes=j.read_json(c['current_scope_findings']['path']);issues=rows(c['current_issues']['path'])
    selected=remaining|set(scopes['current_claim_hashes'])|{r['claim_id'] for r in issues}|{r['claim_id'] for r in old_incidents}
    require(not selected & {r['claim_id'] for r in p['events']},'current planned mutation overlaps read-only scope')
    old_projection={r['claim_id']:r for r in rows(j.OUTPUT/'round45_semantic_sources/CURRENT_CLAIM_PROJECTIONS.jsonl')}
    node_ids=set(original_nodes)
    for cid in selected & set(old_projection):
        node_ids.update(old_projection[cid]['science'][s+'_id'] for s in ('subject','object'))
    names={name_key(n).casefold() for n in INTENDED_NAMES}
    names.update(name_key(r['name']).casefold() for r in original_nodes.values())
    needle=re.compile('|'.join(re.escape(n) for n in sorted(original_nodes)))
    claims={};nodes={};name_candidates={};owned=[];references=[];incidents=[];counts=Counter();seen_old=set()
    code=j.fingerprint(Path(__file__));census=j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    progress('FULL_CURRENT_REMAINING_CLAIMS_EXISTING_NODE_SCOPE_AND_ALL_EXACT_REFERENCES',selected_claims=len(selected),reviewed_nodes=len(original_nodes))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in selected:
                    claims[key]=projection(row)
                    if key in old_projection:require(claims[key]==old_projection[key],'selected prior claim changed')
                if key in node_ids:nodes[key]=scope_projection(row)
                if key in original_nodes:
                    require(node_projection(row)==original_nodes[key],'existing target changed');seen_old.add(key)
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata']
                    for side in ('subject','object'):
                        if md.get(side+'_id') in original_nodes:
                            incidents.append(dict(node_id=md[side+'_id'],claim_id=key,side=side,claim_sha256=digest(row),name=md.get(side+'_name')))
                else:
                    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
                    if labels & names:name_candidates[key]=scope_projection(row)
            elif kind=='edge' and edge_owner(row) in selected:owned.append(owned_projection(int(key),row))
            encoded=compact(row)
            if needle.search(encoded):
                refs=list(exact_references(row,set(original_nodes)))
                if refs:references.append(dict(record_kind=kind,record_key=key,record_sha256=digest(row),references=refs))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'current full source SHA differs')
    require(set(claims)==selected and seen_old==set(original_nodes),'selected source closure missing')
    expected=[{k:r[k] for k in ('node_id','claim_id','side','claim_sha256','name')} for r in old_incidents]
    require(incidents==expected,'existing complete-node incident scope differs')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for cid,row in claims.items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(row['claim_sha256'],),'claim/census differs')
    finally:db.close()
    detail=sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro',uri=True)
    try:deps=detail_references(detail,set(original_nodes))
    finally:detail.close()
    artifacts={'CURRENT_CLAIM_PROJECTIONS.jsonl':list(claims.values()),'CURRENT_NODE_SCOPE_WITNESSES.jsonl':list(nodes.values()),
        'CURRENT_FULL_NAME_CANDIDATES.jsonl':list(name_candidates.values()),'CURRENT_OWNED_EDGE_PROJECTIONS.jsonl':owned,
        'EXACT_EXISTING_NODE_REFERENCES.jsonl':references,'CURRENT_EXISTING_INCIDENCES.jsonl':incidents}
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced during inspection')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    for file,records in artifacts.items():write_rows(OUTPUT/file,records)
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        current_census=c['current_paper_census'],current_detail_store=c['current_detail_store'],pending_disjoint_plan=plan_fp,historic_source_review=r47_fp,
        code=code,full_source_sha_verified=True,full_census_sha_verified=True,counts=dict(counts),selected_claims=len(selected),existing_review_nodes=len(original_nodes),
        existing_claim_incidences=len(incidents),owned_edge_count=len(owned),exact_reference_records=len(references),detail_dependencies=deps,
        graph_modified=False,record_preimages_saved=False,source_science_and_audit_not_revalidated=True,
        artifacts={file:j.fingerprint(OUTPUT/file) for file in artifacts}))
    progress('COMPLETED',claims=len(claims),existing_nodes=len(original_nodes),incidents=len(incidents),exact_reference_records=len(references),owned_edges=len(owned))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='FAILED',at=j.utc_now(),error=repr(error),graph_modified=False))
        raise
