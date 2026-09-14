"""R64 current seven-name scope, all exact references and proposed nominal reuse.

No KG mutation. The complete graph is read once; only bounded projections,
hashes and endpoint deltas are persisted, never claim/node preimages.
"""
from collections import Counter, defaultdict
from pathlib import Path
import os
import re
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from inspect_kg_remaining_scoped_literals import scope_projection, detail_references
from inspect_kg_semantic_hold_sources import projection
from kg_accepted_candidate_lineage import require
from plan_kg_literal_endpoint_repair import expected_shared
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_bulk_identity import change_claim
from neurooracle.src.kg_identity_pilot import digest, nonidentity_claim
from neurooracle.src.kg_literal_endpoint_repair import edge_owner, reviewed_edges
from neurooracle.src.relation_evidence import name_key, relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
from neurooracle.src.correlation_grouping import IndexTerms

OUTPUT=j.OUTPUT/'round64_nominal_scope_review'
R60=j.OUTPUT/'round60_multiple_name_scope_review'
TARGETS={
    'hippocampal and amygdala volume':'CLM_CONCEPT:hippocampal_and_amygdala_volume_60f419dd6879',
    'incident dementia':'CLM_CONCEPT:incident_dementia',
    'major depressive disorder severity':'CLM_CONCEPT:major_depressive_disorder_severity',
    'neuromelanin-sensitive MRI substantia nigra signal':'CLM_CONCEPT:neuromelanin_sensitive_mri_substantia_nigra_signal',
    'resting-state functional connectivity patterns':'CLM_CONCEPT:resting_state_functional_connectivity_patterns_74fa96109660',
    'supplementary motor area activation':'CLM_CONCEPT:supplementary_motor_area_activation',
    'total gray matter volume':'CLM_CONCEPT:total_gray_matter_volume_f007cd0f22dd',
}
SEARCH_NAMES={name_key(name).casefold():name for name in TARGETS}


def progress(phase,**values):
    state=dict(status='READ_ONLY_INSPECTING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'STATE.json',state);print(compact(state),flush=True)


def selected_groups(records):
    selected=[g for g in records if g['name'] in TARGETS]
    require(len(selected)==len(TARGETS) and {g['name'] for g in selected}==set(TARGETS),'finite name set differs')
    for g in selected:
        require(g['review_reasons']==['existing_domain_or_atom_role_differs'],'additional source scope issue')
        require(g['candidate_ids']==g['current_candidate_ids'] and TARGETS[g['name']] in g['candidate_ids'],'target not reviewed')
    return selected


def reference_kind(kind,key,ref,selected,node_ids):
    path=ref['json_path'];nid=ref['claim_id']
    if kind=='node' and key==nid and path==['id']:return 'node_own_id'
    if kind=='node' and key in selected and path in (
        ['metadata','subject_id'],['metadata','object_id'],
        ['metadata','metadata','subject_id'],['metadata','metadata','object_id']):return 'claim_endpoint'
    if kind=='edge' and path in (['source_id'],['target_id']):return 'edge_endpoint'
    return 'other_exact_reference'


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'R64 exists; reuse frozen inspection')
    tests_fp=j.fingerprint(OUTPUT/'TEST_RESULTS.xml');suite=ET.parse(tests_fp['path']).getroot().find('testsuite')
    require(int(suite.get('tests'))>=12 and all(int(suite.get(k,0))==0 for k in ('failures','errors','skipped')),'inspection tests incomplete')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None,'active KG writer')
    require(Path(c['current_acceptance']['path']).parent.name=='round59_observational_semantics','requires R59 source')
    acceptance=j.read_json(c['current_acceptance']['path'])
    for fp in [c['current_acceptance'],c['current_paper_census'],*acceptance['code']]:small_check(fp)
    old_fp=j.fingerprint(R60/'SOURCE_INSPECTION.json');old=j.read_json(old_fp['path'])
    for fp in [old['code'],*old['artifacts'].values()]:small_check(fp)
    r59_plan=j.read_json(acceptance['provenance_plan']['path'])
    require(old['graph']==r59_plan['graph'] and old['pending_disjoint_plan']==acceptance['provenance_plan']
        and acceptance['checks']['no_existing_concept_metadata_change']
        and acceptance['checks']['inverse_reproduces_all_source_node_and_edge_record_digests'],'missing R60 to R59 bridge')
    groups=selected_groups(rows(R60/'MULTIPLE_NAME_GROUP_REVIEW.jsonl'))
    target_ids={nid for g in groups for nid in g['candidate_ids']}
    old_nodes={n['node_id']:n for n in rows(R60/'CURRENT_TARGET_WITNESSES.jsonl') if n['node_id'] in target_ids}
    old_inc=[r for r in rows(R60/'CURRENT_TARGET_INCIDENCES.jsonl') if r['node_id'] in target_ids]
    queue=rows(c['current_gene_holds']['path']);queued_ids={cid for g in groups for cid in g['queued_claim_ids']}
    queued=[r for r in queue if r['claim_id'] in queued_ids and r['name'] in TARGETS]
    require(len(queued)==14 and len(target_ids)==15 and len(old_inc)==76,'finite seven-group size differs')
    selected=queued_ids|{r['claim_id'] for r in old_inc}
    require(not selected&{r['claim_id'] for r in r59_plan['events']},'R60 bridge overlaps R59 mutation')
    gene_ids={r['current_node_id'] for r in queued}
    census=j.read_json(c['current_paper_census']['path'])
    j.guards([c['current_graph'],c['current_detail_store'],census['database'],c['formal_sources']])
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/tests/test_kg_nominal_scope_inspection.py')]
    claims={};nodes={};genes={};incidences=[];references=[];owned=defaultdict(list);others=[];matches=defaultdict(set);counts=Counter()
    needle=re.compile('|'.join(re.escape(nid) for nid in sorted(target_ids)))
    progress('FULL_CURRENT_NODE_SCOPE_INCIDENCES_AND_EXACT_REFERENCES',groups=len(groups),nodes=len(target_ids),claims=len(selected))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in target_ids:nodes[key]=scope_projection(row)
                if key in gene_ids:genes[key]=scope_projection(row)
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata'];inner=md.get('metadata') or {}
                    if key in selected:claims[key]=row
                    for side in ('subject','object'):
                        nid=md.get(side+'_id')
                        if nid in target_ids:
                            incidences.append(dict(node_id=nid,claim_id=key,side=side,name=md.get(side+'_name'),claim_sha256=digest(row),
                                outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type')))
                else:
                    for label in {name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}:
                        if label in SEARCH_NAMES:matches[label].add(key)
            if kind=='edge':
                owner=edge_owner(row)
                if owner in selected:owned[owner].append((int(key),row))
                if not owner and {row.get('source_id'),row.get('target_id')}&target_ids:
                    others.append(dict(ordinal=int(key),edge_sha256=digest(row),source_id=row['source_id'],target_id=row['target_id']))
            payload=compact(row)
            if needle.search(payload):
                for ref in exact_references(row,target_ids):
                    references.append(dict(kind=kind,key=key,node_id=ref['claim_id'],json_path=ref['json_path'],record_sha256=digest(row),
                        reference_kind=reference_kind(kind,key,ref,selected,target_ids),
                        **(dict(edge_owner=edge_owner(row)) if kind=='edge' else {})))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full source graph SHA differs')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'full source counts differ')
    require(nodes==old_nodes and set(claims)==selected and set(genes)==gene_ids,'current bounded source scope differs')
    order=lambda r:(r['node_id'],r['claim_id'],r['side'])
    require(sorted(incidences,key=order)==sorted(old_inc,key=order) and not others,'complete source incidences changed')
    for g in groups:require(matches[g['name'].casefold()]==set(g['candidate_ids']),'name/alias candidate set changed')
    db=sqlite3.connect(Path(c['current_detail_store']['path']).as_uri()+'?mode=ro',uri=True)
    try:details=detail_references(db,target_ids)
    finally:db.close()
    require(sha256(Path(c['current_detail_store']['path']))==c['current_detail_store']['sha256'],'complete detail SHA differs')
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    redirects={nid:TARGETS[g['name']] for g in groups for nid in g['candidate_ids'] if nid!=TARGETS[g['name']]}
    proposed=defaultdict(list)
    for r in incidences:
        if r['node_id'] in redirects:
            proposed[r['claim_id']].append(dict(side=r['side'],name=name_key(r['name']),old_id=r['node_id'],target_id=redirects[r['node_id']],reason='same_nominal_concept'))
    for r in queued:
        row=claims[r['claim_id']];md=row['metadata']
        require(digest(row)==r['claim_sha256'] and md[r['side']+'_id']==r['current_node_id'] and md[r['side']+'_name']==r['name'],'queued source changed')
        proposed[r['claim_id']].append(dict(side=r['side'],name=name_key(r['name']),old_id=r['current_node_id'],target_id=TARGETS[r['name']],reason='gene_endpoint_to_existing_nominal_concept'))
    events=[];edges=[]
    for cid,changes in sorted(proposed.items()):
        original=claims[cid]
        require(len({r['side'] for r in changes})==len(changes),'duplicate proposed endpoint')
        event=dict(claim_id=cid,claim_sha256=digest(original),changes=changes,nonidentity_sha256=digest(nonidentity_claim(original)))
        current=change_claim(original,event);event.update(current_node_sha256=digest(current),
            old_relation_id=relation_id(terms.relation_key(original['metadata'])),new_relation_id=relation_id(terms.relation_key(current['metadata'])))
        events.append(event);edges.extend(reviewed_edges(cid,original,current,owned[cid]))
    removable=[];preserve=[]
    for nid in sorted(redirects):
        refs=[r for r in references if r['node_id']==nid]
        reasons=[]
        if any(details[nid].values()):reasons.append('immutable_detail_store_source_reference')
        if any(r['reference_kind']=='other_exact_reference' for r in refs):reasons.append('other_exact_graph_reference')
        if any(r['kind']=='edge' and r['edge_owner'] not in proposed for r in refs):reasons.append('edge_outside_selected_claims')
        info=dict(node_id=nid,target_id=redirects[nid],node_sha256=nodes[nid]['node_sha256'],detail_references=details[nid],reasons=reasons)
        (preserve if reasons else removable).append(info)
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'complete source census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    try:
        for cid,row in claims.items():require(db.execute('SELECT node_sha FROM claims WHERE cid=?',(cid,)).fetchone()==(digest(row),),'current census claim differs')
        shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']))
    finally:db.close()
    data={
        'CURRENT_CLAIM_PROJECTIONS.jsonl':[projection(claims[cid]) for cid in sorted(claims)],
        'CURRENT_TARGET_WITNESSES.jsonl':[nodes[nid] for nid in sorted(nodes)],
        'CURRENT_GENE_WITNESSES.jsonl':[genes[nid] for nid in sorted(genes)],
        'CURRENT_TARGET_INCIDENCES.jsonl':sorted(incidences,key=order),'CURRENT_EXACT_REFERENCES.jsonl':references,
        'PROPOSED_CLAIM_EVENTS.jsonl':events,'PROPOSED_EDGE_EVENTS.jsonl':sorted(edges,key=lambda e:e['ordinal']),
        'SOURCE_QUEUED_ENDPOINTS.jsonl':queued,'NOMINAL_GROUPS.jsonl':groups,
        'REMOVABLE_SOURCE_NODES.jsonl':removable,'PRESERVED_SOURCE_NODES.jsonl':preserve,
    }
    for name,values in data.items():write_rows(OUTPUT/name,values)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and all(j.fingerprint(fp['path'])==fp for fp in code),'source/code changed while inspecting')
    j.guards([c['current_graph'],c['current_detail_store'],census['database'],c['formal_sources']])
    receipt=dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],census=c['current_paper_census'],r60_inspection=old_fp,code=code,tests=tests_fp,
        full_source_sha_verified=True,full_detail_sha_verified=True,full_census_sha_verified=True,
        groups=len(groups),target_nodes=len(nodes),original_incidences=len(incidences),queued_endpoints=len(queued),
        changed_claims=len(events),changed_endpoints=sum(len(e['changes']) for e in events),changed_edges=len(edges),
        redirects=redirects,canonical_targets=TARGETS,removable_nodes=len(removable),preserved_source_nodes=len(preserve),
        exact_reference_kinds=dict(Counter(r['reference_kind'] for r in references)),expected_shared_claim_ids=shared,
        graph_modified=False,record_preimages_saved=False,scientific_reaudit_performed=False,
        artifacts={name:j.fingerprint(OUTPUT/name) for name in data})
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',receipt)
    progress('COMPLETED',claims=len(events),endpoints=receipt['changed_endpoints'],removable=len(removable),preserved=len(preserve))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'STATE.json',dict(status='FAILED',pid=os.getpid(),at=j.utc_now(),error=repr(error),graph_modified=False));raise
