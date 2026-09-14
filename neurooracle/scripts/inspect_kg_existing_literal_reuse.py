"""R54 read-only whole-name reuse, existing incidence scope and exact source refs."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from classify_kg_expanded_literal_candidates import candidate_gate
from inspect_kg_remaining_scoped_literals import scope_projection
from inspect_kg_semantic_hold_sources import projection
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.src.kg_literal_endpoint_repair import edge_owner,reviewed_edges
from neurooracle.src.relation_evidence import name_key

OUTPUT=j.OUTPUT/'round54_existing_literal_review'
REVIEW=j.OUTPUT/'round52_expanded_literal_review'
METADATA_KEYS={'anchor_role','atom_type','atom_types','curation_scope','staging_source','claim_case_study_ids','paper_case_study_ids'}


def candidate_node_gate(review,node):
    if review['literal_candidate_gate']!='existing_full_name_or_case_alias_requires_reuse_review':return 'not_existing_name_review'
    if len(review['existing_name_or_alias_candidates'])!=1:return 'multiple_candidates'
    reason=candidate_gate(dict(review,existing_name_or_alias_candidates=[]),set())
    if reason:return reason
    if name_key(review['name'])!=name_key(node['name']):return 'case_or_alias_not_exact_primary_name'
    if node['semantic_types'] or node['external_ids'] or node['aliases']:return 'ontology_or_alias_scope'
    if node['definition_sha256']!=digest('') or node['spatial_mapping_sha256']!=digest(None):return 'nonempty_definition_or_spatial_scope'
    if not set(node['metadata_keys'])<=METADATA_KEYS:return 'additional_metadata_scope'
    md=node['reviewed_metadata']
    if set(md)!=set(node['metadata_keys']):return 'metadata_projection_incomplete'
    roles=[md.get('atom_type'),*(md.get('atom_types') or [])]
    if any(v in {'gene','gene_target','protein','pathway'} for v in [*node['domain_tags'],*roles]):return 'existing_molecular_role'
    return None


def incidence_gate(node,incidences,nonclaim_edges):
    if nonclaim_edges:return 'existing_nonclaim_relations_need_separate_scope'
    if not incidences:return 'no_current_claim_incidence'
    if any(name_key(r['name'])!=name_key(node['name']) for r in incidences):return 'existing_incident_full_name_differs'
    return None


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'inspection exists')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='MANUAL_ACTIVE' and c['active_process']['kind']=='expanded_literal_repair','expected pending R53')
    pending_fp=j.fingerprint(j.OUTPUT/'round53_expanded_literal_repair/PLAN.json');pending=j.read_json(pending_fp['path'])
    require(pending['graph']==c['current_graph'] and pending['source_acceptance']==c['current_acceptance'],'pending source differs')
    classification_fp=j.fingerprint(REVIEW/'CANDIDATE_CLASSIFICATION.json');classification=j.read_json(classification_fp['path'])
    small_check(classification['classified']);inspection=j.read_json(classification['source_inspection']['path'])
    for fp in [classification['source_inspection'],classification['code'],inspection['artifacts']['EXISTING_NAME_CANDIDATES.jsonl']]:small_check(fp)
    reviews=rows(classification['classified']['path'])
    nodes={r['node_id']:r for r in rows(inspection['artifacts']['EXISTING_NAME_CANDIDATES.jsonl']['path'])}
    held={(r['claim_id'],r['side']):r for r in rows(pending['remaining_queue']['path'])}
    proposed=[];declines=[]
    for r in reviews:
        if (r['claim_id'],r['side']) not in held or r['literal_candidate_gate']!='existing_full_name_or_case_alias_requires_reuse_review':continue
        choices=r['existing_name_or_alias_candidates']
        reason=candidate_node_gate(r,nodes[choices[0]]) if len(choices)==1 else 'multiple_candidates'
        if reason:declines.append(dict(claim_id=r['claim_id'],side=r['side'],reason=reason))
        else:proposed.append(r)
    selected={r['claim_id'] for r in proposed};targets={r['existing_name_or_alias_candidates'][0] for r in proposed}
    require(selected and not selected&{r['claim_id'] for r in pending['events'] if len(r['changes'])==2},'unexpected full-claim overlap')
    code=j.fingerprint(Path(__file__));j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    claims={};seen_targets=set();owned=defaultdict(list);incidences=defaultdict(list);other_edges=defaultdict(list);counts=Counter()
    progress('FULL_CURRENT_EXISTING_PRIMARY_NAMES_AND_COMPLETE_INCIDENCE_SCOPE',proposed_endpoints=len(proposed),targets=len(targets))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in targets:
                    require(scope_projection(row)==nodes[key],'existing node changed');seen_targets.add(key)
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata'];inner=md.get('metadata') or {}
                    if key in selected:claims[key]=row
                    for side in ('subject','object'):
                        if md.get(side+'_id') in targets:
                            incidences[md[side+'_id']].append(dict(node_id=md[side+'_id'],claim_id=key,side=side,name=md.get(side+'_name'),
                                claim_sha256=digest(row),nonidentity_sha256=digest(nonidentity_claim(row)),
                                outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type')))
            elif kind=='edge':
                owner=edge_owner(row)
                if owner in selected:owned[owner].append((int(key),row))
                if not owner:
                    for nid in {row.get('source_id'),row.get('target_id')}&targets:
                        other_edges[nid].append(dict(ordinal=int(key),edge_sha256=digest(row),relation_type=row['relation_type']))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full current source SHA differs')
    require(set(claims)==selected and seen_targets==targets,'selected scope incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'source counts differ')
    endpoints=[];reasons=Counter()
    for r in proposed:
        cid=r['claim_id'];side=r['side'];nid=r['existing_name_or_alias_candidates'][0];row=claims[cid];md=row['metadata']
        require(digest(row)==r['claim_sha256']==held[cid,side]['claim_sha256'] and md[side+'_id']==r['current_node_id']
            and md[side+'_name']==r['name'],'queued current source differs')
        reason=incidence_gate(nodes[nid],incidences[nid],other_edges[nid])
        try:require(reviewed_edges(cid,row,row,owned[cid])==[],'identity neutral closure differs')
        except ValueError as error:reason='current_owned_closure: '+str(error)
        reasons[reason or 'whole_name_reuse_candidate']+=1
        endpoints.append(dict(r,reuse_candidate_gate=reason,target_id=nid,target_node_sha256=nodes[nid]['node_sha256'],
            owned_edges=[dict(ordinal=o,edge_sha256=digest(e)) for o,e in owned[cid]],
            complete_science_and_old_audit_not_revalidated=True))
    data={'CURRENT_REUSE_CANDIDATES.jsonl':endpoints,'CURRENT_CLAIM_PROJECTIONS.jsonl':[projection(claims[k]) for k in sorted(claims)],
        'CURRENT_TARGET_WITNESSES.jsonl':[nodes[k] for k in sorted(targets)],
        'CURRENT_TARGET_INCIDENCES.jsonl':[r for nid in sorted(targets) for r in incidences[nid]],
        'CURRENT_NONCLAIM_RELATIONS.jsonl':[dict(node_id=nid,edges=other_edges[nid]) for nid in sorted(targets) if other_edges[nid]],
        'PRELIMINARY_DECLINES.jsonl':declines}
    for name,records in data.items():write_rows(OUTPUT/name,records)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    result=dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        source_classification=classification_fp,pending_identity_plan=pending_fp,code=code,full_source_sha_verified=True,
        reviewed_endpoints=len(endpoints),reviewed_targets=len(targets),selected_claims=len(claims),existing_incidences=sum(map(len,incidences.values())),
        gate_counts=dict(reasons),preliminary_declines=dict(Counter(r['reason'] for r in declines)),
        pending_changed_claim_overlap=sorted(selected&{r['claim_id'] for r in pending['events']}),
        existing_node_and_all_detail_records_unchanged=True,graph_modified=False,record_preimages_saved=False,
        artifacts={name:j.fingerprint(OUTPUT/name) for name in data})
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',result)
    progress('COMPLETED',endpoints=len(endpoints),targets=len(targets),reasons=dict(reasons))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='FAILED',at=j.utc_now(),pid=os.getpid(),error=repr(error),graph_modified=False));raise
