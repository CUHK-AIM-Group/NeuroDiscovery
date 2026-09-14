"""R52 read-only expanded queue, full names, gene witnesses and owned closures."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from inspect_kg_historic_literal_candidates import node_projection
from inspect_kg_semantic_hold_sources import projection
from inspect_kg_remaining_scoped_literals import scope_projection
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from neurooracle.src.kg_gene_boundary_repair import word_interior_hits
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import edge_owner,reviewed_edges
from neurooracle.src.claim_semantics import declared_type_atoms
from neurooracle.src.relation_evidence import name_key

OUTPUT=j.OUTPUT/'round52_expanded_literal_review'
PENDING=j.OUTPUT/'round50_literal_scope_reuse/PLAN.json'


def gate(row,endpoint,witness,owned):
    md=row['metadata'];inner=md.get('metadata') or {};side=endpoint['side']
    require(digest(row)==endpoint['claim_sha256'] and md[side+'_id']==endpoint['current_node_id']
        and md[side+'_name']==endpoint['name'],'current queued endpoint differs')
    reasons=[]
    hits=word_interior_hits(md[side+'_name'],witness['labels'])
    if not hits:reasons.append('not_word_interior_alias_only')
    if 'T028' not in witness['semantic_types']:reasons.append('source_not_canonical_gene')
    if inner.get(side+'_id',md[side+'_id'])!=md[side+'_id']:reasons.append('nested_id_conflict')
    outer,inside=md.get(side+'_type'),inner.get(side+'_type')
    if outer and inside and declared_type_atoms(outer)!=declared_type_atoms(inside):reasons.append('nested_type_conflict')
    try:require(reviewed_edges(row['id'],row,row,owned)==[],'identity-neutral edge changed')
    except ValueError as error:reasons.append('owned_closure: '+str(error))
    return dict(word_interior_alias_hits=hits,structural_holds=reasons,outer_type=outer,inner_type=inside,
        declared_roles=sorted(a.value for a in declared_type_atoms(outer or inside)))


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'inspection exists')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='MANUAL_ACTIVE' and c['active_process']['kind']=='reviewed_literal_reuse','expected R50 pending boundary')
    pending_fp=j.fingerprint(PENDING);pending=j.read_json(PENDING)
    require(pending['graph']==c['current_graph'] and pending['source_acceptance']==c['current_acceptance'],'pending source differs')
    small_check(pending['remaining_queue'])
    queue=rows(pending['remaining_queue']['path']);selected={r['claim_id'] for r in queue}
    require(len(queue)==5490 and len({(r['claim_id'],r['side']) for r in queue})==len(queue),'remaining scope differs')
    gene_ids={r['current_node_id'] for r in queue};names={name_key(r['name']).casefold() for r in queue}
    claims={};genes={};candidates={};labels_by_name=defaultdict(set);refs=defaultdict(list);incidences=[];counts=Counter()
    code=j.fingerprint(Path(__file__));j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    progress('FULL_CURRENT_SOURCE_EXPANDED_5490_ENDPOINTS_ALL_EXISTING_NAMES_AND_GENE_WITNESSES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in gene_ids:
                    p=node_projection(row);p['labels']=[v for v in [p['name'],*(p['aliases'] or [])] if isinstance(v,str)]
                    genes[key]=p
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata']
                    if key in selected:claims[key]=row
                    relevant=[side for side in ('subject','object') if str(md.get(side+'_id','')).startswith('CLM_CONCEPT:')]
                    if relevant:
                        claim_sha=digest(row);inner=md.get('metadata') or {}
                        for side in relevant:
                            incidences.append((md[side+'_id'],key,side,md.get(side+'_name'),claim_sha,md.get(side+'_type'),inner.get(side+'_type')))
                else:
                    labels={name_key(v).casefold() for v in [row.get('preferred_name'),*(row.get('aliases') or [])]}
                    matched=names&labels
                    if matched:
                        candidates[key]=scope_projection(row)
                        for value in matched:labels_by_name[value].add(key)
            elif kind=='edge' and edge_owner(row) in selected:refs[edge_owner(row)].append((int(key),row))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'full source SHA differs')
    require(set(claims)==selected and set(genes)==gene_ids,'claim/gene closure differs')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'count differs')
    reviews=[];reasons=Counter();roles=Counter()
    for endpoint in queue:
        row=claims[endpoint['claim_id']];w=genes[endpoint['current_node_id']]
        review=dict(endpoint,**gate(row,endpoint,w,refs[row['id']]),
            source_gene_node_sha256=w['node_sha256'],
            existing_name_or_alias_candidates=sorted(labels_by_name[name_key(endpoint['name']).casefold()]),
            owned_edges=[dict(ordinal=o,edge_sha256=digest(r)) for o,r in refs[row['id']]],
            no_conclusion_or_audit_revalidation=True)
        reasons.update(review['structural_holds'] or ['identity_candidate_needs_finite_semantic_allowlist'])
        roles.update(review['declared_roles'] or ['undeclared']);reviews.append(review)
    current_incidences=[dict(node_id=nid,claim_id=cid,side=side,name=name,claim_sha256=sha,outer_type=outer,inner_type=inside)
        for nid,cid,side,name,sha,outer,inside in incidences if nid in candidates]
    data={'CURRENT_ENDPOINT_REVIEWS.jsonl':reviews,'CURRENT_CLAIM_PROJECTIONS.jsonl':[projection(claims[k]) for k in sorted(claims)],
        'GENE_WITNESSES.jsonl':[genes[k] for k in sorted(genes)],'EXISTING_NAME_CANDIDATES.jsonl':[candidates[k] for k in sorted(candidates)],
        'EXISTING_COMPLETE_CONCEPT_INCIDENCES.jsonl':current_incidences}
    for name,records in data.items():write_rows(OUTPUT/name,records)
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    result=dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        pending_identity_plan=pending_fp,queue=pending['remaining_queue'],code=code,counts=dict(counts),
        full_source_sha_verified=True,reviewed_endpoints=len(reviews),selected_claims=len(claims),source_genes=len(genes),
        existing_candidate_nodes=len(candidates),existing_complete_concept_incidences=len(current_incidences),
        closure_reasons=dict(reasons),declared_role_counts=dict(roles),names_without_existing_match=sum(not r['existing_name_or_alias_candidates'] for r in reviews),
        pending_changed_claim_overlap=sorted(selected&{r['claim_id'] for r in pending['events']}),
        graph_modified=False,record_preimages_saved=False,
        artifacts={name:j.fingerprint(OUTPUT/name) for name in data})
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',result)
    progress('COMPLETED',endpoints=len(reviews),genes=len(genes),existing_nodes=len(candidates),roles=dict(roles),reasons=dict(reasons))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='FAILED',at=j.utc_now(),pid=os.getpid(),error=repr(error),graph_modified=False));raise
