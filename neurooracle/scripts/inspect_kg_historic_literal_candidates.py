"""Read-only complete incidence/closure review for the remaining historic queue."""
from collections import Counter,defaultdict
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import small_check
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_gene_boundary_repair import word_interior_hits
from neurooracle.src.kg_literal_endpoint_repair import edge_owner,reviewed_edges
from neurooracle.src.claim_semantics import declared_type_atoms
from neurooracle.src.relation_evidence import name_key

OUTPUT=j.OUTPUT/'round47_historic_mentions'
REVIEW=j.OUTPUT/'round45_semantic_sources'
GENES=j.OUTPUT/'round42_source_scope/CURRENT_GENE_NODE_WITNESSES.jsonl'


def exact_label(value):return name_key(value)


def node_projection(row):
    return dict(node_id=row['id'],node_sha256=digest(row),name=row.get('preferred_name'),
        aliases=row.get('aliases'),semantic_types=row.get('semantic_types'),external_ids=row.get('external_ids'),
        domain_tags=row.get('domain_tags'),source_vocab=row.get('source_vocab'),
        metadata_keys=sorted((row.get('metadata') or {}).keys()),metadata_sha256=digest(row.get('metadata') or {}),
        complete_record_not_saved=True)


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',pid=os.getpid(),phase=phase,at=j.utc_now(),**values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(),'inspection exists; reuse bound evidence')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' or (c['status']=='MANUAL_ACTIVE' and c['active_process']['kind']=='research_statement_retirement'), 'unreviewed writer boundary')
    source_fp=j.fingerprint(REVIEW/'SOURCE_INSPECTION.json');source=j.read_json(source_fp['path'])
    require(source['graph']==c['current_graph'] and source['source_acceptance']==c['current_acceptance'],'R45 current source has advanced')
    for fp in [source['code'],*source['artifacts'].values()]:small_check(fp)
    watch=rows(source['artifacts']['CURRENT_HISTORIC_WATCHLIST.jsonl']['path'])
    require(len(watch)==447 and len({(r['claim_id'],r['side']) for r in watch})==447,'current historic watchlist differs')
    targets={r['node_id']:r for r in rows(source['artifacts']['CURRENT_COMPLETE_NAME_CANDIDATES.jsonl']['path']) if r['name']!='VCP'}
    selected={r['claim_id'] for r in watch};gene_ids={r['current_hold']['current_node_id'] for r in watch}
    gene_fp=j.fingerprint(GENES);genes={r['node_id']:r for r in rows(GENES) if r['node_id'] in gene_ids}
    require(set(genes)==gene_ids=={'CUI:C1414531','CUI:C1421437'},'two reviewed canonical genes differ')
    code=j.fingerprint(Path(__file__));j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    claims={};owned=defaultdict(list);incidents=[];nodes={};found_genes=set();counts=Counter()
    progress('FULL_CURRENT_SOURCE_EXISTING_LITERAL_INCIDENCES_AND_447_OWNED_CLOSURES')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in gene_ids:
                    require(digest(row)==genes[key]['node_sha256'],'canonical gene witness changed');found_genes.add(key)
                if key in targets:
                    require(digest(row)==targets[key]['node_sha256'],'existing complete-name target changed');nodes[key]=node_projection(row)
                if key.startswith('CLM:'):
                    counts['claims']+=1;md=row['metadata'];inner=md.get('metadata') or {}
                    if key in selected:claims[key]=row
                    for side in ('subject','object'):
                        if md.get(side+'_id') in targets:
                            paper=md.get('source_paper') or {}
                            incidents.append(dict(node_id=md[side+'_id'],claim_id=key,claim_sha256=digest(row),side=side,name=md.get(side+'_name'),
                                outer_type=md.get(side+'_type'),inner_type=inner.get(side+'_type'),
                                declared_roles=sorted(a.value for a in declared_type_atoms(md.get(side+'_type') or inner.get(side+'_type'))),
                                pmid=str(paper.get('pmid') or ''),doi=paper.get('doi'),raw_text_sha256=digest(md.get('raw_text')),
                                scope_is_not_a_new_scientific_validation=True))
            elif kind=='edge' and edge_owner(row) in selected:owned[edge_owner(row)].append((int(key),row))
            if counts[kind]%1000000==0:progress(kind,counts=dict(counts))
        require(h.hexdigest()==c['current_graph']['sha256'],'complete source SHA differs')
    require(set(claims)==selected and set(nodes)==set(targets) and found_genes==gene_ids,'selected source coverage incomplete')
    require((counts['node'],counts['claims'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'current census differs')
    adjudicated=[];reasons=Counter()
    for r in watch:
        cid=r['claim_id'];side=r['side'];row=claims[cid];md=row['metadata'];inner=md.get('metadata') or {};w=genes[r['current_hold']['current_node_id']]
        require(digest(row)==r['current_hold']['claim_sha256'] and md[side+'_name']==r['name'] and md[side+'_id']==w['node_id'],'current endpoint differs')
        hits=word_interior_hits(md[side+'_name'],w['labels']);gate=None
        if not hits:gate='not_word_interior_only_actual_vcp_organism_scope_review'
        if inner.get(side+'_id',md[side+'_id'])!=md[side+'_id']:gate='nested_id_conflict'
        outer_type,inner_type=md.get(side+'_type'),inner.get(side+'_type')
        if outer_type and inner_type and declared_type_atoms(outer_type)!=declared_type_atoms(inner_type):gate='nested_type_conflict'
        try:
            require(reviewed_edges(cid,row,row,owned[cid])==[],'unexpected identity-neutral edge change')
        except ValueError as error:gate='current_owned_closure_requires_review: '+str(error)
        matches=[nid for nid,n in nodes.items() if exact_label(md[side+'_name']) in {exact_label(n['name']),*[exact_label(v) for v in (n['aliases'] or [])]}]
        reasons[gate or 'identity_only_candidate_with_closed_owned_edges']+=1
        adjudicated.append(dict(claim_id=cid,claim_sha256=digest(row),side=side,name=md[side+'_name'],current_node_id=md[side+'_id'],
            source_gene_node_sha256=w['node_sha256'],word_interior_alias_hits=hits,current_gate=gate,
            identity_scope='complete_original_mention_not_single_VCP_or_FANCE_gene' if hits else 'actual_VCP_context_not_resolved',
            outer_type=outer_type,inner_type=inner_type,existing_complete_name_candidates=matches,
            owned_edges=[dict(ordinal=o,edge_sha256=digest(e)) for o,e in owned[cid]],
            nonidentity_science_and_old_audit_not_revalidated=True,graph_modified=False,
            literal_reuse_requires_all_incident_scope_review=True))
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced during read-only review')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    artifacts={'CURRENT_ENDPOINT_ADJUDICATION.jsonl':adjudicated,'EXISTING_NODE_WITNESSES.jsonl':list(nodes.values()),
        'EXISTING_INCIDENTS.jsonl':incidents,'GENE_WITNESSES.jsonl':list(genes.values())}
    for filename,records in artifacts.items():j.atomic_text(OUTPUT/filename,''.join(compact(r)+'\n' for r in records))
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        source_inspection=source_fp,source_gene_witness=gene_fp,code=code,counts=dict(counts),full_source_sha_verified=True,
        reviewed_endpoints=len(watch),existing_candidate_nodes=len(nodes),existing_incident_endpoints=len(incidents),closure_reasons=dict(reasons),
        graph_modified=False,record_preimages_saved=False,all_conclusions_and_audits_unmodified=True,
        artifacts={name:j.fingerprint(OUTPUT/name) for name in artifacts}))
    progress('COMPLETED',reviewed_endpoints=len(watch),existing_nodes=len(nodes),existing_incidents=len(incidents),closure_reasons=dict(reasons))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='FAILED',pid=os.getpid(),at=j.utc_now(),error=repr(error),graph_modified=False))
        raise
