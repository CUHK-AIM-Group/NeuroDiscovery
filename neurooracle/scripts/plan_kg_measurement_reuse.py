"""R38 current, whole-name plan; source-bound network and numeric proof."""
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from plan_kg_literal_endpoint_repair import expected_shared
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import GENES,literal_node,reuse_gate,edge_owner,reviewed_edges
from neurooracle.src.kg_measurement_reuse import public_network_evidence,gate,reviewed_claim,NETWORKS,BOUNDS
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities

OUTPUT=journal.OUTPUT/'round38_measurement_reuse'


def public_proof():
    fetch=journal.read_json(OUTPUT/'PUBLIC_SOURCE_FETCH.json')
    for w in fetch['witnesses']: require(journal.fingerprint(w['response']['path'])==w['response'],'source witness changed')
    return public_network_evidence((OUTPUT/'pubmed_39829963.xml').read_bytes(),(OUTPUT/'PMC11740805.xml').read_bytes())


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',phase=phase,at=journal.utc_now(),**values)
    journal.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/reuse')
    c=journal.read_json(journal.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    inspection=journal.read_json(OUTPUT/'SOURCE_INSPECTION.json')
    require(inspection['graph']==c['current_graph'],'source inspection stale')
    for fp in inspection['artifacts'].values():require(journal.fingerprint(fp['path'])==fp,'inspection evidence changed')
    proof=public_proof();public_summary={k:v for k,v in proof.items() if k!='abstract'}
    code=[journal.fingerprint(p) for p in (Path(__file__),journal.REPO/'neurooracle/src/kg_measurement_reuse.py')]
    endpoints=rows(OUTPUT/'CURRENT_GENE_ENDPOINTS.jsonl');names={name_key(r['name']) for r in endpoints}
    claim_ids={r['claim_id'] for r in endpoints};summaries={r['node_id']:r for r in rows(OUTPUT/'CURRENT_NAME_CANDIDATES.jsonl')}
    created_ids={literal_node(n)['id'] for n in names};already=set()
    selected,candidates,refs,aliases={}, {},defaultdict(list),defaultdict(set);counts=Counter()
    journal.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    progress('FULL_SOURCE_CURRENT_PLAN_BOUNDARY',endpoints=len(endpoints))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in created_ids:already.add(key)
                if key in claim_ids:selected[key]=row
                if not key.startswith('CLM:'):
                    if name_key(row.get('preferred_name')) in names:
                        candidates[key]=row
                        require(key in summaries and digest(row)==summaries[key]['node_sha256'],'candidate changed')
                    if key not in GENES:
                        for alias in row.get('aliases') or []:
                            if name_key(alias) in names:aliases[name_key(alias)].add(key)
            elif kind=='edge' and edge_owner(row) in claim_ids:refs[edge_owner(row)].append((int(key),row))
            if counts[kind]%1000000==0:progress(kind,count=counts[kind])
        require(h.hexdigest()==c['current_graph']['sha256'],'current source full SHA differs')
    require(set(selected)==claim_ids,'current claim closure incomplete')
    by_name=defaultdict(list)
    for node in candidates.values():by_name[name_key(node['preferred_name'])].append(node)
    terms=VerifiedEntityTerms(journal.read_json(c['current_entity_terms']['path']))
    papers=VerifiedPaperIdentities(journal.read_json(c['current_paper_identities']['path']))
    events,edge_events,holds,new,targets=[],[],[],{},{}
    for cid,row in sorted(selected.items()):
        changes=[];pending=[]
        for side in ('subject','object'):
            md=row['metadata']
            if md.get(side+'_id') not in GENES:continue
            reason,basis=gate(row,side,proof);name=name_key(md.get(side+'_name'));options=by_name.get(name,[])
            if reason is None:
                if cid in NETWORKS:
                    require(papers.resolve(md)['paper_key']=='pmid:39829963' and papers.resolve(md)['status']=='verified','owning source not verified')
                if len(options)>1:reason='multiple_complete_name_candidates'
                elif options:
                    target=options[0];reason=reuse_gate(target,name,summaries[target['id']]['all_incident_claim_endpoints'])
                elif aliases[name]:reason='existing_full_alias_requires_identity_proof'
                else:
                    target=literal_node(name);require(target['id'] not in already,'new literal ID collision')
            if reason:
                holds.append(dict(claim_id=cid,claim_sha256=digest(row),side=side,name=name,current_node_id=md[side+'_id'],reason=reason,
                    candidate_ids=sorted(n['id'] for n in options),alias_candidate_ids=sorted(aliases[name])))
            else:
                changes.append(dict(side=side,old_id=md[side+'_id'],target_id=target['id'],name=name,basis=basis,
                    disposition='reuse_existing_exact_literal' if options else 'reuse_or_create_one_complete_literal'))
                pending.append(target)
        if not changes:continue
        event,current=reviewed_claim(row,changes,proof)
        try:
            closure=reviewed_edges(cid,row,current,refs[cid])
            if cid in BOUNDS:require(all(r['relation_type']=='about' for _,r in refs[cid]),'numeric value may also be materialized on science edge; separate review required')
        except ValueError as exc:
            holds.extend(dict(claim_id=cid,claim_sha256=digest(row),side=ch['side'],name=ch['name'],current_node_id=ch['old_id'],
                reason='reference_closure: '+str(exc),candidate_ids=[ch['target_id']]) for ch in changes)
            continue
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(current['metadata'])))
        events.append(event);edge_events.extend(closure)
        for node in pending:
            if node['id'] in candidates:targets[node['id']]=digest(node)
            else:new[node['id']]=dict(id=node['id'],name=node['preferred_name'],node_sha256=digest(node))
    require(set(NETWORKS)<={e['claim_id'] for e in events},'six source-reviewed network records not all repairable')
    census=journal.read_json(c['current_paper_census']['path']);journal.guards(census['database'])
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'source census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for ev in events:
        require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(ev['claim_id'],)).fetchone()==(ev['claim_sha256'],ev['old_relation_id']),'source census differs')
    shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']));db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',holds)
    plan=dict(version='kg.measurement_reuse.v1',status='REVIEWED_NOT_APPLIED',at=journal.utc_now(),graph=c['current_graph'],detail_store=c['current_detail_store'],
        source_acceptance=c['current_acceptance'],source_inspection=journal.fingerprint(OUTPUT/'SOURCE_INSPECTION.json'),
        source_full_sha_verified=True,source_census_full_sha_verified=True,source_census=census['database'],code=code,
        public_source_fetch=journal.fingerprint(OUTPUT/'PUBLIC_SOURCE_FETCH.json'),public_measurement_evidence=public_summary,
        events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),new_literals=sorted(new.values(),key=lambda e:e['id']),
        existing_targets=targets,expected_shared_claim_ids=shared,changed_claims=len(events),changed_endpoints=sum(len(e['changes']) for e in events),
        changed_edges=len(edge_events),added_literal_nodes=len(new),reused_existing_nodes=len(targets),held_endpoints=len(holds),
        held_reasons=dict(Counter(r['reason'] for r in holds)),science_changed_claims=sum(bool(e['science_changes']) for e in events),
        gene_endpoint_source_count=len(endpoints),remaining_queue=journal.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),
        graph_backups=0,record_preimages_saved=False,
        boundaries=['complete original names; lexical eligibility is not alias equivalence','six network mentions proven by own PubMed quote and PMC imaging methods',
            'only three bound-as-point errors and method labels corrected; raw quotes/negation/conditions/independent sources unchanged',
            'COP Table2 Pearson r and nominal p; other predictive records retain the bound, no univariate/regression substitution',
            'no added metadata keys, new scientific edges, models, training, formal sync or old KG copies'])
    require(plan['changed_endpoints']+len(holds)==len(endpoints),'endpoint scope differs')
    require(journal.read_json(journal.OUTPUT/'CAMPAIGN.json')==c and [journal.fingerprint(p['path']) for p in code]==code,'code/source advanced')
    journal.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    journal.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('REVIEWED_NOT_APPLIED',**{k:plan[k] for k in ('changed_claims','changed_endpoints','changed_edges','added_literal_nodes','reused_existing_nodes','held_endpoints','science_changed_claims','held_reasons')})


if __name__=='__main__':main()
