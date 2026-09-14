"""Freeze a batch of full measurement identities, not speculative synonyms."""
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import IncrementalJsonReader,compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from plan_kg_literal_endpoint_repair import expected_shared
from neurooracle.src import kg_gene_boundary_repair as repair
from neurooracle.src.kg_literal_endpoint_repair import reuse_gate
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.claim_semantics import declared_type_atoms
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round43_gene_boundary'
REVIEW=j.OUTPUT/'round42_source_scope'


def progress(phase,**values):
    state=dict(status='READ_ONLY_PLANNING',at=j.utc_now(),phase=phase,**values)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',state);print(compact(state),flush=True)


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'],'source boundary')
    census=j.read_json(c['current_paper_census']['path'])
    review=j.read_json(REVIEW/'ALL_GENE_ENDPOINT_CENSUS.json')
    require(review['graph']==c['current_graph'] and review['not_a_confirmed_error_count'],'source review advanced')
    for fp in [review['code'],review['source_inspection'],*review['artifacts'].values()]:require(j.fingerprint(fp['path'])==fp,'review proof changed')
    for fp in j.read_json(c['current_acceptance']['path'])['code']:require(j.fingerprint(fp['path'])==fp,'current runtime changed')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    OUTPUT.mkdir(exist_ok=True)
    witnesses={r['node_id']:r for r in rows(REVIEW/'CURRENT_GENE_NODE_WITNESSES.jsonl')}
    endpoints=rows(REVIEW/'ALL_GENE_ENDPOINTS.jsonl')
    protected={r['claim_id'] for f in ('current_issues','current_structure_holds') for r in rows(c[f]['path'])}
    protected.update(j.read_json(c['current_scope_findings']['path'])['current_claim_hashes'])
    proposals={};declined={}
    old_queue={(r['claim_id'],r['side']):r for r in rows(c['current_gene_holds']['path'])}
    combined={}
    for r in endpoints:
        k=(r['claim_id'],r['side'])
        if k in old_queue:combined[k]=dict(old_queue[k])
        elif r['category'] in {'word_interior_alias_only','no_label_alignment'}:
            combined[k]=dict(claim_id=r['claim_id'],claim_sha256=r['claim_sha256'],side=r['side'],name=name_key(r['name']),
                current_node_id=r['current_node_id'],reason=r['category'],candidate_ids=[],alias_candidate_ids=[],
                review_origin='R42_all_canonical_gene_endpoint_census',not_a_confirmed_error=True)
        if r['category']!='word_interior_alias_only':continue
        md={r['side']+'_id':r['current_node_id'],r['side']+'_name':r['name'],r['side']+'_type':r['declared_roles']}
        reason=repair.endpoint_gate(md,r['side'],witnesses[r['current_node_id']])
        if r['claim_id'] in protected:reason='existing_scientific_or_structure_hold'
        if reason:declined[k]=reason
        else:proposals[k]=r
    require(len(combined)==len(old_queue)+review['outside_existing_queue_lexical_review_candidates'],'unified review queue differs')
    selected_ids={cid for cid,_ in proposals};names={name_key(r['name']) for r in proposals.values()}
    genes={r['current_node_id'] for r in proposals.values()}
    generated={repair.literal_node(n)['id'] for n in names}
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_gene_boundary_repair.py')]
    claims={};nodes={};matches=defaultdict(set);aliases=defaultdict(set);refs=defaultdict(list);seen_genes=set();counts=Counter()
    progress('FULL_SOURCE_EXACT_NAMES_ALIASES_AND_OWNERS',proposed_endpoints=len(proposals),review_endpoints=len(combined))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node':
                if key in genes:
                    require(digest(row)==witnesses[key]['node_sha256'],'source gene identity changed');seen_genes.add(key)
                if key in selected_ids:
                    claims[key]=row
                    for side in ('subject','object'):
                        if (key,side) in proposals:require(digest(row)==proposals[key,side]['claim_sha256'],'source claim changed')
                if not key.startswith('CLM:'):
                    name=name_key(row.get('preferred_name'))
                    if name in names:matches[name].add(key);nodes[key]=row
                    for alias in row.get('aliases') or []:
                        n=name_key(alias)
                        if n in names:aliases[n].add(key);nodes[key]=row
                    if key in generated:
                        require(name in names and row==repair.literal_node(name),'generated ID collides with a different record')
            elif kind=='edge' and repair.edge_owner(row) in selected_ids:
                refs[repair.edge_owner(row)].append((int(key),row))
            if counts[kind]%1000000==0:progress(kind,records=counts[kind])
        require(h.hexdigest()==c['current_graph']['sha256'],'source full SHA differs')
    require(set(claims)==selected_ids and seen_genes==genes,'source scope incomplete')
    incidents=defaultdict(list)
    progress('ALL_EXISTING_TARGET_CLAIM_INCIDENCES',existing_nodes=len(nodes))
    with Path(c['current_graph']['path']).open(encoding='utf8') as handle:
        for kind,key,row in walk_graph(IncrementalJsonReader(handle)):
            if kind=='edge':break
            if kind!='node' or not key.startswith('CLM:'):continue
            md=row['metadata'];inner=md.get('metadata') or {}
            for side in ('subject','object'):
                if md.get(side+'_id') in nodes:
                    incidents[md[side+'_id']].append(dict(claim_id=key,name=md.get(side+'_name'),
                        declared_roles=sorted(a.value for a in declared_type_atoms(md.get(side+'_type') or inner.get(side+'_type')))))
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    events=[];edge_events=[];new={};reused={};fixed=set();actual_declines=Counter()
    for cid,row in sorted(claims.items()):
        changes=[];local_new={};local_reused={}
        for side in ('subject','object'):
            k=(cid,side)
            if k not in proposals:continue
            r=proposals[k];name=name_key(r['name']);reason=repair.endpoint_gate(row['metadata'],side,witnesses[r['current_node_id']])
            target=None
            if not reason:
                choices=matches[name]|aliases[name]
                if not choices:
                    target=repair.literal_node(name);local_new[target['id']]=dict(id=target['id'],name=name,node_sha256=digest(target))
                elif len(choices)!=1:reason='ambiguous_existing_complete_name_or_alias'
                else:
                    nid=next(iter(choices));reason=reuse_gate(nodes[nid],name,incidents[nid])
                    if not reason:target=nodes[nid];local_reused[nid]=digest(target)
            if reason:declined[k]=reason;actual_declines[reason]+=1
            else:changes.append(dict(side=side,name=name,old_id=r['current_node_id'],target_id=target['id']))
        if not changes:continue
        try:
            event,out=repair.reviewed_claim(row,changes,witnesses)
            own_edges=repair.reviewed_edges(cid,row,out,refs[cid])
        except ValueError as exc:
            for ch in changes:declined[cid,ch['side']]='reference_or_identity_closure:'+str(exc)
            actual_declines['reference_or_identity_closure:'+str(exc)]+=len(changes)
            continue
        event.update(old_relation_id=relation_id(terms.relation_key(row['metadata'])),new_relation_id=relation_id(terms.relation_key(out['metadata'])))
        events.append(event);edge_events.extend(own_edges)
        used={ch['target_id'] for ch in changes}
        new.update({nid:n for nid,n in local_new.items() if nid in used});reused.update({nid:sha for nid,sha in local_reused.items() if nid in used})
        fixed.update((cid,ch['side']) for ch in changes)
    require(fixed<=set(combined) and events,'no eligible batch or wrong candidate scope')
    remaining=[]
    for k,r in sorted(combined.items()):
        if k in fixed:continue
        if k in declined:r=dict(r,latest_review_reason=declined[k])
        remaining.append(r)
    j.guards(census['database']);require(sha256(Path(census['database']['path']))==census['database']['sha256'],'source census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for e in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(e['claim_id'],)).fetchone()==(e['claim_sha256'],e['old_relation_id']),'current relation census mismatch')
    shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']));db.close()
    used_genes={ch['old_id'] for e in events for ch in e['changes']}
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],source_review=j.fingerprint(REVIEW/'ALL_GENE_ENDPOINT_CENSUS.json'),source_full_sha_verified=True,
        source_census_full_sha_verified=True,code=code,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),
        new_literals=sorted(new.values(),key=lambda r:r['id']),existing_targets={**reused,**{g:witnesses[g]['node_sha256'] for g in used_genes}},
        gene_witnesses={g:witnesses[g] for g in sorted(used_genes)},expected_shared_claim_ids=shared,relation_grouping=POLICY,
        changed_claims=len(events),changed_endpoints=len(fixed),changed_edges=len(edge_events),added_literal_nodes=len(new),reused_existing_nodes=len(reused),
        old_watchlist_endpoints=len(old_queue),expanded_review_endpoints=len(combined),newly_registered_lexical_candidates=review['outside_existing_queue_lexical_review_candidates'],
        held_endpoints=len(remaining),preflight_candidates=len(proposals),preflight_declines=dict(actual_declines),
        remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),protected_scientific_claims=sorted(protected),
        record_preimages_saved=False,graph_backups=0,
        boundaries=['Only explicit concrete imaging mentions matched to gene aliases inside ordinary words.',
            'Complete names and all nonidentity science, quotes, types, source identities, negation, conditions and audits preserved.',
            'No gene/protein node deletion, no merged scientific claims, no model/training/formal writes.',
            'Additional lexical candidates are not confirmed errors; existing five scientific holds excluded.'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    progress('PLAN_COMPLETE_NOT_APPLIED',claims=len(events),endpoints=len(fixed),new_nodes=len(new),reused_nodes=len(reused),remaining=len(remaining),declines=dict(actual_declines))


if __name__=='__main__':main()
