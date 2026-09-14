"""R39 current exact clinical references, keeping source anchors and detail rows."""
from collections import defaultdict
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from plan_kg_literal_endpoint_repair import expected_shared
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_literal_endpoint_repair import reviewed_edges,edge_owner,GENES
from neurooracle.src.kg_clinical_literal_reuse import FAMILIES,validate_family,reviewed_claim
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
OUTPUT=j.OUTPUT/'round39_literal_duplicates'


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan already exists')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json'); inspection=j.read_json(OUTPUT/'SOURCE_INSPECTION.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None and inspection['graph']==c['current_graph'],'source advanced/writer active')
    for fp in inspection['artifacts'].values():require(j.fingerprint(fp['path'])==fp,'scope evidence changed')
    scope={r['node_id']:r for r in rows(OUTPUT/'NODE_SCOPE.jsonl')}
    node_ids={nid for f in FAMILIES.values() for nid in f['members']}
    held=rows(c['current_gene_holds']['path'])
    claim_ids={r['claim_id'] for nid in node_ids for r in scope[nid]['incidents']}|{
        r['claim_id'] for r in held if r['name'] in FAMILIES and r['reason']=='multiple_complete_name_candidates'}
    nodes,claims,refs={}, {},defaultdict(list)
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_clinical_literal_reuse.py')]
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='READ_ONLY_PLANNING',at=j.utc_now()))
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            if kind=='node':
                if key in node_ids:nodes[key]=row;require(digest(row)==scope[key]['node_sha256'],'node witness changed')
                if key in claim_ids:claims[key]=row
            elif kind=='edge' and edge_owner(row) in claim_ids:refs[edge_owner(row)].append((int(key),row))
        require(h.hexdigest()==c['current_graph']['sha256'],'source full SHA differs')
    require(set(nodes)==node_ids and set(claims)==claim_ids,'selected source incomplete')
    proofs=[]; aliases=rows(OUTPUT/'FULL_ALIAS_CANDIDATES.jsonl')
    for name,f in FAMILIES.items():
        proofs.append(validate_family(name,{n:nodes[n] for n in f['members']},{n:scope[n]['incidents'] for n in f['members']},
            [a for a in aliases if name_key(a['alias'])==name]))
    terms=VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path']))
    events=[];edge_events=[]
    for cid,row in sorted(claims.items()):
        changes=[];md=row['metadata']
        for side in ('subject','object'):
            name=name_key(md.get(side+'_name'))
            if name not in FAMILIES:continue
            target=FAMILIES[name]['target'];old=md[side+'_id']
            if old!=target:changes.append(dict(side=side,old_id=old,target_id=target,name=name))
        if not changes:continue
        event,out=reviewed_claim(row,changes)
        edge_events.extend(reviewed_edges(cid,row,out,refs[cid]))
        event.update(old_relation_id=relation_id(terms.relation_key(md)),new_relation_id=relation_id(terms.relation_key(out['metadata'])))
        events.append(event)
    fixed={(e['claim_id'],ch['side']) for e in events for ch in e['changes'] if ch['old_id'] in GENES}
    remaining=[r for r in held if (r['claim_id'],r['side']) not in fixed]
    require(len(events)==6 and len(fixed)==4 and len(remaining)==456,'reviewed clinical scope differs')
    census=j.read_json(c['current_paper_census']['path']); j.guards(census['database'])
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for ev in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(ev['claim_id'],)).fetchone()==(ev['claim_sha256'],ev['old_relation_id']),'census mismatch')
    shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']));db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    plan=dict(version='kg.clinical_literal_reuse.v1',status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],
        detail_store=c['current_detail_store'],source_acceptance=c['current_acceptance'],source_inspection=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json'),
        source_full_sha_verified=True,source_census_full_sha_verified=True,code=code,family_proofs=proofs,
        events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),new_literals=[],existing_targets={nid:digest(n) for nid,n in sorted(nodes.items())},
        expected_shared_claim_ids=shared,changed_claims=len(events),changed_endpoints=sum(len(e['changes']) for e in events),
        changed_edges=len(edge_events),added_literal_nodes=0,reused_existing_nodes=2,gene_endpoint_source_count=len(held),
        repaired_gene_endpoints=len(fixed),held_endpoints=len(remaining),remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),
        detail_rows_modified=False,source_anchor_nodes_retired=0,graph_backups=0,record_preimages_saved=False,
        boundaries=['complete literal expressions and explicit incident imaging type, no fuzzy or UMLS equivalence',
            'clinical claim references canonicalized; detail-backed source anchors kept, not physical node deletion',
            'no scientific text/negation/numeric/conditions/independent-source changes, no new fields or edges',
            'hippocampal combined-volume vs bilateral measures and mixed-closure claims remain unmodified'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'frozen source/code changed')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    j.atomic_json(OUTPUT/'PLAN_STATE.json',dict(status='REVIEWED_NOT_APPLIED',at=j.utc_now(),changed_claims=len(events),changed_edges=len(edge_events),held_endpoints=len(remaining)))
    print(compact(dict(claims=len(events),edges=len(edge_events),repaired_gene_endpoints=len(fixed),held_endpoints=len(remaining))),flush=True)


if __name__=='__main__':main()
