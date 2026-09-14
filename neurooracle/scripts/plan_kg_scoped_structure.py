"""R40 complete claim/edge plan; one added neutral about, no record preimages."""
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact,walk_graph,IncrementalJsonReader
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows
from reclaim_kg_backup_storage import sha256
from plan_kg_literal_endpoint_repair import expected_shared
from neurooracle.src import kg_scoped_structure as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence import name_key,relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round40_scoped_structure'


def public_proof():
    fetches=[j.OUTPUT/'round39_literal_duplicates/PUBLIC_SOURCE_FETCH.json',OUTPUT/'PUBLIC_SOURCE_FETCH.json',OUTPUT/'MRI_SOURCE_FETCH.json']
    contents=[]
    for path in fetches:
        f=j.read_json(path);fp=f['response']
        require(j.fingerprint(fp['path'])==fp,'public source changed')
        contents.append(Path(fp['path']).read_bytes())
    return repair.source_proof(contents)


def main():
    require(not (OUTPUT/'PLAN.json').exists(),'plan already exists; inspect/resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');inspection=j.read_json(OUTPUT/'SOURCE_INSPECTION.json');supplement=j.read_json(OUTPUT/'MRI_SOURCE_INSPECTION.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(inspection['graph']==supplement['graph']==c['current_graph'] and inspection['source_full_sha_verified'] and inspection['detail_full_sha_verified'],'source boundary differs')
    for fp in [*inspection['artifacts'].values(),inspection['code'],supplement['code'],supplement['scope'],supplement['public_fetch']]:
        require(j.fingerprint(fp['path'])==fp,'inspection witness changed')
    require(not supplement['existing_full_name_matches'] and not supplement['full_alias_matches'],'existing MRI target requires review')
    scope={r['claim_id']:r for p in (OUTPUT/'CLAIM_SCOPE.jsonl',OUTPUT/'MRI_CLAIM_SCOPE.jsonl') for r in rows(p)}
    selected=set(repair.MRI_SPECS)|{repair.COLLISION,repair.DIRECTION}
    require(selected<=set(scope),'missing claim scope')
    names={repair.ACTIVATION,repair.INTERACTION,*[v[1] for v in repair.MRI_SPECS.values()]}
    generated={repair.literal_node(n)['id'] for n in names}
    claims={};refs=defaultdict(list);counts=Counter()
    code=[j.fingerprint(p) for p in (Path(__file__),j.REPO/'neurooracle/src/kg_scoped_structure.py')]
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    print('R40_CURRENT_COMPLETE_SCOPE_PLAN',flush=True)
    # Current full source SHA was already verified this turn by the inspection;
    # the record/census hashes and before/after native guards bind this reread.
    with Path(c['current_graph']['path']).open('r',encoding='utf8') as handle:
        for kind,key,row in walk_graph(IncrementalJsonReader(handle)):
            counts[kind]+=1
            if kind=='node':
                require(key not in generated,'new literal ID collision')
                if key in selected:
                    require(digest(row)==scope[key]['claim_sha256'],'source claim changed');claims[key]=row
                if not key.startswith('CLM:'):
                    require(name_key(row.get('preferred_name')) not in names,'new same-name target requires review')
                    require(not any(name_key(a) in names for a in row.get('aliases') or []),'new full alias requires review')
            elif kind=='edge' and repair.edge_owner(row) in selected:refs[repair.edge_owner(row)].append((int(key),row))
    require(set(claims)==selected,'selected scope incomplete')
    terms=VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path']));proof=public_proof()
    events=[];edge_events=[];added=[];new={}
    for cid,row in sorted(claims.items()):
        require([dict(ordinal=i,edge_sha256=digest(e)) for i,e in refs[cid]]==scope[cid]['references'],'exact source ownership changed')
        md=row['metadata'];sides=('subject','object') if cid==repair.COLLISION else (() if cid==repair.DIRECTION else ('subject',))
        changes=[]
        for side in sides:
            name=name_key(md[side+'_name']);node=repair.literal_node(name)
            changes.append(dict(side=side,name=name,old_id=md[side+'_id'],target_id=node['id']))
            new[node['id']]=dict(id=node['id'],name=name,node_sha256=digest(node))
        event,out=repair.reviewed_claim(row,changes,proof)
        edge_events.extend(repair.reviewed_edges(cid,row,out,refs[cid]))
        for a in repair.reviewed_added_about(cid,row,out,refs[cid]):
            added.append(dict(ordinal=c['counts']['edges']+len(added)+1,row=a['row'],proof=a))
        event.update(old_relation_id=relation_id(terms.relation_key(md)),new_relation_id=relation_id(terms.relation_key(out['metadata'])))
        events.append(event)
    old_held=rows(c['current_gene_holds']['path']);fixed={(e['claim_id'],ch['side']) for e in events for ch in e['changes']}
    require(fixed<={(r['claim_id'],r['side']) for r in old_held},'changed gene scope not in held queue')
    remaining=[r for r in old_held if (r['claim_id'],r['side']) not in fixed]
    require(len(events)==7 and len(fixed)==7 and len(new)==7 and len(added)==1 and len(remaining)==449,'approved scope differs')
    census=j.read_json(c['current_paper_census']['path']);j.guards(census['database'])
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'source census SHA differs')
    db=sqlite3.connect(Path(census['database']['path']).as_uri()+'?mode=ro',uri=True)
    for e in events:require(db.execute('SELECT node_sha,relation_id FROM claims WHERE cid=?',(e['claim_id'],)).fetchone()==(e['claim_sha256'],e['old_relation_id']),'claim census mismatch')
    shared=expected_shared(db,{e['claim_id']:e for e in events},rows(c['current_shared_relations']['path']));db.close()
    write_rows(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl',remaining)
    plan=dict(version=repair.VERSION,status='REVIEWED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        detail_store=c['current_detail_store'],source_inspection=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json'),
        mri_source_inspection=j.fingerprint(OUTPUT/'MRI_SOURCE_INSPECTION.json'),source_full_sha_verified=True,
        reused_same_turn_source_sha_with_native_guards=True,source_census_full_sha_verified=True,
        code=code,public_proof=proof,events=events,edge_events=sorted(edge_events,key=lambda e:e['ordinal']),added_about_edges=added,
        new_literals=sorted(new.values(),key=lambda e:e['id']),existing_targets={},expected_shared_claim_ids=shared,
        changed_claims=len(events),changed_endpoints=len(fixed),changed_edges=len(edge_events),added_literal_nodes=len(new),reused_existing_nodes=0,
        repaired_gene_endpoints=len(fixed),direction_repaired_claims=1,gene_endpoint_source_count=len(old_held),held_endpoints=len(remaining),
        remaining_queue=j.fingerprint(OUTPUT/'REMAINING_GENE_ENDPOINTS.jsonl'),record_preimages_saved=False,graph_backups=0,
        retained_review_notes=['qMRI source describes episode count; current symptom-severity endpoint needs separate scientific review',
            'three same-name outcome-typed targets remain held; ACC incident quotation is incomplete',
            'BACE1 direction repair does not validate the broader pathway endpoint or other legacy scope metadata',
            'two R39 source anchors remain detail-backed; no physical node deletion'],
        boundaries=['only seven approved current claims and agreeing references; one added neutral about, no new scientific assertion',
            'complete names, laterality, composite expressions, negation, values, conditions and independent sources preserved except one proved direction',
            'no metadata fields added, model/training/formal writes, automation recreation or closed-night ledger changes'])
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(fp['path']) for fp in code]==code,'frozen plan source/code changed')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    j.atomic_json(OUTPUT/'PLAN.json',plan)
    print('R40_PLAN_REVIEWED',dict(claims=len(events),endpoints=len(fixed),edges=len(edge_events),added_about=len(added),new_literals=len(new),held=len(remaining)),flush=True)


if __name__=='__main__':main()
