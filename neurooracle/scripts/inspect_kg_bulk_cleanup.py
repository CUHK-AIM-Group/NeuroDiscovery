"""R41 read-only whole-graph cleanup and reversed-correlation census."""
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from build_umls_simplification_candidate import compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_metadata_compaction import typed_equal
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = j.OUTPUT / 'round41_bulk_cleanup'
HINTS = ('subject_canonical_hint','object_canonical_hint','subject_atlas','object_atlas')
AUDIT_ALIASES = dict(scope_confidence='confidence', scope_decision_basis='decision_basis',
    scope_rubric_version='rubric_version', scope_review_status='review_status', scope_assignment_stage='review_stage')


def progress(phase, **values):
    state = dict(status='READ_ONLY_RUNNING', at=j.utc_now(), phase=phase, **values)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json', state)
    print(compact(state), flush=True)


def main():
    require(not (OUTPUT/'SOURCE_INSPECTION.json').exists(), 'inspection already exists')
    c = j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='COMPLETED' and c['active_process'] is None and not c['rollback_retention'], 'source boundary')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    receipt=j.read_json(c['current_acceptance']['path'])
    for fp in [c['current_acceptance'],c['current_runtime_acceptance'],*receipt['code']]:
        require(j.fingerprint(fp['path'])==fp, 'current code/evidence differs: '+fp['path'])
    OUTPUT.mkdir(exist_ok=True)
    j.atomic_json(OUTPUT/'SOURCE_BOUNDARY.json',dict(at=j.utc_now(),campaign=c,code=receipt['code'],
        no_code_or_graph_changes=True,record_preimages_saved=False))
    terms=VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path']))
    scopes=j.read_json(c['current_scope_findings']['path'])
    queue=rows(c['current_gene_holds']['path'])
    watched=set(scopes['current_claim_hashes']) | {r['claim_id'] for r in queue}
    selected={}; counts=Counter(); cleanup=Counter(); alias_conflicts=Counter(); changed=0
    db=sqlite3.connect(':memory:');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE corr(k TEXT,cid TEXT,orientation INT,paper TEXT)')
    buffer=[]; examples=defaultdict(list); code=j.fingerprint(Path(__file__))
    progress('FULL_CURRENT_SOURCE')
    with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
        for kind,key,row in walk_graph(reader):
            counts[kind]+=1
            if kind=='node' and key.startswith('CLM:'):
                md=row['metadata']; inner=md.get('metadata') or {}; audit=md.get('scope_reaudit') or {}
                counts['claims']+=1
                if key in watched:selected[key]=row
                record_changes=0
                for field in HINTS:
                    if field in inner and type(inner[field]) is str and inner[field]=='':
                        cleanup['empty_hint.'+field]+=1;record_changes+=1
                if 'raw_stats' in inner and type(inner['raw_stats']) is dict and not inner['raw_stats']:
                    cleanup['empty_raw_stats']+=1;record_changes+=1
                for field in ('subject_type','object_type','conditions','population'):
                    if field in inner and field in md and typed_equal(inner[field],md[field]):
                        cleanup['equal_science_alias.'+field]+=1;record_changes+=1
                for field,target in AUDIT_ALIASES.items():
                    if field in inner and target in audit:
                        if typed_equal(inner[field],audit[target]):
                            cleanup['equal_audit_alias.'+field]+=1
                        else:alias_conflicts[field]+=1
                changed+=record_changes>0
                if md.get('predicate')=='correlates_with':
                    k=terms.relation_key(md);a,b=k[:2],k[3:]
                    canonical=(a[0],a[1],k[2],b[0],b[1]) if a<=b else (b[0],b[1],k[2],a[0],a[1])
                    from neurooracle.src.case_study_membership_contract import strongest_paper_key
                    buffer.append((compact(canonical),key,int(a>b),strongest_paper_key(md)))
                    if len(buffer)>=5000:db.executemany('INSERT INTO corr VALUES (?,?,?,?)',buffer);buffer.clear()
            if counts[kind] % 500000==0:progress(kind,counts=dict(counts),changed_claim_candidates=changed)
        require(h.hexdigest()==c['current_graph']['sha256'],'full current graph SHA differs')
    db.executemany('INSERT INTO corr VALUES (?,?,?,?)',buffer)
    db.execute('CREATE INDEX corr_key ON corr(k)')
    merged=[]
    for k,n,papers in db.execute('SELECT k,COUNT(*),COUNT(DISTINCT paper) FROM corr GROUP BY k HAVING COUNT(DISTINCT orientation)>1 ORDER BY k'):
        members=[dict(claim_id=cid,original_orientation=o,source_key=p) for cid,o,p in db.execute('SELECT cid,orientation,paper FROM corr WHERE k=? ORDER BY cid',(k,))]
        merged.append(dict(key=k,claims=n,source_keys=papers,members=members))
    corr_count=db.execute('SELECT COUNT(*) FROM corr').fetchone()[0]
    db.close()
    require(set(selected)==watched,'selected claim coverage incomplete')
    for r in queue:require(digest(selected[r['claim_id']])==r['claim_sha256'],'queue hash differs')
    for cid,sha in scopes['current_claim_hashes'].items():require(digest(selected[cid])==sha,'scope hash differs')
    require(counts['node']==c['counts']['nodes'] and counts['edge']==c['counts']['edges'] and counts['claims']==c['counts']['claims'],'record census differs')
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and j.fingerprint(Path(__file__))==code,'source/code advanced')
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources']])
    result=dict(status='INSPECTED_NOT_APPLIED',at=j.utc_now(),graph=c['current_graph'],source_acceptance=c['current_acceptance'],
        full_source_sha_verified=True,code=code,counts=dict(counts),cleanup_candidates=dict(cleanup),
        changed_claim_candidates=changed,audit_alias_conflicts=dict(alias_conflicts),correlation_claims=corr_count,
        reversed_exact_correlation_groups=len(merged),reversed_correlation_members=sum(r['claims'] for r in merged),
        selected_claims=len(selected),graph_modified=False,record_preimages_saved=False)
    j.atomic_json(OUTPUT/'REVERSED_CORRELATION_GROUPS.json',dict(graph=c['current_graph'],groups=merged,
        interpretation='exact endpoints and complete names; grouping proposal, not fact or study merging'))
    j.atomic_json(OUTPUT/'SOURCE_INSPECTION.json',result)
    j.atomic_json(OUTPUT/'INSPECTION_STATE.json',dict(status='COMPLETED',at=j.utc_now()))
    print(compact(result),flush=True)
    print(compact(dict(scope_records=[selected[cid] for cid in scopes['current_claim_hashes']])),flush=True)


if __name__=='__main__':main()
