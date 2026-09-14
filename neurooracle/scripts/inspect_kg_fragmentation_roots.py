"""R68: full current KG root-cause diagnostics; graph and runtime remain read-only."""
from collections import Counter
from pathlib import Path
import argparse
import json
import os
import sqlite3
import sys
import time
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
import inspect_kg_current_relation_reuse as old
from build_umls_simplification_candidate import compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.correlation_grouping import IndexTerms,enabled
from neurooracle.src.kg_fragmentation_diagnostics import PROBES,PROBE_ONLY,extra_keys,features,orthographic,key,explicit_barriers
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.metadata_field_audit import node_class
from neurooracle.src.shared_relation_catalog import current_shared_relations,find_shared_relations,check_file
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round68_fragmentation_root_census'
DATABASE=OUTPUT/'WORKING_PROJECTIONS.sqlite'
EXTRAS=tuple(p for p in PROBES if p not in old.GROUPINGS)
FILES=[Path(__file__),j.REPO/'neurooracle/src/kg_fragmentation_diagnostics.py',
       j.REPO/'neurooracle/tests/test_kg_fragmentation_diagnostics.py',*old.FILES,
       j.REPO/'neurooracle/scripts/build_umls_simplification_candidate.py']


def progress(phase,**values):
    state=dict(status='READ_ONLY_RUNNING',at=j.utc_now(),pid=os.getpid(),phase=phase,**values)
    j.atomic_json(OUTPUT/'RUN_STATE.json',state);print(compact(state),flush=True)


def initialize(db):
    old.initialize(db)
    for field in EXTRAS: db.execute(f'ALTER TABLE claims ADD COLUMN {field} TEXT')
    for field in ('st','tt','sf','tf'):db.execute(f'ALTER TABLE claims ADD COLUMN {field} TEXT')
    db.executescript('''
        CREATE TABLE nodes(nid TEXT PRIMARY KEY,name TEXT,kind TEXT,node_sha TEXT,domains TEXT);
        CREATE TABLE aliases(label TEXT,nid TEXT,origin TEXT);
    ''')


def append_claims(db,batch):
    if batch:db.executemany('INSERT INTO claims VALUES ('+','.join('?' for _ in batch[0])+')',batch)


def group_statistics(db,field):
    require(field in PROBES,'unknown diagnostic probe')
    db.execute(f'''CREATE TABLE groups_{field} AS SELECT {field} k,COUNT(*) n,
        COUNT(DISTINCT pk) source_keys,
        COUNT(DISTINCT CASE WHEN status='verified' THEN pk END) verified_papers,
        COUNT(DISTINCT CASE WHEN countable=1 THEN pk END) counted_papers,
        COUNT(DISTINCT rid) fine_variants,COUNT(DISTINCT surface) exact_surface_variants,
        COUNT(DISTINCT physical) physical_variants,COUNT(DISTINCT neg) negation_variants,
        COUNT(DISTINCT ctx) evidence_variants FROM claims GROUP BY {field}''')
    db.execute(f'CREATE UNIQUE INDEX groups_{field}_k ON groups_{field}(k)')
    return dict(db.execute(f'''SELECT COUNT(*) groups,SUM(n) claims,
        SUM(n=1) singleton_groups,SUM(n>1) shared_groups,
        SUM(CASE WHEN n>1 THEN n ELSE 0 END) shared_claims,
        SUM(source_keys>1) multi_source_key_groups,
        SUM(verified_papers>1) multi_verified_paper_groups,
        SUM(counted_papers>1) multi_verified_not_retracted_groups,
        SUM(fine_variants>1) multiple_fine_relation_groups,
        SUM(CASE WHEN fine_variants>1 THEN n ELSE 0 END) claims_in_multiple_fine_relation_groups,
        SUM(exact_surface_variants>1) multiple_exact_surface_groups,
        SUM(negation_variants>1) mixed_negation_groups,
        SUM(evidence_variants>1) multiple_evidence_context_groups,
        MAX(n) largest_group,MAX(counted_papers) most_counted_papers,
        SUM(fine_variants>1 AND source_keys>1) cross_source_fragment_candidates,
        SUM(CASE WHEN fine_variants>1 AND source_keys>1 THEN n ELSE 0 END) cross_source_candidate_claim_incidents,
        SUM(fine_variants>1 AND counted_papers>1) cross_verified_fragment_candidates
        FROM groups_{field}''').fetchone())


def scan():
    require(not (OUTPUT/'SCAN.json').exists() and not DATABASE.exists(),'existing scan/working index; inspect before resume')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');require(c['status']=='COMPLETED' and c['active_process'] is None,'writer active')
    require(Path(c['current_acceptance']['path']).parent.name=='round67_relation_semantic_consolidation','requires current R67')
    for field in ('current_acceptance','current_runtime_acceptance','current_paper_census','current_entity_terms','current_paper_identities','current_manual_summary','current_report_validation'):
        check_file(c[field],full_hash=True)
    a=j.read_json(c['current_acceptance']['path']);census=j.read_json(c['current_paper_census']['path'])
    require(a['graph']==c['current_graph']==census['graph'] and enabled(a),'current identity mismatch')
    tests=ET.parse(OUTPUT/'TEST_RESULTS.xml').getroot().find('testsuite')
    require(int(tests.get('tests'))>=20 and not any(int(tests.get(k,0)) for k in ('failures','errors','skipped')),'tests not complete')
    code=[j.fingerprint(p) for p in FILES]
    j.atomic_json(OUTPUT/'BASELINE.json',dict(at=j.utc_now(),campaign=c,code=code,tests=j.fingerprint(OUTPUT/'TEST_RESULTS.xml')))
    j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
    progress('VERIFY_CURRENT_CENSUS_SHA')
    require(sha256(Path(census['database']['path']))==census['database']['sha256'],'current census SHA differs')
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(c['current_entity_terms']['path'])))
    papers=VerifiedPaperIdentities(j.read_json(c['current_paper_identities']['path']))
    db=sqlite3.connect(DATABASE);initialize(db)
    counts,classes,edge_kinds,predicates,feature_counts,role_counts,route_counts= (Counter() for _ in range(7))
    lengths=Counter();node_union=set();edge_union=set();batch=[];nodes=[];aliases=[]
    def flush():
        append_claims(db,batch);batch.clear()
        db.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)',nodes);nodes.clear()
        db.executemany('INSERT INTO aliases VALUES (?,?,?)',aliases);aliases.clear();db.commit()
    try:
        progress('FULL_GRAPH_PROJECTION')
        with hashed_reader(Path(c['current_graph']['path'])) as (reader,h):
            for kind,nid,row in walk_graph(reader):
                counts[kind]+=1
                if kind=='metadata':require(enabled(row),'graph policy differs');continue
                md=row.get('metadata') or {}
                if kind=='node':
                    category=node_class(nid,row);classes[category]+=1;node_union.update(md)
                    if category=='claim':
                        base=old.claim_projection(row,terms,papers);extra,routed=extra_keys(md,terms)
                        inner=md.get('metadata') or {};st=md.get('subject_type') or inner.get('subject_type');tt=md.get('object_type') or inner.get('object_type')
                        sf,tf=features(md['subject_name']),features(md['object_name'])
                        batch.append((*base,*[extra.get(k,'') for k in EXTRAS],compact(st),compact(tt),compact(sf),compact(tf)))
                        predicates[md['predicate']]+=1;route_counts[len(routed)]+=1
                        route_counts['changed_key']+=int(extra['registry']!=base[3])
                        for side,ft,role in (('s',sf,st),('t',tf,tt)):
                            role_counts[side+':'+str(role)]+=1;lengths[ft['token_count']]+=1
                            for name,value in ft.items():
                                if name!='token_count' and value:feature_counts[side+':'+name]+=1
                    else:
                        nodes.append((nid,row.get('preferred_name') or '',category,digest(row),compact(row.get('domain_tags') or [])))
                        # Existing ontology labels are only SEARCH evidence. A
                        # unique alias hit is not scientific identity approval.
                        if category in {'new_umls_cui','existing_entity_or_infrastructure'}:
                            for label in {row.get('preferred_name') or '',*(row.get('aliases') or [])}:
                                if label:aliases.append((orthographic(label),nid,row.get('source_vocab') or ''))
                else:
                    edge_union.update(md)
                    edge_kinds['about' if row['relation_type']=='about' else 'claim_science' if md.get('claim_id') else 'other']+=1
                if counts[kind]%10000==0:flush()
                if counts[kind]%250000==0:progress('SCAN_'+kind.upper(),counts=dict(counts),claims=classes['claim'])
            require(h.hexdigest()==c['current_graph']['sha256'],'full current graph SHA differs')
        flush()
        require((counts['node'],classes['claim'],counts['edge'])==tuple(c['counts'][k] for k in ('nodes','claims','edges')),'full counts differ')
        require((len(node_union),len(edge_union))==(c['node_metadata_field_union'],c['edge_metadata_field_union']),'metadata changed')
        db.execute('ATTACH DATABASE ? AS frozen',(Path(census['database']['path']).as_uri()+'?mode=ro',))
        progress('ALL_CLAIM_HASHES_PAPERS_RELATIONS_VS_CURRENT_CENSUS')
        bad=db.execute('''SELECT COUNT(*) FROM claims c LEFT JOIN frozen.claims f ON c.cid=f.cid
            WHERE f.cid IS NULL OR c.node_sha<>f.node_sha OR c.paper_sig<>f.paper_sig OR c.rid<>f.relation_id''').fetchone()[0]
        require(bad==0 and db.execute('SELECT COUNT(*) FROM frozen.claims').fetchone()[0]==classes['claim'],'census closure differs')
        require(db.execute('''SELECT COUNT(*) FROM claims c LEFT JOIN nodes s ON s.nid=c.sid LEFT JOIN nodes t ON t.nid=c.tid
            WHERE s.nid IS NULL OR t.nid IS NULL''').fetchone()[0]==0,'missing endpoint node')
        db.execute('DETACH DATABASE frozen')
        progress('COMPLETE_LABEL_ALIAS_ROUTE_DIAGNOSTIC')
        db.executescript('''CREATE TABLE alias_unique AS SELECT label,MIN(nid) nid FROM aliases GROUP BY label HAVING COUNT(DISTINCT nid)=1;
            CREATE UNIQUE INDEX alias_unique_label ON alias_unique(label);''')
        lexicon={r['label']:r['nid'] for r in db.execute('SELECT * FROM alias_unique')};updates=[];alias_counts=Counter()
        for row in db.execute('SELECT cid,sn,tn,p,sid,tid FROM claims'):
            sn,tn=orthographic(row['sn']),orthographic(row['tn'])
            s='id:'+lexicon[sn] if sn in lexicon else 'text:'+sn
            t='id:'+lexicon[tn] if tn in lexicon else 'text:'+tn
            alias_counts['endpoint_hits']+=int(sn in lexicon)+int(tn in lexicon)
            alias_counts['claims_with_hit']+=int(sn in lexicon or tn in lexicon)
            updates.append((key(s,row['p'],t,symmetric=row['p']=='correlates_with'),row['cid']))
        db.executemany('UPDATE claims SET alias_route=? WHERE cid=?',updates);del updates,lexicon;db.commit()
        require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','working projection corrupt')
        require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(p) for p in FILES]==code,'state/code changed')
        j.guards([c['current_graph'],c['current_detail_store'],c['formal_sources'],census['database']])
        result=dict(status='FULL_CURRENT_SCAN_COMPLETE',at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
            counts=dict(counts),node_classes=dict(classes),edge_kinds=dict(edge_kinds),predicates=dict(predicates),
            endpoint_feature_counts=dict(feature_counts),endpoint_role_counts=dict(role_counts),endpoint_token_lengths=old.distribution(lengths),
            verified_registry_routes=dict(route_counts),unverified_alias_search=dict(alias_counts),
            source_status_counts=old.all_rows(db,'SELECT status,COUNT(*) claims,COUNT(DISTINCT pk) source_keys FROM claims GROUP BY status'),
            metadata_unions=dict(nodes=len(node_union),edges=len(edge_union)),checked_claims=classes['claim'],
            full_graph_sha_verified=True,full_current_census_sha_verified=True,graph_mutations=0,model_calls=0,
            aliases_are_merge_authority=False,record_preimages_saved=False,code=code)
        j.atomic_json(OUTPUT/'SCAN.json',result)
    finally:db.close()
    progress('SCAN_COMPLETE_ANALYSIS_PENDING')


def analyze():
    require((OUTPUT/'SCAN.json').exists() and not (OUTPUT/'ANALYSIS.json').exists(),'analysis boundary')
    baseline=j.read_json(OUTPUT/'BASELINE.json');c=baseline['campaign'];scan_result=j.read_json(OUTPUT/'SCAN.json')
    require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c,'graph campaign advanced')
    require([j.fingerprint(p) for p in FILES]==baseline['code'],'scan code changed')
    db=sqlite3.connect(DATABASE);db.row_factory=sqlite3.Row
    phase='ANALYZE';last=time.monotonic()
    def pulse():
        nonlocal last
        if time.monotonic()-last>25:progress(phase);last=time.monotonic()
        return 0
    db.set_progress_handler(pulse,200000)
    try:
        stats={};candidates=[];examples=[]
        db.executescript('CREATE TABLE opportunities(rid TEXT,probe TEXT,k TEXT);CREATE TABLE candidate_claims(cid TEXT PRIMARY KEY);')
        for field in PROBES:
            phase='GROUP_'+field;progress(phase)
            db.execute(f'CREATE INDEX claims_{field} ON claims({field})')
            stats[field]=group_statistics(db,field)
            if field=='rid':continue
            db.execute(f'''INSERT INTO opportunities SELECT DISTINCT c.rid,?,g.k FROM claims c JOIN groups_{field} g ON c.{field}=g.k
                WHERE g.fine_variants>1 AND g.source_keys>1''',(field,))
            db.execute(f'''INSERT OR IGNORE INTO candidate_claims SELECT c.cid FROM claims c JOIN groups_{field} g ON c.{field}=g.k
                WHERE g.fine_variants>1 AND g.source_keys>1''')
            for group in db.execute(f'SELECT * FROM groups_{field} WHERE fine_variants>1 AND source_keys>1 ORDER BY counted_papers DESC,source_keys DESC,n DESC,k'):
                candidates.append(dict(probe=field,**dict(group),status=PROBE_ONLY))
            for group in db.execute(f'SELECT * FROM groups_{field} WHERE fine_variants>1 AND source_keys>1 ORDER BY counted_papers DESC,source_keys DESC,n DESC,k LIMIT 5'):
                members=old.all_rows(db,f'''SELECT cid,node_sha,rid,sid,sn,p,tid,tn,pk,status,countable,neg,ctx,st,tt,sf,tf
                    FROM claims WHERE {field}=? ORDER BY countable DESC,pk,cid LIMIT 6''',(group['k'],))
                barriers=Counter(b for r in members[1:] for b in explicit_barriers(members[0],r))
                examples.append(dict(probe=field,group=dict(group),members=members,observed_barriers=dict(barriers),
                    barriers_are_sampled_not_all_group_members=True,status=PROBE_ONLY))
            db.commit()
        phase='DISJOINT_SINGLE_SOURCE_ATTRIBUTION';progress(phase)
        shared=list(current_shared_relations(j.OUTPUT/'CAMPAIGN.json'))
        default=list(find_shared_relations(j.OUTPUT/'CAMPAIGN.json',minimum_papers=2))
        inclusive=list(find_shared_relations(j.OUTPUT/'CAMPAIGN.json',minimum_papers=2,include_retracted=True))
        old.crosscheck_fine_groups(stats['rid'],c,shared,default,inclusive)
        db.executescript('CREATE INDEX opportunities_rid ON opportunities(rid);CREATE INDEX opportunities_probe ON opportunities(probe);')
        flags={field:1<<i for i,field in enumerate(PROBES[1:])}
        case='CASE probe '+' '.join("WHEN '"+k+"' THEN "+str(v) for k,v in flags.items())+' END'
        db.execute('''CREATE TABLE root_masks AS SELECT r.k rid,r.n,r.source_keys,COALESCE(o.mask,0) mask FROM groups_rid r
            LEFT JOIN (SELECT rid,SUM(bit) mask FROM (SELECT DISTINCT rid,'''+case+''' bit FROM opportunities) GROUP BY rid) o ON r.k=o.rid''')
        partitions=old.all_rows(db,'SELECT mask,COUNT(*) fine_relations,SUM(n) claims FROM root_masks WHERE source_keys=1 GROUP BY mask ORDER BY mask')
        require(sum(r['fine_relations'] for r in partitions)==stats['rid']['groups']-stats['rid']['multi_source_key_groups'],'single-source partition not exhaustive')
        for r in partitions:r['probes']=[k for k,bit in flags.items() if r['mask']&bit]
        incidence=old.all_rows(db,'''SELECT o.probe,COUNT(DISTINCT o.rid) fine_relations,COUNT(DISTINCT CASE WHEN r.source_keys=1 THEN o.rid END) single_source_fine_relations
            FROM opportunities o JOIN groups_rid r ON r.k=o.rid GROUP BY o.probe ORDER BY o.probe''')
        phase='ENTITY_USAGE_AND_SOURCE_COVERAGE';progress(phase)
        db.executescript('''CREATE TABLE endpoint_incidents AS SELECT sid nid,cid,pk FROM claims UNION SELECT tid nid,cid,pk FROM claims;
            CREATE INDEX endpoint_incidents_nid ON endpoint_incidents(nid);
            CREATE TABLE endpoint_use AS SELECT nid,COUNT(*) claims,COUNT(DISTINCT pk) sources FROM endpoint_incidents GROUP BY nid;
            CREATE UNIQUE INDEX endpoint_use_nid ON endpoint_use(nid);''')
        used=old.all_rows(db,'''SELECT n.kind,COUNT(*) used_nodes,SUM(e.claims=1) one_claim_nodes,SUM(e.sources=1) one_source_nodes,
            SUM(e.sources>1) multiple_source_nodes,MAX(e.claims) most_claims FROM endpoint_use e JOIN nodes n ON n.nid=e.nid GROUP BY n.kind''')
        exact_names=old.all_rows(db,'''SELECT kind,COUNT(*) name_groups,SUM(ids) endpoint_ids FROM
            (SELECT n.kind,n.name,COUNT(*) ids FROM nodes n JOIN endpoint_use e ON n.nid=e.nid WHERE n.name<>'' GROUP BY n.kind,n.name HAVING COUNT(*)>1)
            GROUP BY kind''')
        source_distribution=old.all_rows(db,'''SELECT CASE WHEN n=1 THEN '1' WHEN n<=5 THEN '2-5' WHEN n<=10 THEN '6-10' ELSE '>10' END claims_per_source,
            COUNT(*) source_keys,SUM(n) claims FROM (SELECT pk,COUNT(*) n FROM claims GROUP BY pk) GROUP BY claims_per_source''')
        from prune_current_kg import rows
        queue=rows(c['current_gene_holds']['path']);current_hold_ids={r['claim_id'] for r in queue}
        queued_candidates=sum(db.execute('SELECT 1 FROM candidate_claims WHERE cid=?',(cid,)).fetchone() is not None for cid in current_hold_ids)
        members=old.all_rows(db,'''SELECT c.cid,c.node_sha,c.rid,c.sid,c.sn,c.p,c.tid,c.tn,c.pk,c.status,c.countable,c.neg,c.ctx,c.st,c.tt,
            c.surface,c.folded,c.physical,c.orthographic,c.registry,c.alias_route,c.predicate_family,c.orientation,c.scope_probe,c.topic
            FROM claims c JOIN candidate_claims k ON k.cid=c.cid ORDER BY c.cid''')
        write_rows(OUTPUT/'CANDIDATE_GROUPS.jsonl',candidates)
        write_rows(OUTPUT/'CANDIDATE_MEMBERS.jsonl',members)
        j.atomic_json(OUTPUT/'BOUNDED_EXAMPLES.json',dict(graph=c['current_graph'],examples=examples,sampling='top five groups per probe; up to six members each; not representative precision estimate'))
        write_rows(OUTPUT/'PRIORITY_QUEUE.jsonl',sorted(candidates,key=lambda r:(-r['counted_papers'],-r['source_keys'],-r['fine_variants'],-r['n'],r['probe'],r['k']))[:100])
        j.atomic_json(OUTPUT/'ROOT_CAUSE_PARTITION.json',dict(graph=c['current_graph'],partitions=partitions,overlapping_probe_incidence=incidence,
            probe_bits=flags,partition_unit='one-current-source fine relation',mask_zero_means='no cross-source candidate found under these bounded probes; NOT proven novel or truly single-study',
            classification_is_semantic_equivalence=False))
        j.atomic_json(OUTPUT/'ENTITY_AND_SOURCE_STATISTICS.json',dict(graph=c['current_graph'],used_entities=used,exact_name_multiple_ids=exact_names,
            source_claim_distribution=source_distribution,source_counts_are_not_independent_cohorts=True,
            queued_endpoint_count=len(queue),queued_claims=len(current_hold_ids),queued_claims_in_cross_source_candidates=queued_candidates))
        require(sum(x['claims'] for x in source_distribution)==scan_result['checked_claims'],'source distribution missing claims')
        require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','analysis integrity failed')
        db.commit()
        require(j.read_json(j.OUTPUT/'CAMPAIGN.json')==c and [j.fingerprint(p) for p in FILES]==baseline['code'],'analysis state/code changed')
        result=dict(status='FULL_CURRENT_FRAGMENTATION_ANALYSIS_COMPLETE',at=j.utc_now(),graph=c['current_graph'],acceptance=c['current_acceptance'],
            groupings=stats,unique_candidate_claims=len(members),candidate_group_probe_pairs=len(candidates),
            candidate_groups_overlap=True,grouping_delta_is_not_confirmed_duplicate_count=True,
            scientific_equivalence_certified=False,literature_absence_certified=False,cohort_independence_certified=False,
            all_claims_checked=scan_result['checked_claims'],actual_queries=dict(shared=len(shared),default=len(default),inclusive=len(inclusive)),
            artifacts=[j.fingerprint(OUTPUT/name) for name in ('SCAN.json','ROOT_CAUSE_PARTITION.json','ENTITY_AND_SOURCE_STATISTICS.json','CANDIDATE_GROUPS.jsonl','CANDIDATE_MEMBERS.jsonl','BOUNDED_EXAMPLES.json','PRIORITY_QUEUE.jsonl')],
            graph_mutations=0,model_calls=0,record_preimages_saved=False)
        j.atomic_json(OUTPUT/'ANALYSIS.json',result)
    finally:db.close()
    progress('ANALYSIS_COMPLETE_RULES_AND_ACCEPTANCE_PENDING',candidate_claims=len(members),groups=len(candidates))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','scan','analyze'))
    args=parser.parse_args()
    try:
        if args.phase in {'run','scan'}:scan()
        if args.phase in {'run','analyze'}:analyze()
    except BaseException as error:
        j.atomic_json(OUTPUT/'RUN_STATE.json',dict(status='FAILED',at=j.utc_now(),pid=os.getpid(),error=repr(error),graph_mutations=0))
        raise
