"""R41 bulk exact redundancy removal and opt-in symmetric correlation index.

One current KG; independently validated temporary output, no graph/preimage
backups. Paper evidence, scientific values, audit objects and edges stay intact.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import CurrentCensus,small_check
from apply_kg_explicit_pmid_provenance import authorities,publication_row
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from apply_kg_scoped_structure import FILES as PREVIOUS_FILES
from build_umls_simplification_candidate import cheap,compact,hashed_reader,walk_graph
from compact_current_kg_metadata import protected_claim
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows,guard_native
from reclaim_kg_backup_storage import native_info,sha256
from resume_kg_bibliography_titles import require_process_ended
from neurooracle.src.kg_bulk_cleanup import simplify_record,verify_simplification
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import bibliography,identifiers,title_key,census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage,node_class
from neurooracle.src.relation_evidence import evidence_member,relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round41_bulk_cleanup'
SOURCE=j.OUTPUT/'round23_source_deletion_candidate/knowledge_graph.candidate.json'
TEMP=SOURCE.with_name(SOURCE.name+'.bulk-cleanup.tmp')
CATALOG=OUTPUT/'CURRENT_SHARED_RELATIONS.jsonl'
DATABASE=OUTPUT/'CURRENT_PAPER_CENSUS.sqlite'
ALLOWED_RUNTIME={str((j.REPO/p).resolve()) for p in (
    'neurooracle/src/kg_metadata_compaction.py','neurooracle/src/graph_manager.py',
    'neurooracle/src/storage.py','neurooracle/src/shared_relation_catalog.py')}
FILES=list(dict.fromkeys([*PREVIOUS_FILES,Path(__file__),*[j.REPO/p for p in (
    'neurooracle/src/kg_bulk_cleanup.py','neurooracle/src/correlation_grouping.py',
    'neurooracle/scripts/inspect_kg_bulk_cleanup.py','neurooracle/scripts/prepare_kg_bulk_cleanup.py',
    'neurooracle/scripts/compare_paired_luna_claim_campaign.py','neurooracle/scripts/manage_paired_luna_postreview.py',
    'neurooracle/scripts/compact_current_kg_metadata.py',
    'neurooracle/tests/test_kg_bulk_cleanup.py','neurooracle/tests/test_kg_bulk_cleanup_runtime.py',
    'neurooracle/tests/test_kg_bulk_cleanup_pipeline.py')]]))


def bindings():return [j.fingerprint(p) for p in FILES]


def progress(phase,**values):
    state=dict(status='RUNNING',pid=os.getpid(),phase=phase,at=j.utc_now(),**values)
    j.atomic_json(OUTPUT/'RUN_STATE.json',state);print(compact(state),flush=True)


def test_evidence():
    counts=Counter();seen=set()
    for suite in ET.parse(OUTPUT/'TEST_RESULTS.xml').getroot().iter('testsuite'):
        counts.update({k:int(suite.get(k,0)) for k in ('tests','failures','errors','skipped')})
        for case in suite.findall('testcase'):
            key=case.get('classname'),case.get('name');require(key not in seen,'duplicate regression');seen.add(key)
    require(counts['tests']==len(seen)>=730 and not any(counts[k] for k in ('failures','errors','skipped')),'passing complete regression required')
    return dict(counts)


def plan():
    require(not (OUTPUT/'PLAN.json').exists(),'plan exists; inspect current state')
    b=j.read_json(OUTPUT/'SOURCE_BOUNDARY.json');c=j.read_json(j.OUTPUT/'CAMPAIGN.json');base=b['campaign']
    require(c['status']=='MANUAL_PREPARING' and c['active_process'] is None,'not preparing')
    require(c['current_graph']==base['current_graph'] and c['current_acceptance']==base['current_acceptance'],'source advanced')
    inspection=j.read_json(OUTPUT/'SOURCE_INSPECTION.json');small_check(inspection['code'])
    require(inspection['graph']==base['current_graph'] and inspection['full_source_sha_verified'],'source scan not current')
    changed=[]
    for before in b['code']:
        after=j.fingerprint(before['path'])
        if before!=after:
            require(str(Path(before['path']).resolve()) in ALLOWED_RUNTIME,'unapproved existing code edit')
            changed.append(dict(before=before,after=after))
    require({str(Path(r['before']['path']).resolve()) for r in changed}==ALLOWED_RUNTIME,'runtime edit set differs')
    j.guards([base['current_graph'],base['current_detail_store'],base['formal_sources']])
    old_groups=rows(base['current_shared_relations']['path']);small_check(base['current_shared_relations'])
    reverse=j.read_json(OUTPUT/'REVERSED_CORRELATION_GROUPS.json')
    require(reverse['graph']==base['current_graph'],'reverse census not current')
    shared={m['claim_id'] for g in old_groups for m in g['members']}
    shared.update(m['claim_id'] for g in reverse['groups'] for m in g['members'])
    j.atomic_json(OUTPUT/'PLAN.json',dict(version='kg.bulk_cleanup.v1',at=j.utc_now(),graph=base['current_graph'],
        source_acceptance=base['current_acceptance'],source_boundary=j.fingerprint(OUTPUT/'SOURCE_BOUNDARY.json'),
        source_inspection=j.fingerprint(OUTPUT/'SOURCE_INSPECTION.json'),
        reversed_groups=j.fingerprint(OUTPUT/'REVERSED_CORRELATION_GROUPS.json'),runtime_changes=changed,
        code=bindings(),tests=j.fingerprint(OUTPUT/'TEST_RESULTS.xml'),test_counts=test_evidence(),
        expected_shared_claim_ids=sorted(shared),expected_relation_groups=base['relation_evidence_counts']['all_fine_grained_relation_groups']-len(reverse['groups']),
        relation_grouping=POLICY,edge_changes=0,node_identity_changes=0,scientific_or_audit_value_changes=0,
        remove_only_exact_empty_hints_and_canonical_duplicates=True,record_preimages_saved=False))
    print('R41_PLAN_FROZEN',flush=True)


def load_plan():
    p=j.read_json(OUTPUT/'PLAN.json');base=j.read_json(p['source_boundary']['path'])['campaign']
    for fp in (p['source_boundary'],p['source_inspection'],p['reversed_groups'],p['tests']):small_check(fp)
    require(bindings()==p['code'] and p['graph']==base['current_graph'],'plan/source/code differs')
    changes={str(Path(r['before']['path']).resolve()):r for r in p['runtime_changes']}
    for fp in j.read_json(base['current_acceptance']['path'])['code']:
        change=changes.get(str(Path(fp['path']).resolve()))
        if change:require(change['before']==fp and j.fingerprint(fp['path'])==change['after'],'planned runtime binding differs')
        else:small_check(fp)
    return p,base


def issue_row(key,row,answer):
    return dict(claim_id=key,claim_sha256=digest(row),reasons=answer['reasons'],
        legacy_source_key=evidence_member(row['metadata'])['paper_key'],action='hold_no_claim_or_source_deletion')


def wanted_hashes(base):
    known={r['claim_id']:r['current_node_sha256'] for r in rows(base['current_issues']['path'])}
    for r in rows(base['current_gene_holds']['path']):
        require(known.get(r['claim_id'],r['claim_sha256'])==r['claim_sha256'],'source queue conflict')
        known[r['claim_id']]=r['claim_sha256']
    for cid,h in j.read_json(base['current_scope_findings']['path'])['current_claim_hashes'].items():
        require(known.get(cid,h)==h,'scope queue conflict');known[cid]=h
    return known


def build():
    require(not any(p.exists() for p in (TEMP,CATALOG,DATABASE,OUTPUT/'BUILD_STATE.json')),'build exists; inspect/resume')
    p,base=load_plan();c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['status']=='MANUAL_PREPARING' and c['active_process'] is None and c['current_graph']==base['current_graph'],'unexpected writer/source')
    require(not base['rollback_retention'] and base['automation_id'] is None,'single-version/manual boundary differs')
    require(shutil.disk_usage(SOURCE).free>base['current_graph']['bytes']+12*1024**3,'insufficient atomic space')
    for key in ('current_acceptance','current_runtime_acceptance','current_entity_terms','current_paper_identities','current_shared_relations',
        'current_paper_census','current_coverage','current_issues','current_structure_holds','current_gene_holds','current_scope_findings'):small_check(base[key])
    papers,_=authorities(base);terms=IndexTerms(VerifiedEntityTerms(j.read_json(base['current_entity_terms']['path'])))
    old_receipt=j.read_json(base['current_acceptance']['path']);old_census=j.read_json(base['current_paper_census']['path'])
    old_audit=j.read_json(old_receipt['identity_audit']['path'])
    pub_path=Path(base['current_acceptance']['path']).parent/'CURRENT_PUBLICATION_REVIEW.json';pub_fp=j.fingerprint(pub_path);pub=j.read_json(pub_path)
    require(pub['graph']==base['current_graph'] and pub['registry']==base['current_paper_identities'],'publication baseline differs')
    pub_rows={r['claim_id']:r for r in pub['claims']};current_pub=[]
    protected=[native_info(fp['path']) for fp in (base['current_graph'],base['current_detail_store'],old_census['database'],*base['formal_sources'].values())]
    j.guards([base['current_graph'],base['current_detail_store'],old_census['database'],base['formal_sources']])
    c.update(status='MANUAL_ACTIVE',phase='R41批量去冗余及对称相关索引构建中',updated_at=j.utc_now(),
        active_process=dict(kind='bulk_cleanup',pid=os.getpid(),state=str(OUTPUT/'RUN_STATE.json')))
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    proof=SourceProof(terms,papers);coverage=Coverage();census=CurrentCensus(DATABASE,terms,set(p['expected_shared_claim_ids']))
    removals=Counter();changed=Counter();science=hashlib.sha256();paper_issues=[];reasons=Counter()
    needed=wanted_hashes(base);current_hashes={};same_records={k:hashlib.sha256() for k in ('nonclaim_nodes','edges')}

    def transformed(records):
        for kind,key,source in records:
            out=source
            if kind=='node' and key.startswith('CLM:'):
                out=simplify_record(kind,source,removals)
                verify_simplification(source,out)
                before=protected_claim(source);after=protected_claim(out)
                require(digest(before)==digest(after),'science/audit contract changed')
                science.update(compact(before).encode()+b'\n');changed['claim_nodes']+=out is not source
                if key in needed:
                    require(digest(source)==needed[key],'current held claim hash differs');current_hashes[key]=digest(out)
                census.add(key,out);answer=papers.resolve(out['metadata']);reasons.update(answer['reasons'])
                if answer['status']=='conflict':paper_issues.append(issue_row(key,out,answer))
                if key in pub_rows:current_pub.append(publication_row(pub_rows[key],source,out,papers))
            elif kind!='metadata':
                same_records['edges' if kind=='edge' else 'nonclaim_nodes'].update(compact(source).encode()+b'\n')
            if kind!='metadata':coverage.add('node/'+node_class(key,out) if kind=='node' else 'edge/all',out)
            yield kind,key,out

    progress('BUILD_APPROVED_BULK_CLEANUP')
    try:
        with TEMP.open('xb',buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader,h):
                result=stream_patch(proof.records(transformed(walk_graph(reader))),handle,[],{},CATALOG,progress,
                    relation_key_func=terms.relation_key,relation_version='kg.relation_evidence.v4',papers=papers,
                    extra_graph_metadata=dict(relation_grouping=POLICY))
                require(h.hexdigest()==base['current_graph']['sha256'],'full source SHA differs')
            handle.flush();os.fsync(handle.fileno())
        require(result['counts']==base['counts'] and not result['changed'],'unexpected record count or edge delta')
        require(all(result[k]==old_receipt[k] for k in ('self_loops','connected_components','isolated_nodes')),'topology changed')
        require(result['metadata_key_unions']==old_receipt['metadata_key_unions'],'top-level metadata union changed')
        require(result['catalog_stats']['all_fine_grained_relation_groups']==p['expected_relation_groups'],'unapproved relation grouping')
        shared={m['claim_id'] for g in rows(CATALOG) for m in g['members']}
        require(shared==set(p['expected_shared_claim_ids']),'shared membership differs')
        require(set(current_hashes)==set(needed),'held claim closure incomplete')
        require(dict(proof.claim_status)==old_audit['claim_status'] and proof.key_changes==old_audit['verified_key_changes'],'paper identities changed')
        require(dict(reasons)==old_audit['hold_reasons'],'paper hold reasons changed')
        audit=dict(claim_status=dict(proof.claim_status),verified_key_changes=proof.key_changes,hold_reasons=dict(reasons),relation_counts=result['catalog_stats'])
        proof.verify(result,audit);census_result=census.finish()
    except BaseException:
        try:census.db.close()
        except Exception:pass
        raise
    expected=dict(old_census['counts']);expected['shared_claims']=len(shared)
    require(census_result['counts']==expected and not census_result['outer_pmid_conflicts'],'census record/identity count differs')
    require(census_result['projection']['observed_collisions']==papers.export_payload()['observed_collisions'],'paper collision projection changed')
    require({r['claim_id'] for r in current_pub}==set(pub_rows),'publication closure differs')
    pub.update(claims=current_pub,binding_method='full current claim hashes; exact scientific and audit object invariants')
    write_rows(OUTPUT/'CURRENT_METADATA_COVERAGE.jsonl',coverage.rows());write_rows(OUTPUT/'CURRENT_PAPER_ISSUES.jsonl',paper_issues)
    j.atomic_json(OUTPUT/'PAPER_IDENTITY_AUDIT.json',audit);j.atomic_json(OUTPUT/'PUBLICATION_REVIEW_BUILD.json',pub)
    db_fp={**cheap(DATABASE),'sha256':sha256(DATABASE)};j.atomic_json(OUTPUT/'CENSUS_BUILD.json',dict(**census_result,database=db_fp))
    guard_native(protected);require(bindings()==p['code'],'frozen code changed');small_check(pub_fp)
    state=dict(status='BUILT_NOT_ADOPTED',at=j.utc_now(),baseline=base,protected_native=protected,temporary=cheap(TEMP),
        result=result,code=p['code'],plan=j.fingerprint(OUTPUT/'PLAN.json'),catalog=j.fingerprint(CATALOG),
        papers=base['current_paper_identities'],terms=base['current_entity_terms'],database=db_fp,tests=p['tests'],test_counts=p['test_counts'],
        removals=dict(removals),changed=dict(changed),science_digest=science.hexdigest(),held_hashes=current_hashes,
        untouched_record_digests={k:v.hexdigest() for k,v in same_records.items()},source_full_sha_verified=True,
        graph_backups=0,record_preimages_saved=False)
    for key,file in dict(census_build='CENSUS_BUILD.json',audit='PAPER_IDENTITY_AUDIT.json',paper_issues='CURRENT_PAPER_ISSUES.jsonl',
        coverage='CURRENT_METADATA_COVERAGE.jsonl',publication='PUBLICATION_REVIEW_BUILD.json').items():
        state[key]=j.fingerprint(OUTPUT/file)
    j.atomic_json(OUTPUT/'BUILD_STATE.json',state);progress('BUILT_NOT_ADOPTED',changed=state['changed'],removals=sum(removals.values()),relations=result['catalog_stats'])


def validate():
    require(not (OUTPUT/'VALIDATED.json').exists(),'already validated')
    state=j.read_json(OUTPUT/'BUILD_STATE.json');p,base=load_plan()
    require(cheap(TEMP)==state['temporary'] and bindings()==state['code'],'candidate/code changed')
    guard_native(state['protected_native'])
    for key in ('plan','catalog','papers','terms','census_build','audit','paper_issues','coverage','publication','tests'):small_check(state[key])
    papers,_=authorities(base);terms=IndexTerms(VerifiedEntityTerms(j.read_json(state['terms']['path'])))
    proof=SourceProof(terms,papers);coverage=Coverage();science=hashlib.sha256();same_records={k:hashlib.sha256() for k in state['untouched_record_digests']}
    matched=0;paper_sigs=set();shared=set();held=set();issues=[];outer=[]
    pub=j.read_json(state['publication']['path']);pub_rows={r['claim_id']:r for r in pub['claims']};seen_pub=set()
    db=sqlite3.connect(DATABASE.as_uri()+'?mode=ro',uri=True)

    def independent(records):
        nonlocal matched
        for kind,key,row in records:
            if kind!='metadata':coverage.add('node/'+node_class(key,row) if kind=='node' else 'edge/all',row)
            if kind=='node' and key.startswith('CLM:'):
                require(simplify_record(kind,row)==row,'approved redundancy remains')
                science.update(compact(protected_claim(row)).encode()+b'\n')
                md=row['metadata'];paper=bibliography(md['source_paper']);sig=digest(paper);member=evidence_member(md)
                observed=db.execute('SELECT node_sha,paper_sig,legacy_key,relation_id,shared FROM claims WHERE cid=?',(key,)).fetchone()
                require(observed is not None and observed[:4]==(digest(row),sig,member['paper_key'],relation_id(terms.relation_key(md))),'current census claim differs')
                if sig not in paper_sigs:
                    ids=identifiers(paper)
                    require(db.execute('SELECT payload,pmid,doi,pmcid,title,year FROM papers WHERE sig=?',(sig,)).fetchone()==
                        (compact(paper),ids['pmid'],ids['doi'],ids['pmcid'],title_key(paper.get('title')),str(paper.get('year') or paper.get('publication_year') or '')),'census bibliography differs')
                    paper_sigs.add(sig)
                if observed[4]:
                    require(db.execute('SELECT member_json FROM shared_members WHERE cid=?',(key,)).fetchone()==(compact(member),),'shared census member differs');shared.add(key)
                for holder in (md,md.get('metadata') or {}):
                    if holder.get('pmid') and str(holder['pmid'])!=str(paper.get('pmid') or ''):outer.append(key)
                answer=papers.resolve(md)
                if answer['status']=='conflict':issues.append(issue_row(key,row,answer))
                if key in pub_rows:
                    require(publication_row(pub_rows[key],row,row,papers)==pub_rows[key],'publication witness differs');seen_pub.add(key)
                if key in state['held_hashes']:
                    require(digest(row)==state['held_hashes'][key],'held current hash differs');held.add(key)
                matched+=1
            elif kind!='metadata':same_records['edges' if kind=='edge' else 'nonclaim_nodes'].update(compact(row).encode()+b'\n')
            yield kind,key,row

    progress('INDEPENDENT_FULL_CANDIDATE_SCIENCE_CENSUS_COVERAGE')
    try:
        with hashed_reader(TEMP) as (reader,h):
            checks=verify_catalog_and_graph(proof.records(independent(walk_graph(reader))),state['result'],CATALOG,progress,
                relation_key_func=terms.relation_key,papers=papers)
            graph_sha=h.hexdigest()
        census=j.read_json(state['census_build']['path'])
        require(matched==db.execute('SELECT COUNT(*) FROM claims').fetchone()[0]==base['counts']['claims'],'claim count closure differs')
        require(len(paper_sigs)==db.execute('SELECT COUNT(*) FROM papers').fetchone()[0],'stale census bibliography')
        require(shared==set(p['expected_shared_claim_ids']) and len(shared)==db.execute('SELECT COUNT(*) FROM shared_members').fetchone()[0],'shared census closure differs')
        require(not outer and census_identifier_projection(db)==census['projection'],'paper normalization differs')
        require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','census SQLite integrity failed')
    finally:db.close()
    require(science.hexdigest()==state['science_digest'],'protected science/audit summary differs')
    require({k:v.hexdigest() for k,v in same_records.items()}==state['untouched_record_digests'],'nonclaim nodes or edge records changed')
    require(held==set(state['held_hashes']) and seen_pub==set(pub_rows),'current holds/publication incomplete')
    require(coverage.rows()==rows(state['coverage']['path']) and issues==rows(state['paper_issues']['path']),'independent coverage/issues differ')
    proof.verify(state['result'],j.read_json(state['audit']['path']))
    progress('FULL_DETAIL_AND_CENSUS_SHA_BOUNDARY')
    require(sha256(Path(base['current_detail_store']['path']))==base['current_detail_store']['sha256'],'detail SHA differs')
    require(sha256(DATABASE)==state['database']['sha256'],'census SHA differs')
    guard_native(state['protected_native']);require(bindings()==state['code'],'frozen code changed')
    checks.pop('records_match_approved_identity_only_transform',None)
    checks.pop('all_nonidentity_claim_fields_preserved',None)
    checks.update(records_match_approved_exact_metadata_cleanup=True,all_scientific_values_and_original_audit_objects_preserved=True,
        nonclaim_nodes_and_all_edges_identical=True,no_node_or_edge_addition_or_deletion=True,
        metadata_coverage_full_scan_verified=True,current_census_all_claims_papers_and_shared_members_verified=True,
        verified_identity_proofs_complete=True,paper_identity_witnesses_validated=True,publication_status_witnesses_validated=True,
        symmetric_grouping_only_exact_correlates_with=True,original_claim_and_edge_orientation_preserved=True)
    j.atomic_json(OUTPUT/'VALIDATED.json',dict(status='VALIDATED_NOT_ADOPTED',at=j.utc_now(),graph={**cheap(TEMP),'sha256':graph_sha},
        temporary_native=native_info(TEMP),database_native=native_info(DATABASE),checks=checks,build_state=j.fingerprint(OUTPUT/'BUILD_STATE.json')))
    progress('VALIDATED_NOT_ADOPTED',counts=state['result']['counts'])


def apply():
    require(not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists(),'already adopted')
    state=j.read_json(OUTPUT/'BUILD_STATE.json');accepted=j.read_json(OUTPUT/'VALIDATED.json');p,base=load_plan()
    small_check(accepted['build_state']);guard_native([*state['protected_native'],accepted['temporary_native'],accepted['database_native']])
    for key in ('catalog','papers','terms','census_build','audit','paper_issues','coverage','publication','tests'):small_check(state[key])
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['current_graph']==base['current_graph'] and c['active_process']['pid']==os.getpid() and c['active_process']['kind']=='bulk_cleanup','writer/source advanced')
    require(SOURCE.resolve().parent==TEMP.resolve().parent and SOURCE.resolve().is_relative_to(j.OUTPUT.resolve()),'unsafe adoption path')
    os.replace(TEMP,SOURCE);graph={**cheap(SOURCE),'sha256':accepted['graph']['sha256']}
    require(graph['bytes']==accepted['graph']['bytes'] and graph['mtime_ns']==accepted['graph']['mtime_ns'],'atomic replacement differs')
    for field,filename in (('current_issues','CURRENT_REMAINING_ISSUES.jsonl'),('current_structure_holds','CURRENT_STRUCTURE_HOLDS.jsonl'),('current_gene_holds','CURRENT_GENE_ENDPOINT_HOLDS.jsonl')):
        records=rows(base[field]['path'])
        for r in records:
            for key in ('current_node_sha256','claim_sha256'):
                if key in r:r[key]=state['held_hashes'][r['claim_id']]
            if field!='current_gene_holds':r.update(current_graph=graph,version_binding_status='FULL_CURRENT_GRAPH_VALIDATED')
        write_rows(OUTPUT/filename,records)
    scopes=j.read_json(base['current_scope_findings']['path']);scopes.update(at=j.utc_now(),graph=graph,
        current_claim_hashes={cid:state['held_hashes'][cid] for cid in scopes['current_claim_hashes']})
    j.atomic_json(OUTPUT/'CURRENT_SCOPE_FINDINGS.json',scopes)
    census=j.read_json(state['census_build']['path']);projection=census.pop('projection')
    j.atomic_json(OUTPUT/'CENSUS.json',dict(**census,status='CURRENT_CENSUS_COMPLETE',at=j.utc_now(),graph=graph,
        source_full_sha_verified=True,independent_full_candidate_verified=True,code=state['code'],record_preimages_saved=False))
    census_fp=j.fingerprint(OUTPUT/'CENSUS.json');j.atomic_json(OUTPUT/'CENSUS_NORMALIZATION.json',dict(**projection,census=census_fp))
    pub=j.read_json(state['publication']['path']);pub.update(at=j.utc_now(),graph=graph,current_census=census_fp)
    j.atomic_json(OUTPUT/'CURRENT_PUBLICATION_REVIEW.json',pub)
    result=state['result'];receipt=dict(status='CURRENT_BULK_CLEANUP_APPLIED',at=j.utc_now(),graph=graph,counts=result['counts'],
        changed=state['changed'],removals=state['removals'],removed_field_occurrences=sum(state['removals'].values()),
        connected_components=result['connected_components'],isolated_nodes=result['isolated_nodes'],self_loops=result['self_loops'],
        metadata_key_unions=result['metadata_key_unions'],metadata=result['metadata'],source_graph=base['current_graph'],source_kg_retained=False,
        shared_relations=state['catalog'],entity_terms=state['terms'],paper_identities=state['papers'],paper_issues=state['paper_issues'],
        metadata_coverage=state['coverage'],current_paper_census=census_fp,relation_counts=result['catalog_stats'],relation_grouping=POLICY,
        checks=accepted['checks'],tests=state['tests'],test_counts=state['test_counts'],code=state['code'],detail_store=base['current_detail_store'],
        identity_audit=state['audit'],provenance_plan=state['plan'],current_gene_holds=j.fingerprint(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl'),
        graph_bytes_reduced=base['current_graph']['bytes']-graph['bytes'],rollback_retention=False,record_preimages_saved=False,formal_apply_performed=False)
    j.atomic_json(OUTPUT/'CURRENT_ACCEPTANCE.json',receipt)
    j.atomic_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json',dict(status='CURRENT_RUNTIME_VALIDATED',at=receipt['at'],graph=graph,
        acceptance=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),code=state['code'],tests=state['tests'],test_counts=state['test_counts']))
    c.update(status='COMPLETED',phase='R41批量metadata精简和对称相关索引已验收',active_process=None,updated_at=receipt['at'],current_graph=graph,
        counts=result['counts'],node_metadata_field_union=len(result['metadata_key_unions']['nodes']),edge_metadata_field_union=len(result['metadata_key_unions']['edges']),
        current_acceptance=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),current_runtime_acceptance=j.fingerprint(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json'),
        current_shared_relations=state['catalog'],current_paper_issues=state['paper_issues'],current_coverage=state['coverage'],current_paper_census=census_fp,
        current_census_normalization=j.fingerprint(OUTPUT/'CENSUS_NORMALIZATION.json'),relation_evidence_counts=result['catalog_stats'],
        current_issues=j.fingerprint(OUTPUT/'CURRENT_REMAINING_ISSUES.jsonl'),current_structure_holds=j.fingerprint(OUTPUT/'CURRENT_STRUCTURE_HOLDS.jsonl'),
        current_gene_holds=receipt['current_gene_holds'],current_scope_findings=j.fingerprint(OUTPUT/'CURRENT_SCOPE_FINDINGS.json'),
        metadata_compaction=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),last_deep_verification=accepted['at'],
        last_current_graph_content_verification=dict(graph=graph,independent_full_scan=True),
        next_steps=['R41已整批去除空提示/规范副本并收拢反向相关索引；不等于科学证据归并完成。',
            '449处端点及本篇范围问题仍须证据核实；不凭缺类型或共享率删除。',
            '统计空值及审核封印仍保留，低非空率不等于字段无用；无模型训练或正式同步。'])
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c);j.atomic_json(OUTPUT/'RUN_STATE.json',dict(status='COMPLETED',at=receipt['at'],graph=graph,
        changed=receipt['changed'],removed_field_occurrences=receipt['removed_field_occurrences'],relations=result['catalog_stats']))
    print(compact(dict(status=receipt['status'],changed=receipt['changed'],removals=receipt['removed_field_occurrences'],saved_bytes=receipt['graph_bytes_reduced'],relations=result['catalog_stats'])),flush=True)


def resume():
    state=j.read_json(OUTPUT/'BUILD_STATE.json');c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists() and c['current_graph']==state['baseline']['current_graph'],'not resumable')
    require(c['active_process']['kind']=='bulk_cleanup','unexpected active writer');require_process_ended(c['active_process']['pid'])
    c['active_process']['pid']=os.getpid();j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    if not (OUTPUT/'VALIDATED.json').exists():validate()
    apply()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('plan','run','resume'));phase=parser.parse_args().phase
    if phase=='plan':plan()
    elif phase=='resume':resume()
    else:build();validate();apply()
