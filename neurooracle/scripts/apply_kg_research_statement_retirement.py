"""R46 exact research-intention retirement with independent kept-record proof."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as j
from apply_kg_bibliography_titles import CurrentCensus,small_check
from apply_kg_explicit_pmid_provenance import authorities,publication_row
from apply_kg_source_scope_resolution import FILES as PREVIOUS_FILES
from apply_kg_gene_boundary_repair import issue_row
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import cheap,compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows,guard_native,no_references
from reclaim_kg_backup_storage import native_info,sha256
from resume_kg_bibliography_titles import require_process_ended
from neurooracle.src import kg_research_statement_retirement as scope
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import bibliography,identifiers,title_key,census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage,node_class
from neurooracle.src.relation_evidence import evidence_member,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round46_research_retirement'
SOURCE=j.OUTPUT/'round23_source_deletion_candidate/knowledge_graph.candidate.json'
TEMP=SOURCE.with_name(SOURCE.name+'.research-retirement.tmp')
CATALOG=OUTPUT/'CURRENT_SHARED_RELATIONS.jsonl'
DATABASE=OUTPUT/'CURRENT_PAPER_CENSUS.sqlite'
FILES=list(dict.fromkeys([*PREVIOUS_FILES,Path(__file__),*[j.REPO/p for p in (
    'neurooracle/scripts/plan_kg_research_statement_retirement.py','neurooracle/src/kg_research_statement_retirement.py',
    'neurooracle/tests/test_kg_research_retirement.py','neurooracle/tests/test_kg_research_retirement_plan.py',
    'neurooracle/tests/test_kg_research_retirement_pipeline.py','neurooracle/scripts/report_kg_source_scope_resolution.py',
    'neurooracle/tests/test_kg_source_scope_report.py','neurooracle/scripts/inspect_kg_semantic_hold_sources.py',
    'neurooracle/tests/test_kg_semantic_source_inspection.py','neurooracle/scripts/fetch_kg_semantic_hold_sources.py')]]))
FILES=list(dict.fromkeys([*FILES,*[j.REPO/p for p in (
    'neurooracle/src/tests/test_claim_ingestion_persistence.py',
    'neurooracle/src/tests/test_compare_paired_luna_claim_campaign.py',
    'neurooracle/src/tests/test_graph.py','neurooracle/src/tests/test_paper_identity.py','neurooracle/src/tests/test_schema.py',
    'neurooracle/tests/test_claim_evidence_identity.py','neurooracle/tests/test_complete_pubmed_title_evidence.py',
    'neurooracle/tests/test_entity_resolution_safety.py','neurooracle/tests/test_kg_bulk_cleanup_report.py',
    'neurooracle/tests/test_kg_bulk_identity.py','neurooracle/tests/test_kg_clinical_literal_report.py',
    'neurooracle/tests/test_kg_identity_pilot.py','neurooracle/tests/test_kg_literal_endpoint_report.py',
    'neurooracle/tests/test_kg_literal_query_contract.py','neurooracle/tests/test_kg_measurement_reuse_report.py',
    'neurooracle/tests/test_kg_metadata_compaction.py','neurooracle/tests/test_kg_metadata_compaction_pipeline.py',
    'neurooracle/tests/test_kg_scoped_structure_report.py','neurooracle/tests/test_kg_verified_terms_pipeline.py',
    'neurooracle/tests/test_pubmed_title_integrity.py','neurooracle/tests/test_relation_evidence.py',
    'neurooracle/tests/test_shared_relation_catalog.py','neurooracle/tests/test_verified_entity_terms.py',
    'neurooracle/tests/test_verified_terms_ingestion_smoke.py')]]))


def bindings():return [j.fingerprint(p) for p in FILES]


def progress(phase,**values):
    state=dict(status='RUNNING',pid=os.getpid(),phase=phase,at=j.utc_now(),**values)
    j.atomic_json(OUTPUT/'RUN_STATE.json',state);print(compact(state),flush=True)


def test_evidence():
    counts,seen=Counter(),set()
    for suite in ET.parse(OUTPUT/'TEST_RESULTS.xml').getroot().iter('testsuite'):
        counts.update({k:int(suite.get(k,0)) for k in ('tests','failures','errors','skipped')})
        for case in suite.findall('testcase'):
            key=(case.get('classname'),case.get('name'));require(key not in seen,'duplicate tests');seen.add(key)
    require(counts['tests']==len(seen)>=870 and not any(counts[k] for k in ('failures','errors','skipped')),'complete distinct passing regression required')
    return dict(counts)


def load_plan(baseline):
    fp=j.fingerprint(OUTPUT/'PLAN.json');p=j.read_json(fp['path'])
    require(p['version']==scope.VERSION and p['status']=='REVIEWED_NOT_APPLIED' and p['relation_grouping']==POLICY,'wrong retirement policy')
    require(p['graph']==baseline['current_graph'] and p['source_acceptance']==baseline['current_acceptance'] and p['detail_store']==baseline['current_detail_store'],'plan/source advanced')
    for item in [*p['code'],p['source_inspection'],p['source_public_manifest'],p['public_source'],p['review_notes']]:small_check(item)
    proof=scope.public_source_proof(Path(p['public_source']['path']).read_text(encoding='utf8'))
    require(proof==p['public_source_proof'],'primary source proof differs')
    inspection=j.read_json(p['source_inspection']['path'])
    require(inspection['graph']==baseline['current_graph'] and inspection['full_source_sha_verified'] and inspection['full_census_sha_verified'],'source inspection differs')
    for item in [inspection['code'],*inspection['artifacts'].values()]:small_check(item)
    source_rows={r['claim_id']:r for r in rows(inspection['artifacts']['CURRENT_CLAIM_PROJECTIONS.jsonl']['path'])}
    for cid in scope.REVIEWS:
        r=source_rows[cid];md=dict(r['science'],id=cid,source_paper=r['source_paper'],metadata=r['inner_science'])
        scope.review_science(cid,r['claim_sha256'],md,proof)
        require(not any(p['deletion_detail_dependencies'][cid].values()),'offline deletion dependency')
    require(len(p['deleted_claims'])==p['deleted_claim_count']==4 and {r['claim_id'] for r in p['deleted_claims']}==set(scope.REVIEWS),'unreviewed claim deletion')
    require(p['removed_edge_ordinals']==sorted(set(p['removed_edge_ordinals'])) and len(p['removed_edge_ordinals'])==p['deleted_edges']==12,'unreviewed edge deletion')
    require(p['new_scientific_claims']==p['new_nodes']==0 and len(p['retired_active_bibliographies'])==4,'unreviewed graph/source expansion')
    return fp,p


class RetirementTransform:
    """Exact source deletions; every kept record enters the unchanged digest."""
    def __init__(self,plan):
        self.plan=plan;self.deleted={r['claim_id']:r for r in plan['deleted_claims']}
        self.removed={e['ordinal']:e for r in plan['deleted_claims'] for e in r['owned_edges']}
        require(set(self.deleted)==set(scope.REVIEWS) and len(self.removed)==12 and set(self.removed)==set(plan['removed_edge_ordinals']),'removal closure differs')
        self.digests={k:hashlib.sha256() for k in ('nodes','edges')}
        self.originals={};self.refs=defaultdict(list);self.seen_removed=set()
        self.needle=re.compile('|'.join(re.escape(cid) for cid in sorted(self.deleted)))

    def records(self,records):
        for kind,key,row in records:
            if kind=='node' and key in self.deleted:
                require(digest(row)==self.deleted[key]['claim_sha256'],'deleted source claim differs')
                scope.review_science(key,digest(row),row['metadata'],self.plan['public_source_proof'])
                require(key not in self.originals,'duplicate source claim');self.originals[key]=row
                continue
            if kind=='edge':
                ordinal=int(key);owner=scope.edge_owner(row)
                if owner in self.deleted:self.refs[owner].append((ordinal,row))
                if ordinal in self.removed:
                    require(owner in self.deleted and digest(row)==self.removed[ordinal]['edge_sha256'],'selected removed edge differs')
                    self.seen_removed.add(ordinal);continue
            no_references(row,compact(row).encode(),set(self.deleted),self.needle)
            if kind!='metadata':self.digests[kind+'s'].update(compact(row).encode()+b'\n')
            if kind=='edge':key=scope.compacted_ordinal(int(key),self.plan['removed_edge_ordinals'])
            yield kind,key,row
        require(set(self.originals)==set(self.deleted) and self.seen_removed==set(self.removed),'selected source record closure differs')
        for cid in self.deleted:
            require(scope.reviewed_claim(self.originals[cid],self.refs[cid],self.plan['public_source_proof'])==self.deleted[cid],'source-specific research retirement differs')


class IndependentRetirement:
    def __init__(self,plan,expected_digests):
        self.plan=plan;self.expected=expected_digests
        self.deleted={r['claim_id'] for r in plan['deleted_claims']}
        self.digests={k:hashlib.sha256() for k in ('nodes','edges')}
        self.needle=re.compile('|'.join(re.escape(cid) for cid in sorted(self.deleted)))

    def records(self,records):
        for kind,key,row in records:
            require(not (kind=='node' and key in self.deleted),'deleted claim remains')
            no_references(row,compact(row).encode(),self.deleted,self.needle)
            if kind!='metadata':self.digests[kind+'s'].update(compact(row).encode()+b'\n')
            yield kind,key,row

    def verify(self):
        require({k:v.hexdigest() for k,v in self.digests.items()}==self.expected,'all kept source record digests differ')


def build():
    require(not any(p.exists() for p in (TEMP,CATALOG,DATABASE,OUTPUT/'BUILD_STATE.json')),'build exists; inspect/resume')
    baseline=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(baseline['status']=='COMPLETED' and baseline['active_process'] is None and not baseline['rollback_retention'],'writer/retention boundary')
    require(Path(baseline['current_graph']['path']).resolve()==SOURCE.resolve(),'wrong source graph')
    window=j.read_json(baseline['night_window'])
    require(window['status']=='ACTIVE' and datetime.now(timezone.utc)<datetime.fromisoformat(window['end_at_utc'].replace('Z','+00:00')),'night window closed; no new writer')
    require(shutil.disk_usage(SOURCE).free>baseline['current_graph']['bytes']+12*1024**3,'insufficient atomic space')
    test_counts=test_evidence()
    for field in ('current_acceptance','current_runtime_acceptance','current_entity_terms','current_shared_relations','current_coverage',
        'current_issues','current_structure_holds','current_paper_issues','current_paper_census','current_gene_holds','current_scope_findings'):small_check(baseline[field])
    old_receipt=j.read_json(baseline['current_acceptance']['path'])
    for fp in old_receipt['code']:small_check(fp)
    plan_fp,plan=load_plan(baseline);papers,_=authorities(baseline)
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(baseline['current_entity_terms']['path'])))
    old_census=j.read_json(baseline['current_paper_census']['path'])
    require(set(scope.REVIEWS)<={r['claim_id'] for r in rows(baseline['current_issues']['path'])},'selected old semantic holds missing')
    pub_path=Path(baseline['current_acceptance']['path']).parent/'CURRENT_PUBLICATION_REVIEW.json';pub_fp=j.fingerprint(pub_path);pub=j.read_json(pub_path)
    require(pub['graph']==baseline['current_graph'] and pub['registry']==baseline['current_paper_identities'],'publication baseline differs')
    pub_rows={r['claim_id']:r for r in pub['claims'] if r['claim_id'] not in scope.REVIEWS};current_pub=[]
    protected=[baseline['current_graph'],baseline['current_detail_store'],old_census['database'],*baseline['formal_sources'].values()]
    guards=[native_info(fp['path']) for fp in protected];j.guards(protected);code=bindings()
    campaign=dict(baseline,status='MANUAL_ACTIVE',phase='R46四条研究意图误压成实测声明的精确移除',updated_at=j.utc_now(),
        active_process=dict(kind='research_statement_retirement',pid=os.getpid(),state=str(OUTPUT/'RUN_STATE.json')))
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',campaign)
    window['active_process']=campaign['active_process'];j.atomic_json(Path(baseline['night_window']),window)
    transform=RetirementTransform(plan);proof=SourceProof(terms,papers);coverage=Coverage();census=CurrentCensus(DATABASE,terms,set(plan['expected_shared_claim_ids']))
    paper_issues=[];reasons=Counter()

    def observed(records):
        for kind,key,row in records:
            if kind!='metadata':coverage.add('node/'+node_class(key,row) if kind=='node' else 'edge/all',row)
            if kind=='node' and key.startswith('CLM:'):
                census.add(key,row);answer=papers.resolve(row['metadata']);reasons.update(answer['reasons'])
                if answer['status']=='conflict':paper_issues.append(issue_row(key,row,answer))
                if key in pub_rows:current_pub.append(publication_row(pub_rows[key],row,row,papers))
            yield kind,key,row

    progress('BUILD_EXACT_RESEARCH_INTENTION_RETIREMENT',deleted_claims=4,removed_edges=12)
    try:
        with TEMP.open('xb',buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader,h):
                result=stream_patch(proof.records(observed(transform.records(walk_graph(reader)))),handle,[],{},CATALOG,progress,
                    relation_key_func=terms.relation_key,relation_version='kg.relation_evidence.v4',papers=papers,extra_graph_metadata=dict(relation_grouping=POLICY))
                require(h.hexdigest()==baseline['current_graph']['sha256'],'full source SHA differs')
            handle.flush();os.fsync(handle.fileno())
        require(result['counts']==plan['expected_counts'] and not result['changed'],'unexpected graph delta')
        require(result['self_loops']==old_receipt['self_loops'] and result['metadata_key_unions']==old_receipt['metadata_key_unions'],'unexpected loop/metadata union change')
        shared={m['claim_id'] for g in rows(CATALOG) for m in g['members']}
        require(shared==set(plan['expected_shared_claim_ids']),'independent membership projection differs')
        require(dict(proof.claim_status)==plan['expected_paper_claim_status'] and proof.key_changes==plan['expected_verified_key_changes'],'source identity count delta differs')
        require(dict(reasons)==plan['expected_paper_hold_reasons'],'source issue count delta differs')
        audit=dict(claim_status=dict(proof.claim_status),verified_key_changes=proof.key_changes,hold_reasons=dict(reasons),relation_counts=result['catalog_stats'])
        proof.verify(result,audit);census_result=census.finish()
    except BaseException:
        try:census.db.close()
        except Exception:pass
        raise
    require(census_result['counts']==plan['expected_census_counts'] and not census_result['outer_pmid_conflicts'],'current census delta differs')
    require(census_result['projection']['observed_collisions']==papers.export_payload()['observed_collisions'],'global paper collision differs')
    require({r['claim_id'] for r in current_pub}==set(pub_rows),'publication closure differs')
    pub.update(claims=current_pub,binding_method='full current hashes; four selected research-intention claims removed, surviving publication witnesses preserved')
    write_rows(OUTPUT/'CURRENT_METADATA_COVERAGE.jsonl',coverage.rows());write_rows(OUTPUT/'CURRENT_PAPER_ISSUES.jsonl',paper_issues)
    j.atomic_json(OUTPUT/'PAPER_IDENTITY_AUDIT.json',audit);j.atomic_json(OUTPUT/'PUBLICATION_REVIEW_BUILD.json',pub)
    db_fp={**cheap(DATABASE),'sha256':sha256(DATABASE)};j.atomic_json(OUTPUT/'CENSUS_BUILD.json',dict(**census_result,database=db_fp))
    guard_native(guards);require(bindings()==code,'frozen code changed');small_check(plan_fp);small_check(pub_fp)
    state=dict(status='BUILT_NOT_ADOPTED',at=j.utc_now(),baseline=baseline,protected_native=guards,temporary=cheap(TEMP),result=result,
        code=code,plan=plan_fp,catalog=j.fingerprint(CATALOG),papers=baseline['current_paper_identities'],terms=baseline['current_entity_terms'],database=db_fp,
        kept_source_record_digests={k:v.hexdigest() for k,v in transform.digests.items()},test_counts=test_counts,source_full_sha_verified=True,graph_backups=0,record_preimages_saved=False)
    for key,file in dict(census_build='CENSUS_BUILD.json',audit='PAPER_IDENTITY_AUDIT.json',paper_issues='CURRENT_PAPER_ISSUES.jsonl',
        coverage='CURRENT_METADATA_COVERAGE.jsonl',publication='PUBLICATION_REVIEW_BUILD.json',tests='TEST_RESULTS.xml').items():state[key]=j.fingerprint(OUTPUT/file)
    j.atomic_json(OUTPUT/'BUILD_STATE.json',state);progress('BUILT_NOT_ADOPTED',counts=result['counts'],relations=result['catalog_stats'])


def validate():
    require(not (OUTPUT/'VALIDATED.json').exists(),'already validated; inspect/resume')
    state=j.read_json(OUTPUT/'BUILD_STATE.json');baseline=state['baseline']
    require(bindings()==state['code'] and cheap(TEMP)==state['temporary'],'build/code changed');guard_native(state['protected_native'])
    for field in ('plan','catalog','papers','terms','census_build','audit','paper_issues','coverage','publication','tests'):small_check(state[field])
    _,plan=load_plan(baseline);papers,_=authorities(baseline)
    terms=IndexTerms(VerifiedEntityTerms(j.read_json(state['terms']['path'])))
    inverse=IndependentRetirement(plan,state['kept_source_record_digests']);proof=SourceProof(terms,papers);coverage=Coverage()
    paper_sigs,shared_ids,seen_pub=set(),set(),set();matched=0;issues=[];outer=[]
    pub=j.read_json(state['publication']['path']);pub_rows={r['claim_id']:r for r in pub['claims']}
    db=sqlite3.connect(DATABASE.as_uri()+'?mode=ro',uri=True)

    def independent_census(records):
        nonlocal matched
        for kind,key,row in records:
            if kind!='metadata':coverage.add('node/'+node_class(key,row) if kind=='node' else 'edge/all',row)
            if kind=='node' and key.startswith('CLM:'):
                md=row['metadata'];paper=bibliography(md['source_paper']);sig=digest(paper);member=evidence_member(md)
                rid=relation_id(terms.relation_key(md));observed=db.execute('SELECT node_sha,paper_sig,legacy_key,relation_id,shared FROM claims WHERE cid=?',(key,)).fetchone()
                require(observed is not None and observed[:4]==(digest(row),sig,member['paper_key'],rid),'census claim differs')
                if sig not in paper_sigs:
                    ids=identifiers(paper)
                    require(db.execute('SELECT payload,pmid,doi,pmcid,title,year FROM papers WHERE sig=?',(sig,)).fetchone()==
                        (compact(paper),ids['pmid'],ids['doi'],ids['pmcid'],title_key(paper.get('title')),str(paper.get('year') or paper.get('publication_year') or '')),'census paper differs')
                    paper_sigs.add(sig)
                if observed[4]:
                    require(db.execute('SELECT member_json FROM shared_members WHERE cid=?',(key,)).fetchone()==(compact(member),),'shared census differs');shared_ids.add(key)
                for holder in (md,md.get('metadata') or {}):
                    if holder.get('pmid') and str(holder['pmid'])!=str(paper.get('pmid') or ''):outer.append(key)
                answer=papers.resolve(md)
                if answer['status']=='conflict':issues.append(issue_row(key,row,answer))
                if key in pub_rows:
                    require(publication_row(pub_rows[key],row,row,papers)==pub_rows[key],'publication identity/hash differs');seen_pub.add(key)
                matched+=1
            yield kind,key,row

    progress('INDEPENDENT_FULL_KEPT_RECORDS_SOURCE_CLOSURE_CENSUS_AND_COVERAGE')
    try:
        with hashed_reader(TEMP) as (reader,h):
            checks=verify_catalog_and_graph(proof.records(independent_census(inverse.records(walk_graph(reader)))),state['result'],CATALOG,progress,
                relation_key_func=terms.relation_key,papers=papers)
            candidate_sha=h.hexdigest()
        census=j.read_json(state['census_build']['path'])
        require(matched==db.execute('SELECT COUNT(*) FROM claims').fetchone()[0]==plan['expected_counts']['claims'],'claim census closure differs')
        for cid in scope.REVIEWS:require(db.execute('SELECT cid FROM claims WHERE cid=?',(cid,)).fetchone() is None,'removed claim in census')
        require(len(paper_sigs)==db.execute('SELECT COUNT(*) FROM papers').fetchone()[0],'stale census paper')
        require(len(shared_ids)==db.execute('SELECT COUNT(*) FROM shared_members').fetchone()[0],'stale shared row')
        require(shared_ids==set(plan['expected_shared_claim_ids'])=={m['claim_id'] for g in rows(CATALOG) for m in g['members']},'shared closure differs')
        require(not outer and not census['outer_pmid_conflicts'] and census_identifier_projection(db)==census['projection'],'paper normalization differs')
        require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','census integrity failed')
    finally:db.close()
    inverse.verify()
    require(issues==rows(state['paper_issues']['path']) and seen_pub==set(pub_rows),'issues/publication closure differs')
    require(coverage.rows()==rows(state['coverage']['path']),'independent metadata coverage differs')
    proof.verify(state['result'],j.read_json(state['audit']['path']))
    progress('FULL_DETAIL_AND_CENSUS_SHA_BOUNDARY')
    require(sha256(Path(baseline['current_detail_store']['path']))==baseline['current_detail_store']['sha256'],'detail SHA differs')
    require(sha256(DATABASE)==state['database']['sha256'],'candidate census SHA differs')
    guard_native(state['protected_native']);require(bindings()==state['code'],'frozen code changed')
    checks.update(exact_source_review_reproduced=True,all_kept_source_node_and_edge_record_digests_reproduced=True,
        verified_identity_proofs_complete=True,paper_identity_witnesses_validated=True,publication_status_witnesses_validated=True,
        selected_claim_deleted_without_synthesizing_a_new_conclusion=True,deleted_claim_exact_references=0,
        only_four_exact_research_intention_claims_retired=True,four_active_bibliographies_retired_without_deleting_source_documents=True,
        all_existing_concepts_and_detail_store_unchanged=True,all_surviving_quotes_negation_conditions_scientific_and_audit_fields_preserved=True,
        no_metadata_fields_added=True,current_census_all_claims_papers_shared_verified=True,metadata_coverage_full_scan_verified=True,
        no_new_scientific_assertion_edges=True,authority_verified_claims=proof.claim_status['verified'])
    j.atomic_json(OUTPUT/'VALIDATED.json',dict(status='VALIDATED_NOT_ADOPTED',at=j.utc_now(),graph={**cheap(TEMP),'sha256':candidate_sha},
        temporary_native=native_info(TEMP),database_native=native_info(DATABASE),checks=checks,build_state=j.fingerprint(OUTPUT/'BUILD_STATE.json')))
    progress('VALIDATED_NOT_ADOPTED',counts=state['result']['counts'],relations=state['result']['catalog_stats'])


def project_issue_rows(records,graph,removed,deleted):
    projected=[]
    for record in records:
        if record['claim_id'] in deleted:continue
        row=deepcopy(record)
        for field in ('related_edge_ordinals','owned_science_ordinals','about_ordinals'):
            if field in row:row[field]=[scope.compacted_ordinal(o,removed) for o in row[field]]
        row.update(current_graph=graph,version_binding_status='FULL_CURRENT_GRAPH_VALIDATED');projected.append(row)
    return projected


def apply():
    require(not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists(),'already adopted')
    state=j.read_json(OUTPUT/'BUILD_STATE.json');accepted=j.read_json(OUTPUT/'VALIDATED.json');baseline=state['baseline']
    small_check(accepted['build_state']);require(bindings()==state['code'],'frozen code changed')
    guard_native([*state['protected_native'],accepted['temporary_native'],accepted['database_native']])
    for field in ('plan','catalog','papers','terms','census_build','audit','paper_issues','coverage','publication','tests'):small_check(state[field])
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json');_,plan=load_plan(baseline)
    require(c['current_graph']==baseline['current_graph'] and c['active_process']['kind']=='research_statement_retirement' and c['active_process']['pid']==os.getpid(),'writer/source advanced')
    require(TEMP.resolve().parent==SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(j.OUTPUT.resolve()),'unsafe replacement target')
    for field in ('current_issues','current_structure_holds','current_gene_holds','current_scope_findings'):small_check(baseline[field])
    issues=rows(baseline['current_issues']['path']);structure=rows(baseline['current_structure_holds']['path']);gene=rows(baseline['current_gene_holds']['path'])
    findings=j.read_json(baseline['current_scope_findings']['path']);deleted=set(scope.REVIEWS)
    require(len(issues)==21 and deleted<={r['claim_id'] for r in issues},'source semantic hold set changed')
    require(not deleted&set(findings['current_claim_hashes']),'unreviewed scientific-scope overlap')
    placeholder={**cheap(TEMP),'path':str(SOURCE),'sha256':accepted['graph']['sha256']}
    issues=project_issue_rows(issues,placeholder,plan['removed_edge_ordinals'],deleted)
    structure=project_issue_rows(structure,placeholder,plan['removed_edge_ordinals'],deleted)
    remaining=[deepcopy(r) for r in gene if r['claim_id'] not in deleted]
    require(len(issues)==plan['expected_remaining_semantic_holds']==17 and len(remaining)==len(gene),'review queue delta differs')
    os.replace(TEMP,SOURCE);graph={**cheap(SOURCE),'sha256':accepted['graph']['sha256']}
    require(graph==placeholder,'replacement differs')
    write_rows(OUTPUT/'CURRENT_REMAINING_ISSUES.jsonl',issues);write_rows(OUTPUT/'CURRENT_STRUCTURE_HOLDS.jsonl',structure)
    write_rows(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl',remaining)
    findings.setdefault('resolved',{})['research_intention_claims_retired']=sorted(deleted)
    findings.update(at=j.utc_now(),graph=graph,latest_research_retirement_plan=state['plan'],
        research_retirement_limits='Four exact research-intention flattenings only; qualified 17 semantic cases and all unrelated endpoint/scientific scopes remain held.')
    j.atomic_json(OUTPUT/'CURRENT_SCOPE_FINDINGS.json',findings)
    census=j.read_json(state['census_build']['path']);projection=census.pop('projection')
    j.atomic_json(OUTPUT/'CENSUS.json',dict(**census,status='CURRENT_CENSUS_COMPLETE',at=j.utc_now(),graph=graph,
        source_full_sha_verified=True,independent_full_candidate_verified=True,code=state['code'],record_preimages_saved=False))
    census_fp=j.fingerprint(OUTPUT/'CENSUS.json');j.atomic_json(OUTPUT/'CENSUS_NORMALIZATION.json',dict(**projection,census=census_fp))
    pub=j.read_json(state['publication']['path']);pub.update(at=j.utc_now(),graph=graph,current_census=census_fp)
    j.atomic_json(OUTPUT/'CURRENT_PUBLICATION_REVIEW.json',pub)
    j.atomic_json(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json',dict(graph=graph,current_remaining_endpoints=len(remaining),
        current_remaining_claims=len({r['claim_id'] for r in remaining}),removed_queue_endpoints_this_batch=0,
        current_remaining_semantic_holds=len(issues),retired_research_intention_claims=sorted(deleted),
        baseline_summary=baseline.get('current_gene_review_summary'),not_all_confirmed_errors=True))
    result=state['result'];delta=dict(deleted_claims=4,deleted_owned_edges=12,deleted_science_edges=4,deleted_about_edges=8,
        active_bibliographies_retired=4,identity_claim_nodes=0,endpoint_ids=0,identity_edge_records=0,new_literal_nodes=0)
    receipt=dict(status='CURRENT_RESEARCH_STATEMENT_RETIREMENT_APPLIED',at=j.utc_now(),graph=graph,counts=result['counts'],changed=delta,
        connected_components=result['connected_components'],isolated_nodes=result['isolated_nodes'],self_loops=result['self_loops'],
        metadata_key_unions=result['metadata_key_unions'],metadata=result['metadata'],relation_grouping=POLICY,
        source_graph=baseline['current_graph'],source_kg_retained=False,shared_relations=state['catalog'],entity_terms=state['terms'],paper_identities=state['papers'],
        paper_issues=state['paper_issues'],metadata_coverage=state['coverage'],current_paper_census=census_fp,relation_counts=result['catalog_stats'],
        checks=accepted['checks'],tests=state['tests'],test_counts=state['test_counts'],code=state['code'],detail_store=baseline['current_detail_store'],
        identity_audit=state['audit'],provenance_plan=state['plan'],current_gene_holds=j.fingerprint(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl'),
        deleted_claim_ids=sorted(deleted),original_source_documents_preserved=True,rollback_retention=False,record_preimages_saved=False,formal_apply_performed=False)
    j.atomic_json(OUTPUT/'CURRENT_ACCEPTANCE.json',receipt)
    j.atomic_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json',dict(status='CURRENT_RUNTIME_VALIDATED',at=receipt['at'],graph=graph,
        acceptance=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),code=state['code'],tests=state['tests'],test_counts=state['test_counts']))
    c.update(status='COMPLETED',phase='R46研究意图误压成实测声明已精确移除并验收',active_process=None,updated_at=receipt['at'],current_graph=graph,counts=result['counts'],
        node_metadata_field_union=len(result['metadata_key_unions']['nodes']),edge_metadata_field_union=len(result['metadata_key_unions']['edges']),
        current_acceptance=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),current_runtime_acceptance=j.fingerprint(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json'),
        current_issues=j.fingerprint(OUTPUT/'CURRENT_REMAINING_ISSUES.jsonl'),current_structure_holds=j.fingerprint(OUTPUT/'CURRENT_STRUCTURE_HOLDS.jsonl'),
        current_gene_holds=receipt['current_gene_holds'],current_scope_findings=j.fingerprint(OUTPUT/'CURRENT_SCOPE_FINDINGS.json'),
        current_shared_relations=state['catalog'],current_paper_issues=state['paper_issues'],current_coverage=state['coverage'],current_paper_census=census_fp,
        current_census_normalization=j.fingerprint(OUTPUT/'CENSUS_NORMALIZATION.json'),relation_evidence_counts=result['catalog_stats'],
        current_gene_review_summary=j.fingerprint(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json'),pending_gene_link_claims=len({r['claim_id'] for r in remaining}),
        pending_gene_link_endpoint_events=len(remaining),last_deep_verification=accepted['at'],last_current_graph_content_verification=dict(graph=graph,independent_full_scan=True),
        confirmed_issue_claims_removed_from_current=baseline['confirmed_issue_claims_removed_from_current']+4,
        historical_confirmed_issue_claims=baseline['historical_confirmed_issue_claims']+4,
        additional_semantic_or_detail_review_candidates=len(issues),
        formal_sync_gap_claims=baseline['formal_sync_gap_claims']+4,
        continuation_status='R46_ACCEPTED_CONTINUE_QUALIFIED_SOURCE_AND_ENDPOINT_REVIEW',
        next_steps=['继续447处历史端点的当前身份审查与扩大队列；已存在完整名称节点先排重。',
            '17条旧语义待核不统一否定；BACE1、产前复合宾语、ACC、小鼠VCP等范围继续审查。',
            '仅当前KG，无模型训练、重新抽取和正式同步；窗口结束只做必要验收收口。'])
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    window=j.read_json(Path(baseline['night_window']));window.update(active_process=None,latest_acceptance=c['current_acceptance'],latest_census=census_fp,
        latest_completed_batch='R46',latest_completed_runtime_batch='R46',latest_runtime_acceptance=c['current_runtime_acceptance'])
    j.atomic_json(Path(baseline['night_window']),window)
    j.atomic_json(OUTPUT/'RUN_STATE.json',dict(status='COMPLETED',at=receipt['at'],graph=graph,changed=delta,relations=result['catalog_stats']))
    print(compact(dict(status=receipt['status'],changed=delta,counts=result['counts'],relations=result['catalog_stats'])),flush=True)


def resume():
    state=j.read_json(OUTPUT/'BUILD_STATE.json')
    require(bindings()==state['code'] and not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists(),'not resumable')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['current_graph']==state['baseline']['current_graph'] and c['active_process']['kind']=='research_statement_retirement','unexpected active batch')
    require_process_ended(c['active_process']['pid']);c['active_process']['pid']=os.getpid();j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    window=j.read_json(c['night_window']);window['active_process']=c['active_process'];j.atomic_json(Path(c['night_window']),window)
    if not (OUTPUT/'VALIDATED.json').exists():validate()
    apply()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('run','build','resume'))
    phase=p.parse_args().phase
    if phase=='run':build();validate();apply()
    elif phase=='build':build()
    else:resume()
