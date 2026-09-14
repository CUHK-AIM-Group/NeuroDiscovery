"""R44 exact source resolution, independently verified, with no record preimages."""
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
from apply_kg_gene_boundary_repair import FILES as PREVIOUS_FILES,issue_row
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import cheap,compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows,write_rows,guard_native,no_references
from reclaim_kg_backup_storage import native_info,sha256
from resume_kg_bibliography_titles import require_process_ended
from neurooracle.src import kg_source_scope_resolution as scope
from neurooracle.src.kg_bulk_identity import change_claim
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import bibliography,identifiers,title_key,census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage,node_class
from neurooracle.src.relation_evidence import evidence_member,relation_id
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT=j.OUTPUT/'round44_source_resolution'
SOURCE=j.OUTPUT/'round23_source_deletion_candidate/knowledge_graph.candidate.json'
TEMP=SOURCE.with_name(SOURCE.name+'.source-resolution.tmp')
CATALOG=OUTPUT/'CURRENT_SHARED_RELATIONS.jsonl'
DATABASE=OUTPUT/'CURRENT_PAPER_CENSUS.sqlite'
FILES=list(dict.fromkeys([*PREVIOUS_FILES,Path(__file__),*[j.REPO/p for p in (
    'neurooracle/scripts/plan_kg_source_scope_resolution.py','neurooracle/src/kg_source_scope_resolution.py',
    'neurooracle/tests/test_kg_source_scope_resolution.py','neurooracle/tests/test_kg_source_scope_plan.py',
    'neurooracle/tests/test_kg_source_scope_pipeline.py','neurooracle/scripts/report_kg_gene_boundary_repair.py',
    'neurooracle/tests/test_kg_gene_boundary_report.py','neurooracle/scripts/publish_kg_gene_boundary_query_contract.py',
    'neurooracle/tests/test_kg_gene_boundary_query_contract.py','neurooracle/scripts/fetch_kg_viral_reuse_sources.py')]]))


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
    require(counts['tests']==len(seen)>=825 and not any(counts[k] for k in ('failures','errors','skipped')),'complete distinct passing regression required')
    return dict(counts)


def load_plan(baseline):
    fp=j.fingerprint(OUTPUT/'PLAN.json');p=j.read_json(fp['path'])
    require(p['version']==scope.VERSION and p['status']=='REVIEWED_NOT_APPLIED' and p['relation_grouping']==POLICY,'wrong repair policy')
    require(p['graph']==baseline['current_graph'] and p['source_acceptance']==baseline['current_acceptance'] and p['detail_store']==baseline['current_detail_store'],'plan/source advanced')
    for item in [*p['code'],p['source_manifest'],*p['public_sources'],p['viral_reuse_source_manifest'],p['viral_reuse_public_source']]:small_check(item)
    proof=scope.public_source_proof(Path(p['public_sources'][0]['path']).read_text(encoding='utf8'),Path(p['public_sources'][1]['path']).read_text(encoding='utf8'))
    require(proof==p['public_source_proof'],'primary source proof differs')
    reuse_sources=scope.viral_reuse_source_proof(Path(p['viral_reuse_public_source']['path']).read_text(encoding='utf8'))
    if p.get('viral_target_witness'):
        scope.validate_viral_reuse_witness(p['viral_target_witness'])
        require(reuse_sources==p['viral_target_witness']['public_identity_sources'],'viral reuse public source differs')
    require(len(p['deleted_claims'])==p['deleted_claim_count']==1 and p['deleted_claims'][0]['claim_id']==scope.MYELIN,'unreviewed claim deletion')
    require(p['removed_edge_ordinals']==sorted(set(p['removed_edge_ordinals'])) and len(p['removed_edge_ordinals'])==p['deleted_edges']==5,'unreviewed edge deletion')
    require(len(p['events'])==p['changed_claims']<=1 and all(e['claim_id']==scope.VIRUS for e in p['events']),'unreviewed identity mutation')
    return fp,p


class ResolutionTransform:
    """Only kept source rows enter the digest; deleted rows have exact proofs."""
    def __init__(self,plan):
        self.plan=plan;self.events={e['claim_id']:e for e in plan['events']}
        self.edges={e['ordinal']:e for e in plan['edge_events']}
        self.removed={r['ordinal']:r for r in [*plan['deleted_claims'][0]['owned_edges'],*plan['retired_branch_edges']]}
        self.gone=plan['removed_edge_ordinals'];self.new={r['id']:scope.viral_node() for r in plan['new_literals']}
        require(len(self.removed)==5 and set(self.removed)==set(self.gone),'removal closure differs')
        require(all(r['node_sha256']==digest(self.new[r['id']]) and r['id']==self.new[r['id']]['id'] for r in plan['new_literals']),'new literal differs')
        self.digests={k:hashlib.sha256() for k in ('nodes','edges')}
        self.originals={};self.refs=defaultdict(list);self.seen_targets=set();self.seen_edges=set();self.seen_removed=set();self.appended=False
        self.needle=re.compile(re.escape(scope.MYELIN))

    def records(self,records):
        for kind,key,row in records:
            if kind=='edge' and not self.appended:
                for nid,node in sorted(self.new.items()):yield 'node',nid,node
                self.appended=True
            if kind=='node':
                require(key not in self.new,'new literal ID collides')
                if key in self.plan['existing_targets']:
                    require(digest(row)==self.plan['existing_targets'][key],'existing target changed');self.seen_targets.add(key)
                if key in scope.CLAIM_HASHES:
                    scope.source_claim(row,self.plan['public_source_proof']);self.originals[key]=row
                if key==scope.MYELIN:continue
            elif kind=='edge':
                ordinal=int(key)
                if scope.edge_owner(row) in scope.CLAIM_HASHES:self.refs[scope.edge_owner(row)].append((ordinal,row))
                if ordinal in self.removed:
                    require(digest(row)==self.removed[ordinal]['edge_sha256'],'selected removed edge changed')
                    self.seen_removed.add(ordinal);continue
            no_references(row,compact(row).encode(),{scope.MYELIN},self.needle)
            if kind!='metadata':self.digests[kind+'s'].update(compact(row).encode()+b'\n')
            out=row
            if kind=='node' and key in self.events:out=change_claim(row,self.events[key])
            if kind=='edge':
                if int(key) in self.edges:out=scope.apply_edge(row,self.edges[int(key)]);self.seen_edges.add(int(key))
                key=scope.compacted_ordinal(int(key),self.gone)
            yield kind,key,out
        require(self.appended and self.seen_targets==set(self.plan['existing_targets']),'source target closure differs')
        require(set(self.originals)==set(scope.CLAIM_HASHES) and self.seen_removed==set(self.removed) and self.seen_edges==set(self.edges),'selected source record closure differs')
        require(scope.reviewed_myelin_removal(self.originals[scope.MYELIN],self.refs[scope.MYELIN],self.plan['public_source_proof'])==self.plan['deleted_claims'][0],'myelin removal review differs')
        require(scope.reviewed_frailty_retirements(self.originals[scope.FRAILTY],self.refs[scope.FRAILTY],self.plan['public_source_proof'])==self.plan['retired_branch_edges'],'frailty retirement review differs')
        if self.events:
            event,out=scope.reviewed_viral_identity(self.originals[scope.VIRUS],self.plan['public_source_proof'],
                self.events[scope.VIRUS]['changes'][0]['target_id'],self.plan.get('viral_target_witness'))
            require(all(self.events[scope.VIRUS][k]==v for k,v in event.items()),'viral identity review differs')
            require(scope.reviewed_edges(scope.VIRUS,self.originals[scope.VIRUS],out,self.refs[scope.VIRUS])==self.plan['edge_events'],'viral whole edge closure differs')


class IndependentResolution:
    def __init__(self,plan,expected_digests):
        self.plan=plan;self.expected=expected_digests
        self.events={e['claim_id']:e for e in plan['events']};self.edges={e['ordinal']:e for e in plan['edge_events']}
        self.new={r['id']:r for r in plan['new_literals']};self.digests={k:hashlib.sha256() for k in ('nodes','edges')}
        self.nodes=set();self.changed_edges=set();self.added=set();self.targets=set();self.originals={};self.refs=defaultdict(list);self.canonical={}
        self.needle=re.compile(re.escape(scope.MYELIN))

    def records(self,records):
        for kind,key,row in records:
            no_references(row,compact(row).encode(),{scope.MYELIN},self.needle)
            original=row
            if kind=='node':
                if key in self.new:
                    require(row==scope.viral_node() and digest(row)==self.new[key]['node_sha256'],'added viral literal differs')
                    self.added.add(key);original=None
                elif key in self.events:
                    original=scope.reverse_claim(row,self.events[key]);self.originals[key]=original;self.nodes.add(key)
                if key==scope.FRAILTY:
                    require(digest(row)==self.plan['frailty_claim_sha256'],'retained frailty claim changed');self.originals[key]=row
                if key in self.plan['existing_targets']:
                    require(digest(row)==self.plan['existing_targets'][key],'retained entity changed');self.targets.add(key)
            elif kind=='edge':
                old=scope.original_ordinal(int(key),self.plan['removed_edge_ordinals'])
                if old in self.edges:original=scope.apply_edge(row,self.edges[old],reverse=True);self.changed_edges.add(old)
                if scope.edge_owner(original) in {scope.FRAILTY,scope.VIRUS}:self.refs[scope.edge_owner(original)].append((old,original))
                if old in {r['ordinal'] for r in self.plan['frailty_current_canonical_edges']}:self.canonical[old]=row
            if kind!='metadata' and original is not None:self.digests[kind+'s'].update(compact(original).encode()+b'\n')
            yield kind,key,row

    def verify(self):
        p=self.plan
        require({k:v.hexdigest() for k,v in self.digests.items()}==self.expected,'all kept source record digests differ')
        require(self.nodes==set(self.events) and self.changed_edges==set(self.edges) and self.added==set(self.new) and self.targets==set(p['existing_targets']),'independent mutation closure differs')
        canonical=p['frailty_current_canonical_edges']
        require(set(self.canonical)=={r['ordinal'] for r in canonical} and len(self.refs[scope.FRAILTY])==3,'frailty current closure differs')
        for r in canonical:require(digest(self.canonical[r['ordinal']])==r['edge_sha256'],'canonical frailty edge changed')
        recreated=list(self.refs[scope.FRAILTY])
        for r in p['retired_branch_edges']:
            row=deepcopy(self.canonical[r['retained_ordinal']]);require(digest(row)==r['retained_edge_sha256'],'retained counterpart differs')
            row.update(r['retired_endpoint_values']);require(digest(row)==r['edge_sha256'],'retired duplicate reconstruction differs')
            recreated.append((r['ordinal'],row))
        require(scope.reviewed_frailty_retirements(self.originals[scope.FRAILTY],recreated,p['public_source_proof'])==p['retired_branch_edges'],'independent frailty retirement proof differs')
        if self.events:
            original=self.originals[scope.VIRUS];event,out=scope.reviewed_viral_identity(original,p['public_source_proof'],
                self.events[scope.VIRUS]['changes'][0]['target_id'],p.get('viral_target_witness'))
            require(all(self.events[scope.VIRUS][k]==v for k,v in event.items()),'independent viral source review differs')
            require(scope.reviewed_edges(scope.VIRUS,original,out,self.refs[scope.VIRUS])==p['edge_events'],'independent viral reference review differs')


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
    require(not {scope.MYELIN,scope.VIRUS}&{r['claim_id'] for r in rows(baseline['current_issues']['path'])},'unreviewed old semantic issue overlap')
    pub_path=Path(baseline['current_acceptance']['path']).parent/'CURRENT_PUBLICATION_REVIEW.json';pub_fp=j.fingerprint(pub_path);pub=j.read_json(pub_path)
    require(pub['graph']==baseline['current_graph'] and pub['registry']==baseline['current_paper_identities'],'publication baseline differs')
    pub_rows={r['claim_id']:r for r in pub['claims'] if r['claim_id']!=scope.MYELIN};current_pub=[]
    protected=[baseline['current_graph'],baseline['current_detail_store'],old_census['database'],*baseline['formal_sources'].values()]
    guards=[native_info(fp['path']) for fp in protected];j.guards(protected);code=bindings()
    campaign=dict(baseline,status='MANUAL_ACTIVE',phase='R44来源错误声明、旧衰弱分支与病毒VCP身份集中收口',updated_at=j.utc_now(),
        active_process=dict(kind='source_scope_resolution',pid=os.getpid(),state=str(OUTPUT/'RUN_STATE.json')))
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',campaign)
    window['active_process']=campaign['active_process'];j.atomic_json(Path(baseline['night_window']),window)
    transform=ResolutionTransform(plan);proof=SourceProof(terms,papers);coverage=Coverage();census=CurrentCensus(DATABASE,terms,set(plan['expected_shared_claim_ids']))
    paper_issues=[];reasons=Counter()

    def observed(records):
        for kind,key,row in records:
            if kind!='metadata':coverage.add('node/'+node_class(key,row) if kind=='node' else 'edge/all',row)
            if kind=='node' and key.startswith('CLM:'):
                census.add(key,row);answer=papers.resolve(row['metadata']);reasons.update(answer['reasons'])
                if answer['status']=='conflict':paper_issues.append(issue_row(key,row,answer))
                if key in pub_rows:current_pub.append(publication_row(pub_rows[key],transform.originals.get(key,row),row,papers))
            yield kind,key,row

    progress('BUILD_EXACT_SOURCE_RESOLUTION',deleted_claims=1,removed_edges=5,identity_claims=plan['changed_claims'])
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
    pub.update(claims=current_pub,binding_method='full current hashes; one selected claim removed, surviving publication witnesses preserved')
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
    inverse=IndependentResolution(plan,state['kept_source_record_digests']);proof=SourceProof(terms,papers);coverage=Coverage()
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
        require(db.execute('SELECT cid FROM claims WHERE cid=?',(scope.MYELIN,)).fetchone() is None,'removed claim in census')
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
        canonical_frailty_claim_and_three_edges_preserved=True,retired_branch_payloads_reconstructed_from_kept_counterparts=True,
        viral_identity_scope_checked=bool(plan['events']),no_global_vcp_alias_or_organism_merge=True,
        all_existing_concepts_and_detail_store_unchanged=True,all_surviving_quotes_negation_conditions_scientific_and_audit_fields_preserved=True,
        no_metadata_fields_added=True,current_census_all_claims_papers_shared_verified=True,metadata_coverage_full_scan_verified=True,
        no_new_scientific_assertion_edges=True,authority_verified_claims=proof.claim_status['verified'])
    j.atomic_json(OUTPUT/'VALIDATED.json',dict(status='VALIDATED_NOT_ADOPTED',at=j.utc_now(),graph={**cheap(TEMP),'sha256':candidate_sha},
        temporary_native=native_info(TEMP),database_native=native_info(DATABASE),checks=checks,build_state=j.fingerprint(OUTPUT/'BUILD_STATE.json')))
    progress('VALIDATED_NOT_ADOPTED',counts=state['result']['counts'],relations=state['result']['catalog_stats'])


def project_issue_rows(records,graph,removed):
    projected=[]
    for record in records:
        row=deepcopy(record)
        require(row['claim_id']!=scope.MYELIN,'deleted claim remains in old semantic holds')
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
    require(c['current_graph']==baseline['current_graph'] and c['active_process']['kind']=='source_scope_resolution' and c['active_process']['pid']==os.getpid(),'writer/source advanced')
    require(TEMP.resolve().parent==SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(j.OUTPUT.resolve()),'unsafe replacement target')
    for field in ('current_issues','current_structure_holds','current_gene_holds','current_scope_findings'):small_check(baseline[field])
    issues=rows(baseline['current_issues']['path']);structure=rows(baseline['current_structure_holds']['path']);gene=rows(baseline['current_gene_holds']['path'])
    findings=j.read_json(baseline['current_scope_findings']['path']);changed={e['claim_id']:e for e in plan['events']}
    require(findings['current_claim_hashes'].get(scope.MYELIN)==scope.CLAIM_HASHES[scope.MYELIN] and findings['current_claim_hashes'].get(scope.FRAILTY)==scope.CLAIM_HASHES[scope.FRAILTY],'source scope bindings changed')
    # Compute all hold projections before replacing the only current graph.
    placeholder={**cheap(TEMP),'path':str(SOURCE),'sha256':accepted['graph']['sha256']}
    issues=project_issue_rows(issues,placeholder,plan['removed_edge_ordinals'])
    structure=project_issue_rows([r for r in structure if r['claim_id']!=scope.FRAILTY],placeholder,plan['removed_edge_ordinals'])
    remaining=[]
    for row in gene:
        if row['claim_id']==scope.MYELIN or (row['claim_id']==scope.VIRUS and row['side']=='subject' and scope.VIRUS in changed):continue
        r=deepcopy(row)
        if r['claim_id'] in changed:r['claim_sha256']=changed[r['claim_id']]['current_node_sha256']
        remaining.append(r)
    os.replace(TEMP,SOURCE);graph={**cheap(SOURCE),'sha256':accepted['graph']['sha256']}
    require(graph==placeholder,'replacement differs')
    write_rows(OUTPUT/'CURRENT_REMAINING_ISSUES.jsonl',issues);write_rows(OUTPUT/'CURRENT_STRUCTURE_HOLDS.jsonl',structure)
    write_rows(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl',remaining)
    for cid in (scope.MYELIN,scope.FRAILTY):findings['current_claim_hashes'].pop(cid,None)
    findings['findings']=[r for r in findings['findings'] if r['claim_id'] not in {scope.MYELIN,scope.FRAILTY}]
    findings.setdefault('resolved',{}).update(qmri_severity_unsupported_claim_removed=scope.MYELIN,frailty_obsolete_owned_branch_retired=scope.FRAILTY)
    if scope.VIRUS in changed:findings['resolved']['source_specific_viral_vcp_identity']=scope.VIRUS
    findings.update(at=j.utc_now(),graph=graph,latest_source_resolution_plan=state['plan'],
        source_resolution_limits='Frailty concepts and other physical anchors are not globally merged; BACE1, prenatal compound, ACC and mouse VCP scopes remain under review.')
    j.atomic_json(OUTPUT/'CURRENT_SCOPE_FINDINGS.json',findings)
    census=j.read_json(state['census_build']['path']);projection=census.pop('projection')
    j.atomic_json(OUTPUT/'CENSUS.json',dict(**census,status='CURRENT_CENSUS_COMPLETE',at=j.utc_now(),graph=graph,
        source_full_sha_verified=True,independent_full_candidate_verified=True,code=state['code'],record_preimages_saved=False))
    census_fp=j.fingerprint(OUTPUT/'CENSUS.json');j.atomic_json(OUTPUT/'CENSUS_NORMALIZATION.json',dict(**projection,census=census_fp))
    pub=j.read_json(state['publication']['path']);pub.update(at=j.utc_now(),graph=graph,current_census=census_fp)
    j.atomic_json(OUTPUT/'CURRENT_PUBLICATION_REVIEW.json',pub)
    j.atomic_json(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json',dict(graph=graph,current_remaining_endpoints=len(remaining),
        current_remaining_claims=len({r['claim_id'] for r in remaining}),removed_queue_endpoints_this_batch=len(gene)-len(remaining),
        baseline_summary=baseline.get('current_gene_review_summary'),not_all_confirmed_errors=True))
    result=state['result'];delta=dict(deleted_claims=1,deleted_owned_edges=5,retired_frailty_branch_edges=3,
        identity_claim_nodes=plan['changed_claims'],endpoint_ids=plan['changed_endpoints'],identity_edge_records=plan['changed_edges'],new_literal_nodes=plan['added_literal_nodes'])
    receipt=dict(status='CURRENT_SOURCE_SCOPE_RESOLUTION_APPLIED',at=j.utc_now(),graph=graph,counts=result['counts'],changed=delta,
        connected_components=result['connected_components'],isolated_nodes=result['isolated_nodes'],self_loops=result['self_loops'],
        metadata_key_unions=result['metadata_key_unions'],metadata=result['metadata'],relation_grouping=POLICY,
        source_graph=baseline['current_graph'],source_kg_retained=False,shared_relations=state['catalog'],entity_terms=state['terms'],paper_identities=state['papers'],
        paper_issues=state['paper_issues'],metadata_coverage=state['coverage'],current_paper_census=census_fp,relation_counts=result['catalog_stats'],
        checks=accepted['checks'],tests=state['tests'],test_counts=state['test_counts'],code=state['code'],detail_store=baseline['current_detail_store'],
        identity_audit=state['audit'],provenance_plan=state['plan'],current_gene_holds=j.fingerprint(OUTPUT/'CURRENT_GENE_ENDPOINT_HOLDS.jsonl'),
        deleted_claim_ids=[scope.MYELIN],rollback_retention=False,record_preimages_saved=False,formal_apply_performed=False)
    j.atomic_json(OUTPUT/'CURRENT_ACCEPTANCE.json',receipt)
    j.atomic_json(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json',dict(status='CURRENT_RUNTIME_VALIDATED',at=receipt['at'],graph=graph,
        acceptance=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),code=state['code'],tests=state['tests'],test_counts=state['test_counts']))
    c.update(status='COMPLETED',phase='R44来源范围及重复旧分支已验收',active_process=None,updated_at=receipt['at'],current_graph=graph,counts=result['counts'],
        node_metadata_field_union=len(result['metadata_key_unions']['nodes']),edge_metadata_field_union=len(result['metadata_key_unions']['edges']),
        current_acceptance=j.fingerprint(OUTPUT/'CURRENT_ACCEPTANCE.json'),current_runtime_acceptance=j.fingerprint(OUTPUT/'CURRENT_RUNTIME_ACCEPTANCE.json'),
        current_issues=j.fingerprint(OUTPUT/'CURRENT_REMAINING_ISSUES.jsonl'),current_structure_holds=j.fingerprint(OUTPUT/'CURRENT_STRUCTURE_HOLDS.jsonl'),
        current_gene_holds=receipt['current_gene_holds'],current_scope_findings=j.fingerprint(OUTPUT/'CURRENT_SCOPE_FINDINGS.json'),
        current_shared_relations=state['catalog'],current_paper_issues=state['paper_issues'],current_coverage=state['coverage'],current_paper_census=census_fp,
        current_census_normalization=j.fingerprint(OUTPUT/'CENSUS_NORMALIZATION.json'),relation_evidence_counts=result['catalog_stats'],
        current_gene_review_summary=j.fingerprint(OUTPUT/'CURRENT_REVIEW_QUEUE_SUMMARY.json'),pending_gene_link_claims=len({r['claim_id'] for r in remaining}),
        pending_gene_link_endpoint_events=len(remaining),last_deep_verification=accepted['at'],last_current_graph_content_verification=dict(graph=graph,independent_full_scan=True),
        confirmed_issue_claims_removed_from_current=baseline['confirmed_issue_claims_removed_from_current']+1,
        historical_confirmed_issue_claims=baseline['historical_confirmed_issue_claims']+1,
        formal_sync_gap_claims=baseline['formal_sync_gap_claims']+1,
        continuation_status='R44_ACCEPTED_CONTINUE_REMAINING_SOURCE_AND_ENDPOINT_REVIEW',
        next_steps=['剩余科学范围与扩大端点待核继续基于来源收口；未知项不强并或强删。',
            'BACE1浓度/通路、产前复合宾语、ACC观察性谓词与截断摘录、小鼠VCP身份仍待核。',
            '保持单当前KG、无模型训练、无正式full_v2同步；夜间窗口到点只做验收收口。'])
    j.atomic_json(j.OUTPUT/'CAMPAIGN.json',c)
    window=j.read_json(Path(baseline['night_window']));window.update(active_process=None,latest_acceptance=c['current_acceptance'],latest_census=census_fp,
        latest_completed_batch='R44',latest_completed_runtime_batch='R44',latest_runtime_acceptance=c['current_runtime_acceptance'])
    j.atomic_json(Path(baseline['night_window']),window)
    j.atomic_json(OUTPUT/'RUN_STATE.json',dict(status='COMPLETED',at=receipt['at'],graph=graph,changed=delta,relations=result['catalog_stats']))
    print(compact(dict(status=receipt['status'],changed=delta,counts=result['counts'],relations=result['catalog_stats'])),flush=True)


def resume():
    state=j.read_json(OUTPUT/'BUILD_STATE.json')
    require(bindings()==state['code'] and not (OUTPUT/'CURRENT_ACCEPTANCE.json').exists(),'not resumable')
    c=j.read_json(j.OUTPUT/'CAMPAIGN.json')
    require(c['current_graph']==state['baseline']['current_graph'] and c['active_process']['kind']=='source_scope_resolution','unexpected active batch')
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
