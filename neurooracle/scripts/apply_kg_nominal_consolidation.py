"""R65 one manually resumed nominal batch; closed nightly history is immutable."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from apply_kg_bibliography_titles import CurrentCensus, small_check
from apply_kg_explicit_pmid_provenance import authorities, publication_row
from apply_kg_observational_semantics import FILES as PREVIOUS_FILES
from plan_kg_nominal_consolidation import load_plan, NEW_FILES
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows, guard_native
from reclaim_kg_backup_storage import native_info, sha256
from resume_kg_bibliography_titles import require_process_ended
from neurooracle.src.kg_bulk_identity import change_claim
from neurooracle.src.kg_identity_pilot import digest, nonidentity_claim
from neurooracle.src.kg_nominal_consolidation import (reviewed_claim, NominalTransform,
    no_retired_references, ordered_incidences, reviewed_edges, edge_owner, apply_edge, reverse_claim)
from neurooracle.src.kg_paper_identity import bibliography, identifiers, title_key, census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage, node_class
from neurooracle.src.relation_evidence import evidence_member, relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
from neurooracle.src.correlation_grouping import POLICY, IndexTerms
from datetime import datetime, timezone

OUTPUT=journal.OUTPUT/"round65_nominal_consolidation"
SOURCE=journal.OUTPUT/"round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP=SOURCE.with_name(SOURCE.name+".nominal-consolidation.tmp")
CATALOG=OUTPUT/"CURRENT_SHARED_RELATIONS.jsonl"
DATABASE=OUTPUT/"CURRENT_PAPER_CENSUS.sqlite"
FILES=list(dict.fromkeys([*PREVIOUS_FILES,*NEW_FILES,
    journal.REPO/"neurooracle/scripts/inspect_kg_nominal_consolidation.py",
    journal.REPO/"neurooracle/tests/test_kg_nominal_scope_inspection.py"]))


def bindings(): return [journal.fingerprint(p) for p in FILES]


def progress(phase,**values):
    state=dict(status="RUNNING",pid=os.getpid(),phase=phase,at=journal.utc_now(),**values)
    journal.atomic_json(OUTPUT/"RUN_STATE.json",state); print(compact(state),flush=True)


def test_evidence():
    counts,seen=Counter(),set()
    for suite in ET.parse(OUTPUT/"TEST_RESULTS.xml").getroot().iter("testsuite"):
        counts.update({k:int(suite.get(k,0)) for k in ("tests","failures","errors","skipped")})
        for case in suite.findall("testcase"):
            key=(case.get("classname"),case.get("name")); require(key not in seen,"duplicate tests"); seen.add(key)
    require(counts["tests"]==len(seen)>=1400 and not any(counts[k] for k in ("failures","errors","skipped")),"complete distinct passing regression required")
    return dict(counts)



def issue_row(key,row,answer):
    return dict(claim_id=key,claim_sha256=digest(row),reasons=answer["reasons"],
        legacy_source_key=evidence_member(row["metadata"])["paper_key"],action="hold_no_claim_or_source_deletion")



def build():
    require(not any(p.exists() for p in (TEMP,CATALOG,DATABASE,OUTPUT/"BUILD_STATE.json")),"build exists; inspect/resume")
    baseline=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(baseline["status"]=="COMPLETED" and baseline["active_process"] is None and not baseline["rollback_retention"],"writer/retention boundary")
    require(Path(baseline["current_graph"]["path"]).resolve()==SOURCE.resolve(),"wrong source graph")
    # This is a new one-batch manual authorization, not a reopened night window.
    _,manual_plan=load_plan(baseline)
    require(journal.fingerprint(baseline["night_window"])==manual_plan["closed_night_window"],"closed nightly history changed")
    require(shutil.disk_usage(SOURCE).free>baseline["current_graph"]["bytes"]+12*1024**3,"insufficient atomic space")
    test_counts=test_evidence()
    for field in ("current_acceptance","current_runtime_acceptance","current_entity_terms","current_shared_relations","current_coverage",
        "current_issues","current_structure_holds","current_paper_issues","current_paper_census"):
        small_check(baseline[field])
    old_receipt=journal.read_json(baseline["current_acceptance"]["path"])
    for fp in old_receipt["code"]: small_check(fp)
    plan_fp,plan=load_plan(baseline)
    events={e["claim_id"]:e for e in plan["events"]}
    for field in ("current_issues","current_structure_holds"):
        require(not set(events)&{r["claim_id"] for r in rows(baseline[field]["path"])},"scientific hold overlaps; separate review required")
    papers,_=authorities(baseline)
    terms=IndexTerms(VerifiedEntityTerms(journal.read_json(baseline["current_entity_terms"]["path"])))
    old_census=journal.read_json(baseline["current_paper_census"]["path"])
    prior_audit=journal.read_json(old_receipt["identity_audit"]["path"])
    pub_path=Path(baseline["current_acceptance"]["path"]).parent/"CURRENT_PUBLICATION_REVIEW.json"
    pub_fp=journal.fingerprint(pub_path); pub=journal.read_json(pub_path)
    require(pub["graph"]==baseline["current_graph"] and pub["registry"]==baseline["current_paper_identities"],"publication baseline differs")
    pub_rows={r["claim_id"]:r for r in pub["claims"]}; current_pub=[]
    guards=[native_info(fp["path"]) for fp in (baseline["current_graph"],baseline["current_detail_store"],old_census["database"],plan["closed_night_window"],*baseline["formal_sources"].values())]
    journal.guards([baseline["current_graph"],baseline["current_detail_store"],old_census["database"],baseline["formal_sources"]])
    code=bindings()
    campaign=dict(baseline,status="MANUAL_ACTIVE",phase="R65手动续作：七组同名科学端点归并及两处无引用节点退役",updated_at=journal.utc_now(),
        active_process=dict(kind="nominal_consolidation",pid=os.getpid(),state=str(OUTPUT/"RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",campaign)
    transform=NominalTransform(plan); proof=SourceProof(terms,papers); coverage=Coverage()
    census=CurrentCensus(DATABASE,terms,set(plan["expected_shared_claim_ids"]))
    paper_issues=[]; reasons=Counter()

    def observed(records):
        for kind,key,row in records:
            if kind!="metadata": coverage.add("node/"+node_class(key,row) if kind=="node" else "edge/all",row)
            if kind=="node" and key.startswith("CLM:"):
                census.add(key,row)
                answer=papers.resolve(row["metadata"]); reasons.update(answer["reasons"])
                if answer["status"]=="conflict": paper_issues.append(issue_row(key,row,answer))
                if key in pub_rows:
                    current_pub.append(publication_row(pub_rows[key],transform.originals.get(key,row),row,papers))
            yield kind,key,row

    progress("BUILD_NOMINAL_IDENTITY_AND_EXACT_RETIREMENT",claims=plan["changed_claims"],removed_concepts=len(plan["removed_nodes"]))
    try:
        with TEMP.open("xb",buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader,h):
                result=stream_patch(proof.records(observed(transform.records(walk_graph(reader)))),handle,[],{},CATALOG,progress,
                    relation_key_func=terms.relation_key,relation_version="kg.relation_evidence.v4",papers=papers,
                    extra_graph_metadata=dict(relation_grouping=POLICY))
                require(h.hexdigest()==baseline["current_graph"]["sha256"],"full source SHA differs")
            handle.flush(); os.fsync(handle.fileno())
        expected_counts=dict(baseline["counts"]); expected_counts["nodes"]-=len(plan["removed_nodes"])
        require(result["counts"]==expected_counts and not result["changed"],"unexpected record delta")
        require(result["self_loops"]==old_receipt["self_loops"],"self loops changed")
        require(result["metadata_key_unions"]==old_receipt["metadata_key_unions"],"metadata field union changed")
        shared={m["claim_id"] for g in rows(CATALOG) for m in g["members"]}
        require(shared==set(plan["expected_shared_claim_ids"]),"independent shared member projection differs")
        require(dict(proof.claim_status)==prior_audit["claim_status"] and proof.key_changes==prior_audit["verified_key_changes"],"paper identities changed")
        require(dict(reasons)==prior_audit["hold_reasons"],"paper hold reasons changed")
        audit=dict(claim_status=dict(proof.claim_status),verified_key_changes=proof.key_changes,hold_reasons=dict(reasons),relation_counts=result["catalog_stats"])
        proof.verify(result,audit)
        census_result=census.finish()
    except BaseException:
        try: census.db.close()
        except Exception: pass
        raise
    expected_census=dict(old_census["counts"]); expected_census["shared_claims"]=len(shared)
    require(census_result["counts"]==expected_census and not census_result["outer_pmid_conflicts"],"bibliography census changed")
    require(census_result["projection"]["observed_collisions"]==papers.export_payload()["observed_collisions"],"global paper collision changed")
    require({r["claim_id"] for r in current_pub}==set(pub_rows),"publication closure differs")
    pub.update(claims=current_pub,binding_method="full current graph claim hashes, preserved source identity and publication witnesses")
    write_rows(OUTPUT/"CURRENT_METADATA_COVERAGE.jsonl",coverage.rows())
    write_rows(OUTPUT/"CURRENT_PAPER_ISSUES.jsonl",paper_issues)
    journal.atomic_json(OUTPUT/"PAPER_IDENTITY_AUDIT.json",audit)
    journal.atomic_json(OUTPUT/"PUBLICATION_REVIEW_BUILD.json",pub)
    db_fp={**cheap(DATABASE),"sha256":sha256(DATABASE)}
    journal.atomic_json(OUTPUT/"CENSUS_BUILD.json",dict(**census_result,database=db_fp))
    guard_native(guards); require(bindings()==code,"frozen code changed"); small_check(plan_fp); small_check(pub_fp)
    state=dict(status="BUILT_NOT_ADOPTED",at=journal.utc_now(),baseline=baseline,protected_native=guards,temporary=cheap(TEMP),
        result=result,code=code,plan=plan_fp,catalog=journal.fingerprint(CATALOG),papers=baseline["current_paper_identities"],terms=baseline["current_entity_terms"],
        database=db_fp,source_record_digests={k:v.hexdigest() for k,v in transform.digests.items()},test_counts=test_counts,
        source_full_sha_verified=True,graph_backups=0,record_preimages_saved=False)
    for key,file in dict(census_build="CENSUS_BUILD.json",audit="PAPER_IDENTITY_AUDIT.json",paper_issues="CURRENT_PAPER_ISSUES.jsonl",
        coverage="CURRENT_METADATA_COVERAGE.jsonl",publication="PUBLICATION_REVIEW_BUILD.json",tests="TEST_RESULTS.xml").items():
        state[key]=journal.fingerprint(OUTPUT/file)
    journal.atomic_json(OUTPUT/"BUILD_STATE.json",state)
    progress("BUILT_NOT_ADOPTED",counts=result["counts"],relations=result["catalog_stats"])


def validate():
    require(not (OUTPUT/"VALIDATED.json").exists(),"already validated; inspect/resume")
    state=journal.read_json(OUTPUT/"BUILD_STATE.json"); baseline=state["baseline"]
    require(bindings()==state["code"] and cheap(TEMP)==state["temporary"],"build/code changed")
    guard_native(state["protected_native"])
    for field in ("plan","catalog","papers","terms","census_build","audit","paper_issues","coverage","publication","tests"): small_check(state[field])
    _,plan=load_plan(baseline); papers,_=authorities(baseline)
    terms=IndexTerms(VerifiedEntityTerms(journal.read_json(state["terms"]["path"])))
    events={e["claim_id"]:e for e in plan["events"]}; edges={e["ordinal"]:e for e in plan["edge_events"]}
    removed={r["node_id"] for r in plan["removed_nodes"]}
    inverse={k:hashlib.sha256() for k in ("nodes","edges")}
    seen_nodes,seen_edges,paper_sigs,shared_ids=set(),set(),set(),set()
    matched=0; issues=[]; outer=[]; coverage=Coverage(); proof=SourceProof(terms,papers)
    pub=journal.read_json(state["publication"]["path"]); pub_rows={r["claim_id"]:r for r in pub["claims"]}; seen_pub=set()
    original_nodes={}; original_refs=defaultdict(list)
    reused_targets=set(plan["target_witnesses"]); source_incidences=defaultdict(list)
    db=sqlite3.connect(DATABASE.as_uri()+"?mode=ro",uri=True)

    def independent(records):
        nonlocal matched
        for kind,key,row in records:
            no_retired_references(kind,key,row,plan)
            original=row
            if kind=="node":
                if key in events:
                    original=reverse_claim(row,events[key]); original_nodes[key]=original; seen_nodes.add(key)
                if key in plan["existing_targets"]: require(digest(row)==plan["existing_targets"][key],"reused node changed")
            elif kind=="edge":
                if int(key) in edges:
                    original=apply_edge(row,edges[int(key)],reverse=True); seen_edges.add(int(key))
                if edge_owner(original) in events: original_refs[edge_owner(original)].append((int(key),original))
            if kind=="node" and key.startswith("CLM:"):
                md_original=original["metadata"]; inner_original=md_original.get("metadata") or {}
                for side in ("subject","object"):
                    nid=md_original.get(side+"_id")
                    if nid in reused_targets:
                        source_incidences[nid].append(dict(node_id=nid,claim_id=key,side=side,name=md_original.get(side+"_name"),
                            claim_sha256=digest(original),
                            outer_type=md_original.get(side+"_type"),inner_type=inner_original.get(side+"_type")))
            elif kind=="edge" and not edge_owner(original):
                require(not {original.get("source_id"),original.get("target_id")}&reused_targets,"unexpected existing nonclaim scope")
            if kind!="metadata":
                coverage.add("node/"+node_class(key,row) if kind=="node" else "edge/all",row)
                if original is not None: inverse[kind+"s"].update(compact(original).encode()+b"\n")
            if kind=="node" and key.startswith("CLM:"):
                md=row["metadata"]; paper=bibliography(md["source_paper"]); sig=digest(paper)
                member=evidence_member(md); rid=relation_id(terms.relation_key(md))
                observed=db.execute("SELECT node_sha,paper_sig,legacy_key,relation_id,shared FROM claims WHERE cid=?",(key,)).fetchone()
                require(observed is not None and observed[:4]==(digest(row),sig,member["paper_key"],rid),"census claim differs")
                if sig not in paper_sigs:
                    ids=identifiers(paper)
                    require(db.execute("SELECT payload,pmid,doi,pmcid,title,year FROM papers WHERE sig=?",(sig,)).fetchone()==
                        (compact(paper),ids["pmid"],ids["doi"],ids["pmcid"],title_key(paper.get("title")),str(paper.get("year") or paper.get("publication_year") or "")),"census paper differs")
                    paper_sigs.add(sig)
                if observed[4]:
                    require(db.execute("SELECT member_json FROM shared_members WHERE cid=?",(key,)).fetchone()==(compact(member),),"shared census differs")
                    shared_ids.add(key)
                for holder in (md,md.get("metadata") or {}):
                    if holder.get("pmid") and str(holder["pmid"])!=str(paper.get("pmid") or ""): outer.append(key)
                answer=papers.resolve(md)
                if answer["status"]=="conflict": issues.append(issue_row(key,row,answer))
                if key in pub_rows:
                    require(publication_row(pub_rows[key],row,row,papers)==pub_rows[key],"publication identity/hash differs"); seen_pub.add(key)
                matched+=1
            yield kind,key,row

    progress("INDEPENDENT_FULL_CANDIDATE_INVERSE_CENSUS_AND_COVERAGE")
    try:
        with hashed_reader(TEMP) as (reader,h):
            checks=verify_catalog_and_graph(proof.records(independent(walk_graph(reader))),state["result"],CATALOG,progress,
                relation_key_func=terms.relation_key,papers=papers)
            candidate_sha=h.hexdigest()
        census=journal.read_json(state["census_build"]["path"])
        require(matched==db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]==baseline["counts"]["claims"],"claim census closure differs")
        require(len(paper_sigs)==db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],"stale census paper")
        require(len(shared_ids)==db.execute("SELECT COUNT(*) FROM shared_members").fetchone()[0],"stale census shared row")
        require(shared_ids==set(plan["expected_shared_claim_ids"])=={m["claim_id"] for g in rows(CATALOG) for m in g["members"]},"shared membership differs")
        require(not outer and not census["outer_pmid_conflicts"] and census_identifier_projection(db)==census["projection"],"paper normalization differs")
        require(db.execute("PRAGMA integrity_check").fetchone()[0]=="ok","census integrity failed")
    finally: db.close()
    require(seen_nodes==set(events) and seen_edges==set(edges),"repair closure differs")
    require({k:v.hexdigest() for k,v in inverse.items()}==state["source_record_digests"],"inverse retained-original-record digests differ")
    reproduced=[]
    for cid,row in original_nodes.items():
        out=reviewed_claim(row,events[cid],plan)
        require(relation_id(terms.relation_key(row["metadata"]))==events[cid]["old_relation_id"]
            and relation_id(terms.relation_key(out["metadata"]))==events[cid]["new_relation_id"],"independent relation identity differs")
        reproduced.extend(reviewed_edges(cid,row,out,original_refs[cid]))
    require(sorted(reproduced,key=lambda e:e["ordinal"])==plan["edge_events"],"independent complete reference closure differs")
    require(ordered_incidences([r for nid in reused_targets for r in source_incidences[nid]])==plan["source_incidences"],
        "independent full existing incidence scope differs")
    require(issues==rows(state["paper_issues"]["path"]) and seen_pub==set(pub_rows),"issues/publication coverage differs")
    require(coverage.rows()==rows(state["coverage"]["path"]),"independent full metadata coverage differs")
    proof.verify(state["result"],journal.read_json(state["audit"]["path"]))
    progress("FULL_DETAIL_AND_CENSUS_SHA_BOUNDARY")
    require(sha256(Path(baseline["current_detail_store"]["path"]))==baseline["current_detail_store"]["sha256"],"detail SHA differs")
    require(sha256(DATABASE)==state["database"]["sha256"],"candidate census SHA differs")
    progress("FULL_PROTECTED_FORMAL_SHA_BOUNDARY")
    for fp in baseline["formal_sources"].values():
        require(sha256(Path(fp["path"]))==fp["sha256"],"protected formal SHA differs")
    guard_native(state["protected_native"]); require(bindings()==state["code"],"frozen code changed")
    checks.update(verified_identity_proofs_complete=True,paper_identity_witnesses_validated=True,publication_status_witnesses_validated=True,
        finite_nominal_name_consolidation_and_current_references_reproduced=True,complete_existing_target_incidence_scope_reproduced=True,
        no_new_concept_nodes=True,no_retained_concept_metadata_change=True,no_inferred_literal_semantic_type=True,
        exact_approved_literal_endpoint_transform=True,inverse_reproduces_all_retained_source_node_and_edge_record_digests=True,
        complete_case_sensitive_names_preserved=True,existing_gene_and_protein_nodes_unchanged=True,
        original_quotes_negation_conditions_independent_sources_preserved=True,no_metadata_fields_added=True,
        current_census_all_claims_papers_and_shared_members_verified=True,metadata_coverage_full_scan_verified=True,
        no_claim_or_edge_deletion=True,no_new_scientific_assertion_edges=True,authority_verified_claims=proof.claim_status["verified"],
        removed_concepts=2,retired_exact_references_absent=True,six_detail_source_anchors_preserved=True,
        closed_night_window_unchanged=True,formal_sources_full_sha_verified=True)
    journal.atomic_json(OUTPUT/"VALIDATED.json",dict(status="VALIDATED_NOT_ADOPTED",at=journal.utc_now(),graph={**cheap(TEMP),"sha256":candidate_sha},
        temporary_native=native_info(TEMP),database_native=native_info(DATABASE),checks=checks,
        build_state=journal.fingerprint(OUTPUT/"BUILD_STATE.json"),metadata_coverage_rows=len(coverage.rows())))
    progress("VALIDATED_NOT_ADOPTED",counts=state["result"]["counts"],relations=state["result"]["catalog_stats"])


def apply():
    require(not (OUTPUT/"CURRENT_ACCEPTANCE.json").exists(),"already adopted")
    state=journal.read_json(OUTPUT/"BUILD_STATE.json"); accepted=journal.read_json(OUTPUT/"VALIDATED.json")
    small_check(accepted["build_state"]); require(bindings()==state["code"],"frozen code changed")
    guard_native([*state["protected_native"],accepted["temporary_native"],accepted["database_native"]])
    for field in ("plan","catalog","papers","terms","census_build","audit","paper_issues","coverage","publication","tests"): small_check(state[field])
    baseline=state["baseline"]; campaign=journal.read_json(journal.OUTPUT/"CAMPAIGN.json"); _,plan=load_plan(baseline)
    require(campaign["current_graph"]==baseline["current_graph"] and campaign["active_process"]["kind"]=="nominal_consolidation" and campaign["active_process"]["pid"]==os.getpid(),"writer/source advanced")
    require(TEMP.resolve().parent==SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()),"unsafe replacement target")
    held_files={}
    for field,filename in (("current_issues","CURRENT_REMAINING_ISSUES.jsonl"),("current_structure_holds","CURRENT_STRUCTURE_HOLDS.jsonl")):
        small_check(baseline[field]); held_files[filename]=rows(baseline[field]["path"])
        require(not {e["claim_id"] for e in plan["events"]}&{r["claim_id"] for r in held_files[filename]},"scientific hold overlap")
    small_check(baseline["current_scope_findings"])
    scope=journal.read_json(baseline["current_scope_findings"]["path"])
    require(not set(ch["claim_id"] for ch in plan["events"]) & set(scope["current_claim_hashes"]),"scientific scope overlap")
    small_check(plan["closed_night_window"])
    os.replace(TEMP,SOURCE)
    graph={**cheap(SOURCE),"sha256":accepted["graph"]["sha256"]}
    require(graph["bytes"]==accepted["graph"]["bytes"] and graph["mtime_ns"]==accepted["graph"]["mtime_ns"],"replacement differs")
    for filename,held in held_files.items(): write_rows(OUTPUT/filename,[dict(r,current_graph=graph,version_binding_status="FULL_CURRENT_GRAPH_VALIDATED") for r in held])
    remaining=rows(plan["remaining_queue"]["path"])
    changed={e["claim_id"]:e for e in plan["events"]}
    for row in remaining:
        if row["claim_id"] in changed: row["claim_sha256"]=changed[row["claim_id"]]["current_node_sha256"]
    write_rows(OUTPUT/"CURRENT_GENE_ENDPOINT_HOLDS.jsonl",remaining)
    scope.update(at=journal.utc_now(),graph=graph)
    journal.atomic_json(OUTPUT/"CURRENT_SCOPE_FINDINGS.json",scope)
    census=journal.read_json(state["census_build"]["path"]); projection=census.pop("projection")
    journal.atomic_json(OUTPUT/"CENSUS.json",dict(**census,status="CURRENT_CENSUS_COMPLETE",at=journal.utc_now(),graph=graph,
        source_full_sha_verified=True,independent_full_candidate_verified=True,code=state["code"],record_preimages_saved=False))
    census_fp=journal.fingerprint(OUTPUT/"CENSUS.json")
    journal.atomic_json(OUTPUT/"CENSUS_NORMALIZATION.json",dict(**projection,census=census_fp))
    pub=journal.read_json(state["publication"]["path"]); pub.update(at=journal.utc_now(),graph=graph,current_census=census_fp)
    journal.atomic_json(OUTPUT/"CURRENT_PUBLICATION_REVIEW.json",pub)
    result=state["result"]
    changed_counts=dict(claim_nodes=plan["changed_claims"],endpoint_ids=plan["changed_endpoints"],edge_records=plan["changed_edges"],
        new_literal_nodes=0,reused_existing_nodes=7,removed_concept_nodes=2,preserved_original_source_nodes=6,repaired_queued_endpoints=14)
    receipt=dict(status="CURRENT_NOMINAL_CONSOLIDATION_APPLIED",relation_grouping=POLICY,at=journal.utc_now(),graph=graph,counts=result["counts"],changed=changed_counts,
        connected_components=result["connected_components"],isolated_nodes=result["isolated_nodes"],self_loops=result["self_loops"],
        metadata_key_unions=result["metadata_key_unions"],metadata=result["metadata"],source_graph=baseline["current_graph"],source_kg_retained=False,
        shared_relations=state["catalog"],entity_terms=state["terms"],paper_identities=state["papers"],paper_issues=state["paper_issues"],metadata_coverage=state["coverage"],
        current_paper_census=census_fp,relation_counts=result["catalog_stats"],checks=accepted["checks"],tests=state["tests"],test_counts=state["test_counts"],code=state["code"],
        detail_store=baseline["current_detail_store"],identity_audit=state["audit"],provenance_plan=state["plan"],
        current_gene_holds=journal.fingerprint(OUTPUT/"CURRENT_GENE_ENDPOINT_HOLDS.jsonl"),rollback_retention=False,record_preimages_saved=False,formal_apply_performed=False)
    journal.atomic_json(OUTPUT/"CURRENT_ACCEPTANCE.json",receipt)
    journal.atomic_json(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json",dict(status="CURRENT_RUNTIME_VALIDATED",at=receipt["at"],graph=graph,
        acceptance=journal.fingerprint(OUTPUT/"CURRENT_ACCEPTANCE.json"),code=state["code"],tests=state["tests"],test_counts=state["test_counts"]))
    campaign.update(status="COMPLETED",phase="R65手动续作验收：七组同名归并、36处端点、两处节点退役",active_process=None,updated_at=receipt["at"],current_graph=graph,
        counts=result["counts"],node_metadata_field_union=len(result["metadata_key_unions"]["nodes"]),edge_metadata_field_union=len(result["metadata_key_unions"]["edges"]),
        current_acceptance=journal.fingerprint(OUTPUT/"CURRENT_ACCEPTANCE.json"),current_runtime_acceptance=journal.fingerprint(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT/"CURRENT_REMAINING_ISSUES.jsonl"),current_structure_holds=journal.fingerprint(OUTPUT/"CURRENT_STRUCTURE_HOLDS.jsonl"),
        current_shared_relations=state["catalog"],current_paper_issues=state["paper_issues"],current_coverage=state["coverage"],current_paper_census=census_fp,
        current_census_normalization=journal.fingerprint(OUTPUT/"CENSUS_NORMALIZATION.json"),relation_evidence_counts=result["catalog_stats"],
        current_scope_findings=journal.fingerprint(OUTPUT/"CURRENT_SCOPE_FINDINGS.json"),
        current_gene_holds=receipt["current_gene_holds"],pending_gene_link_claims=len({r["claim_id"] for r in remaining}),pending_gene_link_endpoint_events=len(remaining),
        last_deep_verification=accepted["at"],last_current_graph_content_verification=dict(graph=graph,independent_full_scan=True),
        continuation_status="MANUAL_R65_ACCEPTED_OLD_NIGHT_REMAINS_CLOSED",
        gene_review_queue_interpretation="扩大后的待核队列，包含R42词面筛查候选，不是已确认错误数",
        next_steps=["七组同名的科学端点已统一；6个原始详情锚点保留、2个无引用节点退役。",
            "剩余2190处端点是待核队列，不全是确认错误；历史449剩1，旧语义14及科学范围6不变。",
            "两条关系词与存储原文的疑点单独待核；不等于本批重新批准科学语义。",
            "无模型、训练或full_v2同步；旧夜间窗口保持关闭，无新目标或定时任务。"])
    summary=dict(graph=graph,original_watchlist=449,original_watchlist_remaining_before=1,
        original_watchlist_repaired_this_batch=plan["original_watchlist_449_repaired"],
        expanded_repaired_this_batch=plan["repaired_queued_endpoints"],
        original_watchlist_remaining=plan["original_watchlist_449_remaining_after"],
        expanded_remaining=len(remaining),expanded_claims=len({r["claim_id"] for r in remaining}),
        not_all_confirmed_errors=True,semantic_holds=baseline["additional_semantic_or_detail_review_candidates"])
    journal.atomic_json(OUTPUT/"CURRENT_REVIEW_QUEUE_SUMMARY.json",summary)
    campaign["current_gene_review_summary"]=journal.fingerprint(OUTPUT/"CURRENT_REVIEW_QUEUE_SUMMARY.json")
    journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",campaign)
    journal.atomic_json(OUTPUT/"RUN_STATE.json",dict(status="COMPLETED",at=receipt["at"],graph=graph,changed=changed_counts,relations=result["catalog_stats"]))
    print(compact(dict(status=receipt["status"],changed=changed_counts,relations=result["catalog_stats"])),flush=True)


def resume():
    state=journal.read_json(OUTPUT/"BUILD_STATE.json")
    require(bindings()==state["code"] and not (OUTPUT/"CURRENT_ACCEPTANCE.json").exists(),"not resumable")
    c=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(c["current_graph"]==state["baseline"]["current_graph"] and c["active_process"]["kind"]=="nominal_consolidation","unexpected active batch")
    require_process_ended(c["active_process"]["pid"])
    c["active_process"]["pid"]=os.getpid(); journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",c)
    if not (OUTPUT/"VALIDATED.json").exists(): validate()
    apply()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("phase",choices=("run","build","resume"))
    phase=parser.parse_args().phase
    if phase=="run": build(); validate(); apply()
    elif phase=="build": build()
    else: resume()


