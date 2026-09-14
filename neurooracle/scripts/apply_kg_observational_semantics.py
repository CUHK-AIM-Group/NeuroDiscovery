"""R59 three qualified observations, all original-record inverse proof."""
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
from apply_kg_scientific_definition_repair import FILES as PREVIOUS_FILES
from plan_kg_observational_semantics import inspection_bridge, source_inputs
from apply_kg_paper_identity import SourceProof
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows, guard_native
from reclaim_kg_backup_storage import native_info, sha256
from resume_kg_bibliography_titles import require_process_ended
from neurooracle.src.kg_identity_pilot import digest, nonidentity_claim
from neurooracle.src.kg_observational_semantics import (literal_node, reviewed_claim,
    reviewed_edges, edge_owner, apply_edge, reverse_claim, change_claim)
from neurooracle.src.kg_paper_identity import bibliography, identifiers, title_key, census_identifier_projection
from neurooracle.src.metadata_field_audit import Coverage, node_class
from neurooracle.src.relation_evidence import evidence_member, relation_id
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
from neurooracle.src.correlation_grouping import POLICY, IndexTerms
from datetime import datetime, timezone

OUTPUT=journal.OUTPUT/"round59_observational_semantics"
SOURCE=journal.OUTPUT/"round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP=SOURCE.with_name(SOURCE.name+".observational-semantics.tmp")
CATALOG=OUTPUT/"CURRENT_SHARED_RELATIONS.jsonl"
DATABASE=OUTPUT/"CURRENT_PAPER_CENSUS.sqlite"
FILES=list(dict.fromkeys([*PREVIOUS_FILES,Path(__file__),*[journal.REPO/p for p in (
    "neurooracle/src/kg_observational_semantics.py","neurooracle/scripts/plan_kg_observational_semantics.py",
    "neurooracle/scripts/inspect_kg_observational_semantics.py",
    "neurooracle/tests/test_kg_observational_semantics.py","neurooracle/tests/test_kg_observational_semantics_plan.py",
    "neurooracle/tests/test_kg_observational_semantics_pipeline.py",
    "neurooracle/scripts/report_kg_scientific_definition_repair.py","neurooracle/tests/test_kg_scientific_definition_report.py")]]))


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
    require(counts["tests"]==len(seen)>=1240 and not any(counts[k] for k in ("failures","errors","skipped")),"complete distinct passing regression required")
    return dict(counts)


class LiteralTransform:
    """Forward scan keeps the original-record digest; inverse scan must reproduce it."""
    def __init__(self,plan):
        self.plan=plan
        self.events={e["claim_id"]:e for e in plan["events"]}
        self.edges={e["ordinal"]:e for e in plan["edge_events"]}
        self.new={r["id"]:literal_node(r["name"]) for r in plan["new_literals"]}
        require(all(digest(self.new[r["id"]])==r["node_sha256"] for r in plan["new_literals"]),"literal definition changed")
        require(len(self.events)==len(plan["events"]) and len(self.edges)==len(plan["edge_events"]),"duplicate reviewed identity")
        self.digests={k:hashlib.sha256() for k in ("nodes","edges")}
        self.originals={}; self.refs=defaultdict(list); self.seen_edges=set(); self.seen_targets=set()
        self.appended=False

    def records(self,records):
        for kind,key,row in records:
            if kind=="edge" and not self.appended:
                for nid,node in sorted(self.new.items()): yield "node",nid,node
                self.appended=True
            if kind!="metadata": self.digests[kind+"s"].update(compact(row).encode()+b"\n")
            out=row
            if kind=="node":
                require(key not in self.new,"new literal ID collides")
                if key in self.plan["existing_targets"]:
                    require(digest(row)==self.plan["existing_targets"][key],"existing target changed")
                    self.seen_targets.add(key)
                if key in self.events:
                    event=self.events[key]; out=change_claim(row,event)
                    reproduced,_=reviewed_claim(row,event["changes"],self.plan["public_source_proof"],self.plan["claim_reviews"])
                    require(all(event[k]==v for k,v in reproduced.items()),"claim review did not reproduce")
                    self.originals[key]=row
            elif kind=="edge":
                owner=edge_owner(row)
                if owner in self.events: self.refs[owner].append((int(key),row))
                if int(key) in self.edges:
                    out=apply_edge(row,self.edges[int(key)]); self.seen_edges.add(int(key))
            yield kind,key,out
        require(self.appended,"source has no edge boundary")
        require(set(self.originals)==set(self.events) and self.seen_edges==set(self.edges),"changed record scope incomplete")
        require(self.seen_targets==set(self.plan["existing_targets"]),"target witnesses incomplete")
        reproduced=[]
        for cid,row in self.originals.items():
            reproduced.extend(reviewed_edges(cid,row,change_claim(row,self.events[cid]),self.refs[cid]))
        require(sorted(reproduced,key=lambda e:e["ordinal"])==self.plan["edge_events"],"complete owned edge review differs")


def issue_row(key,row,answer):
    return dict(claim_id=key,claim_sha256=digest(row),reasons=answer["reasons"],
        legacy_source_key=evidence_member(row["metadata"])["paper_key"],action="hold_no_claim_or_source_deletion")


def load_plan(baseline):
    from neurooracle.src.kg_observational_semantics import SPECS,remaining_issues
    fp=journal.fingerprint(OUTPUT/"PLAN.json");plan=journal.read_json(fp["path"])
    require(plan["status"]=="REVIEWED_NOT_APPLIED" and plan["version"]=="kg.observational_semantics.v1"
        and plan["relation_grouping"]==POLICY,"wrong observational correction policy")
    require(plan["graph"]==baseline["current_graph"] and plan["source_acceptance"]==baseline["current_acceptance"]
        and plan["detail_store"]==baseline["current_detail_store"],"plan baseline differs")
    for item in [*plan["code"],*[plan[k] for k in ("remaining_queue","remaining_semantic_issues","source_review","prior_identity_plan","existing_incidence_evidence")],
        *plan["public_manifests"],plan["public_abstracts"]]:small_check(item)
    review_fp,inspection,reviews,proof=source_inputs()
    require(review_fp==plan["source_review"] and inspection["pending_disjoint_plan"]==plan["prior_identity_plan"],"source binding differs")
    inspection_bridge(inspection,baseline,journal.read_json(baseline["current_acceptance"]["path"]),journal.read_json(plan["prior_identity_plan"]["path"]))
    require([inspection["public_sources"]]==plan["public_manifests"] and inspection["public_response"]==plan["public_abstracts"]
        and proof==plan["public_source_proof"],"own qualified abstract proof differs")
    require({cid:reviews[cid] for cid in SPECS}==plan["claim_reviews"],"finite current reviews changed")
    require({e["claim_id"] for e in plan["events"]}==set(SPECS) and len(plan["events"])==plan["changed_claims"]==3,"finite claim scope differs")
    require(not any(e["changes"] for e in plan["events"]) and plan["changed_endpoints"]==0,"no endpoint changes permitted")
    require(plan["repaired_queued_endpoints"]==0 and plan["held_endpoints"]==plan["expanded_review_endpoints"]==2204
        and plan["original_watchlist_449_repaired"]==0 and plan["original_watchlist_449_remaining_after"]==1,"queue scope differs")
    require(plan["added_literal_nodes"]==len(plan["new_literals"])==plan["reused_existing_nodes"]==0 and not plan["reused_target_ids"],"no concept changes permitted")
    require(remaining_issues(rows(baseline["current_issues"]["path"]),plan)==rows(plan["remaining_semantic_issues"]["path"])
        and plan["remaining_semantic_holds"]==14 and plan["resolved_semantic_holds"]==3,"semantic remaining set differs")
    require(not set(SPECS)&{r["claim_id"] for r in rows(baseline["current_gene_holds"]["path"])}
        and rows(plan["remaining_queue"]["path"])==rows(baseline["current_gene_holds"]["path"]),"gene holds must remain identical")
    require(not set(SPECS)&set(journal.read_json(baseline["current_scope_findings"]["path"])["current_claim_hashes"]),"unexpected other scientific hold overlap")
    require(plan["source_full_sha_verified"] and plan["source_census_full_sha_verified"],"source proof incomplete")
    return fp,plan


def build():
    require(not any(p.exists() for p in (TEMP,CATALOG,DATABASE,OUTPUT/"BUILD_STATE.json")),"build exists; inspect/resume")
    baseline=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(baseline["status"]=="COMPLETED" and baseline["active_process"] is None and not baseline["rollback_retention"],"writer/retention boundary")
    require(Path(baseline["current_graph"]["path"]).resolve()==SOURCE.resolve(),"wrong source graph")
    window=journal.read_json(baseline["night_window"])
    require(window["status"]=="ACTIVE" and datetime.now(timezone.utc)<datetime.fromisoformat(window["end_at_utc"].replace("Z","+00:00")),"night window closed; do not start another writer")
    require(shutil.disk_usage(SOURCE).free>baseline["current_graph"]["bytes"]+12*1024**3,"insufficient atomic space")
    test_counts=test_evidence()
    for field in ("current_acceptance","current_runtime_acceptance","current_entity_terms","current_shared_relations","current_coverage",
        "current_issues","current_structure_holds","current_paper_issues","current_paper_census"):
        small_check(baseline[field])
    old_receipt=journal.read_json(baseline["current_acceptance"]["path"])
    for fp in old_receipt["code"]: small_check(fp)
    plan_fp,plan=load_plan(baseline)
    events={e["claim_id"]:e for e in plan["events"]}
    require(not set(events)&{r["claim_id"] for r in rows(baseline["current_structure_holds"]["path"])},"structural hold overlaps")
    papers,_=authorities(baseline)
    terms=IndexTerms(VerifiedEntityTerms(journal.read_json(baseline["current_entity_terms"]["path"])))
    old_census=journal.read_json(baseline["current_paper_census"]["path"])
    prior_audit=journal.read_json(old_receipt["identity_audit"]["path"])
    pub_path=Path(baseline["current_acceptance"]["path"]).parent/"CURRENT_PUBLICATION_REVIEW.json"
    pub_fp=journal.fingerprint(pub_path); pub=journal.read_json(pub_path)
    require(pub["graph"]==baseline["current_graph"] and pub["registry"]==baseline["current_paper_identities"],"publication baseline differs")
    pub_rows={r["claim_id"]:r for r in pub["claims"]}; current_pub=[]
    guards=[native_info(fp["path"]) for fp in (baseline["current_graph"],baseline["current_detail_store"],old_census["database"],*baseline["formal_sources"].values())]
    journal.guards([baseline["current_graph"],baseline["current_detail_store"],old_census["database"],baseline["formal_sources"]])
    code=bindings()
    campaign=dict(baseline,status="MANUAL_ACTIVE",phase="R59三条观察性谓词及研究设计限定：独立全图验收",updated_at=journal.utc_now(),
        active_process=dict(kind="observational_semantics",pid=os.getpid(),state=str(OUTPUT/"RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",campaign)
    night=journal.read_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json"); night["active_process"]=campaign["active_process"]
    journal.atomic_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json",night)
    transform=LiteralTransform(plan); proof=SourceProof(terms,papers); coverage=Coverage()
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

    progress("BUILD_OBSERVATIONAL_PREDICATE_AND_DESIGN",claims=plan["changed_claims"],new_literals=plan["added_literal_nodes"])
    try:
        with TEMP.open("xb",buffering=1024**2) as handle:
            with hashed_reader(SOURCE) as (reader,h):
                result=stream_patch(proof.records(observed(transform.records(walk_graph(reader)))),handle,[],{},CATALOG,progress,
                    relation_key_func=terms.relation_key,relation_version="kg.relation_evidence.v4",papers=papers,
                    extra_graph_metadata=dict(relation_grouping=POLICY))
                require(h.hexdigest()==baseline["current_graph"]["sha256"],"full source SHA differs")
            handle.flush(); os.fsync(handle.fileno())
        expected_counts=dict(baseline["counts"]); expected_counts["nodes"]+=plan["added_literal_nodes"]
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
    new={r["id"]:r for r in plan["new_literals"]}
    inverse={k:hashlib.sha256() for k in ("nodes","edges")}
    seen_nodes,seen_edges,seen_new,paper_sigs,shared_ids=set(),set(),set(),set(),set()
    matched=0; issues=[]; outer=[]; coverage=Coverage(); proof=SourceProof(terms,papers)
    pub=journal.read_json(state["publication"]["path"]); pub_rows={r["claim_id"]:r for r in pub["claims"]}; seen_pub=set()
    original_nodes={}; original_refs=defaultdict(list)
    reused_targets=set(plan["reused_target_ids"]); source_incidences=defaultdict(list)
    db=sqlite3.connect(DATABASE.as_uri()+"?mode=ro",uri=True)

    def independent(records):
        nonlocal matched
        for kind,key,row in records:
            original=row
            if kind=="node":
                if key in new:
                    require(row==literal_node(new[key]["name"]) and digest(row)==new[key]["node_sha256"],"literal node differs")
                    seen_new.add(key); original=None
                elif key in events:
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
                            claim_sha256=digest(original),nonidentity_sha256=digest(nonidentity_claim(original)),
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
    require(seen_nodes==set(events) and seen_edges==set(edges) and seen_new==set(new),"repair closure differs")
    require({k:v.hexdigest() for k,v in inverse.items()}==state["source_record_digests"],"inverse original-record digests differ")
    reproduced=[]
    for cid,row in original_nodes.items():
        event,out=reviewed_claim(row,events[cid]["changes"],plan["public_source_proof"],plan["claim_reviews"])
        require(all(events[cid][k]==v for k,v in event.items()),"independent public-scientific field gate differs")
        reproduced.extend(reviewed_edges(cid,row,out,original_refs[cid]))
    require(sorted(reproduced,key=lambda e:e["ordinal"])==plan["edge_events"],"independent complete reference closure differs")
    require([r for nid in sorted(reused_targets) for r in source_incidences[nid]]==rows(plan["existing_incidence_evidence"]["path"]),
        "independent full existing incidence scope differs")
    require(issues==rows(state["paper_issues"]["path"]) and seen_pub==set(pub_rows),"issues/publication coverage differs")
    require(coverage.rows()==rows(state["coverage"]["path"]),"independent full metadata coverage differs")
    proof.verify(state["result"],journal.read_json(state["audit"]["path"]))
    progress("FULL_DETAIL_AND_CENSUS_SHA_BOUNDARY")
    require(sha256(Path(baseline["current_detail_store"]["path"]))==baseline["current_detail_store"]["sha256"],"detail SHA differs")
    require(sha256(DATABASE)==state["database"]["sha256"],"candidate census SHA differs")
    guard_native(state["protected_native"]); require(bindings()==state["code"],"frozen code changed")
    checks.update(verified_identity_proofs_complete=True,paper_identity_witnesses_validated=True,publication_status_witnesses_validated=True,
        finite_own_source_observational_field_corrections_reproduced=True,complete_existing_target_incidence_scope_reproduced=True,
        no_existing_concept_metadata_change=True,no_inferred_literal_semantic_type=True,
        exact_approved_observational_semantics_transform=True,inverse_reproduces_all_source_node_and_edge_record_digests=True,
        complete_case_sensitive_names_preserved=True,existing_gene_and_protein_nodes_unchanged=True,
        original_quotes_negation_conditions_independent_sources_preserved=True,all_original_statistics_and_historic_audit_records_preserved=True,
        old_scientific_audit_not_revalidated=True,no_metadata_fields_added=True,no_new_concept_nodes=True,
        current_census_all_claims_papers_and_shared_members_verified=True,metadata_coverage_full_scan_verified=True,
        no_record_deletion=True,no_new_scientific_assertion_edges=True,authority_verified_claims=proof.claim_status["verified"])
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
    require(campaign["current_graph"]==baseline["current_graph"] and campaign["active_process"]["kind"]=="observational_semantics" and campaign["active_process"]["pid"]==os.getpid(),"writer/source advanced")
    require(TEMP.resolve().parent==SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()),"unsafe replacement target")
    from neurooracle.src.kg_observational_semantics import remaining_issues,SPECS
    small_check(baseline["current_issues"]);small_check(baseline["current_structure_holds"])
    current_issues=remaining_issues(rows(baseline["current_issues"]["path"]),plan)
    require(current_issues==rows(plan["remaining_semantic_issues"]["path"]),"current finite semantic difference changed")
    held_files={"CURRENT_REMAINING_ISSUES.jsonl":current_issues,"CURRENT_STRUCTURE_HOLDS.jsonl":rows(baseline["current_structure_holds"]["path"])}
    require(not held_files["CURRENT_STRUCTURE_HOLDS.jsonl"],"unexpected structure holds")
    small_check(baseline["current_scope_findings"])
    scope=journal.read_json(baseline["current_scope_findings"]["path"])
    require(not set(SPECS)&set(scope["current_claim_hashes"]) and len(scope["findings"])==6,"unapproved scope overlap")
    scope["observational_fields_resolved"]=sorted(SPECS)
    scope["old_audits_not_revalidated"]=True
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
    scope["latest_observational_semantics_plan"]=state["plan"]
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
        new_literal_nodes=plan["added_literal_nodes"],reused_existing_nodes=plan["reused_existing_nodes"])
    receipt=dict(status="CURRENT_OBSERVATIONAL_SEMANTICS_APPLIED",relation_grouping=POLICY,at=journal.utc_now(),graph=graph,counts=result["counts"],changed=changed_counts,
        connected_components=result["connected_components"],isolated_nodes=result["isolated_nodes"],self_loops=result["self_loops"],
        metadata_key_unions=result["metadata_key_unions"],metadata=result["metadata"],source_graph=baseline["current_graph"],source_kg_retained=False,
        shared_relations=state["catalog"],entity_terms=state["terms"],paper_identities=state["papers"],paper_issues=state["paper_issues"],metadata_coverage=state["coverage"],
        current_paper_census=census_fp,relation_counts=result["catalog_stats"],checks=accepted["checks"],tests=state["tests"],test_counts=state["test_counts"],code=state["code"],
        detail_store=baseline["current_detail_store"],identity_audit=state["audit"],provenance_plan=state["plan"],
        current_gene_holds=journal.fingerprint(OUTPUT/"CURRENT_GENE_ENDPOINT_HOLDS.jsonl"),rollback_retention=False,record_preimages_saved=False,formal_apply_performed=False)
    journal.atomic_json(OUTPUT/"CURRENT_ACCEPTANCE.json",receipt)
    journal.atomic_json(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json",dict(status="CURRENT_RUNTIME_VALIDATED",at=receipt["at"],graph=graph,
        acceptance=journal.fingerprint(OUTPUT/"CURRENT_ACCEPTANCE.json"),code=state["code"],tests=state["tests"],test_counts=state["test_counts"]))
    campaign.update(status="COMPLETED",phase="R59三条观察性科学表达及研究设计已修正验收",active_process=None,updated_at=receipt["at"],current_graph=graph,
        counts=result["counts"],node_metadata_field_union=len(result["metadata_key_unions"]["nodes"]),edge_metadata_field_union=len(result["metadata_key_unions"]["edges"]),
        current_acceptance=journal.fingerprint(OUTPUT/"CURRENT_ACCEPTANCE.json"),current_runtime_acceptance=journal.fingerprint(OUTPUT/"CURRENT_RUNTIME_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT/"CURRENT_REMAINING_ISSUES.jsonl"),current_structure_holds=journal.fingerprint(OUTPUT/"CURRENT_STRUCTURE_HOLDS.jsonl"),
        current_shared_relations=state["catalog"],current_paper_issues=state["paper_issues"],current_coverage=state["coverage"],current_paper_census=census_fp,
        current_census_normalization=journal.fingerprint(OUTPUT/"CENSUS_NORMALIZATION.json"),relation_evidence_counts=result["catalog_stats"],
        current_scope_findings=journal.fingerprint(OUTPUT/"CURRENT_SCOPE_FINDINGS.json"),
        current_gene_holds=receipt["current_gene_holds"],pending_gene_link_claims=len({r["claim_id"] for r in remaining}),pending_gene_link_endpoint_events=len(remaining),
        last_deep_verification=accepted["at"],last_current_graph_content_verification=dict(graph=graph,independent_full_scan=True),
        continuation_status="R59_ACCEPTED_CONTINUE_QUALIFIED_SEMANTICS_REVIEW",
        additional_semantic_or_detail_review_candidates=plan["remaining_semantic_holds"],
        gene_review_queue_interpretation="扩大后的待核队列，包含R42词面筛查候选，不是已确认错误数",
        next_steps=["3条功能改善/续治观察现用关联与方向及既有设计字段表达；原始文字和旧审核未改。",
            "旧语义17减3为14；历史449仍剩1，扩大队列2204、科学范围6不变。",
            "不得把组内变化当组间效果、把急性期随机性传给续治阶段；无模型/训练/full_v2同步。"])
    summary=dict(graph=graph,original_watchlist=449,original_watchlist_remaining_before=1,
        original_watchlist_repaired_this_batch=plan["original_watchlist_449_repaired"],
        expanded_repaired_this_batch=plan["repaired_queued_endpoints"],
        original_watchlist_remaining=plan["original_watchlist_449_remaining_after"],
        expanded_remaining=len(remaining),expanded_claims=len({r["claim_id"] for r in remaining}),
        not_all_confirmed_errors=True,semantic_holds=plan["remaining_semantic_holds"])
    journal.atomic_json(OUTPUT/"CURRENT_REVIEW_QUEUE_SUMMARY.json",summary)
    campaign["current_gene_review_summary"]=journal.fingerprint(OUTPUT/"CURRENT_REVIEW_QUEUE_SUMMARY.json")
    journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",campaign)
    night=journal.read_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json")
    night.update(active_process=None,latest_acceptance=campaign["current_acceptance"],latest_census=census_fp,latest_completed_batch="R59",
        latest_completed_runtime_batch="R59",latest_runtime_acceptance=campaign["current_runtime_acceptance"])
    journal.atomic_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json",night)
    journal.atomic_json(OUTPUT/"RUN_STATE.json",dict(status="COMPLETED",at=receipt["at"],graph=graph,changed=changed_counts,relations=result["catalog_stats"]))
    print(compact(dict(status=receipt["status"],changed=changed_counts,relations=result["catalog_stats"])),flush=True)


def resume():
    state=journal.read_json(OUTPUT/"BUILD_STATE.json")
    require(bindings()==state["code"] and not (OUTPUT/"CURRENT_ACCEPTANCE.json").exists(),"not resumable")
    c=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(c["current_graph"]==state["baseline"]["current_graph"] and c["active_process"]["kind"]=="observational_semantics","unexpected active batch")
    require_process_ended(c["active_process"]["pid"])
    c["active_process"]["pid"]=os.getpid(); journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",c)
    night=journal.read_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json"); night["active_process"]=c["active_process"]
    journal.atomic_json(journal.OUTPUT/"NIGHT_WINDOW_20260910.json",night)
    if not (OUTPUT/"VALIDATED.json").exists(): validate()
    apply()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("phase",choices=("run","build","resume"))
    phase=parser.parse_args().phase
    if phase=="run": build(); validate(); apply()
    elif phase=="build": build()
    else: resume()


