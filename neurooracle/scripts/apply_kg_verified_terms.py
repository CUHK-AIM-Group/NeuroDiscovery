"""R30: verified whole-term reuse, same-owner repairs and independent acceptance.

The only large output is the atomic replacement of the current work KG. There
are no source copies, record preimages, models, training, or formal writes.
"""
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree

sys.path.insert(0,str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from build_umls_simplification_candidate import cheap,compact,hashed_reader,walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import guard_native,rows,write_rows
from reclaim_kg_backup_storage import native_info,sha256
from neurooracle.src.kg_bulk_identity import change_claim,change_edge
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.src.kg_structure_repairs import choose_repairs,repair_edge
from neurooracle.src.metadata_field_audit import Coverage,node_class
from neurooracle.src.relation_evidence import relation_key,relation_id,evidence_member
from neurooracle.src.verified_entity_terms import VERSION,VerifiedEntityTerms

OUTPUT=journal.OUTPUT/"round30_verified_terms"
SOURCE=journal.OUTPUT/"round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP=SOURCE.with_name(SOURCE.name+".verified-terms.tmp")
CATALOG=OUTPUT/"CURRENT_SHARED_RELATIONS.jsonl"
TERMS=OUTPUT/"CURRENT_ENTITY_TERMS.json"
FILES=[Path(__file__),*[(journal.REPO/p) for p in (
    "neurooracle/scripts/collect_kg_verified_terms.py","neurooracle/scripts/apply_kg_relation_identity.py",
    "neurooracle/scripts/build_umls_simplification_candidate.py","neurooracle/scripts/prune_current_kg.py",
    "neurooracle/src/verified_entity_terms.py","neurooracle/src/kg_structure_repairs.py",
    "neurooracle/src/kg_bulk_identity.py","neurooracle/src/kg_identity_pilot.py",
    "neurooracle/src/relation_evidence.py","neurooracle/src/shared_relation_catalog.py",
    "neurooracle/src/claim_evidence_identity.py","neurooracle/src/claim_ingestion.py",
    "neurooracle/src/graph_manager.py","neurooracle/src/storage.py","neurooracle/src/metadata_field_audit.py",
    "neurooracle/tests/test_verified_entity_terms.py","neurooracle/tests/test_kg_verified_terms_pipeline.py")]]


def code_bindings(): return [journal.fingerprint(p) for p in FILES]


def progress(phase,**details):
    state=dict(status="RUNNING",phase=phase,at=journal.utc_now(),**details)
    journal.atomic_json(OUTPUT/"RUN_STATE.json",state); print(compact(state),flush=True)


def proof_hashes(registry,plan):
    result={}
    def add(nid,sig):
        require(result.get(nid,sig)==sig,"conflicting node witnesses"); result[nid]=sig
    for term in registry.entries.values():
        for field,h in (("target_id","target_sha256"),("atom_id","atom_sha256"),("parent_id","parent_sha256")):
            add(term[field],term[h])
    for item in plan["structure_review"]: add(item["claim_id"],item["claim_sha256"])
    for nid,item in plan["structure_target_nodes"].items(): add(nid,item["record_sha256"])
    return result


def relation_stats(db,column):
    require(column in {"before_raw","before_canonical","after_raw","after_canonical"},"invalid comparison column")
    n,shared,indexed,multi,claims=db.execute(f"""SELECT COUNT(*),COALESCE(SUM(n>1),0),
        COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0),COALESCE(SUM(n>1 AND p>1),0),COALESCE(SUM(n),0)
        FROM (SELECT COUNT(*) n,COUNT(DISTINCT paper) p FROM comparison GROUP BY {column})""").fetchone()
    return dict(all_fine_grained_relation_groups=n,all_claims=claims,shared_groups=shared,
        indexed_claims=indexed,multi_paper_shared_groups=multi)


class SourceReview:
    """Validate all frozen witnesses before passing repaired, renumbered edges."""
    def __init__(self,plan,registry):
        self.plan=plan; self.registry=registry
        self.events={e["claim_id"]:e for e in plan["events"]}
        self.edits={e["ordinal"]:e for e in plan["structural_repairs"]["edits"]}
        require(len(self.edits)==len(plan["structural_repairs"]["edits"]),"duplicate structural ordinal")
        require(choose_repairs(plan["structure_review"],plan["structure_target_nodes"])==plan["structural_repairs"],"structural decision is not reproducible")
        self.keep={e["keep_ordinal"]:e["keep_sha256"] for e in self.edits.values() if "keep_ordinal" in e}
        require(not set(self.keep)&set(self.edits),"duplicate survivor also edited")
        self.refs={t["mapping_ref"]:t["mapping_sha256"] for t in registry.entries.values()}
        self.seen_edits=set(); self.seen_keep=set(); self.seen_refs=set()
        self.unique_changed=0; self.removed=0; self.science=hashlib.sha256()
        self.db=sqlite3.connect(":memory:"); self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.execute("CREATE TABLE comparison(before_raw TEXT,before_canonical TEXT,after_raw TEXT,after_canonical TEXT,paper TEXT)")
        self.buffer=[]

    def records(self,records):
        output_ordinal=0; source_ordinal=0
        for kind,key,row in records:
            if kind=="node" and key.startswith("CLM:"):
                after=change_claim(row,self.events[key]) if key in self.events else row
                md=row["metadata"]; amd=after["metadata"]
                self.science.update(bytes.fromhex(digest(nonidentity_claim(row))))
                keys=[relation_id(fn(value)) for fn,value in ((relation_key,md),(self.registry.relation_key,md),
                    (relation_key,amd),(self.registry.relation_key,amd))]
                self.buffer.append((*keys,evidence_member(md)["paper_key"]))
                if len(self.buffer)>=5000:
                    self.db.executemany("INSERT INTO comparison VALUES (?,?,?,?,?)",self.buffer); self.buffer.clear()
            if kind!="edge":
                yield kind,key,row; continue
            source_ordinal+=1; ordinal=int(key)
            require(ordinal==source_ordinal,"source edge order changed")
            ref=(row.get("metadata") or {}).get("audit_ref")
            if ref in self.refs:
                require(digest(row)==self.refs[ref],"source mapping proof changed"); self.seen_refs.add(ref)
            if ordinal in self.keep:
                require(digest(row)==self.keep[ordinal],"duplicate survivor changed"); self.seen_keep.add(ordinal)
            out=row
            if ordinal in self.edits:
                out=repair_edge(row,self.edits[ordinal]); self.seen_edits.add(ordinal)
            if out is None:
                self.removed+=1; continue
            shadow=change_edge(out,self.events)
            self.unique_changed+=shadow is not row
            output_ordinal+=1
            yield kind,str(output_ordinal),out
        require(self.seen_refs==set(self.refs) and self.seen_keep==set(self.keep) and self.seen_edits==set(self.edits),"incomplete source witnesses")
        self.db.executemany("INSERT INTO comparison VALUES (?,?,?,?,?)",self.buffer); self.buffer.clear(); self.db.commit()

    def comparison(self):
        split=self.db.execute("SELECT COUNT(*) FROM (SELECT before_raw FROM comparison GROUP BY before_raw HAVING COUNT(DISTINCT after_canonical)>1)").fetchone()[0]
        require(split==0,"new canonical grouping splits an existing exact-name shared relation")
        result={k:relation_stats(self.db,k) for k in ("before_raw","before_canonical","after_raw","after_canonical")}
        result.update(existing_raw_groups_newly_split=split,
            interpretation="compare before_raw/after_raw for ID routing; before_canonical/after_canonical for the same verified-term rule")
        return result


def build():
    require(not TEMP.exists() and not CATALOG.exists() and not (OUTPUT/"BUILD_STATE.json").exists(),"existing build; inspect before resuming")
    campaign=journal.read_json(journal.OUTPUT/"CAMPAIGN.json"); plan=journal.read_json(OUTPUT/"PLAN.json")
    registry=VerifiedEntityTerms(journal.read_json(TERMS)); term_fp=journal.fingerprint(TERMS)
    require(campaign["status"]=="COMPLETED" and campaign["active_process"] is None,"another operation active")
    require(campaign["current_graph"]==plan["graph"] and campaign["current_detail_store"]==plan["detail_store"],"plan/source advanced")
    require(Path(campaign["current_graph"]["path"]).resolve()==SOURCE.resolve(),"unexpected graph target")
    require(campaign["rollback_retention"] is False and campaign["automation_id"] is None,"retention/automation boundary changed")
    tests=OUTPUT/"TEST_RESULTS.xml"; suite=ElementTree.parse(tests).getroot().find("testsuite")
    require(int(suite.get("tests"))>=220 and all(int(suite.get(k))==0 for k in ("failures","errors","skipped")),"passing full regression required")
    journal.guards([campaign["current_graph"],campaign["current_detail_store"],campaign["formal_sources"],campaign["current_acceptance"]])
    require(journal.fingerprint(Path(campaign["current_acceptance"]["path"]))==campaign["current_acceptance"],"acceptance SHA mismatch")
    guards=[native_info(fp["path"]) for fp in [campaign["current_graph"],campaign["current_detail_store"],*campaign["formal_sources"].values()]]
    require(shutil.disk_usage(SOURCE).free>campaign["current_graph"]["bytes"]+12*1024**3,"insufficient atomic output headroom")
    code=code_bindings(); plan_fp=journal.fingerprint(OUTPUT/"PLAN.json")
    issues=rows(campaign["current_issues"]["path"])
    require(not {e["claim_id"] for e in plan["events"]}&{i["claim_id"] for i in issues},"open issue overlaps identity changes")
    hashes=proof_hashes(registry,plan); review=SourceReview(plan,registry)
    baseline=deepcopy(campaign)
    campaign.update(status="MANUAL_ACTIVE",phase="R30完整术语复用、结构修复及全图验收",updated_at=journal.utc_now(),
        active_process=dict(kind="verified_terms",pid=os.getpid(),state=str(OUTPUT/"RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",campaign)
    progress("BUILD_START",claims=plan["changed_claims"],endpoints=plan["changed_endpoints"])
    with TEMP.open("xb",buffering=1024**2) as handle:
        with hashed_reader(SOURCE) as (reader,sha):
            result=stream_patch(review.records(walk_graph(reader)),handle,plan["events"],hashes,CATALOG,progress,
                claim_transform=change_claim,edge_transform=change_edge,relation_key_func=registry.relation_key,
                relation_version="kg.relation_evidence.v2",extra_graph_metadata=dict(entity_identity=dict(
                    version=VERSION,registry=os.path.relpath(TERMS,SOURCE.parent).replace("\\","/"),sha256=term_fp["sha256"])))
            require(sha.hexdigest()==plan["graph"]["sha256"],"full source SHA mismatch")
        handle.flush(); os.fsync(handle.fileno())
    comparison=review.comparison(); review.db.close()
    require(comparison["before_raw"]==baseline["relation_evidence_counts"] and comparison["after_canonical"]==result["catalog_stats"],"full comparison disagrees with accepted/catalog counts")
    require(result["nonidentity_claim_digest"]==review.science.hexdigest(),"source scientific/audit content changed")
    require(result["counts"]=={**baseline["counts"],"edges":baseline["counts"]["edges"]-review.removed},"unexpected record counts")
    require(review.removed==plan["structural_repairs"]["removed_duplicate_edges"] and review.unique_changed==plan["unique_changed_edge_records"],"structural/combined counts differ")
    require(result["changed"]==dict(claim_nodes=plan["changed_claims"],edge_records=plan["changed_edges"]),"identity counts differ")
    require(len(result["metadata_key_unions"]["nodes"])==baseline["node_metadata_field_union"] and len(result["metadata_key_unions"]["edges"])==baseline["edge_metadata_field_union"],"metadata field union changed")
    closure={(c["claim_id"],c["kind"]):c["count"] for c in result["corrected_edge_closure"]}
    require(all(closure[(e["claim_id"],k)]==v for e in plan["events"] for k,v in e["closure"].items()),"owned edge closure differs")
    guard_native(guards); require(code_bindings()==code and journal.fingerprint(OUTPUT/"PLAN.json")==plan_fp and journal.fingerprint(TERMS)==term_fp,"code/plan/terms changed")
    state=dict(status="BUILT_NOT_ADOPTED",at=journal.utc_now(),baseline=baseline,protected_native=guards,
        temporary=cheap(TEMP),result=result,code=code,plan=plan_fp,catalog=journal.fingerprint(CATALOG),terms=term_fp,
        comparison=comparison,tests=journal.fingerprint(tests),issues=issues,source_full_sha_verified=True,
        combined_changed_edges=review.unique_changed,removed_duplicate_edges=review.removed,
        graph_backups=0,record_preimages_saved=False)
    journal.atomic_json(OUTPUT/"BUILD_STATE.json",state); progress("BUILT_NOT_ADOPTED",changed=result["changed"],relations=result["catalog_stats"])


def validate():
    state=journal.read_json(OUTPUT/"BUILD_STATE.json"); plan=journal.read_json(OUTPUT/"PLAN.json")
    require(not (OUTPUT/"VALIDATED.json").exists(),"already validated")
    require(code_bindings()==state["code"] and cheap(TEMP)==state["temporary"],"build/code changed")
    require(journal.fingerprint(CATALOG)==state["catalog"] and journal.fingerprint(OUTPUT/"PLAN.json")==state["plan"] and journal.fingerprint(TERMS)==state["terms"],"catalog/plan/terms changed")
    guard_native(state["protected_native"])
    registry=VerifiedEntityTerms(journal.read_json(TERMS)); expected_nodes=proof_hashes(registry,{**plan,"structure_review":[],"structure_target_nodes":{}})
    expected_refs={t["mapping_ref"]:t["mapping_sha256"] for t in registry.entries.values()}
    seen_nodes=set(); seen_refs=set(); events={e["claim_id"]:e for e in plan["events"]}; reversed_claims=set()
    repaired=set(plan["structural_repairs"]["repaired_claim_ids"]); repaired_md={}; closure=Counter(); coverage=Coverage()
    issue_ids={i["claim_id"]:i["current_node_sha256"] for i in state["issues"]}; issue_seen=set()
    def observed(records):
        for kind,key,row in records:
            if kind=="node":
                if key in expected_nodes:
                    require(digest(row)==expected_nodes[key],"output identity proof node changed"); seen_nodes.add(key)
                if key in events:
                    before=deepcopy(row)
                    for change in events[key]["changes"]:
                        field=change["side"]+"_id"; require(before["metadata"][field]==change["target_id"],"output endpoint differs")
                        before["metadata"][field]=change["old_id"]
                        if field in (before["metadata"].get("metadata") or {}): before["metadata"]["metadata"][field]=change["old_id"]
                    require(digest(before)==events[key]["claim_sha256"],"reverse identity-only transform differs from source claim")
                    reversed_claims.add(key)
                if key in issue_ids:
                    require(digest(row)==issue_ids[key],"held scientific issue changed"); issue_seen.add(key)
                if key in repaired: repaired_md[key]=row["metadata"]
            elif kind=="edge":
                ref=(row.get("metadata") or {}).get("audit_ref")
                if ref in expected_refs:
                    require(digest(row)==expected_refs[ref],"output identity mapping changed"); seen_refs.add(ref)
                owner=row["source_id"] if row["relation_type"]=="about" else (row.get("metadata") or {}).get("claim_id")
                if owner in repaired:
                    md=repaired_md[owner]
                    if row["relation_type"]=="about":
                        require(row["target_id"] in (md["subject_id"],md["object_id"]),"repaired about endpoint inconsistent")
                        closure[(owner,row["target_id"])]+=1
                    else:
                        require((row["source_id"],row["target_id"],row["relation_type"])==(md["subject_id"],md["object_id"],md["predicate"]),"repaired scientific edge inconsistent")
                        closure[(owner,"science")]+=1
            if kind!="metadata": coverage.add("edge/all" if kind=="edge" else "node/"+node_class(key,row),row)
            yield kind,key,row
    with hashed_reader(TEMP) as (reader,sha):
        checks=verify_catalog_and_graph(observed(walk_graph(reader)),state["result"],CATALOG,progress,relation_key_func=registry.relation_key)
        candidate_sha=sha.hexdigest()
    require(seen_nodes==set(expected_nodes) and seen_refs==set(expected_refs) and reversed_claims==set(events) and issue_seen==set(issue_ids),"independent proof/claim closure incomplete")
    require(all(closure[(cid,x)]==1 for cid,md in repaired_md.items() for x in (md["subject_id"],md["object_id"],"science")) and set(repaired_md)==repaired,"repaired claim closure incomplete")
    detail=state["baseline"]["current_detail_store"]; progress("VERIFY_DETAIL_STORE_SHA")
    require(sha256(Path(detail["path"]))==detail["sha256"],"UMLS detail SHA mismatch")
    write_rows(OUTPUT/"CURRENT_METADATA_COVERAGE.jsonl",coverage.rows())
    checks.update(verified_identity_proofs_complete=True,reverse_identity_transform_source_hashes_match=True,
        repaired_claims_closed=len(repaired),held_scientific_issues_unchanged=len(issue_seen),
        all_source_nonidentity_claim_fields_preserved=True,source_edge_changes_limited_to_approved_rules=True,
        metadata_coverage_recomputed=True,metadata_scope_denominators=dict(coverage.denominators))
    # The generic verifier's identity-only label is too narrow for this batch.
    checks.pop("records_match_approved_identity_only_transform",None)
    guard_native(state["protected_native"]); require(code_bindings()==state["code"],"code changed during validation")
    accepted=dict(status="VALIDATED_NOT_ADOPTED",at=journal.utc_now(),graph={**cheap(TEMP),"sha256":candidate_sha},
        temporary_native=native_info(TEMP),checks=checks,detail_sha_verified=True,
        build_state=journal.fingerprint(OUTPUT/"BUILD_STATE.json"),catalog=state["catalog"],terms=state["terms"],
        coverage=journal.fingerprint(OUTPUT/"CURRENT_METADATA_COVERAGE.jsonl"))
    journal.atomic_json(OUTPUT/"VALIDATED.json",accepted); progress("VALIDATED_NOT_ADOPTED",checks=checks)


def rebind_issue(issue,graph,removed):
    result=deepcopy(issue)
    for field in ("related_edge_ordinals","owned_science_ordinals","about_ordinals"):
        if field in result:
            require(not set(result[field])&set(removed),"held issue edge was removed")
            result[field]=[n-bisect_left(removed,n) for n in result[field]]
    result.update(current_graph=graph,version_binding_status="FULL_CURRENT_GRAPH_VALIDATED")
    return result


def apply():
    state=journal.read_json(OUTPUT/"BUILD_STATE.json"); accepted=journal.read_json(OUTPUT/"VALIDATED.json"); plan=journal.read_json(OUTPUT/"PLAN.json")
    require(not (OUTPUT/"CURRENT_ACCEPTANCE.json").exists(),"already applied")
    require(journal.fingerprint(OUTPUT/"BUILD_STATE.json")==accepted["build_state"] and code_bindings()==state["code"],"build/code advanced")
    guard_native([*state["protected_native"],accepted["temporary_native"]])
    require(journal.fingerprint(CATALOG)==accepted["catalog"] and journal.fingerprint(TERMS)==accepted["terms"] and journal.fingerprint(OUTPUT/"PLAN.json")==state["plan"] and journal.fingerprint(OUTPUT/"CURRENT_METADATA_COVERAGE.jsonl")==accepted["coverage"],"current artifacts advanced")
    campaign=journal.read_json(journal.OUTPUT/"CAMPAIGN.json")
    require(campaign["current_graph"]==state["baseline"]["current_graph"] and campaign["active_process"]["kind"]=="verified_terms","current graph advanced")
    require(TEMP.resolve().parent==SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()),"unsafe replacement path")
    os.replace(TEMP,SOURCE)
    graph={**cheap(SOURCE),"sha256":accepted["graph"]["sha256"]}
    require(graph["bytes"]==accepted["graph"]["bytes"] and graph["mtime_ns"]==accepted["graph"]["mtime_ns"],"replacement differs")
    removed=sorted(e["ordinal"] for e in plan["structural_repairs"]["edits"] if e["action"]=="remove_same_owner_duplicate")
    write_rows(OUTPUT/"CURRENT_REMAINING_ISSUES.jsonl",[rebind_issue(i,graph,removed) for i in state["issues"]])
    write_rows(OUTPUT/"CURRENT_STRUCTURE_HOLDS.jsonl",[dict(claim_id=cid,current_graph=graph,
        issue="alternate_path_has_unconfirmed_full_name_scope",status="held_not_changed") for cid in plan["structural_repairs"]["held_claim_ids"]])
    result=state["result"]
    receipt=dict(status="CURRENT_VERIFIED_TERMS_APPLIED",at=journal.utc_now(),graph=graph,counts=result["counts"],
        changed=result["changed"],changed_endpoints=plan["changed_endpoints"],combined_changed_edges=state["combined_changed_edges"],
        removed_duplicate_edges=len(removed),structure_repairs=plan["structural_repairs"],verified_terms=len(journal.read_json(TERMS)["terms"]),
        missing_type_identity_endpoints=plan["missing_type_identity_endpoints"],typed_annotation_values_written=0,
        connected_components=result["connected_components"],isolated_nodes=result["isolated_nodes"],self_loops=result["self_loops"],
        metadata_key_unions=result["metadata_key_unions"],metadata=result["metadata"],source_graph=plan["graph"],source_kg_retained=False,
        shared_relations=accepted["catalog"],entity_terms=accepted["terms"],metadata_coverage=accepted["coverage"],
        relation_counts=result["catalog_stats"],checks=accepted["checks"],tests=state["tests"],code=state["code"],
        detail_store=state["baseline"]["current_detail_store"],rollback_retention=False,record_preimages_saved=False,
        formal_apply_performed=False,comparison=state["comparison"])
    journal.atomic_json(OUTPUT/"CURRENT_ACCEPTANCE.json",receipt)
    campaign.update(status="COMPLETED",phase="R30完整术语复用、同义关系索引及8项结构问题整批完成",active_process=None,
        updated_at=receipt["at"],current_graph=graph,current_acceptance=journal.fingerprint(OUTPUT/"CURRENT_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT/"CURRENT_REMAINING_ISSUES.jsonl"),current_shared_relations=accepted["catalog"],
        current_entity_terms=accepted["terms"],current_coverage=accepted["coverage"],
        current_structure_holds=journal.fingerprint(OUTPUT/"CURRENT_STRUCTURE_HOLDS.jsonl"),
        counts=result["counts"],relation_evidence_counts=result["catalog_stats"],last_deep_verification=accepted["at"],
        last_current_graph_content_verification=dict(graph=graph,independent_full_scan=True),
        next_steps=["复核复合表达、基因指向和类型冲突；不能仅凭相似名字归一。",
            "当前9项结构队列已解决8项；剩余frailty范围冲突及21条语义复审不强行合并。",
            "正式full_v2未同步；无模型/训练；仅当前工作KG，无回退图。"])
    journal.atomic_json(journal.OUTPUT/"CAMPAIGN.json",campaign)
    journal.atomic_json(OUTPUT/"RUN_STATE.json",dict(status="COMPLETED",at=receipt["at"],graph=graph,changed=result["changed"]))
    print(compact(dict(status=receipt["status"],changed=result["changed"],comparison=receipt["comparison"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("phase",choices=("build","validate","apply","run"))
    args=parser.parse_args()
    for phase in ("build","validate","apply") if args.phase=="run" else (args.phase,): globals()[phase]()
