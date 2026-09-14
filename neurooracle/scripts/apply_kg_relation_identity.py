"""Apply reviewed endpoint corrections and a compact shared-evidence index.

One temporary file for atomic replacement, no historical KG or preimage copy.
The original claims remain the authoritative evidence; no consensus is inferred.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import Components, cheap, compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import Writer, verify_stream, guard_native, rows, write_rows
from reclaim_kg_backup_storage import native_info, sha256
from neurooracle.src.kg_identity_pilot import choose_events, change_claim, change_edge, nonidentity_claim, digest
from neurooracle.src.relation_evidence import VERSION, relation_key, relation_id, evidence_member, summarize_members

OUTPUT = journal.OUTPUT / "round28_relation_identity"
SOURCE = journal.OUTPUT / "round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP = SOURCE.with_name(SOURCE.name + ".identity.tmp")
CATALOG = OUTPUT / "CURRENT_SHARED_RELATIONS.jsonl"
FILES = [__file__, journal.REPO / "neurooracle/src/kg_identity_pilot.py",
    journal.REPO / "neurooracle/src/relation_evidence.py", journal.REPO / "neurooracle/src/graph_manager.py",
    journal.REPO / "neurooracle/src/storage.py", journal.REPO / "neurooracle/tests/test_relation_evidence.py",
    journal.REPO / "neurooracle/tests/test_kg_identity_pilot.py"]


def code_bindings():
    return [journal.fingerprint(Path(path)) for path in FILES]


def progress(phase, **details):
    state = dict(status="RUNNING", phase=phase, at=journal.utc_now(), **details)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", state)
    print(compact(state), flush=True)


def export_catalog(db, path, *, papers=None):
    db.execute("CREATE INDEX relation_lookup ON evidence(k)")
    counts = db.execute("SELECT COUNT(*),SUM(n),SUM(n>1) FROM (SELECT COUNT(*) n FROM evidence GROUP BY k)").fetchone()
    stats = dict(all_fine_grained_relation_groups=counts[0], all_claims=counts[1], shared_groups=counts[2],
                 indexed_claims=0, multi_paper_shared_groups=0)
    if papers is not None:
        stats.update(verified_multi_paper_shared_groups=0, fully_verified_shared_groups=0, unverified_indexed_claims=0)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for key, in db.execute("SELECT k FROM evidence GROUP BY k HAVING COUNT(*)>1 ORDER BY k"):
            members = [dict(claim_id=cid, paper_key=paper, negated=json.loads(neg), context_signature=ctx,
                            **({"paper_status":status} if papers is not None else {}))
                       for cid,paper,neg,ctx,status in db.execute("SELECT cid,paper,neg,ctx,status FROM evidence WHERE k=? ORDER BY cid", (key,))]
            group = summarize_members(json.loads(key), members)
            stats["indexed_claims"] += group["claim_count"]
            stats["multi_paper_shared_groups"] += group["paper_count"] > 1
            if papers is not None:
                stats["verified_multi_paper_shared_groups"] += group["verified_paper_count"] > 1
                stats["fully_verified_shared_groups"] += group["unverified_claim_count"] == 0
                stats["unverified_indexed_claims"] += group["unverified_claim_count"]
            handle.write(compact(group) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    return stats


def stream_patch(records, handle, events, target_hashes, catalog_path, callback=None, *,
                 claim_transform=change_claim, edge_transform=change_edge,
                 relation_key_func=relation_key, extra_graph_metadata=None, relation_version=VERSION, papers=None):
    by_id = {item["claim_id"]: item for item in events}
    require(len(by_id) == len(events), "duplicate reviewed claim")
    writer, cc = Writer(handle), Components()
    counts, changed, domains, sources, relations = (Counter() for _ in range(5))
    digests = {key: hashlib.sha256() for key in ("nodes", "edges")}
    science = hashlib.sha256()
    unions = {key: set() for key in digests}
    seen, targets, closure = set(), set(), Counter()
    degrees, metadata = None, None
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("CREATE TABLE evidence(cid TEXT PRIMARY KEY,k TEXT,paper TEXT,neg TEXT,ctx TEXT,status TEXT)")
    buffer = []
    try:
        for kind, key, record in records:
            if kind == "metadata":
                require(metadata is None, "duplicate graph metadata")
                metadata = deepcopy(record)
                continue
            require(metadata is not None, "missing graph metadata")
            out = record
            if kind == "node":
                require(degrees is None and key == record["id"], "invalid node identity/order")
                cc.add(key)
                if key in target_hashes:
                    require(digest(record) == target_hashes[key], "reviewed target changed")
                    targets.add(key)
                if key in by_id:
                    out = claim_transform(record, by_id[key])
                    require(digest(nonidentity_claim(record)) == digest(nonidentity_claim(out)), "nonidentity field changed")
                    seen.add(key); changed["claim_nodes"] += 1
                if key.startswith("CLM:"):
                    counts["claims"] += 1
                    science.update(bytes.fromhex(digest(nonidentity_claim(out))))
                    md = out["metadata"]
                    require(md["id"] == key, "claim metadata ID differs")
                    member = evidence_member(md, papers=papers)
                    buffer.append((key, compact(relation_key_func(md)), member["paper_key"], compact(member["negated"]), member["context_signature"], member.get("paper_status")))
                    if len(buffer) >= 5000:
                        db.executemany("INSERT INTO evidence VALUES (?,?,?,?,?,?)", buffer); buffer.clear()
                domains.update(out.get("domain_tags") or [])
                sources[out.get("source_vocab") or ""] += 1
            else:
                require(kind == "edge" and int(key) == counts["edges"] + 1, "edge ordinal differs")
                if degrees is None:
                    degrees = bytearray(len(cc.ids))
                out = edge_transform(record, by_id)
                if out is not record:
                    changed["edge_records"] += 1
                    owner = record["source_id"] if record["relation_type"] == "about" else record["metadata"]["claim_id"]
                    closure[(owner, "about" if record["relation_type"] == "about" else "science")] += 1
                sid, tid = out["source_id"], out["target_id"]
                cc.union(sid, tid)
                degrees[cc.ids[sid]] = degrees[cc.ids[tid]] = 1
                counts["self_loops"] += sid == tid
                owner = (out.get("metadata") or {}).get("claim_id")
                require(owner is None or owner in cc.ids, "missing edge owner")
                relations[out["relation_type"]] += 1
            group = kind + "s"
            counts[group] += 1
            payload = compact(out).encode("utf-8")
            digests[group].update(payload + b"\n")
            unions[group].update(out.get("metadata") or {})
            if kind == "node": writer.node(key, payload)
            else: writer.edge(payload)
            if callback and counts[group] % 250000 == 0:
                callback("BUILD_" + group.upper(), counts=dict(counts), changed=dict(changed))
        require(seen == set(by_id) and targets == set(target_hashes), "reviewed closure incomplete")
        require(all(closure[(cid,"about")] >= 1 for cid in by_id), "missing corrected about edge")
        db.executemany("INSERT INTO evidence VALUES (?,?,?,?,?,?)", buffer)
        db.commit()
        if callback: callback("SHARED_RELATION_INDEX")
        catalog_stats = export_catalog(db, catalog_path, papers=papers)
    finally:
        db.close()
    metadata["stats"] = dict(n_concepts=counts["nodes"], n_edges=counts["edges"], domains=dict(domains),
        sources=dict(sources), relations=dict(relations), connected_components=cc.count)
    metadata["relation_evidence"] = dict(version=relation_version,
        shared_relation_index=os.path.relpath(catalog_path, SOURCE.parent).replace("\\", "/"),
        sha256=sha256(catalog_path), singleton_claims_remain_in_graph=True,
        interpretation="shared_relation_evidence_not_consensus")
    if papers is not None:
        metadata["relation_evidence"].update(paper_count_semantics="source_keys_including_unverified",
            verified_paper_count_semantics="distinct_authority_verified_articles_not_independent_cohorts")
    if extra_graph_metadata:
        metadata.update(deepcopy(extra_graph_metadata))
    writer.finish(metadata)
    return dict(counts={k:counts[k] for k in ("nodes", "claims", "edges")}, changed=dict(changed), metadata=metadata,
        record_digests={k:v.hexdigest() for k,v in digests.items()}, nonidentity_claim_digest=science.hexdigest(),
        metadata_key_unions={k:sorted(v) for k,v in unions.items()}, connected_components=cc.count,
        isolated_nodes=(degrees if degrees is not None else bytearray(len(cc.ids))).count(0), self_loops=counts["self_loops"],
        deleted_claim_ids=[], catalog_stats=catalog_stats,
        corrected_edge_closure=[dict(claim_id=cid,kind=kind,count=n) for (cid,kind),n in sorted(closure.items())])


def verify_catalog_and_graph(records, expected, catalog_path, callback=None, *, relation_key_func=relation_key, papers=None):
    catalog_rows = rows(catalog_path)
    groups = {row["id"]: row for row in catalog_rows}
    require(len(groups) == len(catalog_rows), "duplicate catalog group")
    expected_members = {}
    for rid, group in groups.items():
        require(rid == relation_id(relation_key(group)), "catalog label/key mismatch")
        require(group["evidence_variant_count"] == len(group["evidence_variants"]), "catalog variant count differs")
        contexts = {cid: v["signature"] for v in group["evidence_variants"] for cid in v["claim_ids"]}
        require(len(contexts) == group["claim_count"] == len(group["members"]), "catalog member/variant counts differ")
        require(len({m["paper_key"] for m in group["members"]}) == group["paper_count"], "catalog paper count differs")
        if papers is not None:
            require(group["verified_paper_count"] == len({m["paper_key"] for m in group["members"] if m.get("paper_status") == "verified"}), "catalog verified paper count differs")
            require(group["unverified_claim_count"] == sum(m.get("paper_status") != "verified" for m in group["members"]), "catalog unverified count differs")
        for member in group["members"]:
            cid = member["claim_id"]
            require(cid not in expected_members, "claim in more than one shared group")
            expected_members[cid] = (rid, {**member, "context_signature": contexts[cid]})
    count_keys, science, matched = Counter(), hashlib.sha256(), set()
    def observed():
        for kind, key, record in records:
            if kind == "node" and key.startswith("CLM:"):
                science.update(bytes.fromhex(digest(nonidentity_claim(record))))
                md = record["metadata"]
                rid = relation_id(relation_key_func(md))
                count_keys[rid] += 1
                if rid in groups:
                    require(key in expected_members, "missing current claim in shared relation")
                    stored_rid, member = expected_members[key]
                    require(stored_rid == rid and digest(member) == digest(evidence_member(md, papers=papers)), "catalog claim evidence differs")
                    matched.add(key)
            yield kind, key, record
    checks = verify_stream(observed(), expected, callback)
    require(science.hexdigest() == expected["nonidentity_claim_digest"], "science/audit digest differs")
    require(matched == set(expected_members), "stale catalog claim")
    require({k for k,n in count_keys.items() if n>1} == set(groups), "shared catalog incomplete")
    require(len(count_keys) == expected["catalog_stats"]["all_fine_grained_relation_groups"], "group count differs")
    checks.pop("retained_records_identical", None)
    checks.update(records_match_approved_identity_only_transform=True, all_nonidentity_claim_fields_preserved=True,
        shared_relation_index_complete=True, shared_groups=len(groups), indexed_claims=len(matched))
    return checks


def build():
    require(not TEMP.exists() and not CATALOG.exists() and not (OUTPUT / "BUILD_STATE.json").exists(), "build exists; inspect/resume")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    collection = journal.read_json(OUTPUT / "IDENTITY_EVIDENCE.json")
    require(campaign["current_graph"] == collection["graph"] and campaign["active_process"] is None, "current source advanced")
    require(Path(campaign["current_graph"]["path"]).resolve() == SOURCE.resolve(), "unexpected current path")
    require(campaign["rollback_retention"] is False and campaign["automation_id"] is None, "retention boundary differs")
    events = choose_events(collection)
    issues = rows(campaign["current_issues"]["path"])
    require(not {item["claim_id"] for item in events} & {r["claim_id"] for r in issues}, "held issue overlaps reviewed scope")
    tests = OUTPUT / "TEST_RESULTS.xml"
    suite = ElementTree.parse(tests).getroot().find("testsuite")
    require(int(suite.get("tests")) >= 80 and all(int(suite.get(k)) == 0 for k in ("failures","errors","skipped")), "passing tests required")
    journal.guards([campaign["current_graph"], campaign["current_detail_store"], campaign["formal_sources"], campaign["current_acceptance"]])
    guards = [native_info(fp["path"]) for fp in [campaign["current_graph"], campaign["current_detail_store"], *campaign["formal_sources"].values()]]
    require(shutil.disk_usage(SOURCE).free > campaign["current_graph"]["bytes"] + 12*1024**3, "insufficient atomic-write headroom")
    target_hashes = {r["target_id"]: collection["nodes"][r["target_id"]]["record_sha256"] for r in events}
    code = code_bindings()
    journal.atomic_json(OUTPUT / "DECISIONS.json", dict(events=events,
        decision="claim-specific identity correction, not a whole-node merge or scientific re-audit",
        holds=["unreviewed names/roles and inadequate stored identity context", "cognitive decline vs disorder", "broad/gene endpoint errors"],
        preimages_saved=False))
    campaign.update(status="MANUAL_ACTIVE", phase="38条端点归一与共享关系证据索引", updated_at=journal.utc_now(),
        active_process=dict(kind="relation_identity", pid=os.getpid(), state=str(OUTPUT / "RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    progress("STARTING_APPROVED_IDENTITY_REWRITE")
    with TEMP.open("xb", buffering=1024**2) as handle:
        with hashed_reader(SOURCE) as (reader, sha):
            result = stream_patch(walk_graph(reader), handle, events, target_hashes, CATALOG, progress)
            require(sha.hexdigest() == campaign["current_graph"]["sha256"], "source full SHA differs")
        handle.flush(); os.fsync(handle.fileno())
    require(result["counts"] == campaign["counts"] and result["changed"]["claim_nodes"] == 38, "unexpected count delta")
    guard_native(guards)
    require(code_bindings() == code, "code changed during build")
    state = dict(status="BUILT_NOT_ADOPTED", at=journal.utc_now(), baseline=campaign,
        protected_native=guards, temporary=cheap(TEMP), result=result, code=code,
        catalog=journal.fingerprint(CATALOG), tests=journal.fingerprint(tests), issues=issues,
        graph_backups=0, record_preimages_saved=False, source_full_sha_verified=True)
    journal.atomic_json(OUTPUT / "BUILD_STATE.json", state)
    progress("BUILT_NOT_ADOPTED", changed=result["changed"], relations=result["catalog_stats"])


def validate():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    require(not (OUTPUT / "VALIDATED.json").exists(), "already validated")
    require(code_bindings() == state["code"] and cheap(TEMP) == state["temporary"], "build/code changed")
    guard_native(state["protected_native"])
    require(journal.fingerprint(CATALOG) == state["catalog"], "catalog changed")
    with hashed_reader(TEMP) as (reader, sha):
        checks = verify_catalog_and_graph(walk_graph(reader), state["result"], CATALOG, progress)
        candidate_sha = sha.hexdigest()
    detail = state["baseline"]["current_detail_store"]
    require(sha256(Path(detail["path"])) == detail["sha256"], "detail SHA differs")
    guard_native(state["protected_native"])
    require(code_bindings() == state["code"], "code changed during validation")
    accepted = dict(status="VALIDATED_NOT_ADOPTED", at=journal.utc_now(), graph={**cheap(TEMP), "sha256": candidate_sha},
        temporary_native=native_info(TEMP), checks=checks, detail_sha_verified=True,
        build_state=journal.fingerprint(OUTPUT / "BUILD_STATE.json"), catalog=state["catalog"])
    journal.atomic_json(OUTPUT / "VALIDATED.json", accepted)
    progress("VALIDATED_NOT_ADOPTED", checks=checks)


def apply():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    accepted = journal.read_json(OUTPUT / "VALIDATED.json")
    require(not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "already applied")
    require(journal.fingerprint(OUTPUT / "BUILD_STATE.json") == accepted["build_state"] and code_bindings() == state["code"], "frozen build/code changed")
    guard_native([*state["protected_native"], accepted["temporary_native"]])
    require(journal.fingerprint(CATALOG) == accepted["catalog"], "frozen catalog changed")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["current_graph"] == state["baseline"]["current_graph"], "current KG advanced before apply")
    # Exact current target only; original is atomically replaced, not backed up.
    require(TEMP.resolve().parent == SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()), "unsafe replacement target")
    os.replace(TEMP, SOURCE)
    graph = {**cheap(SOURCE), "sha256": accepted["graph"]["sha256"]}
    require(graph["bytes"] == accepted["graph"]["bytes"] and graph["mtime_ns"] == accepted["graph"]["mtime_ns"], "replacement fingerprint differs")
    for issue in state["issues"]:
        issue.update(current_graph=graph, version_binding_status="FULL_CURRENT_GRAPH_VALIDATED")
    write_rows(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl", state["issues"])
    result = state["result"]
    receipt = dict(status="CURRENT_RELATION_IDENTITY_APPLIED", at=journal.utc_now(), graph=graph,
        counts=result["counts"], changed=result["changed"], connected_components=result["connected_components"],
        isolated_nodes=result["isolated_nodes"], self_loops=result["self_loops"],
        metadata_key_unions=result["metadata_key_unions"], metadata=result["metadata"],
        source_graph=state["baseline"]["current_graph"], source_kg_retained=False,
        shared_relations=accepted["catalog"], relation_counts=result["catalog_stats"], checks=accepted["checks"],
        tests=state["tests"], code=state["code"], detail_store=state["baseline"]["current_detail_store"],
        rollback_retention=False, record_preimages_saved=False, formal_apply_performed=False)
    journal.atomic_json(OUTPUT / "CURRENT_ACCEPTANCE.json", receipt)
    campaign.update(status="COMPLETED", phase="首批38条实体端点归一及共享关系证据索引已应用", active_process=None,
        updated_at=receipt["at"], current_graph=graph, current_acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"), current_shared_relations=accepted["catalog"],
        relation_evidence_counts=result["catalog_stats"], last_deep_verification=accepted["at"],
        last_current_graph_content_verification=dict(graph=graph, independent_full_scan=True),
        next_steps=["继续按完整名称、类型与原文核对实体身份；本批不是全图语义归一完成。",
            "共享关系索引组织独立论文证据，不表示共识；同义表达和过粗/错误端点仍需后续处理。",
            "正式full_v2未同步；无模型或训练；只保留当前工作KG和必要索引明细。"])
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", dict(status="COMPLETED", at=receipt["at"], graph=graph, changed=result["changed"]))
    print(compact(dict(status=receipt["status"], changed=result["changed"], relations=result["catalog_stats"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "validate", "apply", "run"))
    args = parser.parse_args()
    for phase in ("build", "validate", "apply") if args.phase == "run" else (args.phase,):
        globals()[phase]()
