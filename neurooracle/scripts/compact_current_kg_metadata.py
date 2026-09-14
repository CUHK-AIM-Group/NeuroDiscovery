"""Apply the approved exact/empty/import-state metadata cleanup to the current KG.

One temporary output, independent full verification, then atomic replacement.
No old-KG copies, record preimages, models, training, or formal-source writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from html import escape
import os
from pathlib import Path
import shutil
import sys
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import Writer, guard_native, rows, verify_stream, write_rows
from reclaim_kg_backup_storage import native_info, sha256
from record_kg_source_findings import record_digest
from neurooracle.src.case_study_membership_contract import claim_evidence_payload
from neurooracle.src.kg_metadata_compaction import LAYOUT, LAYOUT_KEY, compact_record, typed_equal
from neurooracle.src.metadata_field_audit import Coverage, node_class

OUTPUT = journal.OUTPUT / "round27_metadata_compaction"
REVIEW_DIR = journal.REPO / "neurooracle/data/kg_metadata_cleanup_review"
SOURCE = journal.OUTPUT / "round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP = SOURCE.with_name(SOURCE.name + ".metadata-compact.tmp")
SOURCE_SHA = "d04ea8c3aa1c1454c0f9356f0b9ebb2d660ea27db40111cb48fc2cf9d9f997ac"
EVENT_ID = "metadata_exact_empty_import_compaction_20260908"
CODE_PATHS = (
    "neurooracle/src/kg_metadata_compaction.py", "neurooracle/src/storage.py",
    "neurooracle/src/graph_manager.py", "neurooracle/src/schema.py",
    "neurooracle/src/case_study_scope.py", "neurooracle/src/claim_compatibility.py",
    "neurooracle/src/case_study_membership_contract.py", "neurooracle/src/metadata_field_audit.py",
    "neurooracle/scripts/compact_current_kg_metadata.py", "neurooracle/scripts/prune_current_kg.py",
    "neurooracle/scripts/build_umls_simplification_candidate.py", "neurooracle/scripts/streaming_graph_json.py",
    "neurooracle/tests/test_kg_metadata_compaction.py", "neurooracle/tests/test_kg_metadata_compaction_pipeline.py",
)


def code_fingerprints():
    return [journal.fingerprint(journal.REPO / path) for path in CODE_PATHS]


def progress(phase, **details):
    state = dict(status="RUNNING", phase=phase, at=journal.utc_now(), **details)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", state)
    print(compact(state), flush=True)


def protected_claim(record):
    """Separate existing audit/evidence contract from the storage transform."""
    md = record["metadata"]
    audit = md.get("scope_reaudit") or {}
    return {
        "node_fields": {k: v for k, v in record.items() if k != "metadata"},
        "evidence_contract": claim_evidence_payload(md,
            scope_context_sha256=audit.get("scope_context_sha256") or "",
            include_legacy_case2_chain=True),
        "scope_reaudit": md.get("scope_reaudit"),
        "paper_case_study_ids": md.get("paper_case_study_ids"),
        "claim_case_study_ids": md.get("claim_case_study_ids"),
        "subject_id": md.get("subject_id"), "object_id": md.get("object_id"),
        "confidence": md.get("confidence"), "negated": md.get("negated"),
    }


def stream_compact(records, handle, baseline, issue_rows=(), on_progress=None):
    """Bounded-memory source pass; computes expected output and protected digests."""
    writer = Writer(handle)
    counts, changed, removals = Counter(), Counter(), Counter()
    before_unions = {kind: set() for kind in ("nodes", "edges")}
    after_unions = {kind: set() for kind in ("nodes", "edges")}
    digests = {kind: hashlib.sha256() for kind in ("nodes", "edges")}
    protected_digest = hashlib.sha256()
    issues = {r["claim_id"]: deepcopy(r) for r in issue_rows}
    require(len(issues) == len(issue_rows), "duplicate issue IDs")
    seen_issues = set()
    metadata = None
    for kind, key, record in records:
        if kind == "metadata":
            require(metadata is None and not counts, "metadata missing/duplicate/out of order")
            require(LAYOUT_KEY not in record, "source already has a storage layout")
            metadata = deepcopy(record)
            metadata[LAYOUT_KEY] = deepcopy(LAYOUT)
            continue
        require(metadata is not None, "graph metadata must precede records")
        out = compact_record(kind, record, removals)
        group = kind + "s"
        counts[group] += 1
        before_unions[group].update(record.get("metadata") or {})
        after_unions[group].update(out.get("metadata") or {})
        original_payload, payload = compact(record).encode("utf-8"), compact(out).encode("utf-8")
        changed[group] += original_payload != payload
        require(typed_equal({k: v for k, v in record.items() if k != "metadata"},
                            {k: v for k, v in out.items() if k != "metadata"}), "nonmetadata fields changed")
        if kind == "node":
            require(key == out["id"] and not counts["edges"], "node ID/order changed")
            if key.startswith("CLM:"):
                counts["claims"] += 1
                original_science, new_science = protected_claim(record), protected_claim(out)
                require(typed_equal(original_science, new_science), "protected scientific/audit fields changed: " + key)
                protected_digest.update(compact(original_science).encode("utf-8") + b"\n")
            if key in issues:
                require(issues[key]["current_node_sha256"] == record_digest(record), "issue source witness changed")
                issues[key]["current_node_sha256"] = record_digest(out)
                issues[key].pop("current_graph", None)
                issues[key]["version_binding_status"] = "METADATA_COMPACTION_PENDING_VALIDATION"
                issues[key]["rollback_preimages_saved"] = False
                issues[key]["rollback_retention"] = False
                seen_issues.add(key)
            writer.node(key, payload)
        else:
            require(kind == "edge" and int(key) == counts["edges"], "edge ordinal/order changed")
            writer.edge(payload)
        digests[group].update(payload + b"\n")
        if on_progress and counts[group] % 250000 == 0:
            on_progress("BUILD_" + group.upper(), counts=dict(counts), changed=dict(changed))
    require(metadata is not None and seen_issues == set(issues), "missing metadata or current issue")
    actual_counts = {k: counts[k] for k in ("nodes", "claims", "edges")}
    require(actual_counts == baseline["counts"], "source graph counts changed")
    writer.finish(metadata)
    return dict(counts=actual_counts, changed_records=dict(changed), removals=dict(removals),
        metadata=metadata, source_metadata_key_unions={k: sorted(v) for k, v in before_unions.items()},
        metadata_key_unions={k: sorted(v) for k, v in after_unions.items()},
        record_digests={k: v.hexdigest() for k, v in digests.items()},
        protected_claim_digest=protected_digest.hexdigest(), protected_claims=counts["claims"],
        connected_components=baseline["connected_components"], isolated_nodes=baseline["isolated_nodes"],
        self_loops=baseline["self_loops"], deleted_claim_ids=[],
        issues=list(issues.values()))


def verify_compaction(records, expected, on_progress=None):
    """Independent full output topology/coverage/protected-evidence scan."""
    coverage = Coverage()
    digest = hashlib.sha256()
    ids = {r["claim_id"]: r["current_node_sha256"] for r in expected["issues"]}
    seen = set()
    counts = Counter()

    def observed():
        for kind, key, record in records:
            if kind != "metadata":
                scope = "edge/all" if kind == "edge" else "node/" + node_class(key, record)
                coverage.add(scope, record)
                if kind == "node" and key.startswith("CLM:"):
                    counts["protected_claims"] += 1
                    digest.update(compact(protected_claim(record)).encode("utf-8") + b"\n")
                if kind == "node" and key in ids:
                    require(record_digest(record) == ids[key], "output issue node hash differs")
                    seen.add(key)
            yield kind, key, record

    checks = verify_stream(observed(), expected, on_progress)
    require(digest.hexdigest() == expected["protected_claim_digest"], "protected evidence/audit digest differs")
    require(counts["protected_claims"] == expected["protected_claims"] and seen == set(ids), "protected count differs")
    checks.pop("references_to_deleted_ids", None)
    checks.pop("retained_records_identical", None)
    checks.update(output_records_match_expected=True, protected_science_and_audits_identical=True,
                  protected_claims=counts["protected_claims"], current_issue_hashes_verified=len(seen))
    return checks, coverage.rows(), dict(coverage.denominators)


def passing_tests():
    path = REVIEW_DIR / "COMPACTION_TEST_RESULTS.xml"
    total = Counter()
    for suite in ElementTree.parse(path).getroot().iter("testsuite"):
        total.update({k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
    require(total["tests"] >= 80 and not any(total[k] for k in ("failures", "errors", "skipped")), "passing complete tests required")
    return dict(counts=dict(total), artifact=journal.fingerprint(path))


def build():
    require(not TEMP.exists() and not (OUTPUT / "BUILD_STATE.json").exists(), "build already started; inspect state")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    review = journal.read_json(REVIEW_DIR / "REVIEW.json")
    require(campaign["current_graph"]["sha256"] == SOURCE_SHA and campaign["active_process"] is None,
            "current graph advanced or a KG worker is active")
    require(campaign["automation_id"] is None and campaign["rollback_retention"] is False, "retention/scheduling boundary changed")
    require(campaign["current_graph"] == review["graph"] and review["full_current_graph_sha_verified"], "review binding differs")
    require(Path(campaign["current_graph"]["path"]).resolve() == SOURCE.resolve(), "unexpected current graph target")
    require(journal.fingerprint(campaign["current_acceptance"]["path"]) == campaign["current_acceptance"], "acceptance changed")
    baseline = journal.read_json(campaign["current_acceptance"]["path"])
    require(baseline["graph"] == campaign["current_graph"] and baseline["counts"] == campaign["counts"], "acceptance/current mismatch")
    journal.guards([campaign["current_graph"], campaign["current_detail_store"], campaign["formal_sources"], campaign["current_issues"]])
    tests, code = passing_tests(), code_fingerprints()
    protected = [native_info(fp["path"]) for fp in campaign["formal_sources"].values()]
    source_native, detail_native = native_info(SOURCE), native_info(campaign["current_detail_store"]["path"])
    require(shutil.disk_usage(SOURCE).free > campaign["current_graph"]["bytes"] + 16 * 1024**3, "insufficient temporary space")
    OUTPUT.mkdir(exist_ok=True)
    campaign.update(status="MANUAL_ACTIVE", phase="已授权metadata精简：临时构建与全量验收", active_process={"kind": "metadata_compaction", "pid": os.getpid(), "state": str(OUTPUT / "RUN_STATE.json")}, updated_at=journal.utc_now())
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    progress("STARTING_APPROVED_METADATA_COMPACTION")
    issues = rows(campaign["current_issues"]["path"])
    with TEMP.open("xb", buffering=1024**2) as handle:
        with hashed_reader(SOURCE) as (reader, digest):
            result = stream_compact(walk_graph(reader), handle, baseline, issues, progress)
            require(digest.hexdigest() == SOURCE_SHA, "source full SHA differs")
        handle.flush()
        os.fsync(handle.fileno())
    guard_native([source_native, detail_native, *protected])
    require(code_fingerprints() == code, "code changed during build")
    state = dict(status="BUILT_NOT_ADOPTED", at=journal.utc_now(), source=campaign["current_graph"],
        source_native=source_native, detail=campaign["current_detail_store"], detail_native=detail_native,
        protected_formal_native=protected, temporary=cheap(TEMP), result=result, tests=tests, code=code,
        source_sha_verified=True, rollback_copies=0, record_preimages_saved=False)
    journal.atomic_json(OUTPUT / "BUILD_STATE.json", state)
    progress("BUILT_NOT_ADOPTED", counts=result["counts"], changed=result["changed_records"])


def validate():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    require(not (OUTPUT / "VALIDATED.json").exists(), "already validated")
    require(code_fingerprints() == state["code"], "code changed since build")
    require(cheap(TEMP) == state["temporary"], "temporary file changed")
    protected = [state["source_native"], state["detail_native"], *state["protected_formal_native"]]
    guard_native(protected)
    temp_native = native_info(TEMP)
    with hashed_reader(TEMP) as (reader, digest):
        checks, coverage, denominators = verify_compaction(walk_graph(reader), state["result"], progress)
        candidate_sha = digest.hexdigest()
    require(sha256(Path(state["detail"]["path"])) == state["detail"]["sha256"], "current SQLite SHA changed")
    guard_native([temp_native, *protected])
    require(code_fingerprints() == state["code"], "code changed during verification")
    write_rows(OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl", coverage)
    validated = dict(status="VALIDATED_NOT_ADOPTED", at=journal.utc_now(), graph={**cheap(TEMP), "sha256": candidate_sha},
        temporary_native=temp_native, detail_sha_verified=True, checks=checks, metadata_denominators=denominators,
        coverage=journal.fingerprint(OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl"),
        build_state=journal.fingerprint(OUTPUT / "BUILD_STATE.json"), rollback_retention=False)
    journal.atomic_json(OUTPUT / "VALIDATED.json", validated)
    progress("VALIDATED_NOT_ADOPTED", checks=checks)


def apply():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    accepted = journal.read_json(OUTPUT / "VALIDATED.json")
    require(not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "already adopted; use report")
    require(code_fingerprints() == state["code"] and journal.fingerprint(OUTPUT / "BUILD_STATE.json") == accepted["build_state"], "code/build changed")
    journal.guards(accepted["coverage"])
    guard_native([state["detail_native"], *state["protected_formal_native"]])
    require(SOURCE.resolve().parent == TEMP.resolve().parent == (journal.OUTPUT / "round23_source_deletion_candidate").resolve(), "exact replacement target escaped")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["current_graph"] == state["source"] and campaign["active_process"]["kind"] == "metadata_compaction", "campaign advanced")
    intent = OUTPUT / "ADOPTION_INTENT.json"
    if TEMP.exists():
        guard_native([state["source_native"], accepted["temporary_native"]])
        journal.atomic_json(intent, dict(at=journal.utc_now(), target=str(SOURCE), graph=accepted["graph"],
            temporary_native=accepted["temporary_native"], validated=journal.fingerprint(OUTPUT / "VALIDATED.json"), rollback_retention=False))
        os.replace(TEMP, SOURCE)  # User explicitly requested a single current KG, no backup.
    else:
        saved_intent = journal.read_json(intent)
        require(saved_intent["validated"] == journal.fingerprint(OUTPUT / "VALIDATED.json"), "missing/changed adoption intent")
    actual = native_info(SOURCE)
    for key in ("bytes", "mtime_ns", "file_id"):
        require(actual[key] == accepted["temporary_native"][key], "adopted file identity differs")
    graph = {**cheap(SOURCE), "sha256": accepted["graph"]["sha256"]}
    result = state["result"]
    for issue in result["issues"]:
        issue.update(current_graph=graph, version_binding_status="FULL_CURRENT_GRAPH_VALIDATED")
    write_rows(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl", result["issues"])
    receipt = dict(status="CURRENT_KG_METADATA_COMPACT_SINGLE_VERSION", at=journal.utc_now(), graph=graph,
        detail_store=state["detail"], counts=result["counts"],
        node_metadata_field_union=len(result["metadata_key_unions"]["nodes"]),
        edge_metadata_field_union=len(result["metadata_key_unions"]["edges"]),
        source_node_metadata_field_union=len(result["source_metadata_key_unions"]["nodes"]),
        source_edge_metadata_field_union=len(result["source_metadata_key_unions"]["edges"]),
        connected_components=result["connected_components"], isolated_nodes=result["isolated_nodes"], self_loops=result["self_loops"],
        changed_records=result["changed_records"], removals=result["removals"], removed_field_occurrences=sum(result["removals"].values()),
        graph_bytes_before=state["source"]["bytes"], graph_bytes_reduced=state["source"]["bytes"]-graph["bytes"],
        protected_claim_digest=result["protected_claim_digest"], validation=accepted["checks"], tests=state["tests"],
        metadata_denominators=accepted["metadata_denominators"], current_coverage=accepted["coverage"],
        current_issues=journal.fingerprint(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"),
        last_deep_verification=accepted["at"], rollback_retention=False, rollback_copies=0,
        deleted_nodes=0, deleted_edges=0, scientific_values_filled=0, audit_seals_reissued=0, formal_apply_performed=False,
        deferred=["current audit detail/profile offloading", "batch/provenance consolidation and external legacy-ID dependencies", "nonempty conflicting aliases and canonical-missing values", "metadata.id retained for reader compatibility"])
    journal.atomic_json(OUTPUT / "CURRENT_ACCEPTANCE.json", receipt)
    campaign.update(updated_at=receipt["at"], status="COMPLETED", phase="重复字段、空副本、过期导入状态与重复版本精简完成", active_process=None,
        current_acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), current_graph=graph,
        current_coverage=receipt["current_coverage"], current_issues=receipt["current_issues"],
        node_metadata_field_union=receipt["node_metadata_field_union"], edge_metadata_field_union=receipt["edge_metadata_field_union"],
        last_deep_verification=accepted["at"], last_current_graph_content_verification={"graph": graph, "independent_full_scan": True},
        metadata_compaction=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"))
    campaign["next_steps"] = ["本批仅精简重复存储、空副本、过期导入状态和重复版本，未改变节点/边/科学证据。",
        "后续可集中收口审核配置与详情、批次来源；非空冲突、唯一值及旧ID外部依赖不可直接删除。",
        "21条待复审与1204条基因/15条端点历史候选仍保留，队列可能重叠。",
        "正式full_v2尚未同步，正式图/声明库/明细/状态需一并明确采用，不擅自替换。"]
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    progress("ADOPTED_SINGLE_CURRENT_KG", graph=graph, counts=result["counts"])


def report():
    from prepare_kg_simplified_reading import document
    from report_current_kg import check_html, file_link
    current = journal.read_json(OUTPUT / "CURRENT_ACCEPTANCE.json")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(current["graph"] == campaign["current_graph"], "current graph advanced")
    journal.guards([current["graph"], current["detail_store"], current["current_coverage"], current["current_issues"], campaign["formal_sources"]])
    totals = Counter()
    for key, n in current["removals"].items():
        totals[key.split(":", 1)[0]] += n
    labels = {"exact_duplicate": "完全一致的重复字段", "empty_optional_slot": "可安全省略的空实例",
              "shared_version": "上提共享的重复版本", "retired_import_state": "完成导入后的旧状态"}
    log = journal.read_json(journal.OUTPUT / "WORK_LOG.json")
    if not any(event["id"] == EVENT_ID for event in log["events"]):
        log["events"].append(dict(id=EVENT_ID, at=current["at"], status="completed", title="metadata第一批集中精简完成：数据与读写一起收敛",
            body=f"节点/声明/边数量不变，省去{current['removed_field_occurrences']:,}处字段存储；节点/边metadata字段并集{current['source_node_metadata_field_union']}/{current['source_edge_metadata_field_union']}→{current['node_metadata_field_union']}/{current['edge_metadata_field_union']}。图文件减少{current['graph_bytes_reduced']/1024**3:.3f} GiB逻辑大小。",
            changes=f"{current['tests']['counts']['tests']}项测试通过，源图SHA、新图独立全量结构与科学/审核摘要、当前SQLite SHA通过。无节点/边删除，无证据改写，无新备份，正式full_v2未改。",
            artifacts=[str(OUTPUT / "CURRENT_ACCEPTANCE.json"), str(OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl"), str(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl")]))
    table = "".join(f"<tr><th>{labels[key]}</th><td>{value:,}</td></tr>" for key, value in sorted(totals.items()))
    timeline = "".join(f"<details><summary>{escape(e.get('at', ''))} — {escape(e['title'])}</summary><p>{escape(e.get('body', ''))}</p><p>{escape(e.get('changes', ''))}</p><p>{' · '.join(file_link(p) for p in e.get('artifacts', []))}</p></details>" for e in reversed(log["events"]))
    content = f'''<h1>KG 当前状态与工作记录</h1><p>更新：{escape(current['at'])}。当前优化工作图，正式 full_v2 尚未同步。</p>
<div class="notice">本批是metadata存储精简，不是科学结论修订；无模型或训练。没有新增历史KG或回退副本。</div>
<h2>当前唯一工作版本</h2><p>节点 2,587,871 · 其中声明 905,184 · 边 3,012,015。数量全部不变。</p>
<p>节点 metadata 字段并集：{current['source_node_metadata_field_union']} → {current['node_metadata_field_union']}；边：{current['source_edge_metadata_field_union']} → {current['edge_metadata_field_union']}。字段并集不代表每条记录都有这些字段。</p>
<p>当前图 {current['graph']['bytes']:,} 字节（{current['graph']['bytes']/1024**3:.3f} GiB），本批减少 {current['graph_bytes_reduced']:,} 字节（{current['graph_bytes_reduced']/1024**3:.3f} GiB逻辑文件大小，不是磁盘可用空间测量）。</p>
<p>{file_link(current['graph']['path'], '当前 KG JSON')} · {file_link(current['detail_store']['path'], '当前必要 UMLS 明细库')} · {file_link(OUTPUT / 'CURRENT_ACCEPTANCE.json', '当前验收记录')}</p>
<h2>本批实际精简</h2><table><tr><th>类别</th><th>字段出现次数</th></tr>{table}</table><p>以上不是不同节点数量；同一记录可涉及多个字段。</p>
<p>规范ID metadata.id仍保留，删除重复claim_id；论文只去相同别名，不向source_paper补入未确认值。两类归属标签分别保留一份，不互相合并。内层population的[]等会影响审核摘要的空值仍保留。</p>
<h2>验证与读写</h2><p>{current['tests']['counts']['tests']}项测试通过。源图边读边完整SHA-256验证；输出独立全量读取、节点身份/边所有者/拓扑检查及哈希验证；905,184条声明的科学字段、论文来源、证据和当前审核对象摘要一致。当前SQLite完整SHA-256一致。没有签发新审核结论。</p>
<p>已声明精简格式的图，读取后再次保存会继续应用精简规则，避免Claim兼容写入重新增加重复字段；旧格式默认行为不变。审核详情和来源批次仍保留，未声称metadata已经全部最小化。</p>
<h2>当前覆盖与待办</h2><p>{file_link(current['current_coverage']['path'], '全量实测metadata覆盖')} · {file_link(current['current_issues']['path'], '21条当前待复审声明')}</p>
<p>有效population、conditions、类型和统计值不因稀疏而删除。6,929对主语别名与6,929对宾语别名不一致、48对PMID不一致保留；这些是逐值/类型差异，不自动认定科学错误。唯一文本、唯一论文信息及来源回退ID保留。</p>
<p>后续重点：审核配置/详情去重外置、批次来源收口及旧ID外部依赖。21条待复审、1204条基因声明和15条端点历史候选仍保留，队列可能重叠。metadata精简不代表全图无错误。</p>
<h2>保留策略与正式应用</h2><p>只有当前工作KG和当前必要明细，无逐步回退。旧工作图已被验证后的临时文件原子替换，不能本地逐步回退。此前释放99.608 GiB的历史副本清理另见历史事件，本批不重复计入。</p>
<p>正式full_v2、原始文献和实验快照未修改；正式声明905,274条，工作图905,184条，差额是此前累计删除的90条，不是本批新增删除。</p>
<h2>历史工作日志</h2><p>以下为发生时的记录；已删除的历史文件不会显示为可用回退。</p>{timeline}'''
    page = document("KG 当前状态与工作记录", content)
    qa = check_html(page)
    journal.atomic_json(journal.OUTPUT / "WORK_LOG.json", log)
    journal.atomic_text(journal.REPORT, page)
    journal.atomic_json(OUTPUT / "REPORT_QA.json", qa)
    journal.atomic_text(journal.OUTPUT / "HANDOFF.md", f"""# Current KG — metadata compaction, single-version retention

Updated {current['at']}. Authoritative receipt: {OUTPUT / 'CURRENT_ACCEPTANCE.json'}.
Current graph: {current['graph']['path']}
Current SQLite: {current['detail_store']['path']}
Counts unchanged: 2587871 nodes / 905184 claims / 3012015 edges.
Metadata unions {current['node_metadata_field_union']}/{current['edge_metadata_field_union']}.
{current['removed_field_occurrences']} field occurrences removed; {current['graph_bytes_reduced']} logical bytes saved.
{current['tests']['counts']['tests']} tests passed. Full source SHA, independent full output structure/SHA,
protected scientific/audit digest and unchanged SQLite SHA verified.
Only exact duplicates, selected optional empty slots, completed-import booleans,
and shared membership-version occurrences removed. No node/edge deletion, source
imputation, audit resealing, model/training or formal full_v2 mutation.
Current coverage and 21 held-issue hashes updated in round27_metadata_compaction.
Current reader/writer supports metadata_layout={LAYOUT['version']}; legacy default unchanged.
metadata.id, conflicting/unique values, audit details, batch provenance and old-ID
dependencies remain. Future work must not call metadata fully minimized yet.
No historical KG or rollback preimages retained; temporary output replaced the
sole current graph. Do not rerun old R23/R26 builders or use their old graph hashes.
CAMPAIGN.json points at current R27 acceptance. HTML: KG_OVERNIGHT_20260907.html.
HTML structure/local links checked; no browser visual QA. No active worker/timer.
""")
    journal.atomic_json(OUTPUT / "RUN_STATE.json", dict(status="COMPLETED", at=journal.utc_now(), graph=current["graph"],
        counts=current["counts"], active_process=None, rollback_copies=0, report=str(journal.REPORT)))
    print(compact(dict(status="METADATA_COMPACTION_COMPLETE", receipt=str(OUTPUT / "CURRENT_ACCEPTANCE.json"), **qa)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "apply", "report"))
    {"build": build, "validate": validate, "apply": apply, "report": report}[parser.parse_args().command]()
