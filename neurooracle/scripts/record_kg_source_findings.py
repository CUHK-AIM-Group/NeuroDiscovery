"""Bind manually reviewed source issues to frozen current records, not repairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from retire_kg_duplicate_references import guards, deep_check, fingerprint, jsonl, read_json, same_record
from audit_kg_integrity import JsonlWriter

OUTPUT = journal.OUTPUT / "round05_source_lineage"
NOTES = OUTPUT / "MANUAL_SOURCE_NOTES.json"
CODE = ("neurooracle/scripts/record_kg_source_findings.py", "neurooracle/tests/test_kg_source_findings.py")
ACC = "CLM:f3d09d6c004773ecd6430383f39584ce"
PROTECTED_OTHER = "CLM:4cb07935b48a903e0d2fcae6dca92d2d"


def record_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def issue(node, category, reason, **extra):
    return {"claim_id": node["id"], "category": category, "reason": reason,
            "current_record_sha256": record_digest(node), "current_record": node,
            "current_audit_decision": (node["metadata"].get("scope_reaudit") or {}).get("decision"),
            "decision": "held_for_source_or_semantic_correction_not_applied",
            "whole_paper_or_universal_falsity_certified": False, "source_record_edited": False,
            "new_valid_audit_seal_created": False, **extra}


def batch_findings(pairs, links, current, notes):
    if notes["graph_mutation_authorized_by_this_document"] is not False or len(notes["rows"]) != 50 or {r["index"] for r in notes["rows"]} != set(range(51, 101)):
        raise ValueError("manual review must cover exactly batch positions 51-100 without mutation authority")
    by_index = {r["batch_index"]: r for r in pairs}
    results = []
    for note in notes["rows"]:
        pair = by_index[note["index"]]
        matching = [r for r in links if r["batch_index"] == note["index"] and r["exact_archive_lineage"]]
        if len(matching) != 1:
            raise ValueError("reviewed archive record lacks exactly one verified current lineage")
        node = current[matching[0]["claim_id"]]
        abstract = pair["source_item"]["abstract"]
        old = pair["archived_claim"]
        text = " ".join(old[key] for key in ("subject", "predicate", "object", "evidence_text"))
        if not note["source_anchor"] or note["source_anchor"].casefold() not in abstract.casefold():
            raise ValueError("manual source anchor not present in complete saved abstract")
        if not note["claim_anchor"] or note["claim_anchor"].casefold() not in text.casefold() or note["claim_anchor"].casefold() in abstract.casefold():
            raise ValueError("manual claim/source contrast anchor differs")
        if node["metadata"].get("raw_text") != old["evidence_text"] or (node["metadata"].get("source_paper") or {}).get("pmid") != pair["pmid"]:
            raise ValueError("current issue evidence differs from exact source lineage")
        results.append(issue(node, "unsupported_by_bound_input_abstract", note["reason"],
            batch_index=note["index"], pmid=pair["pmid"], source_title=pair["source_item"]["title"],
            source_abstract=abstract, archived_claim=old, writer_line=pair["writer_line"],
            reviewed_anchors={k: note[k] for k in ("source_anchor", "claim_anchor")},
            support_scope="complete_saved_extraction_input_not_unseen_full_paper"))
    return results


def other_findings(archives, current):
    results = []
    old_case3 = [r for r in archives if r["source_kind"] == "case3_claims"]
    if len(old_case3) != 5:
        raise ValueError("reviewed five-claim type cohort changed")
    for row in old_case3:
        old = row["record"]
        node = current[old["claim_id"]]
        md = node["metadata"]
        if old.get("subject_type") != "DRUG" or md.get("subject_type") != "DRUG" or (md.get("metadata") or {}).get("subject_type") != "DRUG":
            raise ValueError("reviewed pre-existing DRUG declaration changed")
        if old["subject"] != md["subject_name"] or old["object"] != md["object_name"] or old["raw_text"] != md["raw_text"] or not same_record(old["source_paper"], md["source_paper"]):
            raise ValueError("current case3 evidence no longer matches the archived witness")
        results.append(issue(node, "non_drug_subject_declared_as_drug", "该记录的主语是疾病/症状、人群比较、临床量表分数或脑连接，而非药物。DRUG 声明在上游提取记录中已经存在；类型和谓词应独立重审，不能保留旧 finalized 凭据后直接改字段。",
                             archived_source=row, support_scope="stored_subject_and_evidence_type_consistency_only"))
    old_acc = next(r for r in archives if r["source_kind"] == "acc_final_claims" and r["record"].get("id") == ACC)
    old = old_acc["record"]
    node, other = current[ACC], current[PROTECTED_OTHER]
    md = node["metadata"]
    paper = next(r for r in archives if r["source_kind"] == "acc_source_papers")["record"]
    for key in ("subject_name", "object_name", "predicate", "raw_text", "source_paper", "evidence", "scope_reaudit"):
        if not same_record(md.get(key), old.get(key)):
            raise ValueError("ACC issue changed since the archived selected record")
    if md["raw_text"] != other["metadata"]["raw_text"] or md["object_name"] == other["metadata"]["object_name"] or md["predicate"] == other["metadata"]["predicate"]:
        raise ValueError("reviewed nonduplicate repeated-text pair changed")
    if md["raw_text"] not in paper["abstract"] or "greater dorsal ACC global efficiency than the low-risk group" not in paper["abstract"]:
        raise ValueError("saved source abstract no longer contains the reviewed distinct results")
    results.append(issue(node, "stored_evidence_sentence_attached_to_wrong_result", "当前 SPO 指向背侧 ACC 全局效率增加，附句却描述膝下 ACC 局部效率/聚类系数降低。保存的输入摘要另有对应背侧 ACC 结果；错误附句在上游选定记录中已存在。相同原文的另一条记录不是重复项，不删除也不互相捐赠证据。",
                         archived_source=old_acc, bound_source_paper=paper, protected_nonduplicate_claim_id=PROTECTED_OTHER,
                         support_scope="stored_result_to_sentence_alignment_only"))
    return results


def related_edges(issues, edges):
    ids = {r["claim_id"] for r in issues}
    selected = [r for r in edges if r["edge"]["source_id"] in ids or r["edge"]["target_id"] in ids or (r["edge"].get("metadata") or {}).get("claim_id") in ids]
    ordinals = [r["source_candidate_edge_ordinal"] for r in selected]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("duplicate related edge ordinal")
    return selected


def write_rows(name, rows):
    writer = JsonlWriter(OUTPUT / name)
    for row in rows:
        writer.add(row)
    return writer.close()


def render_report(rows, impact, summary):
    categories = summary["category_counts"]
    body = ["# 来源追溯与当前图问题登记", "", "本轮是问题登记，不是修复验收；没有修改节点、边、原文或审核字段。", "",
            f"完整扫描复核当前图后，在 113 个定点 claim 中登记 {len(rows)} 个不同 claim 的问题。类别：{json.dumps(categories, ensure_ascii=False)}。", "",
            "100 条旧批次输出全部有精确当前来源链。逐条阅读第 51–100 项的完整保存摘要后，确认这 50 条输出不受各自提取输入支持。第 1–50 项尚未逐条科学认证；不能把本次结果推广为全部批次或整个图的错误率。", "",
            "三层证据：完整旧输入摘要 → 按数组位置把摘要身份与人工 specs 配对的旧脚本文本 → 旧输出 → 当前完整 claim。只解析 JSON 字面值，未运行旧脚本。当前问题原文与旧输出逐字相同，因此这些错配不是本次 KG 身份补丁制造的。", "",
            f"涉及 {len(impact)} 条当前相关边；这里只登记影响范围，不代表这些边已被隔离或均应直接删除。", "",
            f"与已知基因待审队列交集 {summary['overlap_gene_queue_claims']} 条，与 15 条原端点队列交集 {summary['overlap_opposite_queue_claims']} 条；不要把各队列简单相加为全图问题总数。", "",
            "## 第 51–100 项逐条来源对照", "", "| 批内序号 | PMID | 不对应之处 |", "|---|---|---|"]
    for row in rows:
        if "batch_index" in row:
            body.append(f"| {row['batch_index']} | {row['pmid']} | {row['reason']} |")
    body += ["", "## 其他六条问题", ""]
    for row in rows:
        if "batch_index" not in row:
            body += [f"- `{row['claim_id']}`：{row['reason']}"]
    body += ["", "## 保留边界与下一步", "", "- 这些判定针对保存的提取输入或原有类型/附句，不声称其科学陈述在所有情境都为假，也不认证未查看的论文全文。",
             f"- {summary['existing_finalized_audit_claims']} 条已登记记录仍有 finalized 审核标签；来源对应异常说明标签存在不等于内容正确。任何语义修正必须保留旧像并使旧证据绑定失效，不伪造新凭据。",
             "- 本轮没有隔离这些 claim，没有重跑模型，没有猜新 PMID，也没有把别条 claim 的文字复制成新证据。",
             "- 继续逐条来源定位；若要落地隔离，应另做独立可回退方案、全部入出引用检查、统计/回归和明确的审核状态处理。",
             "- 当前图数量不变：2,587,961 节点、905,274 claims、3,012,308 边；metadata 顶层键并集 130/37。此前 Round 03 安全身份补丁仍暂存，与本问题清单独立。", ""]
    return "\n".join(body)


def build():
    if (OUTPUT / "FINDINGS_BUILD.json").exists():
        raise ValueError("findings already built and bound")
    receipt, inputs = read_json(OUTPUT / "CURRENT_SOURCE_COMPLETE.json"), read_json(OUTPUT / "INPUTS.json")
    guards(receipt)
    guards(inputs)
    for fp in [*receipt["artifacts"].values(), *inputs["artifacts"].values()]:
        deep_check(fp)
    notes = read_json(NOTES)
    current = {r["id"]: r for r in jsonl(OUTPUT / "CURRENT_CLAIMS.jsonl")}
    rows = batch_findings(list(jsonl(OUTPUT / "BATCH_SOURCE_LINEAGE.jsonl")), list(jsonl(OUTPUT / "CURRENT_BATCH_LINKS.jsonl")), current, notes)
    rows += other_findings(list(jsonl(OUTPUT / "TARGETED_ARCHIVE_RECORDS.jsonl")), current)
    if len(rows) != 56 or len({r["claim_id"] for r in rows}) != 56:
        raise ValueError("reviewed 50+5+1 unique-claim issue scope differs")
    impact = related_edges(rows, list(jsonl(OUTPUT / "CURRENT_RELATED_EDGES.jsonl")))
    accepted = read_json(Path(inputs["source_acceptance"]["path"]))
    gene_fp = accepted["artifacts"]["PENDING_GENE_LINKS.jsonl"]
    deep_check(gene_fp)
    genes = {r["claim_id"] for r in jsonl(Path(gene_fp["path"]))}
    opposite_path = journal.OUTPUT / "round01_shared_and_queue_review/OPPOSITE_ENDPOINT_HOLDS.jsonl"
    opposite = {r["claim_id"] for r in jsonl(opposite_path)}
    ids = {r["claim_id"] for r in rows}
    summary = {"status": "SOURCE_FINDINGS_REGISTERED_NOT_CORRECTED", "created_at": journal.utc_now(),
        "unique_claims_with_registered_issues": len(rows), "category_counts": dict(Counter(r["category"] for r in rows)),
        "related_edges": len(impact), "related_edge_relations": dict(Counter(r["edge"]["relation_type"] for r in impact)),
        "overlap_gene_queue_claims": len(ids & genes), "overlap_opposite_queue_claims": len(ids & opposite),
        "existing_finalized_audit_claims": sum(r["current_audit_decision"] == "finalized" for r in rows),
        "graph_changes": 0, "graph_quarantine_applied": False, "safe_round03_patch_bundle_changed": False,
        "models_called": 0, "training_jobs_started": 0, "formal_apply_performed": False,
        "source_receipt": fingerprint(OUTPUT / "CURRENT_SOURCE_COMPLETE.json"), "source_inputs": fingerprint(OUTPUT / "INPUTS.json"),
        "source_graph": receipt["source_graph"], "source_counts": receipt["counts"],
        "manual_notes": fingerprint(NOTES), "gene_queue": gene_fp, "opposite_queue": fingerprint(opposite_path),
        "implementation": {name: fingerprint(journal.REPO / name) for name in CODE}}
    artifacts = {"REGISTERED_SOURCE_ISSUES.jsonl": write_rows("REGISTERED_SOURCE_ISSUES.jsonl", rows),
                 "ISSUE_RELATED_EDGES.jsonl": write_rows("ISSUE_RELATED_EDGES.jsonl", impact)}
    journal.atomic_text(OUTPUT / "SOURCE_FINDINGS.md", render_report(rows, impact, summary))
    artifacts["SOURCE_FINDINGS.md"] = fingerprint(OUTPUT / "SOURCE_FINDINGS.md")
    summary["artifacts"] = artifacts
    guards(summary)
    journal.atomic_json(OUTPUT / "FINDINGS_BUILD.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "unique_claims_with_registered_issues", "related_edges", "overlap_gene_queue_claims", "overlap_opposite_queue_claims")}, ensure_ascii=True))


def accept():
    if (OUTPUT / "SOURCE_REVIEW_COMPLETE.json").exists():
        raise ValueError("source review already frozen")
    built = read_json(OUTPUT / "FINDINGS_BUILD.json")
    guards(built)
    for fp in [*built["artifacts"].values(), *built["implementation"].values(), built["manual_notes"]]:
        deep_check(fp)
    test_path = OUTPUT / "FINDINGS_TEST_RESULTS.xml"
    suites = list(ElementTree.parse(test_path).getroot().iter("testsuite"))
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    if tests["tests"] < 15 or any(tests[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("source-finding binding tests must all pass")
    result = {"status": "SOURCE_REVIEW_COMPLETE_ISSUES_HELD_NOT_REPAIRED", "completed_at": journal.utc_now(),
        "build": fingerprint(OUTPUT / "FINDINGS_BUILD.json"), "tests": tests, "test_artifact": fingerprint(test_path),
        "registered_issues": built["unique_claims_with_registered_issues"], "category_counts": built["category_counts"],
        "current_related_edges": built["related_edges"], "graph_changes": 0, "models_called": 0, "training_jobs_started": 0}
    journal.atomic_json(OUTPUT / "SOURCE_REVIEW_COMPLETE.json", result)
    journal.append_event({"id": "round05-source-issues-reviewed", "status": "held", "at": result["completed_at"],
        "title": "重要发现：登记 50 条输入摘要错配、5 条 DRUG 错标、1 条证据附句错误",
        "body": "对旧批次后 50 项逐条阅读保存的完整摘要，并与旧输出及当前图一一绑定，确认各自提取输入不支持所附 claim。另找到同一青少年研究的 5 个非药物主语全被标为 DRUG，以及 ACC 结果附错证据句。三类问题共 56 个不同 claim，均可追溯到本次优化前的上游记录；原有 finalized 标签不能代替来源正确性检查。",
        "changes": "审查已完成，但问题尚未修复或隔离；不改写证据、不造新审核、不把相同原文的不同 ACC 指标去重。当前节点/边/metadata 统计保持不变。",
        "details": {"unique_issue_claims": 56, "related_edges": built["related_edges"], "queue_overlaps": {"gene": built["overlap_gene_queue_claims"], "opposite": built["overlap_opposite_queue_claims"]}, "tests": tests, "quarantine_applied": False},
        "artifacts": [str(OUTPUT / "SOURCE_FINDINGS.md"), str(OUTPUT / "REGISTERED_SOURCE_ISSUES.jsonl"), str(OUTPUT / "SOURCE_REVIEW_COMPLETE.json"), str(test_path)]})
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "accept"))
    args = parser.parse_args()
    (build if args.phase == "build" else accept)()
