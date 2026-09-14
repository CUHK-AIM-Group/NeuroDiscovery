"""R24: current coverage and usable, lossless read-only KG simplification.

Reuses sealed R23/R15 proofs; no graph rewrite, full graph copy or model call.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import Bindings, require, file_identity
from reclaim_kg_backup_storage import native_info
from retire_kg_duplicate_references import guards, same_record, jsonl
from project_kg_current_metadata_coverage import apply_delta, pack_rows, validate_rows, unions, SPOTLIGHT
from review_kg_metadata_contract import classify
from neurooracle.src.metadata_field_audit import Coverage, nonempty, PLACEHOLDERS, node_class
from neurooracle.src.kg_reading_view import make_view, restore_record, render_record, digest, field_group, edge_layer, iter_visible_edges, LAYERS

OUTPUT = journal.OUTPUT / "round24_structure_metadata_simplification"
R23 = journal.OUTPUT / "round23_source_deletion_candidate"
R15 = journal.OUTPUT / "round15_current_metadata_coverage"
R23_SHA = "04c78fed8ba6173ca30f6dd47ad1874efca44330d7c858915b5b31554cebb01d"
R15_SHA = "f593427ec302c22b72d5c0e42242587f421779e40372f688cc8125fbac518a50"
READS_SHA = "6b1bbe69b65d7950928b95672a15b11331dc79f3be8b303f35077af04236bb5a"
CODE = ("neurooracle/src/kg_reading_view.py", "neurooracle/scripts/prepare_kg_simplified_reading.py",
        "neurooracle/tests/test_kg_simplified_reading.py")


def verify_boundary(accepted, validation, plan, prior, prior_build):
    proof = prior_build["current_version_proof"]
    require(accepted["status"] == "SOURCE_MISMATCH_DELETION_CANDIDATE_ACCEPTED_NOT_FORMALLY_APPLIED", "R23 acceptance missing")
    require(validation["status"] == "FULL_CANDIDATE_AND_EXACT_BYTE_ROLLBACK_VERIFIED", "R23 full validation missing")
    require(prior["status"] == "CURRENT_CANDIDATE_METADATA_COVERAGE_INDEPENDENTLY_VERIFIED_NO_GRAPH_MUTATION", "R15 coverage not accepted")
    require(accepted["upstream_acceptance"] == proof["current_acceptance"] == plan["current_acceptance"], "R12 coverage/graph mismatch")
    require(file_identity(plan["source_graph"]) == file_identity(proof["current_graph"]), "coverage source graph changed")
    require(validation["candidate_graph"] == accepted["artifacts"]["knowledge_graph.candidate.json"], "current graph changed")
    require(accepted["plan"] == validation["plan"] and accepted["build"] == validation["build"], "R23 plan/build link changed")
    require(validation["all_retained_records_identical"] is True and validation["full_original_file_recoverable"] is True, "retained-record/rollback proof missing")
    for key, number in {"changed_nodes": 0, "deleted_nodes": 50, "deleted_edges": 149, "added_nodes": 0, "added_edges": 0}.items():
        require(type(validation[key]) is int and validation[key] == number, "unexpected R23 graph delta: " + key)
    require(validation["rollback"]["sha256"] == plan["source_graph"]["sha256"], "full R12 rollback differs")
    require(validation["rollback"]["counts"] == prior["counts"] == proof["current_counts"], "R12 denominators differ")
    require(validation["counts"] == accepted["counts"], "R23 counts differ")
    for kind in ("nodes", "edges"):
        require(validation["retained_record_digests"][kind] == plan["expected_record_digests"]["retained_" + kind], "retained digest differs")
        require(validation["rollback"]["record_digests"][kind] == plan["expected_record_digests"]["before_" + kind], "source digest differs")
    require(accepted["formal_apply_performed"] is False, "formal scope changed")


def deletion_records(nodes, edges):
    records, node_ids = [], set()
    for kind, rows, ordinal_key, scope in (("node", nodes, "source_node_ordinal", "node/claim"),
                                          ("edge", edges, "source_edge_ordinal", "edge/all")):
        seen = set()
        for row in rows:
            ordinal, record = row[ordinal_key], row["before"]
            require(type(ordinal) is int and ordinal > 0 and ordinal not in seen, "duplicate/invalid deletion ordinal")
            seen.add(ordinal)
            require(row["after"] is None and digest(record) == row["before_sha256"], "deletion preimage changed")
            if kind == "node":
                require(record["id"] == row["node_id"] and node_class(record["id"], record) == "claim", "deletion node is not the selected claim")
                require(record["id"] not in node_ids, "duplicate deletion ID")
                node_ids.add(record["id"])
            records.append({"kind": kind, "source_ordinal": ordinal, "scope": scope, "record": record})
    require(len(nodes) == 50 and len(edges) == 149, "incomplete 50/149 deletion delta")
    return records


def independent_subtract(rows, den, records):
    """Separate full-preimage interpreter, not Coverage.add/apply_delta."""
    original = validate_rows(rows, den)
    observations = defaultdict(list)
    removed = Counter()
    for item in records:
        scope, record = item["scope"], item["record"]
        removed[scope] += 1
        for key, value in record.items():
            if key != "metadata":
                observations[scope, "top." + key].append(value)
        md = record.get("metadata") or {}
        require(isinstance(md, dict), "invalid metadata container")
        for key, value in md.items():
            observations[scope, "metadata." + key].append(value)
            if isinstance(value, dict):
                for child, nested in value.items():
                    observations[scope, "metadata." + key + "." + child].append(nested)
    require(set(observations) <= set(original), "deletion contains uncensused fields")
    fields, types = {}, {}
    for key, row in original.items():
        values = observations.get(key, [])
        fields[key] = {"present": row["present"] - len(values),
                       "nonempty": row["nonempty"] - sum(nonempty(v) for v in values),
                       "placeholder_like": row["placeholder_like"] - sum(isinstance(v, str) and v.strip().casefold() in PLACEHOLDERS for v in values)}
        removed_types = Counter(type(v).__name__ for v in values)
        require(set(removed_types) <= set(row["types"]), "uncensused JSON type")
        types[key] = {name: count - removed_types[name] for name, count in row["types"].items()}
    current_den = {scope: count - removed[scope] for scope, count in den.items()}
    return pack_rows(fields, types, current_den), current_den


def project_coverage(rows, den, records, expected_counts):
    minus, plus = Coverage(), Coverage()
    for item in records:
        minus.add(item["scope"], item["record"])
    current, current_den = apply_delta(rows, den, minus, plus)
    other, other_den = independent_subtract(rows, den, records)
    require(same_record(current, other) and current_den == other_den, "independent coverage differs")
    restored, restored_den = apply_delta(current, current_den, plus, minus)
    require(same_record(restored, sorted(rows, key=lambda r: (r["scope"], r["field"]))) and restored_den == den, "full counter/type inverse differs")
    require(sum(v for k, v in current_den.items() if k.startswith("node/")) == expected_counts["nodes"]
            and current_den["node/claim"] == expected_counts["claims"] and current_den["edge/all"] == expected_counts["edges"], "R23 count reconciliation differs")
    old = validate_rows(rows, den)
    now = validate_rows(current, current_den)
    require(set(old) == set(now), "field scope emerged/disappeared")
    changed = [{"scope": k[0], "field": k[1], "before": old[k], "after": now[k]}
               for k in sorted(now) if not same_record(old[k], now[k])]
    return current, current_den, changed


def structure_summary(topology):
    counts = Counter()
    for relation, count in topology["relations"].items():
        require(type(count) is int and count > 0, "invalid relation count")
        counts[edge_layer({"relation_type": relation})] += count
    require(sum(counts.values()) == topology["edges"], "incomplete relation counts")
    return {"stored_edges": topology["edges"], "layers": dict(counts), "default_visible_edges": counts["relations"],
            "default_collapsed_edges": counts["claim_anchors"] + counts["mappings"],
            "default_collapsed_pct": round(100*(counts["claim_anchors"] + counts["mappings"])/topology["edges"], 6),
            "all_layers_restore_original_order_and_duplicates": True,
            "meaning": "display_selection_not_deletion_or_scientific_admission", "nodes_deleted": 0, "edges_deleted": 0}


def load_inputs():
    b = Bindings()
    c = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(Path(c["current_acceptance"]["path"]) == R23 / "REPAIR_ACCEPTANCE.json" and c["current_acceptance"]["sha256"] == R23_SHA, "current acceptance advanced")
    require(c["automation_id"] is None and c["active_process"] is None, "another worker/timer is active")
    accepted = b.json(c["current_acceptance"])
    validation, plan = b.json(accepted["validation"]), b.json(accepted["plan"])
    prior = journal.read_json(b.pin(R15 / "REVIEW_COMPLETE.json", R15_SHA))
    prior_build = b.json(prior["build"])
    verify_boundary(accepted, validation, plan, prior, prior_build)
    guards((accepted, plan, prior_build["current_version_proof"], c["formal_sources"]))
    for fp in [*prior_build["implementation"].values(), *plan["implementation"].values(), prior["test_results"]]:
        b.check(fp)
    require([native_info(x["path"]) for x in plan["protected_native"]] == plan["protected_native"], "formal/R12 native guards changed")
    current_native = [native_info(accepted["artifacts"][n]["path"]) for n in ("knowledge_graph.candidate.json", "umls_details.sqlite", "NODE_REPAIRS.jsonl")]
    for info in current_native:
        fp = next(f for f in accepted["artifacts"].values() if f["path"] == info["path"])
        require((info["bytes"], info["mtime_ns"]) == (fp["bytes"], fp["mtime_ns"]), "current native/accepted fingerprint mismatch")
    rows = b.rows(prior_build["artifacts"]["CURRENT_METADATA_COVERAGE.jsonl"])
    require(len(rows) == prior["coverage_rows"] == 453, "R15 census incomplete")
    records = deletion_records(b.rows(plan["artifacts"]["DELETE_NODE_PREIMAGES.jsonl"]), b.rows(plan["artifacts"]["DELETE_EDGE_PREIMAGES.jsonl"]))
    reads = journal.read_json(b.pin(journal.OUTPUT / "round04_metadata_reading_contract/REVIEWED_CODE_READS.json", READS_SHA))
    require(len(reads) == 6, "code-use witness set incomplete")
    for item in reads:
        path = b.check(item["fingerprint"])
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        require("\n".join(lines[item["start"]-1:item["end"]]) == item["source_excerpt"], "reviewed consumer changed")
    # This helper is reused for labels only; all claims of code use stay tied
    # to the six unchanged source blocks above, not to substring searches.
    b.pin(journal.REPO / "neurooracle/scripts/review_kg_metadata_contract.py")
    nodes = b.rows(accepted["artifacts"]["CURRENT_PROTECTED_NODES.jsonl"])
    edges = b.rows(accepted["artifacts"]["CURRENT_PROTECTED_EDGES.jsonl"])
    isolates = b.rows(accepted["artifacts"]["NEWLY_ISOLATED_NODES.jsonl"])
    issues = b.rows(accepted["artifacts"]["CURRENT_REMAINING_ISSUES.jsonl"])
    companion = b.json(c["round23_isolated_companion_review"])
    require(companion["accepted_candidate"] == c["current_acceptance"] and companion["full_graph_validation"] == accepted["validation"], "isolate review version changed")
    require(companion["nodes_with_offloaded_atom_reference"] == len(isolates) == 61
            and companion["offloaded_atom_references"] == 81 and companion["core_atom_references"] == 2, "isolate dependency evidence changed")
    require(len(nodes) == 62 and len(edges) == 186 and Counter(i["classification"] for i in issues) == {"confirmed": 40, "additional_review": 21}, "protected/issue cohort incomplete")
    proof = {"current_acceptance": c["current_acceptance"], "current_graph": validation["candidate_graph"],
             "current_full_validation": accepted["validation"], "full_graph_verified_at": validation["candidate_full_verified_at"],
             "full_companion_and_formal_hash_verified_at": validation["completed_at"], "prior_coverage": journal.fingerprint(R15 / "REVIEW_COMPLETE.json"),
             "prior_counts": prior["counts"], "current_counts": accepted["counts"],
             "deletion_preimages": {n: plan["artifacts"][n] for n in ("DELETE_NODE_PREIMAGES.jsonl", "DELETE_EDGE_PREIMAGES.jsonl")},
             "retained_record_digests": validation["retained_record_digests"], "exact_source_rollback": validation["rollback"],
             "current_native_start": current_native, "protected_native": plan["protected_native"],
             "formal_sources": c["formal_sources"], "companion_dependency_review": c["round23_isolated_companion_review"],
             "full_graph_scanned_or_rehashed_this_round": False, "large_file_integrity_method": "reuse_accepted_SHA_with_unchanged_size_mtime_and_intra_round_native_guards",
             "scientific_validity_certified": False, "live_experiment_usage_measured": False}
    return dict(bindings=b, campaign=c, accepted=accepted, validation=validation, prior=prior, rows=rows, records=records,
                reads=reads, nodes=nodes, edges=edges, isolates=isolates, issues=issues, companion=companion, proof=proof)


def witness_views(data):
    issues = {row["claim_id"]: row for row in data["issues"]}
    require(len(issues) == len(data["issues"]) == 61, "duplicate/incomplete issue IDs")
    nodes = {row["id"]: row for row in data["nodes"]}
    for row in data["isolates"]:
        require(digest(row["node"]) == row["record_sha256"], "isolate record differs")
        require(row["node"]["id"] not in nodes, "duplicate witness node")
        nodes[row["node"]["id"]] = row["node"]
    for row in data["companion"]["full_current_referencing_atom_witnesses"]:
        require(digest(row["node"]) == row["sha256"] == row["full_scan_reference"]["record_sha256"], "atom full record differs")
        require(row["node"]["id"] not in nodes, "duplicate atom witness")
        nodes[row["node"]["id"]] = row["node"]
    for cid, issue in issues.items():
        require(issue["current_graph"] == data["proof"]["current_graph"] and digest(nodes[cid]) == issue["current_node_sha256"], "known-issue view points to another record")
        require(issue["repaired"] is False and issue["deleted"] is False and issue["quarantined"] is False, "known-issue disposition changed")
    def notices(cid):
        if cid not in issues:
            return ["本记录未被这份问题清单标为确认错误；不等于已全面核实科学正确性。"]
        item = issues[cid]
        label = "已确认问题，尚未修复" if item["classification"] == "confirmed" else "待复审，不是已确认错误"
        return [label + "：" + item["reason"], "旧审核状态保留；不是本轮重新认证。" + item["preparation_action"]]
    output = []
    isolated_ids = {r["node"]["id"] for r in data["isolates"]}
    for nid, node in nodes.items():
        note = ["图内度数为 0，但仍是离线原子的来源父概念，必须保留。"] if nid in isolated_ids else notices(nid)
        output.append({"identity": nid, "kind": "node", "view": make_view(node, notices=note)})
    for row in data["edges"]:
        edge = row["edge"]
        cid = (edge.get("metadata") or {}).get("claim_id")
        if edge.get("relation_type") == "about":
            cid = edge.get("source_id") if str(edge.get("source_id", "")).startswith("CLM:") else edge.get("target_id")
        output.append({"identity": "edge:" + str(row["candidate_edge_ordinal"]), "kind": "edge", "layer": edge_layer(edge),
                       "view": make_view(edge, kind="edge", notices=notices(cid))})
    require(len(nodes) == 125 and len(output) == 311 and len({r["identity"] for r in output}) == len(output), "witness set differs")
    originals = list(nodes.values()) + [r["edge"] for r in data["edges"]]
    for row, original in zip(output, originals):
        restored = restore_record(row["view"])
        require(same_record(restored, original) and json.dumps(restored, ensure_ascii=False) == json.dumps(original, ensure_ascii=False), "full type/order round trip failed")
    raw_edges = [r["edge"] for r in data["edges"]]
    require(same_record(list(iter_visible_edges(raw_edges, layers=LAYERS)), raw_edges), "all-layer edge order/content changed")
    return output


def compute(data):
    current, den, changed = project_coverage(data["rows"], data["prior"]["current_denominators"], data["records"], data["accepted"]["counts"])
    union = unions(current)
    accepted_union = {key: data["validation"]["metadata_key_unions"][key + "s"] for key in ("node", "edge")}
    require(union == unions(data["rows"]) == accepted_union
            and len(union["node"]) == 130 and len(union["edge"]) == 37, "field unions changed")
    index = validate_rows(current, den)
    spotlight = [{"label": label, **index["node/claim", "metadata." + field]} for label, field in SPOTLIGHT]
    catalogue = [{**classify(row), "actual_reading_group": field_group(row["field"].removeprefix("top.").split(".")),
                  "consumer_source_rechecked": True, "experiment_usage_measured": False} for row in current]
    views = witness_views(data)
    summary = {"counts": data["accepted"]["counts"], "current_denominators": den, "coverage_rows": len(current),
               "changed_scope_field_rows": len(changed), "changed_by_scope": dict(Counter(r["scope"] for r in changed)),
               "node_metadata_field_union": 130, "edge_metadata_field_union": 37,
               "independent_coverage_matches": True, "all_R15_counters_and_types_restored": True,
               "node_view_witnesses": 125, "edge_view_witnesses": 186, "lossless_real_record_round_trips": len(views),
               "code_use_blocks_rechecked": 6, "structure": structure_summary(data["validation"]["topology"]),
               "new_isolates_retained": 61, "offloaded_atom_dependencies": 81, "core_atom_dependencies": 2,
               "known_confirmed_unrepaired": 40, "additional_review": 21,
               "read_only_view_implemented": True, "production_server_integrated": False,
               "graph_changes": 0, "new_graph_fields": [], "deleted_graph_fields": [], "missing_values_filled": 0,
               "large_graph_copies": 0, "formal_apply_performed": False, "models_called": 0, "training_jobs_started": 0,
               "scientific_validity_certified": False, "experiment_usage_measured": False}
    return {"CURRENT_METADATA_COVERAGE.jsonl": current, "CHANGED_COVERAGE_ROWS.jsonl": changed,
            "CLAIM_FIELD_SPOTLIGHT.jsonl": spotlight, "FIELD_READING_CATALOGUE.jsonl": catalogue,
            "CURRENT_RECORD_VIEWS.jsonl": views}, summary


def link(path, label):
    relative = os.path.relpath(path, OUTPUT).replace("\\", "/")
    return f'<a href="{quote(relative, safe="/.")}">{escape(label)}</a>'


def document(title, body):
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{escape(title)}</title><style>body{{margin:0;background:#f3f5f7;color:#203344;font:16px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:1.7rem}}h2{{font-size:1.3rem;margin-top:28px}}h3,summary,code{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid #dbe3e9;text-align:left;vertical-align:top}}pre{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}}details{{padding:10px;margin:8px 0;background:white;border:1px solid #dbe3e9}}summary{{cursor:pointer}}.notice,.warning{{padding:12px;background:#fff3d7;border-left:4px solid #bb8323}}.scroll{{overflow-x:auto}}small{{color:#607283}}a{{color:#235f92}}@media(max-width:680px){{main{{padding:15px}}td,th{{padding:6px}}}}</style></head><body><main>{body}</main></body></html>'''


def pages(outputs, summary, proof):
    spotlight = outputs["CLAIM_FIELD_SPOTLIGHT.jsonl"]
    stats = "".join(f'<tr><td>{escape(r["label"])}</td><td>{r["present"]:,}</td><td>{r["nonempty"]:,}</td><td>{r["nonempty_pct"]:.6f}%</td></tr>' for r in spotlight)
    structure = summary["structure"]
    layers = "".join(f'<tr><td>{label}</td><td>{structure["layers"][key]:,}</td><td>{state}</td></tr>' for key, label, state in
                     (("relations", "其余关系（不代表均已科学认证）", "默认查看"), ("claim_anchors", "about：声明锚点", "按需展开"), ("mappings", "maps_to：映射", "按需展开")))
    body = f'''{link(journal.REPORT, "返回 KG 工作记录")}<h1>第二大步：图不乱删，查看更简单</h1>
<p class="notice">R23 候选：2,587,911 节点（905,224 claims）、3,012,135 边。metadata 顶层键并集仍为节点 130 / 边 37。此次没有物理删除或正式替换；已实现独立只读视图，尚未接入正式服务。</p>
<h2>1. 结构分三层查看</h2><table><tr><th>层</th><th>全图边数</th><th>独立读取默认行为</th></tr>{layers}</table>
<p>默认关注 951,860 条其余关系，约 {structure['default_collapsed_pct']:.2f}% 的边归入按需层。选择全部层会保留原顺序、重复边、自环和端点；隐藏不等于删除或质量合格。61 个新增孤立概念仍有 81 条离线原子、2 条核心原子依赖，全部保留。</p>
<h2>2. metadata 当前覆盖（R23 全部 claim 为分母）</h2><div class="scroll"><table><tr><th>字段</th><th>键存在数</th><th>非空数</th><th>非空率</th></tr>{stats}</table></div>
<p>不是 100% 覆盖。存在、空值、缺席分别计数；0 和 false 是已记录值，不自动当缺失。占位字符串可能计入非空，具体数量见完整表。非空率不等于正确率、可计算率或实验使用率。本轮未做实验。</p>
<h2>3. 简明查看器已可用</h2><p>关系与原文、科学限定优先展示；来源统计、审核与其他字段折叠，完整 JSON 随时展开。所有 JSON 路径和值都保留，不覆盖嵌套冲突，不将 p 上界或 ICC 强行转换为数值。</p>
<p>{link(OUTPUT / 'KG_RECORD_BROWSER.html', '打开当前记录的简明查看器')}。收录 125 个节点和 186 条边，是已验收的问题与保护样例，不是全图库或随机样本。共 311 份真实记录逐一通过完整类型/字段顺序往返校验。</p>
<h2>4. 用途与安全边界</h2><p>重新核对了 6 组现有代码读取证据：详情展示、端点类型、动态 claim 字段、证据哈希、质量检查、扩展字段保留。源文件未变；静态代码存在读取路径，不证明某次实验用过。完整 evidence/source_paper 参与证据哈希，低覆盖字段没有被直接删除。</p>
<p class="warning">仍有 40 条已确认问题尚未修复，另有 21 条待复审。查看器显式显示这些提示；历史 finalized 不是新的科学认证。下一步是统一交付验收及正式采用的范围确认，不能把它当成已清零问题的正式版本。</p>
<h2>5. 数字怎样核对？</h2><p>R15 的 R12 全量 453 行统计减去 R23 完整 50 节点/149 边原像，共 {summary['changed_scope_field_rows']} 行统计发生变化。两套计算一致，逆向恢复全部原计数、分母和 JSON 类型。字段行包括嵌套与顶层属性，不等于字段并集。</p>
<p>复用已验收完整保留记录/字节回退证明；最后完整图校验 UTC：{escape(proof['full_graph_verified_at'])}。本轮只完整校验小型输入/输出，检查大文件未变状态与原生身份，没有重新扫描大图或生成副本。</p>
<p>{' · '.join(link(OUTPUT / name, label) for name, label in (('CURRENT_METADATA_COVERAGE.jsonl','453 行当前覆盖'),('FIELD_READING_CATALOGUE.jsonl','字段用途与保留目录'),('CURRENT_VERSION_PROOF.json','版本与差量证据'),('CODE_USE_RECHECK.json','6 组代码用途'),('REVIEW_COMPLETE.json','本步验收'),('USAGE.md','读取模块与命令说明')))}</p>'''
    cards = defaultdict(list)
    for row in outputs["CURRENT_RECORD_VIEWS.jsonl"]:
        category = "nodes" if row["kind"] == "node" else row["layer"]
        content = render_record(row["view"])
        cards[category].append(f'<details><summary>{escape(row["identity"])}</summary>{content}</details>')
    sections = "".join(f'<details><summary>{label} · {len(cards[key])} 份样例</summary>{"".join(cards[key])}</details>' for key, label in
                       (("nodes", "节点"), ("relations", "其余关系"), ("claim_anchors", "about 锚点层"), ("mappings", "maps_to 映射层")))
    browser = f'''{link(OUTPUT / 'KG_SIMPLIFIED.html', '返回覆盖与结构总览')}<h1>简明读取 · R23 已绑定记录</h1>
<p class="notice">125 节点 + 186 边的有意选择样例，非全图库或随机抽样。点击类别和 ID 展开。默认把审核与其他字段折叠，原字段并未删除；节点/关系不合并。主图仍为 2,587,911 节点 / 3,012,135 边。</p>
<p>已知问题提示优先于历史审核状态。原值、原位置与 JSON 类型原样显示；未提供的 negated 不会自动补成 false，未提供的统计不会补成 0。未知字段保留并可展开，不被称为无用。CLI 精确查找见 {link(OUTPUT / 'USAGE.md', '使用说明')}。</p>{sections}'''
    return {"KG_SIMPLIFIED.html": document("KG 简明读取与当前覆盖", body), "KG_RECORD_BROWSER.html": document("R23 KG 记录简明查看器", browser)}


def check_html(page, *, files=True, pending=False):
    class Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = 0
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            require(tag not in {"script", "iframe", "object", "embed", "form", "img", "link"}, "active/resource HTML forbidden")
            require(not any(k.startswith("on") for k in attrs) and not {"src", "srcset", "action"} & attrs.keys(), "active attributes forbidden")
            if tag == "a":
                href = attrs.get("href", "")
                parsed = urlparse(href)
                require(href and not parsed.scheme and not parsed.netloc and not parsed.query, "nonlocal link")
                path = (OUTPUT / unquote(parsed.path)).resolve()
                path.relative_to(journal.REPO.resolve())
                require(not files or path.is_file() or pending and path == (OUTPUT / "REVIEW_COMPLETE.json").resolve(), "broken evidence link")
                self.links += 1
    parser = Checker()
    parser.feed(page)
    parser.close()
    return {"local_links": parser.links, "broken_links": 0 if files else None, "scripts": 0, "remote_resources": 0}


def write_once(name, value):
    path = OUTPUT / name
    require(not path.exists(), "frozen output exists: " + name)
    if name.endswith(".jsonl"):
        journal.atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n" for r in value))
    elif isinstance(value, str):
        journal.atomic_text(path, value)
    else:
        journal.atomic_json(path, value)
    return {**journal.fingerprint(path), **({"rows": len(value)} if name.endswith(".jsonl") else {})}


def tests():
    counts = Counter()
    for suite in ElementTree.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"):
        counts.update({k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
    require(counts["tests"] >= 35 and not any(counts[k] for k in ("failures", "errors", "skipped")), "passing current tests required")
    return dict(counts)


def write_build_artifact(name, value):
    """Resume an interrupted, unsealed build only for exactly equal artifacts.

    Never replace a file or revise a sealed build. Mutual HTML links are
    checked after both files exist, not after writing just the first page.
    """
    require(Path(name).name == name and name not in {"BUILD_RECEIPT.json", "BUILD_BINDING.json", "REVIEW_COMPLETE.json"}, "not a build artifact name")
    require(not any((OUTPUT / n).exists() for n in ("BUILD_RECEIPT.json", "BUILD_BINDING.json", "REVIEW_COMPLETE.json")), "build already sealed")
    path = OUTPUT / name
    if not path.exists():
        return write_once(name, value)
    if name.endswith(".jsonl"):
        matches = same_record(list(jsonl(path)), value)
    elif isinstance(value, str):
        matches = path.read_text(encoding="utf-8") == value
    else:
        matches = same_record(journal.read_json(path), value)
    require(matches, "unsealed artifact differs; refusing overwrite: " + name)
    return {**journal.fingerprint(path), **({"rows": len(value)} if name.endswith(".jsonl") else {})}


def guard_end(data):
    guards((data["bindings"].files, data["proof"]))
    for key in ("protected_native", "current_native_start"):
        require([native_info(x["path"]) for x in data["proof"][key]] == data["proof"][key], "native identity changed during step")
    current = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(current["current_acceptance"] == data["proof"]["current_acceptance"], "candidate advanced during step")


def usage():
    return '''# KG 简明读取入口

KG_SIMPLIFIED.html 是当前覆盖总览，KG_RECORD_BROWSER.html 是已绑定 125 节点/186 边样例的可展开查看器。可直接打开本地文件，无网络、无服务和模型。不是全图库搜索器，也没有改正式服务。

在项目根目录精确查看已绑定样例（只向终端输出视图，不写图）：

    uv run --no-project --with networkx python -X utf8 neurooracle/scripts/prepare_kg_simplified_reading.py show --identity CLM:63af446c43fd
    uv run --no-project --with networkx python -X utf8 neurooracle/scripts/prepare_kg_simplified_reading.py show --identity edge:59616

新纯函数模块 neurooracle.src.kg_reading_view 可对调用者已有的任意单个节点/边 dict 工作：make_view(record, kind="node")、restore_record(view)、render_record(view)。它不加载大图，不解码/改写 Claim，不填空、不合并字段；视图中的 schema/groups/notices 不是写入 KG 的 metadata 字段。

iter_visible_edges(edges) 默认选其余关系层；iter_visible_edges(edges, layers=LAYERS) 显示全部层，保持原顺序、重复边和自环。只用 relation_type 精确区分 about 与 maps_to，不推测科学有效性。任何层都不删节点或改端点；不将该视图另存为正式图。

未知字段默认完整保留在其余字段组。科学条件、类型、否定及原谓词优先展示。缺席字段仍缺席，null/空/false/0 保留原值；顶层与嵌套位置冲突并列保留，不自动决定谁正确。已知问题提示来自 R23 40 确认/21 待复审登记，不因旧 finalized 清除。

本轮图节点、边与字段并集未减，未跑实验或模型。源覆盖定义仍只展开 metadata 一层；简明单记录视图可展开更深的字典，两者统计口径不同。所有数组作为原值整体保留。
'''


def build():
    require(not (OUTPUT / "BUILD_RECEIPT.json").exists(), "step already built")
    data = load_inputs()
    outputs, summary = compute(data)
    test_counts = tests()
    html = pages(outputs, summary, data["proof"])
    for page in html.values():
        check_html(page, files=False)
    guard_end(data)
    artifacts = {name: write_build_artifact(name, rows) for name, rows in outputs.items()}
    artifacts["CURRENT_VERSION_PROOF.json"] = write_build_artifact("CURRENT_VERSION_PROOF.json", data["proof"])
    artifacts["CODE_USE_RECHECK.json"] = write_build_artifact("CODE_USE_RECHECK.json", data["reads"])
    artifacts["USAGE.md"] = write_build_artifact("USAGE.md", usage())
    for name, page in html.items():
        artifacts[name] = write_build_artifact(name, page)
    for page in html.values():
        check_html(page, pending=True)
    guard_end(data)
    receipt = {"status": "R23_SIMPLIFIED_READING_BUILT_NOT_FORMAL_GRAPH_CHANGE", "created_at": journal.utc_now(),
               **summary, "proof": data["proof"], "bindings": data["bindings"].files, "artifacts": artifacts,
               "implementation": {name: journal.fingerprint(journal.REPO / name) for name in CODE},
               "policy": journal.fingerprint(OUTPUT / "POLICY.md"), "tests": test_counts,
               "development_diagnostics": journal.fingerprint(OUTPUT / "INITIAL_DIAGNOSTICS.md"),
               "test_results": journal.fingerprint(OUTPUT / "TEST_RESULTS.xml")}
    fp = write_once("BUILD_RECEIPT.json", receipt)
    write_once("BUILD_BINDING.json", fp)
    print(json.dumps(summary, ensure_ascii=False))


def accept():
    require(not (OUTPUT / "REVIEW_COMPLETE.json").exists(), "step already accepted")
    b = Bindings()
    fp = journal.read_json(OUTPUT / "BUILD_BINDING.json")
    require(Path(fp["path"]) == OUTPUT / "BUILD_RECEIPT.json", "build binding path changed")
    receipt = b.json(fp)
    for item in [*receipt["bindings"].values(), *receipt["artifacts"].values(), *receipt["implementation"].values(), receipt["policy"], receipt["test_results"], receipt["development_diagnostics"]]:
        b.check(item)
    data = load_inputs()
    outputs, summary = compute(data)
    require(all(same_record(receipt[k], value) for k, value in summary.items()) and same_record(receipt["proof"], data["proof"]), "saved summary/proof differs")
    for name, rows in outputs.items():
        require(same_record(rows, b.rows(receipt["artifacts"][name])), "saved current rows differ: " + name)
    require(b.json(receipt["artifacts"]["CURRENT_VERSION_PROOF.json"]) == data["proof"], "saved proof differs")
    require(b.json(receipt["artifacts"]["CODE_USE_RECHECK.json"]) == data["reads"], "saved consumers differ")
    require((OUTPUT / "USAGE.md").read_text(encoding="utf-8") == usage(), "saved usage differs")
    html = pages(outputs, summary, data["proof"])
    for name, page in html.items():
        require((OUTPUT / name).read_text(encoding="utf-8") == page, "saved page differs")
        check_html(page, pending=True)
    require(tests() == receipt["tests"], "test counts changed")
    guard_end(data)
    complete = {"status": "STRUCTURE_METADATA_READ_ONLY_SIMPLIFICATION_ACCEPTED_NOT_FORMALLY_APPLIED",
                "completed_at": journal.utc_now(), "build": fp, **summary, "tests": receipt["tests"],
                "test_results": receipt["test_results"], "current_acceptance": data["proof"]["current_acceptance"]}
    write_once("REVIEW_COMPLETE.json", complete)
    qa = {name: check_html(page) for name, page in html.items()}
    write_once("HTML_QA.json", qa)
    print(json.dumps({"status": complete["status"], "tests": complete["tests"], "html": qa}, ensure_ascii=False))


def show(identity):
    b = Bindings()
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    fp = campaign["round24_structure_metadata_simplification"]["receipt"]
    require(Path(fp["path"]) == OUTPUT / "REVIEW_COMPLETE.json", "current reading receipt path changed")
    complete = b.json(fp)
    require(complete["status"] == "STRUCTURE_METADATA_READ_ONLY_SIMPLIFICATION_ACCEPTED_NOT_FORMALLY_APPLIED", "reading view not accepted")
    receipt = b.json(complete["build"])
    require(complete["current_acceptance"] == campaign["current_acceptance"], "saved view no longer current")
    b.json(complete["current_acceptance"])
    for item in receipt["implementation"].values():
        b.check(item)
    guards(receipt["proof"])
    rows = b.rows(receipt["artifacts"]["CURRENT_RECORD_VIEWS.jsonl"])
    selected = [row for row in rows if row["identity"] == identity]
    require(len(selected) == 1, "ID absent from the 311 bound examples; this is not a whole-KG search index")
    restore_record(selected[0]["view"])
    print(json.dumps(selected[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build", "accept", "show"))
    parser.add_argument("--identity")
    args = parser.parse_args()
    if args.command == "check":
        data = load_inputs()
        outputs, summary = compute(data)
        guard_end(data)
        print(json.dumps({**summary, "spotlight": outputs["CLAIM_FIELD_SPOTLIGHT.jsonl"]}, ensure_ascii=False))
    elif args.command == "show":
        require(args.identity, "--identity is required")
        show(args.identity)
    else:
        {"build": build, "accept": accept}[args.command]()
