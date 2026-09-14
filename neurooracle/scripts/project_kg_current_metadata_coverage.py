"""Project frozen full-census metadata counts through the exact accepted delta.

No graph scan, model execution, metadata filling or graph mutation. The current
graph's already completed full ordered inverse proof anchors every operation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_accepted_candidate_lineage as lineage
import kg_overnight_report as journal
from kg_accepted_candidate_lineage import require, indexed, file_identity
from retire_kg_duplicate_references import guards, deep_check, jsonl, read_json, same_record
from neurooracle.src.metadata_field_audit import Coverage, node_class, nonempty, PLACEHOLDERS

OUTPUT = journal.OUTPUT / "round15_current_metadata_coverage"
CENSUS = journal.OUTPUT / "round02_current_metadata_and_identity"
CENSUS_SHA = "0f832fbd14dac186a3641b2fcd942b897b906a86712f6663f49b71ca50f0f921"
R12 = lineage.R12
COUNT_KEYS = ("present", "nonempty", "placeholder_like")
CODE = ("neurooracle/scripts/project_kg_current_metadata_coverage.py", "neurooracle/tests/test_kg_current_metadata_coverage.py",
        "neurooracle/src/metadata_field_audit.py", "neurooracle/scripts/kg_accepted_candidate_lineage.py",
        "neurooracle/scripts/retire_kg_duplicate_references.py", "neurooracle/scripts/kg_overnight_report.py")
SPOTLIGHT = (("主语ID", "subject_id"), ("关系谓词", "predicate"), ("宾语ID", "object_id"),
             ("原始证据文字", "raw_text"), ("PMID", "source_paper.pmid"), ("DOI", "source_paper.doi"),
             ("论文标题", "source_paper.title"), ("研究类型", "evidence.study_type"), ("方法说明", "evidence.methodology"),
             ("样本量", "evidence.sample_size"), ("p值", "evidence.p_value"), ("效应量", "evidence.effect_size"),
             ("效应度量", "evidence.effect_metric"), ("证据方向", "evidence.direction"))


def load_data():
    b = lineage.Bindings()
    campaign = read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(Path(campaign["current_acceptance"]["path"]) == R12 / "REPAIR_ACCEPTANCE.json"
            and campaign["current_acceptance"]["sha256"] == lineage.ACCEPTANCE_SHA, "current candidate has advanced")
    accepted = b.json(campaign["current_acceptance"])
    guards(accepted)
    values = {}
    for name in ("INPUTS.json", "PLAN.json", "BUILD_COMPLETE.json", "VALIDATION_COMPLETE.json"):
        fp = accepted["artifacts"][name]
        require(Path(fp["path"]) == R12 / name, "accepted artifact path substituted")
        values[name] = b.json(fp)
    inputs, plan, build, validation = [values[n] for n in ("INPUTS.json", "PLAN.json", "BUILD_COMPLETE.json", "VALIDATION_COMPLETE.json")]
    census = read_json(b.pin(CENSUS / "CENSUS_COMPLETE.json", CENSUS_SHA))
    require(census["status"] == "CURRENT_GRAPH_CENSUS_VERIFIED_NO_MUTATION", "metadata census incomplete")
    lineage.verify_boundary(accepted, inputs, plan, build, validation, census)
    for fp in [*accepted["implementation"].values(), *census["implementation"].values()]:
        b.check(fp)
    for value in (inputs, build, census, campaign["formal_sources"]):
        guards(value)
    fp = census["artifacts"]["METADATA_NESTED_COVERAGE.jsonl"]
    require(Path(fp["path"]) == CENSUS / "METADATA_NESTED_COVERAGE.jsonl", "source coverage path substituted")
    rows = b.rows(fp)
    require(len(rows) == census["coverage_rows"] == 453, "full source coverage incomplete")
    ops = {}
    for name in ("NODE_PATCHES.jsonl", "EDGE_PATCHES.jsonl", "EDGE_RETIREMENTS.jsonl"):
        require(same_record(plan["artifacts"][name], accepted["artifacts"][name]), "plan operation binding differs")
        ops[name] = b.rows(accepted["artifacts"][name])
    require([len(ops[n]) for n in ops] == [5, 1, 24], "incomplete accepted operation list")
    issue_path = journal.OUTPUT / "round14_batch946_source_review/REVIEW_COMPLETE.json"
    issue_snapshot = read_json(b.pin(issue_path, "a92f83d48c026e24ef50a5d5f0d68ab76eededcecc342f1d412220570ab5e986"))
    issue_build = b.json(issue_snapshot["build"])
    require(issue_snapshot["status"] == "BATCH946_SOURCE_EXPRESSION_FINDINGS_VERIFIED_NOT_REPAIRED_NOT_QUARANTINED"
            and issue_snapshot["combined_confirmed_issue_claims"] == 87 and issue_snapshot["combined_additional_review_candidates"] == 21
            and same_record(file_identity(issue_build["current_lineage"]["current_graph"]), file_identity(build["graph"])), "source issue snapshot differs")
    proof = {"method": "frozen_full_BASE_metadata_census_plus_complete_accepted_R12_delta_and_full_ordered_inverse_proof",
        "source_census": journal.fingerprint(CENSUS / "CENSUS_COMPLETE.json"), "source_coverage": fp,
        "source_acceptance": inputs["source_acceptance"], "current_acceptance": campaign["current_acceptance"],
        "source_graph": inputs["source_graph"], "current_graph": build["graph"],
        "source_counts": inputs["source_counts"], "current_counts": accepted["counts"],
        "full_current_verified_at": validation["completed_at"], "accepted_complete_operation_counts": [5, 1, 24],
        "operation_bindings": {n: accepted["artifacts"][n] for n in ops}, "full_inverse_digests": validation["digests"],
        "source_coverage_definitions": census["implementation"], "formal_sources": campaign["formal_sources"],
        "source_issue_snapshot": {"receipt": journal.fingerprint(issue_path), "confirmed": 87, "additional": 21,
                                  "repaired_or_quarantined_this_round": 0},
        "full_graph_rehashed_this_round": False, "graph_changes_this_round": 0,
        "historical_source_counts_rewritten": False, "models_called": 0, "training_jobs_started": 0}
    return {"bindings": b, "census": census, "accepted": accepted, "proof": proof, "rows": rows, "ops": ops}


def validate_rows(rows, denominators):
    require(all(isinstance(s, str) and type(n) is int and n > 0 for s, n in denominators.items()), "invalid scope denominator")
    result = {}
    for row in rows:
        key = (row["scope"], row["field"])
        require(key not in result, "duplicate scope/field row")
        require(row["scope"] in denominators and row["field"].startswith(("metadata.", "top.")), "unknown scope/field")
        require(type(row["denominator"]) is int and row["denominator"] == denominators[row["scope"]], "row denominator differs")
        require(all(type(row[k]) is int for k in (*COUNT_KEYS, "empty_or_null", "absent")), "coverage count must be integer, not bool/float")
        p, n, t = row["present"], row["nonempty"], row["denominator"]
        require(0 <= row["placeholder_like"] <= n <= p <= t and p > 0, "invalid count decomposition")
        require(row["empty_or_null"] == p - n and row["absent"] == t - p, "empty/absent decomposition differs")
        require(same_record(row["present_pct"], round(100*p/t, 6)) and same_record(row["nonempty_pct"], round(100*n/t, 6)), "percentage denominator differs")
        require(isinstance(row["types"], dict) and all(isinstance(k, str) and type(v) is int and v > 0 for k, v in row["types"].items())
                and sum(row["types"].values()) == p, "JSON type counts differ from present count")
        result[key] = row
    require({s for s, _ in result} == set(denominators), "scope missing from field census")
    return result


def contributions(ops):
    np = indexed(ops["NODE_PATCHES.jsonl"], "node_id")
    ep = indexed(ops["EDGE_PATCHES.jsonl"], "source_candidate_edge_ordinal")
    retired = indexed(ops["EDGE_RETIREMENTS.jsonl"], "source_candidate_edge_ordinal")
    require(not set(ep) & set(retired), "patched and retired edge overlap")
    require(all(type(i) is int and i > 0 for i in set(ep) | set(retired)), "invalid source edge ordinal")
    minus, plus = Coverage(), Coverage()
    records = []
    for nid, row in sorted(np.items()):
        require(row["before"]["id"] == row["after"]["id"] == nid, "node patch changes identity")
        before_scope = "node/" + node_class(nid, row["before"])
        after_scope = "node/" + node_class(nid, row["after"])
        minus.add(before_scope, row["before"])
        plus.add(after_scope, row["after"])
        records.append({"operation": "node_patch", "identity": nid, "before_scope": before_scope, "after_scope": after_scope,
                        "before": row["before"], "after": row["after"]})
    for ordinal, row in sorted(ep.items()):
        minus.add("edge/all", row["before"])
        plus.add("edge/all", row["after"])
        records.append({"operation": "edge_patch", "identity": ordinal, "before_scope": "edge/all", "after_scope": "edge/all",
                        "before": row["before"], "after": row["after"]})
    for ordinal, row in sorted(retired.items()):
        # The hypothetical corrected record is NOT an applied afterimage.
        minus.add("edge/all", row["before"])
        records.append({"operation": "edge_retirement", "identity": ordinal, "before_scope": "edge/all", "after_scope": None,
                        "before": row["before"], "after": None})
    return minus, plus, records


def pack_rows(fields, types, denominators):
    result = []
    for (scope, field), counters in sorted(fields.items()):
        require(scope in denominators, "delta scope missing")
        p, n, h = [counters.get(k, 0) for k in COUNT_KEYS]
        t = denominators[scope]
        require(type(t) is int and t > 0, "invalid projected scope denominator")
        counts = {k: v for k, v in types.get((scope, field), {}).items() if v != 0}
        require(all(type(v) is int and v >= 0 for v in (p, n, h, *counts.values())), "negative/noninteger delta counter")
        if p == 0:
            require(n == h == 0 and not counts, "vanished field retains evidence counts")
            continue
        result.append({"scope": scope, "field": field, "denominator": t, "present": p, "nonempty": n,
            "empty_or_null": p-n, "absent": t-p, "placeholder_like": h,
            "present_pct": round(100*p/t, 6), "nonempty_pct": round(100*n/t, 6), "types": counts})
    validate_rows(result, denominators)
    return result


def apply_delta(rows, denominators, minus, plus):
    """Counter arithmetic; never sum rounded percentages or discard negatives."""
    original = validate_rows(rows, denominators)
    den = {scope: denominators.get(scope, 0) - minus.denominators[scope] + plus.denominators[scope]
           for scope in set(denominators) | set(minus.denominators) | set(plus.denominators)}
    require(all(type(n) is int and n > 0 for n in den.values()), "invalid projected scope denominator")
    fields, types = {}, {}
    for key in set(original) | set(minus.fields) | set(plus.fields):
        row = original.get(key, {})
        before_counts, after_counts = minus.fields.get(key, {}), plus.fields.get(key, {})
        fields[key] = {name: row.get(name, 0) - before_counts.get(name, 0) + after_counts.get(name, 0) for name in COUNT_KEYS}
        prior_types = row.get("types", {})
        before_types, after_types = minus.types.get(key, {}), plus.types.get(key, {})
        types[key] = {name: prior_types.get(name, 0) - before_types.get(name, 0) + after_types.get(name, 0)
                      for name in set(prior_types) | set(before_types) | set(after_types)}
    return pack_rows(fields, types, den), den


def independent_project(rows, denominators, records):
    """Separate record-by-record signed interpreter (not Coverage.add/delta)."""
    initial = validate_rows(rows, denominators)
    den = dict(denominators)
    fields = {k: {c: r[c] for c in COUNT_KEYS} for k, r in initial.items()}
    types = {k: dict(r["types"]) for k, r in initial.items()}
    for operation in records:
        for side, sign in (("before", -1), ("after", 1)):
            record, scope = operation[side], operation[side + "_scope"]
            if record is None:
                require(side == "after" and scope is None and operation["operation"] == "edge_retirement", "unexpected missing pre/postimage")
                continue
            den[scope] = den.get(scope, 0) + sign
            md = record.get("metadata") or {}
            require(isinstance(md, dict), "metadata must be object")
            observed = []
            for key, value in md.items():
                observed.append(("metadata." + key, value))
                if isinstance(value, dict):
                    observed.extend(("metadata." + key + "." + child, v) for child, v in value.items())
            observed.extend(("top." + key, value) for key, value in record.items() if key != "metadata")
            for field, value in observed:
                key = (scope, field)
                counts = fields.setdefault(key, dict.fromkeys(COUNT_KEYS, 0))
                counts["present"] += sign
                counts["nonempty"] += sign * int(nonempty(value))
                counts["placeholder_like"] += sign * int(isinstance(value, str) and value.strip().casefold() in PLACEHOLDERS)
                typed = types.setdefault(key, {})
                name = type(value).__name__
                typed[name] = typed.get(name, 0) + sign
    return pack_rows(fields, types, den), den


def unions(rows):
    result = {"node": set(), "edge": set()}
    for row in rows:
        if row["field"].startswith("metadata.") and row["present"]:
            result[row["scope"].split("/")[0]].add(row["field"].split(".")[1])
    return {k: sorted(v) for k, v in result.items()}


def results(data):
    baseline, den = data["rows"], data["census"]["coverage_denominators"]
    minus, plus, records = contributions(data["ops"])
    projected, current_den = apply_delta(baseline, den, minus, plus)
    independently, independent_den = independent_project(baseline, den, records)
    require(same_record(projected, independently) and same_record(current_den, independent_den), "independent projection differs")
    restored, restored_den = apply_delta(projected, current_den, plus, minus)
    require(same_record(restored, sorted(baseline, key=lambda r: (r["scope"], r["field"]))) and same_record(restored_den, den), "full counter/type rollback differs")
    counts = data["accepted"]["counts"]
    require(sum(n for s, n in current_den.items() if s.startswith("node/")) == counts["nodes"]
            and current_den["node/claim"] == counts["claims"] and current_den["edge/all"] == counts["edges"], "current count/scope reconciliation differs")
    union = unions(projected)
    require(same_record(unions(baseline), union) and len(union["node"]) == data["accepted"]["node_metadata_field_union"] == 130
            and len(union["edge"]) == data["accepted"]["edge_metadata_field_union"] == 37, "top-level metadata field union differs")
    old = validate_rows(baseline, den)
    now = validate_rows(projected, current_den)
    require(set(old) == set(now) and len(now) == 453, "unexpected field/scope emergence or disappearance")
    changed = [{"scope": k[0], "field": k[1], "before": old[k], "after": now[k],
                "delta": {c: now[k][c]-old[k][c] for c in (*COUNT_KEYS, "denominator", "empty_or_null", "absent")}}
               for k in sorted(now) if not same_record(old[k], now[k])]
    require(all(same_record(old[k], row) for k, row in now.items() if k[0] == "node/claim"), "claim coverage unexpectedly changed")
    spotlight = [{"label": label, **now["node/claim", "metadata." + field]} for label, field in SPOTLIGHT]
    summary = {"counts": counts, "source_counts": data["census"]["counts"], "coverage_rows": len(projected),
        "source_denominators": den, "current_denominators": current_den, "changed_scope_field_rows": len(changed),
        "changed_rows_by_scope": dict(Counter(r["scope"] for r in changed)), "claim_coverage_all_rows_unchanged": True,
        "node_metadata_field_union": 130, "edge_metadata_field_union": 37, "metadata_key_unions": union,
        "accepted_operations_reused": {"node_patches": 5, "edge_patches": 1, "edge_retirements": 24},
        "independent_projection_matches": True, "all_source_counters_and_types_recovered": True,
        "graph_changes_this_round": 0, "new_graph_fields": [], "deleted_graph_fields": [], "missing_values_filled": 0,
        "source_claim_register_modified": False, "scientific_validity_certified": False,
        "live_usage_measured": False, "runtime_view_implemented": False, "models_called": 0, "training_jobs_started": 0}
    return {"CURRENT_METADATA_COVERAGE.jsonl": projected, "CHANGED_COVERAGE_ROWS.jsonl": changed,
            "EXACT_DELTA_RECORDS.jsonl": records, "CLAIM_FIELD_SPOTLIGHT.jsonl": spotlight}, summary


def report(output, summary):
    lines = ["# 当前候选的 metadata 覆盖度（Round12版本）", "",
        "这次统计针对已接受但未替换正式图的候选：2,587,961节点，其中905,274个claim，3,012,284条边。节点metadata顶层键并集130、边37；不是每条记录都有130/37个字段，也不代表100%有值。", "",
        "旧Round02/04的边分母3,012,308属于夜间起点BASE。现已将453条类别×字段路径的全量统计，按已验收5节点/1边补丁及24条退役边的完整前后记录重新计算。44条统计行变化：43条edge/all行、1条node/source_mention.atom_types行。不是新增44种字段或44个新错误。", "",
        "原统计定义保持不变：按全部相应类别记录计分母，metadata展开至一层子字段，不展开列表；键缺席、显式null/空内容、非空值分别计数。0与false不是缺失；非空对象不保证其每个子字段有值。字符串占位符另计但仍可能被原定义计入非空，不能把非空数等同有效证据数。", "",
        "使用已完成的完整当前候选SHA/有序回退证明，重新绑定冻结全量BASE普查及全部已接受差量；两种独立累加逻辑结果相同，并逐字段/类型逆向恢复全部453行BASE统计。本轮没有重新扫描大图或生成副本，候选最后完整验证时间为2026-09-07 10:26:36香港时间；本轮新小型统计产物独立验收。", "",
        "## Claim字段：有值不等于正确或适用", "",
        "以下每项分母均为905,274个claim；这次候选改动未改变任何claim覆盖统计行。", "",
        "| 字段 | 键存在数 | 有值数 | 有值率 |", "|---|---:|---:|---:|"]
    for row in output["CLAIM_FIELD_SPOTLIGHT.jsonl"]:
        lines.append(f"| {row['label']} ({row['field']}) | {row['present']:,} | {row['nonempty']:,} | {row['nonempty_pct']:.2f}% |")
    lines += ["", "样本量键存在93.87%，真正非空仅5.45%；p值、效应量同样不能只看键存在率。其类型包含数值和字符串，需结合研究设计、单位、具体结局/分组及来源再使用，不把所有字符串直接视为可计算数值。没有用0填空，也没有把多组、多时间点、多样本类型压成一个统一样本量。", "",
        "source_mention类别的atom_types有值数从166,915变为166,919（分母1,478,646，有值率11.288368%→11.288638%），恰好对应已应用的4个影像概念类型槽位。字段种类未增加；这不是所有未填类型都应批量补齐的依据。", "",
        "## 精简与用途边界", "",
        "沿用核心关系与原文、科学上下文、来源/可选统计、后台审核四组阅读建议；是文档分组，未接入运行界面。来源信息便于回溯，可选统计按需要展开，审核/版本信息保留但可折叠阅读。低覆盖不等于无用，不据此删除字段。", "",
        "Round04的六组静态代码用途证据仍是其时间点的历史记录，不是实验实测使用率，也没有在本轮重新执行服务、模型、假设生成或训练。旧METADATA_REVIEW.html保持冻结，里面的‘当前’指当时BASE；阅读今天候选覆盖请使用本轮文件。", "",
        "来源问题登记仍为87条确认、21条额外复审，均未修复或隔离。本轮没有增加来源错误计数，也没有把形式一致性或软件测试通过称作论文科学事实已认证。", ""]
    return "\n".join(lines)


def local_link(path, label):
    import os
    relative = os.path.relpath(path, OUTPUT).replace("\\", "/")
    return f'<a href="{quote(relative, safe="/.")}">{journal.e(label)}</a>'


def html_page(output, summary):
    table = "".join(f'<tr><td>{journal.e(r["label"])}</td><td>{r["present"]:,}</td><td>{r["nonempty"]:,}</td><td>{r["nonempty_pct"]:.2f}%</td></tr>'
                    for r in output["CLAIM_FIELD_SPOTLIGHT.jsonl"])
    links = " · ".join(local_link(OUTPUT / name, label) for name, label in (
        ("CURRENT_METADATA_COVERAGE.jsonl", "453行当前统计"), ("CHANGED_COVERAGE_ROWS.jsonl", "44行前后对照"),
        ("METADATA_CURRENT_REVIEW.md", "完整解释与边界"), ("CURRENT_VERSION_PROOF.json", "版本与完整差量证明"),
        ("REVIEW_COMPLETE.json", "独立验收收据")))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>KG 当前候选 metadata 覆盖度</title><style>
body{{margin:0;background:#f4f6f8;color:#193044;font:16px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1050px;margin:auto;padding:30px}}h1{{font-size:1.7rem}}h2{{font-size:1.25rem;margin-top:30px}}.note{{background:#e6eff8;padding:16px;border-left:4px solid #36638f}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;padding:9px 13px;border-bottom:1px solid #dbe2e8}}th{{background:#eaf0f5}}a{{color:#235d91}}footer{{font-size:.9rem;color:#4f6272;margin-top:28px}}@media(max-width:650px){{main{{padding:16px}}th,td{{padding:8px}}}}
</style></head><body><main>{local_link(journal.REPORT, "返回八小时工作记录")}<h1>metadata：当前候选中，哪些确实有值？</h1>
<p class="note">2,587,961节点 · 905,274 claims · 3,012,284边。节点字段并集130、边37，<strong>不是100%覆盖</strong>。这份报告绑定Round12已接受候选；正式full_v2未替换。</p>
<h2>关键字段覆盖</h2><p>统一分母：905,274个claim。键存在与有值分开；有值不保证科学正确、数值可算或适用于当前研究。</p>
<div class="scroll"><table><thead><tr><th>字段</th><th>键存在数</th><th>有值数</th><th>有值率</th></tr></thead><tbody>{table}</tbody></table></div>
<p>例如样本量键存在93.87%，实际有值仅5.45%。0和false不是缺失，null与键缺席分别计数；占位字符串另行记录。没有填补数值或合并不同组别、时间点的分母。</p>
<h2>这次校正了什么统计？</h2><p>453条类别×字段路径统计中，44行随已接受差量改变：43条边字段统计和1条来源概念类型字段统计。边分母由3,012,308变为3,012,284；不是增加44个字段。所有claim字段覆盖行不变。</p>
<p>source_mention的atom_types有值数166,915→166,919，恰好对应4个已应用类型槽位；分母1,478,646，有值率11.288368%→11.288638%。字段种类不变，未把所有空类型自动补齐。</p>
<h2>如何让阅读更简单？</h2><p>核心关系与原文优先展示；科学上下文、来源与可选统计按需展开；审核和版本信息折叠保留。这只是阅读建议，未接入应用、未删字段，未运行模型、KGE或训练。低覆盖本身不能证明字段无用。</p>
<h2>统计可信到什么程度？</h2><p>冻结BASE全量普查＋已接受5节点/1边/24退役完整前后像；两种计算结果一致，逆向恢复全部原计数及JSON类型。沿用2026-09-07 10:26:36香港时间的候选全量校验并检查文件状态，本轮不重复大图扫描。非空率不是可用率或科学正确率。</p>
<p>旧第4轮metadata页面是当时BASE的历史快照，不重写。此前87条确认来源/表达问题和21条额外复审仍未修复，本轮没有增加问题数量。</p>
<footer>{links}<p>本页面离线运行，不加载外部资源；时间与验收状态见工作记录和本轮收据。</p></footer></main></body></html>'''


def check_html(text, require_files=True, allow_pending_receipt=False):
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            require(tag != "script", "report must not execute scripts")
            attrs = dict(attrs)
            require(not any(k.lower().startswith("on") for k in attrs), "inline event handler forbidden")
            require("src" not in attrs, "report must not load resources")
            if tag == "a":
                href = attrs.get("href", "")
                parsed = urlparse(href)
                require(href and not parsed.scheme and not parsed.netloc and not parsed.query, "link must be local")
                path = (OUTPUT / unquote(parsed.path)).resolve()
                path.relative_to(journal.REPO.resolve())
                if require_files:
                    require(path.is_file() or allow_pending_receipt and path == (OUTPUT / "REVIEW_COMPLETE.json").resolve(),
                            "missing report artifact: " + str(path))
                self.links.append(str(path))
    parser = Parser()
    parser.feed(text)
    parser.close()
    require(len(parser.links) == 6, "report link set incomplete")
    return {"local_links": len(parser.links), "broken_links": 0 if require_files else None, "remote_resources": 0, "scripts": 0}


def write_once(path, value, rows=False, raw=False):
    require(not path.exists(), "immutable output already exists: " + str(path))
    if rows:
        journal.atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n" for r in value))
    elif raw:
        journal.atomic_text(path, value)
    else:
        journal.atomic_json(path, value)
    return {**journal.fingerprint(path), **({"rows": len(value)} if rows else {})}


def test_counts():
    counts = Counter()
    for suite in ElementTree.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"):
        counts.update({k: int(suite.attrib.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
    require(counts["tests"] >= 35 and not any(counts[k] for k in ("failures", "errors", "skipped")), "coverage tests missing or failed")
    return dict(counts)


def build():
    require(not (OUTPUT / "BUILD_RECEIPT.json").exists(), "coverage report already built")
    data = load_data()
    output, summary = results(data)
    tests = test_counts()
    check_html(html_page(output, summary), require_files=False)
    guards(data["bindings"].files)
    guards(data["proof"])
    artifacts = {n: write_once(OUTPUT / n, rows, rows=True) for n, rows in output.items()}
    artifacts["CURRENT_VERSION_PROOF.json"] = write_once(OUTPUT / "CURRENT_VERSION_PROOF.json", data["proof"])
    artifacts["METADATA_CURRENT_REVIEW.md"] = write_once(OUTPUT / "METADATA_CURRENT_REVIEW.md", report(output, summary), raw=True)
    artifacts["METADATA_CURRENT.html"] = write_once(OUTPUT / "METADATA_CURRENT.html", html_page(output, summary), raw=True)
    receipt = {"status": "CURRENT_CANDIDATE_METADATA_COVERAGE_BUILT_NOT_GRAPH_MUTATION", "created_at": journal.utc_now(),
        **summary, "bindings": data["bindings"].files, "current_version_proof": data["proof"], "artifacts": artifacts,
        "implementation": {p: journal.fingerprint(journal.REPO / p) for p in CODE},
        "tests": tests, "test_results": journal.fingerprint(OUTPUT / "TEST_RESULTS.xml")}
    guards(data["proof"])
    guards(data["bindings"].files)
    fp = write_once(OUTPUT / "BUILD_RECEIPT.json", receipt)
    write_once(OUTPUT / "BUILD_BINDING.json", fp)
    print(json.dumps({k: summary[k] for k in ("coverage_rows", "changed_scope_field_rows", "changed_rows_by_scope", "counts")}, ensure_ascii=False))


def accept():
    require(not (OUTPUT / "REVIEW_COMPLETE.json").exists(), "coverage report already accepted")
    build_fp = read_json(OUTPUT / "BUILD_BINDING.json")
    require(Path(build_fp["path"]) == OUTPUT / "BUILD_RECEIPT.json", "coverage build path substituted")
    deep_check(build_fp)
    receipt = read_json(OUTPUT / "BUILD_RECEIPT.json")
    guards(receipt)
    for fp in [*receipt["bindings"].values(), *receipt["implementation"].values(), *receipt["artifacts"].values(), receipt["test_results"]]:
        require(fp["bytes"] < 256*1024**2, "coverage evidence boundary must not rehash a whole graph")
        deep_check(fp)
    data = load_data()
    output, summary = results(data)
    require(all(same_record(receipt[k], value) for k, value in summary.items()), "saved summary differs")
    require(same_record(data["proof"], read_json(OUTPUT / "CURRENT_VERSION_PROOF.json")), "saved current version proof differs")
    for name, rows in output.items():
        require(same_record(rows, list(jsonl(OUTPUT / name))), "saved exact coverage/witness rows differ: " + name)
    require(report(output, summary) == (OUTPUT / "METADATA_CURRENT_REVIEW.md").read_text(encoding="utf-8"), "saved explanation differs")
    page = (OUTPUT / "METADATA_CURRENT.html").read_text(encoding="utf-8")
    require(page == html_page(output, summary), "saved HTML differs")
    check_html(page, require_files=True, allow_pending_receipt=True)
    tests = test_counts()
    require(same_record(tests, receipt["tests"]), "test counts changed")
    guards(receipt)
    completed = {"status": "CURRENT_CANDIDATE_METADATA_COVERAGE_INDEPENDENTLY_VERIFIED_NO_GRAPH_MUTATION",
        "completed_at": journal.utc_now(), "build": build_fp, **summary, "tests": tests, "test_results": receipt["test_results"]}
    write_once(OUTPUT / "REVIEW_COMPLETE.json", completed)
    qa = check_html(page, require_files=True)
    write_once(OUTPUT / "HTML_QA.json", qa)
    print(json.dumps({"status": completed["status"], "completed_at": completed["completed_at"], "tests": tests, "html_qa": qa}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "accept"))
    args = parser.parse_args()
    {"build": build, "accept": accept}[args.command]()
