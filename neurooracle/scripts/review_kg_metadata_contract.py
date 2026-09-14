"""Build an offline, evidence-linked metadata review; never rewrite the KG.

The proposed four-group reading contract is documentation, not an importer,
runtime schema, materialized projection, deletion list or experiment result.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from retire_kg_duplicate_references import guards, deep_check, fingerprint, jsonl, read_json
from audit_kg_integrity import JsonlWriter

CENSUS = journal.OUTPUT / "round02_current_metadata_and_identity"
OUTPUT = journal.OUTPUT / "round04_metadata_reading_contract"
HTML = OUTPUT / "METADATA_REVIEW.html"
CODE = ("neurooracle/scripts/review_kg_metadata_contract.py", "neurooracle/tests/test_kg_metadata_contract.py")

CORE = frozenset("id subject_id subject_name predicate object_id object_name negated confidence raw_text".split())
CONTEXT = frozenset("subject_type object_type conditions population".split())
EVIDENCE = frozenset("study_type methodology p_value effect_size effect_metric sample_size replicability direction".split())
PAPER = frozenset("pmid doi title authors year journal".split())
SPOTLIGHT = (
    ("主体 ID", "subject_id"), ("关系谓词", "predicate"), ("客体 ID", "object_id"),
    ("原始证据文字", "raw_text"), ("PMID", "source_paper.pmid"), ("DOI", "source_paper.doi"),
    ("论文标题", "source_paper.title"), ("研究类型", "evidence.study_type"),
    ("方法说明", "evidence.methodology"), ("sample_size", "evidence.sample_size"),
    ("p_value", "evidence.p_value"), ("effect_size", "evidence.effect_size"),
    ("effect_metric", "evidence.effect_metric"), ("证据方向", "evidence.direction"),
)
READS = (
    {"id": "web_claim_detail", "path": "core/web/server.py", "start": 2144, "end": 2185,
     "needles": ['meta = node.metadata or {}', 'evidence = meta.get("evidence") or {}', '"sample_size": evidence.get("sample_size")'],
     "lineage": "KG 的 claim 节点 → node.metadata → paper/evidence → 详情序列化；确认静态读取路径，未启动服务。"},
    {"id": "endpoint_semantics", "path": "neurooracle/src/claim_semantics.py", "start": 327, "end": 343,
     "needles": ['def concept_atom_roles(', 'metadata.get("atom_types")'],
     "lineage": "概念节点 metadata.atom_types 及 domain_tags → 规范类型集合；不是依据覆盖率猜测类型。"},
    {"id": "claim_endpoint_reads", "path": "neurooracle/src/claim_semantics.py", "start": 401, "end": 503,
     "needles": ['def _claim_value(', 'metadata.get(field)', 'subject_id = str(_claim_value(claim, "subject_id")'],
     "lineage": "claim 顶层/嵌套字段回退 → ID、全名和声明类型的端点检查；动态字段读取也有作用。"},
    {"id": "immutable_evidence_hash", "path": "neurooracle/src/case_study_membership_contract.py", "start": 104, "end": 137,
     "needles": ['def claim_evidence_payload(', '"evidence": payload.get("evidence") or {}', '"source_paper": payload.get("source_paper") or {}'],
     "lineage": "来源、原文、声明类型、条件、人群及整个 evidence/source_paper 对象进入证据哈希输入；删除子字段也可能使已有审核失效。"},
    {"id": "claim_quality_checks", "path": "neurooracle/src/kg_quality_checks.py", "start": 22, "end": 91,
     "needles": ['md = record.get("metadata") or {}', 'p = evidence.get("p_value")', 'n = evidence.get("sample_size")'],
     "lineage": "实际 claim 记录 → metadata.evidence → p 值数值范围和样本量检查；检查范围有限，不等于验证论文真实性。"},
    {"id": "schema_extension_preservation", "path": "neurooracle/src/schema.py", "start": 347, "end": 496,
     "needles": ['serialized = deepcopy(self.extra_fields)', 'class Claim:', 'extra_fields={k: v for k, v in d.items() if k not in known}'],
     "lineage": "Evidence、PaperRef、Claim 的扩展字段有序列化保留通道；未发现字面读取不代表可删除。旧 membership 字段另有规范化，不能把整个 round-trip 当作逐字无损证明。"},
)


def indexed_coverage(rows):
    result = {}
    for row in rows:
        key = (row["scope"], row["field"])
        if key in result:
            raise ValueError("duplicate scope/field coverage row")
        total, present, nonempty = (row[k] for k in ("denominator", "present", "nonempty"))
        if not 0 <= nonempty <= present <= total or total <= 0:
            raise ValueError("invalid coverage denominator/counts")
        if row["absent"] != total - present or row["empty_or_null"] != present - nonempty:
            raise ValueError("coverage count decomposition differs")
        if row["present_pct"] != round(100 * present / total, 6) or row["nonempty_pct"] != round(100 * nonempty / total, 6):
            raise ValueError("coverage percentage differs from its declared denominator")
        result[key] = row
    return result


def classify(row):
    scope, field = row["scope"], row["field"]
    key = field.removeprefix("metadata.")
    group, rationale, sites = "后台保留／待逐项核实", "本轮没有证明此字段可安全删除；默认保留，阅读时可折叠。", []
    if scope == "node/claim":
        if key in CORE:
            group, rationale, sites = "核心关系与原文", "用于关系识别、否定/置信信息或证据追溯。", ["web_claim_detail", "claim_quality_checks"]
        elif key in CONTEXT or key.removeprefix("metadata.") in CONTEXT:
            group, rationale, sites = "科学上下文", "保留原有顶层与嵌套位置；未知不猜测，冲突不覆盖。", ["claim_endpoint_reads", "immutable_evidence_hash"]
        elif key == "source_paper" or key.startswith("source_paper."):
            group, rationale, sites = "来源与可选证据", "可折叠展示，但保留所有原始来源字段及哈希输入。", ["immutable_evidence_hash", "schema_extension_preservation"]
            if key == "source_paper" or key.split(".")[-1] in PAPER:
                sites.append("web_claim_detail")
        elif key == "evidence" or key.startswith("evidence."):
            group, rationale, sites = "来源与可选证据", "低覆盖率不构成删除理由；非空也不代表有效统计值。", ["immutable_evidence_hash", "schema_extension_preservation"]
            if key == "evidence" or key.split(".")[-1] in EVIDENCE:
                sites.append("web_claim_detail")
            if key in {"evidence.p_value", "evidence.sample_size"}:
                sites.append("claim_quality_checks")
        else:
            sites = ["schema_extension_preservation"] if field.startswith("metadata.") else []
    elif field in {"metadata.atom_types", "top.domain_tags"} and scope.startswith("node/"):
        group, rationale, sites = "核心关系与原文", "概念规范类型的静态读取来源；不能按字段填充率推断身份正确。", ["endpoint_semantics"]
    return {**row, "reading_group": group, "recommendation": rationale,
            "reviewed_consumer_ids": sites, "consumer_evidence": "static_source_review_not_execution",
            "deletion_authorized": False, "storage_action": "retain_unchanged",
            "scientific_validity_certified": False}


def reviewed_read(spec, inventory):
    path = (journal.REPO / spec["path"]).resolve(strict=True)
    fp = next((v for v in inventory.values() if Path(v["path"]).resolve() == path), None)
    if fp is None:
        raise ValueError("reviewed source was not in the frozen census inventory")
    deep_check(fp)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    snippet = "\n".join(lines[spec["start"] - 1:spec["end"]])
    if not all(needle in snippet for needle in spec["needles"]):
        raise ValueError("reviewed code-line witness changed")
    return {**spec, "fingerprint": fp, "source_excerpt": snippet, "execution_performed": False}


def local_link(path, label):
    path = Path(path).resolve(strict=True)
    path.relative_to(journal.REPO.resolve())
    import os
    return f'<a href="{quote(Path(os.path.relpath(path, HTML.parent)).as_posix(), safe="/")}">{journal.e(label)}</a>'


def check_html(source):
    class Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag in {"script", "iframe", "img", "link", "object", "embed", "form"}:
                raise ValueError("offline report cannot load or send external resources")
            if tag == "a":
                href = attrs.get("href", "")
                if not href or urlparse(href).scheme or urlparse(href).netloc:
                    raise ValueError("all evidence links must be local")
                path = (HTML.parent / unquote(href)).resolve(strict=True)
                path.relative_to(journal.REPO.resolve())
    if '<meta charset="utf-8">' not in source or "905,274" not in source:
        raise ValueError("report encoding or claim denominator missing")
    Parser().feed(source)


def render(index, reads, summary):
    metrics = []
    for label, field in SPOTLIGHT:
        row = index["node/claim", "metadata." + field]
        metrics.append(f'<tr><td>{journal.e(label)}</td><td>{row["nonempty"]:,}</td><td>{row["nonempty_pct"]:.2f}%</td><td>{row["present_pct"]:.2f}%</td></tr>')
    lineage = []
    for item in reads:
        lineage.append(f'<details><summary>{journal.e(item["id"])} — {journal.e(item["lineage"])}</summary><p>{local_link(journal.REPO / item["path"], item["path"])}，{item["start"]}–{item["end"]} 行</p><pre>{journal.e(item["source_excerpt"])}</pre></details>')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KG metadata 覆盖与用途审查</title>
<style>body{{margin:0;background:#f5f7fa;color:#172c40;font-family:Segoe UI,Microsoft YaHei,sans-serif;line-height:1.7}}main{{max-width:1080px;margin:auto;padding:32px}}h1{{font-size:1.9rem}}h2{{margin-top:32px;font-size:1.3rem}}.notice{{background:#e6eef9;border-left:4px solid #28598a;padding:14px 18px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:9px 14px;border-bottom:1px solid #dbe3eb;text-align:left}}th{{background:#eaf0f6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:14px;font-size:.85rem}}details{{background:white;padding:14px;margin:12px 0;border:1px solid #dbe3eb}}a{{color:#235b91}}.scroll{{overflow-x:auto}}footer{{margin-top:30px;font-size:.9rem;color:#506477}}@media(max-width:640px){{main{{padding:18px}}td,th{{padding:8px}}}}</style></head><body><main>
<p>{local_link(journal.REPORT, "返回八小时工作日志")}</p><h1>metadata：哪些有值，哪些有用途</h1>
<p class="notice">不是 100% 覆盖，也不能靠删低覆盖字段来证明图谱更干净。本页只提出阅读分组，未删除字段、未改应用代码、未运行模型或实验。</p>
<p>当前已接受候选：2,587,961 节点、905,274 claims、3,012,308 边。节点 metadata 顶层字段并集 130，边为 37；不是每个节点都有 130 个字段。下表均以全部 905,274 条 claim 为分母，使用本次已冻结全图普查，不使用抽样估算。</p>
<h2>1. 关键字段覆盖率</h2><div class="scroll"><table><thead><tr><th>字段</th><th>非空记录数</th><th>非空率</th><th>键存在率</th></tr></thead><tbody>{''.join(metrics)}</tbody></table></div>
<p>键存在不等于有值，非空不等于科学有效。0 和 false 不当作缺失；字符串占位符另有记录。统计量可能是数值、字符串或空值，仍需格式与原文验证。缺少 p 值也不自动意味着一条关系不合格。</p>
<h2>2. 建议的四组阅读信息</h2><div class="scroll"><table><thead><tr><th>阅读层</th><th>包含内容</th><th>存储处置</th></tr></thead><tbody>
<tr><td>核心关系与原文</td><td>ID、名称、谓词、否定、置信信息、原始证据</td><td>保留；日常优先展示</td></tr>
<tr><td>科学上下文</td><td>主客体声明类型、条件、人群；保留原位置</td><td>保留；未知不补造，冲突不覆盖</td></tr>
<tr><td>来源与可选证据</td><td>PMID/DOI/标题；研究类型、方法、样本量、统计量</td><td>保留；按需要展开，不把空值补成 0</td></tr>
<tr><td>后台保留／待逐项核实</td><td>审核、版本、来源定位、扩展信息等</td><td>折叠阅读，不从正式图移除</td></tr></tbody></table></div>
<p>这是一份阅读合同建议，不是已经接入应用的新视图。删除清单为空。字段级清单包含 {summary["coverage_rows"]} 个“记录类别 × 路径”条目，包含嵌套和非 metadata 顶层属性，因此不能与 130/37 的顶层并集直接比较。</p>
<h2>3. 已核对的代码读取路径</h2><p>下面 {len(reads)} 组代码证据来自冻结的本地源文件。只检查了代码，没有启动服务、生成假设或做任何训练。它们证明存在相应读取路径，不证明某次实验实际使用了某个值。</p>{''.join(lineage)}
<h2>4. 明确的限制与下一步</h2><p>全量静态索引有 1,983 个字面读取候选和 254 处动态读取位置；键名相同可能来自实验结果表或其他对象，不能全部计为 KG 消费者。没有字面命中也不能证明未使用。完整 evidence/source_paper 还进入证据哈希；删掉子字段可能破坏已有审核。</p>
<p>下一步优先检查数值格式、来源与原文是否对应、明确类型和 ID 是否一致。若需要实际精简界面，先实现独立只读视图，保留回到完整记录的定位；不要把展示视图替换成正式图谱。</p>
<p>{local_link(OUTPUT / "FIELD_REVIEW.jsonl", "453 行字段覆盖与处置依据")} · {local_link(OUTPUT / "READING_CONTRACT.json", "机器可读的分组建议")} · {local_link(OUTPUT / "REVIEWED_CODE_READS.json", "冻结的代码证据")}</p>
<footer>生成时间 {journal.e(summary["created_at"])}。本轮未新增节点、边或 metadata 字段，未改变任何图谱统计。四个类型槽位补丁仍是另一轮的暂存结果，不在本页计作已应用。</footer></main></body></html>'''


def build():
    if OUTPUT.exists():
        raise ValueError("versioned review directory already exists")
    campaign = read_json(journal.OUTPUT / "CAMPAIGN.json")
    if campaign["active_process"]:
        raise ValueError("another campaign process is active")
    census = read_json(CENSUS / "CENSUS_COMPLETE.json")
    guards(census)
    for name in ("METADATA_NESTED_COVERAGE.jsonl", "STATIC_KEY_READ_CANDIDATES.jsonl", "DYNAMIC_KEY_READS.jsonl"):
        deep_check(census["artifacts"][name])
    index = indexed_coverage(jsonl(CENSUS / "METADATA_NESTED_COVERAGE.jsonl"))
    reads = [reviewed_read(spec, census["code_inventory"]) for spec in READS]
    rows = [classify(row) for row in index.values()]
    if len(rows) != census["coverage_rows"] or {r["denominator"] for r in rows if r["scope"] == "node/claim"} != {census["counts"]["claims"]}:
        raise ValueError("current coverage scope differs")
    OUTPUT.mkdir()
    summary = {"status": "READING_CONTRACT_PROPOSED_NO_STORAGE_OR_RUNTIME_CHANGE", "created_at": journal.utc_now(),
               "coverage_rows": len(rows), "reading_group_counts": dict(Counter(r["reading_group"] for r in rows)),
               "reviewed_code_blocks": len(reads), "delete_fields": [], "new_graph_fields": [],
               "graph_counts": census["counts"], "node_metadata_key_union": 130, "edge_metadata_key_union": 37,
               "runtime_view_implemented": False, "experiment_usage_measured": False,
               "source_graph": census["source_graph"], "source_acceptance": census["source_acceptance"],
               "source_census": fingerprint(CENSUS / "CENSUS_COMPLETE.json"),
               "source_artifacts": {name: census["artifacts"][name] for name in ("METADATA_NESTED_COVERAGE.jsonl", "STATIC_KEY_READ_CANDIDATES.jsonl", "DYNAMIC_KEY_READS.jsonl")},
               "implementation": {name: fingerprint(journal.REPO / name) for name in CODE},
               "formal_sources": census["formal_sources"], "models_called": 0, "training_jobs_started": 0}
    writer = JsonlWriter(OUTPUT / "FIELD_REVIEW.jsonl")
    for row in rows:
        writer.add(row)
    writer.close()
    journal.atomic_json(OUTPUT / "READING_CONTRACT.json", summary)
    journal.atomic_json(OUTPUT / "REVIEWED_CODE_READS.json", reads)
    journal.atomic_text(HTML, render(index, reads, summary))
    check_html(HTML.read_text(encoding="utf-8"))
    guards(census)
    journal.atomic_json(OUTPUT / "BUILD_RECEIPT.json", {"summary": fingerprint(OUTPUT / "READING_CONTRACT.json"),
        "artifacts": {name: fingerprint(OUTPUT / name) for name in ("FIELD_REVIEW.jsonl", "REVIEWED_CODE_READS.json", "METADATA_REVIEW.html")}})
    print(json.dumps({"status": summary["status"], "fields": len(rows), "code_blocks": len(reads)}, ensure_ascii=True))


def accept():
    if (OUTPUT / "REVIEW_COMPLETE.json").exists():
        raise ValueError("review already frozen")
    receipt = read_json(OUTPUT / "BUILD_RECEIPT.json")
    summary = read_json(OUTPUT / "READING_CONTRACT.json")
    for item in (receipt, summary):
        guards(item)
    for fp in [receipt["summary"], *receipt["artifacts"].values(), *summary["implementation"].values()]:
        deep_check(fp)
    tests_path = OUTPUT / "TEST_RESULTS.xml"
    suites = list(ElementTree.parse(tests_path).getroot().iter("testsuite"))
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    if tests["tests"] < 15 or any(tests[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("review tests must all pass")
    complete = {"status": "METADATA_READING_REVIEW_COMPLETE_NO_GRAPH_MUTATION", "completed_at": journal.utc_now(),
                "build": fingerprint(OUTPUT / "BUILD_RECEIPT.json"), "tests": tests, "test_artifact": fingerprint(tests_path),
                "storage_deletions": 0, "runtime_changes": 0, "graph_changes": 0,
                "models_called": 0, "training_jobs_started": 0}
    journal.atomic_json(OUTPUT / "REVIEW_COMPLETE.json", complete)
    journal.append_event({"id": "round04-metadata-reading-contract", "status": "completed", "at": complete["completed_at"],
        "title": "metadata 用途核对：四组阅读建议，删除清单为空",
        "body": "整理了 453 个类别×字段路径的覆盖和处置依据，并逐段核对 6 组实际代码读取路径。样本量 5.45%、p 值 0.62%、效应量 1.51% 的非空覆盖很低，但确有展示、质量检查或证据哈希用途，不能据此删除。将核心关系、科学上下文、来源/可选证据、后台信息分组阅读；这份建议尚未接入应用。",
        "changes": "未删字段、未新增 schema、未改模型或训练。metadata 仍是节点 130 / 边 37 个顶层键并集，图谱数量不变。",
        "details": {"tests": tests, "reading_group_counts": summary["reading_group_counts"], "deletion_list": [], "experiment_usage_measured": False},
        "artifacts": [str(HTML), str(OUTPUT / "REVIEW_COMPLETE.json"), str(OUTPUT / "FIELD_REVIEW.jsonl"), str(tests_path)]})
    print(json.dumps(complete, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "accept"))
    args = parser.parse_args()
    (build if args.phase == "build" else accept)()
