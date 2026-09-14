"""Current-only control state and an HTML journal with retired links disabled."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import os
import sys
from urllib.parse import quote, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
import prune_current_kg as prune
from kg_accepted_candidate_lineage import Bindings, require
from prepare_kg_simplified_reading import document
from project_kg_current_metadata_coverage import SPOTLIGHT

EVENT_ID = "confirmed40_deleted_single_version_retention_20260908"


def file_link(path, label=None):
    p = Path(path)
    label = label or p.name
    if not p.is_file():
        return f'<span>{escape(label)}（历史文件已按单版本策略移除）</span>'
    p.resolve().relative_to(journal.REPO.resolve())
    url = quote(os.path.relpath(p, journal.REPORT.parent).replace("\\", "/"), safe="/.")
    return f'<a href="{url}">{escape(label)}</a>'


def html(campaign, current, coverage, events, retention):
    counts = current["counts"]
    scope = "当前优化工作图；正式 full_v2 尚未同步"
    rows = "".join(f'<tr><th>{label}</th><td>{value:,}</td></tr>' for label, value in (
        ("节点总数", counts["nodes"]), ("其中声明节点", counts["claims"]), ("边总数", counts["edges"]),
        ("节点 metadata 字段并集", current["node_metadata_field_union"]),
        ("边 metadata 字段并集", current["edge_metadata_field_union"]),
        ("连接分量", current["connected_components"]), ("孤立节点（保留）", current["isolated_nodes"])))
    index = {(r["scope"], r["field"]): r for r in coverage}
    field_rows = ""
    for label, field in SPOTLIGHT:
        item = index["node/claim", "metadata." + field]
        field_rows += f'<tr><td>{escape(label)}</td><td>{item["present"]:,}</td><td>{item["nonempty"]:,}</td><td>{item["nonempty_pct"]:.6f}%</td></tr>'
    timeline = ""
    for event in reversed(events):
        links = " · ".join(file_link(p) for p in event.get("artifacts", []))
        timeline += (f'<details><summary>{escape(event.get("at", ""))} — {escape(event["title"])}</summary>'
                     f'<p>{escape(event.get("body", ""))}</p><p>{escape(event.get("changes", ""))}</p><p>{links}</p></details>')
    body = f'''<h1>KG 当前状态与工作记录</h1><p>更新：{escape(campaign["updated_at"])}。{escape(scope)}。</p>
<div class="notice">本次已删除40条确认有问题的声明及120条关联边。此登记批次的确认问题剩余0条，21条待复审保留；这不是全图无错误证明。不涉及模型、KGE或训练。</div>
<h2>当前唯一工作版本</h2><table>{rows}</table><p>{file_link(current["graph"]["path"], "当前 KG JSON")} · {file_link(current["detail_store"]["path"], "当前必要 UMLS 明细库")} · {file_link(prune.OUTPUT / "CURRENT_ACCEPTANCE.json", "当前验收记录")}</p>
<p>保留所有非目标概念节点，包括本次新增的{current["newly_isolated_nodes"]}个孤立节点；未按孤立状态或名称相似度继续删除。剩余历史候选队列：基因{campaign["pending_gene_link_claims"]:,}条声明、端点{campaign["remaining_opposite_endpoint_claims"]}条，队列可能重叠，不直接相加。</p>
<h2>单版本保留策略</h2><p>用户已取消逐步回退保留。当前图经临时写入、独立完整扫描后原子替换；没有新备份或删除前像。38个旧图、重复明细库及回退文件已永久移除，释放{retention["allocated_bytes_reclaimed"] / 1024**3:.3f} GiB实际分配空间。没有本地逐步回退可用。</p>
<p>只清理本轮优化链及明确的 KG 变更备份。正式应用当前版本、实验快照、原始文献和小型历史说明没有删除。UMLS明细库包含当前节点需要的离线数据，不是旧图备份。</p>
<h2>验收结果</h2><p>38项删除测试通过。源图边读边完整SHA-256校验；新图独立全量读取、重新计算拓扑及哈希；当前明细库完整SHA-256一致。保留记录的值、JSON类型和顺序一致；悬空端点、重复节点ID、失效claim所有者、指向本次删除ID的精确引用均为0。</p>
<p>HTML已做结构和本地链接检查；未宣称浏览器视觉验收。来源审阅不是本次重新执行的工作。</p>
<h2>当前 metadata 覆盖</h2><p>以下以{counts["claims"]:,}条声明为分母，依据上一版全量统计扣除本次实际删除记录，并由两套计算交叉核对。字段存在不等于有值，有值不等于科学有效或实验可用；未补填数据。</p>
<div class="scroll"><table><tr><th>字段</th><th>字段存在</th><th>非空</th><th>非空率</th></tr>{field_rows}</table></div>
<p>{file_link(current["current_coverage"]["path"], "完整当前覆盖统计")} · {file_link(current["current_issues"]["path"], "21条当前待复审声明")}</p>
<h2>正式应用边界</h2><p>full_v2仍是原正式版本，905,274条声明；当前工作图905,184条，相差累计已删除的90条。正式同步尚未执行，不能只替换图文件，也不能把上述差额视为新的90条问题。</p>
<h2>历史工作日志</h2><p>以下记录按发生时的版本理解，不代表当前状态；历史验收凭据可能引用已经主动删除的旧图，不能再用于重放或回退。</p>{timeline}'''
    return document("KG 当前状态与工作记录", body)


def check_html(page):
    class Audit(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = 0
        def handle_starttag(self, tag, attrs):
            require(tag not in {"script", "iframe", "object", "form"}, "active HTML content")
            for key, value in attrs:
                require(not key.lower().startswith("on"), "event handler")
                if key in {"src", "href"}:
                    parsed = urlparse(value)
                    require(not parsed.scheme and not parsed.netloc, "external link/resource")
                    if parsed.path:
                        path = (journal.REPORT.parent / unquote(parsed.path)).resolve(strict=True)
                        path.relative_to(journal.REPO.resolve())
                        require(path.is_file(), "broken local link")
                        self.links += 1
    audit = Audit()
    audit.feed(page)
    return dict(local_links=audit.links, broken_links=0, scripts=0, remote_resources=0, browser_visual_qa_completed=False)


def deliver():
    old = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    current = journal.read_json(prune.OUTPUT / "CURRENT_ACCEPTANCE.json")
    retention = journal.read_json(prune.OUTPUT / "RETENTION_RESULT.json")
    journal.guards([current["graph"], current["detail_store"], current["current_issues"], current["current_coverage"], old["formal_sources"]])
    b = Bindings()
    aux = journal.read_json(prune.R23 / "AUXILIARY_COHORT_CONTINUITY.json")
    gene = b.rows(aux["PENDING_GENE_LINKS.jsonl"]["source"])
    opposite = b.rows(aux["FINAL_OPPOSITE_ENDPOINT_AUDITS.jsonl"]["source"])
    deleted = set(current["deleted_claim_ids"])
    require(not deleted & {r["claim_id"] for r in gene + opposite}, "auxiliary queue needs projection")
    at = journal.utc_now()
    campaign = dict(schema="kg.current_single_version.v2", created_at=old["created_at"], updated_at=at,
                    status="COMPLETED", phase="40条确认问题删除及单版本整理完成", active_process=None,
                    automation_id=None, automation_state="user_deleted_not_recreated", execution_mode="manual_user_resumed",
                    current_acceptance=journal.fingerprint(prune.OUTPUT / "CURRENT_ACCEPTANCE.json"),
                    current_graph=current["graph"], current_detail_store=current["detail_store"], counts=current["counts"],
                    node_metadata_field_union=current["node_metadata_field_union"], edge_metadata_field_union=current["edge_metadata_field_union"],
                    current_coverage=current["current_coverage"], current_issues=current["current_issues"],
                    pending_gene_link_claims=len({r["claim_id"] for r in gene}), pending_gene_link_endpoint_events=len(gene),
                    remaining_opposite_endpoint_claims=sum(r["reason"] != "ok" for r in opposite),
                    historical_confirmed_issue_claims=90, confirmed_issue_claims_removed_from_current=90,
                    confirmed_source_or_expression_issue_claims=0, additional_semantic_or_detail_review_candidates=21,
                    issue_register_is_exhaustive=False, queue_counts_may_overlap=True,
                    formal_sources=old["formal_sources"], formal_apply_performed=False, formal_claims=905274,
                    formal_sync_gap_claims=90, rollback_retention=False,
                    retention_result=journal.fingerprint(prune.OUTPUT / "RETENTION_RESULT.json"),
                    report_path=str(journal.REPORT), last_deep_verification=journal.read_json(prune.OUTPUT / "VALIDATED.json")["at"],
                    last_current_graph_content_verification={"graph": current["graph"], "independent_full_scan": True},
                    retention_policy="只保留当前优化KG和必要明细；不再保存每步备份或删除前像。历史小型工作说明保留，但不是可用回退。",
                    next_steps=["21条待复审按来源集中审阅，未确认之前不自动删除。", "1204条基因声明/15条端点历史候选队列保留，可能与复审队列重叠，不重新计为确认错误。", "等待用户对正式full_v2同步的选择：尚未授权/执行正式替换；正式采用需图、声明库、明细、状态一起对齐，不产生新回退副本。"],
                    boundaries=["仅KG；无模型或训练", "单当前版本，不恢复自动定时任务", "历史实验快照和原始数据不属于本次优化副本清理范围"])
    log = journal.read_json(journal.OUTPUT / "WORK_LOG.json")
    require(not any(r["id"] == EVENT_ID for r in log["events"]), "delivery event already recorded")
    event = dict(id=EVENT_ID, at=at, status="completed", title="删除40条确认问题；改为单版本保留，清理旧KG副本",
                 body=f"删除40个声明节点和120条边，当前2587871节点/905184声明/3012015边。21条待复审与其他历史候选保留。永久移除38个明确旧图/重复库/回退文件，实际释放{retention['allocated_bytes_reclaimed']/1024**3:.3f} GiB；没有本地逐步回退。正式full_v2未同步。",
                 changes="38项删除回归通过。源图完整SHA，新图独立完整拓扑/记录一致性/SHA，当前SQLite完整SHA通过。仅临时文件后原子替换，无新回退前像。",
                 artifacts=[str(prune.OUTPUT / "CURRENT_ACCEPTANCE.json"), str(prune.OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"), str(prune.OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl"), str(prune.OUTPUT / "RETENTION_RESULT.json")])
    log["events"].append(event)
    page = html(campaign, current, prune.rows(current["current_coverage"]["path"]), log["events"], retention)
    qa = check_html(page)
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    journal.atomic_json(journal.OUTPUT / "WORK_LOG.json", log)
    journal.atomic_text(journal.REPORT, page)
    journal.atomic_text(journal.OUTPUT / "HANDOFF.md", f'''# Current KG — user-requested single-version retention

Updated {at}. Current acceptance: round26_confirmed_deletion/CURRENT_ACCEPTANCE.json.
Current KG: {current['graph']['path']}
Current required detail store: {current['detail_store']['path']}
Counts: 2587871 nodes, 905184 claims, 3012015 edges. Metadata unions130/37.

User explicitly approved removal of40confirmed claims and no per-step rollback.
Deleted40nodes/120edges;64new isolates intentionally retained.21additional review
claims and historical1204gene/15endpoint candidate queues remain, overlap possible.
No new scientific adjudication; no models/training. Formal full_v2 unchanged,
905274claims; cumulative formal sync gap90. Asynchronous user choice about formal
adoption is pending; do not silently replace formal graph alone.

38oldKG/duplicateSQLite/rollback files permanently deleted, allocated bytes freed
{retention['allocated_bytes_reclaimed']}. User supersedes all earlier mandatory
rollback/immutable-version retention policies. No per-step local rollback exists.
One working KG remains in optimization tree. Formal live graph, experiment
snapshots/smoke fixtures, raw corpora and small historical reports retained.
Current NODE_REPAIRS rollback log removed; its top-metadata link removed as well.
Do not regenerate deleted versions or treat old receipts as live dependencies.

38deletion tests passed. Build session74947 and validation session65457 completed.
Source complete SHA verified during rewrite; output full SHA+independent topology
and retained JSON record order/types verified; current SQLite fullSHA verified.
No dangling endpoint, duplicate node ID, missing claim owner or remaining exact
reference to40deleted IDs. All non-deleted record payloads unchanged.

Current HTML: KG_OVERNIGHT_20260907.html, newest event {EVENT_ID}.
Renderer: neurooracle/scripts/report_current_kg.py. Old R23/R24/R25 build/render
commands are historical and may depend on intentionally removed files; do not run.
CAMPAIGN.json now a compact v2 current-only state, WORK_LOG.json retains history.
No active process/timer. HTML structural/link checks pass, no browser visual QA.
''')
    journal.atomic_json(prune.OUTPUT / "REPORT_QA.json", qa)
    journal.atomic_json(prune.OUTPUT / "RUN_STATE.json", dict(status="COMPLETED", phase="CURRENT_GRAPH_AND_RETENTION_COMPLETE",
                       updated_at=at, graph=current["graph"], counts=current["counts"], active_process=None))
    print(dict(status="CURRENT_ONLY_HANDOFF_COMPLETE", events=len(log["events"]), **qa), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("deliver",))
    deliver()
