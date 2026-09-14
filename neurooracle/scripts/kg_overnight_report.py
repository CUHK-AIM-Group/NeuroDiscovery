"""Local, offline KG work journal. No graph writes, model calls or scheduling.

The product's heartbeat owns scheduling; this script only records completed
work and renders a self-contained HTML report from an atomic JSON journal.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "neurooracle/data/umls_mapping/umls_2026AA_atomic_mentions_v1_20260906"
OUTPUT = DATA / "kg_overnight_20260907"
BASELINE = DATA / "remaining_endpoint_repair_v1_20260907"
REPORT = REPO / "KG_OVERNIGHT_20260907.html"
HKT = timezone(timedelta(hours=8))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("file changed while binding: " + str(path))
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def guards(value):
    if isinstance(value, dict):
        if {"path", "bytes", "mtime_ns", "sha256"} <= value.keys():
            path = Path(value["path"]).resolve(strict=True)
            stat = path.stat()
            if str(path) != value["path"] or (stat.st_size, stat.st_mtime_ns) != (value["bytes"], value["mtime_ns"]):
                raise ValueError("frozen file guard changed: " + str(path))
        else:
            for child in value.values():
                guards(child)
    elif isinstance(value, list):
        for child in value:
            guards(child)


def local_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(HKT).strftime("%Y-%m-%d %H:%M:%S")


def e(value):
    return html.escape(str(value), quote=True)


def link(path, label=None):
    path = Path(path).resolve(strict=True)
    path.relative_to(REPO.resolve())
    href = quote(os.path.relpath(path, REPORT.parent).replace("\\", "/"), safe="/")
    return f'<a href="{e(href)}">{e(label or path.name)}</a>'


def init():
    if (OUTPUT / "CAMPAIGN.json").exists() or REPORT.exists():
        raise ValueError("existing campaign/report must be resumed, never reset")
    accepted = read_json(BASELINE / "REPAIR_ACCEPTANCE.json")
    guards(accepted)
    now = utc_now()
    campaign = {
        "schema": "kg-overnight-local-journal.v1", "created_at": now,
        "deadline": "2026-09-07T06:15:00+00:00", "timezone": "Asia/Hong_Kong",
        "minimum_work_window_hours": 8, "latest_user_extension_at": "2026-09-06T22:08:01+00:00",
        "automation_id": "kg-html", "status": "ACTIVE", "phase": "共享节点全引用审查",
        "updated_at": now, "report_path": str(REPORT),
        "baseline_acceptance": fingerprint(BASELINE / "REPAIR_ACCEPTANCE.json"),
        "current_acceptance": fingerprint(BASELINE / "REPAIR_ACCEPTANCE.json"),
        "last_deep_verification": read_json(BASELINE / "VALIDATION_COMPLETE.json").get("completed_at"),
        "counts": accepted["counts"], "node_metadata_field_union": accepted["node_metadata_field_union"],
        "edge_metadata_field_union": accepted["edge_metadata_field_union"],
        "remaining_opposite_endpoint_claims": accepted["remaining_opposite_endpoint_claims"],
        "pending_gene_link_claims": accepted["pending_gene_link_claims"],
        "pending_gene_link_endpoint_events": accepted["pending_gene_link_endpoint_events"],
        "baseline_tests": accepted["tests"], "formal_sources": accepted["formal_sources"],
        "accepted_rounds": [], "active_process": None,
        "next_steps": [
            "核对 adolescent depression 和 negative symptoms 的全部引用，区分疾病/风险、单症状/复合症状。",
            "保留其余 13 条端点的完整限定条件，寻找有证据支持的现有身份；不按截短名称自动修复。",
            "对 1,205 条 FANCE/VCP 待审 claim 分层，优先找精确可验证的小批修复。",
            "把 metadata 的实际覆盖率、运行时读取情况及可精简建议整理成有证据的清单。",
            "按需要合并成一份独立候选，完成回归、全量完整性检查和可回退验证；不替换正式图。",
        ],
        "boundaries": ["仅 KG，不训练、不运行 KGE 或模型实验", "正式 full_v2 与旧候选只读",
                       "不按多数票填类型，不改写证据", "不删历史版本，不提交 Git，不上传或发布报告"],
    }
    atomic_json(OUTPUT / "CAMPAIGN.json", campaign)
    atomic_json(OUTPUT / "WORK_LOG.json", {"schema": "kg-work-log.v1", "events": [{
        "id": "night-start", "at": now, "status": "completed", "title": "夜间工作已启动，已冻结候选作为起点",
        "body": "已检查最新候选、实现代码与正式文件的冻结大小/时间绑定，均未变化。用户追加要求至少八小时；后台按约 20 分钟一轮续跑，香港时间 14:15 开始收尾。此处是计划窗口，不是已经完成了八小时工作。",
        "changes": "本步骤未修改图数据。接续此前 270 项测试通过的候选；这 270 项不是今晚新跑的测试。",
        "artifacts": [str(BASELINE / "REPAIR_REPORT.md"), str(BASELINE / "REPAIR_ACCEPTANCE.json")],
    }]})
    render()


def append_event(event):
    for key in ("id", "title", "body", "status"):
        if not isinstance(event.get(key), str) or not event[key].strip():
            raise ValueError("missing event " + key)
    if event["status"] not in {"completed", "running", "held", "failed", "planned"}:
        raise ValueError("invalid event status")
    journal = read_json(OUTPUT / "WORK_LOG.json")
    if any(row["id"] == event["id"] for row in journal["events"]):
        raise ValueError("event ID already recorded; never count it twice")
    event = dict(event, at=event.get("at") or utc_now())
    for artifact in event.get("artifacts", []):
        Path(artifact).resolve(strict=True).relative_to(REPO.resolve())
    journal["events"].append(event)
    atomic_json(OUTPUT / "WORK_LOG.json", journal)
    campaign = read_json(OUTPUT / "CAMPAIGN.json")
    campaign["updated_at"] = event["at"]
    atomic_json(OUTPUT / "CAMPAIGN.json", campaign)
    render()


def render():
    campaign, journal = read_json(OUTPUT / "CAMPAIGN.json"), read_json(OUTPUT / "WORK_LOG.json")
    accepted = read_json(campaign["current_acceptance"]["path"])
    guards(campaign["current_acceptance"])
    baseline = read_json(campaign["baseline_acceptance"]["path"])
    counts = accepted["counts"]
    rows = "".join(f'<tr><th>{label}</th><td>{baseline[key]:,}</td><td>{accepted[key]:,}</td></tr>' for label, key in [
        ("节点 metadata 字段并集", "node_metadata_field_union"), ("边 metadata 字段并集", "edge_metadata_field_union"),
        ("这批端点告警（非全图）", "remaining_opposite_endpoint_claims"), ("FANCE/VCP 待审 claim（单独队列）", "pending_gene_link_claims")])
    statuses = {"completed": "已完成", "running": "执行中", "held": "暂缓", "failed": "失败", "planned": "计划"}
    events = []
    for row in reversed(journal["events"]):
        files = " · ".join(link(path) for path in row.get("artifacts", []))
        detail = ""
        if row.get("details"):
            detail = f'<details><summary>详细依据</summary><pre>{e(json.dumps(row["details"], ensure_ascii=False, indent=2))}</pre></details>'
        events.append(f'<article class="entry"><div class="stamp"><time>{e(local_time(row["at"]))}</time><span class="tag {e(row["status"])}">{e(statuses[row["status"]])}</span></div><h3>{e(row["title"])}</h3><p>{e(row["body"])}</p><p class="muted">{e(row.get("changes", ""))}</p>{detail}<p class="files">{files}</p></article>')
    stats = "".join(f'<div class="metric"><span>{label}</span><strong>{counts[key]:,}</strong><small>较夜间起点 {counts[key]-baseline["counts"][key]:+,}</small></div>' for label, key in [("已验证候选节点", "nodes"), ("已验证候选边", "edges"), ("其中 claim 节点", "claims")])
    next_steps = "".join(f'<li>{e(item)}</li>' for item in campaign["next_steps"])
    boundary = " · ".join(e(item) for item in campaign["boundaries"])
    state = {"ACTIVE": "工作窗口进行中", "COMPLETED": "本次夜间工作已收尾", "BLOCKED": "需处理阻碍", "PAUSED": "已暂停"}.get(campaign["status"], campaign["status"])
    last_deep = campaign.get("last_deep_verification")
    last_deep = local_time(last_deep) if last_deep else "见候选 VALIDATION_COMPLETE.json"
    snapshot_at = max([campaign["updated_at"]] + [row["at"] for row in journal["events"]])
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>KG 夜间工作记录 · 2026-09-07</title><style>
:root{{color-scheme:light;--ink:#142333;--muted:#536679;--line:#dce4ec;--blue:#1359a0;--bg:#f3f6fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1120px;margin:auto;padding:36px 28px 72px}}header{{border-top:5px solid var(--blue);padding:24px 0 20px}}h1{{font-size:2rem;line-height:1.35;margin:6px 0 12px}}h2{{font-size:1.4rem;margin:36px 0 14px}}h3{{font-size:1.05rem;margin:10px 0 5px}}p{{margin:8px 0}}a{{color:var(--blue);text-underline-offset:3px}}code,pre{{overflow-wrap:anywhere}}.eyebrow,.muted,small{{color:var(--muted)}}.eyebrow{{font-size:.88rem;letter-spacing:.06em}}.status{{padding:14px 18px;background:#e5effa;border-left:4px solid var(--blue)}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}}.metric,.entry,section.panel{{background:white;border:1px solid var(--line);border-radius:8px;padding:20px}}.metric span,.metric strong,.metric small{{display:block}}.metric strong{{font-size:2rem;font-variant-numeric:tabular-nums;letter-spacing:-.035em}}small{{font-size:.88rem}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:580px;text-align:left}}th,td{{padding:10px 14px;border-bottom:1px solid var(--line);font-size:.95rem}}thead{{background:#edf2f7}}td{{font-variant-numeric:tabular-nums}}.entry{{margin:14px 0}}.stamp{{display:flex;gap:16px;justify-content:space-between;align-items:center;font-size:.88rem;color:var(--muted)}}.tag{{padding:2px 10px;background:#edf2f7;border-radius:4px;white-space:nowrap}}.completed{{color:#11603b;background:#e3f4ea}}.held{{color:#815b12;background:#fff1ce}}.failed{{color:#9c2828;background:#fde8e8}}.running{{color:#155799;background:#e4effb}}.files{{font-size:.9rem;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;font-size:.86rem;line-height:1.6;background:#f3f6fa;padding:12px;border-radius:5px;max-height:550px;overflow:auto}}summary{{cursor:pointer;color:var(--blue)}}li{{margin:8px 0}}footer{{margin-top:35px;padding-top:16px;border-top:1px solid var(--line);font-size:.88rem;color:var(--muted)}}
@media(max-width:720px){{main{{padding:20px 16px 40px}}.metrics{{grid-template-columns:1fr}}h1{{font-size:1.6rem}}.metric strong{{font-size:1.75rem}}.stamp{{align-items:flex-start}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}.entry{{break-inside:avoid}}details{{display:block}}a{{color:inherit}}}}
</style></head><body><main>
<header><div class="eyebrow">NEUROCLAW / KG QUALITY JOURNAL</div><h1>KG 夜间工作记录</h1><p>本地离线报告 · 所有时间为香港时间（UTC+8）</p><p class="muted">记录截至 {e(local_time(snapshot_at))}；计划窗口 {e(local_time(campaign["created_at"]))} — {e(local_time(campaign["deadline"]))}。</p></header>
<div class="status"><strong>{e(state)}</strong><br>当前阶段：{e(campaign["phase"])}。这是最近一次保存的状态，不是电脑或后台任务的实时在线指示。</div>
<div class="metrics">{stats}</div>
<section class="panel"><strong>结论边界</strong><p>{boundary}。</p><p>本报告只把通过独立校验的候选计入图谱变化。规则告警通过 ≠ 原论文事实已认证；暂缓项仍保留原始证据。当前正式图没有被夜间候选替换。</p></section>
<h2>图谱与 metadata 变化</h2><div class="table-wrap"><table><thead><tr><th>指标</th><th>夜间起点</th><th>最新已验证候选</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="muted">字段并集不是每个节点/边都应填写的模板，也不表示覆盖率为 100%。已知 raw_text 非空覆盖率为 97.004664%；证据容器存在不等于其每个子字段有效。</p>
<p class="files">当前候选凭据：{link(campaign["current_acceptance"]["path"])} · 起点覆盖率：{link(BASELINE / "METADATA_COVERAGE.json")}</p>
<h2>工作时间线</h2>{''.join(events)}
<h2>待办与收尾顺序</h2><ol>{next_steps}</ol>
<section class="panel"><strong>完整性与执行说明</strong><p>最近一次已接受候选的完整验证：{e(last_deep)}。日常状态检查复用冻结哈希，并核对路径、大小、修改时间；重要候选和最终交付按约定深验。后台续跑由当前任务的定时机制负责，不依赖这个 HTML 页面保持打开。</p><p>起点回归测试 {baseline["tests"]["tests"]} 项通过；新轮次测试在时间线单独列出，不重复计数。电脑需保持开机、应用运行且不休眠；如果任务中断，以最后保存的时间和证据为准。</p></section>
<footer>本文件不加载外部资源、不调用模型、不发送数据。重新打开或刷新可读取下一次保存的内容。机器可读记录：{link(OUTPUT / "CAMPAIGN.json")} · {link(OUTPUT / "WORK_LOG.json")}</footer>
</main></body></html>'''
    atomic_text(REPORT, page)
    print(json.dumps({"report": str(REPORT), "events": len(journal["events"]), "snapshot_at": snapshot_at}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "render", "event"))
    parser.add_argument("--event-file", type=Path)
    args = parser.parse_args()
    if args.phase == "init":
        init()
    elif args.phase == "event":
        if not args.event_file:
            parser.error("event requires --event-file")
        append_event(read_json(args.event_file))
    else:
        render()


if __name__ == "__main__":
    main()
