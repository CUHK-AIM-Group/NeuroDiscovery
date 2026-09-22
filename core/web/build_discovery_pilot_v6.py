"""Build the v6 expert-study pack: 34 externally validated, strongest-first cases.

Owner decision (2026-09-18): the panel evaluates only hypotheses that passed the
frozen external triage, strongest effects first, because the system presents its
best candidates to users. Panel size 34 comes from the study design
(10 experts x 10 materials x 3 reviewers each). The frozen ranked selection is
in study_materials/sources_v6/selection_v6.json (build_discovery_v6_selection.py).

Three cases already in v5 passed the triage and are reused verbatim (their card
content is byte-identical after renumbering). The other seven v5 cards failed
the external triage and leave the panel; the v5 pack itself stays immutable.
Thirty-one cards are newly built from the frozen handoffs with the same schema
and the same honest-source conventions as v2. Nothing here runs experiments.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

from core.web.build_discovery_pilot import clean_rationale, PAPER_NOTES

HERE = Path(__file__).resolve().parent
MATERIALS = HERE / "study_materials"
SOURCES = MATERIALS / "sources_v6"
PARENT = MATERIALS / "cs1_discovery_pilot_v5.json"
TARGET = MATERIALS / "cs1_discovery_pilot_v6.json"

PACK_ID = "cs1-discovery-capabilities-20260918-v6"
PROTOCOL_VERSION = "cs1-complete-output-single-round-v6"
PARENT_HASH = "d5648f5f4781936dddffc678be56a9be21827b4469ccd51f3527f2cf0fd2025b"
SELECTION_HASH = "84e79d82f5f248c2c8d9818eeb1f2bc94e7628c3516fd5a3c8d92984e2122c85"
SOURCE_HASHES = {
    "old_r1/candidate_index.json": "a16ff8cc00d87dcf2c1d6eeafb2d0ad4ae3ea8e55f7d8f061d2f7ad309ce1b74",
    "old_r1/proposal_slots.json": "0c84a047842f696eab92a484037edadebb55b0d7b93cb9f0f4f30c6356bf8239",
    "old_r1/feedback_lineage.json": "3cdaf2a2fc6aeeb9d9b56fb2213a987034b77ee49acb9aa0f2418c373ca82468",
    "r7/candidate_index.json": "40964c8ba056a2cc44615857030f48a177e24baff50a25a2a6b168bc2f74f19d",
    "r7/proposal_slots.json": "cb40e97a0e8190e8fe1bc724dc525f656a8a5afca809cb76e792caa09e08ca2d",
    "r7/feedback_lineage.json": "59cb63a2f658ee01352bc82393075b0ddf63c2dee6bef61dd16e3531e0e5d981",
}

NETWORKS = {"Vis": "视觉网络", "SomMot": "感觉运动网络", "DorsAttn": "背侧注意网络",
            "SalVentAttn": "显著性/腹侧注意网络", "Limbic": "边缘网络", "Cont": "控制网络", "Default": "默认网络"}
PLAIN_NETWORKS = {"Vis": "视觉网络", "SomMot": "躯体运动网络", "DorsAttn": "背侧注意网络",
                  "SalVentAttn": "显著性/腹侧注意网络", "Limbic": "边缘网络", "Cont": "控制网络", "Default": "默认网络"}
GROUPS_ZH = {"ADHD": "ADHD 患者", "bipolar": "双相障碍患者", "psychosis_SZ_SZA": "精神分裂症/分裂情感性障碍患者"}
CASES = {"ADHD": 13, "bipolar": 26, "psychosis_SZ_SZA": 16}
CONTROLS = 91
AUDIT_NOTE = "完整原始提议、六次模型评议、预检验锁定、真实父反馈和外部行均保留来源绑定。此页是汇总展示，不是重新运行实验。"
AUDIT_NOTE_FIRST_ROUND = "完整原始提议、六次模型评议、预检验锁定和外部行均保留来源绑定；该假设来自首轮提议，生成时同 seed 尚无已完成 TCP 反馈，无父假设。此页是汇总展示，不是重新运行实验。"

# v5 cards that passed the external triage stay in the panel, content unchanged.
REUSE_FROM_V5 = {
    "e3b5ea76fb92c756d9e3": "packet-01",
    "c94c1f0143413b424323": "packet-07",
    "02ef10ac7980d2a6c859": "packet-02",
}


def load_checked(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source changed: {Path(path).name}; build a separately reviewed new version.")
    return json.loads(raw)


def records_of(doc):
    return doc if isinstance(doc, list) else doc["records"]


def roi_info(record):
    info = record.get("source_ROI_label")
    if not info:
        info = (record.get("endpoint_definition") or {}).get("source_ROI_record")
    if not info:
        raise ValueError(f"ROI card without a source parcel label: {record['canonical_hypothesis_id']}")
    return info


def describe(record, counts):
    spec = record["canonical_record"]["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        edges = counts[a] * counts[b] if a != b else counts[a] * (counts[a] - 1) // 2
        title = f"{NETWORKS[a]}与{NETWORKS[b]}" if a != b else f"{NETWORKS[a]}内部连接"
        definition = (f"Schaefer-400 / 7-network；{a}（{counts[a]} 个分区）与 {b}（{counts[b]} 个分区），"
                      f"共 {edges:,} 条唯一非对角 Pearson r 边的算术均值。")
    else:
        if spec["kind"] != "ROI_to_network":
            raise ValueError("Unrecognised measurement definition.")
        roi = roi_info(record)
        net = record["endpoint"].split(":")[-1]
        edges = counts[net] - (1 if roi["network"] == net else 0)
        title = f"{'左' if '_LH_' in roi['name'] else '右'}侧 ROI{roi['roi']:03} 与{NETWORKS[net]}"
        definition = (f"Schaefer-400 / 7-network；ROI{roi['roi']:03} = {roi['name']}。"
                      f"该分区到 {net} 的 {edges} 条非对角 Pearson r 边的算术均值；不等同于整个同名解剖区。")
    return title, definition + "不作 Fisher 变换或结果驱动选边。"


def plain_hypothesis(record):
    """Plain-language, dataset-free restatement; same template family as v5."""
    canonical = record["canonical_record"]
    named = canonical["named"]
    if canonical["direction"] != -1:
        raise ValueError("Plain template only covers joint-decrease hypotheses; write deliberately.")
    if any(group not in GROUPS_ZH for group in named):
        raise ValueError(f"Uncovered diagnosis group: {named}")
    spec = canonical["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        subject = (f"{PLAIN_NETWORKS[a]}（{a}）与{PLAIN_NETWORKS[b]}（{b}）之间的功能连接强度，"
                   if a != b else f"{PLAIN_NETWORKS[a]}（{a}）内部各分区之间的功能连接强度，")
    else:
        roi = roi_info(record)
        net = record["endpoint"].split(":")[-1]
        subject = f"{'左' if '_LH_' in roi['name'] else '右'}侧 ROI{roi['roi']:03} 分区与{PLAIN_NETWORKS[net]}（{net}）之间的功能连接强度，"
    if len(named) == 2:
        groups = f"在{GROUPS_ZH[named[0]]}和{GROUPS_ZH[named[1]]}中预计均低于各自的对照组——两组下降方向一致。"
    elif len(named) == 3:
        bare = "、".join(GROUPS_ZH[g].replace(" 患者", "") for g in named)
        groups = f"在{bare}三组患者中预计均低于各自的对照组——三组下降方向一致。"
    else:
        raise ValueError(f"Plain template covers two or three named groups, got {named}")
    return subject + groups


def external_rows(record):
    rows = []
    for i, entry in enumerate(record["external_component_rows"], 1):
        r = entry["record"]
        boot_std = r.get("bootstrap_standardized_beta")
        if boot_std is None:
            field = entry.get("declared_direction_standardized_bootstrap_field")
            if not field or field not in r:
                raise ValueError(f"External row lacks declared-direction standardized bootstrap: {record['endpoint']}")
            boot_std = r[field]
        for key in ("bootstrap_direction_fraction", "bootstrap_percentiles_2_5_97_5_descriptive",
                    "bootstrap_valid", "bootstrap_total"):
            if key not in boot_std:
                raise ValueError(f"Standardized bootstrap lacks {key}: {record['endpoint']}")
        rows.append({"row_id": f"E{i}", **{key: r[key] for key in [
            "cohort", "domain", "variant", "role", "cases", "controls", "beta", "standardized_beta",
            "residual_RMS", "columns", "bootstrap_beta"]},
            "bootstrap_standardized_beta": boot_std})
    return rows


def make_card(record, proposal, lineage, card_id, counts):
    canonical = record["canonical_record"]
    decision = proposal.get("final_decision") or proposal["resolved_final_decision"]
    final = decision["final_card"]
    assert proposal["canonical_hypothesis_id"] == record["canonical_hypothesis_id"]
    assert proposal["committed_card"]["hypothesis_id"] == record["canonical_hypothesis_id"]
    assert proposal["committed_card"]["endpoint"] == record["endpoint"]
    assert set(final["named"]) == set(canonical["named"]) and final["direction"] == canonical["direction"]
    if lineage:
        assert lineage["child_hypothesis_id"] == record["canonical_hypothesis_id"]
    refs, aliases = [], {}
    evidence_by_id = {entry["record"]["id"]: entry["record"]
                      for entry in record["first_occurrence_cited_KG_records"] if entry.get("present", True)}
    assert len(canonical["evidence_ids"]) == len(canonical["model_evidence_ids"])
    for i, (eid, model_id) in enumerate(zip(canonical["evidence_ids"], canonical["model_evidence_ids"]), 1):
        if eid not in evidence_by_id:
            raise ValueError(f"Cited KG record absent from handoff: {eid}")
        record_, rid = evidence_by_id[eid], f"P{i}"
        paper = record_["source_paper"]
        aliases[model_id] = rid
        refs.append({"id": rid, "title": paper.get("title", ""), "year": paper.get("year"),
                     "journal": paper.get("journal", ""), "pmid": str(paper.get("pmid", "")),
                     "url": "https://doi.org/" + paper["doi"] if paper.get("doi") else "",
                     "recorded_sentence": record_.get("sentence", ""),
                     "note": PAPER_NOTES.get(str(paper.get("pmid", "")),
                                             "这是历史记录，不代表本版已核实其与精确候选的直接匹配；请结合原文判断。"),
                     "verification": "历史 KG 记录；未在本版独立核验原文，年份保留原记录。"})
    internal = [{"domain": name, "cases": CASES[name], "controls": CONTROLS, **result}
                for name, result in canonical["components"].items()]
    if lineage:
        parent_entry = lineage["parent_prior_same_seed_TCP_records"][0]
        parent = parent_entry.get("record", parent_entry)
        verified = lineage.get("same_feedback_verified")
        if verified is None:
            verified = lineage.get("feedback_values_and_temporal_order_verified")
        feedback = {"parent_endpoint": parent["endpoint"], "parent_named": parent["named"],
                    "parent_components": parent["feedback"]["components"],
                    "parent_joint": parent["feedback"]["joint_direction_stability"],
                    "action_text": clean_rationale(re.sub(r"[a-f0-9]{20}", "[父假设]", final["feedback_action_zh"]), aliases),
                    "binding_verified": bool(verified)}
        audit_note = AUDIT_NOTE
    else:
        feedback = {"parent_endpoint": None, "parent_named": [], "parent_components": {},
                    "parent_joint": None, "no_parent": True,
                    "action_text": clean_rationale(re.sub(r"[a-f0-9]{20}", "[父假设]", final["feedback_action_zh"]), aliases),
                    "binding_verified": False}
        audit_note = AUDIT_NOTE_FIRST_ROUND
    title, definition = describe(record, counts)
    return {"id": card_id,
            "pre": {"title": title, "hypothesis": canonical["hypothesis_zh"],
                    "hypothesis_plain": plain_hypothesis(record), "definition": definition,
                    "rationale": clean_rationale(final["rationale_zh"], aliases),
                    "transfer_limit": clean_rationale(final["evidence_transfer_limit_zh"], aliases),
                    "references": refs,
                    "source_note": "假设为检验前执行卡的程序渲染标题；理由和限制来自实际模型回答，仅规范化引文别名。"},
            "post": {"internal_rows": internal, "external_rows": external_rows(record),
                     "tcp_joint": canonical["joint_direction_stability"],
                     "ucla_joint": record["original_external_assessment"]["joint_direction_frequency"],
                     "system_narrative_available": False, "feedback": feedback,
                     "discussion": {"action": decision["action"],
                                    "text": clean_rationale(decision["discussion_application_zh"], aliases)},
                     "audit_note": audit_note}}


def build():
    raw_parent = PARENT.read_bytes()
    if hashlib.sha256(raw_parent).hexdigest() != PARENT_HASH:
        raise ValueError("Unexpected v5 pack; do not silently rebase.")
    parent = json.loads(raw_parent)
    selection = load_checked(SOURCES / "selection_v6.json", SELECTION_HASH)
    sources = {name: load_checked(SOURCES / name, expected) for name, expected in SOURCE_HASHES.items()}

    counts = sources["old_r1/candidate_index.json"]["counts"]["network_parcel_counts"]
    batches = {}
    for batch, prefix in (("old_r1", "old_r1"), ("r7_expansion", "r7")):
        records = {r["canonical_hypothesis_id"]: r for r in records_of(sources[f"{prefix}/candidate_index.json"])}
        proposals = {x["proposal_event_id"]: x for x in records_of(sources[f"{prefix}/proposal_slots.json"])}
        lineages = {x["proposal_event_id"]: x for x in records_of(sources[f"{prefix}/feedback_lineage.json"])}
        batches[batch] = (records, proposals, lineages)

    parent_cards = {card["id"]: card for card in parent["cards"]}
    parent_mapping = {entry["id"]: entry for entry in parent["organizer"]["mapping"]}
    for hypothesis_id, card_id in REUSE_FROM_V5.items():
        if parent_mapping[card_id]["hypothesis_id"] != hypothesis_id:
            raise ValueError(f"v5 mapping mismatch for {card_id}; verify reuse deliberately.")

    pack = copy.deepcopy(parent)
    pack.update({"pack_id": PACK_ID, "protocol_version": PROTOCOL_VERSION})
    pack["cards"], pack["organizer"]["mapping"] = [], []
    retained, built = [], 0
    for entry in selection["selected"]:
        hypothesis_id = entry["hypothesis_id"]
        card_id = f"packet-{entry['rank']:02d}"
        if hypothesis_id in REUSE_FROM_V5:
            old_id = REUSE_FROM_V5[hypothesis_id]
            card = copy.deepcopy(parent_cards[old_id])
            card["id"] = card_id
            private = copy.deepcopy(parent_mapping[old_id])
            private["id"] = card_id
            private["reused_from_v5_card_id"] = old_id
            retained.append({"v5_id": old_id, "v6_id": card_id, "hypothesis_id": hypothesis_id, "rank": entry["rank"]})
        else:
            records, proposals, lineages = batches[entry["batch"]]
            record = records[hypothesis_id]
            event = record["raw_proposal_event_ids"][0]
            proposal = proposals[event]
            lineage = lineages.get(event)
            card = make_card(record, proposal, lineage, card_id, counts)
            private = {"id": card_id, "hypothesis_id": hypothesis_id, "source_run": record["source_run"],
                       "source_evidence": {"candidate": record, "first_proposal": proposal,
                                           "feedback_lineage": [lineage] if lineage else []},
                       "human_value_ground_truth": None}
            built += 1
        pack["cards"].append(card)
        pack["organizer"]["mapping"].append(private)

    retired = [{"v5_id": old_id, "hypothesis_id": parent_mapping[old_id]["hypothesis_id"],
                "reason": "外部整理条件未通过，按 v6 固定选取规则离开面板；v5 包本身保留不变。"}
               for old_id in parent_cards if old_id not in REUSE_FROM_V5.values()]

    pack["public_meta"].update({
        "version_label": "v6 · 2026-09-18 · 单轮",
        "card_count": 34, "target_card_count": 34,
        "estimated_minutes": "每份 6 题；按分配表每位专家评审 10 份、共 60 题；可暂停，实际耗时待试评测定",
        "scope": "34 份已完成且外部验证通过的研究材料能力评分；按 10 名专家 × 每人 10 份 × 每份 3 人设计；尚未完成正式伦理与招募安排。",
        "selection_note": ("两批已完成发现活动外部整理条件全部通过的假设共 38 条，按最小方向标准化效应降序取前 34 条"
                           "（并列按联合方向频率），最弱 4 条未纳入；面板按外部效应强弱排序，不能用来估计自然发现成功率。")})
    pack["common_post"][0]["text"] = (
        "34 份材料全部来自已完成的发现臂与扩充臂及其两阶段外部分析。两批中外部整理条件全部通过的假设共 38 条，"
        "本面板按最小方向标准化效应降序取前 34 条（并列按联合方向频率），最弱 4 条未纳入；其中 3 份沿用上一版材料，"
        "31 份按相同版式新建。面板按外部效应强弱排序，不是代表性抽样，不能用来估计自然发现成功率。")
    pack["organizer"].update({
        "parent_v5_sha256": PARENT_HASH,
        "v6_amendment": {
            "selection_v6_sha256": SELECTION_HASH,
            "selection_rule_zh": selection["selection_rule_zh"],
            "study_design_basis": selection["study_design_basis"],
            "pool_size": selection["pool_size"],
            "retained_from_v5": retained,
            "retired_from_v5": retired,
            "newly_built_cards": built,
            "source_hashes": SOURCE_HASHES,
            "frontend_change": "反馈区块新增首轮提议（无父假设）展示分支；其余版式与 v5 一致。",
            "unchanged": "问题、选项、计分规则、共同说明其余区块及三份保留卡片的内容均与 v5 一致。",
        }})
    return pack


def main():
    pack = build()
    if TARGET.exists():
        if json.loads(TARGET.read_text(encoding="utf-8")) != pack:
            raise ValueError("v6 already exists with different content; create a new version.")
    else:
        with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(pack, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"pack_id": pack["pack_id"], "cards": len(pack["cards"]),
                      "external_rows": sum(len(c["post"]["external_rows"]) for c in pack["cards"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
