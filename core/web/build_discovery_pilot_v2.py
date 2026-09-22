"""Build ten real presentation cases from frozen metadata only; never run experiments."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from core.web.build_discovery_pilot import clean_rationale, PAPER_NOTES
from core.web.discovery_questionnaire_v2 import PROTOCOL_VERSION, QUESTIONS, SCORING

PACK_ID = "cs1-discovery-capabilities-20260915-v2"
SOURCE_HASHES = {
    "candidate_index.json": "a16ff8cc00d87dcf2c1d6eeafb2d0ad4ae3ea8e55f7d8f061d2f7ad309ce1b74",
    "proposal_slots.json": "0c84a047842f696eab92a484037edadebb55b0d7b93cb9f0f4f30c6356bf8239",
    "feedback_lineage.json": "3cdaf2a2fc6aeeb9d9b56fb2213a987034b77ee49acb9aa0f2418c373ca82468",
}
V1_HASH = "04a486728a51f43879c31079c2d7c5d2686159b1a89dc22e257a7bc15eac22a0"
NETWORKS = {"Vis": "视觉网络", "SomMot": "感觉运动网络", "DorsAttn": "背侧注意网络",
            "SalVentAttn": "显著性/腹侧注意网络", "Limbic": "边缘网络", "Cont": "控制网络", "Default": "默认网络"}


def load_checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source changed: {path.name}; build a separately reviewed new version.")
    return json.loads(raw)


def select_additions(candidates, proposals, lineages, existing):
    """Fixed hash order among complete first-occurrence traces; no effect/triage input."""
    endpoints = {c["endpoint"] for c in existing}
    groups = {(c["pre_external_metric_group"] or {}).get("group_anchor", c["endpoint"]) for c in existing}
    eligible = []
    for c in candidates:
        event = c["raw_proposal_event_ids"][0]
        proposal, lineage = proposals[event], lineages.get(event)
        if (c["complete_internal_external_empirical_package"] and c["external_status"] == "EXTERNAL_COMPLETE"
                and proposal["selection_locked_before_TCP"] and proposal["round_boundary_hashes_verified"]
                and len(proposal["six_model_reviews"]) == 6 and lineage and lineage["same_feedback_verified"]):
            eligible.append(c)
    eligible.sort(key=lambda c: hashlib.sha256(("cs1-expert-v2-coverage:" + c["canonical_hypothesis_id"]).encode()).hexdigest())
    selected = []
    for c in eligible:
        group = (c["pre_external_metric_group"] or {}).get("group_anchor", c["endpoint"])
        if c["endpoint"] in endpoints or group in groups:
            continue
        selected.append(c)
        endpoints.add(c["endpoint"])
        groups.add(group)
        if len(selected) == 7:
            break
    if len(selected) != 7:
        raise ValueError("Not enough complete, distinct-group records; do not manufacture cases.")
    return selected, len(eligible)


def describe(c, counts):
    spec = c["canonical_record"]["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        edges = counts[a] * counts[b] if a != b else counts[a] * (counts[a] - 1) // 2
        title = f"{NETWORKS[a]}与{NETWORKS[b]}" if a != b else f"{NETWORKS[a]}内部连接"
        definition = f"Schaefer-400 / 7-network；{a}（{counts[a]} 个分区）与 {b}（{counts[b]} 个分区），共 {edges:,} 条唯一非对角 Pearson r 边的算术均值。"
    else:
        if spec["kind"] != "ROI_to_network":
            raise ValueError("Unrecognised measurement definition.")
        roi = c["source_ROI_label"]
        net = c["endpoint"].split(":")[-1]
        edges = counts[net] - (1 if roi["network"] == net else 0)
        title = f"{'左' if '_LH_' in roi['name'] else '右'}侧 ROI{roi['roi']:03} 与{NETWORKS[net]}"
        definition = f"Schaefer-400 / 7-network；ROI{roi['roi']:03} = {roi['name']}。该分区到 {net} 的 {edges} 条非对角 Pearson r 边的算术均值；不等同于整个同名解剖区。"
    return title, definition + "不作 Fisher 变换或结果驱动选边。"


def make_card(c, proposal, lineage, card_id, counts):
    canonical = c["canonical_record"]
    final = proposal["final_decision"]["final_card"]
    assert proposal["canonical_hypothesis_id"] == c["canonical_hypothesis_id"] == lineage["child_hypothesis_id"]
    assert proposal["committed_card"]["hypothesis_id"] == c["canonical_hypothesis_id"]
    assert proposal["committed_card"]["endpoint"] == c["endpoint"]
    assert final["named"] == canonical["named"] and final["direction"] == canonical["direction"]
    refs, aliases = [], {}
    evidence_by_id = {entry["record"]["id"]: entry["record"] for entry in c["first_occurrence_cited_KG_records"] if entry["present"]}
    assert len(canonical["evidence_ids"]) == len(canonical["model_evidence_ids"])
    for i, (eid, model_id) in enumerate(zip(canonical["evidence_ids"], canonical["model_evidence_ids"]), 1):
        record, rid = evidence_by_id[eid], f"P{i}"
        paper = record["source_paper"]
        aliases[model_id] = rid
        refs.append({"id": rid, "title": paper.get("title", ""), "year": paper.get("year"),
                     "journal": paper.get("journal", ""), "pmid": str(paper.get("pmid", "")),
                     "url": "https://doi.org/" + paper["doi"] if paper.get("doi") else "",
                     "recorded_sentence": record.get("sentence", ""),
                     "note": PAPER_NOTES.get(str(paper.get("pmid", "")), "这是历史记录，不代表本版已核实其与精确候选的直接匹配；请结合原文判断。"),
                     "verification": "历史 KG 记录；未在本版独立核验原文，年份保留原记录。"})
    internal = [{"domain": name, "cases": {"ADHD": 13, "bipolar": 26, "psychosis_SZ_SZA": 16}[name],
                 "controls": 91, **result} for name, result in canonical["components"].items()]
    external = [{"row_id": f"E{i}", **{key: entry["record"][key] for key in [
        "cohort", "domain", "variant", "role", "cases", "controls", "beta", "standardized_beta",
        "residual_RMS", "columns", "bootstrap_beta", "bootstrap_standardized_beta"]}}
        for i, entry in enumerate(c["external_component_rows"], 1)]
    # All parents are preserved in the private source. The displayed parent must
    # be the earliest matching same-seed occurrence that was in the real context.
    parent = lineage["parent_prior_same_seed_TCP_records"][0]
    title, definition = describe(c, counts)
    return {"id": card_id,
            "pre": {"title": title, "hypothesis": canonical["hypothesis_zh"], "definition": definition,
                    "rationale": clean_rationale(final["rationale_zh"], aliases),
                    "transfer_limit": clean_rationale(final["evidence_transfer_limit_zh"], aliases), "references": refs,
                    "source_note": "假设为检验前执行卡的程序渲染标题；理由和限制来自实际模型回答，仅规范化引文别名。"},
            "post": {"internal_rows": internal, "external_rows": external,
                     "tcp_joint": canonical["joint_direction_stability"],
                     "ucla_joint": c["original_external_assessment"]["joint_direction_frequency"],
                     "system_narrative_available": False,
                     "feedback": {"parent_endpoint": parent["endpoint"], "parent_named": parent["named"],
                                  "parent_components": parent["feedback"]["components"],
                                  "parent_joint": parent["feedback"]["joint_direction_stability"],
                                  "action_text": clean_rationale(re.sub(r"[a-f0-9]{20}", "[父假设]", final["feedback_action_zh"]), aliases),
                                  "binding_verified": lineage["same_feedback_verified"]},
                     "discussion": {"action": proposal["final_decision"]["action"],
                                    "text": clean_rationale(proposal["final_decision"]["discussion_application_zh"], aliases)},
                     "audit_note": "完整原始提议、六次模型评议、预检验锁定、真实父反馈和外部行均保留来源绑定。此页是汇总展示，不是重新运行实验。"}}


def build(source_dir, v1_path):
    old = load_checked(v1_path, V1_HASH)
    source = {name: load_checked(source_dir / name, expected) for name, expected in SOURCE_HASHES.items()}
    index = source["candidate_index.json"]
    candidates = index["records"]
    lookup = {c["canonical_hypothesis_id"]: c for c in candidates}
    proposals = {x["proposal_event_id"]: x for x in source["proposal_slots.json"]["records"]}
    lineages = {x["proposal_event_id"]: x for x in source["feedback_lineage.json"]["records"]}
    existing = [lookup[x["hypothesis_id"]] for x in old["organizer"]["mapping"]]
    additions, eligible_count = select_additions(candidates, proposals, lineages, existing)
    pack = copy.deepcopy(old)
    pack.update({"schema_version": 2, "pack_id": PACK_ID, "protocol_version": PROTOCOL_VERSION,
                 "questions": QUESTIONS, "scoring": SCORING, "issues": []})
    for c in additions:
        event = c["raw_proposal_event_ids"][0]
        card_id = f"packet-{len(pack['cards']) + 1:02}"
        pack["cards"].append(make_card(c, proposals[event], lineages[event], card_id, index["counts"]["network_parcel_counts"]))
        pack["organizer"]["mapping"].append({"id": card_id, "hypothesis_id": c["canonical_hypothesis_id"],
            "source_run": c["source_run"], "source_evidence": {"candidate": c, "first_proposal": proposals[event], "feedback_lineage": [lineages[event]]},
            "human_value_ground_truth": None})
    # Do not prime scientific-value ratings with a curator's evaluative verdict.
    # Historical v1 summaries remain in the immutable v1 pack.
    for card, private in zip(pack["cards"], pack["organizer"]["mapping"]):
        model_ids = private["source_evidence"]["candidate"]["canonical_record"]["model_evidence_ids"]
        aliases = {eid: f"P{i}" for i, eid in enumerate(model_ids, 1)}
        card["post"]["feedback"]["action_text"] = clean_rationale(card["post"]["feedback"]["action_text"], aliases)
        card["post"].pop("curator_summary", None)
        card["post"].get("trace_labels", {}).pop("summary", None)
    pack["public_meta"].update({"version_label": "v2 · 2026-09-15", "card_count": 10, "target_card_count": 10,
        "estimated_minutes": "每份 6 题，共 60 题；可暂停，实际耗时待试评测定",
        "scope": "十份已完成研究材料的能力评分试用；尚未完成正式抽样、伦理与招募安排。",
        "selection_note": "十份材料按既定示例及固定覆盖规则选取，不能用来估计自然发现成功率。",
        "show_curator_summary": False})
    pack["common_post"][0]["text"] = (
        "十份材料全部来自已完成的发现臂及其两阶段外部分析，未包含在途扩充。原三份示例保留；追加七份从完整记录中按固定规则选取，避免重复指标及同一既定高相关组。追加排序不使用外部支持强弱。原三份有目的选取，整个十份面板仍不是代表性抽样。")
    pack["common_post"][-1]["text"] = (
        "外部分析后没有新增模型总结原文。本版只显示实际提议、反馈、模型讨论与实验结果，不展示整理者的评价性结论。这里的模型评议不是人类专家判断。")
    pack["organizer"].update({"source_hashes_v2": SOURCE_HASHES, "parent_v1_sha256": V1_HASH,
        "selection_rule": "Keep three v1 examples. Among complete first-occurrence proposals with six reviews, pre-test locks and verified real feedback, sort SHA256('cs1-expert-v2-coverage:' + canonical_id), skip existing endpoint/group_anchor, take seven. No external effect or triage value used to order/select additions.",
        "eligible_first_occurrence_feedback_packages": eligible_count,
        "sampling": "Purposive three-example base plus seven fixed hash/coverage additions. Not representative, independent, or outcome-blind sampling of the entire pipeline.",
        "profile_change": "Prior participation/exposure and usage questions removed; unmeasured exposure is not recoded as no.",
        "questionnaire_status": "Six anchored 1–5 capability ratings; descriptive equal-weight index, not a validated psychometric scale."})
    return pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = build(args.source, args.v1)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(pack, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"pack_id": pack["pack_id"], "cards": len(pack["cards"]), "questions_per_case": len(QUESTIONS),
                      "external_rows": sum(len(c["post"]["external_rows"]) for c in pack["cards"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
