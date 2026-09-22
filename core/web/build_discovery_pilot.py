"""Compile the three handoff examples; no model, NAS, experiment or re-fit access.

This creates a new presentation pack. Organiser records never go to a static URL.
The source files remain untouched; existing outputs cannot be overwritten.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from core.web.discovery_questionnaire import PROTOCOL_VERSION, QUESTIONS, ISSUES

PACK_ID = "cs1-discovery-pilot-20260915-v1"
SOURCE_HASH = "d340a32aeba222cc5bcf21d58f013c7d94800864f2fdea4540ba8bf5d9a28a9b"
TITLES = ["左侧 ROI108 与感觉运动网络", "左侧 ROI098 与感觉运动网络", "边缘网络与感觉运动网络"]
DEFINITIONS = [
    "Schaefer-400 / 7-network；ROI108 = 7Networks_LH_SalVentAttn_Med_2。该分区到全部 77 个 SomMot 分区的 77 条非对角 Pearson r 边的算术均值；不是整个旁中央小叶或整个岛叶。",
    "Schaefer-400 / 7-network；ROI098 = 7Networks_LH_SalVentAttn_FrOperIns_2。该分区到全部 77 个 SomMot 分区的 77 条非对角 Pearson r 边的算术均值；不是整个解剖岛叶。",
    "Schaefer-400 / 7-network；Limbic 26 个分区、SomMot 77 个分区之间 2,002 条唯一非对角 Pearson r 边的算术均值；不是单个边缘系统脑区。",
]
SUMMARIES = [
    "该固定 ROI—网络测量在 UCLA 的双相和 SCHZ 比较中均呈降低，并在 COBRE/HCP-EP 的精神病组成或推广中得到同向效应支持。它是有探索性外部支持的候选跨诊断关联；不能称双相在三个队列均被验证，也不建立因果、特异性或临床诊断价值。",
    "该 ROI—网络测量有探索性外部支持，但 UCLA 跨诊断联合方向频率接近既定门槛。精神病组成在另两队列同向。结果不足以声称首次发现岛叶—感觉运动连接异常，也不建立因果关系。",
    "精神病组成得到外部同向结果，但 UCLA 的 ADHD 效应较小，跨诊断联合稳定性有限，当前不足以支持较强的 ADHD—精神病共同表型。不能用精神病队列结果代替 ADHD 验证，也不能把缺乏支持称为显著证伪。",
]
PAPER_NOTES = {
    "35264921": "既有 BD-I 感觉运动连接/执行功能线索。所存原句是机制不清楚的背景，不是该精确 ROI 降低的直接结果；方向需原文核查。",
    "30837901": "既有 SZ/MDD/BD 跨诊断研究，说明跨诊断概念本身并非新提出。不能直接支持精确 parcel—全网络均值的方向。",
    "28360829": "视觉言语启动任务中的局部连接结果；任务与本次静息全网络均值不同。",
    "32688026": "既有双相岛叶—中央后回连接降低，也包含其他连接升高；人群、侧别、疾病阶段和测量不同，不能只保留有利方向。",
    "37502819": "儿童 ADHD 的 NIRS 局部连接，不是本次静息 fMRI 跨网均值；模态和人群存在距离。",
    "35634207": "SZ 正则化区域/体素连接，缺少本候选 Limbic 直接锚点；不是完整全文核查。",
}
COMMON_PRE = [
    {"title": "研究问题与信息条件", "text": "检验 ADHD、双相及 SZ/SZA 中具名诊断组合，是否在固定连接测量上分别相对对照呈同向改变。假设在该候选 TCP 测量前锁定，但形成过程中使用了更早同 seed 的 TCP 反馈。阶段 A 统一不展示父反馈数值或当前候选结果；相关记录在阶段 B 提供。因此本阶段不是完整原始生成上下文的再现。"},
    {"title": "人群与分析", "text": "TCP 基础样本 234：ADHD 13、双相 26、SZ/SZA 16，各比较使用预定对照 91，具名组可能重叠。使用既有预处理、修复及 QC 后 FC；OLS 调整年龄、性别、站点。连接是唯一非对角 Pearson r 边均值，不作 Fisher 变换或结果驱动选边。"},
    {"title": "外部设计与数据接触", "text": "UCLA：ADHD 40、双相 49、SCHZ 49、对照 119，主要模型调整年龄、性别、扫描仪/站点、mean FD，另保留不加 FD 敏感性。COBRE：SZ/SZA 74 vs 84，另做仅 SZ。HCP-EP：非情感性 90 vs 55；情感性仅为背景，不等同双相验证。研究团队此前接触过外部队列；跨队列身份零重叠尚未穷尽认证。ADHD200 未执行。"},
    {"title": "效应与不确定性", "text": "标准化效应为调整后病例系数 β / 残差 RMS，不是 Cohen d。TCP 512 次、外部 2,000 次共享站点分层参与者 bootstrap。报告描述性百分位范围和同向频率，未提供校准 P/q 值或确证置信区间。联合频率不是假设为真的概率。"},
    {"title": "证据边界", "text": "给出的文献是当时知识图谱的已有记录及整理者限制说明，尚未完成逐项最接近文献/全文新颖性核查。部分原句是背景或不同模态结果。可选择“材料不足”，不要求将未见报道等同于首次发现。所有评分无默认选项。"},
]
COMMON_POST = [
    {"title": "当前来源范围", "text": "本试用版取自已完成发现臂及其两阶段外部分析，未包含在途扩充。来源共有 104 个不同已检验假设、50 个指标，38 个假设/25 个指标完成外部分析；8 个合格诊断变体未做完整外部分析，其余 58 项未进入外部。这三份示例为材料试评而选，不代表自然成功率。"},
    {"title": "实际外部整理标准", "text": "探索性整理条件：UCLA 全部原具名主要效应同向；其中最小方向标准化效应 ≥0.15；联合方向频率 ≥0.8；不加 FD 敏感性均同向。没有本家族的确证性多重比较推断。方向一致、通过整理条件、正式确认是不同概念。"},
    {"title": "共享与缺口", "text": "内部 seed 共用 TCP，外部两阶段共用同一批 588 人；302 组成行对应 214 个唯一端点—设计估计，不能按表格行计独立发现。各队列预处理/头动信息并不统一，药物、病程等混杂没有统一控制。已有统计校准资格失败保留，本批只能按探索性效应复制解释。"},
    {"title": "结论归属", "text": "外部分析后没有新增模型总结回答。下面的自然语言摘要由材料整理者撰写；原始程序只提供结构化判定。不能将摘要评分解释为系统独立撰写结论的能力。内部模型评议角色也不是此次人类专家。"},
]


def clean_rationale(text, aliases):
    """Replace identity aliases only; retain an auditable original privately."""
    return re.sub(r"(?<![A-Za-z0-9])E\d{3}(?![A-Za-z0-9])", lambda m: aliases.get(m.group(0), "[已移除引文]"), text)


def build(source_dir):
    raw = (source_dir / "three_example_source_records.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError("Handoff source changed. Prepare a new reviewed material version; do not overwrite v1.")
    examples = json.loads(raw)["records"]
    index = json.loads((source_dir / "candidate_index.json").read_text(encoding="utf-8"))
    cards, private = [], []
    for i, ex in enumerate(examples):
        candidate, proposal = ex["candidate"], ex["first_proposal"]
        canonical = candidate["canonical_record"]
        assert candidate["external_status"] == "EXTERNAL_COMPLETE"
        final_card = proposal["final_decision"]["final_card"]
        references, aliases = [], {}
        for j, evidence in enumerate(candidate["first_occurrence_cited_KG_records"]):
            record = evidence["record"]
            paper = record["source_paper"]
            ref_id = f"P{j + 1}"
            if j < len(canonical["model_evidence_ids"]):
                aliases[canonical["model_evidence_ids"][j]] = ref_id
            references.append({
                "id": ref_id, "title": paper.get("title", ""), "year": paper.get("year"),
                "journal": paper.get("journal", ""), "pmid": str(paper.get("pmid", "")),
                "url": "https://doi.org/" + paper["doi"] if paper.get("doi") else "",
                "recorded_sentence": record.get("sentence", ""),
                "note": PAPER_NOTES.get(str(paper.get("pmid", "")), "未完成直接原文与新颖性核查。"),
                "verification": "历史 KG 记录；未在本版独立核验原文，年份保留原记录。",
            })
        internal = [{"domain": domain, "cases": {"ADHD": 13, "bipolar": 26, "psychosis_SZ_SZA": 16}[domain],
                     "controls": 91, **component} for domain, component in canonical["components"].items()]
        external = []
        for j, entry in enumerate(candidate["external_component_rows"]):
            r = entry["record"]
            external.append({"row_id": f"E{j + 1}", **{k: r[k] for k in [
                "cohort", "domain", "variant", "role", "cases", "controls", "beta", "standardized_beta",
                "residual_RMS", "columns", "bootstrap_beta", "bootstrap_standardized_beta"]}})
        lineage = ex["feedback_lineage"][0]
        parent = lineage["parent_prior_same_seed_TCP_records"][0]
        decision = proposal["final_decision"]
        card_id = f"packet-{i + 1:02}"
        cards.append({
            "id": card_id,
            "pre": {"title": TITLES[i], "hypothesis": canonical["hypothesis_zh"], "definition": DEFINITIONS[i],
                    "rationale": clean_rationale(final_card["rationale_zh"], aliases),
                    "transfer_limit": clean_rationale(final_card["evidence_transfer_limit_zh"], aliases),
                    "references": references,
                    "source_note": "假设是检验前执行卡的程序渲染标题；理由和限制来自实际模型回答（引文别名规范化）。"},
            "post": {"internal_rows": internal, "external_rows": external,
                     "tcp_joint": canonical["joint_direction_stability"],
                     "ucla_joint": candidate["original_external_assessment"]["joint_direction_frequency"],
                     "curator_summary": SUMMARIES[i], "system_narrative_available": False,
                     "feedback": {"parent_endpoint": parent["endpoint"], "parent_named": parent["named"],
                                  "parent_components": parent["feedback"]["components"],
                                  "parent_joint": parent["feedback"]["joint_direction_stability"],
                                  "action_text": re.sub(r"[a-f0-9]{20}", "[父假设]", final_card["feedback_action_zh"]),
                                  "binding_verified": lineage["same_feedback_verified"]},
                     "discussion": {"action": decision["action"],
                                    "text": clean_rationale(decision["discussion_application_zh"], aliases)},
                     "audit_note": "交接包记录了检验前锁定、六次模型评议、真实父反馈与内部/外部结果绑定。此界面展示汇总记录，不宣称评审者已经重新运行分析。",
                     "trace_labels": {"internal": "R1 · 固定 canonical 内部结果", "external": "R2 · 完整外部组成与敏感性", "feedback": "R3 · 已核对的父反馈到后轮提议", "summary": "R4 · 整理者摘要"}},
        })
        private.append({"id": card_id, "original_material_id": ex["material_id"],
                        "hypothesis_id": candidate["canonical_hypothesis_id"], "source_run": candidate["source_run"],
                        "source_record_pointer": f"/records/{i}", "source_evidence": ex,
                        "human_value_ground_truth": None})
    return {
        "schema_version": 1, "pack_id": PACK_ID, "protocol_version": PROTOCOL_VERSION,
        "status": "pilot_only", "questions": QUESTIONS, "issues": ISSUES,
        "public_meta": {"title": "Case Study 1 · 完整科研成果评审", "version_label": "v1 · 2026-09-15",
                        "card_count": len(cards), "estimated_minutes": "约 35–50 分钟，可暂停",
                        "scope": "三份完整材料试用；尚未冻结正式抽样、伦理与招募安排。不是正式人类评价结果。",
                        "blinding": "隐藏方法、提供方和内部评分；仅做单来源成果评审，无完整对照。",
                        "novelty_caveat": "既往文献与原文核查仍有缺口；允许无法判断，不预设新发现。"},
        "common_pre": COMMON_PRE, "common_post": COMMON_POST, "cards": cards,
        "organizer": {"source_hash": digest, "source_dir": str(source_dir), "mapping": private,
                      "sampling": "Three purposive handoff examples, not representative sampling; no success-rate estimation.",
                      "inventory_sha256": hashlib.sha256((source_dir / "candidate_index.json").read_bytes()).hexdigest(),
                      "source_inventory_top_level_keys": list(index) if isinstance(index, dict) else [],
                      "questionnaire_status": "Unvalidated pilot rubric; no ethics approval or novelty certification asserted."},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"created": str(args.output), "pack_id": result["pack_id"], "cards": len(result["cards"]), "external_rows": sum(len(c["post"]["external_rows"]) for c in result["cards"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
