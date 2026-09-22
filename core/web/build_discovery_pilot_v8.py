"""Build a multi-topic expert panel from completed, source-matched experiments.

This is a display/assignment build only. Historical packs and experiments are
never modified. Repeated proposals for the same predictor/outcome pair use the
earliest recorded seed, round and hypothesis ID, independent of scores/results.
"""
import argparse
import copy
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials"
PACK_ID = "neurodiscovery-multitopic-20260920-v8"
SOURCE_NAME = "NeuroDiscovery_Eight_Topic_Records_EN_20260920.json"
SOURCE_SHA = "d6f53824c6ec492f812617b4292d7e0cfe737e0bf343ec1d7d62dd281e72010d"
V7_SHA = "787094275b59cc037dc5633d1bd5cb6695ce9666ab9851085c49c59a0c528568"
RETAIN = [1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 17, 22, 24, 27, 28, 30, 31, 32, 33, 34]
NAMES = {
    "apoe_e4_dosage": ("APOE ε4 等位基因剂量", "APOE ε4 allele dosage"),
    "ad_prs_p1em03_avg": ("阿尔茨海默病多基因风险评分", "Alzheimer's disease polygenic risk score"),
    "pathway_prs__curated__ad_risk_gwas__p5em02": ("AD 风险 GWAS 基因集多基因评分", "AD-risk GWAS gene-set polygenic score"),
    "pathway_prs__curated__mendelian_ad__p5em02": ("孟德尔型 AD 基因集多基因评分", "Mendelian-AD gene-set polygenic score"),
    "Hippocampus_icv": ("海马体积", "hippocampal volume"),
    "Entorhinal_icv": ("内嗅皮层体积", "entorhinal volume"),
    "MidTemp_icv": ("颞中回体积", "middle temporal gyrus volume"),
    "Ventricles_icv": ("脑室体积", "ventricular volume"),
    "WholeBrain_icv": ("全脑体积", "whole-brain volume"),
}
RATIONALES = {
    ("imaging_genetics", "ad_prs_p1em03_avg", "Hippocampus_icv"): (
        "既有研究将 AD 多基因风险与较小的海马体积联系起来。系统据此提出，用可用的遗传风险评分检验基线海马体积的差异。",
        "Prior research linked AD polygenic risk to smaller hippocampal volume. The system proposed testing baseline hippocampal volume against the available genetic risk score."),
    ("imaging_genetics", "apoe_e4_dosage", "Entorhinal_icv"): (
        "既有研究报告 APOE ε4 携带者在进展至痴呆后内嗅皮层萎缩更快。系统将这一线索转化为 ε4 剂量与基线内嗅皮层体积之间的关联假设。",
        "Prior research reported faster entorhinal atrophy in APOE ε4 carriers after dementia transition. The system translated this lead into an association between ε4 dosage and baseline entorhinal volume."),
    ("imaging_genetics", "apoe_e4_dosage", "Hippocampus_icv"): (
        "既有队列研究报告 APOE ε4 纯合子具有较小的海马。系统进一步以 ε4 剂量作为连续变量，检验其与基线海马体积的关系。",
        "A previous cohort study reported smaller hippocampi in APOE ε4 homozygotes. The system tested ε4 dosage as a continuous predictor of baseline hippocampal volume."),
    ("imaging_genetics", "apoe_e4_dosage", "MidTemp_icv"): (
        "既有研究将 APOE ε4 与更快的颞中回萎缩联系起来。系统据此检验 ε4 剂量是否与较小的基线颞中回体积相关。",
        "Prior research linked APOE ε4 to faster middle temporal atrophy. The system tested whether ε4 dosage was associated with smaller baseline middle temporal volume."),
    ("imaging_genetics", "apoe_e4_dosage", "Ventricles_icv"): (
        "既有研究观察到 APOE ε4 携带者的脑室体积增加更快。系统选择基线脑室体积，检验 ε4 剂量与脑室大小的横断面关联。",
        "Prior research observed faster ventricular expansion in APOE ε4 carriers. The system selected baseline ventricular volume to test the cross-sectional association between ε4 dosage and ventricular size."),
    ("imaging_genetics", "pathway_prs__curated__ad_risk_gwas__p5em02", "Hippocampus_icv"): (
        "既有影像遗传学研究报告多基因评分与海马体积相关。系统将这一思路扩展至 AD 风险 GWAS 基因集的多基因评分。",
        "Previous imaging-genetics research reported associations between polygenic scores and hippocampal volume. The system extended this approach to a polygenic score for the AD-risk GWAS gene set."),
    ("imaging_genetics", "pathway_prs__curated__mendelian_ad__p5em02", "Hippocampus_icv"): (
        "既有研究将较高的 AD 多基因风险与较小的海马及临床进展联系起来。系统据此提出孟德尔型 AD 基因集评分与海马体积的关联假设。",
        "Prior research linked higher AD polygenic risk to smaller hippocampi and clinical progression. The system proposed testing the Mendelian-AD gene-set score against hippocampal volume."),
}
PROGNOSIS_RATIONALES = {
    "ad_prs_p1em03_avg": ("既有研究显示 AD 多基因风险与纵向认知下降相关。系统进一步检验这一遗传风险评分能否解释 MCI 患者进展至痴呆的时间差异。", "Prior research linked AD polygenic risk to longitudinal cognitive decline. The system tested whether the genetic risk score explained differences in time to dementia among participants with MCI."),
    "Entorhinal_icv": ("既有随访研究显示基线内嗅皮层体积可预测 MCI 进展。系统据此检验经颅内容积归一化的内嗅皮层体积与痴呆发生时间的关系。", "A follow-up study found that baseline entorhinal volume predicted MCI progression. The system tested ICV-normalized entorhinal volume against time to dementia."),
    "Hippocampus_icv": ("既有形态学研究的荟萃分析提示内侧颞叶萎缩有助于预测 AD。系统用基线海马体积检验 MCI 患者后续进展风险。", "A meta-analysis of morphometric studies identified medial temporal atrophy as a predictor of AD. The system tested baseline hippocampal volume against subsequent progression risk in MCI."),
    "MidTemp_icv": ("既有纵向 MRI 研究将颞中回萎缩与 MCI 进展联系起来。系统据此检验基线颞中回体积与后续痴呆风险的关联。", "A longitudinal MRI study linked middle temporal atrophy to MCI progression. The system tested baseline middle temporal volume against subsequent dementia risk."),
    "pathway_prs__curated__mendelian_ad__p5em02": ("既有研究报告遗传变异与 MCI 进展速度有关。系统将遗传风险线索扩展为孟德尔型 AD 基因集多基因评分与痴呆发生时间的关联假设。", "Prior research associated genetic variation with MCI progression. The system extended this genetic-risk lead to a hypothesis linking a Mendelian-AD gene-set polygenic score to time to dementia."),
    "Ventricles_icv": ("既有研究显示脑室及脑脊液体积有助于预测认知下降。系统进一步检验基线脑室体积是否与 MCI 患者的痴呆发生时间相关。", "Prior research found that ventricular and CSF volumes helped predict cognitive decline. The system tested whether baseline ventricular volume was associated with time to dementia in MCI."),
    "WholeBrain_icv": ("既有研究将全脑体积纳入 MCI 进展的预测模型。系统据此检验基线全脑体积与不同随访时间窗内痴呆风险的关系。", "Prior research included whole-brain volume in models of MCI progression. The system tested baseline whole-brain volume against dementia risk over different follow-up windows."),
}


def serialize(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build(source_zip):
    parent_raw = (HERE / "cs1_discovery_pilot_v7.json").read_bytes()
    assert hashlib.sha256(parent_raw).hexdigest() == V7_SHA
    parent = json.loads(parent_raw)
    with zipfile.ZipFile(source_zip) as archive:
        raw = archive.read(SOURCE_NAME)
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    source = json.loads(raw)
    catalog = json.loads((HERE / "cs1_discovery_en_v4.json").read_bytes())
    translations = catalog["strings"]

    def t(zh, en):
        translations[zh] = en
        return zh

    pack = copy.deepcopy(parent)
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v8")
    pack["cards"] = [copy.deepcopy(parent["cards"][i - 1]) for i in RETAIN]
    pack["organizer"] = {
        "source_pack": "cs1_discovery_pilot_v7.json", "source_pack_sha256": V7_SHA,
        "source_export": SOURCE_NAME, "source_export_sha256": SOURCE_SHA,
        "selection": "20 retained connectivity cases plus all 14 distinct completed topic/predictor/outcome pairs. For repeated pairs, earliest seed/round/ID, without consulting scores or outcomes. No parent-feedback requirement.",
        "mapping": [copy.deepcopy(parent["organizer"]["mapping"][i - 1]) for i in RETAIN],
    }
    pack["public_meta"] = {
        "title": t("NeuroDiscovery · 完整科研成果评审", "NeuroDiscovery · Complete research output review"),
        "version_label": t("v8 · 2026-09-20 · 多主题", "v8 · 2026-09-20 · Multiple topics"),
        "card_count": 34, "target_card_count": 34, "review_flow": "single_round",
        "material_layout": "multitopic", "show_curator_summary": False,
        "estimated_minutes": t("每位专家评审 10 份，每份约 5 分钟，可保存与暂停。", "Each expert reviews 10 cases, about 5 minutes each; save and pause at any time."),
        "scope": t("跨诊断脑连接、影像遗传学与预后研究。", "Transdiagnostic connectivity, imaging genetics and prognosis."),
    }
    pack["common_pre"] = []
    pack["common_post"] = []
    help_text = {
        "grounding": ("评价已有证据如何支持提出这项假设。", "Assess how existing evidence motivates this hypothesis."),
        "novelty": ("结合给定文献与专业知识，评价假设的新颖性。", "Assess novelty using the supplied literature and your expertise."),
        "design": ("评价测量、样本、统计模型与验证安排是否适合该假设。", "Assess whether the measures, samples, statistical models and validation suit the hypothesis."),
        "validation": ("结合内部与外部实验，评价验证的严谨性与可信度。", "Assess the rigor and credibility of the internal and external experiments."),
        "value": ("评价完整研究结果提供的科学信息与研究价值。", "Assess the scientific information and research value provided by the complete results."),
        "feedback": ("评价系统如何根据前序实验改进当前假设。", "Assess how the system used earlier experiments to refine the current hypothesis."),
    }
    for q in pack["questions"]:
        q["help"] = t(*help_text[q["id"]])
    for card in pack["cards"]:
        card["pre"]["topic"] = t("跨诊断脑连接", "Transdiagnostic connectivity")
        card["pre"].pop("study_context", None)

    # All 14 measurement pairs are represented, not only externally passing ones.
    selected = {}
    for row in sorted(source["scored_and_tested"], key=lambda r: (r["seed"], r["round"], r["hypothesis_id"])):
        key = (row["topic"], *row["measurement_pair"])
        selected.setdefault(key, row)
    assert len(selected) == 14
    for number, (key, row) in enumerate(sorted(selected.items()), 35):
        topic, predictor, outcome = key
        zh, en = NAMES[predictor]
        is_ig = topic == "imaging_genetics"
        target_zh, target_en = NAMES[outcome] if is_ig else ("MCI 至痴呆的进展", "MCI-to-dementia progression")
        title = t(f"{zh}与{target_zh}", f"{en} and {target_en}")
        original = row["original_proposal"]
        if is_ig:
            if original["expected_direction"] == "negative":
                plain = f"{zh}越高，基线经颅内容积归一化的{target_zh}预计越小。"
            else:
                plain = f"{zh}与基线经颅内容积归一化的{target_zh}相关。"
            rationale = RATIONALES[key]
            definition = t(f"预测变量为 {predictor}；结局为基线 {outcome}，即{target_zh}除以颅内容积。",
                           f"Predictor: {predictor}. Outcome: baseline {outcome}, defined as {target_en} divided by intracranial volume.")
            methods = [
                {"title": t("数据集", "Datasets"), "text": t("内部分析使用 ADNI1/GO/2，外部分析使用 ADNI3；各模型的有效样本量见结果表。", "Internal analyses use ADNI1/GO/2 and external analyses use ADNI3; effective sample sizes are shown for each model.")},
                {"title": t("测量与统计模型", "Measures and statistical models"), "text": t("使用基线结构 MRI 体积与遗传数据，调整年龄、性别、教育、遗传祖源 PC1–PC5 和站点。分别进行线性关联、秩关联和 Huber 稳健回归。", "Baseline structural MRI volumes and genetic data are analyzed with adjustment for age, sex, education, ancestry PCs 1–5 and site, using linear association, rank association and robust Huber regression.")},
                {"title": t("结果指标", "Outcome measures"), "text": t("线性与秩关联报告偏相关系数 r；Huber 模型报告标准化回归系数 β。P 为双侧检验值，q 为同任务、同 seed 全部配置的 FDR 校正值。", "Linear and rank associations report partial correlation r; Huber models report standardized regression coefficient β. P values are two-sided; q values use FDR correction across all configurations within the same task and seed.")},
            ]
        else:
            if predictor.endswith("_icv") and original["expected_direction"] == "negative":
                plain = f"在轻度认知障碍（MCI）人群中，较大的基线{zh}预计与较低的后续痴呆风险相关。"
            elif predictor == "ad_prs_p1em03_avg":
                plain = f"在轻度认知障碍（MCI）人群中，较高的{zh}预计与更短的痴呆发生时间相关。"
            else:
                plain = f"在轻度认知障碍（MCI）人群中，基线{zh}与后续痴呆发生时间相关。"
            rationale = PROGNOSIS_RATIONALES[predictor]
            definition = t(f"预测变量为基线 {predictor}；结局为从 MCI 进展至痴呆的时间。影像体积均以颅内容积归一化。",
                           f"Predictor: baseline {predictor}. Outcome: time from MCI to dementia. Imaging volumes are normalized to intracranial volume.")
            methods = [
                {"title": t("数据集", "Datasets"), "text": t("内部分析使用 ADNI1/GO/2 的 MCI 参与者，外部分析使用 ADNI3 的 MCI 参与者；样本量及进展事件数见结果表。", "Internal analyses use participants with MCI in ADNI1/GO/2; external analyses use participants with MCI in ADNI3. Sample sizes and progression events are shown in the results.")},
                {"title": t("测量与统计模型", "Measures and statistical models"), "text": t("使用 Cox 比例风险模型，调整年龄、性别、教育和 APOE ε4 剂量；分别分析 2 年、3 年与 5 年随访窗口。", "Cox proportional hazards models adjust for age, sex, education and APOE ε4 dosage, with separate 2-, 3- and 5-year follow-up windows.")},
                {"title": t("结果指标", "Outcome measures"), "text": t("HR 表示预测变量每增加一个标准差的风险比，并报告 95% 置信区间。P 为双侧检验值，q 为同任务、同 seed 全部配置的 FDR 校正值。", "HR is the hazard ratio per standard-deviation increase in the predictor, with a 95% confidence interval. P values are two-sided; q values use FDR correction across all configurations within the same task and seed.")},
            ]
        refs = []
        for i, claim in enumerate(row["source_claim_records"], 1):
            paper = claim["source_paper"]
            refs.append({"id": f"P{i}", "title": paper["title"], "year": paper["year"],
                         "journal": paper.get("journal", ""), "pmid": paper.get("pmid", ""),
                         "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/" if paper.get("pmid") else "",
                         "recorded_sentence": claim["raw_text"]})
        results = []
        for cfg in row["configurations"]:
            results.append({"model": cfg["model"], "horizon_years": cfg["horizon_years"],
                            **{phase: copy.deepcopy(cfg[phase]) for phase in ("internal", "external")}})
        cid = f"packet-{number:02d}"
        pack["cards"].append({"id": cid, "pre": {
            "title": title, "topic": t("影像遗传学", "Imaging genetics") if is_ig else t("预后研究", "Prognosis"),
            "hypothesis": row["hypothesis"], "hypothesis_plain": t(plain, row["hypothesis"]),
            "definition": definition, "rationale": t(*rationale), "rationale_is_summary": True,
            "source_note": "", "references": refs, "methods": methods,
        }, "post": {"result_kind": topic, "experimental_results": results,
                    "feedback": {"no_parent": True}, "discussion": {"action": "", "text": ""}}})
        pack["organizer"]["mapping"].append({"card_id": cid, "topic": topic, "record_id": row["record_id"],
            "hypothesis_id": row["hypothesis_id"], "seed": row["seed"], "round": row["round"],
            "measurement_pair": row["measurement_pair"], "source_files": row["source_files"],
            "original_proposal": original, "novelty_status_original": row["novelty_status_original"]})

    scoring = pack["scoring"]
    scoring.update(version="applicable-capabilities-single-round-v4",
                   case_composite="Arithmetic mean of all applicable numeric ratings; otherwise null. Feedback is inapplicable where no previous experiment feedback is documented.",
                   session_composite="Mean of complete applicable-item case composites; report dimension counts and case coverage separately.")
    scoring["applicable_items_by_card"] = {
        c["id"]: [q for q in scoring["item_ids"] if q != "feedback" or bool(c["post"].get("feedback", {}).get("parent_endpoint"))]
        for c in pack["cards"]}
    pack_bytes = serialize(pack)
    pack_hash = hashlib.sha256(pack_bytes).hexdigest()
    source_cards = [f"packet-{i:02d}" for i in RETAIN]
    ig = [f"packet-{i:02d}" for i in range(35, 42)]
    prognosis = [f"packet-{i:02d}" for i in range(42, 49)]
    experts = {f"P{i+1:02d}": [source_cards[(6*i+j) % 20] for j in range(6)]
               + [ig[(2*i+j) % 7] for j in range(2)]
               + [prognosis[(2*i+j+3) % 7] for j in range(2)] for i in range(10)}
    binding = {"version": 2, "pack_id": PACK_ID, "pack_sha256": pack_hash}
    assignments = {**binding, "cards_per_expert": 10, "experts": experts,
                   "coverage": dict(Counter(c for dealt in experts.values() for c in dealt)),
                   "rule_zh": "每位专家评审 6 份跨诊断脑连接、2 份影像遗传学、2 份预后材料。按固定循环分配，每份材料由 2–3 位专家评审。"}
    notes = json.loads((HERE / "cs1_discovery_reference_notes_v2.json").read_bytes())
    sig = json.loads((HERE / "cs1_discovery_significance_v2.json").read_bytes())
    notes.update(binding)
    notes["notes"] = {cid: value for cid, value in notes["notes"].items() if cid in source_cards}
    notes["definition_plain"] = {cid: value for cid, value in notes.get("definition_plain", {}).items() if cid in source_cards}
    sig.update(binding)
    sig["significance"] = {cid: value for cid, value in sig["significance"].items() if cid in source_cards}
    for card in pack["cards"][20:]:
        cid = card["id"]
        notes["notes"][cid] = {ref["id"]: {"did_zh": "", "did_en": "", "relation_zh": "", "relation_en": ""} for ref in card["pre"]["references"]}
        ig_card = card["post"]["result_kind"] == "imaging_genetics"
        sig["significance"][cid] = {
            "zh": "这项研究检验遗传风险能否在脑结构中体现。" if ig_card else "这项研究检验基线特征能否提供后续疾病进展的信息。",
            "en": "This study tests whether genetic risk is reflected in brain structure." if ig_card else "This study tests whether baseline features inform subsequent disease progression."}
    notes = {**binding, "notes": notes["notes"], "definition_plain": notes["definition_plain"],
             "note_count": sum(len(card["pre"]["references"]) for card in pack["cards"])}
    sig = {**binding, "significance": sig["significance"]}
    catalog.update(version="discovery-en-v5", source_pack="cs1_discovery_pilot_v8.json", source_sha256=pack_hash)
    return {"cs1_discovery_pilot_v8.json": pack_bytes,
            "cs1_discovery_assignments_v3.json": serialize(assignments),
            "cs1_discovery_reference_notes_v3.json": serialize(notes),
            "cs1_discovery_significance_v3.json": serialize(sig),
            "cs1_discovery_en_v5.json": serialize(catalog)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, default=Path("C:/Users/45846/Desktop/NeuroDiscovery_Eight_Topic_Records_EN_20260920.zip"))
    parser.add_argument("--refresh-draft", action="store_true", help="Regenerate only this unreleased v8 draft and its new sidecars.")
    args = parser.parse_args()
    for name, raw in build(args.source_zip).items():
        path = HERE / name
        if path.exists() and path.read_bytes() != raw and not args.refresh_draft:
            raise ValueError(f"Refusing to overwrite changed version: {name}")
        path.write_bytes(raw)
        print(name, hashlib.sha256(raw).hexdigest())
