"""Version the five-item questionnaire and source-bound runtime descriptions."""
import copy
import hashlib
import json

from core.web.build_discovery_pilot_v15 import HERE, ROOT, save_new, serialize

PARENT = HERE / "cs1_discovery_pilot_v15.json"
PARENT_SHA = "32a6ff9280ccd22317190156756aaf14524ea64cb4303c1f334e3df945b58858"
PACK_ID = "neurodiscovery-multitopic-20260922-v16"
REVISION = "five-capabilities-runtime-process-v1"
RUN = ROOT / ".codex_tmp/supplemental_two_case_full_nd_20260914_v1"


def bilingual(zh, en):
    return {"zh": zh, "en": en}


GROUNDING = [
    bilingual("证据与反馈整合", "Evidence and feedback integration"),
    bilingual("系统如何利用既有证据及此前的真实实验反馈提出或改进假设？", "How did the system use existing evidence and actual earlier experimental feedback to formulate or refine hypotheses?"),
    bilingual("结合文献依据与已记录的实验反馈，评价它们如何转化为当前或后轮假设；没有前序实验时，评价已有证据的利用。", "Assess how literature evidence and documented experimental feedback inform the current or later-round hypotheses; where no earlier experiment exists, assess the use of existing evidence."),
    bilingual("主要依据不相关，或明显误读证据与反馈", "The main basis is irrelevant, or evidence and feedback are clearly misinterpreted"),
    bilingual("提及相关证据或反馈，但未支撑关键推断", "Relevant evidence or feedback is mentioned but does not support the key inference"),
    bilingual("据证据或反馈作出相关推断或调整，但关键连接仍需解释", "Evidence or feedback informs a relevant inference or change, but key connections need explanation"),
    bilingual("证据、反馈与假设衔接清楚，调整理由可追溯", "Evidence, feedback and hypotheses are clearly connected, with traceable reasons for changes"),
    bilingual("充分整合已有证据与可用反馈，形成依据清晰、有针对性的假设", "Existing evidence and available feedback are thoroughly integrated into well-grounded, targeted hypotheses"),
]
EXECUTION = [
    bilingual("实验流程完成", "Experimental process completion"),
    bilingual("系统是否成功完成了实验完整过程，包括数据预处理、模型选择、实验记录和分析", "Did the system successfully complete the entire experimental process, including data preprocessing, model selection, experiment recording, and analysis?"),
    bilingual("根据展示的实际执行过程和记录，评价四个环节的完成情况及衔接；模型选择包括按任务配置分析方法。", "Use the actual execution process and records to assess completion and continuity across the four stages; model selection includes configuring analysis methods for the task."),
    bilingual("关键环节未执行，未形成可用的完整实验过程", "Key stages were not executed, leaving no usable complete experimental process"),
    bilingual("仅完成部分环节，关键处理、模型、记录或分析缺失", "Only some stages were completed; key processing, models, records or analyses are missing"),
    bilingual("主要环节已执行，但存在未完成步骤或记录缺口", "The main stages were executed, but some steps or records remain incomplete"),
    bilingual("四个环节均已完成，主要输入、配置和输出可追溯", "All four stages were completed, with traceable main inputs, configurations and outputs"),
    bilingual("四个环节完整衔接，处理与模型配置、执行记录及结果分析均清楚可核查", "All four stages form a complete process, with clearly verifiable processing, model configurations, execution records and analyses"),
]
LABELS = [bilingual("数据预处理", "Data preprocessing"), bilingual("模型选择", "Model selection"),
          bilingual("实验记录", "Experiment recording"), bilingual("分析", "Analysis")]

CONNECTIVITY = [
    bilingual("NeuroRuntime 使用已有的静息态功能连接数据，按本例锁定的脑区或网络端点提取平均 Pearson r，并与诊断分组和协变量对应。这里记录的是分析输入准备；现有执行材料没有显示重新进行原始 MRI 去噪、配准等预处理。", "NeuroRuntime uses existing resting-state functional-connectivity data, extracts mean Pearson r for the region or network endpoint locked for this case, and aligns it with diagnostic groups and covariates. This documents analysis-input preparation; the execution materials do not show a new raw-MRI denoising or registration run."),
    bilingual("按预先设定的连接分析方案采用 OLS，分别比较各病例组与对照，调整年龄、性别和站点；外部结果同时保留运动调整等敏感性配置。这是任务对应的模型配置，不是比较多种算法后选出最优模型。", "The configured connectivity analysis uses OLS for each case group versus controls, adjusting for age, sex and site; external results retain sensitivity configurations including motion adjustment. This is task-specific model configuration, not selection of a best algorithm after a model competition."),
    bilingual("已保存假设 ID、端点、seed/轮次、内部反馈及外部逐行结果；结果行包含队列、模型列、β、残差 RMS 和重采样方向频率，可按假设及来源记录对应回查。", "Saved records include hypothesis IDs, endpoints, seeds/rounds, internal feedback and external result rows. Rows retain cohort, model columns, beta, residual RMS and resampled direction frequencies, linked back to the hypothesis and source records."),
    bilingual("内部用 512 次、外部用 2,000 次重采样检查方向稳定性，并报告标准化效应及各队列结果。方向频率不是 P 值；下表和逐表分析保留具体效应与不同队列间的差异。", "Internal analyses use 512 resamples and external analyses 2,000 to assess directional stability, alongside standardized effects and cohort-specific results. Direction frequency is not a P value; the tables and accompanying analyses retain the effects and differences between cohorts."),
]
ADNI_PREPROCESS = bilingual("NeuroRuntime 从已整理的 ADNI 遗传与访视表出发，按受试者合并最早访视，转换数值字段，并用颅内容积归一化脑区体积；ADNI1/GO/2 与 ADNI3 分别用于内部和外部分析。这是表格级处理，使用既有影像测量，不是重新处理原始 MRI。", "NeuroRuntime starts from prepared ADNI genetics and visit tables, joins each participant's earliest visit, converts numeric fields and normalizes regional volumes by intracranial volume. ADNI1/GO/2 and ADNI3 supply internal and external analyses respectively. This is table-level processing of existing imaging measurements, not a new raw-MRI preprocessing run.")
RECORDING = bilingual("逐配置记录包含假设与配置 ID、seed、内部/外部阶段、开始与结束时间、完成状态、主结果、交叉验证结果、警告及失败字段；configuration_ledger 汇集配置执行记录，final_readout 汇总结果。本例的三个配置在内部及外部均记录为 complete。", "Per-configuration records contain hypothesis and configuration IDs, seed, internal/external phase, start and finish times, completion status, primary and cross-validation results, warnings and failure fields. The configuration_ledger collects execution records and final_readout summarizes results. All three configurations for this case are recorded as complete in both internal and external analyses.")
GENETICS = [
    bilingual(ADNI_PREPROCESS["zh"] + " 对每个测量对排除非有限值；协变量进行缺失值填补、缩放和站点编码，交叉验证时这些处理仅在训练折拟合。", ADNI_PREPROCESS["en"] + " Non-finite measurement pairs are excluded; covariates undergo missing-value imputation, scaling and site encoding, fitted only on the training fold during cross-validation."),
    bilingual("对入选测量对按登记方案运行线性偏相关、秩偏相关和 Huber 稳健回归三种配置，调整年龄、性别、教育、遗传主成分和站点。三种结果全部保留，不按显著性择优报告。", "Each selected measurement pair receives the registered linear partial-correlation, rank partial-correlation and robust Huber configurations, adjusting for age, sex, education, ancestry principal components and site. All three are retained rather than selecting the most significant result."),
    RECORDING,
    bilingual("分别报告内部与外部的样本量、效应、P 值及按任务/seed 校正的 FDR q 值；线性与秩方法报告 r，Huber 报告标准化 β。内部另有五折交叉验证记录，未通过的配置与队列差异仍显示在结果中。", "Internal and external analyses report sample size, effect, P value and task/seed-level FDR q value. Linear and rank methods report r; Huber reports standardized beta. Internal five-fold cross-validation is also recorded; non-passing configurations and cohort differences remain visible."),
]
PROGNOSIS = [
    bilingual(ADNI_PREPROCESS["zh"] + " 在基线 MCI 人群中根据后续访视构造进展事件与随访时间，并在各分析窗口截尾；排除无有效随访时间或缺少本例体积测量的记录，对协变量填补并标准化。", ADNI_PREPROCESS["en"] + " Subsequent visits define progression events and follow-up time for baseline MCI participants, censored at each analysis horizon. Records without valid follow-up time or the case's volume measurement are excluded; covariates are imputed and standardized."),
    bilingual("按预后任务配置调整 Cox 模型，分别运行 2、3、5 年窗口，调整年龄、性别、教育和 APOE ε4 剂量。这是预定窗口的同类模型检验，不是用外部结果挑选算法。", "The prognosis task configures adjusted Cox models for the registered 2-, 3- and 5-year horizons, adjusting for age, sex, education and APOE epsilon-4 dosage. These are tests of the same model family across prespecified horizons, not algorithm selection using external results."),
    RECORDING,
    bilingual("报告各窗口的样本量、事件数、每标准差体积对应的 HR、95% 置信区间及 FDR q 值；执行记录还保留内部五折检查和 499 次 bootstrap 配置。结果分析同时核对方向、校正结果和事件数，不将模型完成或单项显著等同于全部支持条件通过。", "Each horizon reports sample size, event count, HR per standard-deviation volume, 95% confidence interval and FDR q value; execution records also retain internal five-fold checks and the 499-bootstrap configuration. Analysis considers direction, multiplicity correction and event counts, rather than equating model completion or one significant statistic with passing all support criteria."),
]


def build():
    raw_parent = PARENT.read_bytes()
    if hashlib.sha256(raw_parent).hexdigest() != PARENT_SHA:
        raise ValueError("Frozen parent changed")
    pack = copy.deepcopy(json.loads(raw_parent))
    catalog = json.loads((HERE / "cs1_discovery_en_v14.json").read_bytes())
    translations = catalog["strings"]
    pack["questions"] = [question for question in pack["questions"] if question["id"] != "feedback"]
    for question in pack["questions"]:
        entries = {"grounding": GROUNDING, "validation": EXECUTION}.get(question["id"])
        if entries:
            for key, entry in zip(("capability", "title", "help"), entries[:3]):
                question[key] = entry["zh"]
                translations[entry["zh"]] = entry["en"]
            for number, entry in enumerate(entries[3:], 1):
                label = f"{number} 分 · {entry['zh']}"
                question["options"][number - 1]["label"] = label
                translations[label] = f"{number} · {entry['en']}"
    item_ids = [question["id"] for question in pack["questions"]]
    pack["scoring"].update(version=REVISION, item_ids=item_ids,
        case_composite="Arithmetic mean of all five numeric ratings; otherwise null. Evidence and available prior feedback are assessed together.",
        applicable_items_by_card={card["id"]: list(item_ids) for card in pack["cards"]})
    pack["scoring"]["interpretation"] += " Do not pool with earlier six-item or experimental-rigor ratings; historical answers are not recoded."
    ledger_path = RUN / "paper_ready/configuration_ledger.json"
    ledger = json.loads(ledger_path.read_bytes())
    mapping = {entry.get("card_id", entry.get("id")): entry for entry in pack["organizer"]["mapping"]}
    audit = {}
    for card in pack["cards"]:
        entry = mapping[card["id"]]
        kind = entry.get("topic", "connectivity")
        paragraphs = {"imaging_genetics": GENETICS, "prognosis": PROGNOSIS, "connectivity": CONNECTIVITY}[kind]
        if kind != "connectivity":
            records = [row for row in ledger if row["configuration"]["hypothesis_id"] == entry["hypothesis_id"]
                       and row["configuration"]["seed"] == entry["seed"]]
            if len(records) != 3 or any(row[phase]["status"] != "complete" for row in records for phase in ("internal", "external")):
                raise ValueError(f"Execution status needs a case-specific description: {card['id']}")
            audit[card["id"]] = [{"configuration": row["configuration"], **{
                phase: {key: row[phase][key] for key in ("status", "started_at", "finished_at", "failure", "warning_categories")}
                for phase in ("internal", "external")}} for row in records]
        else:
            audit[card["id"]] = {"source": "parent organizer.mapping.source_evidence", "hypothesis_id": entry["hypothesis_id"]}
        card["pre"]["reading"]["runtime_process"] = [
            {"title": copy.deepcopy(title), "text": copy.deepcopy(text)} for title, text in zip(LABELS, paragraphs)]
    sources = [ledger_path, RUN / "src/statistics_runner.py", ROOT / "core/scripts/run_adni_closed_loop_experiments.py"]
    pack["organizer"]["runtime_process"] = {"parent_sha256": PARENT_SHA, "cases": audit,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v16")
    pack["public_meta"].update(version_label="v16 · 2026-09-22 · 多主题", questionnaire_revision=REVISION, runtime_process_revision="record-bound-runtime-process-v1")
    translations[pack["public_meta"]["version_label"]] = "v16 · 2026-09-22 · Multiple topics"
    raw = serialize(pack)
    binding = {"pack_id": PACK_ID, "pack_sha256": hashlib.sha256(raw).hexdigest()}
    outputs = {"cs1_discovery_pilot_v16.json": raw}
    for kind, old, new in (("assignments", 11, 12), ("reference_notes", 10, 11), ("significance", 10, 11)):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v{old}.json").read_bytes())
        sidecar.update(binding)
        outputs[f"cs1_discovery_{kind}_v{new}.json"] = serialize(sidecar)
    catalog.update(version="discovery-en-v15", source_pack="cs1_discovery_pilot_v16.json", source_sha256=binding["pack_sha256"])
    outputs["cs1_discovery_en_v15.json"] = serialize(catalog)
    return outputs


if __name__ == "__main__":
    for name, raw in build().items():
        save_new(HERE / name, raw)
        print(name, hashlib.sha256(raw).hexdigest())
