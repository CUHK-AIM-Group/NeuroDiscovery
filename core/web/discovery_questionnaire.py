"""Pilot rubric. These are human judgments, never scientific ground-truth labels."""

PROTOCOL_VERSION = "cs1-complete-output-expert-pilot-v1"
UNABLE = [
    {"value": "insufficient", "label": "材料不足，无法判断"},
    {"value": "outside", "label": "超出本人专业范围"},
]


def question(key, stage, title, labels, help_text="", rating=False):
    return {
        "id": key, "stage": stage, "title": title, "help": help_text,
        "rating": rating,
        "options": [{"value": str(i + 1), "label": text} for i, text in enumerate(labels)] + UNABLE,
    }


QUESTIONS = [
    question("grounding", "A", "既往证据是否为提出这项假设提供了合理依据？", [
        "主要依据不相关或被误读", "部分有依据，但关键推断缺乏解释", "有清晰依据，并说明了证据迁移的限制",
    ], "评价提出假设的依据，不要求前人已经证实同一个假设。"),
    question("novelty", "A", "相对于已有研究，这项假设具有多大的实质新颖性？", [
        "1 · 基本重复：相同问题和关系已有直接报道",
        "2 · 很小：主要是标签、相邻分区或技术细节变化",
        "3 · 有限但实质：扩展至新的诊断组合、人群或测量定义",
        "4 · 明确：提出前人证据未直接覆盖的重要关系或适用边界",
        "5 · 很强：提出明显不同、可能改变既有理解的科学命题",
    ], "基于给定文献及本人知识判断；不是全球首次性认证。现有文献来自历史 KG 记录，尚缺系统的新颖性检索。", True),
    question("design", "A", "所列数据和分析能否检验核心假设？", [
        "不能检验核心假设", "只能检验部分组成或较弱的命题", "能够检验所述关联及明确限定的范围",
    ]),
    question("validity", "B", "实际分析是否形成可供科学解释的结果？", [
        "没有形成有效分析结果", "形成结果，但存在影响解释的关键方法问题", "形成结果，未发现使其不可解释的关键缺陷",
    ], "分析有效不等于假设获得支持；探索性分析不自动无效。"),
    question("outcome", "B", "核心假设目前得到怎样的经验结果？", [
        "出现明确的反向证据", "证据不确定或只支持部分组成", "在所述探索性范围内获得支持",
    ], "效应方向一致不一定足够；没有通过标准不等于被证伪。", False),
    question("evidence", "B", "这些结果对核心跨诊断关联提供了多充分的支持？", [
        "1 · 很弱：核心关系缺乏支持或结果相互冲突",
        "2 · 较弱：只有部分组成或很不稳定的证据",
        "3 · 中等：有实质支持，但不确定性或覆盖限制明显",
        "4 · 较充分：主要组成与相关敏感性结果提供一致支持",
        "5 · 很充分：在所述范围内证据充分，主要替代解释得到妥善处理",
    ], "结合效应、不确定性、混杂与验证覆盖，不只看方向或是否越过筛选门槛。", True),
    question("value", "B", "该研究结果增加了多少有意义的科学信息？", [
        "1 · 很少：没有可辨认的可信信息增量",
        "2 · 较少：主要重复已有认识，增量有限",
        "3 · 中等：提供有价值的验证、扩展或适用边界",
        "4 · 明确：对具体科学问题提供重要的新信息",
        "5 · 很大：对该问题的理解构成实质推进，并值得深入研究",
    ], "准确的不支持结果也可能有价值。新颖性与科学意义分开评价，不以显著性代替价值。", True),
    question("conclusion_fit", "B", "材料中的整理者摘要是否与结果及其适用范围一致？", [
        "核心说明缺乏支持或与结果矛盾", "主要有依据，但需实质性收窄", "范围恰当，无需实质性修改",
    ], "本批没有外部分析后的模型总结原文。本题评价明确标注的整理者摘要，不评价系统独立撰写结论的能力。"),
    question("traceability", "B", "展示的结论能否追溯到测量定义、分析和结果？", [
        "关键依据无法追溯", "部分环节缺失", "主要依据可追溯",
    ], "审阅汇总记录不等于已经重新运行分析。"),
    question("coverage", "B", "外部分析覆盖并支持了哪些具名诊断组成？", [
        "尚不能支持完整具名诊断组合", "部分组成有支持，整体仍有明显不确定性", "在所述探索性范围内支持全部具名诊断组成",
    ], "COBRE/HCP-EP 的精神病结果不能替代 ADHD 或双相验证。"),
    question("feedback", "B", "展示的后轮提议是否合理利用了此前的真实实验反馈？", [
        "与反馈脱节或推论不合理", "部分利用，但推断依据有限", "合理利用结果并保留推断限制",
    ], "评价已记录的反馈到新提议关系，不证明反馈带来了性能提升。"),
]

# A scientifically unavailable analysis is not automatically negative evidence.
for _q in QUESTIONS:
    if _q["id"] in {"outcome", "evidence", "coverage"}:
        _q["options"] = _q["options"] + [{"value": "not_assessable", "label": "无有效分析结果，不作科学支持判断"}]

ISSUES = ["文献匹配", "新颖性依据", "测量或诊断定义", "混杂控制", "多重比较与统计校准", "验证覆盖或数据暴露", "结论外推", "执行或记录缺失", "其他"]

QUESTIONS.insert(7, {
    "id": "contribution_type", "stage": "B", "title": "综合来看，这项结果主要属于哪类科研产出？",
    "help": "这是产出类型判断，不是分数排序；“有证据的新关联”不等于首次性、因果机制或临床用途已获确认。",
    "rating": False,
    "options": [
        {"value": "replication", "label": "对已知关系的有价值复核"},
        {"value": "extension", "label": "对已知关系的新范围、测量或适用边界的扩展"},
        {"value": "new_association", "label": "具有实质新颖性、并获得当前证据支持的观察性关联"},
        {"value": "candidate", "label": "值得继续研究，但尚未得到充分支持的候选假设"},
        {"value": "negative_information", "label": "有信息价值的不支持或反向结果"},
        {"value": "not_interpretable", "label": "目前没有可解释的科学产出"},
    ] + UNABLE,
})
