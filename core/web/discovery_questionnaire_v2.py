"""Six explicitly anchored capability ratings; not a validated psychometric scale."""

PROTOCOL_VERSION = "cs1-complete-output-expert-capabilities-v2"


def question(key, stage, capability, title, anchors, help_text):
    return {
        "id": key, "stage": stage, "capability": capability, "title": title,
        "help": help_text, "rating": True,
        "options": [
            {"value": str(i), "score": i, "label": f"{i} 分 · {anchor}"}
            for i, anchor in enumerate(anchors, 1)
        ] + [
            {"value": "insufficient", "score": None, "label": "材料不足，无法判断"},
            {"value": "outside", "score": None, "label": "超出本人专业范围"},
        ],
    }


QUESTIONS = [
    question("grounding", "A", "证据整合", "系统如何利用既有证据来提出这项假设？", [
        "主要依据不相关，或存在关键误读",
        "依据部分相关，但核心推断缺乏支撑",
        "依据基本相关，关键连接仍需解释",
        "依据与推断衔接清楚，主要迁移限制已说明",
        "多条证据形成充分、清晰且有边界的推断依据",
    ], "评价证据到假设的推理，不要求已有研究直接证实同一假设。"),
    question("novelty", "A", "新假设提出", "系统提出的假设具有多大的实质新颖性？", [
        "基本重复已有的相同命题",
        "主要改变标签、相邻分区或技术细节",
        "扩展了有意义的人群、测量或诊断组合",
        "提出已有证据未直接覆盖的重要关系或边界",
        "提出明显不同、可能实质推进既有理解的科学命题",
    ], "依据给定文献及本人知识判断；这是新颖性评价，不是全领域首次性认证。"),
    question("design", "A", "实验设计", "系统采用的实验方案能在多大程度上检验所述假设？", [
        "核心命题未被转化为可检验的分析",
        "数据、测量或比较设计存在关键不匹配",
        "可以检验部分命题，但关键设计限制仍明显",
        "能够检验核心关联，主要混杂和比较范围明确",
        "检验设计充分对应命题，主要替代解释有针对性检查",
    ], "结合测量定义、诊断覆盖、混杂与验证安排；评价展示的流程，不推定未展示的自主设计过程。"),
    question("validation", "B", "实验验证", "系统实际完成的实验验证有多严谨、可信？", [
        "没有有效结果，或关键错误使结果不可解释",
        "虽有结果，但重大执行或统计问题限制可信度",
        "产生可解释结果，但稳健性或验证覆盖有限",
        "主要检查充分，结果与关键不确定性可追溯",
        "所需验证与敏感性检查完整，主要不确定性得到充分处理",
    ], "评价验证质量，不以阳性与否评分。严谨得到不支持结果也可高分；方向一致不能替代统计校准。"),
    question("value", "B", "科学信息增量", "这项研究结果增加了多少有意义的科学信息？", [
        "没有可辨认的可信信息增量",
        "主要重复已有认识，增量很小",
        "提供有价值的复核、扩展或适用边界",
        "对具体科学问题提供重要的新信息",
        "对该问题的理解构成实质推进，值得深入研究",
    ], "结合完整结果判断；可信的不支持结果也可能有价值。新颖不自动代表重要。"),
    question("feedback", "B", "反馈驱动迭代", "系统如何利用此前的真实实验反馈形成后轮假设？", [
        "提议与反馈脱节，或明显曲解反馈",
        "仅提及反馈，未体现可解释的调整",
        "据反馈作出相关调整，但推断依据有限",
        "反馈与后轮提议衔接清楚，并保留推断限制",
        "充分利用反馈及不确定性，形成有针对性的后续检验",
    ], "评价已记录的父反馈到当前提议的关系；不把模型评议当作人类意见，也不据此证明迭代优于无反馈。"),
]

SCORING = {
    "version": "six-capabilities-equal-weight-v1",
    "minimum": 1, "maximum": 5,
    "item_ids": [q["id"] for q in QUESTIONS],
    "non_numeric": ["insufficient", "outside"],
    "case_composite": "Arithmetic mean of all six numeric ratings; otherwise null.",
    "session_composite": "Mean of complete six-item case composites; report case coverage separately.",
    "study_composite": "Equal-weight mean of case-level mean composites; withhold the full-panel mean unless every case has a valid composite.",
    "interpretation": "Descriptive expert-rated research-capability index (1–5), not a validated latent scale, discovery probability, or scientific ground truth.",
}
