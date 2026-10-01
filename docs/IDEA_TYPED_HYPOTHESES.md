# Idea 问题2：类型化链条与证据后处理

实现入口是 `core/idea_hypotheses.py`。复用问题1的 topic 检索、`AcceptedClaimLayer.query_batch`、原始 claim 索引，以及已有 `TaskChain` / `HypothesisEngine`。图谱只读；本模块没有模型调用、训练或评分。

## 使用

idea 模式提供 `generate_idea_hypotheses(topic, templates?, limit?, offset?, page_size?)` 工具。默认尝试已有的五种模板，可显式选择其中一种或多种。**候选池默认500条，`limit`为1–2000**；这是待筛选池的规模，不是最终推荐数。

工具默认每页返回20条，`page_size`为1–50，`candidate_count`说明实际池大小；用`next_offset`继续读取。同一图谱版本、topic、模板和limit的分页复用最近一份候选池，每次仍检查依赖版本。CLI与HTTP返回完整池。

```powershell
python -B -m core.idea_hypotheses --topic "APOE hippocampal atrophy" --limit 500 --output apoe_pool.json
python -B -m core.idea_hypotheses --topic "Marker" --template genetic_imaging_disease --output candidates.json
```

`--output` 只创建新文件，不覆盖旧输出。HTTP 入口：`GET /api/kg/idea-hypotheses?topic=...&limit=500`，可重复传 `template`。

| 模板 | 节点顺序 |
|---|---|
| genetic_imaging_disease | 基因/靶点 → 影像标志物 → 疾病 |
| drug_imaging_outcome | 药物 → 影像标志物 → 结局 |
| task_brain_behavior | 认知任务 → 影像标志物 → 个体行为数据 |
| disease_biomarker_prognosis | 疾病 → 影像标志物 → 结局 |
| pathway_polygenic_mediation | 基因/通路 → 影像标志物 → 结局 |

每种模板的逐边谓词白名单和方向由 `STEP_RULES` 明确声明，完整规则随 `template_spec` 返回。最后一种模板沿用旧注册名；名称中的 mediation 不代表已证明中介因果。只有显式对称的关联关系允许反向遍历，输出仍保留原始边方向。

## 生成和提交约束

1. 从既有原始索引读取可用谓词与声明类型，复用topic词组/同义词及共享目录的检索文字寻找锚点，再枚举锚点的全部已索引邻接记录。取消旧版每篇论文只选一条、前32条检索记录/64条观察/64条邻接边的生成限制；问题1的论文展示接口不变。没有新增索引或图谱写入。
2. 根据原始记录声明的类型枚举连续链条，逐边验证关系、方向、类型、实际 CLM ID；再查询实际图谱概念确认端点存在。未知类型不靠名称推断补齐。补充来源记录仍在证据档案内，但不能冒充原始图边参与遍历。
3. 先去重/枚举，再按需分批调用现有`query_batch`和真实节点读取器，避免把全部原始证据都展开进内存。同一节点序列与原始有向谓词链的不同论文组合不重复计为新hypothesis。最终topic条件必须在这条路径自己的metadata/论文记录中出现，共享目录其他论文的诊断不能替代。`topic_match`明确完整/部分词组覆盖，不把词命中当语义相关性验证。
4. 后处理保留完整原始 metadata、论文身份、来源锚点与核查状态，并单列人群、条件、测量、时间、方向、否定、调节和交互信息。未知字段保持空值。不同已知范围或明确否定/不支持证据进入 `held_chains`；相同文本不等于已证明跨研究条件一致。
5. `candidates` 中每条保留机器链条、逐边证据、条件化假设、依据、可证伪预测和限制。默认预测是验证两条观察及中间变量的额外预测信息，整链推断标为待检验；不从路径推出因果或创新性。
6. idea 写入工具要求结构字段。提交候选 JSON 时，即使关闭评分/独立评审，或用尽评分重试次数，也会重新绑定当前图谱、原始边、真实端点和版本；候选自填的证据上下文被服务器重建。合规结构与上下文继续传给现有评审接口。现有评审每批最多8条的限制仍保留，它不再限制前面的生成池；大池的评分与排名属于问题3。

候选交付应同时提交 JSON 与 IDEA.md。仅交付文字的证据不足报告不被当作链条验证通过。旧的节点顺序型 `TaskChain` 调用保持兼容；如果把带关系约束的新模板传给旧宽松路径接口，会明确拒绝，要求使用严格枚举接口。

## 输出与范围

顶层包括 `candidates`、`held_chains`、`template_path_counts`、`retrieval`、`gap` 和图谱/claim层/索引版本。每条候选包括：

- `chain` / `kg_triples`：模板、真实端点、逐边谓词、CLM ID。
- `chain_context` / `evidence_queries` / `source_ids`：逐边来源与限制；提交评审时再读取完整证据。
- `hypothesis` / `rationale` / `prediction`：可由 idea 主模型据证据细化的条件化草稿，结构字段必须保留。
- `structural_validation: passed` 与 `scientific_validation: not_established`：两种结论严格分开。

这是有界候选生成，未证明全图召回。每个模板最多枚举`max(1000, limit*2)`条不同路径，默认1000条；每个池最多返回`limit`条合规草稿及`limit`条待核查链条。`retrieval`报告路径截断、输出截断、缺失节点、topic不匹配和待核查数。已有路径按词组覆盖优先展开，这不是科学质量ranking。类型缺失、谓词不在模板内、真实端点不连续、截断或条件冲突仍可能使结果不足目标数量；不靠虚构关系或重复表述凑满。

问题2的交付到此为止。随后问题3已接入全池预排序、GNN适配器与三方回应，见 `docs/IDEA_HYPOTHESIS_RANKING.md`；当前尚缺真实GNN/模型评审运行输入，不表示质量验证已完成。

## 验证

候选池扩展后的离线检查：279项通过（新增大池组合、跨页复用、不重复计算论文变体、共享topic必须经自身证据核对、缓存版本失效等检查）；8项旧完整大图加载测试仍因已知内存限制排除。临时微型图谱中的17条前半边与18条后半边形成306条不同链条，只来自两篇测试论文，全部通过实际loader与只读检查。记录：`tmp/idea_chain_pool_20260929/offline_tests.xml`。真实大池检查结果以同目录`verified/RESULTS.json`为准。

2026-09-29真实大池结果（limit=500，原始图谱与证据输入不变）：

| topic | 旧版草稿 | 当前合规草稿 | 其中完整topic词组命中 | 单独待核查 |
|---|---:|---:|---:|---:|
| Hippocampal atrophy in MCI | 0 | 135 | 18 | 4 |
| Cerebellum ADHD | 0 | 500 | 239 | 179 |
| APOE hippocampal atrophy | 1 | 300 | 48 | 120 |
| qzxnonexistent | 0 | 0 | 0 | 0 |

其余草稿均明确标为部分词组匹配，完整词组命中也不是语义相关性/科学正确性验证。ADHD达到500条输出上限，其中一个模板也触及路径枚举上限；MCI/APOE在这五种模板和当前声明类型下分别找到139/420条不同链条，未因输出上限截断。935条草稿与303条待核查链条共1,238条结构检查通过，三个真实topic各抽首/中/尾3条再次绑定新读取的证据，9次通过；生成过程中每条链都已读取原始记录、核对类型/方向/端点。待核查链条不混计为可用草稿。

cold首题342.977秒，后续ADHD125.855秒、APOE76.064秒，空topic5.420秒；此处不声称大量生成是瞬时操作。4,829个绑定输入stat及campaign字节均未变。没有模型调用、评分、训练或图谱修改。新池数量足以接续问题3，实际相关性、科学价值与新颖性仍需筛选。

以下保留旧小批量版本的验证记录；其0–1条输出不能作为大量候选生成已足够的结论。

离线检查：275项通过，包含24项本次新增测试。覆盖微型真实 loader 的正例与只读性、错误类型/关系/方向/断链、假 ID、版本变化、补充来源、条件冲突、否定、候选篡改、HTTP 和真实 runtime 接线。8项旧完整大图加载测试因 `json.load` 内存不足未完成，最终回归明确排除，没有计入通过数。最终记录在 `tmp/idea_chain_generation_20260929/final_offline_tests.xml`。

真实数据检查记录在 `tmp/idea_chain_generation_20260929/`。首次 `run/` 在补充来源 ID 查询处失败，保留诊断；修正后的 `verified/` 独立记录，不覆盖旧结果。使用每题20条检索记录、最多4条输出的预算：

| topic | 论文数 | 原始观察数 | 合规候选 | 待核查链条 |
|---|---:|---:|---:|---:|
| Hippocampal atrophy in MCI | 19 | 76 | 0 | 0 |
| Cerebellum ADHD | 21 | 85 | 0 | 0 |
| APOE hippocampal atrophy | 25 | 81 | 1 | 0 |
| qzxnonexistent | 0 | 0 | 0 | 0 |

唯一候选是原图记录的 `APOE epsilon4 —modulates→ hippocampal atrophy —is_risk_factor_for→ cognitive impairment`，关联 PMID:15557508 与 PMID:39989286。真实端点、原始边、类型和版本在提交评审前重新绑定通过；这仍是**未验证的条件化草稿**，两个来源未在当前登记中核查，部分范围字段缺失，第二条观察记录 A+T+ 条件。这里不确认其科学合理性或新颖性。三个非空topic的邻接检索均达到截断边界，不能以两个零候选推断全图没有路径。

首次加载及首题362.596秒，后续ADHD/APOE为29.482/36.565秒，空查询10.514秒。4,829个绑定输入的大小/修改时间以及campaign字节均未变。

最终后处理还修正了 `evidence.direction` 的读取与全空人群字典的未知标记；保留原始metadata不改写。用刚核查的真实观察离线重放生成 `postprocessed_genetics.json`，断言结构、来源和条件不变、两个方向均保留positive、人群仍未知。该文件明确记录是离线后处理复核；`verified/RESULTS.json` 保留此前实际读图运行的代码哈希。不能用这些开发检查代表实际 idea 质量提升。
