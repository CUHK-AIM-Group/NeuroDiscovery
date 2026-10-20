# Idea：检索、假设构造与筛选

本文件统一 Idea 当前功能说明，检索、类型化结构和排序分别解释；排序不等于科学确认。

历史过程与旧版原文统一保存在 [说明文档历史归档](../dev/archive/docs-history_20261019.zip)，不再另建逐次进度稿。


## 筛选偏好

## Hypothesis selection in the client

The **Hypothesis selection / 假设筛选偏好** control sets `novelty_mode` for the
current chat. It appears in the model menu only for the two AutoResearch modes
that produce hypotheses: **Idea** and **Full**. Data and Model runs cannot change
how hypotheses are selected, so the route is hidden there and a stored mode is
kept unchanged. The default is `balanced` for new chats and requests that omit
the mode. Existing saved chat selections are preserved. Selection is saved with
the chat and captured on each new or edited/resubmitted turn, including the
exported agent work log. Changing it does not modify an already frozen workflow.

- `novelty_first`: only evidence-backed novelty candidates; fewer or zero is valid.
- `balanced`: fill the budget from every scientifically valid candidate, ranked by
  novelty 40%, structure 20%, GNN 20%, scientific review 20%.

There are two modes. The earlier three were consolidated by a behaviour-preserving
rename: the previous `strict` became `novelty_first` and both the previous
`novelty_first` and `weighted` became `balanced`. The unambiguous old names
`strict` and `weighted` are still accepted at the request/CLI boundary and mapped
to their current equivalent; a live `novelty_first` request uses the new meaning. The client
only offers the two current modes. Stored chat selections written before the rename
are migrated once, using the recorded store schema version: an old stored
`novelty_first` meant the permissive composite mode and therefore migrates to
`balanced`, not to the current conservative `novelty_first`.

The within-tier quality score is 30% structure, 30% GNN, and 40% scientific
review. Novelty utility is 1 for supported substantive extension, 0.7
for a supported potential new relation, 0.2 for uncertainty, and 0 for known
relations. These are decision utilities, not probabilities or first-report claims.

HTTP and WebSocket chat handlers validate the mode and supply a scoped selection
instruction to the agent. The client does not itself perform literature searches
or certify novelty. A generated narrative alone is not a validated selection.
The deterministic selection boundary is `neurooracle.src.novelty_policy.select_reviewed`,
also exposed as `POST /api/hypotheses/select`. Both modes require the same
scientific validity and operationalization gates. A known-prior warning from any
expert vetoes novelty eligibility; unresolved evidence does not become novelty.

### Reviewed candidate contract

Each candidate needs `hypothesis_id`, `structural_score`, `gnn_path_score`,
`science: {verdict, critic_score}`, three independently obtained `expert_novelties`,
and an independent literature `adjudication`. Scores must be finite in [0, 1].
Each review contains:

- `classification`: `exact_prior`, `same_scientific_conclusion`,
  `substantive_extension`, `potential_new_relation`, or `uncertain`;
- `reference_ids`: an array of actual reviewed source identifiers;
- `scientific_delta`: documented substantive difference, empty when unknown;
- `executable_test_covers_delta`: boolean.

Adjudication additionally requires `search_evidence_sufficient` and
`registered_test_matches_hypothesis` booleans. Missing data must be disclosed,
not fabricated. The API recomputes gates from these reviews rather than trusting
client-supplied gate flags. Schema validation is not verification of paper contents.

```json
{"novelty_mode": "novelty_first", "limit": 5, "candidates": []}
```

The response retains every original candidate and literature label, appends
selection reasons, and returns ranked `selected_ids`. Known candidates selected
by a permissive mode remain replication; no mode grants a new-finding claim.

For local reviewed JSON lists, the agent can use the same policy without a server:

```bash
python -m neurooracle.src.novelty_policy --input reviewed.json --output selection.json --mode novelty_first --limit 5
```

The output must be a new file. Historical experiments under `.codex_tmp` are not
imported or modified. No model calls, statistics, KG edits, or experiment reruns
are performed by this selector.

### Interface reference

The visual refresh was informed by the locally installed DeepSeek Harness
`0.1.1-rc.2` frontend (`dsh-client-ui-layout/lib/client.js` and
`dsh-web-frontend/dist/assets/index-C6eRlFa6.css`): neutral surfaces, a compact
sidebar, unobtrusive controls, restrained borders, and clear menu hierarchy.
NeuroDiscovery uses its own CSS and retains its identity and existing workflows;
it does not bundle DSH code, logos, or runtime dependencies.


## 主题检索

## Idea topic 检索

输入 topic，返回相关命题、按论文去重的 `studies`、来源证据和 metadata。复用 `AcceptedClaimLayer` 的当前接受版本；查询不修改图谱，不生成假设，也不执行评分。

### 使用

```powershell
& C:/Users/45846/anaconda3/envs/neuroclaw/python.exe -B -m core.topic_evidence --topic "海马萎缩与轻度认知障碍" --minimum-coverage 0.5 --json
```

网页服务提供 `GET /api/kg/topic-evidence?topic=Hippocampal%20atrophy%20in%20MCI&minimum_coverage=0.5`。服务进程复用已加载的图谱证据；每次重新启动 CLI 都有首次加载成本。

`--term`（HTTP 的可重复 `term`）补充检索词；自动中英文别名仅覆盖一小组常用词和缩写。`--limit` 控制最多返回多少命题/原始观察，`--evidence-claims` 控制展开前多少条的论文证据，默认分别为 20 和 10；共享与单篇结果共用预算。`--minimum-coverage` 默认 0.5，是命中查询概念的比例，不是科学相关概率；同一个概念的多个别名只算一次。设为 1 可要求全部查询概念命中。`--minimum-papers` 沿用现有接口的“经核查支持该命题的论文数”定义；单篇候选展开后才有确切计数，正数过滤会排除未展开/核查计数不足的候选，可能少于请求数量。`--output` 可保存 JSON，拒绝覆盖已有文件。

### 返回内容与边界

- `studies`：按图谱已记录的 work 身份去重；含 PMID/DOI、完整 bibliography、出版版本与状态、来源身份、相关命题及逐项 observation。逐论文检查查询覆盖率及显式疾病词，`missing_terms` 标明部分匹配；其他论文只保留在命题上下文中，不算 topic 命中的论文。
- observation：原有 evidence、population、conditions、完整原记录 metadata、source_review、source_anchor、抽取文本及来源类别。没有来源锚点时明确为图谱抽取文本；缺失字段保持缺失，不补造。
- `claims`：命题匹配词、查询覆盖率、已有论文计数及证据展开状态。`retrieval_origin` 区分共享目录与原始单篇观察，`query_id` 可直接交给现有 `CLM:`/`REL:` 证据接口。先按检索覆盖排序，再沿用已知论文计数；未展开单篇的论文/支持计数为 null。论文列表另按自身记录命中词和来源可用性排序。
- `claim_total`：共享目录条目数加未被目录覆盖的原始观察数；`shared_claim_total` / `shared_claim_matched` 单列共享部分，`original_retrieval` 列出全量索引数、排除重复后的范围、命中观察及来源数。命中数在单篇展开后的支持论文过滤之前统计，不是论文总数。
- `graph_revision` / `claim_layer_revision` / `original_index_revision`：搜索和展开使用同一版本；即使空结果或不展开证据，也提供版本。文件或版本中途变化时整次查询失败，不返回部分证据。

当前范围为已发布共享命题目录及其接受的扩展，加上其绑定的既有全量原始 claim 索引。单篇检索扫描索引已有的概念端点、关系、PMID/DOI，排除已在共享目录中的观察，每个原始来源保留一条最佳匹配进入统一预算。没有新建索引或合并命题；**原始索引不含全文与完整摘要，也不等于全库文献搜索**。未配置该索引的图谱仍可搜索共享目录，并明确报告这个缺口。空结果不能证明没有相关研究；命中不等于证据支持某个新假设，更不证明创新性。

单篇证据通过既有 `query_batch` 返回，使用索引中的字节位置读取完整原记录，再核对独立 census 的 ID、hash、单篇成员范围及关系身份。复用原有论文身份、来源审阅和 evidence 聚合函数，缓存只限当前版本；原始 metadata、不完整抽取文本、未核查状态均保留。

历史验收将应用代码也记录为审计文件。读取器现将仅位于 `review_inputs` 的三个准确路径（`server.py`、`claim_layer_v7.py`、`claim_layer_support_v1.py`）视为本次修改的运行代码，报告 `historical_code_drift`，不要求当前应用字节等于旧代码。图谱、projection、来源材料、其他审计文件以及这三个路径若被用作来源输入时的校验继续执行。原验收文件不变；这项修复不意味着旧验收在当前代码上被重新完成。

首次加载仍校验完整原始 XML 文件；补充证据缓存只保留本层实际使用的完整论文，复用原解析器，随后逐项核对与已封存 observation 完全一致，避免保留同一 XML 中大量无关论文。

### 验证

```powershell
& C:/Users/45846/anaconda3/envs/neuroclaw/python.exe -B -m pytest -q -p no:cacheprovider core/test_topic_evidence.py core/web/test_claim_layer_v8.py core/web/test_claim_layer_support_v1.py core/web/test_claim_evidence.py core/agent/test_novelty_gate.py
```

120 项相关测试通过，含原始单篇与旧查询证据逐字段一致、只读、共享去重、错误字节位置/ID、版本变化及预算/过滤测试。真实图谱的有界开发检查脚本及结果保存在 `tmp/idea_topic_retrieval_20260928/`；这是检索可用性检查，不是科学准确率或全库召回率评估。

补充单篇后的本机实测（`singletons_20260929/RESULTS.json`，共享与单篇共用前10条展开预算，minimum_coverage=0.5）：

| 查询 | 返回论文记录 | 耗时 |
|---|---|---|
| Hippocampal atrophy in MCI | 9，全部查询词命中；9篇此前未返回 | 新进程首次362.393s |
| 海马萎缩与轻度认知障碍 | 与英文相同的9篇、相同CLM/REL选择 | 37.698s |
| Cerebellum ADHD | 11，其中8篇此前未返回、3篇部分匹配 | 29.540s |
| qzxnonexistent（空结果负例） | 0 | 10.426s |

扫描既有905,110条原始观察，排除共享目录已覆盖的8,232条，余下896,878条参与单篇检索；加4,081条共享目录，`claim_total=900,959`。MCI原始观察命中2,258条/1,869个来源，ADHD命中19,367条/13,543个来源（含部分词命中）；这些不是最终论文数量或科学相关率。每个原始来源只有一条代表观察进入预算，最终MCI选择9条单篇、ADHD选择8条单篇。由于共用预算、重新排序，新返回集合不是旧返回集合的简单叠加。

4,829个绑定输入文件大小/mtime及campaign字节未变，索引数据库和图谱只读。中英文一致、原记录metadata保留、空结果及新增单篇路径检查通过。首次加载仍约6分钟，全量旧索引扫描使后续topic查询约30–38秒；空负例约10秒。未控制操作系统缓存，不声称已经快速或高召回；当前足够供问题2开发，后续由端到端使用判断是否值得继续优化速度。

来源缺口保持显式：MCI的9篇均未在现有身份登记中核查，也没有已核查来源锚点；8条有原抽取文本，9条有原metadata。ADHD的11篇含3篇身份已核查记录，共12条观察，其中4条有来源核查锚点。这两组的结构化population/conditions仍为空。**查询词命中、标题相关或来源身份核查，不证明每条观察支持该topic的研究结论。** 问题2应把缺字段、未核查和不支持的观察作为约束，不能直接当作已确立事实。

以下是先前仅共享目录的本机实测（`refined/`，前10个命题展开，minimum_coverage=0.5，保留作比较）：

| 查询 | 命题命中/目录总数 | 返回论文 | 耗时 |
|---|---|---|---|
| Hippocampal atrophy in MCI | 23 / 4,081 | 3；其中2篇部分匹配 | 新进程首次323.307s |
| 海马萎缩与轻度认知障碍 | 23 / 4,081，与英文相同 | 同上 | 2.568s |
| Cerebellum ADHD | 137 / 4,081 | 19；其中18篇部分匹配 | 2.473s |
| qzxnonexistent（空结果负例） | 0 / 4,081 | 0 | 2.221s |

首次校验仍需约5.4分钟，适合复用常驻网页服务的已加载实例。未控制操作系统磁盘缓存，不把这个数字当作通用启动性能。最终运行的4,089个输入文件大小/mtime及campaign字节保持不变。

样例完整匹配首条为 PMID 39350364（MRI/血浆生物标志物研究）和 PMID 26482248（ADHD/自闭症结构影像比较综述）。逐项 observation 的来源核查状态不同；有核查锚点也有旧抽取文本。返回样例的结构化人群/测量条件为空，原文与metadata原样保留，未自行补充。共享命题下其他疾病的论文保留为上下文，不混入主 `studies` 列表；部分匹配仍需用户或后续生成环节判断用途。


## 类型化假设

## Idea 问题2：类型化链条与证据后处理

实现入口是 `core/idea_hypotheses.py`。复用问题1的 topic 检索、`AcceptedClaimLayer.query_batch`、原始 claim 索引，以及已有 `TaskChain` / `HypothesisEngine`。图谱只读；本模块没有模型调用、训练或评分。

### 使用

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

### 生成和提交约束

1. 从既有原始索引读取可用谓词与声明类型，复用topic词组/同义词及共享目录的检索文字寻找锚点，再枚举锚点的全部已索引邻接记录。取消旧版每篇论文只选一条、前32条检索记录/64条观察/64条邻接边的生成限制；问题1的论文展示接口不变。没有新增索引或图谱写入。
2. 根据原始记录声明的类型枚举连续链条，逐边验证关系、方向、类型、实际 CLM ID；再查询实际图谱概念确认端点存在。未知类型不靠名称推断补齐。补充来源记录仍在证据档案内，但不能冒充原始图边参与遍历。
3. 先去重/枚举，再按需分批调用现有`query_batch`和真实节点读取器，避免把全部原始证据都展开进内存。同一节点序列与原始有向谓词链的不同论文组合不重复计为新hypothesis。最终topic条件必须在这条路径自己的metadata/论文记录中出现，共享目录其他论文的诊断不能替代。`topic_match`明确完整/部分词组覆盖，不把词命中当语义相关性验证。
4. 后处理保留完整原始 metadata、论文身份、来源锚点与核查状态，并单列人群、条件、测量、时间、方向、否定、调节和交互信息。未知字段保持空值。不同已知范围或明确否定/不支持证据进入 `held_chains`；相同文本不等于已证明跨研究条件一致。
5. `candidates` 中每条保留机器链条、逐边证据、条件化假设、依据、可证伪预测和限制。默认预测是验证两条观察及中间变量的额外预测信息，整链推断标为待检验；不从路径推出因果或创新性。
6. idea 写入工具要求结构字段。提交候选 JSON 时，即使关闭评分/独立评审，或用尽评分重试次数，也会重新绑定当前图谱、原始边、真实端点和版本；候选自填的证据上下文被服务器重建。合规结构与上下文继续传给现有评审接口。现有评审每批最多8条的限制仍保留，它不再限制前面的生成池；大池的评分与排名属于问题3。

候选交付应同时提交 JSON 与 IDEA.md。仅交付文字的证据不足报告不被当作链条验证通过。旧的节点顺序型 `TaskChain` 调用保持兼容；如果把带关系约束的新模板传给旧宽松路径接口，会明确拒绝，要求使用严格枚举接口。

### 输出与范围

顶层包括 `candidates`、`held_chains`、`template_path_counts`、`retrieval`、`gap` 和图谱/claim层/索引版本。每条候选包括：

- `chain` / `kg_triples`：模板、真实端点、逐边谓词、CLM ID。
- `chain_context` / `evidence_queries` / `source_ids`：逐边来源与限制；提交评审时再读取完整证据。
- `hypothesis` / `rationale` / `prediction`：可由 idea 主模型据证据细化的条件化草稿，结构字段必须保留。
- `structural_validation: passed` 与 `scientific_validation: not_established`：两种结论严格分开。

这是有界候选生成，未证明全图召回。每个模板最多枚举`max(1000, limit*2)`条不同路径，默认1000条；每个池最多返回`limit`条合规草稿及`limit`条待核查链条。`retrieval`报告路径截断、输出截断、缺失节点、topic不匹配和待核查数。已有路径按词组覆盖优先展开，这不是科学质量ranking。类型缺失、谓词不在模板内、真实端点不连续、截断或条件冲突仍可能使结果不足目标数量；不靠虚构关系或重复表述凑满。

问题2的交付到此为止。随后问题3已接入全池预排序、GNN适配器与三方回应，见 `docs/IDEA.md`；当前尚缺真实GNN/模型评审运行输入，不表示质量验证已完成。

### 验证

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


## 排序与模型边界

## Idea 问题3：候选池评分与排序

既有接线阶段连接候选池、GNN推理和新颖性评审，没有训练模型。2026-09-29用户进一步允许在图谱上训练；当前范围仍仅为问题3的辅助评分，图谱只读，不扩展论文抽取工程。

**最新一轮优化（2026-09-29，用户“建议一轮做完”）**：完成20开发+12固定检查候选的来源阅读、真实评审、主审裁决及无/旧/新GNN同池比较。25条完整评审、7条缺来源零调用HOLD；主审条件保留6条（开发3、检查3），不等于科学成立或已证实新颖。检查集模型自动只选T11，但其实际文本有ε4/ε2条件混用及预测基线缺口，主审HOLD；主审保留T06/T07/T12。三种GNN配置在检查集保留相同3条，仅排序不同：无GNN及新R-GCN为T06→T07→T12，旧GraphSAGE为T06→T12→T07。没有同等级提案的独立价值排序标准，且5条参考可保留候选只形成1个共享来源簇，不能据此宣布新模型更优。

本轮后处理区分已观察前提和待检验增量；自适应评审通常5次、有回应8次，critic取三个最新视角的中位数，来源疑点仍阻止入选。实验使用冻结v4.3；生产v4.3.1另修主审标记漏项，18条/118份缓存重放确认分数、科学字段和入选不变。336项广泛回归+26项针对性测试（338个不同案例）通过；独立复算420个评审引文、128个边分、36组排序，并核对来源边界、旧记录和默认配置。总256次HTTP尝试，237次200，Ollama-only，约91.92分钟请求跨度；开发修复、D15未知回应未重发、T10引用空格修复及原轮检查10完整/1缺来源/1格式失败均保留。图谱与模型只读，无新训练或默认切换。完整说明见 `tmp/idea_chain_review_round_20260929/run/REPORT.md`，可读提案及条件在 `REVIEWED_HYPOTHESES.json`，客观复核在 `VERIFIED.json`。本轮有界优化完成，自动评审可靠性及GNN排序收益仍未证实；下面早先“无最终推荐项”是上一轮历史结果。

**当前修正（2026-09-29）**：用户指出之前的KGE网络也是GNN。核对后，历史`GraphSAGE + DistMult`确实是GNN，先前把idea入口限定为R-GCN过窄。现已通过`FrozenGraphSAGEScorer`复用其冻结导出，并默认读取显式配置`neurooracle/configs/idea_gnn.json`；对935条既有真实草稿实测635条完整链，剩余300条词表不覆盖，继续保持null。另一个`kge_complex.pt`是ComplEx模型，不是消息传递GNN，未作为GNN代入。

**全量新模型（2026-09-29）**：用户明确授权全量训练与Goal模式后，R-GCN + directional解码器完成28轮、验证选择23，独立数值复核通过。1179903条合规关系；935条既有草稿中724条可测、211条null。使用配置`tmp/idea_full_gnn_20260929_final/idea_gnn.json`（`kind=rgcn_export`），由`FrozenRGCNScorer`核对权重、导出、词表及训练曝光掩码后推理，未训练词表不打分。项目默认仍保留旧GraphSAGE：新模型抽样MRR仅比度数基线高.006983、Hits@10更低，没有充分证据确认更优。测试全体88123条中仅1379条可测；高sigmoid分数不代表假设成立概率。训练、失效旧划分、同一候选复算和资源记录见该目录README及VERIFIED.json。

**来源后处理与真实三方对照（2026-09-29）**：用户同意优先改善链条后处理和综合评审。本轮修复中间节点名称被后一篇覆盖的问题；935条旧草稿离线重建保持ID、链、三元组和来源不变，其中586条同时保留不同的来源描述（不把名称差异当成错误）。来源细化→三方初评→三方回应→裁决正常共8次调用，各阶段保留测量、前提、范围和关系判断及逐字锚点；原文不足HOLD，来源反对不能被GNN高分覆盖。真实实验冻结v4代码；单独生产修复v4.1复用明确的DOI别名恢复3份摘要，不回填旧实验，也未宣称新提示已重新完成整轮科学验证。37条固定候选（6开发+31新来源）实际完成29条，8条未完成；共330次HTTP尝试、280次200，全部沿用Ollama，保留本地PoolTimeout诊断、有限恢复和恢复前统计，没有重发未知结果。无GNN/旧GraphSAGE/新R-GCN复用同一组真实评审，balanced和novelty_first均无最终入选，不能据此比较好假设的排序增益。31条新参考在调用前冻结为18拒绝/13待核查/0保留，题材失衡、不是独立专家金标准；根助手另记录模型把共同测量说成未测量、用相反相关方向断言不可能等错误，三方一致不是科学权威。329项回归及独立原响应/来源/六种排序复算通过，默认GNN、图谱、旧权重与旧记录未改。完整证据见`tmp/idea_source_ranking_20260929/run/REPORT.md`和`VERIFIED.json`。本轮接线和有界对照完成，能否保留并排好有价值的假设仍未证实；不自动重训或恢复KG工程。

### 使用流程

**扩大来源审阅（2026-09-29）**：按用户要求新增120条不同候选，累计138条。每topic30条条件抽样框内固定随机样本，另10条靠前/补足；新增判断在逐条分数揭示前冻结。主90条排序一致旧497/809（61.4%）、新307/809（37.9%）；移除共同来源PMID:35383335的9条后变50.5%/45.0%，topic等权均值48.8%/48.5%，说明旧优势很受重复错误来源驱动。剩余抽样框内每模型前5/topic，两者均仅2/15达到条件保留；不能宣布旧模型普遍更优。105份来源中104份摘要已读/复用，2条相关候选保留原始来源缺失HOLD。复用已验证真实评分，无新训练、推理或模型API；抽样、计算、来源/模型绑定与只读检查通过。记录在`tmp/idea_gnn_comparison_expanded_20260929/run/REPORT.md`和`VERIFIED.json`。仍只是来源兼容性开发诊断，未评新颖性，不是独立科学金标准；下方18条是历史子集，判断未改。

**同池人工来源审阅（2026-09-29）**：已按用户要求完成两模型实际评分及主助手逐条审阅。共同可测600条中选18条，29份摘要，先固定判断后揭示分数；旧模型与判断一致15/30个同topic排序对，新模型7/30，旧模型的优势集中在APOE组。两者都把部分测量混接/证据不支持的链排前，故保持旧默认辅助分，不宣称新模型更优或旧模型足以筛选科学新颖假设。该有意分层的小样本、共享来源和主助手判断不构成独立专家验证。完整逐条分数、来源与复核见`tmp/idea_gnn_comparison_20260929/run_v2/REPORT.md`、`VERIFIED.json`；没有修改图谱、权重或运行逻辑。

Idea runtime 提供 `rank_idea_hypotheses`，参数与 `generate_idea_hypotheses` 相同（topic、templates、limit、offset、page_size）。它复用同一个生成池，默认500、最多2000条；全池评分后分页返回，不把每次评审的8条上限套在生成池上。完整的轻量排序表保存在 run 的 `idea_pool_ranking`。

1. **全池预排序**：topic词组覆盖 → GNN边分（仅当全池均可测）→ 两端节点在既有原始索引中的来源数量所形成的稀疏度 → 原始边confidence。部分GNN覆盖时仍保留逐项实测值，但不靠缺分把图外词表候选排到末尾。这里没有文献新颖性判定。
2. **证据细化与小批评审**：运行时重新读取CLM证据、真实端点并校验条件；另从既有来源库只读绑定原摘要，先检查CORPUS/heldout和已知版本边界再读载荷。增加一次来源细化调用，只允许改写hypothesis、prediction、rationale、limitations；保留原稿、稳定candidate_id、链条和版本。明确比较两侧中间节点的实际测量，保留人群、比较组、时间、主效应与调节效应。条件不同可以形成明确提出的迁移检验，不能直接当成已成立机制。自填分数、来源文本和证据不作为可信输入；128 KiB证据上限仍按每条候选检查。
3. **三方独立评分与必要回应（v4.3）**：prior_art、scientific_delta、operationalization各独立调用一次，初评不看细化者的判断。意见一致时直接裁决；有实质分歧时，各视角看到相同初评包并回应一次，再裁决。三方来源判断均为hold/reject时，不为新颖性分类差异追加讨论。含细化通常5次、有回应8次，格式修复另计；缺少必要原文则0次调用并单列source_unavailable，不算完成三方评审或科学拒绝。各阶段保留source_assessment及逐字锚点。全部来源判断均有锚点和明确错误依据的reject才为fail；混合意见或未解决疑点为revise并标记root_review_required。主助手实际读取来源和提案后另记复核，标记本身不自动调用主助手，也不自动放行。独立调用不是独立专家验证，逐字锚点不证明解释正确。
4. **最终排序**：复用 `novelty_policy` 两种模式及现有权重。balanced = 新颖性40% + 结构20% + GNN20% + critic20%；novelty_first仍要求新颖性门控。当前runtime的critic取三个视角最新分数的中位数（有回应取回应，无回应取初评）；旧接口adaptive_review=False仍保留最低分聚合。来源HOLD不因中位分或GNN高分通过。初始三个独立意见构成新颖性投票，已知结论否决和0.6质量阈值不变。`ranked_ids` / `selection_rank`表示已评审可入选项的顺序，全池和所有原评分仍保留。

未评审不等于淘汰；条件冲突的held链也单独保留。正常预算不足时不开始下一条，格式不合规最多补问一次，传输失败不盲目重发。后续候选失败会保留已完成结果及当前部分评审。改写已入选假设会使旧评审失效，既有最多3次重评限制不变。

后处理区分已有两条前提和拟检验的整链增量；增量尚未证实本身不是拒绝理由。基线体积差异不能当作已观察到的萎缩速率，分层显著性不能单独证明交互，图谱中的空字段不能抹去摘要已给出的事实。来源未覆盖只能写“所给来源未证实”，不能写成全域不存在或新颖性证明。

### 分数含义与缺失

GNN复用 `GNNEdgeScorer`（R-GCN/GraphSAGE/GAT标准checkpoint）、`FrozenGraphSAGEScorer`（历史两层GraphSAGE冻结导出）或 `FrozenRGCNScorer`（本次全量R-GCN冻结导出），共用 `hypothesis_path.score_candidate`：只按实际ID/关系词表逐边推理，取几何均值，最弱边小于0.3时乘既有0.7惩罚。不生成新实体嵌入、不猜测映射、不将ComplEx当作GNN。输出每条边的实际分数、模型输入哈希、覆盖率与缺失原因。

结构分复用既有权重：confidence^0.20 × traceability^0.20 × graph_only_novelty^0.25 × testability^0.35。confidence来自绑定记录；traceability只表示保留的论文及文本锚点，不代表来源核查；graph_only_novelty是原始索引内两端节点的来源稀疏度；testability是本次裁决的两个可执行性标志均通过时为1，否则0。组件按原有0.01下限参与结构乘积；任一组件缺失则整个结构分为null。

缺失的GNN/结构分保持null；固定权重的部分效用中缺项贡献为0，但不输出“实测0分”，不重调权重。`score_complete=false`、`unmeasured_scores`和`score_scope`说明限制。GNN分衡量已知边的模型相容性；图内稀疏、模型评审及入选都不证明整链机制、全域新颖性或科学真实性。

### GNN配置

运行环境变量 `NEUROCLAW_IDEA_GNN_MANIFEST` 可覆盖项目默认`neurooracle/configs/idea_gnn.json`。相对路径相对于配置文件解析。当前默认类型为`graphsage_export`，绑定既有`model/inference_manifest.json`的SHA-256；加载时核对权重、实体/关系导出、词表、TRAINED记录和checkpoint中保存的原始训练身份，缓存命中也重新核对文件。

训练图谱版本`bbccd816…`与当前评分目标`487179e2…`分别报告，`trained_on_current_graph=false`；使用已导出向量，不重算消息图、不改旧权重的版本标签。DistMult的正反向分数对称，因此该分数不能用作方向验证。历史执行锁在磁盘清理后发生过修订，仍标记`frozen_execution_audit_revalidated=false`；本次绑定检查没有重新授予历史科学结论或完整审计通过。

标准GNN checkpoint也可用下述配置（默认kind为`gnn_checkpoint`）：

```json
{
  "graph_revision": "当前图谱SHA-256",
  "training_scope": "实际训练来源、范围与限制；不是质量认证",
  "checkpoint": {"path": "checkpoint.pt", "sha256": "checkpoint文件SHA-256"},
  "message_graph": {"path": "message_graph.json", "sha256": "消息图文件SHA-256"}
}
```

消息图是与权重词表匹配的 `[[source_id, relation, target_id], ...]`；checkpoint使用现有GNN适配器格式，含state_dict/config/entity_to_id/relation_to_id。加载只做CPU推理，验证版本及文件哈希；缓存命中仍检查绑定。配置声明与哈希不能证明训练科学有效性。不得为了通过加载检查给旧权重虚填当前图谱版本。

首次接线时缺少GNN和可用评审凭据的记录保留为历史状态。现在已实跑上方来源细化、真实三方评审及固定候选三种GNN对照；本轮单独复用已有Ollama凭据，没有改全局客户端配置。没有最终可推荐项，不能声称idea质量已经提高。

### 验证记录

全量R-GCN补充验收：`tmp/idea_full_gnn_20260929_final/acceptance_v2/README.md`。新增逐关系纳入/排除/划分计数、全部原始边与训练行及来源的对应记录、可执行的断点恢复入口；25项针对性测试、真实检查点重载和独立进程的小图恢复通过。正式权重没有重训，同一935候选再次实测724条，与原报告一致。新验证入口使用`--output`保存独立报告，并按实际训练源码快照检查版本，原始报告不覆盖。技术检查不授予科学排序增益或三方真实评审完成的结论。

`tmp/idea_ranking_20260929/`保留检查脚本、测试报告和真实池预排序。合成测试覆盖真实mini-loader、全池保留/分页、GNN改变排序、OOV缺分、checkpoint绑定、两轮否决、越界引用、预算中断、传输失败、候选改写失效及真实runtime接线；R-GCN小图测试只做前向推理，不训练，不计科学效果。

首次真实池检查复用问题2已有MCI135、ADHD500、APOE300条草稿及303条held，不重新生成、不调用模型。`tmp/idea_ranking_20260929/verified/RESULTS.json`记录每池输入哈希、来源统计、抽样重新绑定、耗时及4,829个绑定输入只读检查；当时真实GNN覆盖数、已评审数、实测最终分均为0，保留该历史结果，不覆盖。

最终回归：311项流程/策略/证据测试加16项GNN接口测试，合计327通过。GNN接口测试在本机先导入pyarrow再运行，避开首次混合测试的native导入诊断；没有修改环境。真实loader330.289秒，三个非空池预排序与首条证据复核各2.708–6.582秒。输入stat和campaign字节均未变；真实检查记录的运行代码哈希复核一致。

旧模型复用记录：`tmp/idea_gnn_reuse_20260929/verified/RESULTS.json`，输入为上轮完全相同且哈希通过的935条真实草稿，无新增图谱查询/来源审阅。MCI125/135、ADHD277/500、APOE233/300可完整测量，合计635；均与冻结解码及路径聚合公式独立复算一致，300条OOV为null，held303不混入评分。载入与绑定检查5.12秒，三个池评分/复算各0.021–0.028秒。分数范围分别0.445–0.611、0.375–0.595、0.350–0.620；不是校准概率。模型文件哈希和stat未变，训练/模型API调用0。189项相关回归通过，包括新增GraphSAGE checkpoint、冻结导出与原模型前向一致、只读/OOV、文件篡改和旧/新图谱版本分离检查。
