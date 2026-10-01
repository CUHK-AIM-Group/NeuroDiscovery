# Idea topic 检索

输入 topic，返回相关命题、按论文去重的 `studies`、来源证据和 metadata。复用 `AcceptedClaimLayer` 的当前接受版本；查询不修改图谱，不生成假设，也不执行评分。

## 使用

```powershell
& C:/Users/45846/anaconda3/envs/neuroclaw/python.exe -B -m core.topic_evidence --topic "海马萎缩与轻度认知障碍" --minimum-coverage 0.5 --json
```

网页服务提供 `GET /api/kg/topic-evidence?topic=Hippocampal%20atrophy%20in%20MCI&minimum_coverage=0.5`。服务进程复用已加载的图谱证据；每次重新启动 CLI 都有首次加载成本。

`--term`（HTTP 的可重复 `term`）补充检索词；自动中英文别名仅覆盖一小组常用词和缩写。`--limit` 控制最多返回多少命题/原始观察，`--evidence-claims` 控制展开前多少条的论文证据，默认分别为 20 和 10；共享与单篇结果共用预算。`--minimum-coverage` 默认 0.5，是命中查询概念的比例，不是科学相关概率；同一个概念的多个别名只算一次。设为 1 可要求全部查询概念命中。`--minimum-papers` 沿用现有接口的“经核查支持该命题的论文数”定义；单篇候选展开后才有确切计数，正数过滤会排除未展开/核查计数不足的候选，可能少于请求数量。`--output` 可保存 JSON，拒绝覆盖已有文件。

## 返回内容与边界

- `studies`：按图谱已记录的 work 身份去重；含 PMID/DOI、完整 bibliography、出版版本与状态、来源身份、相关命题及逐项 observation。逐论文检查查询覆盖率及显式疾病词，`missing_terms` 标明部分匹配；其他论文只保留在命题上下文中，不算 topic 命中的论文。
- observation：原有 evidence、population、conditions、完整原记录 metadata、source_review、source_anchor、抽取文本及来源类别。没有来源锚点时明确为图谱抽取文本；缺失字段保持缺失，不补造。
- `claims`：命题匹配词、查询覆盖率、已有论文计数及证据展开状态。`retrieval_origin` 区分共享目录与原始单篇观察，`query_id` 可直接交给现有 `CLM:`/`REL:` 证据接口。先按检索覆盖排序，再沿用已知论文计数；未展开单篇的论文/支持计数为 null。论文列表另按自身记录命中词和来源可用性排序。
- `claim_total`：共享目录条目数加未被目录覆盖的原始观察数；`shared_claim_total` / `shared_claim_matched` 单列共享部分，`original_retrieval` 列出全量索引数、排除重复后的范围、命中观察及来源数。命中数在单篇展开后的支持论文过滤之前统计，不是论文总数。
- `graph_revision` / `claim_layer_revision` / `original_index_revision`：搜索和展开使用同一版本；即使空结果或不展开证据，也提供版本。文件或版本中途变化时整次查询失败，不返回部分证据。

当前范围为已发布共享命题目录及其接受的扩展，加上其绑定的既有全量原始 claim 索引。单篇检索扫描索引已有的概念端点、关系、PMID/DOI，排除已在共享目录中的观察，每个原始来源保留一条最佳匹配进入统一预算。没有新建索引或合并命题；**原始索引不含全文与完整摘要，也不等于全库文献搜索**。未配置该索引的图谱仍可搜索共享目录，并明确报告这个缺口。空结果不能证明没有相关研究；命中不等于证据支持某个新假设，更不证明创新性。

单篇证据通过既有 `query_batch` 返回，使用索引中的字节位置读取完整原记录，再核对独立 census 的 ID、hash、单篇成员范围及关系身份。复用原有论文身份、来源审阅和 evidence 聚合函数，缓存只限当前版本；原始 metadata、不完整抽取文本、未核查状态均保留。

历史验收将应用代码也记录为审计文件。读取器现将仅位于 `review_inputs` 的三个准确路径（`server.py`、`claim_layer_v7.py`、`claim_layer_support_v1.py`）视为本次修改的运行代码，报告 `historical_code_drift`，不要求当前应用字节等于旧代码。图谱、projection、来源材料、其他审计文件以及这三个路径若被用作来源输入时的校验继续执行。原验收文件不变；这项修复不意味着旧验收在当前代码上被重新完成。

首次加载仍校验完整原始 XML 文件；补充证据缓存只保留本层实际使用的完整论文，复用原解析器，随后逐项核对与已封存 observation 完全一致，避免保留同一 XML 中大量无关论文。

## 验证

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
