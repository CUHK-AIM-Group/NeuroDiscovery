# 图谱：正式输入、安装与语义合同

正式运行使用已验收 v2；v3 筛选是独立 staging 工作，不能据此替换正式图谱。
当前筛选入口仍为 [native scope](KG_V3_NATIVE_SCOPE_20261008.md)。

历史过程与旧版原文统一保存在 [说明文档历史归档](../dev/archive/docs-history_20261019.zip)，不再另建逐次进度稿。


## 正式图谱及实验输入

## Current graph inputs and version-pinned experiments

Ordinary graph readers now resolve the current **accepted publication**, not a
date-named directory or the most recently modified JSON file:

1. Use the campaign configured by `NEUROCLAW_KG_CAMPAIGN` or
   `neurooracle/configs/graph_explorer.json`. Check its completed publication,
   acceptance receipt, graph binding and file fingerprint.
2. If no campaign is configured, use the canonical graph recorded in
   `neurooracle/data/full_v2/CURRENT_STATE.json`.
3. A configured but unavailable, changing or unaccepted publication is an error.
   There is no silent fallback to `full_snapshot_v1`, `full_snapshot_v2`, a small
   display export, or an empty graph.

During a graph rebuild reset, `neurooracle/configs/graph_explorer.json` may be
absent intentionally. In that state there is no active accepted evidence
campaign, so the default graph loader uses the canonical `full_v2/CURRENT_STATE.json`
entry above; the evidence web layer remains unavailable until a new accepted
campaign is published. The old campaign pointer is retained only in the cleanup
manifest and is never used as the formal graph.

Inspect the resolved input without loading the large graph:

```bash
python -m neurooracle.current_graph
```

Normal Python callers use `load_graph()`; hypothesis CLI and query commands
resolve the same default. An explicit `--graph /path/to/frozen/graph.json` (or an
explicit path to `load_graph`) keeps experiments pinned to their original input.
This change does not republish graph data or relabel historical results.

### Writes and model assets

- `save_graph` requires an explicit output path. Ingestion templates, marker
  ingestion and optimization commands require a separate `--output`.
- Fresh batch-extraction builds use `data/build_artifacts/claim_extraction`, or
  their explicitly supplied output directory. Creating a new build is separate
  from selecting a published graph for research.
- KGE plausibility requires an explicitly supplied checkpoint. Selecting a new
  graph does not make an old graph's KGE checkpoint compatible. This change does
  not train models or certify user-supplied checkpoint compatibility.
- Shell runners accept `NEUROCLAW_GRAPH_PATH` for an explicit frozen input and
  `NEUROCLAW_KGE_CHECKPOINT` for the corresponding model. Otherwise their graph
  input follows the current publication. Set `NEUROCLAW_NOVELTY_CACHE` separately
  if an existing literature-query cache should be reused.

Legacy abstract caches, original extracted claims and marker/hypothesis exports
are **source or experiment assets**, not the published graph selector. Their
historical audit references are deliberately retained until a separately
verified migration or cleanup. Frozen implementations and completed experiment
manifests must not be rewritten to point to a newer graph.

The existing accepted-claim web interface already follows the campaign and its
accepted evidence layer. The legacy packaged display export is a separate UI
artifact, not a substitute for the current research graph. Rebuild desktop
runtime bundles after source changes; do not patch sealed experiment copies.


## 安装与版本核验

## 图谱与证据的统一安装

概念图、Claims & evidence 和图谱驱动的 AutoResearch 共同选择一个版本。默认使用正式发布的图谱，不按文件修改时间选择新实验结果，也不依赖已清理的旧研发 campaign。当前正式版本由 `neurooracle/data/full_v2/CURRENT_STATE.json` 指定。

正式图谱的 claim 节点已保存论文书目、提取文本、统计字段及 Case Study 标记。`core/formal_graph_evidence.py` 直接读取这些原始记录；单独的 `extracted_claims.jsonl` 审计库不再是浏览这些记录的安装前提。原始记录不等于新近核验的论文身份、获审命题支持或多论文共识，界面和研究检索保留这个区别。

### 安装选择

默认安装入口为 `<backend-root>/neurooracle/data/graph-installation.json`。它指向同一目录中的 `graph-packages/<manifest-sha256>/manifest.json`，使用相对路径。

可以用 `NEUROCLAW_GRAPH_INSTALLATION` 指定其他位置的安装入口或包目录。移动时保留入口和整个 `graph-packages` 目录的相对结构即可。包内的原始来源路径是不可变溯源信息，实际读取通过清单解析，不要求目标机器具有原研发目录。

选择顺序为：已选安装包、显式 `NEUROCLAW_KG_CAMPAIGN` 实验覆盖、本地正式 CURRENT_STATE、历史配置或演示下载。显式实验覆盖保留原有校验规则；默认正式版本存在时，历史 `graph_explorer.json` 不会覆盖它。选定正式文件、安装包或显式 campaign 损坏时会报告错误，不悄悄回退到旧图。v3 的筛选和暂存产物不参与正式版本选择。

正式来源可以是原始 CURRENT_STATE，也可以是公开快照原有的 `neurodiscovery.graph-publication.v1` manifest。两者按各自原始字段绑定实际 graph 的字节数和 SHA-256，不补造已删除的 campaign 或验收记录。包使用 `source_kind: formal_current`，包含原字节的正式清单和 graph；旧的完整验收 campaign 包仍受原证据读取器校验。

普通关键词搜索名称、谓词和论文书目，不匹配随机技术 ID 片段；完整 claim 或端点 ID 仍支持精确查询。

首次检索会在系统临时目录的 `neurodiscovery-kg-cache` 中建立 SQLite 缓存，保存节点字节位置与检索字段。缓存绑定 graph SHA-256、实际 claim 数和缓存文件 SHA-256；返回的完整观察仍从原 graph 读取。缓存不会写入 full_v2，也不是新的科研审核。`minimum_papers > 0` 仍表示获审支持论文门槛，不能用原始来源记录填充获审支持计数。

### 从正式版本制作包

```powershell
python -m core.graph_installation export --state 'neurooracle/data/full_v2/CURRENT_STATE.json' --output '<new-formal-package-directory>' --release-id 'full_v2_20260906_bbccd816de32'
python -m core.graph_installation verify '<new-formal-package-directory>/manifest.json'
```

图谱与正式清单按原字节复制。安装后的相对位置替代原开发机器的绝对文件位置，图谱身份仍由原清单中的内容 hash 决定。该导出不新增声明、不重做科学审核、不更改正式选择器。

### 从已验收来源制作包

在具有完整验收数据的维护环境执行，输出目录必须尚不存在：

```powershell
python -m core.graph_installation export --campaign '<accepted-campaign.json>' --output '<new-package-directory>' --release-id '<release-id>'
python -m core.graph_installation verify '<new-package-directory>/manifest.json'
```

导出器先使用现有 AcceptedClaimLayer 检查来源，再递归收集活跃文件中的完整指纹依赖。原始 graph、campaign、验收记录、索引和来源文件按原字节复制；不改写旧 JSON 内的路径、不补造审核记录、不修改正式图谱。缺失、变化或冲突的依赖会阻止导出。

包清单保存原文件的路径、字节数、mtime 和 SHA-256，并为其分配包内相对文件路径。安装后的物理时间戳可以不同，源指纹身份仍需匹配，并校验实际内容 SHA-256。该映射仅在一次包读取的上下文中生效，普通研发和冻结实验的存储解析不受影响。

### 制作下载文件

```powershell
python -m core.graph_installation pack '<package-directory>/manifest.json' --output '<new-graph-package.zip>' --remote-path 'neurooracle/data/releases/<release-id>/graph-package.zip'
```

ZIP 使用 Zip64，支持大于 4 GiB 的文件。命令同时生成 `<archive>.release.json`，绑定 archive 大小、archive SHA-256 和 package manifest SHA-256。重复命令不会覆盖旧 archive。

客户端下载协议读取 Hugging Face Space 内的 `neurooracle/data/releases/desktop/latest.json`。发布时，该文件应使用生成的 release descriptor 内容，其 archive 路径指向同仓库中实际上传的包。没有实际完整包时不能先发布一个不可用入口。

若该入口返回 404，下载器读取已经公开的 `neurooracle/data/releases/full_v2_20260906_bbccd816de32/manifest.json`，下载其完整 `knowledge_graph.json.zst`。安装前核对压缩文件大小与 SHA-256，解压时再次核对完整 graph 的大小与 SHA-256，并保留原 publication manifest 字节。无需为原始声明浏览另行发布旧研发依赖。

只有两个正式发布入口都不存在时才保留历史演示下载兼容路径；这种情况下明确显示证据未安装。已经安装的完整包不会被演示下载覆盖。公开正式快照约 693 MB，解压后的 graph 为 10,278,527,239 字节；首次内容核验和建立检索缓存需要额外磁盘读取与缓存空间。

### 本地安装

```powershell
python -m core.graph_installation install '<graph-package.zip>' --data-directory '<backend-root>/neurooracle/data'
```

安装先在新临时目录解包，拒绝越界路径、重复目标和符号链接，验证全部文件及现有证据验收，再原子切换安装入口。下载 hash 或证据检查失败时保留旧入口。后端后续请求会识别新包，不需要通过重发研究任务来刷新数据。

也可以将包安装到其他目录，然后让 `NEUROCLAW_GRAPH_INSTALLATION` 指向该目录中的 `graph-installation.json`。这不会改变正式 `full_v2/CURRENT_STATE.json` 或 benchmark 的固定输入。

### 接口和显示

- `/api/neurooracle/graph/status` 提供实际选择的 graph path、包 release、graph revision 和 evidence 配置状态。
- `/api/kg/*` 证据接口、`core.topic_evidence.default_layer()` 和 CLI/MCP 的 `neurooracle.src.graph_paths` 默认解析使用同一安装选择。显式 `--graph` 冻结输入不改绑，缺失时仍报错。正式版本返回 `source: formal_current_graph`、`evidence_kind: original_extraction`；来源审核状态保持未审核，不能将 Case Study 归属审核改作命题支持审核。
- 正式版本的声明页默认显示全部原始提取记录，保留查看来源书目、原字段、统计及导出的能力；原有已审核共享主张包仍按获审支持条件显示。
- 缺少证据数据时返回 HTTP 503 和 `code: evidence_not_installed`，界面说明缺失数据，不提供无效的重复读取提示。
- 已配置但损坏、更新中或校验失败的证据仍按原规则拒绝读取；不会把未审核的概念图记录提升成已审核证据。

### 验证边界

回归测试同时使用合成正式图和既有验收绑定图，覆盖过期配置、换目录、中文路径、原来源目录不可用、Unicode 与流式边界、原文和计数一致、概念图真实加载、研究检索、压缩及解压 hash、下载切换和失败保留旧版本。

合成测试包不能作为真实科研数据发布。2026-10-19 已额外用本地正式 graph 完成全文件 SHA-256 检查和全部 905,274 条 claim 的检索索引构建，并实际查询原始声明；记录见 `CLIENT_GRAPH_INSTALLATION_REPAIR_20261019.md`。这属于读取与检索完整性验证，不是新的命题支持或论文科学准确性验收，也不代表完整 AutoResearch 已进行模型实跑。大型图谱首次 hash 检查和加载会进行较多磁盘读取；后端将相关读取放入工作线程，避免阻塞事件循环。


## 论文抽取合同

**从论文源材料抽取的合同**

v3 契约明确 population/exposure/outcome/comparator/timepoint/disease_stage 与逐字段来源绑定、原子化和比较特定的显著性。实现见 [paper_extraction_contract_v3.py](../neurooracle/src/paper_extraction_contract_v3.py)，离线校验入口见 [check_paper_extraction_v3.py](../neurooracle/scripts/check_paper_extraction_v3.py)。这些契约校验不放宽严格等价标准，也不代表真实发送器已切换或旧产物已迁移。

每个抽取包绑定具体论文发布记录、作品身份状态、源快照 SHA256、材料类型（摘要/全文）和正文。抽取输入不包含旧 claim、旧共享键、旧审核答案或旧投产计数。论文身份、同作品版本与复用队列分别处理。

| 对象 | 必须保存的字段 |
|---|---|
| 论文与发布记录 | work identity/status，PMID/DOI/PMCID，标题、作者、期刊、年份，文章类型、预印本/正式版、更正/撤稿、来源及全部原始书目字段 |
| 源材料 | snapshot SHA256、原文、材料类型、语言、获取记录、原文件路径/偏移、许可与可用性；摘要不能标成全文 |
| 原子观察 | observation ID、原句及字符/段落/表格位置、主语、测量、关系语义、宾语、具体结果；复合句拆分的来源链接 |
| 共享命题 | statement、规范实体引用及测量身份、relation kind、必要命题限定；reuse/create/related/unresolved 决定与依据 |
| 证据连接 | claim/observation/work 引用、supports/opposes/partial/contextual/unresolved、概括或对应关系的理由；是否已审核 |
| 研究条件 | 物种、疾病、人群、阶段、年龄/性别、组织/脑区/体液、模态/测量/任务/量表、干预、剂量、对照、时间、调整变量 |
| 数值结果 | 样本量及单位、效应量及单位/类型、CI、p 值运算符及数值/校正、方向、阴性/不确定结果、模型；原文数值与解析结果并存 |
| 证据角色 | primary_result / synthesis_result / background / hypothesis / protocol / method；依据具体段落，不只看整篇论文类型 |
| 独立性与审计 | study/cohort/experiment 引用、重叠状态（允许未知）、抽取/审核记录、全部原始输出与 extras、字段缺失原因 |

共享命题主键不包含论文 ID、标题、作者、年份、样本量、p 值或整句原文。原子观察与共享命题是不同对象：观察保留具体研究事实，命题负责组织跨论文证据。

抽取规则：

1. 阅读源材料后生成观察和命题；禁止把旧三元组重新命名当成重抽取。
2. 不把蛋白自动替换成其编码基因，不把模态替换成特定指标，不猜脑图谱、研究队列或人群。
3. 区分因果、关联、预测、组间差异和机制解释；保留模态词和不确定性，不把改善统一写成治疗有效。
4. 不同方向或未显著结果是否反对共享命题，取决于该命题的含义与比较范围；不得只按 p 值自动判定。
5. 背景陈述、方案、假说可以保留在图中，但不能计作该论文的新实验支持。
6. 对既有固定实体仅创建引用；未能准确对齐的候选概念留在论文层，不修改固定层名称或映射。
7. 一个作品在同一命题中只计一次；多个结果、版本、缓存、队列不变成多篇论文。研究独立性单独记录。
8. 所有保存记录必须有原文锚点或明确的材料不足状态；字段缺失保留 null 与原因，不填零、不杜撰。

抽取完成之后，先验证来源锚点、对象完整性和 metadata 保留，再做语义归组与科学抽查。程序通过 JSON 校验不等于命题得到科学支持。


## Case Study 范围规则

## KG v3 Case Study membership policy

### Status

- Canonical rubric: `neurooracle/src/CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md`
- Rubric epoch: `2026-08-10.peer17.v2`
- Formal Case Study registry: 17 ordered, non-exclusive IDs from
  `neurooracle.src.case_study_scope.CASE_STUDY_IDS`
- New-claim seal: `case_study_membership_contract.v4`
- Prior extraction seal: `case_study_membership_contract.v3` (preserved and
  still verifiable)
- Historical full-v2 seal: `case_study_reaudit_contract.v1` (preserved)

`general` is implicit shared-corpus membership. `hindcasting` is a validation
protocol. Neither value is a Case Study ID.

### One policy, two consumers

`neurooracle.src.case_study_membership_policy` parses the canonical rubric and
provides its version, SHA-256, ordered definitions, mandatory gates, prompt
block, and strict decision validator. Both standalone re-audit and new claim
extraction import this module. Do not copy the definitions into another prompt,
script, notebook, or manual top-up builder.

Changing the rubric file or the ordered registry changes its hash and therefore
starts a new, explicitly versioned audit epoch. Never overwrite a finalized
decision in place.

Case 2 follows the same evidence-contribution principle as Case 1. A Case 1
paper may contribute one disease-specific neural result without itself being
transdiagnostic. Likewise, a Case 2 claim may contribute either the
genetic/pathway-to-neural link or the baseline-neural-to-later-outcome link;
the paper need not contain the complete mediation chain.

### v3 expansion lifecycle

All Case Study search campaigns use the frozen publication window 1980–2026.

1. Search broadly for relevant papers. Query provenance is candidate-generation
   evidence only and never assigns Case Study membership.
2. Deduplicate candidates against both the formal KG and every staged queue by
   PMID, DOI, PMCID, arXiv/OpenAlex ID, then normalized title and year.
3. Retrieve the abstract/full text and retain its source identity.
4. Extract source-linked claims. Every returned claim must also contain:
   `case_study_ids`, all nine `case_study_gates`, verbatim
   `scope_evidence_spans`, `scope_confidence`, and
   `scope_decision_basis`.
5. Validate all IDs and gates against the frozen policy. Unknown IDs, duplicate
   IDs, missing gates, string booleans, labels with a false mandatory gate,
   ungrounded evidence spans, or missing decision reasons fail the response and
   trigger extraction retry.
   Case 2 is then canonicalized deterministically: every validated
   `imaging_genetics`, `progression_prediction`, or `prognosis` claim also
   receives `case2_pathway_mediation`; a standalone Case 2 label is invalid.
6. Seal the accepted decision with the claim evidence hash, source-context
   hash, rubric hash, ordered-registry hash, labels, gates, confidence, and
   decision basis. The serialized record is stored in `scope_reaudit` with
   `review_status=final_complete`.
7. Stage claims and run entity resolution/deduplication. At actual KG mutation,
   call `ingest_claims(..., require_final_scope_audit=True)`. A missing or
   mismatched seal refuses the whole batch before graph mutation.
8. Recompute paper membership deterministically as the ordered union of all
   retained claim memberships from that paper.

### Reuse and correction

A finalized seal is canonical for its evidence and policy hashes. A model,
reasoning-effort, concurrency, or session change is not a reason to review it
again. Identical evidence reuses the stored decision.

Model inference before sealing is not mathematically deterministic, and a
sealed decision may still contain a semantic error. A correction must be an
explicit audited replacement with its reason and superseded decision hash; it
must not silently mutate the old decision.

### Compatibility boundary

The completed full-v2 re-audit remains valid under its historical v1 contract,
and claims extracted under the strict same-paper Case 2 rule retain their v3
seals. New v3-expansion claims use the component-based v4 contract. Existing
claims are not silently relabelled during collection; after KG v3 collection is
complete, one planned full-graph re-audit will project all retained claims to
the current rubric epoch.
