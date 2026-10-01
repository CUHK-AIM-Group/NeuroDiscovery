# v2 验证集：源优先 root 裁决、门禁与有界模型对比（2026-09-27）

## 结论（先说要点）

1. **门禁层是安全的，但会压掉真实的合并。** 在 53 对上，离线门基线 **0 次误并**（42 个非合并对全部正确保留），但它**一个真实合并也放不出来**：10 个 root 认为可合并的对，全部被降级（8 个 `model_must_justify` 走模型、4 个被硬阻断）。
2. **模型层没有过度合并，但也没有召回。** 22 个可判定对全部真实调用（Ollama 19 + 重试 3），**0 次把非合并对提升为合并**，包括 12 个困难负例；同时 **0 次判 `equivalent`**，只给出 2 次 `narrower_than`。也就是说，当前管线在验证集上的多论文召回 ≈ **0（严格）/ 20%（把嵌套视为合并）**。
3. **瓶颈是“把同命题的不同粒度/不同对照当成了不同命题”，而不是模型能力。** root 的 10 个正例里，4 个被 `disease_stage`/`comparator` 硬阻断，其余被软差异推给模型后模型一律选了 `related_to`。
4. **重抽样本（v2 提示）质量可用，但接受率不是 100%。** 16 篇有界重抽：12 篇 `CANDIDATE_READY`，4 篇 held（2 篇一致性门，2 篇结构门）。一致性门触发 2 次，都是**保守但可辩护**的判定，没有误伤合法结论。

> 重要限制：root 参考**不是独立金标准**。抽取提示与范围门都是 root 自己写的，所以“root 正例”本身带确认偏置；这里量化的是**管线相对 root 的一致性**，不是全库准确率，也不构成发布信用。

## 验证集构成

`tmp/kg_pilot_20260927_v2/run/VALIDATION_CANDIDATES.json`，共 **53 对 / 41 篇唯一论文**：

| 类别 | 数量 | 说明 |
|---|---|---|
| `shared_condition_candidate` | 41 | 共享具体疾病 + 共享解剖/度量家族（召回池） |
| `hard_negative_disjoint_condition` | 12 | 措辞相近但疾病互斥（防误并） |

配对来源是“较宽松的源绑定阻塞”，比旧的字面端点召回宽得多；它只提出候选，不判定等价。

## Root 源优先参考

产物 `tmp/kg_pilot_20260927_v2/ROOT_VALIDATION_REFERENCE.json`（由 `neurooracle/scripts/build_root_validation_reference.py` 生成，53 对全部覆盖，无缺项）。

每对先读 `side_a/side_b.source_abstract` 与 quotes，再落裁决；门禁提示不作为答案。裁决口径：

- **可合并** = 两篇论文确实共享**同一个命题**：`equivalent`（完全一致）或严格嵌套的 `narrower_than`/`broader_than`（一方是子群/单侧）。
- **不可合并** = `related_to`（同主题、不同测量/构念/对照）、`distinct`（不同结构/构念/相反极性）、`unresolved`（证据不足）。
- 同一命题的“断言 vs 不显著”算**同一命题节点的反证**，必须保留为 dispute，**不得合并成正例**。

结果分布：

| root 裁决 | 数量 | 可合并 |
|---|---|---|
| `equivalent` | 5 | 是（精确） |
| `narrower_than` | 5 | 是（嵌套） |
| `related_to` | 23 | 否 |
| `distinct` | 16 | 否 |
| `unresolved` | 4 | 否 |
| **合计** | **53** | **10 个可合并** |

附带标记（保留科学含义，避免裁决丢失信息）：`secondary_source`（综述/荟萃，非独立原始复现）、`background_role`（把命题当前提）、`same_node_dispute`（同命题相反结果）、`atrophy_volume_inversion` / `direction_retyped`（同一关系被反写或方向编码错误，文本仍一致）、`scope_narrower` / `scope_variant`（粒度/队列措辞差异，不改变命题）。

## 门禁对比（离线基线）

产物 `run/VALIDATION_GATE.json`（`--validation`）、评分 `run/VALIDATION_SCORE_GATE.json`。

| 指标 | 结果 |
|---|---|
| 误并（非合并对被放行/合并） | **0 / 42** |
| 正确保留 | 42 / 42 |
| 遗漏合并（root 可合并但门禁降级） | **10 / 10** |
| 困难负例误并 | **0 / 12** |

10 个遗漏中：**4 个被硬阻断**（V001、V014、V016、V021），6 个进入 `model_must_justify`。硬阻断的槽位统计：`disease_stage` 3 次（慢性 vs 首发、早期 vs MCI、早期 vs 前驱）、`comparator` 1 次（创伤暴露对照措辞）。

## 有界模型对比（真实调用）

产物 `run/VALIDATION_MODEL.json` + `run/VALIDATION_MODEL_RETRY.json`，账本 `run/ADJUDICATION_V2_VALIDATION.sqlite` / `..._RETRY.sqlite`。22 个非硬阻断对全部真实调用（Ollama 优先，`think: high`）。

| 指标 | 结果 |
|---|---|
| 成功裁决 | 22 / 22（19 + 3 重试，0 未知结果丢弃） |
| 逐次记账 | 25 行，全部含 provider/slot/耗时/verdict |
| 成功耗时 | 均值 **3.3 s/对**（1.5–7.8 s） |
| 模型合并（`equivalent`） | **0** |
| 模型提升非合并对 | **0** |
| 困难负例误并 | **0 / 12** |
| 严格召回（只认 `equivalent`） | **0 / 10** |
| 含嵌套召回 | **2 / 10** |
| 含嵌套精确率 | **2 / 2 = 1.0** |

可判定的 6 个 root 正例（V006、V007、V009、V011、V029、V035）里，模型给 V011/V029 判 `narrower_than`，其余 4 个判 `related_to`。模型从不越权，但也从不合并。

## 有界重抽样本（v2 提示 + 一致性门）

`neurooracle/scripts/paper_extract_v2.py --papers 16`，账本 `run/PILOT_v2val16.sqlite`，产物 `run/v2val16/`。16 篇全部来自 CORPUS，heldout 未触碰。

| 结果 | 数量 |
|---|---|
| `CANDIDATE_READY` | 12 |
| `HELD_CONSISTENCY` | 2 |
| 结构门 held | 2 |

一致性门两次触发均为**保守但可辩护**：

- `PMID:40489111`：把 HR < 1 编码成 `direction=lower` + `reported_significant`，但摘要里显著性措辞只出现在**交互项**上，HR 本身没有逐项显著性词。门按“有符号声明必须源绑定”拦下，属于可辩护的保守。
- `PMID:24557771`：把“moderate to strong correlations”编码成 `direction=positive`。摘要只有 `r = .51-.73`（本就全正），但无 “positive” 字样，门按“有符号方向必须源绑定”拦下。

两例都**不是**误伤合法结论（如显式 `P < .05` 的合法声明不被拦），也未改写原始候选，原始响应完整保留可审计。

## 失败模式与修复方向（需另行授权）

1. **`disease_stage` 硬阻断过强。** “慢性 vs 首发”同属精神分裂症、“早期 vs MCI”同属 MCI 谱系，属于**同一疾病的阶段限定**而非不同人群；当前被升级为硬冲突。建议：仅当两侧阶段**互斥且分属不同疾病**时硬阻断，同疾病阶段差异降为软差异。
2. **`comparator` 硬阻断误伤对照措辞。** “创伤暴露对照 vs 含/不含创伤暴露对照”是同一对照类型的不同措辞。建议：对照措辞接近时降为变体（variant），仅当对照是**不同测量维度**（如对侧半球 vs 健康对照）时才作差异。
3. **模型对“同命题不同粒度”一律保守。** 提示把 `narrower_than` 定义为“严格更窄范围”，但模型更倾向 `related_to`。建议：在提示中明确“子群/单侧/子区属于同一命题的嵌套范围，应判 `narrower_than` 而非 `related_to`”，并给出正例。
4. **同命题 dispute 处理。** V012/V017/V030/V038 模型判 `related_to` 而非 `unresolved`，尚可，但需显式规定：同命题的“断言 vs 不显著”必须落到**同一 claim 节点 + 反证**，绝不能变成正合并。

## 产物与复现

新增脚本：

- `neurooracle/scripts/build_root_validation_reference.py` — root 源优先参考（53 对）。
- `neurooracle/scripts/score_validation_v2.py` — 门禁/模型 vs root 评分（误并、漏并、召回、精确率、可达性）。
- `neurooracle/scripts/paper_adjudicate_v2.py` 新增 `--validation` / `--labels`，可在冻结验证集上复用同一提示与证据包。

```powershell
# 门禁基线（离线）
.venv/Scripts/python.exe neurooracle/scripts/paper_adjudicate_v2.py --pilot-dir tmp/kg_pilot_20260927_v2 --ledger tmp/kg_pilot_20260927_v2/run/PILOT_ollama200.sqlite --packets tmp/kg_pilot_20260927_v2/SUBSET_PACKETS.jsonl --validation tmp/kg_pilot_20260927_v2/run/VALIDATION_CANDIDATES.json --out VALIDATION_GATE.json
# 有界模型裁决
.venv/Scripts/python.exe neurooracle/scripts/paper_adjudicate_v2.py --pilot-dir tmp/kg_pilot_20260927_v2 --ledger tmp/kg_pilot_20260927_v2/run/PILOT_ollama200.sqlite --packets tmp/kg_pilot_20260927_v2/SUBSET_PACKETS.jsonl --validation tmp/kg_pilot_20260927_v2/run/VALIDATION_CANDIDATES.json --run-model --keys "$env:USERPROFILE\Downloads\keys.txt" --out VALIDATION_MODEL.json --out-dir tmp/kg_pilot_20260927_v2/run/adjudication_v2_validation --ledger-out tmp/kg_pilot_20260927_v2/run/ADJUDICATION_V2_VALIDATION.sqlite
# 评分
.venv/Scripts/python.exe neurooracle/scripts/score_validation_v2.py --gate-file VALIDATION_GATE.json --model-file VALIDATION_MODEL.json VALIDATION_MODEL_RETRY.json --out VALIDATION_SCORE.json
```

38 项 v2 测试通过（29 原有 + 9 新增评分/口径测试）。本轮未重跑 200 篇抽取、未重跑旧 303 裁决、未覆盖旧产物；无生产/固定层/语料/heldout/E: 写入；无删除。

## 顺带修复的实现缺陷（真实缺陷，非实验产物）

`paper_adjudicate_v2.py` 的账本 `persist()` 之前把 verdict 字典直接写入 TEXT 列，真实模型调用时会在成功路径抛出 `sqlite3.ProgrammingError` 并令整个裁决中断。已改为写入 verdict 名，并令 `invalid_verdict` 分支保留原始响应用于审计（之前只留 gate 兜底、原始响应丢失）。该缺陷此前只被离线/缓存路径掩盖。

另有两处既有缺陷（非本轮引入）：

- 某一篇候选（`PMID:38976036`）含未声明的限定槽 `adjustment_note`，`shared_proposition_registry.normalize_qualifiers` 会 `ValueError`，使**整个**标准路径裁决中断。已在 `paper_adjudicate_v2.proposition_identity` 做防御：已声明槽走注册表校验，未声明槽以独立键参与 seen-set，既不再崩溃、也不会把仅在该槽不同的两条观测静默合并。修复后标准路径（200 篇账本）召回 349 对 / 88 篇，此前记录的 114 对是 `proposition_scope_v2` 概念方向容错（9:10 加入）**之前**的数值；未覆盖 `run/ADJUDICATION_V2.json`（mtime 仍为 9:01:30）。
- 已删除的根读者草稿（`run/_*.txt`、`run/_*.py`）共 55 个文件、约 317.0 KB，均为可再生成的阅读辅助，已记入 `CLEANUP_MANIFEST.json`。
