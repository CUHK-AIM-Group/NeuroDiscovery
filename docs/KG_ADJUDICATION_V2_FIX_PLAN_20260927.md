# 修复计划执行报告：抽取提示、范围门、来源一致性与裁决层（2026-09-27）

## 结论

计划已按四层落地实现并通过测试，但**尚未授权全库**。修复目标是让 API 能承担候选抽取、把合并限制在可证明的范围差异之内，并让每个结论都带源绑定证据。本轮没有重跑 200 篇、没有写固定层/生产图/heldout/E:。

关键判断得到数据支持：把旧提示换成 v2 提示后，**模型对 7 个存疑对全部判 related_to，0 次越权合并**。此前 25 个错误 equivalent 全部来自我们的范围和提示规则，不是模型能力不足。

## 1. 规则层：范围、方向与空结果

`neurooracle/src/proposition_scope_v2.py`

- 只有**同一关系家族**、**方向可交换（或非对称关系方向一致）**、**无硬性范围冲突**时才允许进入合并判定。
- 硬阻断只用于可靠差异：物种、疾病阶段（闭集）、以及两侧各自指明**互斥疾病/年龄段/遗传风险**（如精神分裂症 vs 临床前痴呆、儿童 vs 成人）。
- 度量、解剖、人群措辞等自由文本差异为**软差异**，不静默合并也不静默拒绝，必须由模型给出理由。
- 缺失范围、范围不可比、以及“不显著 vs 断言”一律**不得直接等价**；不显著只是未证实，不是反证。
- 队列/样本/站点/模态/校正方法记为 variant，**不阻断跨队列复现**，避免把真正的多论文支持也拒掉。

效果：旧 25 个自动 equivalent 现在 **21 个 related、4 个 distinct、0 个放行**；全部被阻止，无越权合并。

## 2. 数据传递层：证据包不再丢条件

`neurooracle/scripts/paper_adjudicate_v2.py`

裁决包现在包含完整源摘要、role、result、limitations、scope_reason、publication_types/notices、source_sha256，而不再只有三元组和引句。上一轮的教训是：模型其实在 `limitations` 里保留了“未通过 FDR”，是我们的裁决包把它丢了。现在该信息随包进入判定。

## 3. 程序校验层：来源绑定与账本

`neurooracle/src/pilot_validation_v2.py`

- 有符号方向（positive/negative）必须有源文的有符号词；只有 “correlated/associated” 时判为 `*_not_source_bound`。
- 声称 `reported_significant` 必须有显著性措辞，否则报错。
- 已知缺陷 PMID:28138428 第 8 条被正确拦下，而合法的 P<0.05 不被误伤。

`neurooracle/src/candidate_ledger_v2.py`

- 逐作业挑选合格尝试，拒绝**失败前驱覆盖成功结果**的旧加载方式；重复候选、digest 不匹配、paper_id 不符都单独记录而不是静默接受。

`neurooracle/scripts/paper_extract_v2.py`

- 复用已验证的传输（Ollama 优先、仅在确认额度耗尽时回退 NVIDIA、逐次记账、未知结果 HELD 不盲发、密钥只用槽位号）。
- 换用修正提示 `EXTRACTION_SYSTEM_V2`（哈希 `b7d0ddbdd5d102bb…`，取代 `ce781805b306e69a…`）：方向必须源绑定、不显著不等于否定、疾病/人群/物种/阶段/对照是必要差异必须保留。
- 抽取后跑一致性门；违反者记为 `HELD_CONSISTENCY`，`candidate_ledger_v2` 不会把它拿去合并，原始响应保留不重写。

## 4. 验证层：两套样本

- 开发回归：旧 25 个 equivalent + 已知缺陷观测，全部按预期被拦。
- 新验证：仍需另行构建含**真实等价正例、困难负例、待定例**的样本；当前 24/25 对只作回归，不能算未见验证。

## 实测（复用既有 200 篇产物，未重抽）

| 项目 | 结果 |
|---|---|
| 旧 25 个自动 equivalent → v2 门 | 21 related / 4 distinct / 0 放行 |
| 200 篇 114 个候选对 → v2 门（旧候选） | 106 related / 7 需模型判定 / 1 unresolved |
| 7 个存疑对 → v2 模型裁决 | 7/7 related_to，0 次越权，全部逐次记账 |
| 2 篇 v2 抽取冒烟 | 1 接受、1 因方向未源绑定被一致性门拦下 |

模型裁决 7 对的耗时（仅成功记录）：约 2.3–8.3 秒/对；账本 `run/ADJUDICATION_V2_BOUNDED.sqlite`。

## 未做与仍需授权

- 未重跑 200 篇抽取（旧候选带旧提示缺陷，仅用于诊断门是否生效）。
- 未写入固定层、生产图、语料、heldout、E:；未删除任何文件。
- 未构建新验证集、未做全库投放；这些需要具体的有界授权。

## 复现

```powershell
.venv/Scripts/python.exe neurooracle/scripts/paper_adjudicate_v2.py --pilot-dir tmp/kg_pilot_20260927_v2 --ledger tmp/kg_pilot_20260927_v2/run/PILOT_ollama200.sqlite --packets tmp/kg_pilot_20260927_v2/SUBSET_PACKETS.jsonl
.venv/Scripts/python.exe neurooracle/scripts/paper_extract_v2.py --pilot-dir tmp/kg_pilot_20260927_v2 --run-tag v2dry --papers 2 --dry-run
.venv/Scripts/python.exe -m unittest neurooracle.tests.test_proposition_scope_v2 neurooracle.tests.test_pilot_validation_v2 neurooracle.tests.test_candidate_ledger_v2 neurooracle.tests.test_paper_adjudicate_v2 -q
```

29 项 v2 测试通过（另有 11 项范围门 + 7 项一致性测试在前述集合内）。`test_kg_systematic_consolidation` 因环境缺 `openai` 模块无法导入，属既有环境问题，与本次改动无关。
