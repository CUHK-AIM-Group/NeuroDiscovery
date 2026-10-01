# 来源先行10篇对照：抽取可用作候选，尚不能放手合并

用户“可以继续”后执行。只处理已经冻结的10篇；Ollama `deepseek-v4.1-flash:cloud` / think high，API仅收到原始来源的HTML/空白规范化视图和既有v2提示，未收到人工答案。无裁决调用、全库运行、生产发布、固定层/CORPUS/heldout/E:修改或删除。

## 1. 先核实参考边界

通过Europe PMC core元数据检查两对候选涉及的四篇：15780850、15038994、19714565均未列出开放全文，24828364列出JAMA PDF但本轮未阅读全文。这只是有限查找，不代表不存在其他公开版本。病程、用药、测量/调整兼容性没有补齐，**S01/S02不升格为严格等价正例**。

因此本轮可以评价“摘要抽取是否忠实”，不能评价严格跨论文合并召回。原参考保持哈希 `07ea7b1b95442882ed7df7220ef96e1c03b6713e290bc708004975fa77f8ffeb` 不变。摘要参考本身有复合单元，不能充当完整原子命题金标准。

## 2. 执行与速度

`tmp/kg_source_first_20260927/CONTINUATION_PROTOCOL.json`在本批输出前记录边界；`run_blind_comparison.py`执行。使用独立的小型首响应包装器复用既有传输/解析/校验，不改变主流程：

- 固定10篇，单worker；每篇收到第一个HTTP200响应后停止，不因质量不佳换key重抽；未知结果不重发。
- Ollama优先，仅所有可用Ollama额度明确耗尽才允许NVIDIA。认证错误停用slot，不当成额度耗尽；普通限流或服务错误HOLD。
- 发送前持久化intent，发送后保存原始body和outcome；已有运行目录拒绝再次启动。留下intent+outcome是一对收据，不是待重发任务。
- **10次请求、10次HTTP200、10篇各1个响应、全部finish_reason=stop、全部Ollama，0次NVIDIA，0次重试。**
- 平均 **17.31秒/篇**，中位15.62秒，最大29.42秒；顺序运行总计173.23秒。prompt 16,104 tokens，completion 47,245 tokens；不据此推断全库吞吐。
- 输出57条观察：50 primary_result、4 synthesis_result、1 method、1 hypothesis、1 background。
- 技术检查8篇通过、2篇HOLD；这不是80%科学正确率。

## 3. 人工对照结果

root检查了10篇候选的statement/conditions/result/role/quotes，部分proposition/statistics；不是所有字段的独立双人复核。逐篇映射和问题见 `ROOT_COMPARISON_REVIEW.json`，源/候选/body哈希绑定及技术统计见 `BLIND_COMPARISON_SUMMARY.json`。

53个人工参考单元都有内容对应，但有多对一/一对多：例如API一条双侧结果对应参考左右两条；三诊断一条对应三个疾病单元。**不能报告53/53准确率或原子召回100%。** 60个模型引文均唯一逐字命中规范化摘要，但仍发现语义问题。

### 3.1 确证的问题并不只是选样

| 例子 | 实际问题 | 当前门控 |
| --- | --- | --- |
| 19714565：各自人群内部的基因型与体积变化关系 | 患者观察的comparator填健康人，健康人观察填患者；把人群内关联写成潜在组间对照 | 未捕获此错误；该篇因另一个显著性问题HOLD |
| 19714565：genotype-by-diagnosis interaction | 没有p或significant字样却填reported_significant；baseline又填进疾病stage | 显著性错误被捕获 |
| 15038994：对照显著激活、患者不显著 | proposition写成group_difference / higher / comparator=patients / reported_significant；两组各自显著性不同不等于组间差异显著 | **TECHNICAL_PASS** |
| 24828364：三个疾病组 | 文字保留三诊断，但结构population压成psychotic disorders，没有分别建立疾病限定结果 | **TECHNICAL_PASS** |
| 31712617：OPC与RG两个PRS | 两种暴露没有拆开；反而把相邻句的关联和方向拆成两个观察；subject还是patient，实际PRS暴露塞进measurement | **TECHNICAL_PASS** |
| 11378309：controls | 摘要未说明healthy，模型自动补healthy controls | **TECHNICAL_PASS** |

还有较弱/需要进一步核实的问题：把unchanged/no association写成not_significant、把来源未说明的分析人群填为combined、PRS误入adjustment、短引文只保留数值而遗漏解剖/比较条件。这些分开标记，不把所有与root写法不同之处一律算模型错误。

### 3.2 同时存在误拦截

10963985的原始摘要是 `P=0. 003`。模型保留了原文raw并填数值0.003，校验器正则却匹配成0，产生 `p_value_or_operator_changed`。这是可定位的解析缺陷，不是模型捏造p值。

本轮不现场修改规则重评以追求高通过率，原始状态保留。改解析需单独回归测试，同时保留“不明确数字不能静默修复”的边界。

## 4. 根因与下一步

**现在有直接证据：即使换成较连贯的样本，结构化表示仍会丢失/填充比较对象、混用人群/暴露/测量、合并不同疾病结果，或把一句结果按语句切成不同节点。** 所以原来的碎片化问题不能只靠语义embedding或放松门控解决。与此同时，模型能快速抓住多数摘要结果的内容，角色区分和相关方向也有正确案例；无需据此全盘否定模型。

建议下一轮先做离线修复，不扩大API批次：

1. **收紧数据契约而非放宽等价。** 明确study population、exposure/predictor、outcome、comparison、time是不同字段；stage不能填baseline；unknown不补常识。
2. **定义原子化单位。** 不同疾病/暴露拆开；同一对变量的效应方向、显著性、相邻来源句合成一个结果，保留多锚点。参考的grouped单元也要另立版本细化，不能只要求模型原子化。
3. **补条件支持校验。** 检查比较关系和显著性绑定到哪一组/哪个测试，不再仅检查关键词在引文中出现。加入“显著与不显著不等于组间显著差异”的负例。
4. **修p值空白解析并冻结回归。** 本10篇已经曝光，今后只能作开发回归；修复后的泛化效果需再找未曝光、来源先行的新样本验证，不能在这10篇上调好后声称盲测通过。
5. **严格正例仍要补证。** 两个暂定同义对保持HOLD，补全文或寻找范围明确的来源；合并召回仍为N/A。不为提高多论文率降低等价标准。

**目前适合让API生成待审候选，不适合无人审核直接写共享命题。** 此结论来自具体缺陷，不是把8/10技术通过或53单元映射当成科学准确率。

## 5. 留存与验证

首响应运行文件463,639字节；参考及本轮材料在写本报告前合计728,423字节逻辑大小，未测物理占用。只保留小来源包、原始响应、候选、人工复核、绑定与脚本，不复制KG或数据库。清理追加清单见 `CONTINUATION_CLEANUP.json`；本轮不删除证据。人工阅读速度未可靠计时。

离线4项路由检查通过；原参考10个源绑定及57个root锚点复验通过；汇总验证每篇恰好一个body、模型文件哈希未变、所有映射ID有效。旧 `ADJUDICATION_V2.json` 仍为967147字节、mtime 2026-09-27 09:01:30。代码/出处检查不等于独立科学验证。
