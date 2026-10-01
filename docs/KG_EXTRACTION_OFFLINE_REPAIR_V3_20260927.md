# 论文抽取离线修正 v3

2026-09-27，用户授权“可以修正”。本轮只修代码、契约与离线回归；没有API、子代理、任务、重抽取、删除、生产/固定层/CORPUS/heldout/E:写入。

## 已落地

### 1. p值解析修复，接入现有结构校验

`neurooracle/src/source_statistics.py::parse_p_literal` 支持原始 `P=0. 003` 的数值0.003、Unicode空白、`.005`、科学计数法和不等号。原始raw、引文和历史候选不改写。错误数值0、错误运算符仍不通过；区间、多p值、模糊断字和越界值不猜成单个标量。

`paper_batch_model.validate_candidate()` 已使用该解析器。旧10963985第4条观察的 `p_value_or_operator_changed` 在离线复核中消失；其它9篇的原结构错误列表不变。**不把旧HELD账本自动改成READY。** 老pilot的validator_limitations兼容层仍存在，不据本修复声称所有旧宽免都已科学验证。

### 2. 新的来源绑定契约，不改变旧版提示哈希

`neurooracle/src/paper_extraction_contract_v3.py` 提供追加式v3提示，保留原v2提示与旧产物。生成的完整提示已冻结在 `tmp/kg_source_first_repairs_v3_20260927/PROMPT_V3.json`，**尚未接入真实发送器**。

每个primary_result新增structured_result：

- population、exposure/predictor、outcome、comparator、timepoint、disease_stage分别存放。population不代替实际基因型/PRS暴露；baseline不是stage；不把controls补成healthy。
- 源字段保存原文短语及quote_index/start/end绑定，未知值带原因；canonical概念另存于proposition，不用“必须逐字”限制实体规范化。
- contrast绑定具体比较句；test中的显著性必须指向同一contrast，统计依据绑定到该比较句内部，不能借另一个结果的p值。
- 两组分别显著/不显著不推出组间显著；unchanged/no association不自动改写成明确报告的not_significant；CI推导与明确报告分开。
- 不同疾病/风险评分拆开，同一结果的相邻来源句与方向/显著性合并；复合测量不能凭空拆分。不能确定时atomicity=unresolved，不发布。

### 3. 可执行离线校验器

`neurooracle/src/pilot_validation_v3.py` 检查源锚点、字段绑定、未知原因、时间/病程混用、比较与显著性绑定、组合字段、重复结果身份；旧输出缺v3结构时明确标记缺失，**不自动转换旧答案**。

入口：

```powershell
.venv/Scripts/python.exe -B neurooracle/scripts/check_paper_extraction_v3.py --candidate candidate.json --source source.json --out new_report.json
```

source必须是该候选实际收到的来源视图，包含paper_id/abstract或packet.paper_id/packet.abstract。入口核对身份、记录哈希，只能新建报告，不能覆盖原文件；状态最多CONTRACT_CHECK_ONLY，绝不获得生产或独立科学验证信用。

**局限：** 原文位置一致不等于语义蕴含。同一句也可能包含多个测试；仅凭包含关系无法完全证明哪个p值属于哪个结果。组合字段的and/or检查会拦住合法复合名词，属于人工复核信号，不自动拆词。新契约没有证明解决所有比较对象误归属、语义去重或跨论文等价，仍需来源审阅及未曝光新样本测试。

## 旧10篇回归，不是新盲测

执行 `neurooracle/scripts/audit_source_first_repairs_v3.py --out-dir tmp/kg_source_first_repairs_v3_20260927`，只读原始首响应及来源；每个候选/body哈希保持一致。没有best-of-attempts挑选、重新调用模型或升格账本。

- 共11条诊断提示：7条“null报告不等于明确统计显著性”、2条baseline误作stage、2条组内显著性需要比较关系复核。
- 15038994观察4现在能被离线诊断提示为组内显著性/组间差异风险。
- **11条不是11个已证实错误。** 另一个显著性提示落在15780850观察0：PBP而非NPBP相对HC的结果本身可能合法，保守模式需要人工确认它没有编码成PBP-NPBP比较。保留这一边界例，不为减少HOLD而在开发集上特判放行。
- 原10篇是v2输出，没有structured_result，v3契约缺失属于版本不兼容，不能计为模型“新失败率”。

## 验证与留存

- 151项测试通过：新v3、结构校验、旧pilot/v2、来源门控、加载器、裁决、评分、模拟传输、防重发测试；不代表科学准确率。
- 原10篇来源绑定与57个root锚点复验通过，冻结参考哈希仍为 `07ea7b1b95442882ed7df7220ef96e1c03b6713e290bc708004975fa77f8ffeb`。
- 新回归/提示/清理清单共21,723字节逻辑大小；没有复制响应、KG或数据库，没有删除旧证据。
- 新v3发送实验尚未进行，不宣称模型已经改善。下一轮应先在新的来源先行样本上测契约可用性与实际错误，再决定是否接入自动抽取；严格等价标准未放宽。
