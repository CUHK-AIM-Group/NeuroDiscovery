# Hindcasting 结果图 — 绘图代码与思路交接文档

> 用途：交给 Codex（可读图）接手后续改图/换数据工作。本文覆盖「图长什么样、代码在哪、数据怎么流、设计思路是什么、当前状态」。

---

## 1. 图长什么样（一句话）

一张 15.8 in 宽的 manuscript 风格大图，分两部分：

- **Panel a–i**：9 个「Topic」（代表性预测任务），每个 Topic 占一行两列——
  - 左列：**发现曲线**（discovery curve），x 轴是评估的假设数 K（10→1000），y 轴是「未来被证实的命中数」（future-supported hits），三条方法曲线 + 同色阴影带。
  - 右列：**lift 柱状图**，x 轴是 freeze year（2016 / 2020 两个窗口），y 轴是「相对随机基线的 lift」（k=100），三组柱子 + 误差棒 + 抖动散点，虚线 y=1 表示随机基线。
- **Panel k**：整行宽度的**代表性假设卡片**。5 张 semantic-v2 可追溯卡片显示 PMID；暂缺的 Topic 2、5、6、9 复用上一版内容。9 张卡片在图面上使用完全相同的背景、边框和 JIF 格式。

三种方法（颜色固定，全项目统一）：

| 方法 | 颜色 | 标记 | 图例标签 |
|---|---|---|---|
| SciAgents | `#F28E2B` 橙 | `^` | SciAgents |
| OpenScholar-RAG | `#4C78A8` 蓝 | `s` | OpenScholar（图内缩短，避免挡曲线） |
| NeuroDiscovery | `#D9544D` 红 | `o` | NeuroDiscovery |

红色 `#D9544D` 全项目唯一留给 NeuroDiscovery（我们的方法，图里的「主角」）。

---

## 2. 脚本与指标数据的职责

| 脚本 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `core/scripts/generate_hindcasting_v5_design_data.py` | 生成**占位设计数据**（DESIGN PREVIEW） | 硬编码的设计参数（代码常量） | 目录树：`<method>/seed_XX/<task>/kg<Y>_to_<A>_<B>/hindcasting/{metrics.json, recovered_examples.csv}` + `design_data_provenance.json` |
| `core/scripts/plot_case3_taskwise_hindcasting_figure.py` | **主绘图脚本**（读数据 → 汇总 → 画图 → 导出） | 上面的 metrics.json + recovered_examples.csv | PNG(450dpi)/PDF/SVG/TIFF + `section_svgs/` + 6 个 source CSV + `hindcasting_figure_provenance.json` |
| `core/scripts/verify_hindcasting_v5_design_figure.py` | **无人工校验**（确定性断言，不画图） | 绘图脚本产出的 source CSV + PNG | 控制台 PASS/FAIL |
| `materials/hindcasting_journal_impact_factors_2025.csv` | 出版社/JCR 核验的 **2025 JIF 与来源** | 期刊名、JIF、指标年份、来源 URL、核验日期 | panel k 的年份限定 JIF + provenance |
| `materials/hindcasting_verified_representative_examples_semantic_v2_20260812.csv` | **真实卡片快照**（5 张） | 清理前 semantic-v2 导出 + PubMed 核验的 PMID/DOI | panel k；缺失的 4 个 Topic 保持未验证状态 |
| `materials/hindcasting_provisional_representative_examples_20260827.csv` | **临时卡片快照**（4 张） | 上一版 Topic 2/5/6/9 内容 | 仅填补空位；状态只保留在源 CSV/provenance 中 |

**数据流**：设计数据生成器 → 绘图脚本 → 图 + 源数据 CSV → 校验脚本。

---

## 3. 绘图脚本内部逻辑（`plot_case3_taskwise_hindcasting_figure.py`）

### 3.1 数据读取与汇总

- `collect_metrics()`：glob `*/seed_*/*/kg*_to_*_*/hindcasting/metrics.json`，解析 `topk` 里的 `observed` 和 `random_same_hypothesis_pool`。
- `select_balanced_test_set()`：只保留 9 个选定 Topic × 3 方法 × 共同 freeze year × 共同 seed × 7 个 K 的**平衡测试集**（三方法必须共享同一批窗口和 seed，否则报错）。
- `summarize_curves()`：曲线 = 每个 seed 内对共同窗口求和 `primary_hits`，再对 seed 取均值 + bootstrap 95% CI（5000 次，rng 固定 seed）。
- `summarize_freeze_lifts()`：lift = `primary_hits / random_mean_primary_hits`（k=100），同样 bootstrap。
- `summarize_lead_times()`：lead time = freeze year 到未来支撑年份的间隔（只统计 rank≤100 且 primary_hit 的假设）。
- `select_representative_examples()`：每 Topic 选 NeuroDiscovery top-100 内**最早出现的 strict hit**（rank 最小，composite score 只用于 rank 并列）；**JIF 不参与选卡**，只在选定期刊之后查表附加。
- `load_verified_representative_examples()`：用 `--representative-examples` 载入可追溯卡片快照；`--provisional-examples` 可选填补缺失 Topic，来源状态保留在导出的 source CSV 和 provenance 中。

### 3.2 布局（gridspec）

```python
fig = plt.figure(figsize=(15.8, 13.6))
grid = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.0, 1.0, 2.4],
                        left=0.125, right=0.900, top=0.940, bottom=0.045,
                        wspace=0.16, hspace=0.50)
```

- 前 3 行 = 9 个 Topic（每 Topic 占 2 列：curve + lift），第 4 行 = panel k（`grid[3,:].subgridspec(1,1)`）。
- panel id（a–i、k）用 `position_panel_labels()` 按「整个面板块」（axes + 刻度 + 轴标题 + 标题）的 tightbbox 计算位置，放在块左上角、右对齐。

### 3.3 关键绘制函数

- `draw_task_hit_curve()`：曲线 + marker + `fill_between` 阴影带（同色 alpha 0.15，带 = per-seed min/max）。x 刻度 `["10","","50","","200","","1,000"]`。a–i 不再各放图例；全图唯一共享图例放在**整张图顶部中央的带框固定区**。y 轴标签只在每行最左 panel 显示。
- `draw_task_freeze_bars()`：三组柱（width 0.23，`alpha=0.7`，白边 0.55），误差棒 `#333333`，柱上叠 per-seed 抖动散点（markersize 1.5）。**没有 y 轴标签**（用户要求删掉 "Lift" 单词）。虚线 `y=1`。
- `draw_examples_table()`：panel k 保留 3 列 × 3 行卡位；所有卡片统一使用浅灰底和灰色边框，JIF 统一写成 `JIF 2025: x.x`。不显示临时状态说明行、星号或特殊配色。

### 3.4 导出

- PNG 450 dpi、PDF（`pdf.fonttype=42` 保证字体可嵌入）、SVG、TIFF（600 dpi LZW），全部 `bbox_inches="tight"`。
- `save_section_svgs()`：把每个 panel 单独裁成 SVG（供论文分图用）。
- 同时写出 6 个 source CSV（曲线/曲线按 seed/lift/lead time/代表性卡片/平衡原始数据）+ provenance JSON。

---

## 4. 绘图规范（`materials/figure_style_spec.md` 要点）

- 画布宽固定 **15.8 in**，绘图主体宽 **12.24 in**（left 0.125 → right 0.900）。
- 字体全图 **Times New Roman**，两档字号：**14 pt**（标题/轴标题/panel id 粗体）、**12 pt**（刻度/图例）。
- 柱状图 y 轴从 0 开始；柱子 `alpha=0.7` + 白边 `linewidth=0.55`；误差棒 `#333333` 不透明。
- 网格只留 y 向，`#E7E9EB` 浅灰，置于柱子下层（`set_axisbelow(True)`）。
- 文字全黑 `#000000`，背景白。
- panel id 放面板块左上角、纵轴标题上方（本图用 `position_panel_labels` 自动算）。
- 方法色见上表，红色只给 NeuroDiscovery。

---

## 5. 设计数据思路（占位数据，非真实结果）

真实 v2 五窗口实验还没跑完，所以用 `generate_hindcasting_v5_design_data.py` 造了一套**自包含的占位数据**，让图能先给 advisor 看设计。核心「故事」：

1. **NeuroDiscovery 每个 Topic、每个 K 都领先**（明显 SOTA）。
2. **OpenScholar-RAG 清晰可见**（不是死值），SciAgents 有真实但较小的结果。
3. **2016 和 2020 两个窗口明显不同**：`lift_year_factor` 让 ND 2016×1.45 / 2020×0.875，baseline 2016×0.80 / 2020×1.25（早期知识少→ND 相对 lift 大；2020 知识库变大→baseline 变强、ND lift 缩小）。
4. **baseline 方法互有胜负**：`connectome_behavior` 和 `differential_diagnosis` 两个任务里 SciAgents 反超 OpenScholar-RAG（图推理 vs 平面检索），其余任务 OpenScholar-RAG > SciAgents。
5. **曲线形状有区分度**：4 个 family（early/steady/late/sat），9 个 panel 形状两两不同，避免「看起来都一样」。
6. **ND lead time 最长**（≤4 年）、top-100 命中数最大（封顶 100）。
7. 设计生成器中的 `support_journal` 只是版面占位，**不得用于 panel k 或 JIF**；绘图脚本已增加防误用检查。

数据生成器是**自包含**的：不依赖已删除的 semantic-v2 模板，目录结构、metrics schema、CSV 表头都从代码常量重建。`run_self_check()` 会在生成后断言「ND 在 9 任务 × 2 窗口 × K=1000 都领先」。

---

## 6. 当前状态与换回真实数据的步骤

- **a–i 当前仍是占位设计数据**；panel k 为 5 张真实卡片 + 4 张上一版卡片。来源差异只在 source CSV/provenance 中记录，不能把这张图表述成完整实验结果。
- 当前修订版输出目录：`neurooracle/data/experiments/hindcasting/hindcasting_v5_design_preview_20260827_uniform_cards_single_row_legend/`。
- 真实 v1（2019/2020，540 runs）已跑完；**v2 五窗口（2016–2020）尚未跑完**。
- 换回真实数据时，**绘图脚本不用改**，只需：

```bash
python core/scripts/plot_case3_taskwise_hindcasting_figure.py \
  --neurodiscovery-root <真实 ND eval root> \
  --baseline-root <真实 frozen-baseline root> \
  --representative-examples <真实 representative card CSV；可省略以从本次 recovered_examples 按 rank 选择> \
  --provisional-examples <可选；只填补缺失 Topic 的临时卡片 CSV> \
  --output-root <新输出目录> \
  --file-prefix <新文件名前缀>
```

---

## 7. 给 Codex 的注意事项（踩过的坑）

1. **JIF 必须有指标年份和来源**：当前采用 2026 年发布、反映 2025 数据的 JCR/出版社页面；数据在 `materials/hindcasting_journal_impact_factors_2025.csv`，图内写成 `JIF 2025`，provenance 保存来源 URL 和核验日期。JIF 只能在卡片选定后查表，不能参与选卡。
2. **a–i 共用一个图例**：不要再把图例放进 panel b；唯一共享图例固定在全图顶部中央，三种方法保持严格单行，不设第二行标题。
3. **Topic 标题位置**：标题用 figure coordinates 放在每组 curve + lift axes 上方并左对齐，避免被相邻 axes patch 截断；panel id 位置仍由 title bbox 参与自动计算。
4. **lift 柱状图不要 y 轴标签**（删掉 "Lift" 单词）。
5. **x 轴刻度用 `1,000` 不是 `1k`**。
6. **panel k 卡片**：y_positions 已上移 0.07（`0.685, 0.3775, 0.070`），hypothesis 行距 `linespacing=1.15`（1.35 会跟 evidence 重叠）。
7. **导出 PDF 到桌面**：沙盒里 `cp` 到桌面不可信，用 PowerShell `Copy-Item` + md5 校验；字体只允许 Times New Roman 三变体。
8. **改图范围只限用户当前指定的图**，不主动审计其他绘图脚本。
