# 专家研究与专家研究（扩展）：v1.0.0 统一风格、滚轮修复、阅读结构与 v6 面板

日期：2026-09-18 至 2026-09-19。客户端版本维持 **1.0.0**。最终分发为 **study-client-v1.0.0-20260919-v5** 目录（此前的 -20260919-v4、-20260919-v3、-20260919、-20260919-v2、-20260918 各版为其前身）。

## 2026-09-19 下午追加：Human Evaluation 改名与扩展题库 v2

- **品牌**：主研究改为 **Human Evaluation 1 · COMPLETE RESEARCH OUTPUT**；扩展研究改为 **Human Evaluation 2 · HYPOTHESIS RANKING**（中文界面显示"Human Evaluation 2 · 假设排序"）。客户端菜单为 Human Evaluation 1（专家研究）/ Human Evaluation 2（扩展）；顶栏视图名、iframe 标题、解锁页、欢迎弹窗与设置重置文案同步更名。
- **扩展题库换到最新结果**：经核查，原 120 对题库是 2026-07-30 的旧外部验证策展，已非最新。新题库 `case1_tcp_expert_study_v2.json`（SHA256 `ff9b1d51…`）改用最新全空间（case_study_closed_loop_v8，426,555 候选）的内部实验结果：每对 = 一个内部验证通过 + 一个未通过（`validated`：家族 FDR q≤0.05、|调整 Cohen d|≥0.15、留出方向 AUC≥0.55 且方向一致率≥0.7），且前者的冻结 ND 打分显著更高（差距 ≥0.20，实际最小 0.629——旧库仅 0.03）。240 对、480 个假设、难度按打分差三分位各 80、主持预览 6 会话 × 40 道均衡。假设方向自由表述（v8 注册空间无预定方向，观测方向只进组织者真值文件 `private_truth/`，不打入客户端）。文献面板：v1 人工复核卡片在假设重现时逐字复用（9 张），其余由本地知识图谱记录自动匹配并如实标注未人工复核。
- **扩展分配表 v2** `case1_tcp_expert_pair_assignments_v2.json`（SHA256 `a37c6e8c…`）：每题 3 人复评，720 人次 ÷ 10 位专家 = 每人 72 道 = 6 次会话 × 12 道（每次约 10 分钟，单人约一小时，与主研究对齐）；按难度分层轮流确定性发牌，同题 3 个实例必落在不同专家，每人每次会话恰 4 易/4 中/4 难。服务端按分配表逐会话过滤，会话记录分配编号。v1 表与 v1 库保留用于旧会话结果回看。
- **入口介绍改写**：两个研究入口都先简介 NeuroDiscovery 系统（自动提出假设并在真实数据上验证），再说明为什么需要人工评估，再说明本次材料与请求；删除专家研究欢迎页的"数据保存与评价边界"折叠块和材料页的"完整解释边界 · 必读"块（同意勾选文案改为"我自愿参与本次评审，并且可以随时停止"以避免引用已删除的说明）。
- **注意**：今天 13:55 前后发现本机 `~/.neurodiscovery/discovery-expert-study/` 目录被移除（其中均为历次界面验证的测试会话，尚无真实专家数据；客户端日志无删除记录，应用重启后该目录会按需自动重建）。`user-studies` 库未受影响。

（以下各节为同日早些时候的变更与验证记录，不再重复。）

## 本次变更

**两个专家研究的统一**

- “假设排序”在客户端菜单、顶栏视图名、iframe 标题和页面内部文案统一更名为 **专家研究（扩展）** / Expert Study (Extension)；源码与打包产物中均不再出现旧名。
- 扩展页（`/study`）与专家研究页（`/discovery-study`）共用 `workspace-tokens.css` 与 `study-workspace.css`：扩展页按 `data-study-ui="legacy"` 整套换肤为专家研究的扁平视觉语言，仅保留排序协议本身的左右对比工作区结构；页面内“关闭”按钮与专家研究对齐为“← 返回工作台 / Back to workbench”。
- 两个研究入口都收集 **名字/代号 + 相关研究年限 + 自愿参与勾选**；导出均为 v2 信封（研究 / 参与者含 id 与经验年限 / 会话 / 计时 / 汇总 / 逐题记录）。扩展页设置面板修复了内容垂直居中造成的大片顶部空白（`margin: 0 auto`）与窄窗口下文字无右边距的问题（ intro 文本块上限 620px ）。
- **滚轮修复**：嵌入模式下 Chromium/Electron 可能把滚轮手势路由到 overflow 被有意隐藏的固定 iframe 视口，导致进入具体案例后各栏无法滚动。`study-workspace.js` 在捕获阶段接管滚轮：沿祖先链找到第一个真正能滚动的容器手动滚动（材料栏、评分栏、目录栏、扩展页文献列表等），触底链式传递，无处可滚时收容 overscroll；Ctrl/Meta 缩放与纯横向手势不受影响。独立打开时不挂接该逻辑。
- **目录栏宽度可调**：专家研究进入案例后，左侧材料目录栏右缘可拖拽调宽（120–400px），支持键盘方向键与双击复位，宽度保存在本机；窄屏堆叠布局下手柄自动隐藏。
- **材料四段式阅读结构**（单轮流程；历史两阶段分支不变）：①这个发现是什么 ②它从何而来 · 相关研究与前序实验的启发（含"前序假设实验与真实反馈 · 自动迭代"小节）③实验设计 · 模型、数据与指标 ④实验结果 · 内部与外部验证。右侧六题顺序一致：新颖性 → 证据整合 → 反馈驱动迭代 → 实验设计 → 实验验证 → 科学信息增量，编号 01–06 连续；答案按题目 id 存储，不受显示顺序影响。

**v6 专家面板（34 份）与评审分配**

- 面板从 10 份扩到 **34 份**：全部选自外部验证通过（pragmatic_triage_pass）且效果最好的案例（旧批 11 + r7 批 27 共 38 条，按最小方向标准化效应降序取前 34）；详见 `CS1_V6_PANEL_HANDOFF_20260918.md`。3 张保留卡与 v5 逐字节一致，4 张首轮卡展示专门的"无父假设"反馈说明。
- **评审分配表** `cs1_discovery_assignments_v1.json`（SHA256 `92f134507a08700fa8229af7edb976079460972906a7d556062849b4a1f73085`）：rank 1–32 各 3 位评审、rank 33–34 各 2 位（共 100 人次；覆盖优先级随效应排名，已向用户披露此设计），确定性轮流发牌给 P01–P10，每人恰好 10 份互不重复。入口表单新增必选"分配编号"下拉（全部 34 份 · 主持预览 / P01–P10 各 10 份），选定后简介数量联动；服务端按分配过滤会话卡片，导出只含分配卡，分配编号记入会话与创建事件。
- 名字/代号校验与表单对齐：2–40 位字母、数字或中文（入口本已允许"张三"类填写，服务端此前误拒，已修正）。
- 运行时缓存键升级为 `1.0.0-study-v5`，旧缓存目录保留不删除。
- **专家研究（扩展）对齐**：移除指向专家研究的超链接；不再让用户选择"是否显示生成器打分"（固定不显示，手动条件）；120 道策展比较题按难度分层轮流**分成 10 份**（P01–P10，每位 12 道 = 4 易/4 中/4 难，每题恰好归入一份，分配表绑定候选库哈希 `57dbdea4…`；主持预览保留完整 6 次会话 × 20 道），入口有必选"分配编号"下拉，会话题量与说明文案随份额联动（1 次会话 · 12 道）；会话记录分配编号。
- **名字/代号两边通用**：两个研究入口共享同一个本机保存的名字/代号（一处填写，另一处自动带入）；首次打开都默认本机账户名。
- **导出格式对齐**：专家研究的导出现在与扩展研究共用同一信封（`schema_version: neurodiscovery-expert-study-results-v2`）：`study`（包与协议版本）、`participant`（id / experience_years / consent）、`session`（含 assignment_id）、`timing`（含 active_answering_time 双格式）、`summary`（已答/总数/无法判断及原因/备注数）、`question_results`（逐题行：题号、card_id、question_id、能力维度、答案值与标签、是否无法判断；题目顺序与客户端单轮显示一致）。完整审计载荷（session 全量状态、cards、events、score_summary 等）原样保留；文件名统一为 `NeuroDiscovery-user-study-{代号}-{时间戳}.json`（服务端导出与浏览器下载一致）。历史两阶段包导出不受影响。
- **参考文献注释**：按用户要求删除每条文献下的两行历史记录提示（"这是历史记录…"、"历史 KG 记录…"），替换为逐条双语注释：**该文献做了什么** + **与本假设的关系**（覆盖预定诊断组合、方向是否一致、属关联/背景证据、为提出本假设时引用的既有证据之一）。注释由 `build_reference_notes_v1.py` 从冻结的 KG 记录字段（subject/predicate/object/domains）和选取表的 named/direction 模板化生成，逐句绑定包内引文，不新增任何科学内容；产物 `cs1_discovery_reference_notes_v1.json`（99 条注释 + 34 条通俗定义，SHA256 `939520fe…`）作为展示层注释文件绑定 v6 包哈希，随第 4 个分发文件打包；文献层面的核查缺口仍由 common_pre 的"证据边界"统一披露。语言切换时材料区就地重渲染并保留滚动位置。
- **为何提出这项假设 · 叙述化**：从一句话结论扩展为可读的证据链——先一句话引入（"系统形成这个猜想，是受了这些已发表研究的启发"），再逐条列出 P1/P2/P3 各做了什么（复用文献注释），最后保留模型推理原文与来源说明供溯源。
- **精确定义 · 通俗解读**：技术定义下方新增一句"通俗理解"（ROI→网络：把某侧某分区到某网络所有分区的连接取平均，只是该分区到该网络的平均，不代表整个脑区；网络→网络：把两网络间全部分区间的连接取平均，不针对单个分区），由解析冻结定义文本模板化生成。
- **前序反馈 · 通俗引语**："前序假设实验与真实反馈"块开头新增一句通俗说明（有父假设：上一轮已实测过相近猜想、方向稳定度、本猜想再进一步；首轮卡：迭代链起点），由包内反馈字段渲染。

## 本次分发目录

`C:/Users/45846/Documents/Code/NeuroClaw/desktop/dist/study-client-v1.0.0-20260919-v2`

- `NeuroDiscovery Setup 1.0.0.exe`：Windows x64 安装版。
- `NeuroDiscovery 1.0.0.exe`：Windows x64 免安装版。
- `NeuroDiscovery-1.0.0-win.zip`：完整便携目录。
- `BUILD_INFO.json`：本次文件哈希与实际检查记录。

使用本目录的新文件；旧日期目录保留，不覆盖历史交付。升级前请先关闭正在运行的旧客户端。

## 验证边界

- 测试全过：discovery 系列与分配表 unittest 130 项、desktop 打包/窗口 + 工作区 pytest 74 项、Node 36 项。
- 真实 Chromium 同源 iframe 实测（嵌入模式）：v6 固定材料 34 份显示正确；分配下拉 12 个选项、选 P03 后简介联动为 10 份/60 题；P03 会话恰好 10 张分配卡且会话信息含 P03；四段式结构与题目新顺序渲染正确；首轮卡"无父假设"反馈块正常；参考文献两行旧提示已删除、新双语注释渲染正确；"为何提出这项假设"叙述化列表、"精确定义"通俗解读、反馈块通俗引语均渲染正确；英文切换无漏翻；三栏滚轮与侧栏拖拽/键盘/双击复位回归通过。扩展页：无专家研究超链接、无打分选择；分配下拉 12 项（全部 120 道·主持预览 / P01–P10·各 12 道）；选 P03 后文案联动为"一次定时会话 · 12 道"；P03 会话恰好 12 道、显示"第 1 / 1 次会话"；表单与专家研究一致；设置面板间距修复经布局探针与截图确认。
- 名字/代号共享实测：专家研究默认本机账户名（45846）→ 扩展页同默认；扩展页填"联合测试"创建会话后，专家研究入口自动带入同值。
- 导出对齐实测：UI"导出答案"对话框文件名为 `NeuroDiscovery-user-study-VERIFY-EXPORT-….json`；导出 API 实测含 v2 信封（participant/session/timing/summary/60 行逐题记录），会话分配编号正确。
- 打包后端路由实测通过，打包内容含全部新资产且无临时验证文件；ZIP CRC 通过。验证用 VERIFY-V6 / VERIFY-NOTES / VERIFY-PLAIN / CURL-TEST / 联合测试 会话已删除；用户的真实会话（123、45846 等）与早前 VERIFY-V5 测试会话均保留未动。
- 未自动化 Electron 原生窗口本身。已生成 Windows 文件，未自动安装，未推送 Git 或发布；文件未数字签名；本次没有生成 macOS 安装包。

本次不改变任何科学材料内容、已收集答案或实验结果；材料包与英文目录逐字节沿用冻结版本，变化仅在展示层与分配逻辑。

## 2026-09-19 下午追加：Human Evaluation 改名、扩展题库 v2 与本轮四项收口

**品牌与题库（本轮前半段已完成）**

- 两个研究更名为 **Human Evaluation 1 · COMPLETE RESEARCH OUTPUT**（原专家研究 / CASE STUDY 1）与 **Human Evaluation 2 · HYPOTHESIS RANKING**（原假设排序/扩展）；菜单、页眉、导出信封同步。
- 扩展题库重建为 v2：`neurooracle/data/user_study/case1_tcp_expert_study_v2.json`（SHA256 `ff9b1d51…`），240 对 / 480 假设，真值文件留在 `private_truth/`，不打入客户端。
- 扩展分配表 v2 `case1_tcp_expert_pair_assignments_v2.json`（每题 3 人复评，P01–P10 各 72 道 = 6 会话 × 12 道）。

**本轮四项收口**

1. **不再向参与者提供完整题库入口**：两个研究的分配下拉都过滤掉"全部 · 主持预览"（ALL 仍由服务端保留，供组织者编程/内部使用；前端不出现）。Human Evaluation 1 下拉现为 P01–P10（各 10 份），Human Evaluation 2 为 P01–P10（各 72 道）。
2. **修复"删除后直接进入题目"**：Human Evaluation 2 的"删除进度并重新开始"确认后原先直接沿用旧会话的名字/年限/分配自动建会话并进入答题；现在删除后停留在初始设置页——清空分配与年限选择、重新启用"开始排序会话"、聚焦名字/代号输入框，参与者重新填写后才会开始。`createFreshSession` 的旧会话复用形参随之移除。
3. **删除 Human Evaluation 2 首次打开弹窗**：说明内容本已写在初始界面，整块弹窗（DOM、`show/closeStudyIntro`、本地"不再提示"标记、约 50 行专用 CSS、16 组 i18n 字符串）全部移除，`showView` 不再触发弹窗。
4. **入口介绍删句**：删除"每对一个经内部实验验证、一个未获验证，且系统对前者的冻结打分显著更高"（中英文同步），不再向标注者泄露真值结构与分差信息。

**测试与实证**

- Node 40 项全过（含本轮新增 4 条防回归：两处 ALL 过滤、删除后停留设置页、无 intro 弹窗、文案无真值泄露句）；Python unittest 128 项全过（runtime python）；conda pytest 119 项全过。desktop/tests 下 5 个模块在 runtime python 下因缺 pytest 报导入错误，为既有环境限制，conda 环境全过。
- 真实 Chromium 实测：Human Evaluation 2 解锁后直接进入设置页（无弹窗）；分配下拉无主持预览；欢迎文案无泄露句。创建会话（P10）→ 重载 →"删除进度并重新开始"→ 确认后停留在设置页，年限/分配清空、开始按钮恢复、未进入答题。Human Evaluation 1 标题与分配下拉（P01–P10）核实一致。

**分发目录**

当日最终以 `desktop/dist/study-client-v1.0.0-20260919-v5` 为准（见下文傍晚追加；本节构建的 v4 中间目录已被后续轮次取代，作废不用）。v3 目录是在改名与新题库之前构建的中间产物，作废不用；旧日期目录保留不覆盖。

## 2026-09-19 傍晚追加：两入口设置页完全一致化与同意勾选移除（v5）

**两项变更**

1. **两个研究入口设置页前端完全一致**：Human Evaluation 1（`discovery-study.html`）与 Human Evaluation 2（`study.html`）的设置/入口页在文本宽度、字体、字号上逐项对齐。实际基准是共享样式层 `study-workspace.css` 的 legacy 变体（HE2 的实际渲染形态：扁平布局、28px 标题、14px 正文、569px 简介文本宽）；HE1 的 discovery 变体规则改写为同一组计算样式（标题 28px/37.8px/-0.98px 字距、正文 14px/22.4px、步骤圆 30px、步骤标题 13px/说明 12px、表单标签 12px、输入框 12px/10px 11px 内边距/7px 圆角、按钮 14px/650 字重），介绍栏改为与 HE2 相同的右分隔线结构；HE1 基础层恢复接近原版以避免双重覆盖；1120/820/480 三个断点行为同步。
2. **两个入口移除同意勾选**：删除 HE1"我自愿参与本次评审，并且可以随时停止。"与 HE2"我已阅读数据保存说明，自愿参与本次研究；可随时停止。"勾选框（DOM、CSS、i18n 字符串、提交载荷与前端必填校验一并移除）。服务端 `discovery_study.py` 非 legacy 资料校验改为仅必填代号+年限、consent 转为可选字段，`profile_schema_version` 升为 `anonymous-experience-v4`；导出信封中 consent 仅对勾选移除前创建的历史会话输出 true，新会话为 null/缺省，信封版本 `neurodiscovery-expert-study-results-v2` 不变。扩展研究服务端本就不强制 consent，无需改动。
3. **文字缩放一致性（当天二次修复）**：客户端外观设置 textScale（用户当前为 1.1）在 HE2 上通过 `zoom` 整页生效、在 HE1 上原本只设 CSS 变量，导致两页在该设置下字号/宽度实际不同（15.4px vs 14px、615px vs 569px）。HE1 的 `hostAppearance` 现在与 HE2 相同地对 `#app` 应用 zoom（含 `min-width: 960/scale`），并去掉 `.welcome` 上的固定字号改为继承 body 的 `calc(14px × scale)`，两页在任意 textScale 下渲染一致。
4. **静态资源缓存修复（界面"看起来没变"的根因）**：`/static` 挂载的 Starlette StaticFiles 只发 ETag/Last-Modified、不发 Cache-Control，Chromium 启发式缓存会在文件更新后继续供旧 JS/CSS（外壳只给 iframe 文档加了 v= 破缓存参数，子资源没有）。`/static` 与 `/materials` 现在统一发送 `Cache-Control: no-cache`（每次校验、未变则 304，开销极小），客户端今后更新即刻可见。两个页面路由（`/study`、`/discovery-study`）原本已是 no-store，未动。

**测试与实证**

- Python unittest 125 项、desktop 打包/窗口 + 工作区 pytest 127 项（conda neuroclaw 环境）、Node 63 项全部通过（含 consent 移除的防回归断言：静态扫描两页无勾选残留、提交载荷无 consent、无 consent 创建会话成功；hostAppearance 沙箱契约测试兼容无 DOM 环境）。
- **真实 Electron 客户端窗口实测**（CDP 附加到运行中的客户端，非独立浏览器）：两页在 textScale 1.1 下逐项一致——zoom 均 1.1、标题 30.8px/41.58px/-1.078px、正文 15.4px/24.64px、简介文本宽均 615px、标签 12px/400；两页均无同意勾选与文案残留，分配下拉均仅 P01–P10；用户已保存会话的"继续评审"横幅完好。scale 1.0 下两者共用同一组基础令牌，天然一致。
- 打包校验：win-unpacked 内 8 个改动文件与源逐字节一致（SHA256）；打包后端 7191 端口路由实测 /api/health、/discovery-study、/study、/static/study.html、/static/discovery-study.html 全部 200，/static 响应含 `no-cache` 头、包内 JS 含 zoom 对齐代码，检查后进程已停；ZIP CRC 通过、19078 个成员、无验证残留文件。四产物哈希见 v5 目录 `BUILD_INFO.json`。

**分发目录**

`C:/Users/45846/Documents/Code/NeuroClaw/desktop/dist/study-client-v1.0.0-20260919-v5`（NSIS 安装版 + 免安装版 + zip + BUILD_INFO.json）。v4 目录是本轮改动之前的中间产物，作废不用；更早目录保留不覆盖。

## 2026-09-19 傍晚第二轮追加：免密进入、返回确认弹窗、占位符平滑与父假设说明（v5 同名重建）

**五项变更（用户实测反馈驱动）**

1. **两入口眉标间距完全一致**：HE1 的 discovery 变体此前把 `.eyebrow` 纳入 `calc(11px × textScale)` 再叠加整页 zoom（双重缩放，视觉约 13.3px），HE2 为固定 11px 仅 zoom（12.1px）。已将 eyebrow 从该 calc 规则拆出，回退到固定 11px 基础层；无头实测两页 eyebrow→标题间距均 15.4px、标题→正文均 19.8px（embedded + textScale 1.1）。
2. **两个研究入口默认免密**：`server.py` 的 `study_password` 不再有默认值 `123456`（环境变量 `NEUROORACLE_STUDY_PASSWORD`/`NEURODISCOVERY_STUDY_PASSWORD` 为空即关闭密码门）；middleware 在空密码时放行全部 `/api/studies/*`，`/api/studies/auth` 在空密码时仍直接发 token 以兼容旧前端。HE2 的 `bootstrapAuth()` 改为直接进入加载（401 时才回退解锁页），HE1 本就隐藏 auth-form 直接加载。配置密码后行为与旧版完全一致（有专项测试）。
3. **返回工作台确认弹窗（两页一致）**：点击"返回工作台/关闭评审"时，若有未完成会话则弹窗，两个选项——**放弃填写内容**（服务端删除该未完成会话：HE1 新增 `DELETE /api/studies/discovery/sessions/{id}`，会话 token 鉴权；HE2 复用已有删除端点；已完成会话均不可删除，保持研究记录不可变）与**保存，下次继续**（先保存再关闭）；Esc/点击遮罩=留在页面，无任何动作。无活动会话时直接关闭不弹窗。
4. **"[已移除引文]示"不是运行时 bug**：这是冻结材料包构建期 `clean_rationale()` 的盲审脱敏占位符（未映射的 E 编号引文别名被替换），"示"是原句动词残留，句子不通顺。冻结包 JSON 不改，在 i18n walker 的 `apply()` 增加 `smooth()` 显示层后处理（中英文输出都生效）：`[已移除引文]示`→"有研究提示"、残留 `[已移除引文]`→"某文献（引文已略）"、`父节点<hex>`→"父假设"、"[父假设]"→"该父假设"、en `[citation removed]`→"an earlier study (citation omitted)"。translate() 保持无损，目录精确匹配不受影响。
5. **父假设具体内容说明**：材料"从真实反馈到后轮假设"块在原有原始字段之外，新增双语通俗句"父假设具体是什么"：把端点编码解码为可读测量（`ROI_to_network:NNN:NET`→"NNN 号感兴趣区与 X 网络中全部分区之间连接的平均强度"；`network_pair:A:B`→两网络间全部分区连接平均强度），并给出诊断组中文/英文名、相对健康对照的方向（按父反馈标准化效应符号）与内部重复方向一致比例。已核对 v6 包全部 30 个父假设端点均被两种解码器覆盖（4 张首轮卡保持"无父假设"说明），网络映射含 Vis/SomMot/DorsAttn/SalVentAttn/Default。

**测试与实证**

- Python unittest 173 项（core/web 全量，含新增：默认免密路径、配置密码路径、discard 服务端与路由、两页弹窗静态断言、父假设说明静态断言）、desktop pytest 127 项、Node 64 项全部通过。`test_autoresearch_help.py` 中 4 个与前几轮已接受改动脱节的过时断言（首次启动 intro 弹窗已移除、语言开关数量、condition 选择器已移除并固定 manual、菜单标签更名）已按当前 UI 事实更新。
- 无头 Chrome（CDP，embedded + textScale 1.1）实测：两页均直接显示设置表单、无密码门；两页弹窗按钮文案"放弃填写内容 / 保存，下次继续"；HE1 弹窗 save/discard/Esc 三分支实测正确；占位符 smooth 各形态实测（含向 DOM 注入占位文本经 walker 平滑）；父假设说明中英文、两种端点形态实测；eyebrow 11px 两页一致。
- 打包校验：win-unpacked 内 7 个改动文件与源逐字节一致（SHA256）；包内后端 7191 实测 /api/health、/api/studies/config（无 token 200）、/api/studies/discovery/config（无 token 200）、/discovery-study、/study、/static 全部 200，no-cache 头在，新代码字符串在包内，DELETE 路由存在（无 token 403），检查后进程已停；ZIP CRC 通过、19078 个成员。四产物哈希见 v5 目录 `BUILD_INFO.json`（已重写）。

**分发目录**

`C:/Users/45846/Documents/Code/NeuroClaw/desktop/dist/study-client-v1.0.0-20260919-v5`（同名重建，NSIS 安装版 + 免安装版 + zip + BUILD_INFO.json）。

## 2026-09-19 晚追加：实验设计板块简化与 v5 原地重建

- **实验设计板块只留专家需要的关键信息**：Human Evaluation 1 材料页"03 实验设计 · 模型、数据与指标"改为三个简明块——数据集（内部 TCP 234 人：ADHD 13 / 双相 26 / SZ/SZA 16 / 对照 91；外部 UCLA 257 / COBRE 158 / HCP-EP 145）、测量指标（静息态功能连接，每个假设预先锁定一条连接；读数为标准化效应与联合方向频率）、统计模型（OLS 回归，调整年龄/性别/站点，外部另调头动；内部 512 次、外部 2,000 次重抽样）。删除单轮展示中的"共同方法、数据接触与统计边界 · 必读"折叠块与"逐行模型与原尺度参数"折叠块（含证据边界内容）；这些细节仍保留在导出记录中，只是不进评审面板。历史两阶段分支（旧包）的展示不受影响。新增中文文案全部登记英文显示译文并有防回归测试。
- **重启与"界面没变"**：界面不更新的根因是 /static 子资源的 Chromium 启发式缓存，已由前一轮的 Cache-Control: no-cache 修复；本轮打包前确认运行中后端已发送该头。客户端已随本节改动再次重启。
- **v5 原地重建**：`desktop/dist/study-client-v1.0.0-20260919-v5` 已按当前全部代码重新打包（覆盖当天较早的同名 v5；更早日期目录不动）。包内核验：设计简化、无 intro 弹窗、两处分配下拉仅 P01–P10 均在 win-unpacked 资产中确认；ZIP CRC 通过；最新产物哈希见该目录 `BUILD_INFO.json`（旧哈希作废）。
- 测试：Node 41 项（含设计板块防回归）、core.web 各 unittest 套件与 conda pytest 119 项、desktop 打包 unittest 11 项全过。验证会话 VERIFY-DESIGN 已删除，真实会话未动。

## 2026-09-19 晚第二轮追加：成真意义句、删迁移限制块、间距修复与排版收敛（新 v6 分发目录）

**五项变更（本轮用户实测反馈驱动）**

1. **"开始评审"与下拉 0 间距修复**：HE1/HE2 的设置表单（`#setup-form`/`#auth-form`）改为 `display:grid; gap:15px(#study-workspace 嵌入层 16px)`，修复年限/分配下拉紧贴"开始评审"按钮的问题。此前表单靠各控件自身 margin 撑开，下拉与按钮之间恰好为 0。
2. **假设块末尾新增"成真意义"句**：新增边车 `core/web/study_materials/cs1_discovery_significance_v1.json`（`build_significance_v1.py` 确定性生成，绑定 v6 包哈希；34 卡中英文）。模板："如果成真，这条假设意味着{诊断组}在「{标题}」这一指标上存在稳定的同方向降低——对脑研究，它把不同诊断连接到同一个可检验的网络机制上；对临床，这类可复现的连接指标有望成为辅助分型或追踪疗效的候选线索。"前端 `significanceHTML(cardId)` 仅在 v6 包会话内展示（老包会话不继承）；暂存与打包绑定校验与 reference_notes 同构。
3. **删除"证据迁移的限制"展示块**：单轮阅读结构 02 节不再展示 transfer_limit 块（题库 rubric 与真实历史反馈文字中的"转移限制"原样保留——那是冻结题库内容与历史评议记录，不是展示框）。历史两阶段分支（旧包）不受影响。
4. **"三方两轮均判 keep"说清来源**：反馈块新增小字说明——"三方"是三个独立的大语言模型评审角色（生物统计、方法学、临床神经科学），各自先独立评审、看到另两份意见后复核，两轮共六份评议；评议来自模型角色而非人类专家，其赞同不构成独立实验支持。
5. **统计模型块补深度学习说明**：句末追加"候选假设由系统在神经影像知识图谱上提出、经三个大语言模型评审角色讨论；验证刻意使用可解释的线性模型，没有使用深度学习。"——本批专家评审案例的实验验证全部为 OLS 线性模型；假设提出端为知识图谱规则采样 + LLM 角色讨论（项目内另有 GraphSAGE/GNN 工作，不属于本批评审范围）。

**排版统一**：全站字号收敛到 {12, 13, 14, 15, 18, 24}px 六档（正文 14px 基准；HE1 材料页大标题等曾出现的 16–28px 各档向两端收拢），两个专家研究的字体族与字号阶梯保持一致；node 防回归测试含三文件字号白名单断言。

**测试与实证**

- Python unittest 117 项（core/web 全量，新增 `test_discovery_significance_v1.py`：构建确定性与哈希守卫/34 卡中英文覆盖/服务端 config 暴露/异包忽略/错哈希拒绝/前端渲染断言/两层 CSS 都有 .sig 样式）、desktop 打包 unittest 12 项（含边车暂存逐字节与绑定校验）、conda pytest 120 项、Node 43 项（27 workspace + 16 i18n，i18n 断言同步更新）全部通过。`test_discovery_v5.py` 中 `hypothesisHTML` 签名断言随新参数更新。
- 浏览器实测（真实后端，中文+英文）：验证会话 VERIFY-SIG2 / P01 走过全流程；材料 01 确认意义句出现在假设块内（中英文各有对应译文）、无"证据迁移的限制"块、反馈块含"三个独立的大语言模型评审角色"说明、统计模型块含无深度学习说明；欢迎页"开始评审"与下拉间距恢复；HE2 排序页排版与代号带入确认未受影响。验证会话已删（events 2 + sessions 1），45846 三条真实会话原样保留。
- 打包校验与产物哈希见 `desktop/dist/study-client-v1.0.0-20260919-v6/BUILD_INFO.json`。

**分发目录**

`C:/Users/45846/Documents/Code/NeuroClaw/desktop/dist/study-client-v1.0.0-20260919-v6`（NSIS 安装版 + 免安装版 + zip + BUILD_INFO.json）。此前的 v3–v5 目录均为中间产物；**v6 为当前最新分发**。

## 2026-09-21 追加：HE2 自动匹配参考文献补齐"做了什么 / 关系 / 最相关原文"

**背景**：HE2（假设排序）v2 题库 2,400 条参考文献中，45 条来自 v1 人工复核（逐字复用，含完整"本文研究的是 / 对当前 hypothesis 的具体支持 / 证据强度"三段评议），其余 2,355 条为知识图谱记录自动匹配。此前自动匹配条目把声明句"知识图谱记录自动匹配；本轮未逐条人工复核。"当作相关性结论文本直接展示，专家看不到该文献做了什么、与假设有什么关系。

**改动（仅展示层，题库与真相文件不动）**

1. `study.html` 文献行构建器新增 `manualRelevanceStatus` 字段透传。
2. `paperRelevanceReason`：仅当参考文献**不是**自动匹配时才把题库存的相关性结论文本当作权威结论展示；自动匹配条目改走已有的计算分析路径——关系芯片（疾病/脑区/指标/方向四要素）+"该研究在……样本中报告了……"（指标与方向由摘录文本识别）+ 关系程度分级。声明句保留为文献卡内小字来源说明（`.paper-auto-note`，中英双语取自题库原字段），不作结论展示。
3. "与假设最相关的原文"（按句子与疾病/脑区/指标/结果词匹配打分，最多 4 句）对自动匹配条目同样可用，经"展开相关原文"按钮展示。
4. 顺手修正计算路径英文句缺主语的问题（`What this paper studied: The paper reports …`），避免醒目行出现 "reports higher or increased…" 这种无主语残句。

**测试与实证**

- Node 28 项（study-workspace，新增自动匹配展示防回归断言）、conda pytest 135 项、其余 node 套件（i18n 16 / reading 12 / significance 2 / evaluation-export 4 等）全过；core.web unittest discover 171 项中仅 `test_discovery_v2` 2 个用例因历史 Downloads 交接目录被清理而报 FileNotFoundError（环境遗留，与本轮无关，未改动）。
- 另一会话 09-20 用共享 `evaluation-export.js` 替换 HE1 旧导出弹窗但未同步测试契约；本轮把 `test_study_workspace.py` 的期望 ID 列表更新为新设计（`export-complete`/`export-saved`/`save-progress`，并断言旧 `export-dialog` 标记已移除、共享导出模块已挂载）。
- 浏览器中英双语实证（VERIFY-REF / P01 会话）：自动匹配文献卡显示四要素关系芯片、"该研究在精神分裂症谱系精神病相关样本中报告了低频振幅分数（fALFF）升高或增大"式研究发现句、小字自动匹配声明、"展开相关原文"可看到最相关原句；英文 "The paper reports …" 完整成句。验证会话已从 `~/.neurodiscovery/user-studies/study.sqlite3` 删除（events 5 + candidates 480 + sessions 1），库内无其他会话。

**边界**：文献匹配本身仍为题库构建时的自动匹配，未做人工复核——声明句保留正是为了向专家明示这一点；本轮只改展示，不改题库、分配表、真相文件与任何已保存会话。运行中的后端逐请求读取静态文件，改动已即时生效，未重启（端口 7080 后端同时为 KG 生产服务 PID 40168，不宜动）。未重打包：09-20 另一工作流（v7+ 多主题面板）尚未自行打包，避免把在途工作混入本轮分发。

## 2026-09-21 追加（二）：HE2 Study details 本地摘要补全边车

**背景**：承接当日"自动匹配参考文献补齐"改动，用户进一步要求把 HE2 文献卡的 Study details（摘要、队列规模、相关 p 值）尽力补全，指出摘要应都在本地。

**数据与构建**

- 本地摘要来源：冻结语料库 `tmp/kg_rebuild_full_20260920_v1/CORPUS.sqlite`（只读，优先 HISTORICAL_ABSTRACT_CACHE 质量层），回落到 `neurooracle/data/full_snapshot_v2/abstract_cache.jsonl`（PubMed 摘要快照）。
- 新构建器 `core/web/build_study_bank_reference_notes_v1.py`（题库 SHA 钉住 ff9b1d51…）：生成边车 `neurooracle/data/user_study/case1_tcp_expert_study_v2_reference_notes_v1.json`，SHA256 `27594c586432d71978cfe05aea6aa728fabaaeedfe807bb39f6b38ca125780d2`。
- 覆盖：143 条去重参考文献中 135 条原本无摘要，本地解析出 **106 条摘要**（其余 29 条本地确无记录，前端维持"暂无"）；规则提取 **33 条队列规模**（首个 `N = n` 或最大的"n participants/patients/subjects"计数，展示恒带"≈…摘要自动提取，未经人工核对"中英标注）与 **18 条 p 值提及**（状态固定为 `reported_elsewhere_in_abstract`，即"摘要另有 p 值，但与所选证据不直接对应"——不做超出证据的断言）。
- 题库、分配表、真相文件、已有会话一律不动；边车只补缺口（已有 abstract/credibility 的 45 条人工复核条目完全不受影响）。

**服务端合并**：`neurooracle/src/user_study.py` 新增 `load_reference_notes`（缺边车则跳过；哈希不匹配即报错，与 pair_assignments 同构）与 `apply_reference_notes`（仅填空缺字段），在 `create_session` 加载候选后应用，随会话快照进入 `session_candidates.payload_json`。手动（manual）会话的成绩字段剥离逻辑不受影响。

**测试与实证**

- 新增 `core/web/test_study_bank_reference_notes_v1.py` 8 项（构建确定性与绑定、键完整性、字段规范与"自动提取"标注、提取器保守性、缺边车/错哈希行为、只补缺口、真实会话 2,000+ 自动匹配槽位带摘要且分数不泄露）；`test_user_study`、`test_study_bank_v2`、`test_evaluation_assignments_v12`、`test_evaluation_export` 共 51 项与 conda pytest 125 项、node 28 项全过。
- 真实后端 API 实证（桌面后端 7081，Electron 自动让出被 KG 生产服务占用的 7080）：P01 会话 2,400 个参考文献槽位中 2,205 带摘要（含 2,160 自动匹配）、811 带队列、784 带 p 值；验证会话经 DELETE 路由删除，7080/7081 两侧均无残留。
- `desktop/scripts/prepare-bundled-runtime.ps1` 的 HE2 分发文件清单已加入边车，重暂存通过；未重打包（同前，v7+ 在途工作流未自行打包）。

**生效说明**：补全是 Python 侧合并，需要后端进程重载；桌面客户端已于本轮重启，其自有后端（7081）已生效。端口 7080 现为 KG v29 生产服务（AGENTS.md 钉住的 PID 40168，附带挂载研究路由），未动；若用户经 7080 查看研究页，Study details 补全要等该服务下次自然重启。

## 2026-09-21 追加（三）：研究详情默认展开、联网 JIF、队列/p 值提取加强（边车 v2）

**三项用户要求**

1. **研究详情默认展开**：文献卡 `paper-details` 由折叠改为默认展开（`<details open>`），node 防回归断言已加。
2. **期刊当前 JIF 联网检索**：6 个并行检索代理 WebSearch 题库 61 本归一化期刊的最新官方 JCR 影响因子（优先 2024 JCR/2025-06 发布，否则 2023），53 本有值、8 本无现行 JIF（3 个预印本服务器"不适用"；《中华医学杂志》《Japanese Journal of Psychiatry and Neurology》（2008 停刊）《Archives of General Psychiatry》（2013 更名 JAMA Psychiatry）《BioMed Research International》（2023 被 WoS 剔除）《Chronic Diseases and Translational Medicine》如实记 null）。检索代理的两处错归因（把 JAMA Psychiatry 的 17.1 归给已停刊的 Archives of General Psychiatry；给已被剔除的 BioMed Research International 回收旧值）已由根任务纠正为 null。整理表 `neurooracle/data/user_study/journal_impact_factors_2024_v1.json`（SHA256 `48cb353debae40023cae040d7146451bc369f73de6b78dda6bff0ece26babd40`，逐刊保留来源 URL），一次性生成脚本 `tmp/build_jif_table_20260921.py`。
3. **p 值与队列规模为何稀少（已答用户并改进）**：p 值——88 篇本地摘要的正文里根本没有 p 字样（这些论文只在全文报告），规则提取如实只拿到 19 篇，不编造；队列——v1 提取器过保守（仅大写 `N =` 优先 + 人群关键词），v2 改为大小写不敏感的 `n =` 与关键词计数（含 controls）合并取最大，覆盖率 33→51 篇唯一文献（含复用卡的自有摘要）。提取值恒带"≈…摘要自动提取，未经人工核对"中英标注；p 值状态固定 `reported_elsewhere_in_abstract`。

**构建与服务端**

- 新构建器 `core/web/build_study_bank_reference_notes_v2.py`（题库与 JIF 表双哈希钉住）生成 `case1_tcp_expert_study_v2_reference_notes_v2.json`（SHA256 `229746366d11c1aff6f6c43988319bf876ea015be7859ae5102297bd263c01b1`）：143/143 文献均有条目，106 摘要、51 队列、19 p 值、114 文献带 JIF 值。
- `user_study.py` 边车加载改为 v2→v1 回退；合并新增 JIF 规则：仅当现有记录无数值时升级（45 条人工复核卡的"未核实"状态由此获得联网检索值；已有数值的记录不动）。题库/分配表/真相文件/已有会话不变。
- 前端无需新字符串：JIF 走既有 credibility 渲染（值 + 年份；"不适用/暂无/未核实"文案已有）；JIF 免责说明（期刊层面指标）原样保留。
- `prepare-bundled-runtime.ps1` 分发清单加入 v2 边车。

**测试与实证**：新增 `test_study_bank_reference_notes_v2.py` 8 项（构建确定性/双哈希绑定/143 键全覆盖/JIF 状态与数值规范/提取器/规范化/v2 优先 v1 回退/只升无值 JIF/真实会话 2,000+ 摘要槽位与 1,900+ JIF 槽位且分数不泄露）；unittest 59、conda pytest 125、node 29 全过。重启桌面客户端后真实后端实证：2,400 槽位中摘要 2,205、队列 1,139、p 值 784、JIF 2,177，验证会话 VERIFY-NOTES4 经 DELETE 删除；用户自己的两条 45846 会话未动。注意：KG v29 生产服务（原 7080，PID 40168）本轮重启前已不在，桌面后端本次绑定在 7080；KG 侧按其 STOP 与恢复脚本由其工作流自行管理，本轮未重启或修改任何 KG 资产。
