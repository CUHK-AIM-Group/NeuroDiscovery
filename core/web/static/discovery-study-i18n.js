/* Local display layer. No model calls, no form replacement and no score changes. */
"use strict";
(function (root) {
  const STORAGE = "neurodiscovery.discovery-study.language.v1";
  const ui = {
    "保存进度":"Save progress", "导出结果":"Export results", "保存并退出":"Save and exit",
    "导出并交给研究者":"Export and send to the researcher",
    "已请求下载，请将 JSON 文件交给研究者。":"Download requested. Send the JSON file to the researcher.",
    "完成本例能力评分":"Rate the capabilities for this case",
    "研究依据":"Research rationale",
    "研究依据：":"Research rationale:",
    "每项 1–5 分，适用项等权。无法判断不计分。":"Each item is scored 1–5, with equal weights for applicable items. Cannot judge is unscored.",
    "每次阅读一份完整材料，完成能力评分后继续。提交前可回看和修改。":"Read one complete case, rate its capabilities and continue. Revisit and edit before submission.",
    "完整科研成果评审 · Human Evaluation 1":"Complete research output review · Human Evaluation 1",
    "Human Evaluation 1":"Human Evaluation 1",
    "从假设，到结果。":"From hypothesis to results.",
    "评价一项研究带来了什么。":"Evaluate what a study adds.",
    "审阅十份跨诊断神经影像研究材料，评价系统从提出假设到形成研究成果的六项能力。每份 6 题，共 60 题，无需撰写长篇评审。":"Review ten transdiagnostic neuroimaging research materials and assess six capabilities from hypothesis to output. 6 questions per case, 60 total; no lengthy review required.",
    "本面板包含 30 份实验一致的发现材料和 4 份实验不一致的真实校准材料；本次分配给你 34 份，每份 6 题，共 204 题。所有材料使用同一套“越高越好”的能力评分，结果不一致本身不是低分理由。":"This panel contains 30 findings with results consistent with their hypotheses and 4 real calibration cases with inconsistent results. You are viewing all 34 cases, with 6 questions per case and 204 questions in total. Every case uses the same higher-is-better capability ratings; an inconsistent result is not itself a reason for a low score.",
    "本面板包含 30 份实验一致的发现材料和 4 份实验不一致的真实校准材料；本次分配给你 10 份，每份 6 题，共 60 题。所有材料使用同一套“越高越好”的能力评分，结果不一致本身不是低分理由。":"This panel contains 30 findings with results consistent with their hypotheses and 4 real calibration cases with inconsistent results. You are assigned 10 cases, with 6 questions per case and 60 questions in total. Every case uses the same higher-is-better capability ratings; an inconsistent result is not itself a reason for a low score.",
    "阅读一个完整案例":"Read one complete case",
    "假设、依据、方法、内外部结果和反馈同页展示。":"Hypothesis, rationale, methods, internal/external results and feedback appear together.",
    "完成本例六项评分":"Rate six capabilities for this case",
    "评分区保留当前假设，可随时对照材料。":"The current hypothesis stays beside the questions for easy reference.",
    "继续下一个案例":"Continue to the next case",
    "一轮完成分配的案例，不再分两轮往返。":"Review the assigned cases in one round, without revisiting them in a second phase.",
    "先看假设与方法":"Read hypotheses and methods first",
    "完成全部材料的初评，不显示候选实验结果。":"Complete initial ratings for all materials before candidate results are shown.",
    "再看完整结果":"Then read complete results",
    "锁定初评后，查看内外部分析及真实反馈链。":"After locking initial ratings, review internal/external analyses and the actual feedback chain.",
    "独立作出判断":"Make independent judgments",
    "选择题与 1–5 级评分，允许无法判断。":"Multiple-choice questions and 1–5 ratings, with Cannot judge available.",
    "数据保存与评价边界":"Data storage and evaluation boundaries",
    "只填写名字/代号和研究年限；请勿填写邮箱、患者信息等其他个人信息。答案、提交时间和客户端估计的活动时长保存在本机独立数据库，无第三方统计脚本；继续码保存在当前客户端。时长不作为能力分数。":"Provide only your name or a code and years of research experience; do not enter your email, patient information or other personal details. Answers, submission times and client-estimated active time are stored in a separate local database, without third-party analytics. The resume code is stored in the current client. Time is not a capability score.",
    "材料含探索性统计与文献核查缺口。请按所见证据回答，可查给出的论文并在备注中注明。":"Materials include exploratory statistics and gaps in literature verification. Judge the evidence shown; you may check the supplied papers and record this in a note.",
    "本版同页提供完整研究材料；评分是在结果可见条件下作出的评价，不是结果盲评。":"Complete research materials appear together. Ratings are made with outcomes visible, not under outcome blinding.",
    "阶段 A 不显示候选结果，这一显示顺序不保证评审者此前没有接触过结果。":"Stage A hides candidate results; this display order does not guarantee that reviewers have never seen those results.",
    "开始一次独立评审":"Start an independent review",
    "当前入口使用原有研究访问密码。":"This entry uses the existing study access password.",
    "访问密码":"Access password",
    "解锁入口":"Unlock",
    "此浏览器有已保存的会话":"This browser has a saved session",
    "继续评审":"Resume review",
    "开始独立的新会话":"Start a separate new session",
    "名字/代号":"Name / code",
    "例如：张三 或 P023":"e.g. Zhang San or P023",
    "相关研究年限":"Relevant research experience",
    "请选择":"Please select",
    "0–2 年":"0–2 years", "3–5 年":"3–5 years", "6–10 年":"6–10 years", "10 年以上":"Over 10 years",
    "审阅分配给你的研究材料，评价系统从提出假设到形成研究成果的六项能力，无需撰写长篇评审。":"Review the research materials assigned to you and rate the system's six capabilities from hypothesis to output—no lengthy review required.",
    "NeuroDiscovery 是一个自动研究系统：它在神经影像知识图谱上自动提出可检验的假设，用真实数据完成内部与外部实验验证，并把假设、依据、方法和结果整理成可评审的研究材料。":"NeuroDiscovery is an automated research system: it proposes testable hypotheses on a neuroimaging knowledge graph, validates them on real data through internal and external experiments, and packages hypotheses, evidence, methods and results into reviewable research materials.",
    "自动指标无法替代领域专家对科学价值与可信度的判断——系统眼中的好成果，是否经得起专家的独立评审，需要你来判断。":"Automated metrics cannot replace expert judgment about scientific value and credibility—whether the system's best-rated outputs hold up under independent expert review is yours to judge.",
    "开始评审":"Start review", "开始评审 →":"Start review →", "进入阶段 A →":"Enter stage A →",
    "用另存的继续码恢复":"Restore with a saved resume code",
    "继续码相当于本次会话的钥匙，请勿公开。":"The resume code is the key to your session. Do not share it publicly.",
    "继续码":"Resume code", "恢复会话":"Restore session",
    "正在连接本机评审服务…":"Connecting to the local review service…",
    "保存继续码":"Save resume code", "导出答案":"Export answers", "暂停并保存":"Pause and save",
    "评审材料":"Review materials", "独立判断":"Independent judgments",
    "十份材料顺序随机。没有标准答案。":"The ten materials are randomly ordered. There are no standard answers.",
    "已保存":"Saved", "上一份":"Previous", "下一份":"Next", "保存并评下一份":"Save and review next",
    "取消":"Cancel", "确认":"Confirm", "完成":"Done", "关闭评审":"Close review",
    "返回工作台？":"Return to the workbench?",
    "本次评审尚未完成。可以保存已填写的内容、下次在本客户端继续，或放弃本次填写的内容。":"This review is not finished. You can save your answers and continue later in this client, or discard what you have entered.",
    "放弃填写内容":"Discard my answers", "保存，下次继续":"Save and continue later",
    "答案已保存为本机文件":"Answers saved to a local file",
    "不包含继续码或隐藏来源映射。即使浏览器不支持下载，下列文件也已写入本机。":"The file excludes the resume code and hidden source mapping. It is already saved locally, even if the browser does not support downloads.",
    "文件绝对路径":"Absolute file path", "复制路径":"Copy path", "尝试浏览器另存":"Try browser download",
    "此问卷需 JavaScript；所有数据保存于本机服务。":"This questionnaire requires JavaScript; all data is stored by the local service.",
    "研究入口需要重新解锁。请保存继续码后回到入口，原答案仍保留。":"The study entry needs to be unlocked again. Save your resume code and return to the entry; your answers remain stored.",
    "请先解锁研究入口。":"Please unlock the study entry first.",
    "已完成 · 答案锁定":"Complete · Answers locked", "锁定初评，解锁结果":"Lock initial ratings and reveal results", "完成评审":"Complete review",
    "查原文 ↗":"View source ↗", "R1 · TCP 固定内部测量（β / 残差 RMS）":"R1 · Fixed internal TCP measurements (β / residual RMS)",
    "比较":"Comparison", "病例 / 对照":"Cases / controls", "标准化效应":"Standardized effect",
    "R2 · 全部外部组成及敏感性结果（横向滚动）":"R2 · All external component and sensitivity results (scroll horizontally)",
    "队列 / 诊断":"Cohort / diagnosis", "n 病例/对照":"n cases/controls", "描述性百分位范围":"Descriptive percentile range", "同向频率":"Same-direction frequency",
    "百分位范围是描述性 bootstrap 结果，不是校准置信区间；没有 P/q 值。请勿按是否跨零作显著性判断。":"Percentile ranges are descriptive bootstrap results, not calibrated confidence intervals; no P/q values are provided. Do not infer significance from whether a range crosses zero.",
    "父假设测量：":"Parent-hypothesis measure:", "具名组成：":"Named components:", "父反馈标准化效应：":"Standardized effects in parent feedback:", "联合方向频率：":"Joint direction frequency:",
    "真实后轮利用说明：":"Recorded use in a later round:", "原始讨论后的决策：":"Decision after the original discussion:",
    "文中的“三方”是三个独立的大语言模型评审角色（生物统计、方法学、临床神经科学视角）：先各自独立评审，再看到另外两份意见后复核，两轮共六份评议。评议来自模型角色，不是人类专家意见；其赞同不增加独立实验支持。":"“Three parties” means three independent large-language-model reviewer roles (biostatistics, methodology and clinical neuroscience perspectives): each role reviews independently first, then re-checks after seeing the other two reviews — six reviews across two passes in total. The reviews come from model roles, not human experts; their agreement does not add independent experimental support.",
    "R3 · 首轮提议：生成该假设时同 seed 尚无已完成的 TCP 反馈，无父假设绑定。":"R3 · First-round proposal: when this hypothesis was generated, no completed TCP feedback existed for the same seed, so there is no parent-hypothesis binding.",
    "首轮提议说明：":"First-round proposal note:",
    "请分别评价系统能力，并保留测量、诊断与队列的边界。":"Rate each system capability separately, respecting the boundaries of the measures, diagnoses and cohorts.",
    "系统提出的假设":"System-proposed hypothesis", "待检验的假设":"Hypothesis to be tested", "程序原始表述":"Original program-rendered statement", "精确定义":"Exact definition", "为何提出这项假设":"Why this hypothesis was proposed", "证据迁移的限制":"Limits of evidence transfer",
    "共同方法、数据接触与统计边界 · 必读":"Common methods, data exposure and statistical boundaries · Required reading",
    "内部与外部结果":"Internal and external results", "TCP · 联合方向频率":"TCP · Joint direction frequency", "UCLA · 联合方向频率":"UCLA · Joint direction frequency",
    "逐行模型与原尺度参数":"Row-level models and original-scale parameters",
    "完整解释边界 · 必读":"Full interpretation boundaries · Required reading", "从真实反馈到后轮假设":"From actual feedback to a later-round hypothesis",
    "整理者摘要":"Curator summary", "非模型总结原文":"Not an original model summary", "实验结果尚未解锁":"Experimental results are not yet revealed",
    "前往本份评审题 ↓":"Go to the questions for this material ↓",
    "查看本次能力评分汇总":"View this session's capability scores", "综合均分：":"Composite mean:", "能力":"Capability", "均分 / 5":"Mean / 5", "有效材料数":"Number of valid materials",
    "仅汇总本次个人评价；不同能力可有不同有效数。未评分项及原因单独保留。全部统计使用导出中的未舍入数值。":"This summarizes only your ratings in this session. Valid counts may differ by capability. Unscored responses and their categories are retained separately. Use unrounded exported values for statistics.",
    "已完成的独立判断":"Completed independent judgments", "假设与设计初评":"Initial hypothesis and design review", "完整成果评审":"Complete output review",
    "答案已锁定，可导出完整记录。":"Answers are locked; you can export the complete record.",
    "每题选择一项，不预选答案。对不同维度分别判断，避免用单一印象代替评价。":"Choose one option per question; none is preselected. Judge each dimension separately rather than using a single overall impression.",
    "回看材料 ↑":"Review the material ↑", "当前案例 · 假设对照":"Current case · Hypothesis reference", "回看假设、方法与结果 ↑":"Review hypothesis, methods and results ↑",
    "请阅读本例完整材料后作答。":"Read the complete case before answering.",
    "查看阶段 A 已锁定判断":"View locked stage A judgments", "未回答":"Unanswered", "可选 · 主要问题标签":"Optional · Main issue tags",
    "可选 · 关键理由、缺失证据或原文核查线索":"Optional · Key reasons, missing evidence or source-verification notes",
    "例如：P2 不能直接支持本次测量；还需双相独立样本…":"For example: P2 does not directly support this measure; an independent bipolar sample is still needed…",
    "每项能力按所选描述记为 1–5 分；六项等权。无法判断的回答单独记录，不纳入数值均分。":"Each capability is scored 1–5 according to the selected description; all six have equal weight. Cannot judge is recorded separately and excluded from numeric means.",
    "此会话保留旧版问卷与计分规则，不自动转换为新版能力分数。":"This session retains its original questionnaire and scoring rules; it is not automatically converted to the new capability scores.",
    "有更改，正在自动保存…":"Changes detected; autosaving…", "正在保存到本机…":"Saving locally…", "保存失败，尚未保存的输入仍在当前页面。":"Save failed; unsaved input remains on this page.",
    "评审已完成":"Review completed", "逐例评审 / 完整研究成果":"Case-by-case review / Complete research outputs", "阶段 A / 假设与方法":"Stage A / Hypotheses and methods", "阶段 B / 完整研究结果":"Stage B / Complete research results",
    "感谢你的独立判断。答案、各能力均分及综合均分已保存，可查看与导出。":"Thank you for your independent judgments. Answers, capability means and the composite mean are saved and available to view or export.",
    "旧版答案已保存，可查看与导出；不会自动改用新版计分规则。":"Original-version answers are saved and available to view or export; new scoring rules are not applied automatically.",
    "单轮 · 逐例完成":"Single round · One case at a time", "假设与设计":"Hypotheses and design", "结果与贡献":"Results and contribution",
    "开始独立的新会话？":"Start a separate new session?",
    "旧答案不会删除，但此浏览器会改记新会话的继续码。若需要返回旧会话，请先进入原会话保存继续码。":"Old answers will not be deleted, but this browser will store the new session's resume code. To return to the old session later, first open it and save its resume code.",
    "开始新会话":"Start new session", "继续码格式不正确。":"The resume code format is invalid.",
    "请完成这份材料的所有必答问题；确实无法判断时可选择对应选项。":"Please answer all required questions for this material; use Cannot judge when appropriate.",
    "锁定阶段 A 并显示结果？":"Lock stage A and reveal results?", "完成并锁定本次评审？":"Complete and lock this review?",
    "全部答案将按当前材料与计分版本保存并锁定。提交后可查看能力均分、综合均分及有效数量。":"All answers will be saved and locked with the current material and scoring versions. After submission, you can view capability means, the composite mean and valid counts.",
    "旧版答案将保存并锁定；保留原有题目与计分含义。":"Original-version answers will be saved and locked, retaining their original questions and scoring meaning.",
    "锁定并解锁结果":"Lock and reveal results", "初评已锁定 · 结果已解锁":"Initial ratings locked · Results revealed", "完成记录已保存":"Completion record saved",
    "路径已复制。":"Path copied.", "请使用 Ctrl+C 复制选中的路径。":"Press Ctrl+C to copy the selected path.",
    "已请求浏览器另存；若未出现下载，请使用已保存的本机文件。":"Browser download requested. If it does not appear, use the file already saved locally.",
    "继续码已复制，请存到安全位置，不要发在公开群或论文材料中。":"Resume code copied. Keep it somewhere safe; do not post it in public groups or paper materials.",
    "无法复制。请保留本浏览器，可随时使用入口的“继续评审”。":"Could not copy. Keep this browser; you can use Resume review at the entry.",
    "已保存，可关闭页面。下次在本浏览器继续。":"Saved. You can close the page and resume in this browser later.",
    "双相":"Bipolar disorder", "精神病":"Psychosis", "非情感性精神病":"Non-affective psychosis", "情感性精神病":"Affective psychosis", "SZ（不含 SZA）":"SZ (excluding SZA)",
    "主要模型 · 调整 FD":"Primary model · FD-adjusted", "敏感性 · 不调整 FD":"Sensitivity · No FD adjustment", "主要模型 · SZ/SZA":"Primary model · SZ/SZA", "敏感性 · 仅 SZ":"Sensitivity · SZ only", "主要 · 非情感性早期精神病":"Primary · Early non-affective psychosis", "背景 · 情感性早期精神病（非双相验证）":"Context · Early affective psychosis (not bipolar validation)",
    "keep":"Keep", "revise":"Revise",
    "请完整填写评审信息。":"Please complete the review information.",
    "分配编号":"Assignment", "未知的分配编号，请按入口列表选择。":"Unknown assignment; choose one of the identifiers listed at the entry.",
    "请使用 2–40 位的名字或代号（字母、数字或中文），不填写真实联系方式。":"Use a 2–40 character name or code (letters, numbers or Chinese); do not enter real contact details.",
    "请填写研究年限。":"Please select your research experience.", "请确认已阅读数据保存说明并自愿参与评审。":"Please confirm that you have read the data storage information and voluntarily agree to take part in the review.",
    "无法访问此会话，请使用原浏览器或保存的继续码。":"Cannot access this session. Use the original browser or the saved resume code.",
    "另一标签页已更新答案。请刷新后继续，未覆盖已保存答案。":"Another tab updated the answers. Refresh to continue; saved answers were not overwritten.",
    "已完成的评审不可改写；请导出或开始独立的新会话。":"Completed review answers cannot be changed. Export them or start a separate new session.",
    "显示版本已更新，请保存继续码后刷新页面。":"The display version has changed. Save your resume code, then refresh.",
    "本机尚未配置该评审材料包，请联系主办方。原有问卷不受影响。":"This review material pack is not configured locally. Contact the organizer. The original questionnaire is unaffected.",
    "备注最多 4,000 字符。":"Notes may contain at most 4,000 characters.",
    "答案不属于当前阶段，或选项无效。":"The answer is outside the current stage or the option is invalid.",
    "历史版本有未翻译文本，将按原文显示；题目和答案不改变。":"This historical version has untranslated text, shown in its original language; questions and answers are unchanged.",
    "英文为显示译文；数值及论文所录原句保持原样，可随时切回中文。":"English is a display translation. Numbers and recorded paper excerpts remain unchanged; switch back to Chinese at any time.",
    "材料不足，无法判断":"Insufficient material to judge", "超出本人专业范围":"Outside my expertise",
    "这个发现是什么":"What this finding is",
    "系统提出的假设与它的精确定义。":"The proposed hypothesis and its exact definition.",
    "它从何而来 · 相关研究与前序实验的启发":"Where it came from · Prior studies and earlier experiment feedback",
    "相关研究做了什么；系统前序假设实验留下了什么反馈。":"What prior studies did, and what earlier hypothesis experiments fed back.",
    "实验设计 · 模型、数据与指标":"Study design · Models, data and measures",
    "队列、测量指标、模型与统计安排。":"Cohorts, measures, models and the statistical plan.",
    "所有案例共用同一套数据、测量与模型。":"Every case shares the same datasets, measures and model.",
    "数据集":"Datasets", "测量指标":"Measures", "统计模型":"Statistical model",
    "内部实验用 TCP 队列 234 人（ADHD 13 人、双相 26 人、SZ/SZA 16 人，各比较的对照组 91 人）；外部验证用三个互不相干的独立队列：UCLA 257 人、COBRE 158 人、HCP-EP 145 人。":"Internal experiments use the TCP cohort of 234 participants (ADHD 13, bipolar 26, SZ/SZA 16, with 91 controls per comparison); external validation uses three mutually independent cohorts: UCLA 257, COBRE 158 and HCP-EP 145.",
    "静息态功能连接（fMRI）。每个假设预先锁定一条连接：某个脑分区到某个网络、或两个网络之间连接强度的平均值。看两个读数：病例组与对照的差异大小（标准化效应），以及差异方向在重复抽样中的一致程度（联合方向频率）。":"Resting-state functional connectivity (fMRI). Each hypothesis locks one connection in advance: the average connection strength from one brain parcel to a network, or between two networks. Two readouts matter: the size of the case–control difference (standardized effect), and how consistently that direction repeats across resampling (joint direction frequency).",
    "普通最小二乘（OLS）回归，比较病例组与对照，调整年龄、性别和扫描站点；外部分析另调整头动。结果的稳定性用重复抽样检验：内部 512 次、外部 2,000 次。候选假设由系统在神经影像知识图谱上提出、经三个大语言模型评审角色讨论；验证刻意使用可解释的线性模型，没有使用深度学习。":"Ordinary least squares (OLS) regression compares each case group with controls, adjusting for age, sex and scanning site; external analyses additionally adjust for head motion. Stability is checked by resampling: 512 internal and 2,000 external repetitions. Candidate hypotheses were proposed by the system on a neuroimaging knowledge graph and discussed by three large-language-model reviewer roles; the validation deliberately uses an interpretable linear model, with no deep learning.",
    "实验结果 · 内部与外部验证":"Results · Internal and external validation",
    "内部固定测量与外部组成、敏感性结果。":"Fixed internal measurements, external components and sensitivity results.",
    "前序假设实验与真实反馈 · 自动迭代":"Earlier hypothesis experiments and actual feedback · Automated iteration",
    "调整目录栏宽度":"Resize the materials sidebar", "拖动调整目录栏宽度":"Drag to resize the materials sidebar",
    "返回工作台":"Back to workspace"
  };
  const rules = [
    [/^我们从系统自动提出并完成实验验证的成果中，选取表现最好的 (\d+) 个完整案例；本次分配给你 (\d+) 份，每份 (\d+) 题，共 (\d+) 题，无需撰写长篇评审。$/, (_,pool,n,q,total)=>`From the outputs the system proposed and experimentally validated, we selected the ${pool} strongest complete cases; your assignment covers ${n} of them, ${q} questions each, ${total} total—no lengthy review required.`],
    [/^审阅 (\d+) 份跨诊断神经影像研究材料，评价系统从提出假设到形成研究成果的(六项能力|科学贡献)。每份 (\d+) 题，共 (\d+) 题，无需撰写长篇评审。$/, (_,n,type,q,total)=>`Review ${n} transdiagnostic neuroimaging research materials and assess ${type==='六项能力'?'six capabilities':'scientific contributions'} from hypothesis to output. ${q} questions per case, ${total} total; no lengthy review required.`],
    [/^全部 (\d+) 份 · 主持预览$/, (_,n)=>`All ${n} materials · Organizer preview`],
    [/^(P\d{2}) · (\d+) 份$/, (_,id,n)=>`${id} · ${n} materials`],
    [/^审阅跨诊断脑连接、影像遗传学和预后研究材料。本次分配 (\d+) 份，每份约 5 分钟，完成 5–6 项能力评分。$/, (_,n)=>`Review transdiagnostic connectivity, imaging-genetics and prognosis studies. ${n} cases are assigned, about 5 minutes each, with 5–6 capability ratings per case.`],
    [/^· 适用项均有数值的材料 (\d+) \/ (\d+) 份。$/, (_,n,total)=>`· Materials with numeric values for all applicable items: ${n} / ${total}.`],
    [/^固定材料：(.*) · (\d+) 份材料 · 自动保存$/, (_,id,n)=>`Fixed materials: ${translate(id)} · ${n} materials · Autosave`],
    [/^(.*)答案与材料版本已绑定。继续原会话不会改用更新的材料。$/, (_,id)=>`${id.replace(/。$/,' · ')}Answers are bound to their material version. Resuming does not replace it with newer materials.`],
    [/^(.*) · 单轮( · 试用版)?$/, (_,v,pilot)=>`${v} · Single round${pilot?' · Pilot':''}`],
    [/^(v[^\n]*) · 试用版$/, (_,v)=>`${v} · Pilot`],
    [/^(单轮评审|阶段 [AB]) · (\d+) \/ (\d+) 项已回答$/, (_,stage,n,total)=>`${stage==='单轮评审'?'Single-round review':stage.replace('阶段','Stage')} · ${n} / ${total} answered`],
    [/^材料 (\d+)( \/ \d+)?$/, (_,n,total)=>`Material ${n}${total||''}`],
    [/^(\d+) \/ (\d+) 项( · 已回答)?$/, (_,n,total,done)=>`${n} / ${total}${done?' · Answered':''}`],
    [/^(\d+) 份材料顺序随机。没有标准答案。$/, (_,n)=>`${n} materials in random order. There are no standard answers.`],
    [/^有效重复 (\d+) \/ (\d+)$/, (_,n,total)=>`Valid replicates ${n} / ${total}`],
    [/^R3 · 同 seed 的历史反馈在当前提议前已可见，绑定核对：(通过|未确认)。$/, (_,ok)=>`R3 · Earlier feedback from the same seed was available before this proposal. Binding check: ${ok==='通过'?'passed':'unconfirmed'}.`],
    [/^既有文献与核查缺口（(\d+) 份）$/, (_,n)=>`Prior literature and verification gaps (${n} records)`],
    [/^模型列：(.*)$/, (_,cols)=>`Model columns: ${cols}`],
    [/^β (.*)；残差 RMS (.*)$/, (_,beta,rms)=>`β ${beta}; residual RMS ${rms}`],
    [/^评测能力 · (.*)$/, (_,cap)=>`Capability · ${translate(cap)}`],
    [/^· 六项均有数值的材料 (\d+) \/ (\d+) 份。$/, (_,n,total)=>`· Materials with numeric values for all six capabilities: ${n} / ${total}.`],
    [/^每次阅读一个完整案例，完成六项评分后继续下一份。共 (\d+) 份，不再分两轮；提交前可回看和修改。$/, (_,n)=>`Read one complete case, rate six capabilities and continue to the next. ${n} cases in one round; revisit and edit before submission.`],
    [/^每份先评 (\d+) 项能力；完成全部 (\d+) 份初评后统一显示实验结果。$/, (_,q,n)=>`Rate ${q} capabilities per material first. Results are revealed after initial ratings for all ${n} materials.`],
    [/^初评已锁定。现在结合内外部证据，完成每份剩余 (\d+) 题。$/, (_,q)=>`Initial ratings are locked. Use internal/external evidence to answer the remaining ${q} questions per material.`],
    [/^完成全部 (\d+) 份材料的阶段 A 后统一解锁，避免先看到相关材料的结果。无需猜测实验是否成功。$/, (_,n)=>`Results unlock after stage A for all ${n} materials, avoiding early exposure to related outcomes. You do not need to guess whether experiments succeeded.`],
    [/^全部 (\d+) 份材料的初评将不可回改。随后显示全部内外部结果及反馈链。$/, (_,n)=>`Initial ratings for all ${n} materials will be locked. All internal/external results and feedback chains will then be revealed.`],
    [/^已保存 · (.*)$/, (_,time)=>`Saved · ${time}`],
    [/^连接失败：(.*)$/, (_,error)=>`Connection failed: ${translate(error)}`],
    [/^请求失败 \((\d+)\)$/, (_,code)=>`Request failed (${code})`],
    [/^([A-Z][A-Z0-9-]*) · (.*)$/, (_,prefix,value)=>`${prefix} · ${translate(value)}`],
    [/^((?:ADHD|SZ\/SZA|双相|非情感性精神病|情感性精神病|精神病) [−\-\d.]+)(.*)$/, (whole)=>whole.replace(/双相|非情感性精神病|情感性精神病|精神病/g, word=>ui[word]).replace(/；/g,'; ')]
  ];
  let messages = {...ui};
  function translate(source) {
    const text=String(source??''), key=text.trim();
    if(!key) return text;
    let value=messages[key];
    if(value===undefined) for(const [pattern,replacement] of rules) {
      if(pattern.test(key)) {value=key.replace(pattern,replacement);break;}
    }
    return value===undefined?text:text.replace(key,value);
  }
  // Frozen-pack de-identification placeholders read awkwardly mid-sentence.
  // Smooth them at the display layer only; the stored material is untouched.
  function smooth(text) {
    return String(text??'')
      .replace(/\[已移除引文\]示/g,'有研究提示')
      .replace(/\[已移除引文\]/g,'某文献（引文已略）')
      .replace(/父节点[0-9a-f]{6,20}/g,'父假设')
      .replace(/\[父假设\]/g,'该父假设')
      .replace(/\[citation removed\]/gi,'an earlier study (citation omitted)');
  }
  function mount(doc, {initialLanguage, onLanguage=()=>{}}={}) {
    let saved=''; try {saved=root.localStorage.getItem(STORAGE)||'';} catch {}
    let language=initialLanguage==='en'||initialLanguage==='zh'?initialLanguage:(saved==='en'?'en':'zh');
    const texts=new WeakMap(), attributes=new WeakMap();
    const excluded='script,style,textarea,[data-original],[data-user-content],[contenteditable="true"]';
    function apply() {
      const walker=doc.createTreeWalker(doc.documentElement,4);
      for(let node=walker.nextNode();node;node=walker.nextNode()) {
        if(node.parentElement?.closest(excluded)) continue;
        const previous=texts.get(node);
        const source=previous&&node.nodeValue===previous.rendered?previous.source:node.nodeValue;
        const rendered=language==='en'?smooth(translate(source)):smooth(source);
        texts.set(node,{source,rendered});
        if(node.nodeValue!==rendered) node.nodeValue=rendered;
      }
      doc.querySelectorAll('[placeholder],[aria-label],[title]').forEach(el=>{
        if(el.closest(excluded))return;
        let record=attributes.get(el);if(!record){record={};attributes.set(el,record);}
        for(const name of ['placeholder','aria-label','title']) {
          if(!el.hasAttribute(name)) continue;
          const value=el.getAttribute(name),previous=record[name];
          const source=previous&&value===previous.rendered?previous.source:value;
          const rendered=language==='en'?smooth(translate(source)):smooth(source);
          record[name]={source,rendered};if(value!==rendered)el.setAttribute(name,rendered);
        }
      });
      doc.documentElement.lang=language==='en'?'en':'zh-CN';
      doc.querySelectorAll('[data-discovery-language]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.discoveryLanguage===language)));
      const note=doc.getElementById('translation-note');if(note)note.hidden=language!=='en';
    }
    function setLanguage(next,{notify=true}={}) {
      if(!['zh','en'].includes(next)) return;
      const changed=language!==next;language=next;
      try{root.localStorage.setItem(STORAGE,language);}catch{}
      // Only text nodes and labels change: inputs, selection, scroll, open
      // details, focus, pending saves and the current card all remain in place.
      apply();if(changed&&notify)onLanguage(language);
    }
    doc.querySelectorAll('[data-discovery-language]').forEach(button=>button.addEventListener('click',()=>setLanguage(button.dataset.discoveryLanguage)));
    const observer=new root.MutationObserver(apply);
    observer.observe(doc.documentElement,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['placeholder','aria-label','title']});
    apply();
    return {apply,setLanguage,get language(){return language;},addCatalog(catalog){messages={...ui,...(catalog?.strings||{})};apply();},disconnect(){observer.disconnect();}};
  }
  const api={translate,smooth,mount,addMessages(value){messages={...ui,...value};},ui};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.DiscoveryI18n=api;
})(typeof window==='undefined'?globalThis:window);
