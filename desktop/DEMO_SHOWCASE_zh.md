# NeuroDiscovery 离线演示客户端

> 本文记录旧专用界面原型，现已停用。当前客户端直接复用正式版界面，启动方式和五个 prompt 见 [正式版界面 Demo](DEMO_NATIVE_PROMPTS_zh.md)。旧页面不再加载或打包。

五个预设场景，无需 LLM API key、Python、GPU 或网络。选择场景会自动选择 GPT-6-Astra 的演示选项及对应的 AutoResearch 模式，填入设计好的中文 prompt；点击发送后分块输出文字、展示工具步骤和最终产物。模型名称仅用于展示，未调用该模型。

**演示回复、图谱线索、评分、训练曲线与实验指标均为预设模拟。** 数据文件是可加载的合成数组；不存在真实患者、真实下载记录或科学验证。图谱和实际科研执行后端不会启动。

## 启动

- Windows：运行 `NeuroDiscovery-Offline-Demo-1.0.0-Portable-x64.exe`，或解压 ZIP 后打开 `NeuroDiscovery Demo.exe`。ZIP 必须完整解压。
- 源码：在 `desktop` 执行 `npm run prepare:demo:showcase`，然后 `npm run dev:demo`；也可双击 `desktop/Start-Offline-Demo.cmd`。
- 网页预览：直接用浏览器打开 `desktop/demo/index.html`，无需本地服务器。浏览器版不提供“打开本地演示数据”按钮。
- 独立应用数据目录：`%APPDATA%/NeuroDiscovery-OfflineDemo`。不会读取正式版模型配置或密钥。
- 源码重建 Windows 包：`npm run dist:demo:showcase`。输出到 `desktop/dist-demo-showcase`，旧的联网 Demo 与评估包保留。

## 五个预设场景

完整 prompt、逐步回复、工具输入输出统一位于 `desktop/demo/scenarios.js`，客户端直接使用这些内容，避免文稿和播放器不一致。

| 场景 | 用户输入概要 | 展示过程与交付 |
| --- | --- | --- |
| 普通聊天 | 你是谁，你能做什么？ | GPT-6-Astra、AutoResearch Off；介绍 NeuroDiscovery / NeuroOracle / NeuroRuntime 和四种模式 |
| Idea | ADHD 儿童的小脑—默认模式网络连接与注意症状；5 个候选，筛选 3 个 | 可点击示意子图、GNN/结构评分、新颖性、统计/临床/方法学三个 agent 的意见；按总分降序的完整 hypothesis 表；IDEA.md、CSV |
| Data | 处理 `/data/demo` 下四类样例 | 下载状态和模态、真实流程与简化样例的差别、分析/处理/QC/划分；model_ready 目录和张量形状 |
| Experiment | 指定 `/ideas/adhd_network/IDEA.md` 和 `/data/demo/adhd200/model_ready` | 项目模型选择、BrainNetCNN 示例代码、配置、训练曲线、验证集调参、结果表；H1/H2 通过模拟门槛，H3 未通过 |
| Full | 给出同一 topic 和数据位置，至少 3 项验证通过或达到 20 轮停止 | 全流程、每轮四阶段、不同假设去重计数、验证反馈；默认第 3 轮 H1/H2/H6 达标；可选择 20 轮仍不足的分支 |

所有工具等待 0.5–1.6 秒；文字每 32ms 输出 7 字符，不会瞬间填满。训练曲线分六次更新。支持随时停止、切换场景、从头重播。切换场景会取消正在播放的计时器。任意改写 prompt 不会触发伪造的通用回答，界面提示恢复预设。右上角可导出整个播放记录（JSON）；右侧产物可预览并逐个下载。

Full 的每轮判定使用开发集嵌套验证剧情；不会反复读取测试集来循环调参。达到开发集目标后才展示冻结方案的一次最终测试。这里的“验证通过”是演示状态，不能解释为真实得到已验证的科学假设。

## 四套样例和真实下载状态

| 数据集 | 本演示选择的模态 | 真实下载状态/必要处理 |
| --- | --- | --- |
| ADHD-200 | T1w、rs-fMRI、表型 | 原始 NIfTI、按站点组织、独立表型 CSV；需要 BIDS、运动与混杂处理、ROI 提取、FC 和标签对齐。见 [官方入口](https://fcon_1000.projects.nitrc.org/indi/adhd200/)。 |
| ABIDE | T1w、rs-fMRI、表型 | 示例解释 PCP 衍生物：`.1D` ROI 时序和 `.nii.gz` 影像；需核对 pipeline/atlas/GSR、QC、生成 FC、统一 ID 和划分。见 [PCP 下载说明](https://preprocessed-connectomes-project.org/abide/download.html)。 |
| ADNI | T1w、rs-fMRI、临床 | 真实数据需 IDA 审批；影像与临床分开导出，需按受试者/访视匹配并核实模态可用性，再做影像预处理。见 [ADNI 官方数据说明](https://adni.loni.usc.edu/data-samples/adni-data/)。 |
| SEED-IV | EEG、情绪标签 | 原始信号 `.mat` 与 DE/PSD 衍生特征分别存放；原始 EEG 需要降采样、滤波、伪迹检查和分窗；已有特征不能再当波形处理。官方同时含眼动，本演示未选择。见 [SEED-IV 官方说明](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)。 |

随包文件**不是这些数据集的真实子集**：每套 12 个合成受试者，MRI 使用 12 个玩具 ROI 和 12³ 体素；EEG 使用 24 个 62×800 窗口（200 Hz、4 秒）。构建脚本确实由合成时序计算 FC、标准化张量并生成划分。未执行真实 BIDS 验证、fMRIPrep、FreeSurfer、空间配准或 EEG 伪迹去除。

准备文件位于仓库 `data/demo/`，安装包中位于 `resources/demo-workspace/data/demo/`。`/data/demo` 是演示工作空间中的逻辑路径，不是 Windows 系统盘根目录。预置 idea 文件在 `resources/demo-workspace/ideas/adhd_network/`。实验文件由播放器完成后导出，不会宣称在播放时真实训练或写出 checkpoint。

数据总计约 10 MB，含原始合成数组、处理后数组、`subjects.csv`、`splits.json`、`qc.json` 和带 SHA-256 的 `MANIFEST.json`。划分为 8/2/2 人，同人所有窗口同组；玩具站点 C 仅在测试组。三个 MRI 队列分开存储。

`load_demo.py` 可用 NumPy 直接加载；`--torch` 转为 Tensor；`--export-pt NEW_DIRECTORY` 转为项目 BrainGNN / BNT / BrainNetCNN 所需字典。实际使用需安装 NumPy/PyTorch 和对应模型依赖。图谱打分 GNN 与实验 BrainGNN 是两个不同角色。

## 与项目设计的对应

- 模式映射：普通=`off`，Idea=`idea`，Data=`data`，Experiment=`model`，Full=`end_to_end`。
- 三视角沿用 `neurooracle/src/critic_agent.py` 的 statistical / clinical / methodological。
- 均衡排序沿用 `neurooracle/src/novelty_policy.py` 的 40% 新颖性 + 20% 结构 + 20% GNN + 20% 评审；类别优先点映射到 100/70/20/0。这是排序效用，不是真实概率。
- 模型输入参考 `skills/brain_gnn/SKILL.md`、`skills/brainnetcnn/SKILL.md`、`skills/bnt/SKILL.md`。Fisher-z 存储，经 tanh 还原 Pearson，清零对角。
- 数据流程参考 `skills/{adhd200,abide,adni,seed-iv}-skill/SKILL.md`。这里只使用契约编写演示，不启动其中的下载或科研工作流。

## 检查与范围

`npm run test:demo:showcase` 检查评分排序、两个 Full 分支、取消/切换/重播、流式文字、离线请求和产物预览，并用隐藏 Electron 窗口实际播放五个场景。20 轮分支使用测试时钟加速，正式播放使用相同逻辑、正常时钟。

`python tests/verify-showcase-data.py` 使用独立的 NumPy 实现重新读取 NPY、验哈希、检查 ID/标签/划分、重算 FC 和 EEG 训练集归一化。截图与 UI 检查记录位于 `tmp/demo-showcase-check/`。

Windows 包未签名；macOS 可用 `npm run dist:demo:showcase:mac` 构建，但本次 Windows 环境不代表已验证 macOS。没有执行真实 LLM、GNN 推理、深度学习训练或科学验证。
