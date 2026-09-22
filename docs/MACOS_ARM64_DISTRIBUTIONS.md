# macOS Apple Silicon：三个桌面发行版

三个版本复用 Electron 桌面框架，目标为 **原生 ARM64 macOS**，不是浏览器版或 WKWebView 版。

| 变体 | 应用名称 | 内容 |
| --- | --- | --- |
| `full` | NeuroDiscovery | 常规桌面功能 + Human Evaluation 1/2 |
| `evaluation` | NeuroDiscovery Human Evaluation | 仅冻结的 HE1/HE2 参与者材料、作答、保存恢复及导出；无研究/模型/KG API |
| `demo` | NeuroDiscovery Demo | 常规桌面功能；不包含 HE 材料、评估专用模块和入口，HE 路由返回 404 |

“完整”指桌面功能集合，不包含大型知识图谱、科研数据集、模型权重或全部可选科研依赖。真实模型调用、神经影像分析等仍需单独配置凭据、数据和工具。HE1 保留既有 pilot 状态，不改变材料的科学验收等级。

## 当前状态

Windows 可验证暂存文件、路由边界和构建逻辑；**不能据此声称 macOS 二进制、GUI、签名或公证已通过**。须在 Apple Silicon Mac 上生成 DMG/ZIP，并完成文末验收。

## Mac 构建环境

1. 安装 Xcode Command Line Tools：`xcode-select --install`。
2. 安装原生 ARM64 Node.js（建议 Node 22 或 24）及 ARM64 Miniforge。
3. 创建构建驱动环境，并确认没有使用 Rosetta：

```bash
conda create -n neurodiscovery-build python=3.12 packaging -y
conda activate neurodiscovery-build
python3 -c 'import platform; print(platform.machine())'
node -p process.arch
```

两项架构输出都应为 `arm64`。构建驱动要求 Python 3.12+；应用内另外创建独立 Python 3.11 前缀，不使用本机已有开发环境或 Windows runtime。`CONDA_SUBDIR` 如已设置，必须为 `osx-arm64`。

在仓库根目录：

```bash
cd desktop
npm ci
npm run dist:mac:arm64:all
```

也可单独构建：

```bash
npm run dist:mac:arm64:full
npm run dist:mac:arm64:evaluation
npm run dist:mac:arm64:demo
```

`dist:mac:arm64` 是 full 的别名。旧的 `dist:mac`、`dist:mac:x64`、`dist:mac:skip-runtime` 不属于本三版本流程，不要混用或把 Windows runtime 直接交给这些命令。

构建需要联网下载 conda、pip、Electron 及打包工具依赖；不调用模型、执行科研任务或迁移 KG。各次 pip 解析版本记录在 `PYTHON_PACKAGES.txt`，这不是完全锁定依赖的可复现构建承诺。

## 输出和重试

默认输出：

```text
desktop/dist-mac-arm64/full/artifacts/*.dmg、*.zip
desktop/dist-mac-arm64/evaluation/artifacts/*.dmg、*.zip
desktop/dist-mac-arm64/demo/artifacts/*.dmg、*.zip
```

每个变体独立暂存后端/Python、生成 electron-builder 配置、运行含空格目录的 Python 搬迁检查和离线后端冒烟测试，最终记录文件大小与 SHA-256 到 `BUILD_RECEIPT.json`。GUI 状态始终标为 pending，不自动授予实机验收。

脚本拒绝覆盖既有变体目录，失败时保留诊断资料。重试用全新的输出根，不需删除旧结果：

```bash
npm run dist:mac:arm64:all -- --output-root ./dist-mac-arm64-r2
```

`all` 按顺序构建；中途失败不会删除之前成功的变体。只做运行时准备可使用：

```bash
python3 scripts/build-mac-arm64.py --variant evaluation --prepare-only \
  --output-root ./dist-mac-arm64-prepare
```

准备成功后，可用生成配置手动打包（这条手动命令不会生成脚本的最终 BUILD_RECEIPT）：

```bash
CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder \
  --config dist-mac-arm64-prepare/evaluation/electron-builder.json --mac dmg zip --arm64
```

## 用户数据与运行时

Electron 配置目录分别位于 `~/Library/Application Support/NeuroClaw`、`NeuroDiscovery-Human-Evaluation` 和 `NeuroDiscovery-Demo`。full 沿用既有目录以保持兼容。评估专版的答案位于自己目录下的 `evaluation-data`；full 的研究评估存储沿用既有 `~/.neurodiscovery` 规则，两者并非同一数据库。

三个版本均在可写用户目录运行搬迁后的 Python；不在 `.app` 包内执行 conda-unpack。评估专版按运行时清单指纹复用成功缓存，升级缓存不删除答案。full/demo 复用现有桌面缓存机制。评估专版端口为 17890；端口被占用时拒绝启动，不接管陌生进程。

## 签名与分发

当前脚本显式关闭 Developer ID 自动签名，生成用于内部验收的未公证包（ARM64 工具可能施加 ad-hoc 签名，这不等于可信分发签名）。Gatekeeper 可能阻止下载包；仅对可信测试构建使用 macOS 提供的单应用授权，不要全局关闭安全策略。公开分发前需另外配置 Apple Developer ID、完整嵌套代码签名、entitlements、公证和 stapling，并重新验收。

## Mac 实机验收清单

- 分别从 DMG 和解压 ZIP 启动，移动应用到含空格目录后再启动；确认无开发机路径依赖。
- full：正常聊天页面、模型设置、技能入口和原生菜单可用；无需为冒烟测试调用付费模型。
- full/evaluation：HE1/HE2 材料正确，作答保存、关闭重开恢复、双评估 JSON 导出和原生保存对话框正常。
- evaluation：只显示评估，研究、模型和 organizer API 不存在；关闭窗口后所属后端退出。
- demo：常规功能可用，评估入口/材料/API 均不可用，不能只检查“按钮隐藏”。
- 分别检查端口冲突、重复启动和升级缓存；不得丢失既有答案或终止无关服务。
- 记录 macOS 版本、芯片、各产物 SHA-256 和结果；通过后才作为验收包分发。

## Git 交付边界

本流程不自动 commit/push。向 Mac 交付前，必须确认源码、`desktop/evaluation-materials` 中的参与者投影及哈希清单、三份 HE2 冻结材料、runtime helper allowlist 及所列源码全部包含在交付中。干净检出不需要私有作者材料：暂存脚本在作者输入不存在时验证并复制已冻结的公开投影。不要使用宽泛 `git add .`：本仓库另有未提交科研工作。不要提交生成 runtime、答案数据库、凭据、权重或 KG 数据。新默认输出及 `dist-mac-arm64-*` 重试目录已被 desktop `.gitignore` 排除。
