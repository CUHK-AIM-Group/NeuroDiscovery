# Human Evaluation 专用桌面包

这是新增的独立打包方式，不替换完整版。Windows 产品名为
**NeuroDiscovery Human Evaluation**，应用 ID 为
`org.neurodiscovery.human-evaluation`。

## 功能范围

- Human Evaluation 1：冻结的 v16 十案例材料、相关文献、中英文界面、评审与计分。
- Human Evaluation 2：冻结的 v3 题库、v4 文献说明及分配表、两两比较和六次限时会话。
- 本地保存、退出后继续，以及两项评估统一导出 JSON。两项评估需使用相同名字/代号。
- 不启动聊天、模型、密钥管理、KG、agent、研究执行或主办方结果导入接口。
- 材料、题库、评分和分配规则沿用原实现，不重新生成科学结果。HE1 当前仍标记为 pilot。
- 本地阅读和填写不需要联网；打开外部论文链接需要网络。文献链接不是离线论文全文。

## 构建 Windows x64

在 `desktop` 目录中，使用已准备好的完整版构建 Python（只读使用，不覆盖它）：

```powershell
npm ci
npm run dist:evaluation:win
```

构建 Python 需包含 `packaging`、`fastapi`、`uvicorn` 及其依赖。准备脚本只复制
Python 标准库、必要原生 DLL 和 FastAPI/Uvicorn 依赖闭包，不复制完整科学计算环境。
`evaluation-manifest.json` 记录后端文件 SHA256 和依赖版本。

独立 runtime 默认位于 `desktop/runtime-evaluation`，不覆盖 `desktop/runtime`。
已有非空 runtime 会被拒绝覆盖。同一批源文件的校验续用方式：

```powershell
npm run prepare:evaluation:win:resume
npm run dist:evaluation:win:skip-runtime
```

源文件变更时，请保留旧目录并用 `--target` 准备新目录，再调整专用 builder 配置的
`extraResources.from`；不要将新旧材料混合覆盖。`--resume` 会核对当前源文件、
已有 manifest 和完整后端清单，差异时拒绝继续。`skip-runtime` 仅用于已校验的 runtime。

开发启动：`npm run dev:evaluation`。仅生成解包目录：`npm run pack:evaluation`。
原 `dev`、`pack`、`dist:win`、`dist:mac` 等完整版命令保持不变。
专用 runtime 的自动准备目前仅支持 Windows；macOS 未构建或验收。

## 输出与数据隔离

### 系统 WebView2 轻量试用版（2026-09-22）

当前交付 ZIP：
`desktop/dist-evaluation-webview-r3/NeuroDiscovery-Human-Evaluation-1.0.0-WebView-win-x64.zip`

实测 16,776,531 字节（16.00 MiB），不含整个浏览器；与系统浏览器版相比增加约 0.25 MiB。
解压后双击 `NeuroDiscovery Human Evaluation.exe`，在原生窗口中使用系统 WebView2 Runtime。
不显示控制台；保存后关闭窗口会关闭其拥有的后端，父进程异常退出也通过管道 EOF 通知后端停止。
系统保存对话框接管 JSON 下载；外部 HTTPS 论文链接使用系统浏览器。

需要 Windows x64、.NET Framework 4.6.2+、已安装的 Microsoft Edge WebView2 Runtime。
本机实际使用 Runtime 153.0.4234.48；缺失时报告错误，不自动安装或把整个 Runtime 打入 ZIP。
使用官方 NuGet SDK 1.0.4191.47，构建脚本固定归档 SHA256，产物附 SDK LICENSE/NOTICE。
此 EXE 未签名。详情见 `desktop/EVALUATION_WEBVIEW_README.md`。

数据和 WebView 用户配置独立位于 `%APPDATA%\NeuroDiscovery-Human-Evaluation-WebView`。
不迁移其他版本或历史备份中的答案。服务只绑定 `127.0.0.1:17893`，端口占用时拒绝启动；
不复用外部服务、不结束其他应用。此端口与系统浏览器版 17892、Electron 版 17890 分离。

构建：`npm run dist:evaluation:webview`（`desktop` 目录）。需要已下载的固定 NuGet 归档，
默认在 `desktop/build/webview2-sdk-1.0.4191.47`；脚本不自动下载依赖，已有输出拒绝覆盖。
需要重新构建时为 `scripts/package-evaluation-webview.py` 指定新的 `--output`。

实际 WebView2 测试通过 HE1/HE2 保存恢复、主办方入口隐藏、JSON 下载及双评估记录导出。
对 ZIP 解压产物再次检查 1,236 个 SHA256 并重复 UI 测试，验证窗口关闭后后端退出、17893 释放。
24 项 Python 测试通过，包括父进程管道关闭、端口冲突拒绝和原独立评估 API 回归。
UI 测试：运行 EXE `--smoke-test`，使用临时隔离数据；通过回执位于临时目录
`evaluation-webview-test-*/SMOKE_PASSED.json`。不要在真实 WebView 评估窗口开启时执行。

最初 `dist-evaluation-webview` 和 `dist-evaluation-webview-r2` 为未验收试制输出，不用于分发。
原烟雾测试在 DOM 元素出现后、延迟脚本绑定前点击按钮，已改为等待页面 complete；
frame 导航允许初始化用的 about:blank，随后仍限制为同源。

### 系统浏览器轻量试用版（2026-09-22）

新增不含 Electron 的 Windows x64 ZIP，保留原桌面包，不覆盖已有 runtime 或导出产物：

`desktop/dist-evaluation-browser/NeuroDiscovery-Human-Evaluation-1.0.0-Browser-win-x64.zip`

本次实测 16,514,268 字节（15.75 MiB），相对原 141.01 MiB ZIP 减少约 88.8%。
包含精简 Python 和原独立包 manifest 校验通过的冻结后端/材料；不包含浏览器、KG 或用户答案。
解压整个目录后双击 `Start Evaluation.cmd`，使用默认系统浏览器访问 `http://127.0.0.1:17892`。
浏览器 JSON 下载替代 Electron 保存对话框。保持控制台开启，保存后按 Ctrl+C 停止服务；
仅关闭网页不会停止服务。端口冲突时拒绝启动，不复用或终止其他服务。

答案独立保存到 `%APPDATA%\NeuroDiscovery-Human-Evaluation-Browser\evaluation-data`。
恢复引用依赖浏览器本地存储，继续时须使用相同浏览器、用户配置和地址；不使用隐私模式。
此试用版不迁移 Electron 版或完整版答案。详见 `desktop/EVALUATION_BROWSER_README.md`。

构建：在 `desktop` 下运行 `npm run dist:evaluation:browser`。
输出已存在时拒绝覆盖；再次构建需通过脚本 `--output` 指定新目录。
脚本为 ZIP 生成 CRC 检查、SHA256 构建回执及逐文件 manifest。

验证：22 项 Python 测试通过（启动器、冻结后端边界、评估 API、打包和导出）。
实际系统 Edge 无头测试通过 HE1/HE2 保存与恢复、HE1 中英文切换、主办方入口隐藏、
无 Electron 桥的真实 JSON 下载。另将交付 ZIP 解压到临时目录，校验 1,230 个文件 SHA256，
并对解压产物重复通过 Edge 测试。测试使用隔离临时记录，不污染真实用户数据。
这不代表所有浏览器或所有历史回归套件均已验收。

复验：`node desktop/tests/evaluation-browser-smoke.cjs [解压后的软件目录]`。
默认使用本机 Edge，可通过 `EVALUATION_TEST_BROWSER` 指定其他 Chromium 浏览器。

### 原 Electron 桌面版

输出目录为 `desktop/dist-evaluation`：

| 产物 | 用途 | 本次大小（约） |
| --- | --- | --- |
| `NeuroDiscovery-Human-Evaluation-1.0.0-Setup-x64.exe` | 安装程序 | 101.4 MiB |
| `NeuroDiscovery-Human-Evaluation-1.0.0-Portable-x64.exe` | 便携程序 | 101.2 MiB |
| `NeuroDiscovery-Human-Evaluation-1.0.0-win-x64.zip` | 解压运行 | 141.0 MiB |

这些产物当前未签名，Windows 可能提示未知发布者；未执行系统安装/卸载验收。

浏览器配置与答案独立存放于
`%APPDATA%\NeuroDiscovery-Human-Evaluation`，其中答案在
`evaluation-data\discovery` 和 `evaluation-data\ranking`。
便携版也使用这一持久目录，不将答案写入临时解压目录。不迁移、不清理完整版旧答案。

独立后端仅绑定 `127.0.0.1:17890`，不占用完整版的 7080。端口冲突会报错，
不会复用或终止其他服务。只关闭本应用启动的 Python 子进程。
Host/Origin、浏览器跨站请求和 iframe 限制用于本机浏览器隔离，不是多用户身份认证；
具有本机账户权限的人仍可访问本地评审数据。

后端仅包含 29 个白名单文件，包括五个 HE1 发布材料和三个 HE2 材料文件。
不包含 organizer truth 文件、收集的答案、密钥、源审计缓存或 KG 数据库。
HE2 冻结评分仍保留以维持原协议/哈希绑定；此包不是防止参与者逆向读取材料的安全边界。

## 验证（2026-09-22）

- 30 项针对性 Python 测试通过：专用 API、安全边界、两项评估持久化/恢复、
  HE1 保存/提交/计分、HE2 提交、统一导出以及现有材料打包回归。
- Node 原生导出桥契约测试通过；专用 JS 语法检查通过。
- 隐藏 Electron 窗口完成实际页面测试：主页、中英文切换、HE1 保存退出后恢复、
  HE2 创建/保存/恢复，主办方入口不可见。测试使用独立临时资料目录，不触及真实答案。
- 精简 Python 与产物中的 Python 均能加载 SQLite/SSL；实际专用后端启动成功。
- Windows Setup、Portable、ZIP 构建成功；`app.asar` 仅含专用 main/preload 和 package.json，
  resources 不包含完整版 runtime。
- 额外现有回归套件 49 项中 47 项通过；两项既有问题未在本任务修改：
  v16 测试使用少于 8 字符的请求 ID，被已有校验拒绝；翻译覆盖测试缺少“数据预处理”映射。
  这不等于所有历史套件全部通过。

复验命令（仓库根目录）：

```powershell
desktop/runtime/python/python.exe -m unittest core.web.test_evaluation_app desktop.tests.test_evaluation_packaging core.web.test_evaluation_export desktop.tests.test_discovery_packaging -q
node --test desktop/tests/evaluation-export.test.cjs
desktop/node_modules/.bin/electron.cmd desktop/tests/evaluation-ui-smoke.cjs
```

UI 测试保留并打印其临时目录；不自动删除任何已有参与者记录。
