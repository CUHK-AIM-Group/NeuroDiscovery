# NeuroDiscovery workspace design

The desktop shell keeps the existing NeuroDiscovery frontend and NeuroRuntime
backend. The visual layer lives in `core/web/static/research-workspace.css`;
`index.html` retains the application events, model configuration, research
workflows, bilingual content and historical desktop/storage identifiers.

## References reviewed

DeepSeek Harness source was reviewed at revision
`c291e7961a515f6d7af9304e7fd1d257929aef26` on 2026-09-15:

- [AppFrame](https://github.com/deepseek-ai/deepseek-harness/blob/c291e7961a515f6d7af9304e7fd1d257929aef26/packages/client/ui-layout/src/client/AppFrame.module.css): restrained workspace surfaces and separate navigation/content columns.
- [SidebarRoot](https://github.com/deepseek-ai/deepseek-harness/blob/c291e7961a515f6d7af9304e7fd1d257929aef26/packages/client/ui-sidebar/src/client/SidebarRoot.module.css): prominent new-session action, quiet navigation and bottom-pinned settings.
- [HeroShell](https://github.com/deepseek-ai/deepseek-harness/blob/c291e7961a515f6d7af9304e7fd1d257929aef26/packages/client/ui-conversation/src/client/skeleton/HeroShell.module.css) and [InputBar](https://github.com/deepseek-ai/deepseek-harness/blob/c291e7961a515f6d7af9304e7fd1d257929aef26/packages/client/ui-conversation/src/client/skeleton/InputBar.module.css): shared content width, soft composer elevation and compact controls.
- [ModelsSection](https://github.com/deepseek-ai/deepseek-harness/blob/c291e7961a515f6d7af9304e7fd1d257929aef26/packages/client/ui-settings-models/src/client/ModelsSection.module.css): bounded settings width, connection cards, quiet borders and compact actions.
- The user-provided local DSH project's `claude.html`: compact chat layout,
  separated settings and expandable tool records.

The implementation is original HTML/CSS using these interaction and layout
ideas; it does not vendor upstream source, logos, assets, dependencies or runtime.
The upstream project is MIT-licensed. The local DSH reference is read-only.

## Preserved behavior

- Projects, sessions, checkpoints, skill selection, graph and study views.
- Existing model/provider/protocol controls, output limits and credentials flow.
- AutoResearch modes, stopping, request ownership and delivery safeguards.
- Strict / novelty-first / weighted selection and unchanged literature labels.
- English/Chinese, text scaling, themes and desktop IPC/storage compatibility.

Small windows use a dismissible navigation drawer, close it after navigation,
and retain all research shortcuts. Empty chats can scroll in short windows;
ongoing conversations retain their independent transcript and composer.
Reduced-motion preferences and keyboard focus are supported. Theme-token text
contrast and frontend contracts are checked by `core/web/test_workspace_theme.py`.

Run the offline contracts with:

```console
python -m pytest core/web/test_workspace_theme.py -o addopts= -q
```

Visual checks should use an isolated local preview with stub endpoints, never
real credentials, model requests, graph downloads, or research-campaign state.

## 第二轮全面对比（2026-09-15）

对照对象分开处理：上游 DeepSeek Harness 是主参考；本地 `DSH/claude.html`
是额外的 Claude 聊天页，并非完整 DSH 前端。借鉴其布局和层次，不引入其权限模式、
认证配置或运行时。下面的“已调整”指源码，不代表旧安装包已经更新。

| 区域 | 参考设计与此前差异 | 本轮结果 / 保留差异 |
| --- | --- | --- |
| 应用框架 | DSH 用分栏、细分隔线和紧凑工作区 | 顶栏收至 48px，品牌及导航缩小；保留 NeuroRuntime 状态 |
| 侧栏 | 新会话突出，其余导航安静；原界面行高和留白偏大 | 收紧导航与会话行，日期退到键盘焦点状态，设置仍在底部 |
| 空会话首页 | DSH 标志与标题同行，内容居中；此前 ND 欢迎区偏大 | 32px 标志配 26px 标题，统一居中；修复内联 display 覆盖布局的问题 |
| 输入区 | DSH 输入框与正文共享宽度，底部工具紧凑 | 延续共享宽度、柔和边框与轻阴影；保留附件、发送/停止、AutoResearch 和生成参数 |
| 首页快捷任务 | 通用编码入口不适合神经科学研究 | 保留 BIDS、fMRI、FreeSurfer、图谱四个入口，改为低对比度紧凑卡片 |
| 模型入口 | 此前打开聊天菜单会拉取并展示全部远端模型 | 改成设置内显式发现；聊天菜单只读已添加列表，并提供“管理模型”入口 |
| 模型设置 | DSH 用连接卡片；原 ND 将所有参数平铺 | 三步卡片：接入点与密钥 → 拉取并勾选 → 已添加列表；默认不勾选任何新模型 |
| 高级参数 | 常用连接字段与专业参数混在一起 | 协议、思考、输出上限、温度、密钥文件折叠；保留原控制语义，按界面语言显示 |
| 模型管理细节 | 发现结果与用户配置此前没有清晰边界 | 搜索、默认模型、移除、手动 ID、空/错误状态；切换地址清除旧选择与凭据草稿 |
| 消息与工具记录 | 本地 Claude 页有折叠工具记录；ND 有额外研究状态 | 保留现有折叠、流式回答、停止和研究进度，不把研究步骤替换成通用终端记录 |
| 主题与可访问性 | 参考统一字号、边框及状态层次 | 保留 ND 绿色系与脑图标；浅/深主题、文字对比度、长模型名换行和勾选焦点已检查 |
| 小窗口 | 固定双栏会挤压设置表单 | 窄屏设置改单列、侧栏抽屉；在 390px 和 1280px 宽度检查主要模型流程 |

### 仍未对齐的结构差异

- **多接入点卡片并存**：目前是一条活动连接管理多个已选模型，不是同时保存多组独立
  URL / key 的连接管理器。切换接入点会清空旧模型和凭据草稿，不是新增第二条连接。
- **保存后热更新**：ND 仍明确要求保存并重启后应用连接/模型库；本轮没有改为 DSH 式
  连接热更新，也不在研究执行中切换认证配置。
- **右侧文件/终端工作区**：ND 仍使用现有研究视图和产物入口，没有新增 DSH 的完整右栏。
- **输入区的工作目录选择**：后续工作台功能更新已加入目录 chip，可检查真实执行路径，
  并移动会话所属工作区；仍保留 ND 的项目/会话模型，不使用 DSH 的工作区实现。

这是保留 NeuroDiscovery 研究功能的设计适配，不是逐像素复制；以上结构差异不应被
装饰性按钮伪装成已经具备的功能。

### 本轮验证

214 项离线回归测试通过（1 项现有 Starlette 弃用警告）。覆盖真实环境归一化、桌面
发现模块、前端选择状态、协议适配、打包清单及界面契约。隔离 UI 使用合成列表验证
选择性添加、保存/重启、默认模型、移除、搜索、错误和空状态；不读取真实密钥或调用模型。

## 工作台逻辑补齐（2026-09-15）

在保留科研流程的基础上，已加入可恢复的实时执行事件、真实用量账本、设置内的用量
筛选与导出、工作区安全移除和会话置顶/归档/移动。按钮与对话框统一使用明确的忙碌、
危险操作、键盘焦点和错误状态；响应完成不会强制打断用户阅读历史。

实现边界、存储位置、用量口径和离线/隔离界面验证记录见
[Client workbench implementation](CLIENT_WORKBENCH_IMPLEMENTATION.md)。

## NeuroOracle 模块整合（2026-09-15）

NeuroOracle 保留独立文档边界以隔离图谱交互，但不再呈现为另一套应用外壳：

- 主工作台和图谱从 `workspace-tokens.css` 读取同一份深浅色主题；嵌入时隐藏
  重复主题按钮，移除外层圆角网页框，统一标题、字号、按钮、证据卡片与空状态。
- “主张与证据 / 概念图谱”使用固定页签，支持方向键、Home / End 和明确的选中态。
  检索栏可收起；图谱详情在宽窗口停靠、窄窗口使用可关闭抽屉，图谱选项集中折叠。
- 主界面通过限定同源父窗口的消息同步主题、语言和 80%–150% 字号。
  页面往返不再因为语言或主题改变而重载图谱；点击当前会话也能正常返回聊天。
- 图中标签大小和界面文字大小分开控制。Sigma 的标签颜色同步主题，保留的图谱
  实例允许导航期间容器暂时隐藏，恢复后重新测量画布，不丢失当前选择。
- 修复原 ForceAtlas2 CommonJS worker 被当作浏览器脚本载入的问题；同版本
  0.10.1 现在由一个本地构建产物提供，附锁文件和 MIT 许可。布局定时器绑定其
  原始实例，不能停止后来创建的新布局。构建说明见 `core/web/vendor/README.md`。

论文原句、支持/不支持及否定标签、论文身份、支持篇数与独立性提示、来源字段、
严格链开关条件和证据接口保持原有语义；未更新、下载或修订真实图谱。

验证使用独立合成端点，检查主张阅读、非支持证据、概念选择、详情开合、深浅主题、
中文窄窗口、字号与语言同步、聊天往返和本地布局依赖。Graphology 与 Sigma 仍保留
原有版本化 CDN 地址，不把本次整合描述为完整离线图谱支持。离线契约入口是
`core/web/test_oracle_workspace.py`，包含 Node 交互测试，并已纳入默认测试与 CI。
这项更新只涉及源码；没有重新生成客户端安装包，也没有自动提交或推送。

最终相关回归为 470 项通过（1 项已有 Starlette/AnyIO 弃用警告）；其中 NeuroOracle
交互契约包含 10 项 Node 子测试。最终图谱隐藏/恢复、主题切换和布局预览未产生新的
浏览器警告或错误。临时预览服务与标签页验证后关闭。

## 标题与菜单层级压缩（2026-09-15）

针对实际客户端中“系统标题 → 常驻菜单 → 品牌栏 → 模块大标题 → 页签”
叠加的问题，Windows 现在仅保留 48px 应用栏和一行模块页签：

- 使用 Electron 的 `titleBarStyle: hidden` 与原生 `titleBarOverlay`，把窗口标题
  合入现有应用栏。保留原生最小化、最大化、恢复与关闭；按钮区域预留系统安全宽度，
  空白区域可拖动，交互控件排除在拖动区之外。启动页和错误页也保留拖动区。
- 常驻菜单收起，通过应用栏的省略号或原有键盘菜单访问；菜单项及快捷键不删除。
  弹出入口只接受主窗口主 frame 的 IPC。macOS 保留原生标题和系统菜单；Linux
  保留原生标题、自动隐藏菜单，本轮没有宣称已做这两个平台的原生视觉验收。
- NeuroOracle 嵌入模式不显示独立大标题、眉题和介绍。主栏随当前视图显示名称，
  正文直接从“主张与证据 / 概念图谱”页签开始。独立浏览器页面保留标题。
- 图谱更新操作集中到页签行的省略号菜单，仍需用户显式检查、确认更新。
  菜单支持 Escape 和外部点击关闭；窄屏检索按钮以有可访问名称的图标显示。

实现参考 [Electron custom title bar](https://www.electronjs.org/docs/latest/tutorial/custom-title-bar)，
并核对本地 Electron 39 类型定义。通过 computer-use 核验实际 Windows 客户端的
菜单展开/收起、图谱两种视图、最大化/恢复和深浅色同步；恢复浅色后保持客户端打开。
482 项相关回归测试通过（1 项已有 Starlette/AnyIO 弃用警告），NeuroOracle 的 Node
交互契约现为 12 项。标题栏测试加入 CI 和默认测试路径。未调用模型、下载或修改
图谱；未打包、提交或推送。
