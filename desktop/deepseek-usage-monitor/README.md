# DeepSeek 用量监控（自建 Electron 客户端）

自己实现的 DeepSeek API 余额 / 用量监控客户端，参考了
[deepseek-monitor](https://github.com/Leiuo/deepseek-monitor) 的功能，
但只保留明确端点、无任何第三方运行时依赖、凭证不出本机。

## 功能

- **多账号**：支持添加/切换/编辑/删除多个 DeepSeek 账号，顶栏下拉一键切换；
  每个账号独立的 API Key / Session Token 与独立的历史记录（曲线互不混淆）
- 余额查询：总余额 / 充值余额 / 赠送余额 / 账户可用状态（官方接口）
- 消耗明细：按天 Token（缓存命中/未命中/输出）堆叠柱状图、每日费用折线图、
  模型费用占比环形图、模型与按天明细表（平台接口）
- 本地历史记录：每次刷新自动把当前账号的余额/消耗快照追加到本机
  `userData/history-<账号id>.jsonl`（JSONL 追加文件，零依赖；余额每 5 分钟、消耗每 10 分钟至少一条）
- 趋势曲线：余额趋势（近 30 天）、本月消耗累计曲线——数据来自本地历史，
  即使平台接口暂时不可用也能查看已记录的部分
- 月份切换：查看任意月份
- 日间/夜间模式：顶栏一键切换，选择持久化保存，图表配色跟随
- 自动刷新（10–3600 秒可调）
- 低余额系统通知（阈值可调）
- 连接测试：一键验证 API Key / Session Token
- 设置页可一键清空当前账号的历史数据（清空后曲线从下次刷新重新积累）

> 旧版（单账号）配置首次启动自动迁移为账号列表，历史数据一并保留。

## 运行

```powershell
cd desktop/deepseek-usage-monitor
npm install        # 首次
npm start          # 启动客户端
```

启动后在「设置」页填入凭证：

| 配置项 | 用途 | 获取方式 |
|---|---|---|
| 账号（API Key） | 查询余额 | platform.deepseek.com → API Keys（每个账号一个） |
| 账号（Session Token，可选） | 查询消耗明细 | 浏览器登录 platform.deepseek.com → F12 → Network → 任意请求的 `Authorization: Bearer xxx` |

## 安全设计

- **零远程内容**：所有页面/脚本/图表库均为本地文件，CSP 禁止任何网络连接（`connect-src 'none'`）；
- **网络只在主进程**：渲染进程（sandbox + contextIsolation + nodeIntegration:false）无网络与 Node 权限；
- **域名白名单**：只访问 `api.deepseek.com` 与 `platform.deepseek.com`，外链仅限官方域名并交给系统浏览器；
- **凭证保护**：凭证保存在 `userData/config.json`，Windows 下用 `icacls` 锁定为仅当前用户可读写；
  凭证不会回传渲染进程（界面只显示"是否已配置"），错误信息自动打码；
- **禁止导航**：`will-navigate` 全部拦截，不加载任何外部页面。

## 数据接口

| 数据 | 端点 | 凭证 |
|---|---|---|
| 余额 | `GET https://api.deepseek.com/user/balance`（官方公开） | API Key |
| 消耗明细 | `GET https://platform.deepseek.com/api/v0/usage/amount?month=&year=`<br>`GET https://platform.deepseek.com/api/v0/usage/cost?month=&year=`（平台内部，未公开文档） | Session Token |

⚠️ Session Token 等同网页控制台登录凭证，请勿提供给任何第三方工具或网站。
用量明细接口为未公开的内部接口，DeepSeek 调整时可能失效，届时会给出明确的 HTTP 错误提示。

## 项目结构

```
deepseek-usage-monitor/
├── main.js           # 主进程：窗口、IPC、凭证存储(ACL)、数据抓取、通知、烟雾测试
├── preload.js        # contextBridge 暴露最小 API
├── history.js        # 本地历史存储：JSONL 追加 + 内存缓存 + 降采样 + ACL 保护
├── renderer/
│   ├── index.html    # 页面（含 CSP）
│   ├── styles.css    # 暗色主题
│   └── app.js        # 图表与交互逻辑
└── package.json
```

## 开发 / 测试

```powershell
npm run smoke        # 烟雾测试：加载窗口 → 校验 Chart.js 与桥接层 → 自动退出
```

## 待办（后续可加）

- 系统托盘与开机自启
- 打包为安装程序（electron-builder）
- 历史文件按周轮转（当前单文件追加，约 1MB/月，长期使用后再考虑）
