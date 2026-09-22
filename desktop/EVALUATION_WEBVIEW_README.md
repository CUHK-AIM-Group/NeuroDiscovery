# Human Evaluation · 系统 WebView2 试用版

将整个 ZIP 解压后，双击 `NeuroDiscovery Human Evaluation.exe`。
使用 Windows 已安装的 Microsoft Edge WebView2 Runtime，在独立窗口中显示评估页面。
不需要保持控制台打开；保存后关闭窗口，本应用启动的本地后端也会退出。
导出 JSON 时使用系统保存文件对话框。外部 HTTPS 论文链接在系统浏览器打开。

## 系统要求

- Windows x64、.NET Framework 4.6.2 或更高版本，以及 Microsoft Edge WebView2 Runtime。
- WebView2 Runtime 在多数新版 Windows 上已存在，但不是每台电脑都有。
- 缺失时本程序报告启动失败，不自动安装软件。可从微软官方下载：
  https://developer.microsoft.com/microsoft-edge/webview2/
- ZIP 不包含整个浏览器或 WebView2 Runtime，因此体积较小；首次缺少 Runtime 的电脑需单独安装。
- 无需安装 Python；本包包含精简 Python。首次使用可能遇到 Windows 未签名程序提示。

## 数据隔离

答题数据库：`%APPDATA%\NeuroDiscovery-Human-Evaluation-WebView\evaluation-data`。
恢复引用：同一目录下的 `browser` 用户配置。请不要清除该目录，以免丢失恢复信息。
两项评估使用相同名字/代号。退出前先保存，评估完成后导出 JSON 交给研究者。
不会迁移、读取或覆盖 Electron 版、系统浏览器版、完整版或历史备份中的记录。
固定题库、材料、评分和分配与原独立评估包一致；HE1 仍是 pilot。

本地服务仅监听 `127.0.0.1:17893`。端口被其他应用占用时拒绝启动，不复用或关闭他人服务。
只停止本窗口拥有的后端；窗口进程异常退出时管道关闭也会通知后端停止。
本机账户可以访问本地服务和磁盘数据，这不是多用户安全隔离或防逆向措施。
后端意外停止时会提示重新打开，已保存的数据仍保留。

## English

Extract the entire ZIP and open `NeuroDiscovery Human Evaluation.exe`.
Requires Windows x64, .NET Framework 4.6.2+ and the installed Microsoft WebView2 Runtime.
No Python installation is required. Save before closing the window; closing also stops the owned backend.
Use the same participant code in both evaluations. Export JSON through the Save dialog.
This edition keeps records separately from the Electron and system-browser editions.
