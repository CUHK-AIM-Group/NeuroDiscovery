# Human Evaluation · 系统浏览器版

1. 将整个 ZIP 解压到一个文件夹，不要在压缩包内直接运行。
2. 双击 `Start Evaluation.cmd`，默认浏览器将打开 `http://127.0.0.1:17892`。
3. 保持启动时的控制台窗口打开；若浏览器没有自动打开，手动访问该地址。
4. 两项评估使用同一个名字/代号。先保存，再退出；JSON 导出使用浏览器下载功能。
5. 结束时，先保存，再回到控制台按 Ctrl+C。仅关闭网页不会停止本地服务。

无需安装 Python、Electron 或模型环境。需要 Windows x64 和现代浏览器，建议 Edge/Chrome。
材料和答题支持离线使用，外部论文链接仍需联网。本包不包含离线论文全文。

## 记录与恢复

答案保存在 `%APPDATA%\NeuroDiscovery-Human-Evaluation-Browser\evaluation-data`。
恢复会话的引用保存在所使用浏览器的本地存储；请一直使用同一个浏览器、用户配置和地址，
不要使用隐私模式或清理站点数据。更换浏览器不会自动带入恢复引用。
此试用版不迁移 Electron 版或完整版记录，也不会读取之前清空的记录备份。
即使移动或删除解压后的软件文件夹，已保存的答案也不会自动删除。
导出 JSON 后，请检查浏览器下载列表并将文件交给研究者。

## 本机服务边界

服务只监听 `127.0.0.1`，不向局域网开放。端口被占用时拒绝启动，不复用、不终止其他服务。
重复双击不会另开第二个服务；在原浏览器页面继续即可。
没有登录鉴权的本地服务不是多用户隔离机制；其他具有本机账户权限的人仍可能访问记录。
不要将服务转发至公网。结束评估后请按 Ctrl+C 关闭。

冻结题库、评分、分配和评估流程沿用现有独立桌面包；HE1 仍是 pilot。
这不是防止参与者逆向读取材料的安全边界。

## English quick start

Extract the entire ZIP, then double-click `Start Evaluation.cmd`. Keep the console open.
Use the same browser/profile and participant code for both evaluations. Save before leaving.
Exports are regular browser JSON downloads. After saving, press Ctrl+C in the console to stop.
Closing the browser tab alone does not stop the local server. No Python installation is required.
