# NeuroDiscovery Demo · Windows x64

本包面向演示使用，不提供 Human Evaluation 1/2、评估问卷、评分、参与者材料或评估 API。常规聊天、Skills、项目和 NeuroOracle 入口保留。正式评估版不受影响。

## 启动

- 推荐：双击 `NeuroDiscovery-Demo-1.0.0-Portable-x64.exe`。首次启动需要解压内置运行环境，请等待。
- 备用：完整解压 `NeuroDiscovery-Demo-1.0.0-win-x64.zip` 到可写目录，再启动其中的 `NeuroDiscovery Demo.exe`。不要只拷贝 EXE。
- 包含 Python，无需安装 Python/Conda。仅支持 Windows x64，不是 macOS 包。
- 发布文件未附发行者签名；只运行可信来源且 SHA-256 与随包校验文件一致的文件，遵循所在单位安全策略。

## 演示前准备

1. 打开 Settings，配置演示者自己的模型服务、模型名、URL 和 API key，按提示重启。未配置时显示 Setup required；不会附带作者的密钥。
2. 可先输入 `/help` 检查本地交互，无需调用模型。
3. 真实聊天/生成需要可用模型服务及网络。演示者需自行承担其模型服务费用。
4. 包中没有完整知识图谱、研究数据、实验结果或聊天记录。图谱和数据处理演示需要提前提供相应数据及外部工具；不要将空白安装视为可离线完成全部科学流程。

## 隔离与退出

- 配置、缓存和日志使用 `%APPDATA%\NeuroDiscovery-Demo`，不复用常规版的 `%APPDATA%\NeuroClaw`。
- 不连接常规评估后端；默认端口被占用时会尝试其他可用端口。
- 关闭客户端即可停止由本客户端启动的后端。
- 本次打包未重建、修改或发布知识图谱，未进行外部模型推理。

## 检查范围

已测试 Demo API 禁用、保留版路由、Windows/macOS 菜单构造、打包规则、内置 Python 后端启动、首页/Settings 和评估视图拦截。macOS 菜单测试不代表提供或测试了 macOS 安装包。没有验证真实模型推理或完整科研工作流。

现有普通版 multi-topic HTTP 测试夹具缺少当前服务要求的三个 revision 字段，单独执行该历史测试仍失败；Demo 不走该兼容性分支。新增 Demo 兼容性测试单独通过。
