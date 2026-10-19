# 图谱与证据的统一安装

概念图、Claims & evidence 和图谱驱动的 AutoResearch 共同选择一个版本。默认使用正式发布的图谱，不按文件修改时间选择新实验结果，也不依赖已清理的旧研发 campaign。当前正式版本由 `neurooracle/data/full_v2/CURRENT_STATE.json` 指定。

正式图谱的 claim 节点已保存论文书目、提取文本、统计字段及 Case Study 标记。`core/formal_graph_evidence.py` 直接读取这些原始记录；单独的 `extracted_claims.jsonl` 审计库不再是浏览这些记录的安装前提。原始记录不等于新近核验的论文身份、获审命题支持或多论文共识，界面和研究检索保留这个区别。

## 安装选择

默认安装入口为 `<backend-root>/neurooracle/data/graph-installation.json`。它指向同一目录中的 `graph-packages/<manifest-sha256>/manifest.json`，使用相对路径。

可以用 `NEUROCLAW_GRAPH_INSTALLATION` 指定其他位置的安装入口或包目录。移动时保留入口和整个 `graph-packages` 目录的相对结构即可。包内的原始来源路径是不可变溯源信息，实际读取通过清单解析，不要求目标机器具有原研发目录。

选择顺序为：已选安装包、显式 `NEUROCLAW_KG_CAMPAIGN` 实验覆盖、本地正式 CURRENT_STATE、历史配置或演示下载。显式实验覆盖保留原有校验规则；默认正式版本存在时，历史 `graph_explorer.json` 不会覆盖它。选定正式文件、安装包或显式 campaign 损坏时会报告错误，不悄悄回退到旧图。v3 的筛选和暂存产物不参与正式版本选择。

正式来源可以是原始 CURRENT_STATE，也可以是公开快照原有的 `neurodiscovery.graph-publication.v1` manifest。两者按各自原始字段绑定实际 graph 的字节数和 SHA-256，不补造已删除的 campaign 或验收记录。包使用 `source_kind: formal_current`，包含原字节的正式清单和 graph；旧的完整验收 campaign 包仍受原证据读取器校验。

普通关键词搜索名称、谓词和论文书目，不匹配随机技术 ID 片段；完整 claim 或端点 ID 仍支持精确查询。

首次检索会在系统临时目录的 `neurodiscovery-kg-cache` 中建立 SQLite 缓存，保存节点字节位置与检索字段。缓存绑定 graph SHA-256、实际 claim 数和缓存文件 SHA-256；返回的完整观察仍从原 graph 读取。缓存不会写入 full_v2，也不是新的科研审核。`minimum_papers > 0` 仍表示获审支持论文门槛，不能用原始来源记录填充获审支持计数。

## 从正式版本制作包

```powershell
python -m core.graph_installation export --state 'neurooracle/data/full_v2/CURRENT_STATE.json' --output '<new-formal-package-directory>' --release-id 'full_v2_20260906_bbccd816de32'
python -m core.graph_installation verify '<new-formal-package-directory>/manifest.json'
```

图谱与正式清单按原字节复制。安装后的相对位置替代原开发机器的绝对文件位置，图谱身份仍由原清单中的内容 hash 决定。该导出不新增声明、不重做科学审核、不更改正式选择器。

## 从已验收来源制作包

在具有完整验收数据的维护环境执行，输出目录必须尚不存在：

```powershell
python -m core.graph_installation export --campaign '<accepted-campaign.json>' --output '<new-package-directory>' --release-id '<release-id>'
python -m core.graph_installation verify '<new-package-directory>/manifest.json'
```

导出器先使用现有 AcceptedClaimLayer 检查来源，再递归收集活跃文件中的完整指纹依赖。原始 graph、campaign、验收记录、索引和来源文件按原字节复制；不改写旧 JSON 内的路径、不补造审核记录、不修改正式图谱。缺失、变化或冲突的依赖会阻止导出。

包清单保存原文件的路径、字节数、mtime 和 SHA-256，并为其分配包内相对文件路径。安装后的物理时间戳可以不同，源指纹身份仍需匹配，并校验实际内容 SHA-256。该映射仅在一次包读取的上下文中生效，普通研发和冻结实验的存储解析不受影响。

## 制作下载文件

```powershell
python -m core.graph_installation pack '<package-directory>/manifest.json' --output '<new-graph-package.zip>' --remote-path 'neurooracle/data/releases/<release-id>/graph-package.zip'
```

ZIP 使用 Zip64，支持大于 4 GiB 的文件。命令同时生成 `<archive>.release.json`，绑定 archive 大小、archive SHA-256 和 package manifest SHA-256。重复命令不会覆盖旧 archive。

客户端下载协议读取 Hugging Face Space 内的 `neurooracle/data/releases/desktop/latest.json`。发布时，该文件应使用生成的 release descriptor 内容，其 archive 路径指向同仓库中实际上传的包。没有实际完整包时不能先发布一个不可用入口。

若该入口返回 404，下载器读取已经公开的 `neurooracle/data/releases/full_v2_20260906_bbccd816de32/manifest.json`，下载其完整 `knowledge_graph.json.zst`。安装前核对压缩文件大小与 SHA-256，解压时再次核对完整 graph 的大小与 SHA-256，并保留原 publication manifest 字节。无需为原始声明浏览另行发布旧研发依赖。

只有两个正式发布入口都不存在时才保留历史演示下载兼容路径；这种情况下明确显示证据未安装。已经安装的完整包不会被演示下载覆盖。公开正式快照约 693 MB，解压后的 graph 为 10,278,527,239 字节；首次内容核验和建立检索缓存需要额外磁盘读取与缓存空间。

## 本地安装

```powershell
python -m core.graph_installation install '<graph-package.zip>' --data-directory '<backend-root>/neurooracle/data'
```

安装先在新临时目录解包，拒绝越界路径、重复目标和符号链接，验证全部文件及现有证据验收，再原子切换安装入口。下载 hash 或证据检查失败时保留旧入口。后端后续请求会识别新包，不需要通过重发研究任务来刷新数据。

也可以将包安装到其他目录，然后让 `NEUROCLAW_GRAPH_INSTALLATION` 指向该目录中的 `graph-installation.json`。这不会改变正式 `full_v2/CURRENT_STATE.json` 或 benchmark 的固定输入。

## 接口和显示

- `/api/neurooracle/graph/status` 提供实际选择的 graph path、包 release、graph revision 和 evidence 配置状态。
- `/api/kg/*` 证据接口、`core.topic_evidence.default_layer()` 和 CLI/MCP 的 `neurooracle.src.graph_paths` 默认解析使用同一安装选择。显式 `--graph` 冻结输入不改绑，缺失时仍报错。正式版本返回 `source: formal_current_graph`、`evidence_kind: original_extraction`；来源审核状态保持未审核，不能将 Case Study 归属审核改作命题支持审核。
- 正式版本的声明页默认显示全部原始提取记录，保留查看来源书目、原字段、统计及导出的能力；原有已审核共享主张包仍按获审支持条件显示。
- 缺少证据数据时返回 HTTP 503 和 `code: evidence_not_installed`，界面说明缺失数据，不提供无效的重复读取提示。
- 已配置但损坏、更新中或校验失败的证据仍按原规则拒绝读取；不会把未审核的概念图记录提升成已审核证据。

## 验证边界

回归测试同时使用合成正式图和既有验收绑定图，覆盖过期配置、换目录、中文路径、原来源目录不可用、Unicode 与流式边界、原文和计数一致、概念图真实加载、研究检索、压缩及解压 hash、下载切换和失败保留旧版本。

合成测试包不能作为真实科研数据发布。2026-10-19 已额外用本地正式 graph 完成全文件 SHA-256 检查和全部 905,274 条 claim 的检索索引构建，并实际查询原始声明；记录见 `CLIENT_GRAPH_INSTALLATION_REPAIR_20261019.md`。这属于读取与检索完整性验证，不是新的命题支持或论文科学准确性验收，也不代表完整 AutoResearch 已进行模型实跑。大型图谱首次 hash 检查和加载会进行较多磁盘读取；后端将相关读取放入工作线程，避免阻塞事件循环。
