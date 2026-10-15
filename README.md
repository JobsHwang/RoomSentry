# RoomSentry

隐私优先、本地优先的家庭视频事件分析原型。把视频与可选硬件信号整理为可追溯事件、时间线和日报，帮助用户快速回看，而不是持续盯着监控。

> 实验性项目。婴幼儿检测、抱持遮挡、状态解释和连续追踪尚未通过可靠性验收。本项目不是医疗设备、生命安全系统或成年人照护的替代品。当前不推荐无人值守部署。

项目自有代码与文档采用 Apache-2.0；第三方模型、运行时、数据和摄像头桥接适用各自条款。公开仓库不提供家庭素材、私人标注、运行数据库或模型权重。

## 能做什么

- 通过适配器接入本地视频、RTSP 和可选 Home Assistant 信号。
- 用有界缓冲、检测和匿名短期追踪生成候选片段与关键帧。
- 按需调用一个本地视觉模型，输出严格 JSON，区分可见事实与模型推测。
- 在 SQLite 中保存事件、证据引用、队列和失败状态。
- 提供本地时间线、Markdown/HTML 日报、精华与人工复核入口。
- 提供默认关闭的文本通知、远端复核与独立实验旁路。

检测到 `person` 不代表识别到宝宝；轨迹编号不是身份。缺失、遮挡、过期和断流应明确区分，证据不足时输出 `unknown`，不能解释成无人或安全。

## 成熟度

| 范围 | 状态 |
|---|---|
| 契约、队列、SQLite、报告与合成测试 | 已实现；可运行标准库测试 |
| 文件/RTSP 媒体、检测器与模型适配器 | 已实现可选入口；实际环境需独立验证 |
| 宝宝独立检测、抱持理解、连续状态 | 研究原型，未验收 |
| 可选身份识别与风险提示 | 默认关闭，不提供安全保证 |
| 全天无人值守运行 | 未验收；仅手动启动 |

项目保留 `baby_monitor` 包名、`baby-monitor` CLI 和既有服务前缀。目前暂停功能开发，仅整理发布通用实现与已知限制。

## 快速开始

要求 Ubuntu 24.04 / WSL2、Python 3.11+、Git。媒体功能另需 FFmpeg/ffprobe；核心测试不需要摄像头、GPU、云端账号或模型下载。

```bash
git clone https://github.com/JobsHwang/RoomSentry.git
cd RoomSentry
bash scripts/check_environment.sh
bash scripts/install_core_venv.sh

.venv/bin/python -m baby_monitor --help
.venv/bin/python -m unittest discover -s tests -t . -v
```

可选视觉依赖：

```bash
.venv/bin/python -m pip install '.[vision]'
```

安装依赖不会自动下载权重或启动服务。模型来源、哈希、许可证与运行时必须独立审查；本地视觉模型顺序运行，不同时驻留多个 VLM。

## 安全默认值

- 默认本地处理，回看服务仅监听回环；LAN 访问需单独配置。
- 媒体上传、外部通知、生物特征录入和媒体删除分别授权，默认关闭。
- 不修改防火墙或路由器，不自动启动服务或开启开机自启。
- 凭据、摄像头地址、房间映射、家庭媒体和实验输出放在 Git 外。
- 示例只使用占位符或合成数据，项目日志不得输出秘密和私人路径。

## 开发检查

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python scripts/check_import_rules.py
.venv/bin/python -m baby_monitor release-audit --root . --worktree
```

提交前还需审计 Git index；测试通过不代表家庭识别准确率或安全能力通过验收。

## 文档

- [产品与安全规范](SPEC.md)
- [实现状态与后续门槛](DEVELOPMENT_PLAN.md)
- [文档索引](docs/README.md)
- [技术架构](docs/architecture.md) · [业务架构](docs/business-architecture.md)
- [CLI](docs/cli-reference.md) · [配置](docs/configuration.md)
- [第三方许可证](THIRD_PARTY_LICENSES.md) · [安全政策](SECURITY.md)
- [贡献指南](CONTRIBUTING.md) · [变更日志](CHANGELOG.md)

公开验证记录只保存可复现命令、代码结果与通用限制；家庭试验日志与素材不公开。
