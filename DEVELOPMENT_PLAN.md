# RoomSentry 实现状态与开发计划

状态：研究原型，暂停功能开发。本文件记录通用实现与未通过的产品门槛，不记录任何家庭的设备、作息、位置或部署状态，也不自动授权启动服务、采集素材或下载模型。

## 已有实现

| 范围 | 实现边界 |
|---|---|
| 标准库核心 | 来源/事件/决策契约、严格 JSON、CLI、SQLite |
| 媒体 | 文件/RTSP 适配、有界缓冲、候选片段、关键帧和时间轴检查 |
| 感知 | 可选 YOLO、匿名 IoU/motion 追踪、区域事实和独立实验旁路 |
| 本地理解 | 可替换 command runner、单 VLM 执行租约、严格输出与失败隔离 |
| 持久队列 | 分析、复核、报告刷新、归档、通知 outbox 的幂等和恢复 |
| 回看 | 本地事件界面、历史描述查询、Markdown/HTML 日报、每日精华 |
| 部署 | Ubuntu/WSL 手动运行模板、健康诊断、可选信号与归档 |
| 发布 | Git index 隐私审计、可重复源码包与 source SBOM |

这些是软件能力，不表示目标家庭场景识别可靠，也不保证所有外部部署已验收。

## 尚未通过的门槛

- 宝宝小目标在抱持、遮挡、低照度下的独立检出。
- 多人合框、重复框与连续可见漏检的稳定改善。
- 长时间多人追踪与角色绑定，及可用的宝宝当前位置/可见状态。
- 抱起、放下等完整事件的独立真值验收。
- 可量化的实际误提醒、漏关键变化与端到端延迟。
- 全天运行、资源争用、磁盘压力及外部服务恢复的完整实机验收。
- 可选身份功能的明确登记、独立校准与隐私验收。

小规模同帧实验仅支持开发判断，不作上线依据；不公开私人样本、标注或逐帧模型结果。

## 若恢复开发

必须先由维护者明确恢复，再一次推进一个可验收切片：

1. 确定真实使用价值，例如快速定位事件片段，而非先扩充模型链。
2. 固定一个场景和事件定义，预先写出成功、误报、漏报及未知的判定规则。
3. 在获得素材授权后按事件隔离训练、开发和最终验收，保留独立负例。
4. 冻结基线与输入；分别比较检测、解释、追踪和最终用户输出。
5. 只有真实效果改善、误报与资源门通过才讨论可逆上线。
6. 之后再扩展视角、事件种类或长稳运行，不默认引入跨摄像头身份追踪。

任何检查都不能自行恢复服务、定时任务、留证窗口或媒体上传。

## 开发与发布检查

使用仓库虚拟环境；视觉/厂商依赖不能成为核心测试前提：

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python scripts/check_import_rules.py
.venv/bin/python scripts/generate_cli_reference.py --check --output docs/cli-reference.md
.venv/bin/python scripts/generate_config_reference.py --check --output docs/configuration.md
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m baby_monitor validate-json tests/fixtures/valid_vlm_event.json
.venv/bin/python -m baby_monitor validate-signal-json tests/fixtures/valid_signal_event.json
.venv/bin/python -m baby_monitor validate-decision-json tests/fixtures/valid_decision_record.json
.venv/bin/python -m baby_monitor release-audit --root . --worktree
```

暂存后再运行 index 审计、可重复 source SBOM 与源码包校验。结果追加到 `docs/verification/`，不要将历史测试数当作当前通过数。

## 安全与许可证门

- 私人媒体、凭据、机器路径、设备标识、房间映射和家庭日志在 Git 外保管。
- 默认关闭上传、外部通知、生物特征、删除和自动启动；各自独立授权。
- 下载大权重、执行第三方模型代码、开放网络访问和破坏性操作分别确认。
- 模型/数据/运行时许可证独立登记；不因核心 Apache-2.0 而重新许可。
- 保留用户改动与本地历史；不擅自 force push、改写 Git 历史或删除媒体。

架构与契约见 [SPEC.md](SPEC.md)；第三方边界见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
