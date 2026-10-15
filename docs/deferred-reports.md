# 延迟详细日报：合并刷新与独立恢复

详细日报默认仍随单个分析任务同步生成。连续事件较多时，可以显式把整日读取和渲染交给独立 CPU worker，
让分析与通知路由只等待一次短的 SQLite 请求写入。它不影响聚合隐私日报、媒体、模型准入或外部通知权限。

## 开启方式

这些命令展示应用入口，不代表已经在家庭部署中启用。路径必须替换为既有本地私有配置和数据库。

```bash
.venv/bin/python -m baby_monitor analysis-worker \
  --config /absolute/private/runtime.toml \
  --database /absolute/local/state/events.sqlite3 \
  --defer-reports --max-runtime-seconds 3600

.venv/bin/python -m baby_monitor report-refresh-worker \
  --config /absolute/private/runtime.toml \
  --database /absolute/local/state/events.sqlite3 \
  --max-runtime-seconds 3600
```

这两条是独立进程，应在不同终端或经审阅的 systemd unit 中运行。`analysis-run-once` 同样支持
`--defer-reports`；不加此参数时保留原报告路径与返回结构。延迟响应的 `report.mode` 是 `queued`，
返回持久状态与版本，不能当作已生成的文件。

部署 profile 可显式设置 `"deferred_reports": true`，由 [WSL 部署生成器](deployment/home-wsl-runtime.md)
同时生成对应 analysis 参数和 report worker；默认省略/false 时不改变原 unit/manifest。生成器不安装、
enable 或 start 服务。report worker 没有 GPU 设备和 IP 网络权限，保留前一日同步 timer。

## 请求、版本与失败

| 情况 | 队列行为 |
|---|---|
| 同输出根、同本地日的新事件 | 合并同一 job，增加 requested revision；默认首次请求 30 秒后就绪 |
| pending 期间又来事件 | 不推迟首次就绪，避免持续事件导致永远不生成 |
| 同事件重复交接 | 以事件 ID 的确定性摘要幂等，不增加 revision |
| 正在生成时又来事件 | 本次只完成 claimed revision；保持下一轮 pending |
| 渲染或文件发布失败 | 独立重试，默认 60 秒后再试、最多 3 次；原事件和已完成 analysis 不变 |
| 已失败后又来事件 | 保留新的 dirty revision，不自动解除失败上限；需显式 retry |
| 领取后进程崩溃 | 默认 300 秒 lease 后可回收；旧 token 不能更新新领取的任务 |
| 请求交接本身失败 | analysis 不完成，按其原有规则重试；已落库识别结果复用 |

活动目的日默认最多 366 个，已活动的同日请求仍可合并。历史请求和终态记录不自动删除；长周期账本压缩
另行设计，不借优化删除已有数据。当前没有事件落库/请求/通知的跨账本原子事务。

队列固定规范输出根、本地日期、时区及 UTC 起止，同目的地的不同 timezone 会冲突；重新解释时区后的
UTC 日界与已保存窗口不一致时失败关闭，不能静默重算成另一段时间。输出根不能是根目录、symlink、
包含 `..` 的路径或不属于当前用户的目录。后续更改默认配置/NAS 可用性不会迁移已经排队的输出根。

所有详细日报入口共用该目的日的 0600 锁，从首次读取一直覆盖到发布。生成前及发布前复核 report claim；
失效 worker 不会拿旧快照覆盖已由新 worker 发布的内容。锁文件保留以维持 inode 互斥，等待最多 30 秒。
Markdown/HTML 仍分别原子替换：第二个文件失败时第一个可能已经更新，job 不会声明整份完成，重试修复两份；
不保证崩溃瞬间两种格式总处于同版本。

## 状态、补排与重试

```bash
.venv/bin/python -m baby_monitor report-refresh-status \
  --config /absolute/private/runtime.toml \
  --database /absolute/local/state/events.sqlite3
```

状态只读，不创建数据库或表，不披露输出根、请求 ID、claim token。关注 `counts.failed`、`dirty_jobs`、
`expired_leases` 与 `oldest_pending_age_seconds`；尚未存在日报表时 `table_present=false`，不伪装成已验收。
`unresolved_jobs` 最多返回 20 个未完成任务的脱敏详情，失败任务优先，提供用于显式 retry 的不透明 job ID；
总量继续以 `counts` 为准，不把这一页当作全部任务。
家庭 Dashboard 与 `home-control status` 现在同时提供 `report_refresh_queue` 只读聚合。
首页的“详细日报刷新队列”独立于原有隐私日报产物卡：显示 pending/running/failed、过期 lease 和
最久到期等待；failed > 0、expired lease > 0 或最久等待严格超过 3600 秒时提示“需处理”。
等待从 `available_at` 开始，因此合并窗口和未来 retry delay 不计作积压。旧库无队列表显示尚未使用，
未配置和读取失败不伪装成零任务；失败也不把底层采集服务直接判成停机。

首页/home-control 状态只输出聚合，不输出 job/request ID、路径、错误原文或未完成任务详情；需要 job ID 的显式恢复
仍使用上述独立 CLI。读取状态不会生成日报、领取/重试任务、读取媒体或启动 worker，队列可读也不证明
worker 存活。固定一小时阈值只是运维提示，不是实时处理 SLA。runtime trial 结算尚未纳入该队列。

没有新事件、但要刷新 presence 或人工补排某日时，使用唯一的非个人请求标识：

```bash
.venv/bin/python -m baby_monitor report-refresh-request \
  --config /absolute/private/runtime.toml \
  --database /absolute/local/state/events.sqlite3 \
  --output-root /absolute/local/spool \
  --date 2026-08-03 --request-id manual_refresh_001 --debounce-seconds 0

.venv/bin/python -m baby_monitor report-refresh-run-once \
  --config /absolute/private/runtime.toml \
  --database /absolute/local/state/events.sqlite3
```

同一个手动请求标识重试是幂等的；希望表达一次新的刷新意图时才换标识。输出根必须已经存在。
失败 job 的不透明 ID 从状态中的 `unresolved_jobs` 或原请求/处理结果取得，排查错误原因后显式重新开放
一次有界尝试周期：

```bash
.venv/bin/python -m baby_monitor report-refresh-retry \
  --config /absolute/private/runtime.toml \
  --database /absolute/local/state/events.sqlite3 \
  --job-id 'report_<opaque_hash>'
```

`report-refresh-worker` 复用既有有界调度器，默认最多 32 次处理、空闲 60 秒或运行 3600 秒即结束。
systemd profile 使用一小时进程边界并在授权运行期间重启。入队后只运行 analysis、不运行 report worker
会留下 pending；命令不会偷偷启动后台进程。
