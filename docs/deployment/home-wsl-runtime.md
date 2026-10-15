# Ubuntu / WSL 手动运行模板

本页描述通用运行边界，不包含私人设备、服务实例、机器路径或已安装状态。模板存在不表示系统已部署或通过长期验收。

## 前提

- Ubuntu 24.04 / WSL2 的 Linux 文件系统工作区。
- Python 3.11+ 核心虚拟环境；媒体功能另需 FFmpeg/ffprobe。
- 所有摄像头凭据、无密钥流映射、模型 manifest 和部署 profile 在 Git 外。
- 接入厂商桥时，先单独核对来源、许可证与授权。
- 安装、启动、定时任务和开机自启分别确认；默认只允许离线渲染。

## 渲染与审查

`RuntimeDeploymentProfile` 固定来源、现有绝对路径、资源与可选功能。loopback RTSP 必须无凭据、无查询参数；公开摘要不回显实际路径或 URI。

```bash
.venv/bin/python -m baby_monitor systemd-render \
  --profile /absolute/private/runtime.deployment.local.json \
  --output-dir /absolute/private/rendered-units

systemd-analyze verify /absolute/private/rendered-units/*.service \
  /absolute/private/rendered-units/*.timer \
  /absolute/private/rendered-units/*.target
```

占位目录须由使用者建立并设置私有权限。渲染器独占创建 0600 文件，不覆盖现有内容，不安装、enable 或 start unit。

| 组件 | 职责 |
|---|---|
| init | 幂等建立 SQLite schema |
| buffer | loopback 接流、有界分段与健康状态 |
| candidate | 检测、取证与有界入队 |
| analysis | 顺序消费，一个本地 VLM 执行租约 |
| report-refresh（可选） | 合并与恢复详细日报刷新，不调用模型 |
| notification（可选） | 消费已授权的本地通知 outbox |
| daily-report / health | 仅在明确启动相应任务后生成日报与脱敏健康记录 |

模板使用最小写目录、进程组清理、资源限制与有界 worker 生命周期。默认 core worker 不开放 IP 网络；buffer 只允许 loopback。需要远端网络时另建审查过的 profile，不能修改默认边界绕过媒体授权。

## 可选功能

- `deferred_reports=true` 只渲染独立日报刷新 worker；入队不等于报告完成。
- Detector/Pose shadow 必须与主事实分开，固定输入/配置并共用执行锁，不参与身份或风险判断。
- 生物特征录入、通知、远端复核和删除各自默认关闭，不能因部署而一并启用。
- 缓存淘汰需要持久设置和本次破坏性确认；渲染本身不授予删除权限。

## 本机开发工具

Windows 控制面板必须显式传入 Linux 项目根目录，不内置任何使用者路径。工具仅控制明确配置的本地实验服务，不修改开机启动。

单模型串行试验若需暂停已有 VLM，须先确认卸载并取得共享租约，保持允许继续的接流，退出后按原先授权状态恢复；用户要求停止时不得自动恢复。

## 验证与限制

以 [CLI 参考](../cli-reference.md)、[配置参考](../configuration.md)及对应契约测试核对实际选项。真实接流、长稳、磁盘压力、GPU 争用和看护效果均须在用户环境独立验收。

只读健康检查不应读取媒体、启动 worker、泄露私人路径或给出身份/安全判断。停止状态应显示为停止，不冒充运行故障。
