# 规则清单与规则引擎边界

本页列出系统当前全部规则族、版本、权限边界与扩枚举准入条件。分层契约
（谁能依赖谁）见 [架构概览](architecture.md)；本文回答"哪些规则在跑、
规则能做什么、不能做什么"。

## 规则流水线位置

```text
候选触发（detector/tracker/信号）→ 分析 → 已完成事件（事实固定）
                                            │
                                            ▼
                          ┌─────────────────┴──────────────────┐
                          │ 规则引擎（三族规则 + 路由策略）      │
                          └─────────────────┬──────────────────┘
                                            ▼
                        路由决策 → 通知 outbox → 既有投递闸门
```

核心不变量：**任何规则都不改写事实**。事件类型、风险级别、复核状态在
进入规则层之前已经固定；规则只做升级建议、复核建议和路由选择。

## 规则族一：复核升级规则（policy.py）

| 项 | 值 |
|---|---|
| 契约 | `ReviewPolicy` + `DecisionRecord` v1 |
| 输入 | 已完成事件的置信度、矛盾、同摄像头同信号类型滑动窗口、`severity_hint` |
| 输出 | `review_tier` 提升 + `review_reasons`（枚举原因码） |
| 权限 | 只提升复核层级；**永不**修改 observations、risk_level 或 event_type |
| 版本化 | DecisionRecord v1 Schema；原因码为封闭枚举 |

## 规则族二：原子谓词复核建议（visual_predicate_review.py）

| 项 | 值 |
|---|---|
| 契约 | `atomic_visual_review_v1`（ESC-010/011） |
| 输入 | digest-bound、completed 的原子视觉谓词结果 |
| 输出 | `review_recommended / no_rule_match / unknown` 三态 |
| 权限 | 只读、可重算、path-free；家庭标签校准前不得把模型置信度当概率，不得写 risk_level 或入通知 |
| 边界 | `no_rule_match` 只表示策略未命中，绝不等于画面安全 |

## 规则族三：几何/时序状态机

| 项 | 值 |
|---|---|
| 契约 | TemporalZoneConfig v1、StreamPresenceSample v1 |
| 输入 | 匿名 tracked observations 的 bbox 中心、逐扫描 person/clear 事实 |
| 输出 | `zone_enter/dwell/exit` 信号、房间 presence（两次确认/三次 clear/TTL/coverage gap） |
| 权限 | 系统几何事实；不直接改 event_type、risk_level；`clear` 不证明安全 |

## 规则族四：家庭事件包路由策略（event_policy.py，2026-08-29 新增）

| 项 | 值 |
|---|---|
| 契约 | `HouseholdEventPolicy` v1（公开 Schema `household-event-policy-v1`） |
| 输入 | 已完成事件（event_type/risk_level/review_status/camera/occurred_at） |
| 规则 | 声明式 JSON：匹配集 + 本地 sink + 冷却 + 夜间静默 + 聚合上限 + 可选摄像头过滤 |
| 输出 | `EventPolicyEvaluation` v1：逐事件 route/suppress + 原因（`routed / policy_disabled / no_rule_match / quiet_hours / cooldown / aggregation_cap`） |
| 权限 | **只有路由权，没有投递权**：外部投递仍需 `notification.enabled` + `external_delivery_enabled` + D-012 凭据三重门槛；策略默认 `enabled=false`，规则默认 `local_only=true` |
| 评估 | 纯函数、确定性：事件按时间排序、首条匹配规则生效、冷却锚点只在已路由事件、聚合按（规则×摄像头）滑动窗口 |
| 工具 | `validate-event-policy`、`evaluate-event-policy`（离线重放，不投递）；`evaluate/validate-event-policy-cascade`（DecisionRecord 级联评测）；Dashboard `--event-policy` 卡片；自动本地 file 接线使用 SQLite `event_policy_decisions` 幂等账本 |

示例见 `config/household-event-policy.example.json`（三条规则：中高风险
即时路由、夜间低优先级静默、待复核高注意聚合）。

在线自动通知只有在 `notification.enabled=true`、`auto_file_enabled=true` 且
`event_policy_path` 已严格加载时才评估。决定与完整性摘要先于 notification outbox 持久化；
崩溃重试复用原决定，同一策略的冷却/聚合历史跨进程保持。当前连接器只接受
与 `auto_file_sink_id` 一致的单一本地 sink；非 local route 会保留路由结果但
不会进入 outbox。策略禁用或规则抑制均不产生通知任务。

级联离线评测以 `EventPolicyCascadeCase` v1 接受最小事件元数据和完整
`DecisionRecord`，不接受另一份可能冲突的 risk/review 值：risk 只取
`DecisionRecord.risk_level`，review status 按 result 与 decision 的待复核标志
推导。`EventPolicyCascadeReport` v1 逐事件绑定 DecisionRecord digest 与
route/suppress，并汇总 review status/tier/final source/remote/contradiction ×
路由结果。它只描述规则级联，不评估风险准确率，不读取媒体或运行时数据库，
不写 outbox，也不授予复核、风险或投递权限。

## 为什么还没有"通用规则引擎"

SPEC §8 的既定决策：更细语义（趴睡、遮脸、离床……）先留在
observations/inferences 标签里，等家庭真值校准后再扩顶层枚举。规则先于
数据获得风险权限是明确禁止的（ESC-011）。因此：

1. 升级/建议规则用代码实现并版本化（可审计、可回归）；
2. 路由策略用声明式 JSON（家庭可配置，但不触碰事实与风险）；
3. 新的风险语义规则必须先通过[本地人工真值评测](verification/)的准确率
   门槛，才允许进入 `possible_risk` 判定或通知路由。

## 扩展准入清单

新增或修改任何规则时：

1. 契约版本号 +1 或新增原因码枚举，不得复用旧语义；
2. 补充合成回归测试（正例 + 每种抑制/失败路径）；
3. 验证记录落 `docs/verification/`；
4. 更新本页对应规则族表格；
5. 涉及风险语义的，先过家庭真值闸门（当前未开）。
