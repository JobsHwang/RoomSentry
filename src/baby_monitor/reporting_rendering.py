"""Pure Markdown and HTML rendering for the detailed local daily report."""

from __future__ import annotations

import html
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .database import StoredEvent
from .domain import (
    EventType,
    FinalSource,
    RecognitionProvenance,
    RecognitionRunMetrics,
    RiskLevel,
)
from .presence_history import PresenceDaySummary
from .recognition_attempt import RecognitionAttempt

_MARKDOWN_CONTROL = re.compile(r"([\\`*_{}\[\]()#!|])")


def _markdown_text(value: str) -> str:
    """Render untrusted model text as one literal Markdown line."""
    one_line = " ".join(value.splitlines())
    escaped_html = html.escape(one_line, quote=False)
    return _MARKDOWN_CONTROL.sub(r"\\\1", escaped_html)


def _remote_review_label(event: StoredEvent) -> str:
    if event.remote_used:
        return "已使用"
    if event.decision.needs_remote_review:
        return "待复核"
    return "否"


def _coverage_label(event: StoredEvent) -> str:
    if event.evidence.coverage_status == "unverified":
        return "未验证连续覆盖"
    reasons = ", ".join(gap.reason for gap in event.evidence.coverage_gaps)
    return f"已记录缺口（{reasons or '未记录原因'}）"


def _coverage_status_label(event: StoredEvent) -> str:
    return "已记录缺口" if event.evidence.coverage_status == "gaps_recorded" else "未验证连续覆盖"


def _coverage_overview_lines(events: list[StoredEvent]) -> list[str]:
    counts = Counter(event.evidence.coverage_status for event in events)
    return [
        f"未验证连续覆盖事件：{counts['unverified']}",
        f"已记录证据缺口事件：{counts['gaps_recorded']}",
        "未记录缺口不等于已验证连续覆盖；覆盖未验证也不等于已确认缺失。",
    ]


def _provenance_label(provenance: RecognitionProvenance) -> str:
    revision = provenance.model_revision or "revision 未记录"
    return f"{provenance.adapter_name} @ {revision}"


def _final_provenance(event: StoredEvent) -> RecognitionProvenance:
    if event.decision.final_source is FinalSource.REMOTE and event.remote_provenance:
        return event.remote_provenance
    if event.decision.final_source is FinalSource.LOCAL_SECONDARY and event.secondary_provenance:
        return event.secondary_provenance
    return event.local_provenance


def _final_metrics(event: StoredEvent) -> RecognitionRunMetrics | None:
    if event.decision.final_source is FinalSource.REMOTE:
        return event.remote_metrics
    if event.decision.final_source is FinalSource.LOCAL_SECONDARY:
        return event.secondary_metrics
    return event.local_metrics


def _metrics_label(metrics: RecognitionRunMetrics | None) -> str:
    if metrics is None:
        return "运行指标未记录"
    rss = (
        f"{metrics.process_rss_peak_kib / 1024:.1f} MiB command-tree RSS"
        if metrics.process_rss_peak_kib is not None
        else "RSS 未测量"
    )
    gpu = (
        f"{metrics.gpu_device_memory_peak_mib} MiB device GPU peak"
        if metrics.gpu_device_memory_peak_mib is not None
        else "GPU 峰值未测量"
    )
    compliance = "合规" if metrics.json_compliant else "不合规"
    return f"{metrics.duration_ms} ms；JSON {compliance}；{rss}；{gpu}"


def _recognition_rows(
    event: StoredEvent,
) -> list[tuple[str, RecognitionProvenance, RecognitionRunMetrics | None]]:
    rows = [("local", event.local_provenance, event.local_metrics)]
    if event.secondary_provenance is not None:
        rows.append(("secondary", event.secondary_provenance, event.secondary_metrics))
    if event.remote_provenance is not None:
        rows.append(("remote", event.remote_provenance, event.remote_metrics))
    return rows


def _attempt_counts(attempts: tuple[RecognitionAttempt, ...]) -> dict[str, int]:
    counts = {"running": 0, "succeeded": 0, "failed": 0, "completed": 0, "compliant": 0}
    for attempt in attempts:
        counts[attempt.status.value] += 1
        if attempt.metrics is not None:
            counts["completed"] += 1
            counts["compliant"] += int(attempt.metrics.json_compliant)
    return counts


def _compliance_rate_label(counts: dict[str, int]) -> str:
    if not counts["completed"]:
        return "未产生已完成识别尝试，无法计算"
    rate = counts["compliant"] / counts["completed"] * 100
    return f"{counts['compliant']}/{counts['completed']}（{rate:.1f}%）"


def _sleep_interval_lines(
    intervals: list[tuple[datetime, datetime]],
) -> list[str]:
    if not intervals:
        return ["今日没有 sleeping 分类事件，不能据此断言没有睡眠。"]
    lines: list[str] = []
    for start, end in intervals:
        if start == end:
            lines.append(
                f"{start.strftime('%H:%M:%S')}：仅有一个 sleeping 事件点，无法推断持续时长。"
            )
        else:
            lines.append(
                f"{start.strftime('%H:%M:%S')}–{end.strftime('%H:%M:%S')}："
                "连续事件被模型分类为 sleeping；不是连续生理监测结论。"
            )
    return lines


def _sleep_intervals(
    events: list[StoredEvent], timezone_name: str
) -> list[tuple[datetime, datetime]]:
    timezone = ZoneInfo(timezone_name)
    sleeping = [
        event.occurred_at.astimezone(timezone)
        for event in events
        if event.result.event_type is EventType.SLEEPING
    ]
    if not sleeping:
        return []
    groups: list[list[datetime]] = [[sleeping[0]]]
    for timestamp in sleeping[1:]:
        if timestamp - groups[-1][-1] <= timedelta(minutes=30):
            groups[-1].append(timestamp)
        else:
            groups.append([timestamp])
    return [(group[0], group[-1]) for group in groups]


def _resolved_evidence_frames(event: StoredEvent) -> tuple[str, ...]:
    """Resolve result references only through this event's recorded media evidence."""
    allowed_paths = set(event.evidence.frame_paths)
    paths_by_name: dict[str, list[str]] = {}
    for path in event.evidence.frame_paths:
        paths_by_name.setdefault(Path(path).name, []).append(path)

    resolved: list[str] = []
    for reference in event.result.evidence_frames:
        if reference in allowed_paths:
            candidate = reference
        elif reference == Path(reference).name:
            matches = paths_by_name.get(reference, [])
            if len(matches) != 1:
                continue
            candidate = matches[0]
        else:
            continue
        if candidate == Path(candidate).name:
            continue
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _relative_evidence_url(frame: str, report_dir: Path) -> str:
    relative = os.path.relpath(frame, report_dir).replace(os.sep, "/")
    return quote(relative, safe="/.-_")


def _duration_label(seconds: float | None) -> str:
    if seconds is None:
        return "无可计算样本"
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds / 3600:.1f} 小时"


def _presence_coverage_lines(summary: PresenceDaySummary | None) -> list[str]:
    if summary is None:
        return [
            "presence 历史表尚不可用，不能判断当天是否持续监督。",
            "没有 presence 历史不表示空间无人或环境安全。",
        ]
    if not summary.streams:
        return [
            "当天没有已落库 presence 事实；可能是未运行、无覆盖或尚未接线。",
            "person/clear 均为 0 不能解释为空房或安全结论。",
        ]
    timezone = ZoneInfo(summary.timezone_name)
    first = min(item.first_observed_at for item in summary.streams).astimezone(timezone)
    last = max(item.last_observed_at for item in summary.streams).astimezone(timezone)
    return [
        f"已落库 presence 样本：{summary.sample_count}（{len(summary.streams)} 路）。",
        f"通用 person：{summary.person_detected_count}；clear：{summary.clear_count}；"
        f"coverage gap：{summary.coverage_gap_count}。",
        f"首末观测：{first.strftime('%H:%M:%S')}–{last.strftime('%H:%M:%S')}。",
        "单路最大观测间隔（包含日界空档）："
        + _duration_label(summary.maximum_observation_gap_seconds)
        + "。",
        "person 只表示画面中的通用人物检测，不确认宝宝、成人或具体身份。",
    ]


def _shadow_presence_coverage_lines(summary: PresenceDaySummary | None) -> list[str]:
    if summary is None or not summary.streams:
        return []
    return [
        f"实验旁路样本：{summary.sample_count}（{len(summary.streams)} 路）。",
        f"实验旁路 person：{summary.person_detected_count}；clear：{summary.clear_count}；"
        f"coverage gap：{summary.coverage_gap_count}。",
        "实验旁路与主 presence 分开计数，不改变事件、房间状态、风险、通知或照护结论。",
        "该旁路绑定冻结 profile、时间 holdout 与单人盲审，仍不是官方准确率或部署批准。",
    ]


def render_markdown(
    report_date: date,
    events: list[StoredEvent],
    attempts: tuple[RecognitionAttempt, ...],
    presence: PresenceDaySummary | None,
    shadow_presence: PresenceDaySummary | None,
    report_dir: Path,
    *,
    timezone_name: str,
) -> str:
    counts = Counter(event.result.event_type.value for event in events)
    attempt_counts = _attempt_counts(attempts)
    lines = [
        f"# 宝宝监控日报 — {report_date.isoformat()}",
        "",
        "> 本日报把记录中的可核实事实与模型推测分开。"
        "事件分类、睡眠区间和语义判断仍可能出错，"
        "不能替代成人照护或医疗判断。",
        "",
        "## 概览",
        "",
        f"- 事件总数：{len(events)}",
        f"- 睡眠分类：{counts[EventType.SLEEPING.value]}",
        f"- 醒来分类：{counts[EventType.AWAKE.value]}",
        f"- 哭闹分类：{counts[EventType.CRYING.value]}",
        f"- 成人互动分类：{counts[EventType.ADULT_INTERACTION.value]}",
        f"- 可能风险分类：{counts[EventType.POSSIBLE_RISK.value]}",
        *[f"- {item}" for item in _coverage_overview_lines(events)],
        "",
        "## 数据覆盖",
        "",
        f"- 报告时区：{_markdown_text(timezone_name)}",
        f"- 本地日期范围：{report_date.isoformat()} 00:00:00–24:00:00",
        f"- SQLite 中纳入报告的事件：{len(events)}",
        "- 摄像头、视频源和 NAS 的连续健康采样尚未写入事件库，因此本报告不能证明全天覆盖完整。",
        "- 没有事件记录或没有风险分类，不代表环境安全。",
        "",
        "## 宝宝监督 presence 事实",
        "",
        *[f"- {item}" for item in _presence_coverage_lines(presence)],
        "- 上述是离散系统观测，不是连续生命体征监护，也不能替代成人照护。",
        "",
    ]
    shadow_lines = _shadow_presence_coverage_lines(shadow_presence)
    if shadow_lines:
        lines.extend(
            [
                "## Detector + Pose shadow（实验旁路）",
                "",
                *[f"- {item}" for item in shadow_lines],
                "",
            ]
        )
    lines.extend(
        [
            "## 识别质量与运行覆盖",
            "",
            f"- 识别尝试：{len(attempts)}",
            f"- 成功：{attempt_counts['succeeded']}；失败：{attempt_counts['failed']}；"
            f"运行中：{attempt_counts['running']}",
            "- 严格 JSON 合规率：" + _compliance_rate_label(attempt_counts),
            "- 合规率只反映结构化输出，不代表语义判断正确；失败尝试不包含错误原文或媒体路径。",
            "",
            "## 睡眠区间（模型推测）",
            "",
        ]
    )
    intervals = _sleep_intervals(events, timezone_name)
    lines.extend(f"- {item}" for item in _sleep_interval_lines(intervals))

    lines.extend(
        [
            "",
            "## 事件时间线",
            "",
            "| 时间 | 摄像头 | 模型分类 | 最终识别器 | 识别耗时 | "
            "风险 | 置信度 | 证据覆盖 | 复核层级 | 远端复核 |",
            "|---|---|---|---|---|---|---:|---|---|---|",
        ]
    )
    timezone = ZoneInfo(timezone_name)
    for event in events:
        result = event.result
        final_metrics = _final_metrics(event)
        lines.append(
            "| "
            + " | ".join(
                [
                    event.occurred_at.astimezone(timezone).strftime("%H:%M:%S"),
                    _markdown_text(event.camera_id),
                    result.event_type.value,
                    _provenance_label(_final_provenance(event)),
                    (f"{final_metrics.duration_ms} ms" if final_metrics is not None else "未记录"),
                    result.risk_level.value,
                    f"{result.confidence:.2f}",
                    _coverage_status_label(event),
                    event.decision.review_tier.value,
                    _remote_review_label(event),
                ]
            )
            + " |"
        )
    if not events:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")

    lines.extend(["", "## 事实与推测明细", ""])
    for event in events:
        result = event.result
        timestamp = event.occurred_at.astimezone(timezone).strftime("%H:%M:%S")
        lines.extend(
            [
                f"### {timestamp} · {_markdown_text(event.event_id)}",
                "",
                "模型摘要：" + (_markdown_text(result.summary_zh) if result.summary_zh else "无"),
                "",
                "**决策与复核（不会改写下列事实/推测）**",
                "",
                f"- 复核层级：{event.decision.review_tier.value}",
                f"- 本地待复核：{'是' if event.decision.needs_local_review else '否'}",
                f"- 最终来源：{event.decision.final_source.value}",
                "- 升级原因：" + (", ".join(event.decision.review_reasons) or "无"),
                "- 信号频率："
                + (
                    f"{event.decision.signal_frequency_count} 次 / "
                    f"{event.decision.signal_frequency_window_seconds} 秒"
                    if event.decision.signal_frequency_count
                    else "无"
                ),
                "- 证据覆盖：" + _markdown_text(_coverage_label(event)),
                "",
                "**识别运行 provenance（不包含模型路径）**",
                "",
            ]
        )
        for role, provenance, metrics in _recognition_rows(event):
            lines.append(
                f"- {role}：{_provenance_label(provenance)}；"
                f"prompt={provenance.prompt_revision}；"
                f"schema={provenance.output_schema_revision}；"
                f"{_metrics_label(metrics)}"
            )
        if not event.provenance_metadata_present:
            lines.append("- 该事件来自旧数据库记录，原始识别版本未持久化。")
        lines.extend(
            [
                "",
                "**记录中的可核实事实（observations，仍建议抽查证据）**",
                "",
            ]
        )
        lines.extend(f"- {_markdown_text(item)}" for item in result.observations)
        if not result.observations:
            lines.append("- 无。")
        lines.extend(["", "**模型推测（inferences）**", ""])
        lines.extend(f"- {_markdown_text(item)}" for item in result.inferences)
        if not result.inferences:
            lines.append("- 无。")
        lines.extend(["", "**证据帧**", ""])
        evidence_frames = _resolved_evidence_frames(event)
        for frame in evidence_frames[:3]:
            lines.append(
                f"- [{_markdown_text(Path(frame).name)}]"
                f"({_relative_evidence_url(frame, report_dir)})"
            )
        if not evidence_frames:
            lines.append("- 无。")
        lines.append("")

    lines.extend(["## 异常与待复核", ""])
    noteworthy = [
        event
        for event in events
        if event.result.risk_level.rank >= RiskLevel.MEDIUM.rank or event.decision.needs_review
    ]
    if noteworthy:
        for event in noteworthy:
            local_time = event.occurred_at.astimezone(timezone).strftime("%H:%M:%S")
            lines.append(
                f"- {local_time} · 风险 {event.result.risk_level.value} · "
                f"本地待复核 {'是' if event.decision.needs_local_review else '否'} · "
                f"远端复核 {_remote_review_label(event)} · "
                f"{_markdown_text(event.result.summary_zh or '无模型摘要')}"
            )
    else:
        lines.append("- 未记录到 medium/high 风险或待复核事件；这不等于环境绝对安全。")
    lines.append("")
    return "\n".join(lines)


def render_html(
    report_date: date,
    events: list[StoredEvent],
    attempts: tuple[RecognitionAttempt, ...],
    presence: PresenceDaySummary | None,
    shadow_presence: PresenceDaySummary | None,
    report_dir: Path,
    *,
    timezone_name: str,
) -> str:
    timezone = ZoneInfo(timezone_name)
    counts = Counter(event.result.event_type.value for event in events)
    attempt_counts = _attempt_counts(attempts)
    rows: list[str] = []
    details: list[str] = []
    for event in events:
        result = event.result
        local_time = event.occurred_at.astimezone(timezone).strftime("%H:%M:%S")
        final_metrics = _final_metrics(event)
        final_duration = (
            f"{final_metrics.duration_ms} ms" if final_metrics is not None else "未记录"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(local_time)}</td>"
            f"<td>{html.escape(event.camera_id)}</td>"
            f"<td>{html.escape(result.event_type.value)}</td>"
            f"<td>{html.escape(_provenance_label(_final_provenance(event)))}</td>"
            f"<td>{html.escape(final_duration)}</td>"
            f"<td>{html.escape(result.risk_level.value)}</td>"
            f"<td>{result.confidence:.2f}</td>"
            f"<td>{html.escape(_coverage_status_label(event))}</td>"
            f"<td>{html.escape(event.decision.review_tier.value)}</td>"
            f"<td>{html.escape(_remote_review_label(event))}</td>"
            "</tr>"
        )
        observation_items = (
            "".join(f"<li>{html.escape(item)}</li>" for item in result.observations)
            or "<li>无。</li>"
        )
        inference_items = (
            "".join(f"<li>{html.escape(item)}</li>" for item in result.inferences)
            or "<li>无。</li>"
        )
        image_items: list[str] = []
        for frame in _resolved_evidence_frames(event)[:3]:
            relative = _relative_evidence_url(frame, report_dir)
            image_items.append(
                f'<figure><a href="{html.escape(relative, quote=True)}">'
                f'<img src="{html.escape(relative, quote=True)}" alt="证据帧" '
                f'loading="lazy"></a><figcaption>{html.escape(Path(frame).name)}'
                "</figcaption></figure>"
            )
        frequency = (
            f"{event.decision.signal_frequency_count} 次 / "
            f"{event.decision.signal_frequency_window_seconds} 秒"
            if event.decision.signal_frequency_count
            else "无"
        )
        provenance_items = "".join(
            "<li>"
            f"{html.escape(role)}：{html.escape(_provenance_label(provenance))}；"
            f"prompt={html.escape(provenance.prompt_revision)}；"
            f"schema={html.escape(provenance.output_schema_revision)}；"
            f"{html.escape(_metrics_label(metrics))}"
            "</li>"
            for role, provenance, metrics in _recognition_rows(event)
        )
        if not event.provenance_metadata_present:
            provenance_items += "<li>该事件来自旧数据库记录，原始识别版本未持久化。</li>"
        details.append(
            f"<section><h3>{html.escape(local_time)} · {html.escape(event.event_id)}</h3>"
            f"<p><strong>模型摘要：</strong>{html.escape(result.summary_zh or '无')}</p>"
            f"<h4>决策与复核</h4><p>复核层级："
            f"{html.escape(event.decision.review_tier.value)}；最终来源："
            f"{html.escape(event.decision.final_source.value)}；升级原因："
            f"{html.escape(', '.join(event.decision.review_reasons) or '无')}；"
            f"本地待复核：{'是' if event.decision.needs_local_review else '否'}；"
            f"远端复核：{html.escape(_remote_review_label(event))}；证据覆盖："
            f"{html.escape(_coverage_label(event))}</p>"
            f"<p>信号频率：{html.escape(frequency)}；矛盾标记："
            f"{html.escape(', '.join(event.decision.contradictions) or '无')}</p>"
            f"<h4>识别运行 provenance（不包含模型路径）</h4>"
            f"<ul>{provenance_items}</ul>"
            f"<h4>记录中的可核实事实</h4><ul>{observation_items}</ul>"
            f"<h4>模型推测</h4><ul>{inference_items}</ul>"
            f'<h4>证据帧</h4><div class="frames">'
            f"{''.join(image_items) if image_items else '<p>无。</p>'}</div></section>"
        )

    if not rows:
        rows.append('<tr><td colspan="10">今日没有事件记录；不能据此判断安全。</td></tr>')
    sleep_items = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in _sleep_interval_lines(_sleep_intervals(events, timezone_name))
    )
    noteworthy = [
        event
        for event in events
        if event.result.risk_level.rank >= RiskLevel.MEDIUM.rank or event.decision.needs_review
    ]
    noteworthy_items = (
        "".join(
            "<li>"
            f"{html.escape(event.occurred_at.astimezone(timezone).strftime('%H:%M:%S'))} · "
            f"风险 {html.escape(event.result.risk_level.value)} · "
            f"本地待复核 {'是' if event.decision.needs_local_review else '否'} · "
            f"远端复核 {html.escape(_remote_review_label(event))} · "
            f"{html.escape(event.result.summary_zh or '无模型摘要')}"
            "</li>"
            for event in noteworthy
        )
        or "<li>未记录到 medium/high 风险或待复核事件；这不等于环境绝对安全。</li>"
    )
    coverage_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in _coverage_overview_lines(events)
    )
    presence_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in _presence_coverage_lines(presence)
    )
    shadow_lines = _shadow_presence_coverage_lines(shadow_presence)
    shadow_section = ""
    if shadow_lines:
        shadow_items = "".join(f"<li>{html.escape(item)}</li>" for item in shadow_lines)
        shadow_section = f"<h2>Detector + Pose shadow（实验旁路）</h2><ul>{shadow_items}</ul>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>宝宝监控日报 {html.escape(report_date.isoformat())}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:auto;padding:24px;color:#202124}}
.notice{{padding:12px;background:#fff8df;border-left:4px solid #e0a800}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
.frames{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0}}img{{max-width:240px;height:auto}}
section{{border-top:1px solid #ddd;margin-top:24px;padding-top:12px}}
</style>
</head>
<body>
<h1>宝宝监控日报 — {html.escape(report_date.isoformat())}</h1>
<p class="notice">事实与模型推测分开展示；本系统不能替代成人照护或医疗判断。</p>
<h2>概览</h2>
<ul>
<li>事件总数：{len(events)}</li>
<li>睡眠分类：{counts[EventType.SLEEPING.value]}</li>
<li>醒来分类：{counts[EventType.AWAKE.value]}</li>
<li>哭闹分类：{counts[EventType.CRYING.value]}</li>
<li>成人互动分类：{counts[EventType.ADULT_INTERACTION.value]}</li>
<li>可能风险分类：{counts[EventType.POSSIBLE_RISK.value]}</li>
{coverage_items}
</ul>
<h2>数据覆盖</h2>
<ul>
<li>报告时区：{html.escape(timezone_name)}</li>
<li>本地日期范围：{html.escape(report_date.isoformat())} 00:00:00–24:00:00</li>
<li>SQLite 中纳入报告的事件：{len(events)}</li>
<li>摄像头、视频源和 NAS 的连续健康采样尚未写入事件库，因此本报告不能证明全天覆盖完整。</li>
<li>没有事件记录或没有风险分类，不代表环境安全。</li>
</ul>
<h2>宝宝监督 presence 事实</h2>
<ul>{presence_items}
<li>上述是离散系统观测，不是连续生命体征监护，也不能替代成人照护。</li>
</ul>
{shadow_section}
<h2>识别质量与运行覆盖</h2>
<ul>
<li>识别尝试：{len(attempts)}</li>
<li>成功：{attempt_counts["succeeded"]}；失败：{attempt_counts["failed"]}；运行中：{attempt_counts["running"]}</li>
<li>严格 JSON 合规率：{html.escape(_compliance_rate_label(attempt_counts))}</li>
<li>合规率只反映结构化输出，不代表语义判断正确；失败尝试不包含错误原文或媒体路径。</li>
</ul>
<h2>睡眠区间（模型推测）</h2>
<ul>{sleep_items}</ul>
<h2>事件时间线</h2>
<table><thead><tr><th>时间</th><th>摄像头</th><th>模型分类</th><th>最终识别器</th><th>识别耗时</th><th>风险</th><th>置信度</th><th>证据覆盖</th><th>复核层级</th><th>远端复核</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<h2>事实与推测明细</h2>
{"".join(details) if details else "<p>无事件记录；不能据此判断安全。</p>"}
<h2>异常与待复核</h2>
<ul>{noteworthy_items}</ul>
</body>
</html>
"""
