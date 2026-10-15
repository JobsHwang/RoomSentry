"""Pure HTML rendering for dashboard, review, digest, and control pages."""

from __future__ import annotations

import html
import math
from datetime import datetime

from baby_monitor.home_dashboard.copy import (
    _ACTIONS,
    _CANDIDATE_CAMERA_LABELS,
    _CANDIDATE_FRESHNESS_LABELS,
    _CANDIDATE_STATE_LABELS,
    _COMPONENT_LABELS,
    _DAILY_DIGEST_GAP_LABELS,
    _DAILY_DIGEST_REASON_LABELS,
    _EVENT_TYPE_LABELS,
    _GATE_SOURCE_LABELS,
    _LOCAL_DATE,
    _LOCAL_TIME,
    _OUTBOX_ERROR_LABELS,
    _PREDICATE_ANSWER_LABELS,
    _PREDICATE_LABELS,
    _QUEUE_STATUS_LABELS,
    _REVIEW_ALLOWED_ANSWERS,
    _REVIEW_TOKEN,
    _RISK_LEVEL_LABELS,
    _ROOM_PRESENCE_REASON_LABELS,
    _ROOM_PRESENCE_STATE_LABELS,
    _SOAK_STATE_LABELS,
    _STATE_LABELS,
    _VISUAL_REVIEW_REASON_LABELS,
    _VISUAL_REVIEW_STATE_LABELS,
)
from baby_monitor.home_dashboard.errors import HomeDashboardError

_STYLE = b"""
:root {
  color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #0b1020; color: #eef2ff;
}
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh;
  background: radial-gradient(circle at top left, #172554, #0b1020 48%);
}
main { max-width: 1080px; margin: auto; padding: 34px 22px 64px; }
.top { display: flex; justify-content: space-between; gap: 18px; align-items: end; }
h1 { font-size: clamp(30px, 5vw, 54px); margin: 0; letter-spacing: -.04em; }
.eyebrow {
  color: #93c5fd; text-transform: uppercase; letter-spacing: .18em; font-size: 12px;
}
.badge {
  border: 1px solid #334155; border-radius: 999px; padding: 9px 13px; background: #111827;
}
.ok { color: #86efac; }
.warn { color: #fde68a; }
.grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px; margin-top: 26px;
}
.card {
  background: #111827dd; border: 1px solid #25324a; border-radius: 18px;
  padding: 20px; box-shadow: 0 18px 60px #0005;
}
.card h2 { font-size: 15px; margin: 0 0 14px; color: #cbd5e1; }
.metric { font-size: 34px; font-weight: 750; }
.muted { color: #94a3b8; font-size: 13px; }
.row {
  display: flex; justify-content: space-between; gap: 12px; padding: 11px 0;
  border-top: 1px solid #243047;
}
.row:first-of-type { border-top: 0; }
.state { font-variant-numeric: tabular-nums; }
.controls { display: grid; gap: 10px; margin-top: 12px; }
.action { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
input, button {
  border-radius: 10px; border: 1px solid #334155; padding: 11px 12px;
  background: #0f172a; color: #f8fafc;
}
button {
  cursor: pointer; background: #1d4ed8; border-color: #3b82f6; font-weight: 700;
}
button.stop { background: #7f1d1d; border-color: #b91c1c; }
.control-link {
  display: inline-block; margin-top: 14px; border-radius: 10px; padding: 11px 14px;
  background: #1d4ed8; color: #fff; text-decoration: none; font-weight: 700;
}
.review-card { margin-top: 18px; }
.review-media {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px; margin: 14px 0;
}
.review-media img, .review-media video {
  display: block; width: 100%; max-height: 340px; object-fit: contain;
  border-radius: 12px; background: #020617; border: 1px solid #334155;
}
.review-answer { border-top: 1px solid #243047; padding: 14px 0; }
.review-answer:first-child { border-top: 0; }
.review-choices { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 9px; }
.review-choice { padding: 8px 10px; background: #1e293b; border-color: #475569; }
.review-choice.selected { background: #166534; border-color: #22c55e; }
.digest-history, .digest-reasons {
  display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0;
}
.digest-chip {
  display: inline-block; border: 1px solid #475569; border-radius: 999px;
  padding: 7px 10px; color: #dbeafe; background: #1e293b; text-decoration: none;
  font-size: 12px;
}
.digest-chip.selected { border-color: #60a5fa; background: #1d4ed8; color: #fff; }
.notice { padding: 12px; border-left: 4px solid #3b82f6; background: #111827dd; }
table { width: 100%; border-collapse: collapse; margin-top: 14px; }
th, td { padding: 10px; border: 1px solid #334155; text-align: left; }
.foot { margin-top: 24px; color: #94a3b8; font-size: 12px; line-height: 1.6; }
@media(max-width: 620px) {
  .top { align-items: start; flex-direction: column; }
  .action { grid-template-columns: 1fr; }
}
"""


def _boolean(value: object) -> bool:
    return value is True


def _integer(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _seconds(value: object) -> int:
    return round(float(value)) if type(value) in {int, float} and float(value) >= 0.0 else 0


def _duration_text(value: object) -> str:
    if (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 604_800.0
    ):
        return f"{float(value):.1f} 秒"
    return "暂无样本"


def _long_duration_text(value: object) -> str:
    if type(value) in {int, float} and math.isfinite(float(value)) and float(value) >= 0.0:
        seconds = float(value)
        if seconds >= 172_800.0:
            return f"{seconds / 86_400.0:.1f} 天"
        if seconds >= 3_600.0:
            return f"{seconds / 3_600.0:.1f} 小时"
        if seconds >= 60.0:
            return f"{seconds / 60.0:.1f} 分钟"
        return f"{seconds:.0f} 秒"
    return "暂无"


def _text(value: object, *, fallback: str = "unknown") -> str:
    if isinstance(value, str) and 0 < len(value) <= 80:
        return value
    return fallback


def _confidence(value: object) -> float:
    if type(value) in {int, float} and 0.0 <= float(value) <= 1.0:
        return float(value)
    return 0.0


def _sampling_evidence_html(health: dict[str, object], *, freshness: str) -> str:
    """Defensively render a copy of v4 facts without importing the health service."""
    unknown = '<p class="muted">实际采样证据未知；不能补记为 0 次或据此判断无人。</p>'
    if type(health.get("version")) is not int or health["version"] != 4:
        return unknown
    if health.get("state") not in {"idle", "stopped"}:
        return unknown
    completed_at = health.get("last_scan_completed_at")
    if not isinstance(completed_at, str) or not 1 <= len(completed_at) <= 64:
        return unknown
    try:
        completed = datetime.fromisoformat(completed_at)
        if completed.tzinfo is None or completed.utcoffset() is None:
            return unknown
    except ValueError:
        return unknown
    summary = health.get("last_sampling")
    counter_fields = {
        "sampled_frame_count",
        "zero_detection_sample_count",
        "decoded_frame_count",
        "timestamp_fallback_frame_count",
        "container_pts_regression_count",
    }
    if not isinstance(summary, dict) or set(summary) != counter_fields | {
        "max_adjacent_container_pts_gap_seconds",
        "end_reason",
        "source_exhaustion_verified",
    }:
        return unknown
    # Keep the display projection bounded by the public sampling contract, not
    # by permissive zero-default helpers used for older operational counters.
    if any(
        type(summary[key]) is not int or not 0 <= summary[key] <= 10_000_000
        for key in counter_fields
    ):
        return unknown
    samples = summary["sampled_frame_count"]
    zero = summary["zero_detection_sample_count"]
    decoded = summary["decoded_frame_count"]
    fallback = summary["timestamp_fallback_frame_count"]
    regressions = summary["container_pts_regression_count"]
    pts_frames = decoded - fallback
    if (
        not 1 <= samples <= 100_000
        or samples > decoded
        or zero > samples
        or fallback > decoded
        or regressions > max(0, pts_frames - 1)
        or summary["end_reason"] != "read_returned_false"
        or summary["source_exhaustion_verified"] is not False
    ):
        return unknown
    gap = summary["max_adjacent_container_pts_gap_seconds"]
    gap_text = "未知"
    if gap is not None:
        try:
            valid_gap = type(gap) in {int, float} and math.isfinite(gap) and gap >= 0
        except (OverflowError, TypeError, ValueError):
            valid_gap = False
        if not valid_gap or pts_frames < 2:
            return unknown
        if 0 < gap < 0.001:
            gap_text = "<0.001 秒"
        elif gap >= 1_000_000:
            gap_text = f"{gap:.3e} 秒"
        else:
            gap_text = f"{gap:.3f} 秒"
    historic = health.get("state") == "stopped" or freshness in {"stale", "clock_skew"}
    heading = (
        "历史采样记录（已停止或状态不新鲜）" if historic else "最近完成扫描（历史记录，非实时）"
    )
    lines = [
        f"实际模型调用 {samples} 次 · 零框 {zero} 次 · 解码 {decoded} 帧",
        f"全部解码帧时间戳回退 {fallback} 次 · 容器 PTS 回退 {regressions} 次",
        f"最大相邻源帧 PTS 间隔：{gap_text}",
        "这是源时间轴线索，不是模型采样间隔或确认丢失时长；源完整性未验证。",
        "零框不等于无人或安全；历史调用不证明当前看护或连续观察。",
    ]
    return (
        f'<div class="muted sampling-evidence"><strong>{html.escape(heading)}</strong><ul>'
        + "".join(f"<li>{html.escape(line)}</li>" for line in lines)
        + "</ul></div>"
    )


def render_home_dashboard(status: dict[str, object]) -> bytes:
    if not isinstance(status, dict):
        raise HomeDashboardError("home dashboard status is invalid")
    base_ready = _boolean(status.get("base_operational"))
    manual_only = _boolean(status.get("manual_start_only"))
    components = status.get("components")
    if not isinstance(components, list):
        components = []
    outbox = status.get("ha_signal_outbox")
    if not isinstance(outbox, dict):
        outbox = {}
    analysis_queue = status.get("analysis_queue")
    if not isinstance(analysis_queue, dict):
        analysis_queue = {}
    candidate_health = status.get("candidate_health")
    if not isinstance(candidate_health, dict):
        candidate_health = {}
    candidate_soak = status.get("candidate_soak")
    if not isinstance(candidate_soak, dict):
        candidate_soak = {}
    room_presence = status.get("room_presence")
    event_policy_value = status.get("event_policy")
    if not isinstance(room_presence, dict):
        room_presence = {}
    visual_shadow = status.get("visual_predicate_shadow")
    if not isinstance(visual_shadow, dict):
        visual_shadow = {}
    notification_queue = status.get("notification_queue")
    if not isinstance(notification_queue, dict):
        notification_queue = {}
    event_digest = status.get("event_digest")
    if not isinstance(event_digest, dict):
        event_digest = {}
    daily_report = status.get("daily_report")
    if not isinstance(daily_report, dict):
        daily_report = {}
    report_refresh = status.get("report_refresh_queue")
    if not isinstance(report_refresh, dict):
        report_refresh = {}
    report_counts = report_refresh.get("counts")
    if not isinstance(report_counts, dict):
        report_counts = {}
    raw_report_refresh_status = _text(report_refresh.get("status"), fallback="not_configured")
    report_refresh_status = html.escape(
        _QUEUE_STATUS_LABELS.get(raw_report_refresh_status, raw_report_refresh_status)
    )
    report_refresh_metric = (
        f"{_integer(report_counts.get('pending'))} 个待刷新日"
        if raw_report_refresh_status == "available"
        else "状态不可用"
        if raw_report_refresh_status == "unavailable"
        else "尚未使用异步刷新"
    )
    report_refresh_detail = (
        f"{_integer(report_counts.get('running'))} 个处理中 · "
        f"{_integer(report_counts.get('failed'))} 个失败 · "
        f"{_integer(report_refresh.get('expired_leases'))} 个租约已超时 · "
        f"最久到期等待 {_long_duration_text(report_refresh.get('oldest_pending_age_seconds'))}"
        if raw_report_refresh_status == "available"
        else "暂无可确认的队列计数。"
    )
    daily_digest = status.get("daily_digest")
    if not isinstance(daily_digest, dict):
        daily_digest = {}

    component_rows: list[str] = []
    for item in components[:16]:
        if not isinstance(item, dict):
            continue
        component = _text(item.get("component"), fallback="")
        if component not in _COMPONENT_LABELS:
            continue
        label = html.escape(_COMPONENT_LABELS[component])
        raw_state = _text(item.get("state"))
        state = html.escape(_STATE_LABELS.get(raw_state, raw_state))
        mode = "手动" if not _boolean(item.get("enabled")) else "开机自启"
        component_rows.append(
            f'<div class="row"><span>{label}</span>'
            f'<span class="state">{state} · {mode}</span></div>'
        )

    failed_count = _integer(outbox.get("failed"))
    analysis_failed = _integer(analysis_queue.get("failed"))
    analysis_missing_media = _integer(analysis_queue.get("missing_media_files"))
    visual_shadow_failed = _integer(visual_shadow.get("failed"))
    notification_failed = _integer(notification_queue.get("failed"))
    queue_attention = (
        failed_count > 0
        or analysis_failed > 0
        or analysis_missing_media > 0
        or visual_shadow_failed > 0
        or notification_failed > 0
        or _boolean(candidate_health.get("attention_required"))
        or _boolean(candidate_soak.get("attention_required"))
        or _boolean(daily_report.get("attention_required"))
        or _boolean(report_refresh.get("attention_required"))
        or _boolean(daily_digest.get("attention_required"))
    )
    readiness = (
        "运行中"
        if base_ready and not queue_attention
        else "运行中 · 需处理"
        if base_ready
        else "已停止 / 异常"
    )
    readiness_class = "ok" if base_ready and not queue_attention else "warn"
    manual_policy = "仅手动启动" if manual_only else "启动策略需检查"
    raw_queue_status = _text(outbox.get("status"), fallback="not_configured")
    queue_status = html.escape(_QUEUE_STATUS_LABELS.get(raw_queue_status, raw_queue_status))
    raw_error_counts = outbox.get("failed_error_counts")
    error_counts = raw_error_counts if isinstance(raw_error_counts, dict) else {}
    error_labels: list[str] = []
    for error_code, error_count in sorted(error_counts.items()):
        if not isinstance(error_code, str) or error_code not in _OUTBOX_ERROR_LABELS:
            continue
        count = _integer(error_count)
        if count:
            error_labels.append(f"{_OUTBOX_ERROR_LABELS[error_code]} {count}")
    queue_explanation = (
        "；".join(error_labels) + "。失败记录会保留，系统不会自动改绑摄像头。"
        if error_labels
        else "队列正常；事件正文、设备实体和本地路径不会显示在页面中。"
    )
    raw_analysis_status = _text(analysis_queue.get("status"), fallback="not_configured")
    analysis_status = html.escape(
        _QUEUE_STATUS_LABELS.get(raw_analysis_status, raw_analysis_status)
    )
    analysis_explanation = (
        f"{analysis_missing_media} 个证据文件缺失；系统不会假装已完成分析。"
        if analysis_missing_media
        else "只读聚合，不打开视频或关键帧；模型未启动时不会自动消费待处理任务。"
    )
    raw_candidate_status = _text(
        candidate_health.get("status"),
        fallback="not_configured",
    )
    candidate_status = html.escape(
        _QUEUE_STATUS_LABELS.get(raw_candidate_status, raw_candidate_status)
    )
    raw_candidate_cameras = candidate_health.get("cameras")
    candidate_cameras = raw_candidate_cameras if isinstance(raw_candidate_cameras, list) else []
    candidate_rows: list[str] = []
    candidate_operational = 0
    candidate_camera_count = 0
    for item in candidate_cameras[:8]:
        if not isinstance(item, dict):
            continue
        health = item.get("health")
        if not isinstance(health, dict):
            continue
        camera_id = _text(health.get("camera_id"), fallback="")
        state = _text(health.get("state"), fallback="")
        freshness = _text(item.get("freshness"), fallback="")
        if (
            camera_id not in _CANDIDATE_CAMERA_LABELS
            or state not in _CANDIDATE_STATE_LABELS
            or freshness not in _CANDIDATE_FRESHNESS_LABELS
        ):
            continue
        candidate_camera_count += 1
        candidate_operational += int(_boolean(item.get("operational")))
        wait_text = _duration_text(health.get("last_execution_lock_wait_seconds"))
        duration_text = _duration_text(health.get("last_scan_duration_seconds"))
        candidate_rows.append(
            '<div class="row"><span>'
            f"{html.escape(_CANDIDATE_CAMERA_LABELS[camera_id])} · "
            f"{html.escape(_CANDIDATE_STATE_LABELS[state])}</span>"
            '<span class="state">'
            f"等显卡 {wait_text} · 片段 {duration_text} · "
            f"{html.escape(_CANDIDATE_FRESHNESS_LABELS[freshness])}</span></div>"
        )
        candidate_rows.append(_sampling_evidence_html(health, freshness=freshness))
        history = item.get("history_summary")
        if isinstance(history, dict):
            candidate_rows.append(
                '<p class="muted">累计 '
                f"{_integer(history.get('scans_succeeded'))} 段 · "
                "候选发出 "
                f"{_integer(history.get('candidate_groups_emitted'))} / 抑制 "
                f"{_integer(history.get('candidate_groups_suppressed'))} · "
                "等显卡 P95≤"
                f"{_duration_text(history.get('execution_lock_wait_p95_upper_bound_seconds'))}"
                " / 峰值 "
                f"{_duration_text(history.get('execution_lock_wait_max_seconds'))}"
                " · 失败会话 "
                f"{_integer(history.get('sessions_failed'))} 次"
                " · 非正常重启 "
                f"{_integer(history.get('unclean_restarts'))} 次</p>"
            )
    candidate_markup = "".join(candidate_rows) or (
        '<p class="muted">尚无候选 worker 健康样本；启动 Shadow 后才会生成。</p>'
    )
    soak_latest = candidate_soak.get("latest")
    if isinstance(soak_latest, dict):
        soak_state = _text(soak_latest.get("state"), fallback="")
        soak_label = html.escape(_SOAK_STATE_LABELS.get(soak_state, "状态未知"))
        soak_markup = (
            '<p class="muted"><strong>SOAK24：</strong>'
            f"{soak_label} · 已运行 {_long_duration_text(soak_latest.get('elapsed_seconds'))}"
            f" / {_long_duration_text(soak_latest.get('expected_duration_seconds'))}"
            f" · 成功 {_integer(soak_latest.get('scans_succeeded'))} 段"
            f" · 候选发出 {_integer(soak_latest.get('candidate_groups_emitted'))}"
            f" / 抑制 {_integer(soak_latest.get('candidate_groups_suppressed'))}"
            f" · 扫描失败 {_integer(soak_latest.get('scans_failed'))} 次"
            f" · 非正常重启 {_integer(soak_latest.get('unclean_restarts'))} 次</p>"
        )
    else:
        soak_markup = (
            '<p class="muted"><strong>SOAK24：</strong>尚未运行；'
            "在手动控制页输入 SOAK24 后才会启动，24 小时自动停止。</p>"
        )
    raw_room_status = _text(room_presence.get("status"), fallback="not_configured")
    raw_rooms = room_presence.get("rooms")
    rooms = raw_rooms if isinstance(raw_rooms, list) else []
    room_rows: list[str] = []
    occupied_room_count = 0
    for item in rooms[:16]:
        if not isinstance(item, dict):
            continue
        display_name = _text(item.get("display_name"), fallback="")
        state = _text(item.get("status"), fallback="")
        if not display_name or state not in _ROOM_PRESENCE_STATE_LABELS:
            continue
        occupied_room_count += int(state == "person_present")
        raw_reasons = item.get("reason_codes")
        reasons = raw_reasons if isinstance(raw_reasons, list) else []
        reason_labels = [
            _ROOM_PRESENCE_REASON_LABELS[value]
            for value in reasons
            if isinstance(value, str) and value in _ROOM_PRESENCE_REASON_LABELS
        ]
        reason_text = "；".join(reason_labels) or "没有可展示的充分证据"
        room_rows.append(
            '<div class="row"><span>'
            f"{html.escape(display_name)}</span>"
            '<span class="state">'
            f"{html.escape(_ROOM_PRESENCE_STATE_LABELS[state])}</span></div>"
            f'<p class="muted">{html.escape(reason_text)}</p>'
        )
    if raw_room_status == "available":
        room_metric = f"{occupied_room_count} 个房间检测到人员"
        room_summary = (
            "".join(room_rows) or '<p class="muted">注册表可用，但当前没有可展示的房间结果。</p>'
        )
    elif raw_room_status == "unavailable":
        room_metric = "状态不可用"
        room_summary = '<p class="muted">私有房间映射或 presence 事实校验失败。</p>'
    else:
        room_metric = "等待填写房间映射"
        room_summary = (
            '<p class="muted">三台物理摄像头和四路视频已经分开建模；填写真实房间名后才会显示。</p>'
        )
    raw_event_policy_status = _text(
        (event_policy_value if isinstance(event_policy_value, dict) else {}).get("status"),
        fallback="not_configured",
    )
    event_policy_value = event_policy_value if isinstance(event_policy_value, dict) else {}
    event_policy_rows: list[str] = []
    raw_policy_rules = event_policy_value.get("rules")
    policy_rules = raw_policy_rules if isinstance(raw_policy_rules, list) else []
    for item in policy_rules[:8]:
        if not isinstance(item, dict):
            continue
        rule_id = _text(item.get("rule_id"), fallback="")
        if not rule_id:
            continue
        types_value = item.get("event_types")
        type_names = (
            [value for value in types_value if isinstance(value, str)]
            if isinstance(types_value, list)
            else []
        )
        sinks_value = item.get("sink_ids")
        sink_names = (
            [value for value in sinks_value if isinstance(value, str)]
            if isinstance(sinks_value, list)
            else []
        )
        cooldown_text = _duration_text(item.get("cooldown_seconds"))
        quiet_text = "夜间静默" if item.get("quiet_hours_suppress") is True else "全天可触发"
        event_policy_rows.append(
            '<div class="row"><span>'
            f"{html.escape(rule_id)}</span>"
            '<span class="state">'
            f"{html.escape('、'.join(type_names))} → {html.escape('、'.join(sink_names))}"
            "</span></div>"
            f'<p class="muted">冷却 {html.escape(cooldown_text)} · {quiet_text}</p>'
        )
    if raw_event_policy_status == "configured":
        event_policy_metric = f"{_integer(event_policy_value.get('rule_count'))} 条规则 · " + (
            "已启用" if event_policy_value.get("enabled") is True else "未启用"
        )
        event_policy_summary = (
            "".join(event_policy_rows) or '<p class="muted">策略文件有效，但没有可展示的规则。</p>'
        )
    elif raw_event_policy_status == "invalid":
        event_policy_metric = "策略文件无效"
        event_policy_summary = '<p class="muted">事件包策略解析失败；路由保持关闭。</p>'
    else:
        event_policy_metric = "未配置"
        event_policy_summary = (
            '<p class="muted">通过 --event-policy 提供家庭事件包策略后显示路由规则。</p>'
        )
    raw_visual_shadow_status = _text(visual_shadow.get("status"), fallback="not_configured")
    visual_shadow_status = html.escape(
        _QUEUE_STATUS_LABELS.get(raw_visual_shadow_status, raw_visual_shadow_status)
    )
    raw_shadow_recent = visual_shadow.get("recent_answers")
    shadow_recent = raw_shadow_recent if isinstance(raw_shadow_recent, list) else []
    shadow_answer_labels: list[str] = []
    for observation in shadow_recent[:2]:
        if not isinstance(observation, dict):
            continue
        if _boolean(observation.get("hardware_signal_gate")):
            source_label = "HA 硬件事件"
        elif _boolean(observation.get("person_detection_gate")):
            source_label = "YOLO 人形候选"
        else:
            source_label = "有界抽检"
        raw_answers = observation.get("answers")
        answers = raw_answers if isinstance(raw_answers, list) else []
        for answer in answers[:2]:
            if not isinstance(answer, dict):
                continue
            predicate = _text(answer.get("predicate"), fallback="")
            value = _text(answer.get("answer"), fallback="")
            if predicate not in _PREDICATE_LABELS or value not in _PREDICATE_ANSWER_LABELS:
                continue
            shadow_answer_labels.append(
                f"{source_label} · {_PREDICATE_LABELS[predicate]} "
                f"{_PREDICATE_ANSWER_LABELS[value]} "
                f"{_confidence(answer.get('confidence')):.0%}"
            )
    shadow_answer_summary = (
        "最近原子结果：" + "；".join(shadow_answer_labels)
        if shadow_answer_labels
        else "暂无已完成的 Mage-VL 原子结果。"
    )
    shadow_review_sample_count = _integer(visual_shadow.get("recent_assessed_count"))
    shadow_review_summary = (
        f"最近 {shadow_review_sample_count} 条规则结果：建议人工回看 "
        f"{_integer(visual_shadow.get('recent_review_recommended'))} 条 · 信息不足 "
        f"{_integer(visual_shadow.get('recent_unknown'))} 条 · 未命中规则 "
        f"{_integer(visual_shadow.get('recent_no_rule_match'))} 条。"
        if shadow_review_sample_count
        else "暂无可执行保守复核规则的完整原子结果。"
    )
    raw_notification_status = _text(
        notification_queue.get("status"),
        fallback="not_configured",
    )
    notification_status = html.escape(
        _QUEUE_STATUS_LABELS.get(raw_notification_status, raw_notification_status)
    )
    external_notification_policy = (
        "外部投递：已开启"
        if _boolean(notification_queue.get("external_delivery_enabled"))
        else "外部投递：关闭"
    )
    raw_daily_report_status = _text(
        daily_report.get("status"),
        fallback="not_configured",
    )
    daily_report_status = html.escape(
        _QUEUE_STATUS_LABELS.get(raw_daily_report_status, raw_daily_report_status)
    )
    daily_report_link = (
        '<a class="control-link" href="/daily-report">查看今日隐私日报</a>'
        if _boolean(daily_report.get("available"))
        else ""
    )
    raw_daily_digest_status = _text(
        daily_digest.get("status"),
        fallback="not_configured",
    )
    daily_digest_status = html.escape(
        {
            **_QUEUE_STATUS_LABELS,
            "planned": "计划已生成",
        }.get(raw_daily_digest_status, raw_daily_digest_status)
    )
    daily_digest_link = (
        '<a class="control-link" href="/daily-digest">播放本地精华</a>'
        if _boolean(daily_digest.get("video_available"))
        else ""
    )
    raw_event_type_counts = event_digest.get("event_type_counts")
    event_type_counts = raw_event_type_counts if isinstance(raw_event_type_counts, dict) else {}
    today_labels: list[str] = []
    for event_type, label in _EVENT_TYPE_LABELS.items():
        count = _integer(event_type_counts.get(event_type))
        if count:
            today_labels.append(f"{label} {count}")
    today_breakdown = " · ".join(today_labels) or "今天暂无已落库事件"
    digest_date = html.escape(_text(event_digest.get("local_date"), fallback="日期不可用"))
    raw_recent = event_digest.get("recent")
    recent_items = raw_recent if isinstance(raw_recent, list) else []
    recent_rows: list[str] = []
    for item in recent_items[:8]:
        if not isinstance(item, dict):
            continue
        event_type = _text(item.get("event_type"), fallback="")
        risk_level = _text(item.get("risk_level"), fallback="")
        if event_type not in _EVENT_TYPE_LABELS or risk_level not in _RISK_LEVEL_LABELS:
            continue
        local_time = html.escape(_text(item.get("local_time"), fallback="时间未知"))
        confidence = _confidence(item.get("confidence"))
        review_label = " · 待复核" if _boolean(item.get("needs_review")) else ""
        recent_rows.append(
            '<div class="row"><span>'
            f"{local_time} · {html.escape(_EVENT_TYPE_LABELS[event_type])}</span>"
            '<span class="state">'
            f"风险 {_RISK_LEVEL_LABELS[risk_level]} · {confidence:.0%}{review_label}</span></div>"
        )
    recent_markup = "".join(recent_rows) or '<p class="muted">暂无历史事件记录。</p>'
    component_markup = "".join(component_rows) or ('<p class="muted">Status unavailable.</p>')
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baby Monitor Home</title>
<link rel="stylesheet" href="/style.css">
</head><body><main>
<div class="top"><div>
<div class="eyebrow">仅限本机与家庭局域网</div>
<h1>Baby Monitor</h1></div>
<div class="badge {readiness_class}">{html.escape(readiness)}</div></div>
<section class="grid">
<article class="card"><h2>监控启动策略</h2>
<div class="metric">{html.escape(manual_policy)}</div>
<p class="muted">打开页面不会启动摄像头。START 只确认本次环形缓冲滚动淘汰，
不会设置开机自启。</p></article>
<article class="card"><h2>HA 持久事件队列</h2>
<div class="metric">{_integer(outbox.get("pending"))} 条待处理</div>
<p class="muted">{queue_status} · {_integer(outbox.get("running"))} 条处理中 ·
{failed_count} 条历史失败</p><p class="muted">{html.escape(queue_explanation)}</p></article>
<article class="card"><h2>本地分析队列</h2>
<div class="metric">{_integer(analysis_queue.get("completed"))} 条已完成</div>
<p class="muted">{analysis_status} · {_integer(analysis_queue.get("pending"))} 条待分析 ·
{_integer(analysis_queue.get("running"))} 条处理中 · {analysis_failed} 条失败</p>
<p class="muted">{html.escape(analysis_explanation)}</p></article>
<article class="card"><h2>详细日报刷新队列</h2>
<div class="metric">{html.escape(report_refresh_metric)}</div>
<p class="muted">{report_refresh_status} · {html.escape(report_refresh_detail)}</p>
<p class="muted">同一天的新事件合并刷新；失败、租约超时或到期等待超过 1 小时需处理。
只读状态不证明 worker 正在运行，也不会重试任务。未启用时不影响同步日报；
这里不是下方隐私日报的产物状态。</p></article>
<article class="card"><h2>四路 YOLO 候选健康</h2>
<div class="metric">{candidate_operational}/{candidate_camera_count} 路运行正常</div>
<p class="muted">{candidate_status} · 只读显示最近片段与显卡排队时间；
不读取画面。“运行正常”仅指 worker 状态，不保证输入连续、检出率或安全。</p>
{candidate_markup}{soak_markup}</article>
<article class="card"><h2>房间人员存在</h2>
<div class="metric">{html.escape(room_metric)}</div>{room_summary}
<p class="muted">这里只合并 person/clear/coverage-gap；物理设备与逻辑视角分开计数，
同一台摄像头的不同视角可以覆盖不同房间。它不是宝宝身份或宝宝位置，也不进入风险判断和通知。</p></article>
<article class="card"><h2>事件包路由策略</h2>
<div class="metric">{html.escape(event_policy_metric)}</div>{event_policy_summary}
<p class="muted">规则引擎只把已完成事件路由到本地通知 sink；它不改写事实、风险或复核状态，
外部投递仍受既有双重开关与凭据门槛约束。</p></article>
<article class="card"><h2>INT8 Mage-VL Shadow</h2>
<div class="metric">{_integer(visual_shadow.get("completed"))} 条已复核</div>
<p class="muted">{visual_shadow_status} · {_integer(visual_shadow.get("pending"))} 条待复核 ·
{_integer(visual_shadow.get("running"))} 条处理中 · {visual_shadow_failed} 条失败 ·
{_integer(visual_shadow.get("skipped"))} 条未命中 gate</p>
<p class="muted">{html.escape(shadow_answer_summary)}</p>
<p class="muted">{html.escape(shadow_review_summary)}这里只用于人工回看排序，
不代表危险或安全，也不会触发通知。</p>
<p class="muted">HA 硬件事件、YOLO 人形候选或有界抽检只负责进门；
结果仅写 shadow，不进入风险判断和通知。</p></article>
<article class="card"><h2>通知队列</h2>
<div class="metric">{_integer(notification_queue.get("delivered"))} 条已投递</div>
<p class="muted">{notification_status} · {_integer(notification_queue.get("pending"))} 条待投递 ·
{_integer(notification_queue.get("running"))} 条处理中 · {notification_failed} 条失败</p>
<p class="muted">{html.escape(external_notification_policy)}。当前测试按钮只写本机测试文件，
不代表真实事件、风险判断或真实邮件/微信已接通。</p></article>
<article class="card"><h2>今日摘要</h2>
<div class="metric">{_integer(event_digest.get("today_total"))} 条事件</div>
<p class="muted">北京时间 {digest_date} · {html.escape(today_breakdown)}</p>
<p class="muted">隐私日报：{daily_report_status}。分类与风险仅是已落库的历史模型/规则记录，
不代表全天覆盖或安全结论。</p>{daily_report_link}</article>
<article class="card"><h2>每日精华</h2>
<div class="metric">{daily_digest_status}</div>
<p class="muted">最近日期：{
        html.escape(_text(daily_digest.get("latest_local_date"), fallback="暂无"))
    } ·
{_integer(daily_digest.get("latest_selected_event_count"))} 个片段 ·
{_seconds(daily_digest.get("latest_duration_seconds"))} 秒</p>
<p class="muted">仅由已落库事实做确定性选择和本地剪辑；不代表风险、身份或全天安全。</p>
{daily_digest_link}</article>
<article class="card"><h2>隐私边界</h2>
<div class="metric">远程上传：关闭</div>
<p class="muted">家庭媒体只留本机。身份能力仅限 shadow，不进入风险判断或通知。
页面不会披露本地路径和事件正文。</p></article>
</section>
<section class="grid"><article class="card"><h2>基础组件</h2>
{component_markup}</article>
<article class="card"><h2>最近事件</h2>{recent_markup}
<p class="muted">不显示事件 ID、摄像头 ID、模型摘要、媒体或本地路径。</p></article>
<article class="card"><h2>Shadow 人工回看</h2>
<p class="muted">只在本机查看已完成的候选画面，并给原子问题做人工标签；
标签不进入风险判断和通知。</p>
<a class="control-link" href="/review">打开本地回看</a></article>
<article class="card"><h2>手动控制</h2>
<p class="muted">控制操作与只读状态页隔离；每次操作都使用独立会话和一次性口令。</p>
<a class="control-link" href="/control">打开控制页</a></article></section>
<p class="foot">仅监听 127.0.0.1 · 无外部页面资源 · 无开机自启 · 无远程上传</p>
</main></body></html>"""
    return body.encode("utf-8")


def render_daily_digest_video(
    entries: tuple[dict[str, object], ...],
    *,
    selected_token: str | None = None,
) -> bytes:
    if not isinstance(entries, tuple) or len(entries) > 30:
        raise HomeDashboardError("home dashboard daily digest is invalid")
    admitted: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise HomeDashboardError("home dashboard daily digest is invalid")
        token = entry.get("review_token")
        local_date = entry.get("local_date")
        items = entry.get("items")
        gap_reason_counts = entry.get("gap_reason_counts")
        selected_hours = entry.get("selected_hours")
        selected_event_count = entry.get("selected_event_count")
        selected_camera_count = entry.get("selected_camera_count")
        duration = entry.get("output_duration_seconds")
        if (
            not isinstance(token, str)
            or _REVIEW_TOKEN.fullmatch(token) is None
            or not isinstance(local_date, str)
            or _LOCAL_DATE.fullmatch(local_date) is None
            or not isinstance(items, list)
            or len(items) > 64
            or not isinstance(gap_reason_counts, dict)
            or len(gap_reason_counts) > 64
            or not isinstance(selected_hours, list)
            or any(type(hour) is not int or not 0 <= hour <= 23 for hour in selected_hours)
            or type(selected_event_count) is not int
            or not 0 <= selected_event_count <= 64
            or selected_event_count != len(items)
            or type(selected_camera_count) is not int
            or not 0 <= selected_camera_count <= 64
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not 0.0 <= float(duration) <= 3_600.0
            or type(entry.get("target_met")) is not bool
            or any(
                entry.get(flag) is not expected
                for flag, expected in {
                    "facts_only": True,
                    "paths_disclosed": False,
                    "event_ids_disclosed": False,
                    "camera_ids_disclosed": False,
                    "identity_inference": False,
                    "risk_inference": False,
                    "notification_decision": False,
                }.items()
            )
        ):
            raise HomeDashboardError("home dashboard daily digest is invalid")
        admitted.append(entry)
    if selected_token is not None and (
        not isinstance(selected_token, str) or _REVIEW_TOKEN.fullmatch(selected_token) is None
    ):
        raise HomeDashboardError("home dashboard daily digest is invalid")
    selected = next(
        (entry for entry in admitted if entry["review_token"] == selected_token),
        admitted[0] if selected_token is None and admitted else None,
    )
    if selected_token is not None and selected is None:
        raise HomeDashboardError("home dashboard daily digest is invalid")
    history = "".join(
        '<a class="digest-chip{}" href="/daily-digest/{}">{}</a>'.format(
            " selected" if entry is selected else "",
            entry["review_token"],
            html.escape(str(entry["local_date"])),
        )
        for entry in admitted
    )
    if selected is None:
        main_content = """<article class="card review-card"><h2>还没有每日精华</h2>
<p class="muted">到手动控制页输入 DIGEST 后，才会在本机生成；不会自动调度。</p></article>"""
    else:
        token = str(selected["review_token"])
        gaps: list[str] = []
        for code, count in selected["gap_reason_counts"].items():
            if not isinstance(code, str) or type(count) is not int or not 1 <= count <= 10_000:
                raise HomeDashboardError("home dashboard daily digest is invalid")
            label = _DAILY_DIGEST_GAP_LABELS.get(code, "其他本地覆盖缺口")
            gaps.append(f"{html.escape(label)} {count} 处")
        gap_text = "、".join(gaps) if gaps else "未记录到本地覆盖缺口"
        item_cards: list[str] = []
        for item in selected["items"]:
            if not isinstance(item, dict):
                raise HomeDashboardError("home dashboard daily digest is invalid")
            rank = item.get("rank")
            local_time = item.get("local_time")
            item_duration = item.get("duration_seconds")
            score = item.get("selection_score")
            reasons = item.get("selection_reasons")
            if (
                type(rank) is not int
                or not 1 <= rank <= 64
                or not isinstance(local_time, str)
                or _LOCAL_TIME.fullmatch(local_time) is None
                or isinstance(item_duration, bool)
                or not isinstance(item_duration, (int, float))
                or not 0.0 < float(item_duration) <= 600.0
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0.0 <= float(score) <= 1.0
                or not isinstance(reasons, list)
                or not reasons
                or any(reason not in _DAILY_DIGEST_REASON_LABELS for reason in reasons)
            ):
                raise HomeDashboardError("home dashboard daily digest is invalid")
            reason_chips: list[str] = []
            for reason in reasons:
                reason_label = html.escape(_DAILY_DIGEST_REASON_LABELS[reason])
                reason_chips.append(f'<span class="digest-chip">{reason_label}</span>')
            reason_markup = "".join(reason_chips)
            displayed_duration = round(float(item_duration), 1)
            item_cards.append(
                f'<article class="card"><h2>片段 {rank} · {local_time}</h2>'
                f'<div class="metric">{displayed_duration} 秒</div>'
                f'<div class="digest-reasons">{reason_markup}</div></article>'
            )
        hours = "、".join(f"{hour:02d} 时" for hour in selected["selected_hours"])
        target_text = (
            "已达到目标时长" if selected["target_met"] else "未勉强凑时长：只使用通过校验的本地素材"
        )
        main_content = f"""<article class="card review-card">
<h2>{html.escape(str(selected["local_date"]))} · {selected["selected_event_count"]} 个片段</h2>
<div class="review-media"><video controls preload="metadata">
<source src="/daily-digest-video/{token}" type="video/mp4"></video></div>
<p class="muted">成片 {round(float(selected["output_duration_seconds"]), 1)} 秒 ·
来自 {selected["selected_camera_count"]} 路本地视频 · 时间覆盖 {html.escape(hours or "暂无")}</p>
<p class="muted">{html.escape(target_text)}。覆盖说明：{gap_text}。</p>
</article><section class="grid">{"".join(item_cards)}</section>"""
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baby Monitor Daily Digest</title><link rel="stylesheet" href="/style.css">
</head><body><main><div class="eyebrow">仅限本机私密回看</div>
<h1>每日精华</h1><nav class="digest-history">{history}</nav>{main_content}
<article class="card review-card">
<p class="muted">这是已落库事实的确定性本地剪辑，不是风险、身份或全天安全结论；
“原记录含风险字段”也只解释入选分数，不代表本页面重新作出风险判断。不会自动进入通知。</p>
<a class="control-link" href="/">返回状态页</a></article>
<p class="foot">仅监听 127.0.0.1 · 无远程上传 · 无外部页面资源</p>
</main></body></html>"""
    return body.encode("utf-8")


def _render_review_calibration(calibration: dict[str, object] | None) -> str:
    if calibration is None:
        return ""
    if not isinstance(calibration, dict):
        raise HomeDashboardError("home dashboard review calibration is invalid")
    expected_fields = {
        "version",
        "observation_count",
        "labeled_observation_count",
        "answer_count",
        "labeled_answer_count",
        "unlabeled_answer_count",
        "exact_agreement_count",
        "exact_agreement_rate",
        "groups",
        "human_reviewed_visible_fact_labels",
        "exact_answer_agreement_evaluated",
        "risk_accuracy_evaluated",
        "media_read",
        "paths_disclosed",
        "event_identity_disclosed",
        "subject_identity_inference",
        "notification_decision",
    }
    if set(calibration) != expected_fields or calibration.get("version") != 1:
        raise HomeDashboardError("home dashboard review calibration is invalid")
    count_fields = (
        "observation_count",
        "labeled_observation_count",
        "answer_count",
        "labeled_answer_count",
        "unlabeled_answer_count",
        "exact_agreement_count",
    )
    if any(
        type(calibration.get(field)) is not int or calibration[field] < 0 for field in count_fields
    ):
        raise HomeDashboardError("home dashboard review calibration is invalid")
    observation_count = calibration["observation_count"]
    labeled_observation_count = calibration["labeled_observation_count"]
    answer_count = calibration["answer_count"]
    labeled_answer_count = calibration["labeled_answer_count"]
    unlabeled_answer_count = calibration["unlabeled_answer_count"]
    exact_agreement_count = calibration["exact_agreement_count"]
    rate = calibration["exact_agreement_rate"]
    expected_flags = {
        "human_reviewed_visible_fact_labels": True,
        "exact_answer_agreement_evaluated": True,
        "risk_accuracy_evaluated": False,
        "media_read": False,
        "paths_disclosed": False,
        "event_identity_disclosed": False,
        "subject_identity_inference": False,
        "notification_decision": False,
    }
    if (
        labeled_observation_count > observation_count
        or labeled_answer_count + unlabeled_answer_count != answer_count
        or exact_agreement_count > labeled_answer_count
        or not isinstance(calibration["groups"], list)
        or any(calibration.get(field) is not expected for field, expected in expected_flags.items())
        or (labeled_answer_count == 0 and rate is not None)
        or (
            labeled_answer_count > 0
            and (
                type(rate) not in {int, float}
                or not math.isfinite(float(rate))
                or not 0.0 <= float(rate) <= 1.0
            )
        )
    ):
        raise HomeDashboardError("home dashboard review calibration is invalid")
    agreement = (
        "暂无一致率（尚未完成人工标注）"
        if rate is None
        else f"模型与人工精确一致 {exact_agreement_count}/{labeled_answer_count}"
        f"（{float(rate):.1%}）"
    )
    return (
        '<article class="card review-card"><div class="eyebrow">准确率闭环</div>'
        "<h2>人工标注进度</h2>"
        f'<p class="muted">已完成 {labeled_answer_count}/{answer_count} 个问题；'
        f"{agreement}。每个问题只需点一个答案，不用填写文字。</p>"
        '<p class="muted">这里只衡量可见事实答案一致性，不是危险识别率。</p></article>'
    )


def render_home_review(
    items: tuple[dict[str, object], ...],
    csrf_token: str,
    review_nonces: dict[tuple[str, str], str],
    *,
    calibration: dict[str, object] | None = None,
) -> bytes:
    if (
        not isinstance(items, tuple)
        or not isinstance(csrf_token, str)
        or not 32 <= len(csrf_token) <= 128
        or not isinstance(review_nonces, dict)
    ):
        raise HomeDashboardError("home dashboard review is invalid")
    csrf = html.escape(csrf_token, quote=True)
    cards: list[str] = []
    for item in items[:6]:
        if not isinstance(item, dict):
            continue
        token = item.get("review_token")
        gate_source = item.get("gate_source")
        frame_count = item.get("frame_count")
        clip_available = item.get("clip_available")
        answers = item.get("answers")
        recorded_at = item.get("recorded_at")
        review_state = item.get("review_state")
        review_reasons = item.get("review_reasons")
        manual_review_recommended = item.get("manual_review_recommended")
        if (
            not isinstance(token, str)
            or _REVIEW_TOKEN.fullmatch(token) is None
            or not isinstance(gate_source, str)
            or gate_source not in _GATE_SOURCE_LABELS
            or type(frame_count) is not int
            or not 1 <= frame_count <= 8
            or type(clip_available) is not bool
            or not isinstance(answers, list)
            or not isinstance(recorded_at, str)
            or not 1 <= len(recorded_at) <= 80
            or not isinstance(review_state, str)
            or review_state not in _VISUAL_REVIEW_STATE_LABELS
            or not isinstance(review_reasons, list)
            or len(set(review_reasons)) != len(review_reasons)
            or any(reason not in _VISUAL_REVIEW_REASON_LABELS for reason in review_reasons)
            or type(manual_review_recommended) is not bool
            or manual_review_recommended != (review_state != "no_rule_match")
        ):
            continue
        escaped_token = html.escape(token, quote=True)
        media = "".join(
            f'<img loading="lazy" referrerpolicy="no-referrer" '
            f'src="/review-media/{escaped_token}/frame/{index}" '
            f'alt="本地候选关键帧 {index + 1}">'
            for index in range(min(frame_count, 6))
        )
        if clip_available:
            media += (
                '<video controls preload="none">'
                f'<source src="/review-media/{escaped_token}/clip/0">'
                "浏览器无法播放本地候选片段。</video>"
            )
        answer_markup: list[str] = []
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            predicate = answer.get("predicate")
            model_answer = answer.get("model_answer")
            model_confidence = answer.get("model_confidence")
            human_answer = answer.get("human_answer")
            allowed = answer.get("allowed_human_answers")
            expected_allowed = _REVIEW_ALLOWED_ANSWERS.get(predicate)
            if (
                not isinstance(predicate, str)
                or predicate not in _PREDICATE_LABELS
                or not isinstance(model_answer, str)
                or model_answer not in _PREDICATE_ANSWER_LABELS
                or type(model_confidence) not in {int, float}
                or not 0.0 <= float(model_confidence) <= 1.0
                or human_answer is not None
                and (
                    not isinstance(human_answer, str)
                    or human_answer not in _PREDICATE_ANSWER_LABELS
                )
                or not isinstance(allowed, list)
                or any(not isinstance(value, str) for value in allowed)
                or expected_allowed is None
                or set(allowed) != set(expected_allowed)
                or len(allowed) != len(expected_allowed)
            ):
                continue
            nonce = review_nonces.get((token, predicate))
            if not isinstance(nonce, str) or not 16 <= len(nonce) <= 128:
                continue
            choices: list[str] = []
            for value in expected_allowed:
                selected = " selected" if value == human_answer else ""
                choices.append(
                    f'<button class="review-choice{selected}" type="submit" '
                    f'name="human_answer" value="{html.escape(value, quote=True)}">'
                    f"{html.escape(_PREDICATE_ANSWER_LABELS[value])}</button>"
                )
            current = (
                f" · 人工：{_PREDICATE_ANSWER_LABELS[human_answer]}"
                if isinstance(human_answer, str)
                else " · 尚未人工标注"
            )
            answer_markup.append(
                '<div class="review-answer">'
                f"<strong>{html.escape(_PREDICATE_LABELS[predicate])}</strong>"
                f'<p class="muted">模型：{html.escape(_PREDICATE_ANSWER_LABELS[model_answer])} '
                f"{float(model_confidence):.0%}（未校准）{html.escape(current)}</p>"
                '<form method="post" action="/review-label">'
                f'<input type="hidden" name="csrf" value="{csrf}">'
                f'<input type="hidden" name="nonce" value="{html.escape(nonce, quote=True)}">'
                f'<input type="hidden" name="review_token" value="{escaped_token}">'
                f'<input type="hidden" name="predicate" '
                f'value="{html.escape(predicate, quote=True)}">'
                f'<div class="review-choices">{"".join(choices)}</div></form></div>'
            )
        if not answer_markup:
            continue
        display_time = html.escape(recorded_at.replace("T", " ")[:19])
        reason_text = "、".join(_VISUAL_REVIEW_REASON_LABELS[reason] for reason in review_reasons)
        review_text = _VISUAL_REVIEW_STATE_LABELS[review_state]
        if reason_text:
            review_text += f"：{reason_text}"
        cards.append(
            '<article class="card review-card">'
            f'<div class="eyebrow">{html.escape(_GATE_SOURCE_LABELS[gate_source])}</div>'
            f"<h2>{display_time} UTC · 本地候选证据</h2>"
            f'<p class="muted"><strong>{html.escape(review_text)}</strong>。'
            "该提示只决定回看顺序，不是风险或通知结论。</p>"
            f'<div class="review-media">{media}</div>'
            f"{''.join(answer_markup)}</article>"
        )
    calibration_summary = _render_review_calibration(calibration)
    content = "".join(cards) or (
        '<article class="card review-card"><h2>暂无可回看的 Shadow 结果</h2>'
        '<p class="muted">先手动运行一次 INT8 Shadow；页面本身不会启动模型。</p></article>'
    )
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baby Monitor Shadow Review</title><link rel="stylesheet" href="/style.css">
</head><body><main><div class="eyebrow">仅限本机 · Shadow 人工审计</div>
<h1>候选事件回看</h1>
<p class="muted">关键帧按需读取；视频不会自动播放。这里的标签只用于后续离线评估，
不会改变风险级别、通知或主分析队列。</p>
{calibration_summary}
{content}
<a class="control-link" href="/">返回状态页</a>
<p class="foot">页面不显示摄像头名、事件 ID 或本地文件路径 · 不上传 · 不训练</p>
</main></body></html>"""
    return body.encode("utf-8")


def render_home_controls(
    csrf_token: str,
    action_nonces: dict[str, str],
    *,
    notification_test_enabled: bool = False,
    daily_report_enabled: bool = False,
    daily_digest_enabled: bool = False,
) -> bytes:
    if (
        not isinstance(csrf_token, str)
        or set(action_nonces) != set(_ACTIONS)
        or any(not isinstance(value, str) for value in action_nonces.values())
        or type(notification_test_enabled) is not bool
        or type(daily_report_enabled) is not bool
        or type(daily_digest_enabled) is not bool
    ):
        raise HomeDashboardError("home dashboard controls are invalid")
    csrf = html.escape(csrf_token, quote=True)

    def action_form(action: str, label: str, *, stop: bool = False) -> str:
        expected = _ACTIONS[action]
        if expected:
            field = (
                f'<input name="confirmation" autocomplete="off" '
                f'placeholder="输入 {html.escape(expected)}" aria-label="确认口令">'
            )
        else:
            field = '<input type="hidden" name="confirmation" value="">'
        button_class = ' class="stop"' if stop else ""
        return (
            '<form class="action" method="post" action="/action">'
            f'<input type="hidden" name="csrf" value="{csrf}">'
            f'<input type="hidden" name="nonce" '
            f'value="{html.escape(action_nonces[action], quote=True)}">'
            f'<input type="hidden" name="action" value="{html.escape(action, quote=True)}">'
            f'{field}<button{button_class} type="submit">'
            f"{html.escape(label)}</button></form>"
        )

    notification_test_form = (
        action_form("notification-test", "发送本地测试通知") if notification_test_enabled else ""
    )
    daily_report_form = (
        action_form("daily-report", "生成今日隐私日报") if daily_report_enabled else ""
    )
    daily_digest_form = (
        action_form("daily-digest", "生成最近一天精华") if daily_digest_enabled else ""
    )

    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baby Monitor Controls</title><link rel="stylesheet" href="/style.css">
</head><body><main><div class="eyebrow">每次操作单独授权</div>
<h1>手动控制</h1><section class="grid"><article class="card">
<h2>固定白名单操作</h2><p class="muted">打开页面不会启动任何服务。START、SHADOW、SOAK24
和 INT8 都需要明确输入确认；每个表单口令只能使用一次。</p>
<div class="controls">
{action_form("start", "启动基础监控")}
{action_form("stop", "停止全部监控", stop=True)}
{action_form("shadow-start", "启动有界 Shadow")}
{action_form("shadow-stop", "停止 Shadow", stop=True)}
{action_form("soak-start", "启动四路 YOLO presence 24 小时稳定性测试")}
{action_form("soak-stop", "停止四路稳定性测试", stop=True)}
{action_form("int8-shadow-once", "运行一次 INT8")}
{notification_test_form}
{daily_report_form}
{daily_digest_form}
</div><a class="control-link" href="/">返回状态页</a>
</article></section><p class="foot">TEST 只生成一条明确标记的本机通知测试，不代表真实事件，
也不会发送邮件或微信。隐私日报只汇总已落库记录，不读取媒体或触发通知。
每日精华会在本机读取并转码最近有事件的一天，输入 DIGEST 后才执行，不会外发或自动调度。
SOAK24 只启动本地米家视频、四路环形缓冲和串行 YOLO presence 扫描，
只写 person/clear 事实，不抽取事件证据、不写分析队列，24 小时后自动停止；
不启动 Mage-VL、身份、风险、主分析或通知 worker。
此处不提供媒体删除、远程上传或任意 systemd 操作。</p>
</main></body></html>"""
    return body.encode("utf-8")
