"""Shared UI copy, label maps, and form constants for the local dashboard."""

from __future__ import annotations

import re

_ACTIONS = {
    "start": "START",
    "stop": "",
    "shadow-start": "SHADOW",
    "shadow-stop": "",
    "soak-start": "SOAK24",
    "soak-stop": "",
    "int8-shadow-once": "INT8",
    "notification-test": "TEST",
    "daily-report": "",
    "daily-digest": "DIGEST",
}


_COMPONENT_LABELS = {
    "xiaomi_stream_gateway": "米家本地视频网关",
    "home_assistant_signals": "Home Assistant 事件监听",
    "local_evidence_buffer": "本地环形证据缓冲",
    "local_evidence_buffer_camera_02": "本地证据缓冲 · C500 第二视角",
    "local_evidence_buffer_camera_03": "本地证据缓冲 · 摄像头 2",
    "local_evidence_buffer_camera_04": "本地证据缓冲 · 摄像头 3",
    "ha_signal_buffer_dispatch": "HA 事件与本地视频匹配",
    "bounded_home_shadow": "有界 YOLO 候选分析",
    "candidate_soak_24h": "四路 YOLO presence 24 小时稳定性测试",
    "int8_mage_shadow_once": "INT8 Mage-VL 单次复核",
}


_STATE_LABELS = {
    "running": "运行中",
    "inactive": "未运行",
    "failed": "失败",
    "not-installed": "未安装",
    "unknown": "未知",
}


_CANDIDATE_CAMERA_LABELS = {
    "camera_01": "C500 主视角",
    "camera_02": "C500 第二视角",
    "camera_03": "摄像头 2",
    "camera_04": "摄像头 3",
}


_CANDIDATE_STATE_LABELS = {
    "starting": "正在启动",
    "scanning": "正在分析片段",
    "idle": "等待新片段",
    "failed": "处理失败",
    "stopped": "已停止",
}


_CANDIDATE_FRESHNESS_LABELS = {
    "fresh": "状态新鲜",
    "stale": "状态已过期",
    "clock_skew": "时钟异常",
}


_SOAK_STATE_LABELS = {
    "running": "正在运行",
    "completed": "已完成",
    "stopped_early": "提前停止",
    "interrupted": "异常中断",
}


_ROOM_PRESENCE_STATE_LABELS = {
    "person_present": "检测到人员",
    "temporarily_clear": "暂未检测到人员",
    "unknown": "证据不足",
    "coverage_gap": "覆盖中断",
}


_ROOM_PRESENCE_REASON_LABELS = {
    "person_confirmed": "连续人员检测已确认",
    "confirmation_pending": "等待第二次人员检测确认",
    "all_streams_clear": "所有视角达到连续 clear 门槛",
    "clear_confirmation_pending": "仍在等待连续 clear",
    "no_samples": "尚无扫描样本",
    "stream_stale": "扫描样本已过期",
    "stream_coverage_gap": "至少一路停止、失败或覆盖中断",
}


_QUEUE_STATUS_LABELS = {
    "available": "可用",
    "not_configured": "未配置",
    "not_initialized": "尚未运行",
    "not_generated": "尚未生成",
    "unavailable": "不可用",
}


_PREDICATE_LABELS = {
    "face_visibility": "脸部可见",
    "support_contact_visible": "支撑接触",
    "posture_transition_visible": "姿态变化",
    "covering_near_face_visible": "脸部附近遮挡",
}


_PREDICATE_ANSWER_LABELS = {
    "yes": "是",
    "no": "否",
    "unknown": "未知",
    "clear": "清晰",
    "partially_obscured": "部分遮挡",
    "not_visible": "不可见",
}


_REVIEW_ALLOWED_ANSWERS = {
    "face_visibility": ("clear", "partially_obscured", "not_visible", "unknown"),
    "support_contact_visible": ("yes", "no", "unknown"),
    "posture_transition_visible": ("yes", "no", "unknown"),
    "covering_near_face_visible": ("yes", "no", "unknown"),
}


_GATE_SOURCE_LABELS = {
    "hardware_signal": "HA 硬件事件",
    "person_detection": "YOLO 人形候选",
    "hardware_and_person": "HA + YOLO 复合 gate",
    "periodic_sample": "有界抽检",
}


_VISUAL_REVIEW_STATE_LABELS = {
    "review_recommended": "建议优先人工回看",
    "unknown": "信息不足，建议人工回看",
    "no_rule_match": "未命中回看规则（不代表安全）",
}


_VISUAL_REVIEW_REASON_LABELS = {
    "face_not_visible": "模型未看到脸部",
    "face_partially_obscured": "模型看到脸部部分遮挡",
    "covering_near_face_visible": "模型看到脸部附近有遮挡物",
    "face_visibility_unknown": "脸部可见性未知",
    "covering_near_face_unknown": "脸部附近遮挡未知",
    "required_face_visibility_missing": "缺少脸部可见性结果",
    "required_covering_near_face_missing": "缺少脸部附近遮挡结果",
}


_REVIEW_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}\Z")


_BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")


_LOCAL_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


_LOCAL_TIME = re.compile(r"\d{2}:\d{2}:\d{2}\Z")


_DAILY_DIGEST_REASON_LABELS = {
    "event_recorded": "已记录事件",
    "human_keep": "人工标记保留",
    "person_fact": "检测到人形事实",
    "motion_fact": "检测到运动事实",
    "hardware_signal": "收到硬件信号",
    "zone_signal": "收到区域信号",
    "risk_field": "原记录含风险字段",
    "coverage_complete": "连续覆盖加分（未验证）",
}


_DAILY_DIGEST_GAP_LABELS = {
    "buffer_started_after_event": "本地缓冲开始较晚",
    "buffer_ended_before_event": "本地缓冲结束较早",
    "segment_gap": "本地视频片段不连续",
    "segment_missing": "事件证据片段缺失",
    "missing_before": "事件前画面不足",
    "missing_after": "事件后画面不足",
}


_OUTBOX_ERROR_LABELS = {
    "route_missing": "摄像头路由未接通",
    "analysis_queue_full": "分析队列已满",
    "buffer_evidence_failed": "本地视频取证失败",
    "buffer_window_unavailable": "事件不在本地缓冲覆盖内",
    "media_command_unavailable": "本地 FFmpeg/FFprobe 不可用",
    "buffer_media_processing_failed": "本地视频处理失败",
    "buffer_config_invalid": "本地取证配置不可用",
    "signal_staging_invalid": "HA 事件暂存校验失败",
    "buffer_storage_unavailable": "本地证据存储不可写",
    "buffer_request_invalid": "本地取证请求无效",
    "buffer_runtime_failed": "本地取证运行失败",
    "buffer_command_invalid": "本地取证子命令启动失败",
    "buffer_contract_type_error": "本地取证契约类型错误",
    "buffer_database_failed": "本地取证数据库失败",
    "buffer_command_timeout": "本地取证命令超时",
    "buffer_child_exception": "本地取证子进程异常",
    "buffer_module_unavailable": "本地取证 Python 模块不可用",
}


_EVENT_TYPE_LABELS = {
    "sleeping": "睡眠",
    "awake": "醒着",
    "crying": "哭闹",
    "movement": "活动",
    "adult_interaction": "成人互动",
    "possible_risk": "可能风险",
    "unknown": "未知",
}


_RISK_LEVEL_LABELS = {
    "none": "无",
    "low": "低",
    "medium": "中",
    "high": "高",
}


_MAX_FORM_BYTES = 2_048
