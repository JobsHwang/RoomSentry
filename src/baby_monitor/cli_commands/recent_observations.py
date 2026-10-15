"""Explicit local description query, separate from the public status digest."""

from __future__ import annotations

import argparse
import unicodedata

from ..cli_support import _json_print
from ..recent_observations import read_recent_observations
from ..room_registry import RoomRegistry


def _display_text(value: str) -> str:
    # Stored model prose is untrusted terminal input, not a control channel.
    return "".join(" " if unicodedata.category(c).startswith("C") else c for c in value)


def render_recent_observations_text(payload: dict[str, object]) -> str:
    """Present historical descriptions without inventing a live visibility cause."""
    lines = [
        "宝宝当前位置：无法确认",
        "宝宝当前状态：无法确认（本入口只查询历史记录，不读取实时画面）",
        "看不清时允许保持未知；未知不代表无人、睡着或安全。",
    ]
    statuses = {
        "no_record": "暂无记录",
        "invalid_record": "最新记录不可用",
        "stale_record": "记录已过期，不能代表现在",
        "recent_record": "近期历史记录，仍不是实时状态",
    }
    for entry in payload["entries"]:
        lines.extend(
            [
                "",
                f"{_display_text(entry['mapped_room'])} / {entry['camera_id']}",
                statuses[entry["status"]],
            ]
        )
        if "recorded_at" not in entry:
            continue
        lines.append(f"记录时间：{entry['recorded_at']}；距查询时 {entry['age_seconds']:.1f} 秒")
        reported = entry["model_reported"]
        for field, heading in (
            ("observations", "当时的模型画面描述（未独立验证）"),
            ("inferences", "当时的模型推测（不是事实）"),
        ):
            lines.append(heading + "：")
            values = reported[field]
            if not values:
                lines.append("  - 无")
            for value in values:
                lines.append("  - " + _display_text(value))
    lines.extend(["", "房间名称按当前配置显示；不证明历史机位，亦未判断未知是否由遮挡或断流造成。"])
    return "\n".join(lines)


def command_recent_observations(args: argparse.Namespace) -> int:
    payload = read_recent_observations(
        args.database,
        RoomRegistry.read_private(args.registry),
        recent_seconds=args.recent_seconds,
    )
    if args.format == "text":
        print(render_recent_observations_text(payload))
    else:
        _json_print(payload)
    return 0
