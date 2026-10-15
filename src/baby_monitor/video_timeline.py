"""Shared decoded-frame timestamps for detector and pose scans."""

from __future__ import annotations

import math
from typing import Any


def decoded_frame_timestamp_seconds(
    capture: Any,
    *,
    position_property: object,
    frame_index: int,
    native_fps: float,
) -> float:
    return decoded_frame_timestamp(
        capture,
        position_property=position_property,
        frame_index=frame_index,
        native_fps=native_fps,
    )[0]


def decoded_frame_timestamp(
    capture: Any,
    *,
    position_property: object,
    frame_index: int,
    native_fps: float,
) -> tuple[float, str]:
    """Use finite container PTS, or nominal frame/FPS when it is unavailable.

    Some OpenCV backends report zero for every frame rather than exposing PTS;
    zero is therefore accepted only for the initial decoded frame.
    """
    nominal_timestamp = frame_index / native_fps
    if not isinstance(position_property, int):
        return nominal_timestamp, "nominal_fps"
    try:
        position_msec = float(capture.get(position_property))
    except (TypeError, ValueError):
        return nominal_timestamp, "nominal_fps"
    if (
        math.isfinite(position_msec)
        and position_msec >= 0.0
        and (position_msec > 0.0 or frame_index == 0)
    ):
        return position_msec / 1_000.0, "container_pts"
    return nominal_timestamp, "nominal_fps"
