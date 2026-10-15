"""Explicit per-camera empty-view cadence, independent of expensive fallback checks."""

from __future__ import annotations

import math
import re


def validate_empty_review_intervals(value: object, *, enabled: bool) -> dict[str, float]:
    """The public default stays 1 Hz; opt-in empty-view checks may reach at most 5 Hz."""
    if value is None:
        return {}
    if type(value) is not dict or len(value) > 4 or (value and not enabled):
        raise ValueError("invalid per-camera alternate review cadence")
    result = {}
    for camera_id, interval in value.items():
        if (
            not isinstance(camera_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", camera_id) is None
            or type(interval) not in (int, float)
            or not math.isfinite(interval)
            or not 0.2 <= interval <= 1.0
        ):
            raise ValueError("invalid per-camera alternate review cadence")
        result[camera_id] = float(interval)
    return result


class AlternateReviewCadence:
    """Keep legacy shared rotation spacing; faster overrides affect only empty native views.

    Nonempty shadow views remain limited to 1 Hz after any rotation. Large-scale
    fallback has its own 1 Hz ceiling, so raising empty-view cadence cannot multiply it.
    These are check eligibility limits, not guaranteed inference rates or accuracy claims.
    """

    def __init__(self, intervals: dict[str, float]):
        self.intervals = validate_empty_review_intervals(intervals, enabled=True)
        self.last_rotation: dict[str, float] = {}
        self.last_fallback: dict[str, float] = {}

    def rotation_due(self, camera_id: str, *, native_present: bool, now: float) -> bool:
        interval = 1.0 if native_present else self.intervals.get(camera_id, 1.0)
        return now - self.last_rotation.get(camera_id, -math.inf) >= interval

    def mark_rotation(self, camera_id: str, now: float) -> None:
        self.last_rotation[camera_id] = now

    def fallback_due(self, camera_id: str, now: float) -> bool:
        return now - self.last_fallback.get(camera_id, -math.inf) >= 1.0

    def mark_fallback(self, camera_id: str, now: float) -> None:
        self.last_fallback[camera_id] = now
