"""Deterministic synthetic scheduling diagnostic; no camera, media or model I/O.

Assumes perfect visibility and a perfect detector on sampled frames. Results
measure admission opportunities, NOT YOLO recall, identity or safety accuracy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from baby_monitor.live_detection import FrameMailbox, LiveFrame, MotionBurstSettings


def phase_sweep() -> dict[str, object]:
    durations = (50, 100, 200, 500)
    phases = tuple(range(0, 200, 5))
    blank = bytes(32 * 24 * 3)
    patch = bytearray(blank)
    for y in range(8, 16):
        for x in range(12, 20):
            offset = (y * 32 + x) * 3
            patch[offset : offset + 3] = b"\xff\xff\xff"
    present = bytes(patch)
    rows = []
    for duration in durations:
        for mode, rate, burst in (
            ("full_20fps", None, None),
            ("fixed_5fps", 5, None),
            ("motion_5_to_20fps", 5, MotionBurstSettings()),
        ):
            seen_once = seen_twice = total_selected = 0
            for phase in phases:
                mailbox = FrameMailbox(("synthetic_camera",), sample_fps=rate, motion_burst=burst)
                start = 1000 + phase
                hits = 0
                for millisecond in range(0, 2000, 50):
                    visible = start <= millisecond < start + duration
                    now = millisecond / 1000
                    frame = LiveFrame(
                        "synthetic_camera",
                        millisecond // 50,
                        32,
                        24,
                        datetime(2026, 9, 30, tzinfo=UTC),
                        now,
                        present if visible else blank,
                    )
                    mailbox.put(frame)
                    selected = mailbox.take(now)
                    total_selected += len(selected)
                    hits += len(selected) if visible else 0
                seen_once += hits >= 1
                seen_twice += hits >= 2
            rows.append(
                {
                    "duration_ms": duration,
                    "mode": mode,
                    "phases": len(phases),
                    "at_least_one_admitted_frame": seen_once,
                    "at_least_two_admitted_frames": seen_twice,
                    "total_admitted_frames": total_selected,
                }
            )
    return {
        "scope": "synthetic_sampling_opportunity_not_detection_accuracy",
        "source_fps": 20,
        "phase_step_ms": 5,
        "assumptions": [
            "perfect_contrast_8x8_patch_in_32x24_grid",
            "regular_decoder_receipt_clock_no_queue_delay",
            "perfect_detection_on_every_admitted_visible_frame",
            "no_motion_identification_or_review_success_claim",
        ],
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(phase_sweep(), indent=2))
