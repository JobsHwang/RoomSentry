"""Bounded offline four-frame throughput/parity probe; optional vision runtime required.

Use an explicitly selected public fixture. This measures runtime and output parity,
not accuracy, identity, household acceptance, or camera-to-detector latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.live_detector import PersistentYoloDetector
from baby_monitor.live_detection import LiveFrame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public-image", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np

    image = cv2.imread(str(args.public_image))
    if image is None or image.size > 12_000_000:
        raise ValueError("bounded public fixture required")
    frames = []
    for index, width in enumerate((848, 848, 864, 848), start=1):
        canvas = np.zeros((480, width, 3), dtype=np.uint8)
        if index != 4:
            scale = min(width / image.shape[1], 480 / image.shape[0])
            resized = cv2.resize(
                image, (round(image.shape[1] * scale), round(image.shape[0] * scale))
            )
            if index == 2:
                resized = cv2.flip(resized, 1)
            height, content_width = resized.shape[:2]
            y, x = (480 - height) // 2, (width - content_width) // 2
            canvas[y : y + height, x : x + content_width] = resized
        frames.append(
            LiveFrame(
                "public_" + str(index),
                0,
                width,
                480,
                datetime.now(UTC),
                time.monotonic(),
                canvas.tobytes(),
            )
        )
    command = (
        "/usr/bin/unshare",
        "--user",
        "--map-root-user",
        "--net",
        str(args.vision_python),
        "-m",
        "baby_monitor.local_runners.live_yolo",
    )
    results = []
    for half, height in ((False, None), (True, None), (True, 384)):
        detector = PersistentYoloDetector(
            command=command,
            model_path=str(args.model),
            manifest_path=str(args.manifest),
            half_precision=half,
            image_height=height,
        )
        try:
            detector.start()
            timings, predictions = [], ()
            for index in range(14):
                began = time.perf_counter()
                predictions = detector.predict(tuple(frames))
                duration = time.perf_counter() - began
                if index >= 2:
                    timings.append(duration)
            results.append(
                {
                    "identity": detector.identity,
                    "batch_size": len(frames),
                    "measured_batches": len(timings),
                    "median_batch_ms": round(statistics.median(timings) * 1000, 3),
                    "median_aggregate_fps": round(len(frames) / statistics.median(timings), 3),
                    "predictions": [
                        [
                            {"confidence": item.confidence, "bbox_xyxy": item.bbox_xyxy}
                            for item in row
                        ]
                        for row in predictions
                    ],
                }
            )
        finally:
            detector.close()
    print(
        json.dumps(
            {
                "scope": "public_static_runtime_parity_not_accuracy",
                "recorded_at": datetime.now(UTC).isoformat(),
                "variants_loaded_sequentially": True,
                "remote_used": False,
                "results": results,
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
