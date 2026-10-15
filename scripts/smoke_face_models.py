#!/usr/bin/env python3
"""Run an offline YuNet/SFace load smoke test on synthetic pixels only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--sface", type=Path, required=True)
    args = parser.parse_args()

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    detector = cv2.FaceDetectorYN.create(str(args.yunet), "", (320, 320), 0.9, 0.3, 5_000)
    detector.setInputSize((320, 320))
    _, faces = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    recognizer = cv2.FaceRecognizerSF.create(str(args.sface), "")
    feature = recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))

    print(
        json.dumps(
            {
                "opencv_version": cv2.__version__,
                "yunet_loaded": True,
                "synthetic_faces": 0 if faces is None else len(faces),
                "sface_loaded": True,
                "embedding_shape": list(feature.shape),
                "embedding_finite": bool(np.isfinite(feature).all()),
                "household_media_used": False,
                "network_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
