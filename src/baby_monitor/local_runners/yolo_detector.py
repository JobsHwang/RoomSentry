from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import DetectorError, UltralyticsYoloDetector
from baby_monitor.detector_protocol import (
    DetectorProtocolError,
    DetectorScanRequest,
    DetectorScanResult,
)
from baby_monitor.yolo_model import YoloModelManifest


def run_request(request: DetectorScanRequest) -> DetectorScanResult:
    manifest = YoloModelManifest.read(request.manifest_path)
    if (
        manifest.model_id != request.model_id
        or manifest.revision != request.model_revision
        or manifest.sha256 != request.model_sha256
        or not set(request.labels).issubset(manifest.labels)
    ):
        raise DetectorProtocolError("detector request does not match the reviewed manifest")
    detector = UltralyticsYoloDetector(
        model_path=request.model_path,
        manifest=manifest,
        labels=request.labels,
        confidence=request.confidence,
        sample_fps=request.sample_fps,
        device=request.device,
        image_size=request.image_size,
        rotation_degrees=request.rotation_degrees,
    )
    source = LocalFileCamera(request.camera_id, request.source_path).source()
    return replace(detector.scan_result(source), detector_adapter=request.detector_adapter)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("request")
    args = parser.parse_args(argv)
    try:
        request = DetectorScanRequest.from_json(Path(args.request).read_text(encoding="utf-8"))
        # Reserve stdout for the one strict result object. Third-party model
        # runtimes sometimes emit informational banners even with verbose
        # inference disabled; route those to the already-suppressed stderr
        # channel so they cannot corrupt the parent/child JSON contract.
        with redirect_stdout(sys.stderr):
            result = run_request(request)
    except (DetectorError, DetectorProtocolError, OSError, RuntimeError, ValueError):
        print("detector_runner_failed", file=sys.stderr)
        return 2
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
