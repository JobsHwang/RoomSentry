"""Offline, synthetic FFmpeg/OpenCV integration check; no model weights are loaded.

Run with an already installed optional vision runtime from the repository root.
This deliberately reuses test-only model stubs, while decoding actual temporary
CFR/VFR videos through the production detector and pose sampling loops.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# A checkout is required: model stubs belong to tests, never the installed core.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baby_monitor.adapters.camera import LocalFileCamera  # noqa: E402
from baby_monitor.adapters.detector import UltralyticsYoloDetector  # noqa: E402
from baby_monitor.detector_protocol import AdmittedDetectorScan  # noqa: E402
from baby_monitor.local_runners.yolo_pose import run_request  # noqa: E402
from baby_monitor.video_timeline import decoded_frame_timestamp_seconds  # noqa: E402
from baby_monitor.yolo_model import YoloModelManifest  # noqa: E402
from scripts.detector_timeline_diagnostic import diagnose_detector_timeline  # noqa: E402
from tests.test_yolo_model import _FakeYolo as DetectorStub  # noqa: E402
from tests.test_yolo_model import _write_fixture  # noqa: E402
from tests.test_yolo_pose import _FakeYolo as PoseStub  # noqa: E402
from tests.test_yolo_pose import _request_fixture  # noqa: E402


def _command(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30
    )
    if result.returncode:
        # Do not disclose command arguments or local paths on failure.
        raise RuntimeError(f"synthetic {Path(arguments[0]).name} command failed")
    return result.stdout


def _assert_aligned(actual: list[float], expected: list[float]) -> None:
    if len(actual) != len(expected) or not all(
        math.isclose(left, right, abs_tol=0.00001, rel_tol=0)
        for left, right in zip(actual, expected, strict=True)
    ):
        raise AssertionError("decoded timestamps differ from independent container PTS")


def verify() -> dict[str, object]:
    import cv2  # type: ignore[import-not-found]

    cases = []
    with tempfile.TemporaryDirectory(prefix="baby-monitor-timeline-") as directory:
        root = Path(directory)
        manifest_path, weight_path = _write_fixture(root)
        manifest = YoloModelManifest.read(manifest_path)
        pose_request = _request_fixture(root)
        for variable_rate in (False, True):
            video = root / ("synthetic_vfr.mp4" if variable_rate else "synthetic_cfr.mp4")
            arguments = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x96:rate=10:duration=2",
            ]
            if variable_rate:
                arguments += ["-vf", "select=not(mod(n\\,3))+not(mod(n\\,7))"]
            arguments += [
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ]
            _command(arguments)
            probe = json.loads(
                _command(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_frames",
                        "-show_entries",
                        "frame=best_effort_timestamp_time",
                        "-of",
                        "json",
                        str(video),
                    ]
                )
            )
            reference = [float(item["best_effort_timestamp_time"]) for item in probe["frames"]]
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError("could not open synthetic video")
            decoded = []
            try:
                native_fps = float(capture.get(cv2.CAP_PROP_FPS))
                backend = capture.getBackendName()
                while True:
                    ok, _frame = capture.read()
                    if not ok:
                        break
                    decoded.append(
                        decoded_frame_timestamp_seconds(
                            capture,
                            position_property=cv2.CAP_PROP_POS_MSEC,
                            frame_index=len(decoded),
                            native_fps=native_fps,
                        )
                    )
            finally:
                capture.release()
            _assert_aligned(decoded, reference)
            sample_fps = 2.0
            frame_step = max(1, round(native_fps / sample_fps))
            expected = reference[::frame_step]
            detector = UltralyticsYoloDetector(
                model_path=weight_path,
                manifest=manifest,
                labels=("person",),
                confidence=0.35,
                sample_fps=sample_fps,
                device="cpu",
            )
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=DetectorStub)}):
                detector_result = detector.scan_result(
                    LocalFileCamera("synthetic_cam", video).source()
                )
                detections = detector_result.detections
            ledger = detector_result.sampling
            if ledger is None or ledger.decoded_frame_count != len(reference):
                raise AssertionError("detector ledger did not preserve decoded-frame count")
            _assert_aligned([sample.timestamp_seconds for sample in ledger.samples], expected)
            if [sample.decoded_frame_index for sample in ledger.samples] != list(
                range(0, len(reference), frame_step)
            ):
                raise AssertionError("detector ledger indexes differ from the actual sampling grid")
            if ledger.source_exhaustion_verified or ledger.end_reason != "read_returned_false":
                raise AssertionError("detector ledger overclaims source exhaustion")
            # Synthetic binding only: the pure diagnostic does not authenticate
            # admissions. Real-media callers must independently verify bytes.
            timeline_audit = diagnose_detector_timeline(
                AdmittedDetectorScan(
                    source_admission_sha256="a" * 64,
                    labels=("person",),
                    confidence_threshold=0.35,
                    requested_sample_fps=sample_fps,
                    result=detector_result,
                    version=2,
                ),
                tuple(reference),
                reference_source_admission_sha256="a" * 64,
            )
            if not (
                timeline_audit["decoded_frame_count_matches_reference"]
                and timeline_audit["all_sample_timestamps_match"]
            ):
                raise AssertionError("synthetic independent timeline audit failed")
            with (
                patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=DetectorStub)}),
                patch.object(DetectorStub, "predict", return_value=[SimpleNamespace(boxes=None)]),
            ):
                empty_result = detector.scan_result(
                    LocalFileCamera("synthetic_cam", video).source()
                )
            if empty_result.detections or any(
                sample.detection_count for sample in empty_result.sampling.samples
            ):
                raise AssertionError("empty model did not preserve zero-box calls")
            _assert_aligned(
                [sample.timestamp_seconds for sample in empty_result.sampling.samples], expected
            )
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=PoseStub)}):
                poses = run_request(
                    replace(
                        pose_request,
                        source_path=str(video),
                        sample_fps=sample_fps,
                        max_frames=100,
                    )
                )
            detector_times = [item.timestamp_seconds for item in detections]
            pose_times = [item.timestamp_seconds for item in poses.frames]
            _assert_aligned(detector_times, expected)
            _assert_aligned(pose_times, expected)
            nominal = [index / native_fps for index in range(0, len(reference), frame_step)]
            old_pose_max_error = max(
                abs(left - right)
                for left, right in zip(
                    nominal,
                    expected,
                    strict=True,
                )
            )
            if variable_rate and old_pose_max_error <= 0.01:
                raise AssertionError("VFR fixture must expose nominal-FPS timestamp drift")
            cases.append(
                {
                    "kind": "vfr" if variable_rate else "cfr",
                    "backend": backend,
                    "decoded_frames": len(decoded),
                    "native_fps": native_fps,
                    "sample_fps": sample_fps,
                    "frame_step": frame_step,
                    "ffprobe_pts": reference,
                    "detector_pts": detector_times,
                    "pose_pts": pose_times,
                    "detector_sample_count": len(ledger.samples),
                    "empty_model_sample_count": len(empty_result.sampling.samples),
                    "detector_decoded_frame_count": ledger.decoded_frame_count,
                    "max_adjacent_container_pts_gap_seconds": (
                        ledger.max_adjacent_container_pts_gap_seconds
                    ),
                    "source_exhaustion_verified": ledger.source_exhaustion_verified,
                    "old_pose_max_timestamp_error_seconds": round(old_pose_max_error, 9),
                    "reference_timeline_alignment_verified": (
                        timeline_audit["all_sample_timestamps_match"]
                    ),
                }
            )
    return {
        "version": 1,
        "opencv_version": cv2.__version__,
        "cases": cases,
        "models": "test_stubs_only",
        "weights_loaded": False,
        "household_media_read": False,
        "network_used": False,
        "temporary_synthetic_media_removed": True,
        "accuracy_evaluated": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
