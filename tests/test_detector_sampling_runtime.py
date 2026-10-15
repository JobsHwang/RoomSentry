from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import DetectorError, UltralyticsYoloDetector
from baby_monitor.cli_commands.detection_eval import command_yolo_scan_file
from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanRequest
from baby_monitor.local_runners.yolo_detector import run_request
from baby_monitor.yolo_model import YoloModelManifest
from tests.test_yolo_model import _FakeYolo, _write_fixture


class DetectorSamplingRuntimeTests(unittest.TestCase):
    def run_scan(
        self, *, points=(0, 500, 4000, 4500, 5000), fps=2.0, predictions=None, runner=False
    ):
        class Capture:
            index = -1
            released = False

            def isOpened(self):
                return True

            def get(self, prop):
                return fps if prop == 5 else points[self.index]

            def read(self):
                self.index += 1
                return (True, object()) if self.index < len(points) else (False, None)

            def release(self):
                self.released = True

        capture = Capture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight = _write_fixture(root)
            source = root / "synthetic.mp4"
            source.write_bytes(b"test-only-video-stub")
            manifest = YoloModelManifest.read(manifest_path)
            detector = UltralyticsYoloDetector(
                model_path=weight,
                manifest=manifest,
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="cpu",
            )
            fake_cv = SimpleNamespace(
                VideoCapture=lambda _: capture, CAP_PROP_FPS=5, CAP_PROP_POS_MSEC=0
            )
            with (
                patch.dict(
                    sys.modules, {"cv2": fake_cv, "ultralytics": SimpleNamespace(YOLO=_FakeYolo)}
                ),
                patch.object(UltralyticsYoloDetector, "_release_accelerator_cache") as cleanup,
            ):
                if predictions is None:
                    normal = _FakeYolo("fixture").predict(
                        source=object(), conf=0.35, device="cpu", imgsz=640, verbose=False
                    )
                    predictions = [normal, [SimpleNamespace(boxes=None)], normal]
                with patch.object(_FakeYolo, "predict", side_effect=predictions) as predict:
                    try:
                        if runner:
                            result = run_request(
                                DetectorScanRequest(
                                    camera_id="synthetic_cam",
                                    detector_adapter="runner",
                                    source_path=str(source),
                                    model_path=str(weight),
                                    manifest_path=str(manifest_path),
                                    model_id=manifest.model_id,
                                    model_revision=manifest.revision,
                                    model_sha256=manifest.sha256,
                                    labels=("person",),
                                    confidence=0.35,
                                    sample_fps=1.0,
                                    device="cpu",
                                )
                            )
                        else:
                            result = detector.scan_result(
                                LocalFileCamera("synthetic_cam", source).source()
                            )
                    finally:
                        self.assertTrue(capture.released)
                        cleanup.assert_called_once()
                return result, predict.call_count

    def test_actual_calls_include_empty_frames_and_full_decode_pts_gap(self):
        result, calls = self.run_scan()
        self.assertEqual(result.version, 2)
        ledger = result.sampling
        self.assertEqual(calls, 3)
        self.assertEqual(ledger.decoded_frame_count, 5)
        self.assertEqual([s.decoded_frame_index for s in ledger.samples], [0, 2, 4])
        self.assertEqual([s.timestamp_seconds for s in ledger.samples], [0, 4, 5])
        self.assertEqual([s.detection_count for s in ledger.samples], [1, 0, 1])
        self.assertEqual([s.detection_start for s in ledger.samples], [0, 1, 1])
        self.assertEqual(ledger.max_adjacent_container_pts_gap_seconds, 3.5)
        self.assertEqual(ledger.timestamp_fallback_frame_count, 0)
        self.assertEqual(ledger.container_pts_regression_count, 0)
        self.assertEqual(ledger.end_reason, "read_returned_false")
        self.assertFalse(ledger.source_exhaustion_verified)
        ledger.validate_detections(result.detections)

    def test_regression_between_decoded_frames_is_not_hidden_by_sampling(self):
        result, _ = self.run_scan(points=(0, 500, 250, 1500, 2000))
        self.assertEqual(result.sampling.container_pts_regression_count, 1)
        self.assertEqual(result.sampling.max_adjacent_container_pts_gap_seconds, 1.25)

    def test_timestamp_fallback_is_explicit_for_all_decoded_frames(self):
        result, _ = self.run_scan(points=(0, 0, 0, 0, 0))
        ledger = result.sampling
        self.assertEqual(ledger.timestamp_fallback_frame_count, 4)
        self.assertEqual(
            [s.timestamp_source for s in ledger.samples],
            ["container_pts", "nominal_fps", "nominal_fps"],
        )
        self.assertEqual([s.timestamp_seconds for s in ledger.samples], [0, 1, 2])
        self.assertIsNone(ledger.max_adjacent_container_pts_gap_seconds)

    def test_fps_fallback_does_not_claim_container_fps(self):
        result, calls = self.run_scan(fps=0)
        self.assertEqual(calls, 1)
        self.assertEqual(result.sampling.native_fps_source, "fallback_25")
        self.assertEqual(result.sampling.native_fps, 25)
        self.assertEqual(result.sampling.samples[0].timestamp_source, "container_pts")

    def test_empty_or_multiple_result_objects_are_not_zero_box_success(self):
        for output in ([], [SimpleNamespace(boxes=None), SimpleNamespace(boxes=None)]):
            with (
                self.subTest(output=output),
                self.assertRaisesRegex(DetectorError, "exactly one result"),
            ):
                self.run_scan(predictions=[output])

    def test_runner_preserves_sampling_evidence_and_request_adapter(self):
        result, _ = self.run_scan(runner=True)
        self.assertEqual(result.detector_adapter, "runner")
        self.assertEqual(result.version, 2)
        self.assertEqual(len(result.sampling.samples), 3)

    def test_scan_cli_exports_v2_ledger_and_rechecks_media_after_inference(self):
        result, _ = self.run_scan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, weight = _write_fixture(root)
            source = root / "fixture.mp4"
            source.write_bytes(b"test-only-video-stub")
            output = root / "scan.json"
            args = SimpleNamespace(
                source_admission="mock-admission",
                source=source,
                runner_executable=sys.executable,
                manifest=manifest,
                weights=weight,
                label=["person"],
                confidence=0.35,
                sample_fps=1.0,
                device="cpu",
                image_size=640,
                rotation_degrees=0,
                timeout_seconds=10,
                camera_id="synthetic_cam",
                output=output,
            )
            admission = Mock(asset_kind="media_asset", fingerprint="b" * 64)
            detector = Mock()
            detector.scan_result.return_value = result
            stdout = io.StringIO()
            with (
                patch(
                    "baby_monitor.cli_commands.detection_eval.ExternalSourceAdmission.read",
                    return_value=admission,
                ),
                patch(
                    "baby_monitor.cli_commands.detection_eval.CommandYoloDetector",
                    return_value=detector,
                ),
                redirect_stdout(stdout),
            ):
                self.assertEqual(command_yolo_scan_file(args), 0)
            admission.verify_asset.assert_any_call(source)
            self.assertEqual(admission.verify_asset.call_count, 2)
            detector.scan.assert_not_called()
            saved = AdmittedDetectorScan.read(output)
            self.assertEqual(saved.version, 2)
            self.assertEqual(saved.result.sampling, result.sampling)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["sampled_frame_count"], 3)
            self.assertEqual(summary["zero_detection_sample_count"], 1)
            self.assertFalse(summary["source_exhaustion_verified"])
            self.assertNotIn(str(root), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
