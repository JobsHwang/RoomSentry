from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import DetectorError, UltralyticsYoloDetector
from baby_monitor.local_runners.yolo_pose import run_request
from baby_monitor.pose_protocol import PoseProtocolError
from baby_monitor.yolo_model import YoloModelManifest
from tests.test_yolo_model import _FakeYolo as DetectorModel
from tests.test_yolo_model import _write_fixture as detector_fixture
from tests.test_yolo_pose import _FakeYolo as PoseModel
from tests.test_yolo_pose import _request_fixture as pose_fixture


class YoloDecodeReliabilityTests(unittest.TestCase):
    @contextmanager
    def _scanner(self, backend, *, opened=True, reads=None, fps=25.0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = Mock()
            capture.isOpened.return_value = opened
            capture.get.return_value = fps
            capture.read.side_effect = reads if reads is not None else [(False, None)]
            if backend == "detector":
                manifest_path, weight_path = detector_fixture(root)
                source = root / "fixture.mp4"
                source.write_bytes(b"synthetic-fixture")
                detector = UltralyticsYoloDetector(
                    model_path=weight_path,
                    manifest=YoloModelManifest.read(manifest_path),
                    labels=("person",),
                    confidence=0.35,
                    sample_fps=1.0,
                    device="cpu",
                )

                def invoke():
                    return detector.scan(LocalFileCamera("public_cam", source).source())

                model = DetectorModel
                cleanup_target = (
                    "baby_monitor.adapters.detector."
                    "UltralyticsYoloDetector._release_accelerator_cache"
                )
                error = DetectorError
            else:
                request = pose_fixture(root)

                def invoke():
                    return run_request(request)

                model = PoseModel
                cleanup_target = "baby_monitor.local_runners.yolo_pose._release_accelerator_cache"
                error = PoseProtocolError
            with (
                patch.dict(
                    sys.modules,
                    {
                        "cv2": SimpleNamespace(
                            VideoCapture=Mock(return_value=capture), CAP_PROP_FPS=5
                        ),
                        "ultralytics": SimpleNamespace(YOLO=model),
                    },
                ),
                patch(cleanup_target) as cleanup,
            ):
                yield invoke, capture, cleanup, error, model

    def test_unopened_capture_is_released(self):
        for backend in ("detector", "pose"):
            with self.subTest(backend=backend), self._scanner(backend, opened=False) as case:
                invoke, capture, cleanup, error, _ = case
                with self.assertRaisesRegex(error, "open the local video"):
                    invoke()
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()

    def test_metadata_failure_releases_capture(self):
        for backend in ("detector", "pose"):
            with self.subTest(backend=backend), self._scanner(backend) as case:
                invoke, capture, cleanup, _, _ = case
                capture.get.side_effect = RuntimeError("fixture_metadata_failed")
                with self.assertRaisesRegex(RuntimeError, "fixture_metadata_failed"):
                    invoke()
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()

    def test_zero_decoded_frames_cannot_be_successful_empty_detections(self):
        for backend in ("detector", "pose"):
            with self.subTest(backend=backend), self._scanner(backend) as case:
                invoke, capture, cleanup, error, _ = case
                with self.assertRaisesRegex(error, "frames"):
                    invoke()
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()

    def test_success_flag_with_missing_or_empty_frame_is_rejected_before_inference(self):
        for backend in ("detector", "pose"):
            for frame in (None, SimpleNamespace(size=0)):
                with (
                    self.subTest(backend=backend, frame=frame),
                    self._scanner(backend, reads=[(True, frame), (False, None)]) as case,
                ):
                    invoke, capture, cleanup, error, model = case
                    with patch.object(model, "predict") as predict:
                        with self.assertRaisesRegex(error, "frame"):
                            invoke()
                        predict.assert_not_called()
                    capture.release.assert_called_once_with()
                    cleanup.assert_called_once()

    def test_decode_exception_does_not_publish_partial_success(self):
        for backend in ("detector", "pose"):
            with (
                self.subTest(backend=backend),
                self._scanner(
                    backend,
                    reads=[(True, object()), RuntimeError("fixture_decode_failed")],
                ) as case,
            ):
                invoke, capture, cleanup, _, _ = case
                with self.assertRaisesRegex(RuntimeError, "fixture_decode_failed"):
                    invoke()
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()

    def test_inference_exception_releases_capture_and_model_cache(self):
        for backend in ("detector", "pose"):
            with (
                self.subTest(backend=backend),
                self._scanner(backend, reads=[(True, object())]) as case,
            ):
                invoke, capture, cleanup, _, model = case
                with (
                    patch.object(model, "predict", side_effect=RuntimeError("fixture_inference")),
                    self.assertRaisesRegex(RuntimeError, "fixture_inference"),
                ):
                    invoke()
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()

    def test_empty_unsampled_frame_is_also_a_decode_failure(self):
        for backend in ("detector", "pose"):
            with (
                self.subTest(backend=backend),
                self._scanner(
                    backend, reads=[(True, object()), (True, None), (False, None)]
                ) as case,
            ):
                invoke, capture, cleanup, error, _ = case
                with self.assertRaisesRegex(error, "empty frame"):
                    invoke()
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()

    def test_nonfinite_or_nonpositive_fps_uses_finite_nominal_fallback(self):
        for backend in ("detector", "pose"):
            for fps in (float("nan"), float("inf"), 0.0, -1.0):
                with (
                    self.subTest(backend=backend, fps=fps),
                    self._scanner(
                        backend,
                        reads=[(True, object()) for _ in range(26)] + [(False, None)],
                        fps=fps,
                    ) as case,
                ):
                    invoke, capture, cleanup, _, _ = case
                    result = invoke()
                    samples = result if backend == "detector" else result.frames
                    self.assertEqual([0.0, 1.0], [s.timestamp_seconds for s in samples])
                    capture.release.assert_called_once_with()
                    cleanup.assert_called_once()

    def test_valid_decoded_frame_without_people_remains_valid(self):
        for backend in ("detector", "pose"):
            with (
                self.subTest(backend=backend),
                self._scanner(backend, reads=[(True, object()), (False, None)]) as case,
            ):
                invoke, capture, cleanup, _, model = case
                with patch.object(
                    model, "predict", return_value=[SimpleNamespace(boxes=None, keypoints=None)]
                ):
                    result = invoke()
                if backend == "detector":
                    self.assertEqual([], result)
                else:
                    self.assertEqual(1, len(result.frames))
                    self.assertEqual((), result.frames[0].poses)
                capture.release.assert_called_once_with()
                cleanup.assert_called_once()
