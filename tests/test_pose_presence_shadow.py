from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.adapters.detector import Detector
from baby_monitor.adapters.pose import PoseEstimator
from baby_monitor.cli_commands.candidate_runtime import _SHADOW_DETECTOR_RUNNER_MODULE
from baby_monitor.domain import Detection
from baby_monitor.pose_presence_shadow import (
    PosePresenceShadowCameraRule,
    PosePresenceShadowClassifier,
    PosePresenceShadowError,
    PosePresenceShadowPolicy,
)
from baby_monitor.pose_protocol import (
    COCO_PERSON_KEYPOINTS,
    PoseFrame,
    PoseFrameBatch,
    PoseInstance,
    PoseKeypoint,
)
from baby_monitor.room_presence import StreamPresenceSampleState


class _Detector(Detector):
    runner_adapter = "ultralytics_yolo_subprocess_v1"

    def __init__(self, detections: tuple[Detection, ...]):
        self.detections = detections
        self.calls = 0
        self.manifest = SimpleNamespace(model_id="detector", revision="v1")
        self.model_sha256 = "d" * 64
        self.confidence = 0.1
        self.sample_fps = 0.5
        self.image_size = 640
        self.rotation_degrees = 0

    def scan(self, source: CameraSource) -> tuple[Detection, ...]:
        del source
        self.calls += 1
        return self.detections


class _PoseEstimator(PoseEstimator):
    runner_adapter = "ultralytics_yolo_pose_subprocess_v1"

    def __init__(self, batch: PoseFrameBatch):
        self.batch = batch
        self.calls = 0
        self.manifest = SimpleNamespace(model_id="pose", revision="v1")
        self.model_sha256 = "e" * 64
        self.confidence = 0.1
        self.sample_fps = 0.5
        self.max_frames = 128
        self.image_size = 640
        self.rotation_degrees = 0

    def scan(self, source: CameraSource) -> PoseFrameBatch:
        del source
        self.calls += 1
        return self.batch


def _keypoints(confidence: float = 0.9) -> tuple[PoseKeypoint, ...]:
    return tuple(
        PoseKeypoint(
            name=name,
            x=0.5,
            y=0.5,
            confidence=confidence,
            in_frame=True,
        )
        for name in COCO_PERSON_KEYPOINTS
    )


def _pose_batch(timestamp: float = 9.0) -> PoseFrameBatch:
    return PoseFrameBatch(
        camera_id="cam_01",
        pose_adapter="ultralytics_yolo_pose_subprocess_v1",
        model_id="pose",
        model_revision="v1",
        model_sha256="e" * 64,
        frames=(
            PoseFrame(
                frame_id="frame_01",
                timestamp_seconds=timestamp,
                poses=(
                    PoseInstance(
                        instance_id="pose_01",
                        confidence=0.8,
                        bbox_xyxy=(0.1, 0.1, 0.7, 0.9),
                        keypoints=_keypoints(),
                    ),
                ),
            ),
        ),
    )


def _policy(*, pose_enabled: bool = True) -> PosePresenceShadowPolicy:
    return PosePresenceShadowPolicy(
        profile_sha256="a" * 64,
        temporal_holdout_attestation_sha256="b" * 64,
        human_audit_report_sha256="c" * 64,
        detector_adapter="ultralytics_yolo_subprocess_v1",
        detector_model_id="detector",
        detector_model_revision="v1",
        detector_model_sha256="d" * 64,
        detector_source_confidence_threshold=0.1,
        detector_requested_sample_fps=0.5,
        pose_adapter="ultralytics_yolo_pose_subprocess_v1",
        pose_model_id="pose",
        pose_model_revision="v1",
        pose_model_sha256="e" * 64,
        pose_source_confidence_threshold=0.1,
        pose_requested_sample_fps=0.5,
        pose_requested_max_frames=128,
        image_size=640,
        rotation_degrees=0,
        bin_seconds=5.0,
        cameras=(
            PosePresenceShadowCameraRule(
                camera_id="cam_01",
                detector_confidence_threshold=0.5,
                detector_hold_seconds=0.0,
                pose_enabled=pose_enabled,
                pose_confidence_threshold=0.5,
                pose_hold_seconds=0.0,
                minimum_pose_bbox_area=0.1,
                minimum_pose_bbox_height=0.5,
                keypoint_confidence_threshold=0.5,
                minimum_keypoint_count=8,
            ),
        ),
    )


class PosePresenceShadowTests(unittest.TestCase):
    def test_shadow_detector_runner_module_is_importable(self) -> None:
        self.assertEqual(
            "baby_monitor.local_runners.yolo_detector",
            _SHADOW_DETECTOR_RUNNER_MODULE,
        )
        self.assertIsNotNone(importlib.util.find_spec(_SHADOW_DETECTOR_RUNNER_MODULE))

    SOURCE = CameraSource(
        camera_id="cam_01",
        kind="file",
        read_uri="/tmp/fixture-segment.mkv",
        audit_ref="/tmp/fixture-segment.mkv",
    )

    def test_recent_detector_hit_short_circuits_pose(self) -> None:
        detector = _Detector((Detection(9.0, "person", 0.8, (0.1, 0.1, 0.8, 0.9)),))
        pose = _PoseEstimator(_pose_batch())
        decision = PosePresenceShadowClassifier(_policy(), detector, pose).classify(
            self.SOURCE,
            duration_seconds=10.0,
        )
        self.assertIs(StreamPresenceSampleState.PERSON_DETECTED, decision.state)
        self.assertTrue(decision.detector_hit)
        self.assertFalse(decision.pose_executed)
        self.assertEqual(1, detector.calls)
        self.assertEqual(0, pose.calls)

    def test_pose_recovers_current_bin_when_detector_is_clear(self) -> None:
        detector = _Detector(())
        pose = _PoseEstimator(_pose_batch())
        decision = PosePresenceShadowClassifier(_policy(), detector, pose).classify(
            self.SOURCE,
            duration_seconds=10.0,
        )
        self.assertIs(StreamPresenceSampleState.PERSON_DETECTED, decision.state)
        self.assertFalse(decision.detector_hit)
        self.assertTrue(decision.pose_hit)
        self.assertTrue(decision.pose_executed)

    def test_old_detector_and_pose_hits_do_not_leak_into_latest_bin(self) -> None:
        detector = _Detector((Detection(1.0, "person", 0.8, (0.1, 0.1, 0.8, 0.9)),))
        pose = _PoseEstimator(_pose_batch(timestamp=1.0))
        decision = PosePresenceShadowClassifier(_policy(), detector, pose).classify(
            self.SOURCE,
            duration_seconds=10.0,
        )
        self.assertIs(StreamPresenceSampleState.CLEAR, decision.state)
        self.assertTrue(decision.pose_executed)

    def test_pose_disabled_camera_never_loads_pose(self) -> None:
        detector = _Detector(())
        pose = _PoseEstimator(_pose_batch())
        decision = PosePresenceShadowClassifier(
            _policy(pose_enabled=False),
            detector,
            pose,
        ).classify(self.SOURCE, duration_seconds=10.0)
        self.assertIs(StreamPresenceSampleState.CLEAR, decision.state)
        self.assertFalse(decision.pose_executed)
        self.assertEqual(0, pose.calls)

    def test_runtime_model_or_transform_drift_is_rejected_before_scan(self) -> None:
        detector = _Detector(())
        detector.image_size = 960
        with self.assertRaisesRegex(PosePresenceShadowError, "does not match evidence"):
            PosePresenceShadowClassifier(_policy(), detector, _PoseEstimator(_pose_batch()))


if __name__ == "__main__":
    unittest.main()
