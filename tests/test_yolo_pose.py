from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.adapters.pose import CommandYoloPoseEstimator, PoseEstimatorError
from baby_monitor.cli import main
from baby_monitor.local_runners.yolo_pose import main as pose_runner_main
from baby_monitor.local_runners.yolo_pose import run_request
from baby_monitor.pose_protocol import (
    COCO_PERSON_KEYPOINTS,
    AdmittedPoseScan,
    PoseFrame,
    PoseFrameBatch,
    PoseInstance,
    PoseKeypoint,
    PoseProtocolError,
    PoseScanRequest,
)
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.yolo_pose_model import YoloPoseModelError, YoloPoseModelManifest


def _manifest_payload(content: bytes) -> dict[str, object]:
    return {
        "version": 1,
        "backend": "ultralytics_yolo_pose",
        "model_id": "fixture_pose",
        "revision": "fixture-v1",
        "filename": "fixture-pose.pt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "source_url": "https://example.invalid/models/fixture-pose.pt",
        "license_spdx": "LicenseRef-Fixture-Only",
        "license_url": "https://example.invalid/licenses/fixture",
        "class_name": "person",
        "keypoint_names": list(COCO_PERSON_KEYPOINTS),
        "keypoint_dimensions": 3,
        "pickle_checkpoint": True,
        "subject_identity_inference": False,
        "posture_classification": False,
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    content = b"fixture-pose-weight"
    weight = root / "fixture-pose.pt"
    weight.write_bytes(content)
    manifest = root / "fixture-pose.manifest.json"
    manifest.write_text(json.dumps(_manifest_payload(content)), encoding="utf-8")
    video = root / "public-fixture.mp4"
    video.write_bytes(b"synthetic-public-video")
    return manifest, weight, video


def _keypoints() -> tuple[PoseKeypoint, ...]:
    return tuple(
        PoseKeypoint(name=name, x=0.5, y=0.5, confidence=0.8, in_frame=True)
        for name in COCO_PERSON_KEYPOINTS
    )


def _request_fixture(root: Path) -> PoseScanRequest:
    manifest_path, weight_path, video_path = _write_fixture(root)
    manifest = YoloPoseModelManifest.read(manifest_path)
    return PoseScanRequest(
        camera_id="public_cam",
        pose_adapter="ultralytics_yolo_pose_subprocess_v1",
        source_path=str(video_path.resolve()),
        model_path=str(weight_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        model_id=manifest.model_id,
        model_revision=manifest.revision,
        model_sha256=manifest.sha256,
        confidence=0.35,
        sample_fps=1.0,
        max_frames=10,
        device="cpu",
    )


def _batch() -> PoseFrameBatch:
    return PoseFrameBatch(
        camera_id="public_cam",
        pose_adapter="fixture_pose_v1",
        model_id="fixture_pose",
        model_revision="fixture-v1",
        model_sha256="a" * 64,
        frames=(
            PoseFrame(
                frame_id="f000001",
                timestamp_seconds=0.0,
                poses=(
                    PoseInstance(
                        instance_id="p0001",
                        confidence=0.9,
                        bbox_xyxy=(0.1, 0.1, 0.9, 0.9),
                        keypoints=_keypoints(),
                    ),
                ),
            ),
        ),
    )


def _media_admission(content: bytes) -> ExternalSourceAdmission:
    payload = json.loads(
        Path("tests/fixtures/valid_external_source_admission.json").read_text(encoding="utf-8")
    )
    payload["source_id"] = "public_pose_media_v1"
    payload["asset_kind"] = "media_asset"
    payload["access"]["expected_bytes"] = len(content)
    payload["content"] = {
        "observed_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return ExternalSourceAdmission.from_dict(payload)


class _FakeTensor:
    def __init__(self, value: object):
        self.value = value

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> object:
        return self.value


class _FakeCapture:
    released = False

    def __init__(self, _path: str):
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def get(self, _property: int) -> float:
        return 30.0

    def read(self) -> tuple[bool, object | None]:
        self.read_count += 1
        if self.read_count == 1:
            return True, object()
        return False, None

    def release(self) -> None:
        type(self).released = True


class _FakeYolo:
    task = "pose"
    names = {0: "person"}
    last_kwargs: dict[str, object] = {}

    def __init__(self, _path: str):
        self.task = type(self).task
        self.names = type(self).names
        self.model = SimpleNamespace(kpt_shape=[17, 3])

    def predict(self, **kwargs: object) -> list[object]:
        if set(kwargs) != {"source", "conf", "device", "imgsz", "verbose"}:
            raise AssertionError("unexpected pose prediction arguments")
        type(self).last_kwargs = kwargs
        coordinates = [[0.5, 0.5] for _ in COCO_PERSON_KEYPOINTS]
        coordinates[0] = [1.2, 0.1]
        boxes = SimpleNamespace(
            conf=_FakeTensor([0.91]),
            xyxyn=_FakeTensor([[0.1, 0.2, 0.8, 0.9]]),
            xyxy=_FakeTensor([[64.0, 96.0, 512.0, 432.0]]),
        )
        keypoints = SimpleNamespace(
            xyn=_FakeTensor([coordinates]),
            conf=_FakeTensor([[0.7 for _ in COCO_PERSON_KEYPOINTS]]),
            xy=_FakeTensor([[[320.0, 240.0] for _ in COCO_PERSON_KEYPOINTS]]),
        )
        return [SimpleNamespace(boxes=boxes, keypoints=keypoints)]


class _VariableRateCapture(_FakeCapture):
    position_msec: object = 500.0

    def get(self, property_id: int) -> object:
        if property_id == 5:
            return 5.0
        return type(self).position_msec if self.read_count > 1 else 0.0

    def read(self) -> tuple[bool, object | None]:
        self.read_count += 1
        return (True, object()) if self.read_count <= 6 else (False, None)


class _FakeCuda:
    cache_cleared = False

    @staticmethod
    def is_available() -> bool:
        return True

    @classmethod
    def empty_cache(cls) -> None:
        cls.cache_cleared = True


class YoloPoseTests(unittest.TestCase):
    def test_pose_runner_reserves_stdout_for_exact_json_on_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request_fixture(root)
            request_path = root / "request.json"
            request_path.write_text(request.to_json(), encoding="utf-8")
            for fail in (False, True):
                with self.subTest(fail=fail):
                    result = _batch()

                    def noisy_runtime(
                        _request: PoseScanRequest,
                        *,
                        fail: bool = fail,
                        result: PoseFrameBatch = result,
                    ) -> PoseFrameBatch:
                        print("third_party_pose_banner")
                        if fail:
                            raise RuntimeError("synthetic runtime failure detail")
                        return result

                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        patch(
                            "baby_monitor.local_runners.yolo_pose.run_request",
                            side_effect=noisy_runtime,
                        ),
                        redirect_stdout(stdout),
                        redirect_stderr(stderr),
                    ):
                        code = pose_runner_main([str(request_path)])
                    self.assertEqual(2 if fail else 0, code)
                    if fail:
                        self.assertEqual("", stdout.getvalue())
                        self.assertIn("pose_runner_failed", stderr.getvalue())
                        self.assertNotIn("synthetic runtime failure detail", stderr.getvalue())
                    else:
                        self.assertEqual(result, PoseFrameBatch.from_json(stdout.getvalue()))
                    self.assertNotIn("third_party_pose_banner", stdout.getvalue())
                    self.assertIn("third_party_pose_banner", stderr.getvalue())

    def test_pose_runner_uses_container_pts_with_nominal_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request_fixture(Path(directory))
            for position_msec, expected_last in (
                (500.0, 0.5),
                (0.0, 1.0),
                (-1.0, 1.0),
                (float("nan"), 1.0),
                (float("inf"), 1.0),
                ("invalid", 1.0),
            ):
                with self.subTest(position_msec=position_msec):
                    _VariableRateCapture.released = False
                    cv2_module = SimpleNamespace(
                        VideoCapture=_VariableRateCapture,
                        CAP_PROP_FPS=5,
                        CAP_PROP_POS_MSEC=0,
                    )
                    with (
                        patch.object(_VariableRateCapture, "position_msec", position_msec),
                        patch.dict(
                            sys.modules,
                            {"cv2": cv2_module, "ultralytics": SimpleNamespace(YOLO=_FakeYolo)},
                        ),
                    ):
                        result = run_request(request)
                    self.assertEqual(
                        [0.0, expected_last], [frame.timestamp_seconds for frame in result.frames]
                    )
                    self.assertTrue(_VariableRateCapture.released)

    @unittest.skipUnless(os.name == "posix", "POSIX advisory lock required")
    def test_pose_execution_lock_is_private_and_rejects_unsafe_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            manifest_path, weight_path, _video_path = _write_fixture(root)
            estimator = CommandYoloPoseEstimator(
                model_path=weight_path,
                manifest_path=manifest_path,
                manifest=YoloPoseModelManifest.read(manifest_path),
                confidence=0.1,
                sample_fps=0.5,
                max_frames=10,
                device="cpu",
                command=(sys.executable, "{request}"),
                execution_lock_path=root / "shared-gpu.lock",
                execution_lock_timeout_seconds=2.0,
            )

            with estimator._execution_gate():
                self.assertEqual(0o600, (root / "shared-gpu.lock").stat().st_mode & 0o777)
                self.assertIsNotNone(estimator.last_execution_lock_wait_seconds)

            root.chmod(0o755)
            with (
                self.assertRaisesRegex(PoseEstimatorError, "execution lock is unsafe"),
                estimator._execution_gate(),
            ):
                self.fail("unsafe execution lock must not be acquired")

    def test_reviewed_manifest_pins_official_pose_weight_and_coco_contract(self) -> None:
        manifest = YoloPoseModelManifest.read("config/yolo11n-pose-coco.reviewed.example.json")
        schema = json.loads(
            Path("schemas/yolo-pose-model-manifest-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(manifest.to_dict()))
        self.assertEqual("Ultralytics.YOLO11n.Pose.COCO", manifest.model_id)
        self.assertEqual("ef787449d1fe015a77a27ca4d375ed0fde69495c", manifest.revision)
        self.assertEqual(6_255_593, manifest.size_bytes)
        self.assertEqual(
            "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
            manifest.sha256,
        )
        self.assertEqual(COCO_PERSON_KEYPOINTS, manifest.keypoint_names)
        self.assertTrue(manifest.pickle_checkpoint)
        self.assertFalse(manifest.subject_identity_inference)
        self.assertFalse(manifest.posture_classification)

    def test_manifest_verification_is_path_free_and_does_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path, _ = _write_fixture(root)
            verification = YoloPoseModelManifest.read(manifest_path).verify_weight(weight_path)
            public = verification.to_public_dict()

            self.assertTrue(public["valid"])
            self.assertFalse(public["paths_disclosed"])
            self.assertFalse(public["model_loaded"])
            self.assertFalse(public["network_used"])
            self.assertNotIn(str(root), json.dumps(public))

    def test_manifest_rejects_identity_posture_and_tampered_weights(self) -> None:
        payload = _manifest_payload(b"fixture")
        payload["subject_identity_inference"] = True
        with self.assertRaisesRegex(YoloPoseModelError, "must be false"):
            YoloPoseModelManifest.from_dict(payload)

        payload = _manifest_payload(b"fixture")
        payload["posture_classification"] = True
        with self.assertRaisesRegex(YoloPoseModelError, "must be false"):
            YoloPoseModelManifest.from_dict(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path, _ = _write_fixture(root)
            tampered = bytearray(weight_path.read_bytes())
            tampered[0] ^= 0x01
            weight_path.write_bytes(tampered)
            with self.assertRaisesRegex(YoloPoseModelError, "sha256 does not match"):
                YoloPoseModelManifest.read(manifest_path).verify_weight(weight_path)

    def test_pose_batch_is_strict_path_free_and_preserves_ordered_coco_17(self) -> None:
        batch = _batch()
        round_trip = PoseFrameBatch.from_json(batch.to_json())
        self.assertEqual(batch, round_trip)
        self.assertNotIn("source_path", batch.to_json())
        self.assertNotIn("model_path", batch.to_json())
        self.assertEqual(17, len(round_trip.frames[0].poses[0].keypoints))

        payload = batch.to_dict()
        payload["subject_identity_inference"] = True
        with self.assertRaisesRegex(PoseProtocolError, "identity inference"):
            PoseFrameBatch.from_dict(payload)

        reversed_points = tuple(reversed(_keypoints()))
        with self.assertRaisesRegex(PoseProtocolError, "ordered COCO-17"):
            PoseInstance(
                instance_id="p0001",
                confidence=0.9,
                bbox_xyxy=(0.1, 0.1, 0.9, 0.9),
                keypoints=reversed_points,
            )

    def test_admitted_pose_scan_binds_source_config_and_is_exclusive(self) -> None:
        scan = AdmittedPoseScan(
            source_admission_sha256="b" * 64,
            confidence_threshold=0.35,
            requested_sample_fps=1.0,
            requested_max_frames=10,
            image_size=640,
            rotation_degrees=0,
            result=_batch(),
        )
        schema = json.loads(
            Path("schemas/admitted-pose-scan-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(scan.to_dict()))
        self.assertEqual(scan, AdmittedPoseScan.from_json(scan.to_json()))
        self.assertEqual(64, len(scan.fingerprint))
        self.assertNotIn("source_path", scan.to_json())

        with self.assertRaisesRegex(PoseProtocolError, "conflicts with scan threshold"):
            replace(scan, confidence_threshold=0.95)
        with self.assertRaisesRegex(PoseProtocolError, "fields do not match"):
            AdmittedPoseScan.from_dict({**scan.to_dict(), "private_path": "/private"})
        with self.assertRaisesRegex(PoseProtocolError, "duplicate keys"):
            AdmittedPoseScan.from_json('{"version":1,"version":1}')

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "admitted-pose.json"
            scan.write_new(output)
            self.assertEqual(scan, AdmittedPoseScan.read(output))
            if os.name == "posix":
                self.assertEqual(0o600, output.stat().st_mode & 0o777)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                scan.write_new(output)

    def test_admitted_pose_cli_verifies_exact_asset_before_scanning(self) -> None:
        content = b"synthetic-public-video"
        admission = _media_admission(content)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "public-fixture.mp4"
            source.write_bytes(content)
            admission_path = root / "admission.json"
            admission_path.write_text(admission.to_json(), encoding="utf-8")
            output = root / "admitted-pose.json"

            stdout = io.StringIO()
            with (
                patch("baby_monitor.cli_commands.pose._scan_pose_batch", return_value=_batch()),
                redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "pose-scan-admitted-file",
                        "--source",
                        str(source),
                        "--source-admission",
                        str(admission_path),
                        "--camera-id",
                        "public_cam",
                        "--manifest",
                        str(root / "manifest.json"),
                        "--weights",
                        str(root / "weights.pt"),
                        "--runner-executable",
                        sys.executable,
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(0, code)
            saved = AdmittedPoseScan.read(output)
            self.assertEqual(admission.fingerprint, saved.source_admission_sha256)
            self.assertEqual(saved.fingerprint, json.loads(stdout.getvalue())["scan_sha256"])
            self.assertNotIn(str(root), stdout.getvalue())

            source.write_bytes(b"tampered")
            rejected_stderr = io.StringIO()
            with (
                patch("baby_monitor.cli_commands.pose._scan_pose_batch", return_value=_batch()),
                redirect_stderr(rejected_stderr),
            ):
                rejected_code = main(
                    [
                        "pose-scan-admitted-file",
                        "--source",
                        str(source),
                        "--source-admission",
                        str(admission_path),
                        "--camera-id",
                        "public_cam",
                        "--manifest",
                        str(root / "manifest.json"),
                        "--weights",
                        str(root / "weights.pt"),
                        "--runner-executable",
                        sys.executable,
                        "--output",
                        str(root / "never-created.json"),
                    ]
                )
            self.assertEqual(2, rejected_code)
            self.assertIn("does not match admission manifest", rejected_stderr.getvalue())
            self.assertFalse((root / "never-created.json").exists())

    def test_out_of_frame_keypoints_are_explicitly_null(self) -> None:
        with self.assertRaisesRegex(PoseProtocolError, "must be null"):
            PoseKeypoint(name="nose", x=1.2, y=0.1, confidence=0.8, in_frame=False)
        point = PoseKeypoint(name="nose", x=None, y=None, confidence=0.8, in_frame=False)
        self.assertIsNone(point.x)
        self.assertFalse(point.in_frame)

    def test_runner_uses_normalized_pose_tensors_and_releases_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path, video_path = _write_fixture(root)
            manifest = YoloPoseModelManifest.read(manifest_path)
            request = PoseScanRequest(
                camera_id="public_cam",
                pose_adapter="ultralytics_yolo_pose_subprocess_v1",
                source_path=str(video_path.resolve()),
                model_path=str(weight_path.resolve()),
                manifest_path=str(manifest_path.resolve()),
                model_id=manifest.model_id,
                model_revision=manifest.revision,
                model_sha256=manifest.sha256,
                confidence=0.35,
                sample_fps=1.0,
                max_frames=10,
                device="0",
            )
            self.assertEqual(2, request.version)
            legacy = request.to_dict()
            legacy["version"] = 1
            legacy.pop("image_size")
            legacy.pop("rotation_degrees")
            migrated = PoseScanRequest.from_dict(legacy)
            self.assertEqual(640, migrated.image_size)
            self.assertEqual(0, migrated.rotation_degrees)
            with self.assertRaisesRegex(PoseProtocolError, "image_size"):
                PoseScanRequest.from_dict({**request.to_dict(), "image_size": 641})
            with self.assertRaisesRegex(PoseProtocolError, "rotation_degrees"):
                PoseScanRequest.from_dict({**request.to_dict(), "rotation_degrees": 45})
            _FakeCapture.released = False
            _FakeCuda.cache_cleared = False
            with patch.dict(
                sys.modules,
                {
                    "cv2": SimpleNamespace(VideoCapture=_FakeCapture, CAP_PROP_FPS=5),
                    "torch": SimpleNamespace(cuda=_FakeCuda),
                    "ultralytics": SimpleNamespace(YOLO=_FakeYolo),
                },
            ):
                result = run_request(request)

            self.assertEqual(1, len(result.frames))
            pose = result.frames[0].poses[0]
            self.assertEqual((0.1, 0.2, 0.8, 0.9), pose.bbox_xyxy)
            self.assertFalse(pose.keypoints[0].in_frame)
            self.assertIsNone(pose.keypoints[0].x)
            self.assertEqual(0.7, pose.keypoints[0].confidence)
            self.assertTrue(_FakeCapture.released)
            self.assertTrue(_FakeCuda.cache_cleared)
            self.assertEqual(640, _FakeYolo.last_kwargs["imgsz"])
            self.assertFalse(result.subject_identity_inference)
            self.assertFalse(result.posture_classification)

    def test_pose_runner_rotates_and_maps_boxes_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path, video_path = _write_fixture(root)
            manifest = YoloPoseModelManifest.read(manifest_path)
            request = PoseScanRequest(
                camera_id="public_cam",
                pose_adapter="ultralytics_yolo_pose_subprocess_v1",
                source_path=str(video_path.resolve()),
                model_path=str(weight_path.resolve()),
                manifest_path=str(manifest_path.resolve()),
                model_id=manifest.model_id,
                model_revision=manifest.revision,
                model_sha256=manifest.sha256,
                confidence=0.35,
                sample_fps=1.0,
                max_frames=10,
                device="0",
                image_size=960,
                rotation_degrees=90,
            )
            rotated_frame = object()
            rotate_calls: list[tuple[object, int]] = []
            cv2_module = SimpleNamespace(
                VideoCapture=_FakeCapture,
                CAP_PROP_FPS=5,
                ROTATE_90_CLOCKWISE=7,
                rotate=lambda frame, constant: (
                    rotate_calls.append((frame, constant)) or rotated_frame
                ),
            )
            with patch.dict(
                sys.modules,
                {
                    "cv2": cv2_module,
                    "torch": SimpleNamespace(cuda=_FakeCuda),
                    "ultralytics": SimpleNamespace(YOLO=_FakeYolo),
                },
            ):
                result = run_request(request)

            self.assertEqual(1, len(rotate_calls))
            self.assertIs(rotated_frame, _FakeYolo.last_kwargs["source"])
            self.assertEqual(960, _FakeYolo.last_kwargs["imgsz"])
            self.assertTrue(
                all(
                    abs(actual - expected) < 1e-12
                    for actual, expected in zip(
                        result.frames[0].poses[0].bbox_xyxy,
                        (0.2, 0.2, 0.9, 0.9),
                        strict=True,
                    )
                )
            )

    def test_verify_and_validate_cli_outputs_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path, _ = _write_fixture(root)
            pose_path = root / "pose.json"
            pose_path.write_text(_batch().to_json(), encoding="utf-8")
            admitted_path = root / "admitted-pose.json"
            admitted = AdmittedPoseScan(
                source_admission_sha256="b" * 64,
                confidence_threshold=0.35,
                requested_sample_fps=1.0,
                requested_max_frames=10,
                image_size=640,
                rotation_degrees=0,
                result=_batch(),
            )
            admitted_path.write_text(admitted.to_json(), encoding="utf-8")

            verify_output = io.StringIO()
            with redirect_stdout(verify_output):
                verify_code = main(
                    [
                        "verify-yolo-pose-model",
                        "--manifest",
                        str(manifest_path),
                        "--weights",
                        str(weight_path),
                    ]
                )
            verify = json.loads(verify_output.getvalue())
            self.assertEqual(0, verify_code)
            self.assertFalse(verify["paths_disclosed"])
            self.assertNotIn(str(root), verify_output.getvalue())

            validate_output = io.StringIO()
            with redirect_stdout(validate_output):
                validate_code = main(["validate-pose-frames", str(pose_path)])
            validated = json.loads(validate_output.getvalue())
            self.assertEqual(0, validate_code)
            self.assertEqual(17, validated["keypoints"])
            self.assertFalse(validated["posture_classification"])
            self.assertNotIn(str(root), validate_output.getvalue())

            admitted_output = io.StringIO()
            with redirect_stdout(admitted_output):
                admitted_code = main(["validate-admitted-pose-scan", str(admitted_path)])
            admitted_validated = json.loads(admitted_output.getvalue())
            self.assertEqual(0, admitted_code)
            self.assertEqual(admitted.fingerprint, admitted_validated["scan_sha256"])
            self.assertEqual("b" * 64, admitted_validated["source_admission_sha256"])
            self.assertNotIn(str(root), admitted_output.getvalue())


if __name__ == "__main__":
    unittest.main()
