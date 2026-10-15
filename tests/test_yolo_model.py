from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import DetectorError, UltralyticsYoloDetector
from baby_monitor.cli import main
from baby_monitor.domain import Detection
from baby_monitor.yolo_model import YoloModelError, YoloModelManifest


def _payload(content: bytes, *, filename: str = "fixture.pt") -> dict[str, object]:
    return {
        "version": 1,
        "backend": "ultralytics_yolo",
        "model_id": "fixture_detector",
        "revision": "fixture-v1",
        "filename": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "source_url": "https://example.invalid/models/fixture.pt",
        "license_spdx": "LicenseRef-Fixture-Only",
        "license_url": "https://example.invalid/licenses/fixture",
        "labels": ["person", "dog"],
        "subject_identity_inference": False,
    }


def _write_fixture(root: Path, content: bytes = b"fixture-weight") -> tuple[Path, Path]:
    weight = root / "fixture.pt"
    manifest = root / "fixture.manifest.json"
    weight.write_bytes(content)
    manifest.write_text(json.dumps(_payload(content)), encoding="utf-8")
    return manifest, weight


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

    def __init__(self, _uri: str):
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def get(self, _property: int) -> float:
        return 30.0 if _property == 5 else 0.0

    def read(self) -> tuple[bool, object | None]:
        self.read_count += 1
        if self.read_count == 1:
            return True, object()
        return False, None

    def release(self) -> None:
        type(self).released = True


class _FakeYolo:
    last_path = ""
    last_kwargs: dict[str, object] = {}
    names: object = {0: "person", 1: "dog"}

    def __init__(self, path: str):
        type(self).last_path = path
        self.names = type(self).names

    def predict(self, **kwargs: object) -> list[object]:
        if set(kwargs) != {"source", "conf", "device", "imgsz", "verbose"}:
            raise AssertionError("unexpected predict arguments")
        type(self).last_kwargs = kwargs
        boxes = SimpleNamespace(
            cls=_FakeTensor([0.0, 1.0]),
            conf=_FakeTensor([0.81, 0.92]),
            xyxyn=_FakeTensor(
                [
                    [0.1, 0.2, 0.4, 0.8],
                    [0.5, 0.5, 0.9, 0.9],
                ]
            ),
            # Deliberately incompatible pixels: the adapter must never read this.
            xyxy=_FakeTensor([[10, 20, 40, 80], [50, 50, 90, 90]]),
        )
        return [SimpleNamespace(boxes=boxes)]


class _VariableRateCapture:
    released = False

    def __init__(self, _uri: str):
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == 5:
            return 5.0
        if property_id == 0:
            return max(0, self.read_count - 1) * 100.0
        return 0.0

    def read(self) -> tuple[bool, object | None]:
        self.read_count += 1
        if self.read_count <= 6:
            return True, object()
        return False, None

    def release(self) -> None:
        type(self).released = True


class _FakeCuda:
    cache_cleared = False

    @staticmethod
    def is_available() -> bool:
        return True

    @classmethod
    def empty_cache(cls) -> None:
        cls.cache_cleared = True


class YoloModelTests(unittest.TestCase):
    def test_vision_extra_pins_reviewed_runtime_without_overlapping_cv2_wheels(
        self,
    ) -> None:
        with Path("pyproject.toml").open("rb") as handle:
            metadata = tomllib.load(handle)
        vision = metadata["project"]["optional-dependencies"]["vision"]

        self.assertEqual(["opencv-python==4.12.0.88", "ultralytics==8.4.115"], vision)
        self.assertFalse(any("opencv-python-headless" in value for value in vision))

    def test_public_schema_and_example_match_runtime_contract(self) -> None:
        schema = json.loads(
            Path("schemas/yolo-model-manifest-v1.schema.json").read_text(encoding="utf-8")
        )
        example = json.loads(Path("config/yolo-model.example.json").read_text(encoding="utf-8"))
        manifest = YoloModelManifest.from_dict(example)
        self.assertEqual(set(schema["required"]), set(example))
        self.assertEqual("ultralytics_yolo", manifest.backend)
        self.assertFalse(manifest.subject_identity_inference)

    def test_reviewed_yolo11n_candidate_pins_weight_and_full_coco_map(self) -> None:
        manifest = YoloModelManifest.read("config/yolo11n-coco.reviewed.example.json")

        self.assertEqual("Ultralytics.YOLO11n.COCO", manifest.model_id)
        self.assertEqual(
            "5599de6f3b0ef95f98f37530ed0f3bf149c10b37",
            manifest.revision,
        )
        self.assertEqual(5_613_764, manifest.size_bytes)
        self.assertEqual(
            "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
            manifest.sha256,
        )
        self.assertEqual(80, len(manifest.labels))
        self.assertEqual("person", manifest.labels[0])
        self.assertEqual("sports ball", manifest.labels[32])
        self.assertEqual("teddy bear", manifest.labels[77])
        self.assertEqual("toothbrush", manifest.labels[-1])
        self.assertEqual("AGPL-3.0", manifest.license_spdx)
        self.assertFalse(manifest.subject_identity_inference)

    def test_manifest_verifies_weight_and_public_output_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            verification = YoloModelManifest.read(manifest_path).verify_weight(weight_path)
            public = verification.to_public_dict()

            self.assertTrue(public["valid"])
            self.assertFalse(public["paths_disclosed"])
            self.assertFalse(public["model_loaded"])
            self.assertFalse(public["network_used"])
            self.assertNotIn(str(root), json.dumps(public))

    def test_manifest_rejects_tampered_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            weight_path.write_bytes(b"tampered-byte!")
            with self.assertRaisesRegex(YoloModelError, "sha256 does not match"):
                YoloModelManifest.read(manifest_path).verify_weight(weight_path)

    def test_manifest_rejects_credential_bearing_or_unpinned_semantics(self) -> None:
        payload = _payload(b"fixture")
        # release-audit: allow=fixture-credential
        payload["source_url"] = "https://user:secret@example.invalid/model.pt"
        with self.assertRaisesRegex(YoloModelError, "credential-free"):
            YoloModelManifest.from_dict(payload)

        payload = _payload(b"fixture")
        payload["subject_identity_inference"] = True
        with self.assertRaisesRegex(YoloModelError, "cannot authorize"):
            YoloModelManifest.from_dict(payload)

    @unittest.skipUnless(os.name == "posix", "symlink policy is verified on POSIX/WSL")
    def test_manifest_rejects_symlink_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            link = root / "linked.pt"
            link.symlink_to(weight_path)
            payload = _payload(weight_path.read_bytes(), filename=link.name)
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(YoloModelError, "missing or unsafe"):
                YoloModelManifest.read(manifest_path).verify_weight(link)

    def test_detector_uses_normalized_boxes_and_preserves_generic_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            video = root / "input.mp4"
            video.write_bytes(b"fixture-video")
            manifest = YoloModelManifest.read(manifest_path)
            detector = UltralyticsYoloDetector(
                model_path=weight_path,
                manifest=manifest,
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="0",
            )
            _FakeCapture.released = False
            _FakeCuda.cache_cleared = False
            cv2_module = SimpleNamespace(VideoCapture=_FakeCapture, CAP_PROP_FPS=5)
            ultralytics_module = SimpleNamespace(YOLO=_FakeYolo)
            torch_module = SimpleNamespace(cuda=_FakeCuda)

            with patch.dict(
                sys.modules,
                {
                    "cv2": cv2_module,
                    "torch": torch_module,
                    "ultralytics": ultralytics_module,
                },
            ):
                detections = detector.scan(LocalFileCamera("cam_01", video).source())

            self.assertEqual(1, len(detections))
            self.assertEqual("person", detections[0].label)
            self.assertEqual((0.1, 0.2, 0.4, 0.8), detections[0].bbox_xyxy)
            self.assertTrue(_FakeCapture.released)
            self.assertTrue(_FakeCuda.cache_cleared)
            self.assertEqual(str(weight_path), _FakeYolo.last_path)
            self.assertEqual(640, _FakeYolo.last_kwargs["imgsz"])

    def test_detector_rotates_for_inference_and_maps_boxes_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            video = root / "input.mp4"
            video.write_bytes(b"fixture-video")
            detector = UltralyticsYoloDetector(
                model_path=weight_path,
                manifest=YoloModelManifest.read(manifest_path),
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
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
                detections = detector.scan(LocalFileCamera("cam_01", video).source())

            self.assertEqual(1, len(rotate_calls))
            self.assertIs(rotated_frame, _FakeYolo.last_kwargs["source"])
            self.assertEqual(960, _FakeYolo.last_kwargs["imgsz"])
            self.assertEqual((0.2, 0.6, 0.8, 0.9), detections[0].bbox_xyxy)

    def test_detector_rejects_manifest_class_map_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            video = root / "input.mp4"
            video.write_bytes(b"fixture-video")
            detector = UltralyticsYoloDetector(
                model_path=weight_path,
                manifest=YoloModelManifest.read(manifest_path),
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="cpu",
            )
            _FakeCapture.released = False
            cv2_module = SimpleNamespace(VideoCapture=_FakeCapture, CAP_PROP_FPS=5)
            with (
                patch.object(_FakeYolo, "names", {0: "adult", 1: "dog"}),
                patch.dict(
                    sys.modules,
                    {"cv2": cv2_module, "ultralytics": SimpleNamespace(YOLO=_FakeYolo)},
                ),
                self.assertRaisesRegex(DetectorError, "class map"),
            ):
                detector.scan(LocalFileCamera("cam_01", video).source())
            self.assertTrue(_FakeCapture.released)

    def test_detector_uses_container_pts_instead_of_nominal_fps_for_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            video = root / "input.mp4"
            video.write_bytes(b"fixture-video")
            detector = UltralyticsYoloDetector(
                model_path=weight_path,
                manifest=YoloModelManifest.read(manifest_path),
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="cpu",
            )
            cv2_module = SimpleNamespace(
                VideoCapture=_VariableRateCapture,
                CAP_PROP_FPS=5,
                CAP_PROP_POS_MSEC=0,
            )
            with patch.dict(
                sys.modules,
                {"cv2": cv2_module, "ultralytics": SimpleNamespace(YOLO=_FakeYolo)},
            ):
                detections = detector.scan(LocalFileCamera("cam_01", video).source())

            self.assertEqual([0.0, 0.5], [item.timestamp_seconds for item in detections])

    def test_detection_contract_rejects_pixel_or_degenerate_boxes(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized"):
            Detection(0.0, "person", 0.8, (0.0, 0.0, 640.0, 480.0))
        with self.assertRaisesRegex(ValueError, "positive normalized area"):
            Detection(0.0, "person", 0.8, (0.5, 0.5, 0.5, 0.9))

    def test_verify_cli_does_not_load_model_or_disclose_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, weight_path = _write_fixture(root)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "verify-yolo-model",
                        "--manifest",
                        str(manifest_path),
                        "--weights",
                        str(weight_path),
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(0, exit_code)
            self.assertFalse(payload["model_loaded"])
            self.assertFalse(payload["network_used"])
            self.assertNotIn(str(root), output.getvalue())


if __name__ == "__main__":
    unittest.main()
