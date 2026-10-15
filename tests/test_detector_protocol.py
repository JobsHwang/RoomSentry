from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import main
from baby_monitor.detector_protocol import (
    AdmittedDetectorScan,
    DetectorProtocolError,
    DetectorScanRequest,
    DetectorScanResult,
)
from baby_monitor.domain import Detection
from baby_monitor.local_runners.yolo_detector import main as yolo_runner_main


class DetectorProtocolTests(unittest.TestCase):
    def test_yolo_runner_reserves_stdout_for_exact_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            weight = root / "fixture.pt"
            manifest = root / "manifest.json"
            for path in (source, weight, manifest):
                path.write_text("fixture", encoding="utf-8")
            request = DetectorScanRequest(
                camera_id="cam_01",
                detector_adapter="ultralytics_yolo_subprocess_v1",
                source_path=str(source),
                model_path=str(weight),
                manifest_path=str(manifest),
                model_id="fixture_yolo",
                model_revision="revision_01",
                model_sha256="a" * 64,
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="0",
            )
            request_path = root / "request.json"
            request_path.write_text(request.to_json(), encoding="utf-8")
            result = DetectorScanResult(
                camera_id=request.camera_id,
                detector_adapter=request.detector_adapter,
                model_id=request.model_id,
                model_revision=request.model_revision,
                model_sha256=request.model_sha256,
                detections=(),
            )

            def noisy_model_runtime(_request: DetectorScanRequest) -> DetectorScanResult:
                print("third_party_model_banner")
                return result

            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.yolo_detector.run_request",
                    side_effect=noisy_model_runtime,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                self.assertEqual(0, yolo_runner_main([str(request_path)]))

            self.assertEqual(result, DetectorScanResult.from_json(stdout.getvalue()))
            self.assertNotIn("third_party_model_banner", stdout.getvalue())
            self.assertIn("third_party_model_banner", stderr.getvalue())

    def test_result_round_trips_and_never_contains_local_paths(self) -> None:
        result = DetectorScanResult(
            camera_id="cam_01",
            detector_adapter="ultralytics_yolo_subprocess_v1",
            model_id="fixture_yolo",
            model_revision="revision_01",
            model_sha256="a" * 64,
            detections=(Detection(1.0, "person", 0.8, (0.1, 0.2, 0.5, 0.9)),),
        )

        restored = DetectorScanResult.from_json(result.to_json())

        self.assertEqual(result, restored)
        self.assertNotIn("source_path", result.to_json())
        self.assertNotIn("manifest_path", result.to_json())
        self.assertNotIn("/", result.to_json())

        scan = AdmittedDetectorScan(
            source_admission_sha256="b" * 64,
            labels=("person",),
            confidence_threshold=0.35,
            requested_sample_fps=1.0,
            result=result,
        )
        schema = json.loads(
            Path("schemas/admitted-detector-scan-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(scan.to_dict()))
        self.assertEqual(scan, AdmittedDetectorScan.from_json(scan.to_json()))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            result.write_new(output)
            self.assertEqual(result, DetectorScanResult.read(output))
            self.assertEqual(0o600, output.stat().st_mode & 0o777)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                result.write_new(output)
            if os.name == "posix":
                link = root / "result-link.json"
                link.symlink_to(output)
                with self.assertRaisesRegex(DetectorProtocolError, "unsafe"):
                    DetectorScanResult.read(link)
            scan_output = root / "scan.json"
            scan.write_new(scan_output)
            self.assertEqual(scan, AdmittedDetectorScan.read(scan_output))
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    0,
                    main(["validate-detector-scan", str(scan_output)]),
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(scan.fingerprint, summary["scan_sha256"])
            self.assertEqual(1, summary["detection_count"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertNotIn(str(root), stdout.getvalue())

    def test_result_rejects_extra_duplicate_and_identity_fields(self) -> None:
        payload = {
            "version": 1,
            "camera_id": "cam_01",
            "detector_adapter": "runner",
            "model_id": "fixture_yolo",
            "model_revision": "revision_01",
            "model_sha256": "a" * 64,
            "detections": [],
            "subject_identity_inference": False,
            "paths_disclosed": False,
        }
        with self.assertRaises(DetectorProtocolError):
            DetectorScanResult.from_dict({**payload, "extra": True})
        with self.assertRaises(DetectorProtocolError):
            DetectorScanResult.from_json('{"version":1,"version":1,"camera_id":"cam_01"}')
        with self.assertRaises(DetectorProtocolError):
            DetectorScanResult.from_dict({**payload, "subject_identity_inference": True})

    def test_request_requires_regular_local_paths_and_bounded_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            weight = root / "fixture.pt"
            manifest = root / "manifest.json"
            for path in (source, weight, manifest):
                path.write_text("fixture", encoding="utf-8")
            request = DetectorScanRequest(
                camera_id="cam_01",
                detector_adapter="runner",
                source_path=str(source),
                model_path=str(weight),
                manifest_path=str(manifest),
                model_id="fixture_yolo",
                model_revision="revision_01",
                model_sha256="a" * 64,
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="0",
            )
            self.assertEqual(
                request,
                DetectorScanRequest.from_json(request.to_json()),
            )
            self.assertEqual(2, request.version)
            self.assertEqual(640, request.image_size)
            self.assertEqual(0, request.rotation_degrees)
            legacy = request.to_dict()
            legacy["version"] = 1
            legacy.pop("image_size")
            legacy.pop("rotation_degrees")
            migrated = DetectorScanRequest.from_dict(legacy)
            self.assertEqual(2, migrated.version)
            self.assertEqual(640, migrated.image_size)
            self.assertEqual(0, migrated.rotation_degrees)
            with self.assertRaises(DetectorProtocolError):
                DetectorScanRequest.from_dict({**request.to_dict(), "device": "cuda:0"})
            with self.assertRaisesRegex(DetectorProtocolError, "image_size"):
                DetectorScanRequest.from_dict({**request.to_dict(), "image_size": 641})
            with self.assertRaisesRegex(DetectorProtocolError, "rotation_degrees"):
                DetectorScanRequest.from_dict({**request.to_dict(), "rotation_degrees": 45})


if __name__ == "__main__":
    unittest.main()
