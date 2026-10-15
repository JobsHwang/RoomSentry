from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters.camera import LocalFileCamera, RTSPCamera, redact_uri
from baby_monitor.adapters.detector import (
    CommandYoloDetector,
    DetectorError,
    UltralyticsYoloDetector,
    group_detections,
)
from baby_monitor.adapters.vlm import (
    CommandVLMAdapter,
    MockVLMAdapter,
    OpenAICompatibleVLMAdapter,
    VLMError,
    VLMRequest,
    _GPUDeviceMemorySampler,
)
from baby_monitor.cli_support import _yolo_detector
from baby_monitor.config import load_settings
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    EvidenceGap,
    RecognitionProvenance,
)
from baby_monitor.remote_mock import RemoteMockServer
from baby_monitor.remote_review import RemoteAttemptStatus, RemoteReviewAuditStore
from baby_monitor.yolo_model import YoloModelManifest


class AdapterSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        # These runners are fixtures, not GPU workloads. Exercise the real
        # lock implementation without contending with a live user's model.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        runtime_root = Path(temporary.name).resolve()
        runtime_root.chmod(0o700)
        execution_root = patch(
            "baby_monitor.adapters.vlm_execution._execution_root", return_value=runtime_root
        )
        execution_root.start()
        self.addCleanup(execution_root.stop)

    def test_shadow_yolo_can_override_primary_model_without_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "shadow.pt"
            weight.write_bytes(b"independent-shadow-weight")
            manifest = YoloModelManifest(
                model_id="fixture_shadow_yolo",
                revision="shadow_revision_01",
                filename=weight.name,
                sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                size_bytes=weight.stat().st_size,
                source_url="https://example.invalid/shadow.pt",
                license_spdx="LicenseRef-Fixture",
                license_url="https://example.invalid/license",
                labels=("person",),
            )
            manifest_path = root / "shadow-manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

            detector = _yolo_detector(
                load_settings("config/default.toml"),
                subprocess_backend=True,
                model_path_override=weight,
                manifest_path_override=manifest_path,
                runner_command_override=(sys.executable, "{request}"),
                labels_override=("person",),
                device_override="cpu",
            )

            self.assertEqual("fixture_shadow_yolo", detector.manifest.model_id)
            self.assertEqual(weight.resolve(), detector.model_path)
            self.assertEqual((sys.executable, "{request}"), detector.command)
            self.assertEqual(frozenset({"person"}), detector.labels)
            self.assertEqual("cpu", detector.device)

    @staticmethod
    def _vlm_request() -> VLMRequest:
        return VLMRequest(
            candidate=CandidateEvent("event", "cam", 0.0, 1.0, ()),
            media=EventMedia("", ("missing-private-frame.jpg",)),
        )

    def test_rtsp_credentials_are_redacted_from_audit_ref(self) -> None:
        # release-audit: allow=fixture-credential
        source = RTSPCamera(
            "cam_01", "rtsp://secret-user:secret-pass@192.0.2.10:8554/live?token=x"
        ).source()
        self.assertEqual("rtsp://192.0.2.10:8554/live", source.audit_ref)
        self.assertNotIn("secret", source.audit_ref)
        self.assertNotIn("token", source.audit_ref)
        self.assertEqual(source.audit_ref, redact_uri(source.read_uri))

    def test_yolo_requires_existing_local_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does-not-exist.pt"
            manifest = YoloModelManifest(
                model_id="fixture",
                revision="test",
                filename=missing.name,
                sha256="0" * 64,
                size_bytes=1,
                source_url="https://example.invalid/fixture.pt",
                license_spdx="LicenseRef-Fixture",
                license_url="https://example.invalid/license",
                labels=("person",),
            )
            with self.assertRaisesRegex(DetectorError, "automatic downloads are disabled"):
                UltralyticsYoloDetector(
                    model_path=missing,
                    manifest=manifest,
                    labels=["person"],
                    confidence=0.35,
                    sample_fps=1.0,
                    device="0",
                )

    @staticmethod
    def _command_yolo_fixture(
        root: Path,
        *,
        script_body: str | None = None,
        timeout_seconds: float = 5.0,
        execution_lock_path: Path | None = None,
        execution_lock_timeout_seconds: float = 5.0,
    ) -> tuple[CommandYoloDetector, Path]:
        weight = root / "fixture.pt"
        weight.write_bytes(b"fixture-weight")
        manifest = YoloModelManifest(
            model_id="fixture_yolo",
            revision="revision_01",
            filename=weight.name,
            sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
            size_bytes=weight.stat().st_size,
            source_url="https://example.invalid/fixture.pt",
            license_spdx="LicenseRef-Fixture",
            license_url="https://example.invalid/license",
            labels=("person", "dog"),
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
        source = root / "input.mp4"
        source.write_bytes(b"fixture-video")
        script = root / "runner.py"
        script.write_text(
            script_body
            or """
import json
import os
import sys
from pathlib import Path

if "PRIVATE_TEST_SECRET" in os.environ:
    raise SystemExit(9)
request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "version": 1,
    "camera_id": request["camera_id"],
    "detector_adapter": request["detector_adapter"],
    "model_id": request["model_id"],
    "model_revision": request["model_revision"],
    "model_sha256": request["model_sha256"],
    "detections": [{
        "timestamp_seconds": 1.0,
        "label": "person",
        "confidence": 0.8,
        "bbox_xyxy": [0.1, 0.2, 0.5, 0.9],
    }],
    "subject_identity_inference": False,
    "paths_disclosed": False,
}))
""".strip()
            + "\n",
            encoding="utf-8",
        )
        detector = CommandYoloDetector(
            model_path=weight,
            manifest_path=manifest_path,
            manifest=manifest,
            labels=("person",),
            confidence=0.35,
            sample_fps=1.0,
            device="0",
            command=(sys.executable, str(script), "{request}"),
            timeout_seconds=timeout_seconds,
            execution_lock_path=execution_lock_path,
            execution_lock_timeout_seconds=execution_lock_timeout_seconds,
        )
        return detector, source

    def test_yolo_command_runs_without_inheriting_secrets(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"PRIVATE_TEST_SECRET": "must-not-cross-boundary"},
                clear=False,
            ),
        ):
            detector, source = self._command_yolo_fixture(Path(directory))

            detections = detector.scan(LocalFileCamera("cam_01", source).source())

        self.assertEqual(1, len(detections))
        self.assertEqual("person", detections[0].label)

    def test_yolo_command_rejects_metadata_mismatch_and_kills_timeout(self) -> None:
        mismatch_script = """
import json
import sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
request["model_id"] = "wrong_model"
print(json.dumps({
    "version": 1,
    "camera_id": request["camera_id"],
    "detector_adapter": request["detector_adapter"],
    "model_id": request["model_id"],
    "model_revision": request["model_revision"],
    "model_sha256": request["model_sha256"],
    "detections": [],
    "subject_identity_inference": False,
    "paths_disclosed": False,
}))
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector, source = self._command_yolo_fixture(
                root,
                script_body=mismatch_script,
            )
            with self.assertRaisesRegex(DetectorError, "metadata"):
                detector.scan(LocalFileCamera("cam_01", source).source())

            timeout_script = root / "timeout_runner.py"
            timeout_script.write_text(
                "import time\ntime.sleep(3)\n",
                encoding="utf-8",
            )
            timeout_detector = CommandYoloDetector(
                model_path=detector.model_path,
                manifest_path=detector.manifest_path,
                manifest=detector.manifest,
                labels=("person",),
                confidence=0.35,
                sample_fps=1.0,
                device="0",
                command=(sys.executable, str(timeout_script), "{request}"),
                timeout_seconds=1.0,
            )
            with self.assertRaisesRegex(DetectorError, "timed out"):
                timeout_detector.scan(LocalFileCamera("cam_01", source).source())

    @unittest.skipUnless(os.name == "posix", "POSIX advisory lock required")
    def test_yolo_command_serializes_execution_with_private_lock(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            lock_path = root / "yolo-execution.lock"
            detector, source = self._command_yolo_fixture(
                root,
                execution_lock_path=lock_path,
                execution_lock_timeout_seconds=1.0,
            )
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(DetectorError, "execution lock timed out"):
                    detector.scan(LocalFileCamera("cam_01", source).source())
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

            detections = detector.scan(LocalFileCamera("cam_01", source).source())
            self.assertEqual(1, len(detections))
            self.assertEqual(0o600, lock_path.stat().st_mode & 0o777)

            lock_path.chmod(0o644)
            with self.assertRaisesRegex(DetectorError, "execution lock is unsafe"):
                detector.scan(LocalFileCamera("cam_01", source).source())

    @unittest.skipUnless(os.name == "posix", "POSIX advisory lock required")
    def test_yolo_execution_lock_hands_off_to_an_already_waiting_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            order_path = root / "order.txt"
            script_body = f"""
import json
import time
import sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with Path({str(order_path)!r}).open("a", encoding="utf-8") as handle:
    handle.write(request["camera_id"] + "\\n")
time.sleep(0.25)
print(json.dumps({{
    "version": 1,
    "camera_id": request["camera_id"],
    "detector_adapter": request["detector_adapter"],
    "model_id": request["model_id"],
    "model_revision": request["model_revision"],
    "model_sha256": request["model_sha256"],
    "detections": [],
    "subject_identity_inference": False,
    "paths_disclosed": False,
}}))
""".strip()
            lock_path = root / "yolo-execution.lock"
            detector_a, source = self._command_yolo_fixture(
                root,
                script_body=script_body,
                execution_lock_path=lock_path,
            )
            detector_b, _source = self._command_yolo_fixture(
                root,
                script_body=script_body,
                execution_lock_path=lock_path,
            )
            barrier = threading.Barrier(2)

            def run_a() -> None:
                barrier.wait()
                detector_a.scan(LocalFileCamera("cam_a", source).source())
                detector_a.scan(LocalFileCamera("cam_a", source).source())

            def run_b() -> None:
                barrier.wait()
                detector_b.scan(LocalFileCamera("cam_b", source).source())

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = (executor.submit(run_a), executor.submit(run_b))
                for future in futures:
                    future.result(timeout=5)

            order = order_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(3, len(order))
            self.assertNotEqual(order[0], order[1])

    def test_detections_group_into_bursts(self) -> None:
        detections = [
            Detection(2.0, "person", 0.7, (0, 0, 1, 1)),
            Detection(3.0, "person", 0.8, (0, 0, 1, 1)),
            Detection(10.0, "dog", 0.9, (0, 0, 1, 1)),
        ]
        candidates = group_detections(
            detections,
            camera_id="cam_01",
            gap_seconds=2.0,
            pre_event_seconds=5.0,
            post_event_seconds=7.0,
        )
        self.assertEqual(2, len(candidates))
        self.assertEqual(0.0, candidates[0].start_seconds)
        self.assertEqual(("person",), candidates[0].labels)
        self.assertEqual(("dog",), candidates[1].labels)

    def test_remote_adapter_refuses_before_reading_frames(self) -> None:
        adapter = OpenAICompatibleVLMAdapter(
            base_url="https://example.invalid/v1",
            model="fixture",
            api_key_env="NEVER_SET_FIXTURE_KEY",
            enabled=False,
            upload_consent=False,
        )
        request = VLMRequest(
            candidate=CandidateEvent("event", "cam", 0.0, 1.0, ()),
            media=EventMedia("", ("missing-private-frame.jpg",)),
        )
        with self.assertRaisesRegex(VLMError, "disabled"):
            adapter.analyze(request)

    def test_remote_coverage_payload_is_conservative_without_uploading_or_reading_frames(self):
        adapter = OpenAICompatibleVLMAdapter(
            base_url="https://example.invalid/v1",
            model="fixture",
            api_key_env="UNSET_FIXTURE_KEY",
            enabled=False,
            upload_consent=False,
        )
        request = self._vlm_request()
        started = datetime(2026, 9, 12, tzinfo=UTC)
        gap = EvidenceGap(started, started + timedelta(seconds=3), "sampling_gap")
        for gaps, status in (((), "unverified"), ((gap,), "gaps_recorded")):
            with (
                self.subTest(status=status),
                patch("socket.socket", side_effect=AssertionError("no network")),
            ):
                bounded = replace(
                    request, media=replace(request.media, coverage_gaps=gaps, selected_segments=2)
                )
                facts = adapter._coverage_facts(bounded)
                payload = adapter._payload(bounded, ())
                text = payload["messages"][-1]["content"][0]["text"]
                self.assertEqual(status, facts["coverage_status"])
                self.assertIs(False, facts["continuous_coverage_verified"])
                self.assertEqual(2, facts["selected_segments"])
                self.assertEqual(
                    [{"reason": "sampling_gap", "duration_seconds": 3.0}] if gaps else [],
                    facts["gaps"],
                )
                self.assertNotIn("complete", facts)
                self.assertNotIn("complete_coverage", facts)
                self.assertNotIn('"complete":', text)
                self.assertNotIn('"complete_coverage":', text)
                self.assertIn("未验证不等于已记录缺失，也不等于连续覆盖", text)
                self.assertNotIn(started.isoformat(), text)
                self.assertNotIn("missing-private-frame.jpg", text)

    def test_mock_reports_unverified_coverage_without_fabricating_a_missing_duration(self):
        request = self._vlm_request()
        adapter = MockVLMAdapter()
        event = adapter.analyze(request)
        self.assertEqual(
            ("系统取证窗口未验证连续覆盖；未验证不等于已记录缺失，也不等于连续覆盖。",),
            event.observations,
        )
        self.assertEqual((), event.inferences)
        self.assertLessEqual(event.confidence, 0.35)
        started = datetime(2026, 9, 12, tzinfo=UTC)
        gap = EvidenceGap(started, started + timedelta(seconds=3), "sampling_gap")
        with_gap = adapter.analyze(
            replace(request, media=replace(request.media, coverage_gaps=(gap,)))
        )
        self.assertEqual(
            ("系统取证窗口存在覆盖缺口：sampling_gap；缺失时段不能由模型补全。",),
            with_gap.observations,
        )

    def test_changed_mock_and_remote_prompts_have_new_provenance_without_rewriting_legacy_default(
        self,
    ):
        adapter = OpenAICompatibleVLMAdapter(
            base_url="https://example.invalid/v1",
            model="fixture",
            api_key_env="UNSET_FIXTURE_KEY",
            enabled=False,
            upload_consent=False,
        )
        expected = "baby_monitor_vlm_prompt_v3_coverage_unverified"
        self.assertEqual(expected, adapter.provenance.prompt_revision)
        self.assertEqual(expected, MockVLMAdapter().provenance.prompt_revision)
        legacy = RecognitionProvenance(adapter_name="legacy_fixture")
        self.assertEqual("baby_monitor_vlm_prompt_v2", legacy.prompt_revision)
        self.assertEqual(legacy, RecognitionProvenance.from_dict(legacy.to_dict()))

    def test_remote_adapter_uses_loopback_mock_and_never_sends_local_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_LOCAL_ROOT_") as directory:
            root = Path(directory)
            frames = tuple(root / f"private_frame_{index}.jpg" for index in range(8))
            for index, frame in enumerate(frames):
                frame.write_bytes(f"fixture-{index}".encode("ascii"))
            database_path = root / "events.sqlite3"
            server = RemoteMockServer(("127.0.0.1", 0), capture_payload=True)
            server.timeout = 2
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                adapter = OpenAICompatibleVLMAdapter(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="local-fixture",
                    api_key_env="BABY_MONITOR_TEST_REMOTE_KEY",
                    enabled=True,
                    upload_consent=True,
                    max_frames=3,
                    min_interval_seconds=0,
                    audit_store=RemoteReviewAuditStore(database_path),
                )
                request = VLMRequest(
                    candidate=CandidateEvent(
                        "PRIVATE_EVENT_ID_DO_NOT_UPLOAD",
                        "PRIVATE_CAMERA_ID_DO_NOT_UPLOAD",
                        0.0,
                        7.0,
                        (Detection(1.0, "person", 0.8, (0.1, 0.1, 0.9, 0.9)),),
                    ),
                    media=EventMedia(
                        str(root / "PRIVATE_CLIP_PATH_DO_NOT_UPLOAD.mp4"),
                        tuple(str(frame) for frame in frames),
                    ),
                    question="PRIVATE_CUSTOM_QUESTION_DO_NOT_UPLOAD",
                )
                preview = adapter.preview(request)
                self.assertEqual(3, preview.selected_frame_count)
                self.assertEqual(5, preview.omitted_frame_count)
                with patch.dict(
                    os.environ,
                    {"BABY_MONITOR_TEST_REMOTE_KEY": "PRIVATE_KEY_DO_NOT_CAPTURE"},
                    clear=False,
                ):
                    event = adapter.analyze(request)
            finally:
                thread.join(timeout=3)
                server.server_close()

            self.assertFalse(thread.is_alive())
            self.assertEqual(1, server.request_count)
            self.assertEqual((str(frames[0]),), event.evidence_frames)
            assert server.last_request_payload is not None
            serialized = json.dumps(server.last_request_payload, ensure_ascii=False)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("PRIVATE_EVENT_ID_DO_NOT_UPLOAD", serialized)
            self.assertNotIn("PRIVATE_CAMERA_ID_DO_NOT_UPLOAD", serialized)
            self.assertNotIn("PRIVATE_CLIP_PATH_DO_NOT_UPLOAD", serialized)
            self.assertNotIn("PRIVATE_CUSTOM_QUESTION_DO_NOT_UPLOAD", serialized)
            self.assertNotIn("PRIVATE_KEY_DO_NOT_CAPTURE", serialized)
            image_items = [
                item
                for item in server.last_request_payload["messages"][-1]["content"]
                if item.get("type") == "image_url"
            ]
            self.assertEqual(3, len(image_items))
            attempts = RemoteReviewAuditStore(database_path).recent()
            self.assertEqual(1, len(attempts))
            self.assertEqual(RemoteAttemptStatus.SUCCEEDED, attempts[0].status)
            self.assertEqual(3, attempts[0].frame_count)

    def test_remote_transport_failure_is_sanitized_and_audited(self) -> None:
        class FailingOpener:
            def open(self, request, timeout):  # type: ignore[no-untyped-def]
                raise TimeoutError("PRIVATE_NETWORK_DETAIL")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.jpg"
            frame.write_bytes(b"fixture")
            audit = RemoteReviewAuditStore(root / "events.sqlite3")
            adapter = OpenAICompatibleVLMAdapter(
                base_url="https://provider.example/v1",
                model="fixture",
                api_key_env="BABY_MONITOR_TEST_REMOTE_KEY",
                enabled=True,
                upload_consent=True,
                min_interval_seconds=0,
                audit_store=audit,
            )
            request = VLMRequest(
                candidate=CandidateEvent("event", "cam", 0.0, 1.0, ()),
                media=EventMedia("", (str(frame),)),
            )
            with (
                patch.dict(
                    os.environ,
                    {"BABY_MONITOR_TEST_REMOTE_KEY": "PRIVATE_KEY_DO_NOT_ECHO"},
                    clear=False,
                ),
                patch(
                    "baby_monitor.adapters.vlm.urllib.request.build_opener",
                    return_value=FailingOpener(),
                ),
                self.assertRaisesRegex(VLMError, "request failed") as raised,
            ):
                adapter.analyze(request)
            self.assertNotIn("PRIVATE_NETWORK_DETAIL", str(raised.exception))
            self.assertNotIn("PRIVATE_KEY_DO_NOT_ECHO", str(raised.exception))
            attempts = audit.recent()
            self.assertEqual(RemoteAttemptStatus.FAILED, attempts[0].status)
            self.assertEqual("transport_error", attempts[0].error_code)

    def test_remote_invalid_json_is_sanitized_and_audited(self) -> None:
        class InvalidResponse:
            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, exc_type, exc, traceback):  # type: ignore[no-untyped-def]
                return False

            def read(self, limit):  # type: ignore[no-untyped-def]
                return b"PRIVATE_INVALID_PROVIDER_BODY"

        class InvalidResponseOpener:
            def open(self, request, timeout):  # type: ignore[no-untyped-def]
                return InvalidResponse()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.jpg"
            frame.write_bytes(b"fixture")
            audit = RemoteReviewAuditStore(root / "events.sqlite3")
            adapter = OpenAICompatibleVLMAdapter(
                base_url="https://provider.example/v1",
                model="fixture",
                api_key_env="BABY_MONITOR_TEST_REMOTE_KEY",
                enabled=True,
                upload_consent=True,
                min_interval_seconds=0,
                audit_store=audit,
            )
            request = VLMRequest(
                candidate=CandidateEvent("event", "cam", 0.0, 1.0, ()),
                media=EventMedia("", (str(frame),)),
            )
            with (
                patch.dict(
                    os.environ,
                    {"BABY_MONITOR_TEST_REMOTE_KEY": "fixture"},
                    clear=False,
                ),
                patch(
                    "baby_monitor.adapters.vlm.urllib.request.build_opener",
                    return_value=InvalidResponseOpener(),
                ),
                self.assertRaisesRegex(VLMError, "not valid UTF-8 JSON") as raised,
            ):
                adapter.analyze(request)
            self.assertNotIn("PRIVATE_INVALID_PROVIDER_BODY", str(raised.exception))
            attempts = audit.recent()
            self.assertEqual(RemoteAttemptStatus.FAILED, attempts[0].status)
            self.assertEqual("invalid_response", attempts[0].error_code)

    def test_remote_upload_byte_limit_refuses_before_audit_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.jpg"
            frame.write_bytes(b"too-large-for-fixture")
            database = root / "events.sqlite3"
            adapter = OpenAICompatibleVLMAdapter(
                base_url="https://provider.example/v1",
                model="fixture",
                api_key_env="BABY_MONITOR_TEST_REMOTE_KEY",
                enabled=True,
                upload_consent=True,
                max_upload_bytes=1,
                audit_store=RemoteReviewAuditStore(database),
            )
            request = VLMRequest(
                candidate=CandidateEvent("event", "cam", 0.0, 1.0, ()),
                media=EventMedia("", (str(frame),)),
            )
            with (
                patch.dict(
                    os.environ,
                    {"BABY_MONITOR_TEST_REMOTE_KEY": "fixture"},
                    clear=False,
                ),
                patch("baby_monitor.adapters.vlm.urllib.request.build_opener") as opener,
                self.assertRaisesRegex(VLMError, "byte budget"),
            ):
                adapter.analyze(request)
            opener.assert_not_called()
            self.assertFalse(database.exists())

    def test_remote_adapter_rejects_non_loopback_plain_http_before_frame_access(self) -> None:
        adapter = OpenAICompatibleVLMAdapter(
            base_url="http://192.0.2.10:8080/v1",
            model="fixture",
            api_key_env="BABY_MONITOR_TEST_REMOTE_KEY",
            enabled=False,
            upload_consent=False,
        )
        with self.assertRaisesRegex(VLMError, "loopback"):
            adapter.preview(self._vlm_request())

    def test_remote_adapter_refuses_upload_when_audit_ledger_is_unavailable(self) -> None:
        class BrokenAuditStore:
            def initialize(self) -> None:
                raise RuntimeError("PRIVATE_DATABASE_DETAIL")

        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"fixture")
            adapter = OpenAICompatibleVLMAdapter(
                base_url="https://provider.example/v1",
                model="fixture",
                api_key_env="BABY_MONITOR_TEST_REMOTE_KEY",
                enabled=True,
                upload_consent=True,
                audit_store=BrokenAuditStore(),  # type: ignore[arg-type]
            )
            request = VLMRequest(
                candidate=CandidateEvent("event", "cam", 0.0, 1.0, ()),
                media=EventMedia("", (str(frame),)),
            )
            with (
                patch.dict(
                    os.environ,
                    {"BABY_MONITOR_TEST_REMOTE_KEY": "fixture"},
                    clear=False,
                ),
                patch("baby_monitor.adapters.vlm.urllib.request.build_opener") as opener,
                self.assertRaisesRegex(VLMError, "upload was refused") as raised,
            ):
                adapter.analyze(request)
            opener.assert_not_called()
            self.assertNotIn("PRIVATE_DATABASE_DETAIL", str(raised.exception))

    def test_command_adapter_filters_inherited_secrets(self) -> None:
        runner = """
import json
import os
import sys
if "BABY_MONITOR_REMOTE_API_KEY" in os.environ or "CAMERA_PASSWORD" in os.environ:
    raise SystemExit(9)
with open(sys.argv[1], encoding="utf-8") as handle:
    request = json.load(handle)
print(json.dumps({
    "event_type": "unknown",
    "subjects": [],
    "observations": [],
    "inferences": [],
    "risk_level": "none",
    "confidence": 0.0,
    "needs_remote_review": False,
    "summary_zh": "",
    "evidence_frames": request["frame_paths"][:1],
}))
"""
        adapter = CommandVLMAdapter(
            name="fixture",
            command=(sys.executable, "-c", runner, "{request}"),
            timeout_seconds=5,
        )
        with patch.dict(
            os.environ,
            {
                "BABY_MONITOR_REMOTE_API_KEY": "must-not-leak",
                "CAMERA_PASSWORD": "must-not-leak",
            },
            clear=False,
        ):
            result = adapter.analyze(self._vlm_request())
        self.assertEqual("unknown", result.event_type.value)

    def test_command_adapter_does_not_echo_invalid_private_output(self) -> None:
        marker = "PRIVATE_RAW_SHOULD_NOT_ESCAPE"
        adapter = CommandVLMAdapter(
            name="fixture",
            command=(sys.executable, "-c", f'print("{marker}")', "{request}"),
            timeout_seconds=5,
        )
        with self.assertRaises(VLMError) as raised:
            adapter.analyze(self._vlm_request())
        self.assertNotIn(marker, str(raised.exception))
        self.assertIn("invalid structured output", str(raised.exception))

    def test_command_adapter_maps_timeout_without_runner_output(self) -> None:
        adapter = CommandVLMAdapter(
            name="fixture",
            command=(
                sys.executable,
                "-c",
                "import time; time.sleep(5)",
                "{request}",
            ),
            timeout_seconds=1,
        )
        with self.assertRaisesRegex(VLMError, "timed out"):
            adapter.analyze(self._vlm_request())

    def test_command_adapter_accepts_only_strict_path_free_failure_status(self) -> None:
        status = json.dumps(
            {
                "protocol": "baby_monitor_local_runner_failure_v1",
                "error_code": "invalid_structured_output",
                "oom_status": "not_observed",
            }
        )
        runner = f"import sys; print({status!r}, file=sys.stderr); raise SystemExit(2)"
        adapter = CommandVLMAdapter(
            name="fixture",
            command=(sys.executable, "-c", runner, "{request}"),
            timeout_seconds=5,
        )
        with self.assertRaises(VLMError) as raised:
            adapter.analyze_with_metrics(self._vlm_request())
        self.assertEqual("invalid_structured_output", raised.exception.code)
        self.assertEqual("not_observed", raised.exception.oom_status)
        assert raised.exception.metrics is not None
        self.assertFalse(raised.exception.metrics.json_compliant)
        self.assertGreaterEqual(raised.exception.metrics.duration_ms, 0)

        malformed_runner = (
            "import sys; print('private path /must/not/persist', file=sys.stderr); "
            "raise SystemExit(2)"
        )
        malformed = CommandVLMAdapter(
            name="fixture",
            command=(sys.executable, "-c", malformed_runner, "{request}"),
            timeout_seconds=5,
        )
        with self.assertRaises(VLMError) as malformed_raised:
            malformed.analyze_with_metrics(self._vlm_request())
        self.assertEqual("runner_failed", malformed_raised.exception.code)
        self.assertEqual("unknown", malformed_raised.exception.oom_status)
        assert malformed_raised.exception.metrics is not None
        self.assertFalse(malformed_raised.exception.metrics.json_compliant)

    def test_gpu_sample_parser_is_strict_and_bounded(self) -> None:
        self.assertEqual(1234, _GPUDeviceMemorySampler._parse_sample("1234\n"))
        self.assertIsNone(_GPUDeviceMemorySampler._parse_sample("N/A"))
        self.assertIsNone(_GPUDeviceMemorySampler._parse_sample("10000001"))
        self.assertIsNone(_GPUDeviceMemorySampler._parse_sample("１２３"))

    def test_gpu_sampler_missing_tool_keeps_peak_null(self) -> None:
        with (
            patch("baby_monitor.adapters.vlm.shutil.which", return_value=None),
            _GPUDeviceMemorySampler(device="0", enabled=True) as sampler,
        ):
            pass
        self.assertIsNone(sampler.peak_mib)

    @unittest.skipUnless(os.name == "posix", "fake executable test is POSIX-specific")
    def test_command_metrics_capture_gpu_device_peak_and_reap_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path = root / "sampler.pid"
            fake_smi = root / "nvidia-smi"
            fake_smi.write_text(
                f"#!{sys.executable}\n"
                "import os,pathlib,sys,time\n"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "str(os.getpid()), encoding='utf-8')\n"
                "print(100, flush=True)\n"
                "time.sleep(0.05)\n"
                "print(275, flush=True)\n"
                "time.sleep(5)\n",
                encoding="utf-8",
            )
            fake_smi.chmod(0o700)
            runner = """
import json
import sys
import time
time.sleep(0.2)
with open(sys.argv[1], encoding="utf-8") as handle:
    request = json.load(handle)
print(json.dumps({
    "event_type": "unknown",
    "subjects": [],
    "observations": [],
    "inferences": [],
    "risk_level": "none",
    "confidence": 0.0,
    "needs_remote_review": False,
    "summary_zh": "",
    "evidence_frames": request["frame_paths"][:1],
}))
"""
            adapter = CommandVLMAdapter(
                name="fixture",
                command=(sys.executable, "-c", runner, "{request}"),
                timeout_seconds=5,
            )
            with patch(
                "baby_monitor.adapters.vlm.shutil.which",
                return_value=str(fake_smi),
            ):
                analysis = adapter.analyze_with_metrics(self._vlm_request())
            self.assertEqual(275, analysis.metrics.gpu_device_memory_peak_mib)
            sampler_pid = int(pid_path.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(sampler_pid, 0)

    @unittest.skipUnless(os.name == "posix", "process-group cleanup is POSIX-specific")
    def test_timed_command_timeout_kills_descendant_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "orphan-marker"
            grandchild = (
                "import pathlib,sys,time; time.sleep(2); "
                "pathlib.Path(sys.argv[1]).write_text('orphan', encoding='utf-8')"
            )
            runner = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
                "time.sleep(5)"
            )
            adapter = CommandVLMAdapter(
                name="fixture",
                command=(
                    sys.executable,
                    "-c",
                    runner,
                    grandchild,
                    str(marker),
                    "{request}",
                ),
                timeout_seconds=1,
            )

            with self.assertRaisesRegex(VLMError, "timed out"):
                adapter.analyze_with_metrics(self._vlm_request())
            time.sleep(2.2)
            self.assertFalse(marker.exists())
            self.assertFalse(
                any(thread.name == "baby-monitor-gpu-sampler" for thread in threading.enumerate())
            )


if __name__ == "__main__":
    unittest.main()
