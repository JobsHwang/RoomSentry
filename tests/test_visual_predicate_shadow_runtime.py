from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.domain import CandidateEvent, Detection, SignalEvent, SignalType
from baby_monitor.visual_predicate_shadow_runtime import (
    VisualPredicateShadowRuntimeError,
    load_private_shadow_candidate,
    load_visual_predicate_shadow_runtime_settings,
)

_RESIDENT_RUNNER = r"""
import json
import pathlib
import sys

counter = pathlib.Path(sys.argv[1])
with counter.open("a", encoding="utf-8") as handle:
    handle.write("start\n")
print(json.dumps({
    "version": 1,
    "status": "ready",
    "model_id": "microsoft.Mage-VL",
    "model_revision": "5c78cab61938e73859b63724d9bf5cb88c477eaa",
    "prompt_version": "mage_atomic_visible_v1",
    "max_jobs": 2,
    "idle_exit_seconds": 30.0,
    "max_runtime_seconds": 300.0,
}, separators=(",", ":")), flush=True)
try:
    for line in sys.stdin:
        control = json.loads(line)
        request = json.loads(
            pathlib.Path(control["request_path"]).read_text(encoding="utf-8")
        )
        frame_id = request["frames"][0]["frame_id"]
        answer = "clear" if request["predicate"] == "face_visibility" else "no"
        result = {
            "version": 1,
            "event_id": request["event_id"],
            "predicate": request["predicate"],
            "answer": answer,
            "confidence": 0.8,
            "evidence_frames": [frame_id],
            "model_id": request["model_id"],
            "model_revision": request["model_revision"],
            "prompt_version": request["prompt_version"],
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
        print(json.dumps({
            "version": 1,
            "status": "ok",
            "result": result,
        }, separators=(",", ":")), flush=True)
finally:
    with counter.open("a", encoding="utf-8") as handle:
        handle.write("stop\n")
"""


class VisualPredicateShadowRuntimeTests(unittest.TestCase):
    @staticmethod
    def _config(root: Path, database: Path, *, enabled: bool = True) -> Path:
        config = root / "config.toml"
        config.write_text(
            "\n".join(
                (
                    "[storage]",
                    f'database = "{database.as_posix()}"',
                    "",
                    "[visual_predicate_shadow]",
                    f"enabled = {'true' if enabled else 'false'}",
                    'backend = "conservative"',
                    "command = []",
                    'predicates = ["face_visibility", "covering_near_face_visible"]',
                    'hardware_signal_adapters = ["home_assistant"]',
                    "min_person_confidence = 0.25",
                    'model_id = "microsoft.Mage-VL"',
                    'model_revision = "5c78cab61938e73859b63724d9bf5cb88c477eaa"',
                    'prompt_version = "mage_atomic_visible_v1"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    @staticmethod
    def _candidate(*, with_hardware_signal: bool, detector_id: str | None = None) -> CandidateEvent:
        signals = (
            (
                SignalEvent(
                    signal_id="signal_01",
                    camera_id="camera_01",
                    occurred_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
                    signal_type=SignalType.MOTION,
                    source_adapter="home_assistant",
                    confidence=1.0,
                ),
            )
            if with_hardware_signal
            else ()
        )
        return CandidateEvent(
            event_id="event_01",
            camera_id="camera_01",
            start_seconds=0.0,
            end_seconds=2.0,
            detections=(
                Detection(
                    timestamp_seconds=1.0,
                    label="person",
                    confidence=0.8,
                    bbox_xyxy=(0.1, 0.1, 0.8, 0.9),
                ),
            ),
            signals=signals,
            detector_id=detector_id,
        )

    @staticmethod
    def _write_candidate(path: Path, candidate: CandidateEvent) -> None:
        path.write_text(
            json.dumps(candidate.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def test_default_configuration_keeps_shadow_disabled(self) -> None:
        settings = load_visual_predicate_shadow_runtime_settings("config/default.toml")
        self.assertFalse(settings.enabled)
        self.assertEqual("conservative", settings.backend)
        self.assertEqual((), settings.command)
        self.assertEqual(("home_assistant",), settings.hardware_signal_adapters)

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "visual-predicate-shadow-run",
                    "--candidate",
                    "/private/missing-candidate.json",
                    "--frame",
                    "frame_01=/private/missing-frame.jpg",
                ]
            )
        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertEqual("error: visual predicate shadow is disabled\n", stderr.getvalue())

    def test_private_candidate_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text(
                (
                    '{"event_id":"event_01","event_id":"event_02",'
                    '"camera_id":"camera_01","start_seconds":0.0,'
                    '"end_seconds":1.0,"detections":[]}'
                ),
                encoding="utf-8",
            )
            candidate.chmod(0o600)
            with self.assertRaisesRegex(
                VisualPredicateShadowRuntimeError,
                "candidate is invalid",
            ):
                load_private_shadow_candidate(candidate)

    def test_unselected_gate_does_not_read_frame_or_open_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "missing.sqlite3"
            config = self._config(root, database)
            candidate_path = root / "candidate.json"
            self._write_candidate(
                candidate_path,
                CandidateEvent(
                    event_id="event_01",
                    camera_id="camera_01",
                    start_seconds=0.0,
                    end_seconds=2.0,
                    detections=(),
                    signals=(),
                ),
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "visual-predicate-shadow-run",
                        "--config",
                        str(config),
                        "--candidate",
                        str(candidate_path),
                        "--frame",
                        "frame_01=/private/missing-frame.jpg",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertFalse(payload["gate"]["selected"])
            self.assertFalse(payload["persisted"])
            self.assertFalse(payload["os_network_isolation_enforced"])
            self.assertFalse(database.exists())

    def test_conservative_cli_persists_path_free_shadow_results_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.sqlite3"
            sqlite3.connect(database).close()
            config = self._config(root, database)
            candidate_path = root / "candidate.json"
            frame_path = root / "frame.jpg"
            self._write_candidate(candidate_path, self._candidate(with_hardware_signal=True))
            frame_path.write_bytes(b"local-fixture-frame")
            frame_path.chmod(0o600)
            arguments = [
                "visual-predicate-shadow-run",
                "--config",
                str(config),
                "--candidate",
                str(candidate_path),
                "--frame",
                f"frame_01={frame_path}",
            ]

            first_stdout = StringIO()
            with redirect_stdout(first_stdout):
                first_code = main(arguments)
            first_raw = first_stdout.getvalue()
            first = json.loads(first_raw)
            self.assertEqual(0, first_code)
            self.assertTrue(first["gate"]["selected"])
            self.assertEqual(["hardware_signal"], first["gate"]["triggers"])
            self.assertTrue(first["persisted"])
            self.assertFalse(first["reused_existing"])
            self.assertEqual(["unknown", "unknown"], [item["answer"] for item in first["results"]])
            self.assertFalse(first["risk_inference"])
            self.assertFalse(first["notification_decision"])
            self.assertFalse(first["os_network_isolation_enforced"])
            self.assertNotIn(str(root), first_raw)

            second_stdout = StringIO()
            with redirect_stdout(second_stdout):
                second_code = main(arguments)
            second = json.loads(second_stdout.getvalue())
            self.assertEqual(0, second_code)
            self.assertTrue(second["reused_existing"])

            connection = sqlite3.connect(database)
            row = connection.execute(
                "SELECT binding_json FROM visual_predicate_shadow_runs"
            ).fetchone()
            connection.close()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertNotIn(str(root), row[0])
            self.assertFalse(json.loads(row[0])["risk_inference"])

    def test_cli_uses_persisted_detector_provenance_and_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.sqlite3"
            sqlite3.connect(database).close()
            config = self._config(root, database)
            candidate_path = root / "candidate.json"
            frame_path = root / "frame.jpg"
            detector_id = "yolo_command_v1_0123456789ab"
            self._write_candidate(
                candidate_path,
                self._candidate(
                    with_hardware_signal=False,
                    detector_id=detector_id,
                ),
            )
            frame_path.write_bytes(b"detector-provenance-frame")
            frame_path.chmod(0o600)

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "visual-predicate-shadow-run",
                        "--config",
                        str(config),
                        "--candidate",
                        str(candidate_path),
                        "--frame",
                        f"frame_01={frame_path}",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(["person_detection"], payload["gate"]["triggers"])
            self.assertEqual(detector_id, payload["gate"]["person_detector_id"])
            self.assertEqual(1, payload["gate"]["person_detection_count"])

            conflict_stdout = StringIO()
            conflict_stderr = StringIO()
            with redirect_stdout(conflict_stdout), redirect_stderr(conflict_stderr):
                conflict_code = main(
                    [
                        "visual-predicate-shadow-run",
                        "--config",
                        str(config),
                        "--candidate",
                        str(candidate_path),
                        "--frame",
                        "frame_01=/private/missing-frame.jpg",
                        "--person-detector-id",
                        "different_detector_v1",
                    ]
                )
            self.assertEqual(2, conflict_code)
            self.assertEqual("", conflict_stdout.getvalue())
            self.assertEqual(
                "error: visual predicate shadow detector provenance conflicts\n",
                conflict_stderr.getvalue(),
            )

    def test_resident_command_cli_runs_two_predicates_and_closes_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.sqlite3"
            sqlite3.connect(database).close()
            runner = root / "resident_runner.py"
            counter = root / "worker_lifecycle.txt"
            runner.write_text(_RESIDENT_RUNNER, encoding="utf-8")
            command = json.dumps(
                [sys.executable, str(runner), str(counter)],
                ensure_ascii=True,
            )
            config = root / "resident.toml"
            config.write_text(
                "\n".join(
                    (
                        "[storage]",
                        f'database = "{database.as_posix()}"',
                        "",
                        "[visual_predicate_shadow]",
                        "enabled = true",
                        'backend = "resident_command"',
                        f"command = {command}",
                        'predicates = ["face_visibility", "covering_near_face_visible"]',
                        'hardware_signal_adapters = ["home_assistant"]',
                        'model_id = "microsoft.Mage-VL"',
                        'model_revision = "5c78cab61938e73859b63724d9bf5cb88c477eaa"',
                        'prompt_version = "mage_atomic_visible_v1"',
                        "max_jobs = 2",
                        "idle_exit_seconds = 30.0",
                        "max_runtime_seconds = 300.0",
                        "startup_timeout_seconds = 10.0",
                        "request_timeout_seconds = 10.0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_path = root / "candidate.json"
            frame_path = root / "frame.jpg"
            self._write_candidate(candidate_path, self._candidate(with_hardware_signal=True))
            frame_path.write_bytes(b"resident-local-frame")
            frame_path.chmod(0o600)

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "visual-predicate-shadow-run",
                        "--config",
                        str(config),
                        "--candidate",
                        str(candidate_path),
                        "--frame",
                        f"frame_01={frame_path}",
                    ]
                )
            raw = stdout.getvalue()
            payload = json.loads(raw)
            self.assertEqual(0, code)
            self.assertTrue(payload["persisted"])
            self.assertEqual(["clear", "no"], [item["answer"] for item in payload["results"]])
            self.assertEqual(["start", "stop"], counter.read_text(encoding="utf-8").splitlines())
            self.assertNotIn(str(root), raw)
            self.assertFalse(payload["subject_identity_inference"])
            self.assertFalse(payload["risk_inference"])
            self.assertFalse(payload["notification_decision"])
            self.assertFalse(payload["os_network_isolation_enforced"])

    def test_enabled_resident_backend_requires_an_existing_absolute_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "bad.toml"
            config.write_text(
                "\n".join(
                    (
                        "[visual_predicate_shadow]",
                        "enabled = true",
                        'backend = "resident_command"',
                        'command = ["/private/missing-python", "-m", "runner"]',
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VisualPredicateShadowRuntimeError,
                "executable is unavailable",
            ):
                load_visual_predicate_shadow_runtime_settings(config)


if __name__ == "__main__":
    unittest.main()
