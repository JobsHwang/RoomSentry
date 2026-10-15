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

from baby_monitor.analysis_queue import AnalysisJobStatus, AnalysisQueue
from baby_monitor.cli import main
from baby_monitor.domain import CandidateEvent, Detection, EventMedia

_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"

_RUNNER = r"""
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
    "max_jobs": 4,
    "idle_exit_seconds": 30.0,
    "max_runtime_seconds": 300.0,
}, separators=(",", ":")), flush=True)
try:
    for line in sys.stdin:
        control = json.loads(line)
        request = json.loads(
            pathlib.Path(control["request_path"]).read_text(encoding="utf-8")
        )
        result = {
            "version": 1,
            "event_id": request["event_id"],
            "predicate": request["predicate"],
            "answer": "clear",
            "confidence": 0.8,
            "evidence_frames": [request["frames"][0]["frame_id"]],
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


class VisualPredicateShadowWorkerCLITests(unittest.TestCase):
    @staticmethod
    def _complete_analysis_job(root: Path, database: Path, event_id: str) -> None:
        media_root = root / event_id
        media_root.mkdir()
        clip = media_root / "event.mp4"
        frame = media_root / "frame.jpg"
        clip.write_bytes(b"clip")
        frame.write_bytes(b"frame")
        clip.chmod(0o600)
        frame.chmod(0o600)
        candidate = CandidateEvent(
            event_id=event_id,
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
            detector_id="yolo11n.sha256_fixture",
        )
        queue = AnalysisQueue(database)
        queue.enqueue(
            candidate=candidate,
            occurred_at=datetime(2026, 8, 13, tzinfo=UTC),
            source_kind="file",
            source_ref="fixture.mp4",
            media=EventMedia(str(clip), (str(frame),)),
            storage_root=root,
            max_active=4,
        )
        claimed = queue.claim_next(lease_seconds=10, max_attempts=2)
        assert claimed is not None
        queue.mark_completed(event_id)

    @staticmethod
    def _run(arguments: list[str]) -> tuple[int, str, str]:
        output = StringIO()
        error = StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(arguments)
        return code, output.getvalue(), error.getvalue()

    def test_worker_reuses_one_resident_process_for_two_completed_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = root / "events.sqlite3"
            runner = root / "runner.py"
            counter = root / "counter.txt"
            runner.write_text(_RUNNER, encoding="utf-8")
            runner.chmod(0o600)
            config = root / "config.toml"
            command = json.dumps([sys.executable, str(runner), str(counter)])
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
                        'predicates = ["face_visibility"]',
                        'hardware_signal_adapters = ["home_assistant"]',
                        "min_person_confidence = 0.25",
                        'model_id = "microsoft.Mage-VL"',
                        f'model_revision = "{_REVISION}"',
                        'prompt_version = "mage_atomic_visible_v1"',
                        "max_jobs = 4",
                        "idle_exit_seconds = 30.0",
                        "max_runtime_seconds = 300.0",
                        "startup_timeout_seconds = 5.0",
                        "request_timeout_seconds = 5.0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            self._complete_analysis_job(root, database, "event_01")
            self._complete_analysis_job(root, database, "event_02")

            code, output, error = self._run(
                [
                    "visual-predicate-shadow-worker",
                    "--config",
                    str(config),
                    "--max-jobs",
                    "2",
                    "--max-runtime-seconds",
                    "10",
                    "--poll-seconds",
                    "0.01",
                ]
            )
            self.assertEqual(0, code, error)
            payload = json.loads(output)
            self.assertEqual(2, payload["worker"]["completed_jobs"])
            self.assertEqual("max_jobs", payload["worker"]["stop_reason"])
            self.assertFalse(payload["os_network_isolation_enforced"])
            self.assertFalse(payload["risk_inference"])
            self.assertNotIn(str(root), output)
            self.assertEqual(["start", "stop"], counter.read_text(encoding="utf-8").splitlines())

            connection = sqlite3.connect(database)
            self.assertEqual(
                2,
                connection.execute(
                    "SELECT COUNT(*) FROM visual_predicate_shadow_attempts "
                    "WHERE status = 'completed'"
                ).fetchone()[0],
            )
            self.assertEqual(
                2,
                connection.execute("SELECT COUNT(*) FROM visual_predicate_shadow_runs").fetchone()[
                    0
                ],
            )
            connection.close()
            for event_id in ("event_01", "event_02"):
                job = AnalysisQueue(database).get_read_only(event_id)
                assert job is not None
                self.assertEqual(AnalysisJobStatus.COMPLETED, job.status)


if __name__ == "__main__":
    unittest.main()
