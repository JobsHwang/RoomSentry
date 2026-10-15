from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.domain import CandidateEvent, Detection
from baby_monitor.visual_predicate import VisualPredicate
from baby_monitor.visual_predicate_shadow import select_visual_predicate_shadow
from baby_monitor.visual_predicate_shadow_queue import VisualPredicateShadowAttemptQueue


class VisualPredicateShadowQueueCLITests(unittest.TestCase):
    @staticmethod
    def _config(root: Path, database: Path) -> Path:
        config = root / "config.toml"
        config.write_text(
            "\n".join(
                (
                    "[storage]",
                    f'database = "{database.as_posix()}"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    @staticmethod
    def _gate():
        candidate = CandidateEvent(
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
            detector_id="yolo11n.sha256_fixture",
        )
        return select_visual_predicate_shadow(
            candidate,
            predicates=(VisualPredicate.FACE_VISIBILITY,),
            person_detector_id=candidate.detector_id,
        )

    @staticmethod
    def _run(arguments: list[str]) -> tuple[int, str, str]:
        output = StringIO()
        error = StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(arguments)
        return code, output.getvalue(), error.getvalue()

    def test_init_status_and_retry_are_path_free_and_shadow_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = root / "events.sqlite3"
            config = self._config(root, database)
            code, output, error = self._run(["init-db", "--config", str(config)])
            self.assertEqual(0, code, error)
            self.assertIn("visual_predicate_shadow_attempts", json.loads(output)["schemas"])

            queue = VisualPredicateShadowAttemptQueue(database)
            now = datetime(2026, 8, 13, tzinfo=UTC)
            queue.enqueue(self._gate(), max_active=2, now=now)
            attempt = queue.claim_next(lease_seconds=10, max_attempts=1, now=now)
            assert attempt is not None
            queue.mark_failed_attempt(
                attempt.event_id,
                error_code="assessment_failed",
                retry_delay_seconds=0,
                max_attempts=1,
                now=now,
            )

            code, output, error = self._run(
                [
                    "visual-predicate-shadow-queue-status",
                    "--config",
                    str(config),
                ]
            )
            self.assertEqual(0, code, error)
            status = json.loads(output)
            self.assertEqual(1, status["queue"]["failed"])
            self.assertFalse(status["paths_disclosed"])
            self.assertNotIn(str(root), output)

            code, output, error = self._run(
                [
                    "visual-predicate-shadow-report",
                    "--config",
                    str(config),
                    "--limit",
                    "10",
                ]
            )
            self.assertEqual(0, code, error)
            report = json.loads(output)
            self.assertEqual(1, report["observation_count"])
            self.assertEqual("failed", report["observations"][0]["attempt"]["status"])
            self.assertEqual([], report["observations"][0]["results"])
            self.assertFalse(report["media_read"])
            self.assertFalse(report["risk_inference"])
            self.assertNotIn(str(root), output)

            code, output, error = self._run(
                [
                    "visual-predicate-shadow-retry",
                    "--config",
                    str(config),
                    "--event-id",
                    "event_01",
                ]
            )
            self.assertEqual(0, code, error)
            retried = json.loads(output)
            self.assertEqual("pending", retried["attempt"]["status"])
            self.assertFalse(retried["risk_inference"])
            self.assertNotIn(str(root), output)
            connection = sqlite3.connect(database)
            main_job_count = connection.execute("SELECT COUNT(*) FROM analysis_jobs").fetchone()[0]
            connection.close()
            self.assertEqual(0, main_job_count)


if __name__ == "__main__":
    unittest.main()
