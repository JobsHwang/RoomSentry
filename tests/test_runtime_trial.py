from __future__ import annotations

import io
import json
import sqlite3
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.candidate_health import (
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateWorkerHealth,
)
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.notification import NotificationOutbox
from baby_monitor.presence_history import PresenceFactStore
from baby_monitor.room_presence import StreamPresenceSample, StreamPresenceSampleState
from baby_monitor.runtime_trial import (
    RuntimeTrialError,
    finish_runtime_trial,
    inspect_runtime_trial,
    start_runtime_trial,
)


class RuntimeTrialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.database = self.state / "events.sqlite3"
        EventDatabase(self.database).initialize()
        AnalysisQueue(self.database).initialize()
        NotificationOutbox(self.database).initialize()
        PresenceFactStore(self.database).initialize()
        self.database.chmod(0o600)
        self.ledger = self.state / "runtime-trial.json"
        self.started_at = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
        self.health_paths = tuple(
            self._write_running_health(camera_id) for camera_id in ("camera_01", "camera_02")
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _health_path(self, camera_id: str) -> Path:
        directory = self.root / camera_id
        directory.mkdir(mode=0o700, exist_ok=True)
        return directory / "health.json"

    def _write_running_health(self, camera_id: str) -> Path:
        lifetime = CandidateHealthLifetime.begin_session(
            None,
            updated_at=self.started_at,
        )
        health = CandidateWorkerHealth(
            version=3,
            camera_id=camera_id,
            scanner_identity="runtime_scanner_v1",
            detector_identity="runtime_detector_v1",
            checked_at=self.started_at,
            state=CandidateHealthState.IDLE,
            worker_running=True,
            lifetime=lifetime,
        )
        path = self._health_path(camera_id)
        health.write_atomic_private(path)
        return path

    def _write_stopped_health(
        self,
        camera_id: str,
        *,
        ended_at: datetime,
        failed_scan: bool = False,
    ) -> None:
        previous = CandidateWorkerHealth.read_private(self._health_path(camera_id))
        assert previous.lifetime is not None
        lifetime = previous.lifetime
        lifetime = lifetime.record_scan(
            updated_at=ended_at - timedelta(seconds=3),
            succeeded=True,
            duration_seconds=2.0,
            execution_lock_wait_seconds=0.1,
        )
        if failed_scan:
            lifetime = lifetime.record_scan(
                updated_at=ended_at - timedelta(seconds=2),
                succeeded=False,
                duration_seconds=1.5,
                execution_lock_wait_seconds=0.2,
            )
        lifetime = lifetime.finish_session(updated_at=ended_at, failed=False)
        CandidateWorkerHealth(
            version=3,
            camera_id=camera_id,
            scanner_identity="runtime_scanner_v1",
            detector_identity="runtime_detector_v1",
            checked_at=ended_at,
            state=CandidateHealthState.STOPPED,
            worker_running=False,
            scans_started=2 if failed_scan else 1,
            scans_succeeded=1,
            scans_failed=int(failed_scan),
            segments_scanned=2 if failed_scan else 1,
            last_scan_started_at=ended_at - timedelta(seconds=4),
            last_scan_completed_at=ended_at - timedelta(seconds=3),
            last_scan_duration_seconds=2.0,
            last_execution_lock_wait_seconds=0.1,
            lifetime=lifetime,
        ).write_atomic_private(self._health_path(camera_id))

    def _insert_analysis(
        self,
        *,
        event_id: str,
        status: str,
        updated_at: datetime,
        error_code: str | None = None,
    ) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    event_id, camera_id, occurred_at, source_kind, source_ref,
                    candidate_json, clip_path, frame_paths_json, evidence_json,
                    storage_root, status, attempt_count, available_at, lease_until,
                    last_error_code, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    "camera_01",
                    updated_at.isoformat(),
                    "fixture",
                    "fixture",
                    json.dumps({}),
                    str(self.root / "fixture.mp4"),
                    "[]",
                    None,
                    str(self.root),
                    status,
                    1,
                    updated_at.isoformat(),
                    None,
                    error_code,
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                    updated_at.isoformat() if status == "completed" else None,
                ),
            )

    def _insert_notification(
        self,
        *,
        notification_id: str,
        event_id: str,
        status: str,
        updated_at: datetime,
        error_code: str | None = None,
    ) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO notification_jobs (
                    notification_id, event_id, sink_kind, sink_id, payload_json,
                    status, attempt_count, available_at, lease_until, last_error_code,
                    created_at, updated_at, delivered_at
                ) VALUES (?, ?, 'file', 'local', '{}', ?, 1, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    notification_id,
                    event_id,
                    status,
                    updated_at.isoformat(),
                    error_code,
                    updated_at.isoformat(),
                    updated_at.isoformat(),
                    updated_at.isoformat() if status == "delivered" else None,
                ),
            )

    def _record_primary(self, camera_id: str, observed_at: datetime) -> None:
        PresenceFactStore(self.database).record(
            StreamPresenceSample(
                sample_id=f"primary_{camera_id}_{int(observed_at.timestamp())}",
                camera_id=camera_id,
                observed_at=observed_at,
                state=StreamPresenceSampleState.CLEAR,
                scanner_id="runtime_scanner_v1",
                detector_id="runtime_detector_v1",
                source_fingerprint="a" * 64,
            ),
            recorded_at=observed_at,
        )

    def test_historical_failures_do_not_stain_a_successful_trial(self) -> None:
        self._insert_analysis(
            event_id="historical_failure",
            status="failed",
            updated_at=self.started_at - timedelta(hours=1),
            error_code="queue_full",
        )
        started = start_runtime_trial(
            ledger_path=self.ledger,
            health_paths=self.health_paths,
            database_path=self.database,
            started_at=self.started_at,
            expected_duration_seconds=3_600,
            shadow_expected=True,
        )
        self.assertEqual("running", started["latest"]["state"])
        self.assertEqual(1, started["latest"]["database_baseline"]["analysis_failed"])
        self.assertEqual(0o600, stat.S_IMODE(self.ledger.stat().st_mode))

        ended_at = self.started_at + timedelta(hours=1)
        for camera_id in ("camera_01", "camera_02"):
            self._write_stopped_health(camera_id, ended_at=ended_at)
            self._record_primary(camera_id, ended_at - timedelta(minutes=1))
        self._insert_analysis(
            event_id="completed_during_trial",
            status="completed",
            updated_at=ended_at - timedelta(minutes=2),
        )
        self._insert_notification(
            notification_id="delivered_during_trial",
            event_id="completed_during_trial",
            status="delivered",
            updated_at=ended_at - timedelta(minutes=1),
        )

        finished = finish_runtime_trial(
            ledger_path=self.ledger,
            health_paths=self.health_paths,
            database_path=self.database,
            ended_at=ended_at,
        )
        latest = finished["latest"]
        self.assertEqual("completed", latest["state"])
        self.assertFalse(finished["attention_required"])
        self.assertEqual([], latest["attention_codes"])
        self.assertEqual(["shadow_camera_missing"], latest["advisory_codes"])
        self.assertEqual(1, latest["result"]["analysis_completed"])
        self.assertEqual(0, latest["result"]["analysis_failed"])
        self.assertEqual(2, latest["result"]["primary_camera_count"])
        self.assertEqual(2, latest["result"]["primary_sample_count"])
        self.assertEqual(2, sum(item["result"]["scans_succeeded"] for item in latest["cameras"]))

        reread = inspect_runtime_trial(
            ledger_path=self.ledger,
            checked_at=ended_at + timedelta(minutes=1),
        )
        self.assertEqual(finished, reread)

    def test_new_failures_and_missing_primary_camera_require_attention(self) -> None:
        start_runtime_trial(
            ledger_path=self.ledger,
            health_paths=self.health_paths,
            database_path=self.database,
            started_at=self.started_at,
            expected_duration_seconds=3_600,
        )
        ended_at = self.started_at + timedelta(hours=1)
        self._write_stopped_health("camera_01", ended_at=ended_at, failed_scan=True)
        self._write_stopped_health("camera_02", ended_at=ended_at)
        self._record_primary("camera_01", ended_at - timedelta(minutes=1))
        self._insert_analysis(
            event_id="new_analysis_failure",
            status="failed",
            updated_at=ended_at - timedelta(minutes=2),
            error_code="runtime_failure",
        )
        self._insert_notification(
            notification_id="new_notification_failure",
            event_id="new_analysis_failure",
            status="failed",
            updated_at=ended_at - timedelta(minutes=1),
            error_code="sink_failure",
        )

        finished = finish_runtime_trial(
            ledger_path=self.ledger,
            health_paths=self.health_paths,
            database_path=self.database,
            ended_at=ended_at,
        )
        latest = finished["latest"]
        self.assertTrue(finished["attention_required"])
        self.assertEqual(
            [
                "candidate_runtime_failure",
                "analysis_failure",
                "notification_failure",
                "primary_camera_missing",
            ],
            latest["attention_codes"],
        )
        self.assertEqual(
            {"runtime_failure": 1},
            latest["result"]["analysis_failed_error_counts"],
        )
        self.assertEqual(
            {"sink_failure": 1},
            latest["result"]["notification_failed_error_counts"],
        )

    def test_start_rejects_stale_stopped_health_and_weak_database(self) -> None:
        stopped_at = self.started_at - timedelta(minutes=5)
        for camera_id in ("camera_01", "camera_02"):
            self._write_stopped_health(camera_id, ended_at=stopped_at)
        with self.assertRaisesRegex(RuntimeTrialError, "fresh and running"):
            start_runtime_trial(
                ledger_path=self.ledger,
                health_paths=self.health_paths,
                database_path=self.database,
                started_at=self.started_at,
                expected_duration_seconds=3_600,
            )

    def test_cli_forwards_health_wait_and_shadow_expectation(self) -> None:
        with patch(
            "baby_monitor.cli_commands.candidate_runtime.start_runtime_trial",
            return_value={"attention_required": False},
        ) as starter:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "runtime-trial-start",
                        "--ledger",
                        str(self.ledger),
                        "--health",
                        str(self.health_paths[0]),
                        "--database",
                        str(self.database),
                        "--expected-duration-seconds",
                        "3600",
                        "--health-wait-seconds",
                        "17.5",
                        "--shadow-expected",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual({"attention_required": False}, json.loads(output.getvalue()))
        call = starter.call_args.kwargs
        self.assertEqual(17.5, call["health_wait_seconds"])
        self.assertTrue(call["shadow_expected"])
        self.assertEqual((str(self.health_paths[0]),), call["health_paths"])

        self.database.chmod(0o644)
        self.health_paths = tuple(
            self._write_running_health(camera_id) for camera_id in ("camera_01", "camera_02")
        )
        with self.assertRaisesRegex(RuntimeTrialError, "database is unavailable"):
            start_runtime_trial(
                ledger_path=self.ledger,
                health_paths=self.health_paths,
                database_path=self.database,
                started_at=self.started_at,
                expected_duration_seconds=3_600,
            )


if __name__ == "__main__":
    unittest.main()
