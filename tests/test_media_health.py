from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.media_health import (
    MediaHealthError,
    MediaHealthFreshness,
    MediaSourceHealth,
    MediaSourceState,
    assess_media_health,
)
from baby_monitor.ring_buffer import RTSPSegmentBuffer, SegmentBufferSettings


class MediaSourceHealthTests(unittest.TestCase):
    @staticmethod
    def _health(*, state: MediaSourceState = MediaSourceState.RUNNING) -> MediaSourceHealth:
        started = datetime(2026, 8, 3, tzinfo=UTC)
        terminal = state.terminal
        return MediaSourceHealth(
            camera_id="cam_01",
            source_ref="rtsp://127.0.0.1:8554/test/camera",
            state=state,
            observed_at=started + timedelta(seconds=3),
            session_started_at=started,
            session_ended_at=(started + timedelta(seconds=3) if terminal else None),
            segments=2,
            coverage_seconds=3.8,
            gap_count=1,
            reconnect_count=1,
            capacity_reached=state is MediaSourceState.CAPACITY_REACHED,
            ffmpeg_exit_code=0,
            last_error_code=None,
        )

    def test_round_trip_atomic_file_and_public_schema_states(self) -> None:
        health = self._health(state=MediaSourceState.STOPPED)
        self.assertEqual(health, MediaSourceHealth.from_dict(health.to_dict()))
        schema = json.loads(
            Path("schemas/media-source-health-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {state.value for state in MediaSourceState},
            set(schema["properties"]["state"]["enum"]),
        )
        self.assertFalse(schema["additionalProperties"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            health.write_atomic(path)
            self.assertEqual(health, MediaSourceHealth.read(path))
            self.assertEqual([], list(path.parent.glob(".health.json.*.tmp")))

    def test_credentials_extra_fields_and_inconsistent_terminal_state_fail(self) -> None:
        value = self._health().to_dict()
        # release-audit: allow=fixture-credential
        value["source_ref"] = "rtsp://user:password@127.0.0.1:8554/test?token=x"
        with self.assertRaisesRegex(MediaHealthError, "redacted"):
            MediaSourceHealth.from_dict(value)

        value = self._health().to_dict()
        value["secret"] = "must-not-be-stored"
        with self.assertRaisesRegex(MediaHealthError, "fields"):
            MediaSourceHealth.from_dict(value)

        value = self._health().to_dict()
        value["camera_id"] = 123
        with self.assertRaisesRegex(MediaHealthError, "camera_id"):
            MediaSourceHealth.from_dict(value)

        value = self._health().to_dict()
        value["state"] = "stopped"
        with self.assertRaisesRegex(MediaHealthError, "session_ended_at"):
            MediaSourceHealth.from_dict(value)

    def test_missing_ffmpeg_writes_a_terminal_failed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = SegmentBufferSettings(
                uri="rtsp://127.0.0.1:8554/test/camera",
                camera_id="cam_01",
                buffer_root=root,
                run_seconds=3,
                ffmpeg=str(root / "missing-ffmpeg"),
            )
            with self.assertRaises(FileNotFoundError):
                RTSPSegmentBuffer(settings).run()
            health = MediaSourceHealth.read(settings.health_path)
            self.assertEqual(MediaSourceState.FAILED, health.state)
            self.assertTrue(health.state.terminal)
            self.assertEqual("FileNotFoundError", health.last_error_code)
            self.assertEqual("rtsp://127.0.0.1:8554/test/camera", health.source_ref)
            self.assertIsNotNone(health.session_ended_at)

    def test_assessment_separates_operational_stale_terminal_and_clock_skew(self) -> None:
        checked = datetime(2026, 8, 3, 0, 0, 20, tzinfo=UTC)

        running = self._health()
        fresh = assess_media_health(
            running,
            checked_at=running.observed_at + timedelta(seconds=2),
        )
        self.assertEqual(MediaHealthFreshness.FRESH, fresh.freshness)
        self.assertTrue(fresh.operational)

        stale = assess_media_health(running, checked_at=checked)
        self.assertEqual(MediaHealthFreshness.STALE, stale.freshness)
        self.assertFalse(stale.operational)
        self.assertEqual("active_snapshot_stale", stale.reason_code)

        terminal_health = self._health(state=MediaSourceState.STOPPED)
        terminal = assess_media_health(terminal_health, checked_at=checked)
        self.assertEqual(MediaHealthFreshness.TERMINAL, terminal.freshness)
        self.assertEqual("terminal_stopped", terminal.reason_code)

        skewed = assess_media_health(
            running,
            checked_at=running.observed_at - timedelta(seconds=6),
        )
        self.assertEqual(MediaHealthFreshness.CLOCK_SKEW, skewed.freshness)
        self.assertEqual("observed_at_in_future", skewed.reason_code)

    def test_assessment_rejects_invalid_thresholds(self) -> None:
        with self.assertRaisesRegex(MediaHealthError, "stale_after_seconds"):
            assess_media_health(self._health(), stale_after_seconds=0)
        with self.assertRaisesRegex(MediaHealthError, "future_tolerance_seconds"):
            assess_media_health(self._health(), future_tolerance_seconds=-1)

    def test_status_cli_returns_zero_only_for_fresh_running_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            now = datetime.now(UTC)
            running = MediaSourceHealth(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test/camera",
                state=MediaSourceState.RUNNING,
                observed_at=now,
                session_started_at=now - timedelta(seconds=2),
                session_ended_at=None,
                segments=1,
                coverage_seconds=1.0,
                gap_count=0,
                reconnect_count=0,
                capacity_reached=False,
                ffmpeg_exit_code=None,
                last_error_code=None,
            )
            running.write_atomic(path)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["media-health-status", "--health", str(path)])
            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertTrue(payload["assessment"]["operational"])
            self.assertEqual("fresh", payload["assessment"]["freshness"])

            stopped_value = running.to_dict()
            stopped_value["state"] = MediaSourceState.STOPPED.value
            stopped_value["session_ended_at"] = now.isoformat()
            stopped = MediaSourceHealth.from_dict(stopped_value)
            stopped.write_atomic(path)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["media-health-status", "--health", str(path)])
            payload = json.loads(output.getvalue())
            self.assertEqual(1, code)
            self.assertFalse(payload["assessment"]["operational"])
            self.assertEqual("terminal", payload["assessment"]["freshness"])


if __name__ == "__main__":
    unittest.main()
