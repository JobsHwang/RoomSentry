from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.candidate_health import (
    CandidateHealthError,
    CandidateHealthFreshness,
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateSamplingHealth,
    CandidateWorkerHealth,
    assess_candidate_health,
)
from baby_monitor.cli import main
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger


class CandidateHealthTests(unittest.TestCase):
    NOW = datetime(2026, 8, 19, 8, 0, tzinfo=UTC)

    def _ledger(self) -> DetectorSamplingLedger:
        return DetectorSamplingLedger(
            native_fps=2,
            native_fps_source="reported",
            requested_sample_fps=1,
            frame_step=2,
            decoded_frame_count=5,
            samples=(
                DetectorSampleFrame(0, 0.0, "container_pts", 0, 0),
                DetectorSampleFrame(2, 1.0, "container_pts", 0, 1),
                DetectorSampleFrame(4, 2.0, "container_pts", 1, 0),
            ),
            timestamp_fallback_frame_count=1,
            max_adjacent_container_pts_gap_seconds=3.07,
            container_pts_regression_count=1,
        )

    def _v4_health(self, *, sampling: bool = True) -> CandidateWorkerHealth:
        return replace(
            self._health(),
            version=4,
            lifetime=CandidateHealthLifetime.begin_session(None, updated_at=self.NOW),
            last_sampling=CandidateSamplingHealth.from_ledger(self._ledger()) if sampling else None,
        )

    def test_sampling_summary_retains_actual_counts_and_uncertainty(self) -> None:
        sampling = CandidateSamplingHealth.from_ledger(self._ledger())
        self.assertEqual(
            {
                "sampled_frame_count": 3,
                "zero_detection_sample_count": 2,
                "decoded_frame_count": 5,
                "timestamp_fallback_frame_count": 1,
                "max_adjacent_container_pts_gap_seconds": 3.07,
                "container_pts_regression_count": 1,
                "end_reason": "read_returned_false",
                "source_exhaustion_verified": False,
            },
            sampling.to_dict(),
        )
        self.assertEqual(sampling, CandidateSamplingHealth.from_dict(sampling.to_dict()))
        self.assertNotIn("samples", sampling.to_dict())

    def test_sampling_all_zero_frames_are_evidence_not_missing_data(self) -> None:
        ledger = self._ledger()
        empty = replace(
            ledger,
            samples=tuple(replace(s, detection_start=0, detection_count=0) for s in ledger.samples),
        )
        sampling = CandidateSamplingHealth.from_ledger(empty)
        self.assertEqual(3, sampling.zero_detection_sample_count)
        self.assertFalse(sampling.source_exhaustion_verified)
        self.assertIsNone(self._v4_health(sampling=False).to_dict()["last_sampling"])
        self.assertNotIn("presence", sampling.to_dict())

    def test_sampling_all_fallback_and_single_frame_have_unknown_pts_gap(self) -> None:
        ledger = DetectorSamplingLedger(
            native_fps=25,
            native_fps_source="fallback_25",
            requested_sample_fps=1,
            frame_step=25,
            decoded_frame_count=1,
            samples=(DetectorSampleFrame(0, 0.0, "nominal_fps", 0, 0),),
            timestamp_fallback_frame_count=1,
            max_adjacent_container_pts_gap_seconds=None,
            container_pts_regression_count=0,
        )
        sampling = CandidateSamplingHealth.from_ledger(ledger)
        self.assertIsNone(sampling.max_adjacent_container_pts_gap_seconds)
        self.assertEqual(1, sampling.sampled_frame_count)

    def test_sampling_summary_rejects_invalid_fields_and_counts(self) -> None:
        payload = CandidateSamplingHealth.from_ledger(self._ledger()).to_dict()
        for field in (
            "sampled_frame_count",
            "zero_detection_sample_count",
            "decoded_frame_count",
            "timestamp_fallback_frame_count",
            "container_pts_regression_count",
        ):
            for invalid in (True, -1, 1.5, "1", None, 1_000_000_001):
                with (
                    self.subTest(field=field, invalid=invalid),
                    self.assertRaises(CandidateHealthError),
                ):
                    CandidateSamplingHealth.from_dict({**payload, field: invalid})
        for field, invalid in (
            ("sampled_frame_count", 0),
            ("sampled_frame_count", 100_001),
            ("sampled_frame_count", 6),
            ("zero_detection_sample_count", 4),
            ("decoded_frame_count", 10_000_001),
            ("timestamp_fallback_frame_count", 6),
            ("container_pts_regression_count", 4),
            ("end_reason", "eof"),
            ("source_exhaustion_verified", True),
            ("source_exhaustion_verified", 0),
        ):
            with (
                self.subTest(field=field, invalid=invalid),
                self.assertRaises(CandidateHealthError),
            ):
                CandidateSamplingHealth.from_dict({**payload, field: invalid})
        for invalid in (None, [], {}, {**payload, "private_path": "/not-disclosed"}):
            with self.assertRaises(CandidateHealthError):
                CandidateSamplingHealth.from_dict(invalid)
        for field in payload:
            with self.assertRaises(CandidateHealthError):
                CandidateSamplingHealth.from_dict({k: v for k, v in payload.items() if k != field})
        with self.assertRaises(CandidateHealthError):
            CandidateSamplingHealth.from_ledger(None)

    def test_sampling_summary_rejects_invalid_or_unsupported_pts_gaps(self) -> None:
        payload = CandidateSamplingHealth.from_ledger(self._ledger()).to_dict()
        for invalid in (True, -1, "1", [], float("nan"), float("inf"), 10**1000):
            with self.subTest(invalid=invalid), self.assertRaises(CandidateHealthError):
                CandidateSamplingHealth.from_dict(
                    {**payload, "max_adjacent_container_pts_gap_seconds": invalid}
                )
        with self.assertRaises(CandidateHealthError):
            CandidateSamplingHealth.from_dict(
                {
                    **payload,
                    "timestamp_fallback_frame_count": 4,
                    "container_pts_regression_count": 0,
                }
            )

    def test_v4_private_round_trip_and_schema_have_required_nullable_sampling(self) -> None:
        schema = json.loads(
            Path("schemas/candidate-worker-health-v4.schema.json").read_text(encoding="utf-8")
        )
        for enabled in (False, True):
            health = self._v4_health(sampling=enabled)
            self.assertEqual(set(schema["required"]), set(health.to_dict()))
            self.assertEqual(health, CandidateWorkerHealth.from_dict(health.to_dict()))
            if enabled:
                self.assertEqual(
                    set(schema["$defs"]["sampling"]["required"]),
                    set(health.to_dict()["last_sampling"]),
                )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(0o700)
                path = root / "candidate-health.json"
                health.write_atomic_private(path)
                self.assertEqual(0o600, path.stat().st_mode & 0o777)
                self.assertEqual(health, CandidateWorkerHealth.read_private(path))
                self.assertNotIn(str(root), json.dumps(health.to_dict()))
        payload = self._v4_health().to_dict()
        del payload["last_sampling"]
        with self.assertRaises(CandidateHealthError):
            CandidateWorkerHealth.from_dict(payload)

    def test_v4_assessment_exposes_sampling_without_adding_authority(self) -> None:
        health = self._v4_health()
        assessment = assess_candidate_health(health, checked_at=self.NOW, stale_after_seconds=30)
        payload = assessment.to_dict()
        self.assertEqual(health.last_sampling.to_dict(), payload["health"]["last_sampling"])
        self.assertFalse(payload["paths_disclosed"])
        for flag in ("identity_inference", "risk_assessment", "notification_authority"):
            self.assertFalse(payload["health"][flag])

    def test_v4_sampling_cannot_be_attached_to_missing_or_failed_scan(self) -> None:
        health = self._v4_health()
        with self.assertRaises(CandidateHealthError):
            replace(
                health,
                scans_succeeded=0,
                last_scan_completed_at=None,
                last_scan_duration_seconds=None,
                last_execution_lock_wait_seconds=None,
            )
        with self.assertRaises(CandidateHealthError):
            replace(
                health,
                state=CandidateHealthState.FAILED,
                worker_running=False,
                last_error_code="candidate_scan_failed",
            )
        with self.assertRaises(CandidateHealthError):
            replace(health, last_sampling={})
        with self.assertRaises(CandidateHealthError):
            replace(health, lifetime=None)

    def test_legacy_versions_keep_exact_shape_without_fabricated_sampling(self) -> None:
        for version in (1, 2, 3):
            health = replace(
                self._health(),
                version=version,
                lifetime=(
                    CandidateHealthLifetime.begin_session(None, updated_at=self.NOW)
                    if version > 1
                    else None
                ),
            )
            payload = health.to_dict()
            self.assertNotIn("last_sampling", payload)
            self.assertIsNone(CandidateWorkerHealth.from_dict(payload).last_sampling)
            self.assertEqual(payload, CandidateWorkerHealth.from_dict(payload).to_dict())
            with self.assertRaises(CandidateHealthError):
                CandidateWorkerHealth.from_dict({**payload, "last_sampling": None})
            with self.assertRaises(CandidateHealthError):
                replace(health, last_sampling=CandidateSamplingHealth.from_ledger(self._ledger()))
        for invalid in (True, 4.0, "4", [], None):
            with self.assertRaises(CandidateHealthError):
                CandidateWorkerHealth.from_dict({**self._v4_health().to_dict(), "version": invalid})

    def _health(
        self,
        *,
        camera_id: str = "camera_01",
        checked_at: datetime | None = None,
        state: CandidateHealthState = CandidateHealthState.IDLE,
    ) -> CandidateWorkerHealth:
        running = state in {
            CandidateHealthState.STARTING,
            CandidateHealthState.SCANNING,
            CandidateHealthState.IDLE,
        }
        failed = state is CandidateHealthState.FAILED
        return CandidateWorkerHealth(
            camera_id=camera_id,
            scanner_identity="candidate_v1_fixture",
            detector_identity="yolo_command_v1_fixture",
            checked_at=checked_at or self.NOW,
            state=state,
            worker_running=running,
            scans_started=1,
            scans_succeeded=0 if failed else 1,
            scans_failed=1 if failed else 0,
            segments_scanned=0 if failed else 1,
            detections=0,
            emitted_events=0,
            queue_full_events=0,
            last_scan_started_at=(checked_at or self.NOW) - timedelta(seconds=2),
            last_scan_completed_at=None if failed else (checked_at or self.NOW),
            last_scan_duration_seconds=None if failed else 2.0,
            last_execution_lock_wait_seconds=None if failed else 0.25,
            last_error_code="candidate_scan_failed" if failed else None,
        )

    def test_private_round_trip_schema_and_permissions(self) -> None:
        health = self._health()
        schema = json.loads(
            Path("schemas/candidate-worker-health-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(health.to_dict()))
        self.assertEqual(health, CandidateWorkerHealth.from_dict(health.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = root / "candidate-health.json"
            health.write_atomic_private(path)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            self.assertEqual(health, CandidateWorkerHealth.read_private(path))
            self.assertEqual([], list(root.glob(".candidate-health.json.*.tmp")))

            path.chmod(0o644)
            with self.assertRaisesRegex(CandidateHealthError, "unsafe"):
                CandidateWorkerHealth.read_private(path)

    def test_assessment_distinguishes_fresh_stale_clock_skew_and_stopped(self) -> None:
        fresh = assess_candidate_health(
            self._health(),
            checked_at=self.NOW + timedelta(seconds=10),
            stale_after_seconds=30,
        )
        self.assertEqual(CandidateHealthFreshness.FRESH, fresh.freshness)
        self.assertTrue(fresh.operational)

        stale = assess_candidate_health(
            self._health(),
            checked_at=self.NOW + timedelta(seconds=31),
            stale_after_seconds=30,
        )
        self.assertEqual(CandidateHealthFreshness.STALE, stale.freshness)
        self.assertIn("candidate_health_stale", stale.reason_codes)

        skew = assess_candidate_health(
            self._health(checked_at=self.NOW + timedelta(seconds=10)),
            checked_at=self.NOW,
            stale_after_seconds=30,
            future_tolerance_seconds=5,
        )
        self.assertEqual(CandidateHealthFreshness.CLOCK_SKEW, skew.freshness)

        stopped = assess_candidate_health(
            self._health(state=CandidateHealthState.STOPPED),
            checked_at=self.NOW,
            stale_after_seconds=30,
        )
        self.assertFalse(stopped.operational)
        self.assertIn("candidate_worker_not_running", stopped.reason_codes)

    def test_v2_v3_lifetime_is_bounded_strict_and_migrates(self) -> None:
        lifetime = CandidateHealthLifetime.begin_session(None, updated_at=self.NOW)
        for duration, wait in ((2.5, 0.2), (4.5, 1.5), (9.0, 4.7)):
            lifetime = lifetime.record_scan(
                updated_at=self.NOW,
                succeeded=True,
                duration_seconds=duration,
                execution_lock_wait_seconds=wait,
            )
        lifetime = lifetime.finish_session(updated_at=self.NOW, failed=False)
        health = CandidateWorkerHealth(
            version=2,
            camera_id="camera_01",
            scanner_identity="candidate_v1_fixture",
            detector_identity="yolo_command_v1_fixture",
            checked_at=self.NOW,
            state=CandidateHealthState.STOPPED,
            worker_running=False,
            scans_started=3,
            scans_succeeded=3,
            segments_scanned=3,
            last_scan_started_at=self.NOW - timedelta(seconds=9),
            last_scan_completed_at=self.NOW,
            last_scan_duration_seconds=9.0,
            last_execution_lock_wait_seconds=4.7,
            lifetime=lifetime,
        )
        schema = json.loads(
            Path("schemas/candidate-worker-health-v2.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(health.to_dict()))
        self.assertEqual(health, CandidateWorkerHealth.from_dict(health.to_dict()))
        summary = lifetime.summary()
        self.assertEqual(3, summary["scan_duration_sample_count"])
        self.assertEqual(9.0, summary["scan_duration_p95_upper_bound_seconds"])
        self.assertEqual(4.7, summary["execution_lock_wait_p95_upper_bound_seconds"])
        self.assertEqual(4.7, summary["execution_lock_wait_max_seconds"])
        self.assertNotIn("camera_id", summary)

        v3_lifetime = lifetime.record_candidate_groups(
            updated_at=self.NOW,
            emitted=2,
            suppressed=3,
        )
        v3_health = replace(
            health,
            version=3,
            emitted_events=2,
            suppressed_candidate_groups=3,
            lifetime=v3_lifetime,
        )
        v3_schema = json.loads(
            Path("schemas/candidate-worker-health-v3.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(v3_schema["required"]), set(v3_health.to_dict()))
        self.assertEqual(
            v3_health,
            CandidateWorkerHealth.from_dict(v3_health.to_dict()),
        )
        self.assertEqual(2, v3_lifetime.summary()["candidate_groups_emitted"])
        self.assertEqual(
            3,
            v3_lifetime.summary()["candidate_groups_suppressed"],
        )
        tampered_v3 = v3_health.to_dict()
        tampered_v3["lifetime"]["candidate_groups_suppressed"] = 2
        with self.assertRaisesRegex(CandidateHealthError, "inconsistent"):
            CandidateWorkerHealth.from_dict(tampered_v3)

        legacy_lifetime = CandidateHealthLifetime.from_dict(
            v3_lifetime.to_dict(include_candidate_groups=False)
        )
        self.assertEqual(0, legacy_lifetime.candidate_groups_emitted)
        self.assertEqual(0, legacy_lifetime.candidate_groups_suppressed)
        migrated_v2 = CandidateHealthLifetime.begin_session(
            replace(health, emitted_events=4),
            updated_at=self.NOW + timedelta(seconds=1),
        )
        self.assertEqual(4, migrated_v2.candidate_groups_emitted)

        restarted = CandidateHealthLifetime.begin_session(
            CandidateWorkerHealth(
                version=2,
                camera_id="camera_01",
                scanner_identity="candidate_v1_fixture",
                detector_identity="yolo_command_v1_fixture",
                checked_at=self.NOW,
                state=CandidateHealthState.IDLE,
                worker_running=True,
                lifetime=CandidateHealthLifetime.begin_session(
                    None,
                    updated_at=self.NOW,
                ),
            ),
            updated_at=self.NOW + timedelta(seconds=1),
        )
        self.assertEqual(2, restarted.sessions_started)
        self.assertEqual(1, restarted.unclean_restarts)

        migrated = CandidateHealthLifetime.begin_session(
            self._health(state=CandidateHealthState.STOPPED),
            updated_at=self.NOW + timedelta(seconds=1),
        )
        self.assertEqual(2, migrated.sessions_started)
        self.assertEqual(1, migrated.sessions_stopped)
        self.assertEqual(1, migrated.scans_succeeded)
        self.assertEqual(1, migrated.scan_duration_sample_count)

    def test_cli_aggregates_multiple_health_files_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            paths: list[Path] = []
            for camera_id in ("camera_01", "camera_02"):
                path = root / f"{camera_id}.json"
                health = self._health(
                    camera_id=camera_id,
                    checked_at=datetime.now(UTC),
                )
                health.write_atomic_private(path)
                paths.append(path)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "candidate-health-status",
                        "--health",
                        str(paths[0]),
                        "--health",
                        str(paths[1]),
                        "--stale-after-seconds",
                        "30",
                    ]
                )
            payload = json.loads(output.getvalue())
            status_schema = json.loads(
                Path("schemas/candidate-health-status-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(status_schema["required"]), set(payload))
            self.assertEqual(0, code)
            self.assertEqual(2, len(payload["cameras"]))
            self.assertTrue(payload["operational"])
            self.assertFalse(payload["media_read"])
            self.assertNotIn(str(root), output.getvalue())

            duplicate = root / "duplicate.json"
            self._health(checked_at=datetime.now(UTC)).write_atomic_private(duplicate)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                duplicate_code = main(
                    [
                        "candidate-health-status",
                        "--health",
                        str(paths[0]),
                        "--health",
                        str(duplicate),
                    ]
                )
            self.assertEqual(2, duplicate_code)
            self.assertIn("must be unique", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
