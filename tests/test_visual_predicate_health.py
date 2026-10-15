from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    ResidentVisualPredicateHealth,
    VisualPredicateHealthError,
)
from baby_monitor.cli import main
from baby_monitor.visual_predicate_health import (
    ResidentVisualPredicateFreshness,
    assess_resident_visual_predicate_health,
)


class ResidentVisualPredicateHealthAssessmentTests(unittest.TestCase):
    @staticmethod
    def _health(
        *,
        checked_at: datetime,
        state: str = "ready",
        worker_running: bool = True,
        failed_jobs: int = 0,
        timeout_jobs: int = 0,
        last_error_code: str | None = None,
    ) -> ResidentVisualPredicateHealth:
        return ResidentVisualPredicateHealth(
            checked_at=checked_at,
            state=state,
            worker_running=worker_running,
            worker_starts=1,
            worker_restarts=0,
            successful_jobs=4,
            failed_jobs=failed_jobs,
            timeout_jobs=timeout_jobs,
            current_worker_jobs=4 if worker_running else 0,
            max_jobs=32,
            idle_exit_seconds=30.0,
            max_runtime_seconds=3600.0,
            model_id="mage-vl-4b-int8",
            model_revision="a" * 40,
            prompt_version="visual-predicate-v1",
            last_error_code=last_error_code,
        )

    def test_only_fresh_ready_or_busy_is_operational(self) -> None:
        now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
        for state in ("ready", "busy"):
            with self.subTest(state=state):
                assessment = assess_resident_visual_predicate_health(
                    self._health(checked_at=now - timedelta(seconds=2), state=state),
                    checked_at=now,
                )
                self.assertTrue(assessment.operational)
                self.assertEqual(ResidentVisualPredicateFreshness.FRESH, assessment.freshness)
                self.assertEqual(f"fresh_{state}", assessment.reason_code)

        non_operational = (
            self._health(
                checked_at=now,
                state="stopped",
                worker_running=False,
            ),
            self._health(checked_at=now, state="starting"),
            self._health(
                checked_at=now,
                state="failed",
                worker_running=False,
                failed_jobs=1,
                timeout_jobs=1,
                last_error_code="worker_timeout",
            ),
        )
        for health in non_operational:
            with self.subTest(state=health.state):
                assessment = assess_resident_visual_predicate_health(
                    health,
                    checked_at=now,
                )
                self.assertFalse(assessment.operational)
                self.assertEqual(f"fresh_{health.state}", assessment.reason_code)

    def test_stale_and_clock_skew_fail_closed(self) -> None:
        now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
        stale = assess_resident_visual_predicate_health(
            self._health(checked_at=now - timedelta(seconds=61)),
            checked_at=now,
            stale_after_seconds=60,
        )
        self.assertFalse(stale.operational)
        self.assertEqual(ResidentVisualPredicateFreshness.STALE, stale.freshness)
        self.assertEqual("resident_health_stale", stale.reason_code)

        future = assess_resident_visual_predicate_health(
            self._health(checked_at=now + timedelta(seconds=6)),
            checked_at=now,
            future_tolerance_seconds=5,
        )
        self.assertFalse(future.operational)
        self.assertEqual(ResidentVisualPredicateFreshness.CLOCK_SKEW, future.freshness)
        self.assertEqual("health_checked_at_in_future", future.reason_code)

    def test_thresholds_and_checked_at_are_strict(self) -> None:
        health = self._health(checked_at=datetime.now(UTC))
        for value in (True, 0, math.nan, math.inf, 604_801):
            with (
                self.subTest(stale_after_seconds=value),
                self.assertRaises(VisualPredicateHealthError),
            ):
                assess_resident_visual_predicate_health(
                    health,
                    stale_after_seconds=value,
                )
        with self.assertRaises(VisualPredicateHealthError):
            assess_resident_visual_predicate_health(
                health,
                checked_at=datetime(2026, 8, 13),
            )

    def test_schema_required_fields_match_canonical_output(self) -> None:
        now = datetime.now(UTC)
        payload = assess_resident_visual_predicate_health(
            self._health(checked_at=now),
            checked_at=now,
        ).to_dict()
        schema_path = (
            Path(__file__).parents[1]
            / "schemas"
            / "resident-visual-predicate-health-assessment-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(payload))
        self.assertFalse(payload["paths_disclosed"])
        self.assertFalse(payload["media_metadata_included"])

    def test_cli_is_read_only_path_free_and_returns_attention_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            health_path = root / "resident-health.json"
            health = self._health(checked_at=datetime.now(UTC))
            health.write_atomic_private(health_path)
            before = health_path.read_bytes()
            before_stat = health_path.stat()

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "visual-predicate-health-status",
                        "--health",
                        str(health_path),
                        "--stale-after-seconds",
                        "600",
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertTrue(payload["assessment"]["operational"])
            self.assertFalse(payload["health_path_disclosed"])
            self.assertNotIn(str(health_path), output.getvalue())
            self.assertEqual(before, health_path.read_bytes())
            after_stat = health_path.stat()
            self.assertEqual(before_stat.st_ino, after_stat.st_ino)
            self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)

            stopped = self._health(
                checked_at=datetime.now(UTC),
                state="stopped",
                worker_running=False,
            )
            stopped.write_atomic_private(health_path)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "visual-predicate-health-status",
                        "--health",
                        str(health_path),
                    ]
                )
            self.assertEqual(1, code)
            self.assertEqual(
                "fresh_stopped", json.loads(output.getvalue())["assessment"]["reason_code"]
            )

    def test_cli_missing_file_error_does_not_disclose_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            missing = root / "private-resident-health.json"
            error = io.StringIO()
            with redirect_stderr(error):
                code = main(
                    [
                        "visual-predicate-health-status",
                        "--health",
                        str(missing),
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual(
                "error: resident health file is unavailable\n",
                error.getvalue(),
            )
            self.assertNotIn(str(missing), error.getvalue())


if __name__ == "__main__":
    unittest.main()
