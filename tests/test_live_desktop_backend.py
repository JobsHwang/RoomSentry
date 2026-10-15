from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from baby_monitor.live_monitor import _subject_projection
from scripts import live_desktop_backend as desktop


class LiveDesktopBackendTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.now_monotonic = 100.0
        self.units = {u: {"state": "active", "pid": 42} for u in desktop.UNITS}
        self.status = {
            "worker_pid": 42,
            "boot_id": "synthetic_boot",
            "phase": "detecting",
            "heartbeat_at": self.now.isoformat(),
            "heartbeat_monotonic_seconds": self.now_monotonic,
            "review_phase": "analyzing",
            "review_successes": 7,
            "review_failures": 2,
            "detection_failures": 0,
            "cameras": {
                f"camera_{i:02}": {
                    "phase": "receiving",
                    "inferred_frames": 10,
                    "recent_inference_fps": 5,
                    "last_person_count": 1,
                    "review_evidence": {
                        "status": "stale_completed_review",
                        "evidence_age_seconds": 45,
                    },
                }
                for i in range(1, 5)
            },
            "review_storage": {"available": True, "used_bytes": 100, "limit_bytes": 1000},
            "private_text": "private synthetic material must not appear",
        }

    def snapshot(self, status=None):
        result = desktop.snapshot(
            self.units,
            self.status if status is None else status,
            boot_id="synthetic_boot",
            now=self.now,
            now_monotonic=self.now_monotonic,
        )
        self.assertFalse(result["accuracy_verified"])
        self.assertFalse(result["media_read"])
        self.assertEqual("unknown", result["baby_state"])
        self.assertEqual("unknown", result["baby_location"])
        self.assertNotIn("private synthetic", json.dumps(result))
        return result

    def test_running_summary_is_allowlisted_and_keeps_stale_review(self):
        result = self.snapshot()
        self.assertEqual("running", result["state"])
        self.assertEqual(40, result["detected_frames"])
        self.assertEqual(2, result["review_failures"])
        self.assertEqual("stale_completed_review", result["cameras"][0]["review_status"])

    def test_wrong_pid_never_imports_old_session(self):
        self.status["worker_pid"] = 41
        result = self.snapshot()
        self.assertEqual("starting", result["state"])
        self.assertEqual([], result["cameras"])

    def test_wrong_boot_never_imports_old_session_even_when_pid_reused(self):
        self.status["boot_id"] = "previous_boot"
        self.assertEqual("starting", self.snapshot()["state"])

    def test_missing_or_boolean_worker_pid_is_unknown(self):
        for value in (None, True):
            self.status["worker_pid"] = value
            self.assertEqual("starting", self.snapshot()["state"])

    def test_stopped_and_partial_do_not_display_cached_telemetry(self):
        self.units[desktop.WORKER] = {"state": "inactive", "pid": 0}
        result = self.snapshot()
        self.assertEqual("partial", result["state"])
        self.assertIsNone(result["review_successes"])
        self.units[desktop.GATEWAY] = {"state": "inactive", "pid": 0}
        self.assertEqual("stopped", self.snapshot()["state"])

    def test_stale_heartbeat_does_not_make_service_active_equal_pipeline_healthy(self):
        self.status["heartbeat_at"] = (self.now - timedelta(seconds=11)).isoformat()
        result = self.snapshot()
        self.assertEqual("degraded", result["state"])
        self.assertEqual("heartbeat_stale", result["status_reason"])
        self.assertEqual([], result["cameras"])

    def test_future_naive_and_invalid_heartbeat_fail_closed(self):
        for value in (
            (self.now + timedelta(seconds=1)).isoformat(),
            "2026-01-01T00:00:00",
            None,
            "garbage",
        ):
            with self.subTest(value=value):
                self.status["heartbeat_at"] = value
                self.assertEqual("heartbeat_invalid", self.snapshot()["status_reason"])

    def test_backward_wall_step_does_not_extend_heartbeat(self):
        self.status["heartbeat_at"] = (self.now - timedelta(seconds=5)).isoformat()
        self.status["heartbeat_monotonic_seconds"] = self.now_monotonic - 15
        result = self.snapshot()
        self.assertEqual("heartbeat_stale", result["status_reason"])
        self.assertEqual(15, result["heartbeat_age_seconds"])
        self.assertEqual([], result["cameras"])

    def test_missing_invalid_and_future_elapsed_heartbeat_fail_closed(self):
        for value in (None, True, -1, "100", float("nan"), float("inf"), 101, 10**400):
            with self.subTest(value_type=type(value).__name__):
                self.status["heartbeat_monotonic_seconds"] = value
                result = self.snapshot()
                self.assertEqual("heartbeat_clock_unavailable", result["status_reason"])
                self.assertEqual([], result["cameras"])
        del self.status["heartbeat_monotonic_seconds"]
        self.assertEqual("heartbeat_clock_unavailable", self.snapshot()["status_reason"])

    def test_invalid_elapsed_reader_clock_fails_closed(self):
        for value in (True, -1, "100", float("nan"), float("inf")):
            self.now_monotonic = value
            self.assertEqual("heartbeat_clock_unavailable", self.snapshot()["status_reason"])

    def test_review_deferred_is_visible_as_degraded(self):
        for phase in ("storage_deferred", "memory_deferred", "worker_failed", None):
            self.status["review_phase"] = phase
            self.assertEqual("degraded", self.snapshot()["state"])

    def test_disconnected_camera_is_degraded(self):
        self.status["cameras"]["camera_02"]["phase"] = "reconnecting"
        self.assertEqual("camera_not_receiving", self.snapshot()["status_reason"])

    def test_invalid_counts_not_printed_as_numbers(self):
        for value in (True, -1, "private synthetic", float("nan"), float("inf"), 10**400):
            with self.subTest(value_type=type(value).__name__):
                self.status["review_successes"] = value
                self.assertIsNone(self.snapshot()["review_successes"])

    def test_nonallowlisted_review_text_never_reaches_output(self):
        self.status["cameras"]["camera_01"]["review_evidence"]["status"] = "private synthetic"
        self.assertEqual("unknown", self.snapshot()["cameras"][0]["review_status"])

    def fake_runner(self, *, active):
        def run(*args, **kwargs):
            if args[0] == "show":
                state = "active" if active else "inactive"
                pid = 42 if active else 0
                return f"LoadState=loaded\nActiveState={state}\nMainPID={pid}\n"
            return ""

        return Mock(side_effect=run)

    def test_status_only_queries_two_allowlisted_units(self):
        runner = self.fake_runner(active=True)
        desktop.control("status", runner)
        self.assertEqual(2, runner.call_count)
        self.assertTrue(all(call.args[0] == "show" for call in runner.call_args_list))

    def test_start_orders_gateway_before_worker_and_verifies(self):
        runner = self.fake_runner(active=True)
        desktop.control("start", runner)
        self.assertEqual(("start", desktop.GATEWAY), runner.call_args_list[0].args)
        self.assertEqual(("start", desktop.WORKER), runner.call_args_list[1].args)
        self.assertFalse(any("enable" in call.args for call in runner.call_args_list))

    def test_stop_only_scoped_units_and_verifies_inactive_pids(self):
        runner = self.fake_runner(active=False)
        desktop.control("stop", runner)
        self.assertEqual(("stop", desktop.WORKER, desktop.GATEWAY), runner.call_args_list[0].args)
        self.assertEqual(240, runner.call_args_list[0].kwargs["timeout"])

    def test_unconfirmed_start_or_stop_is_error(self):
        for action, active in (("start", False), ("stop", True)):
            with self.subTest(action=action), self.assertRaises(desktop.DesktopControlError):
                desktop.control(action, self.fake_runner(active=active))

    def test_arbitrary_actions_rejected_before_execution(self):
        runner = Mock()
        with self.assertRaises(desktop.DesktopControlError):
            desktop.control("restart-unrelated", runner)
        runner.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "private status checks require POSIX")
    def test_private_status_file_limits_duplicate_keys_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = root / "status.json"
            path.touch(mode=0o600)
            path.write_text(json.dumps(self.status))
            self.assertEqual(self.status, desktop.read_private_status(path))
            for payload in ('{"a":1,"a":2}', '{"a":NaN}', "[" * 1000, "x" * 65537):
                path.write_text(payload)
                with self.assertRaises((ValueError, RecursionError)):
                    desktop.read_private_status(path)
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                desktop.read_private_status(link)

    @unittest.skipUnless(os.name == "posix", "private status checks require POSIX")
    def test_public_status_permissions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = root / "status.json"
            path.touch(mode=0o644)
            path.write_text("{}")
            with self.assertRaises(ValueError):
                desktop.read_private_status(path)

    def test_snapshot_does_not_modify_source(self):
        before = deepcopy(self.status)
        self.snapshot()
        self.assertEqual(before, self.status)

    def subject_fixture(self):
        self.status["review_task"] = "subjects"
        data = self.status["cameras"]["camera_01"]
        data.update(generation=2, subject_claims=[{"age_claim": "baby", "head_center": [100, 200]}])
        data["review_evidence"] = {
            "status": "recent_completed_review",
            "evidence_age_seconds": 8,
            "frame_received_at": (self.now - timedelta(seconds=8)).isoformat(),
            "frame_generation": 2,
            "model_output_verified": False,
            "live_baby_state_established": False,
        }
        return data

    def test_subject_projection_is_unverified_and_does_not_leak_geometry(self):
        self.subject_fixture()
        result = self.snapshot()["cameras"][0]["current_subjects"]
        self.assertEqual(["baby"], result["age_claims"])
        self.assertEqual("unverified_model_claims", result["semantics"])
        self.assertNotIn("head_center", result)

    def test_subject_projection_ages_between_heartbeat_and_read(self):
        data = self.subject_fixture()
        self.status["heartbeat_at"] = (self.now - timedelta(seconds=4)).isoformat()
        self.assertEqual(
            12, self.snapshot()["cameras"][0]["current_subjects"]["evidence_age_seconds"]
        )
        data["review_evidence"]["evidence_age_seconds"] = 27
        self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])

    def test_wall_receipt_age_expires_young_monotonic_claim(self):
        data = self.subject_fixture()
        data["review_evidence"]["frame_received_at"] = (
            self.now - timedelta(seconds=30.001)
        ).isoformat()
        result = self.snapshot()
        self.assertIsNone(result["cameras"][0]["current_subjects"])
        self.assertEqual([], result["baby_hint_summary"]["hints"])

    def test_both_clocks_bound_displayed_age_without_sliding_expiry(self):
        data = self.subject_fixture()
        data["review_evidence"]["frame_received_at"] = (
            self.now - timedelta(seconds=30)
        ).isoformat()
        self.assertEqual(
            30, self.snapshot()["cameras"][0]["current_subjects"]["evidence_age_seconds"]
        )
        self.status["heartbeat_monotonic_seconds"] -= 4
        data["review_evidence"]["evidence_age_seconds"] = 27
        self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])

    def test_elapsed_age_wins_after_utc_rollback(self):
        self.subject_fixture()
        self.status["heartbeat_monotonic_seconds"] -= 4
        self.assertEqual(
            12, self.snapshot()["cameras"][0]["current_subjects"]["evidence_age_seconds"]
        )

    def test_cross_stage_clock_disagreement_conservatively_expires_early(self):
        data = self.subject_fixture()
        # Receipt->publish ages: mono 25, UTC 21. Publish->read: mono 1,
        # UTC 6. The conservative sum is 31, even though endpoints are 26/27.
        data["review_evidence"]["evidence_age_seconds"] = 25
        data["review_evidence"]["frame_received_at"] = (
            self.now - timedelta(seconds=27)
        ).isoformat()
        self.status["heartbeat_at"] = (self.now - timedelta(seconds=6)).isoformat()
        self.status["heartbeat_monotonic_seconds"] -= 1
        self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])

    def test_new_heartbeat_after_pause_does_not_refresh_old_receipt(self):
        data = self.subject_fixture()
        attempt = {
            "status": "analyzed",
            "frame_received_at": (self.now - timedelta(seconds=81)).isoformat(),
            "received_monotonic": self.now_monotonic - 21,
            "frame_generation": 2,
            "completed_at": self.now.isoformat(),
        }
        data.update(
            _subject_projection(
                attempt,
                None,
                data["subject_claims"],
                now=self.now_monotonic,
                now_utc=self.now,
                source_phase="receiving",
                source_generation=2,
                monitor_phase="detecting",
            )
        )
        self.assertEqual("stale_completed_review", data["subject_evidence"]["status"])
        self.assertIsNone(data["subject_claims"])
        self.status["review_task"] = "hybrid"
        result = self.snapshot()
        self.assertEqual("current_session", result["status_reason"])
        self.assertIsNone(result["cameras"][0]["current_subjects"])
        self.assertEqual([], result["baby_hint_summary"]["hints"])

    def test_private_elapsed_anchor_is_not_exported(self):
        self.assertNotIn("heartbeat_monotonic_seconds", json.dumps(self.snapshot()))

    def test_hybrid_requires_independent_subject_evidence_not_recent_event(self):
        data = self.subject_fixture()
        self.status["review_task"] = "hybrid"
        self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])
        data["subject_evidence"] = deepcopy(data["review_evidence"])
        self.assertIsNotNone(self.snapshot()["cameras"][0]["current_subjects"])
        data["subject_evidence"]["status"] = "latest_analysis_failed"
        self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])

    def test_subject_projection_rejects_stale_failure_reconnect_and_bad_contract(self):
        for key, value in (
            ("status", "analyzing"),
            ("status", "latest_analysis_failed"),
            ("status", "stale_completed_review"),
            ("frame_generation", 1),
            ("frame_generation", True),
            ("model_output_verified", True),
            ("frame_received_at", "invalid"),
            ("frame_received_at", (self.now + timedelta(seconds=1)).isoformat()),
        ):
            data = self.subject_fixture()
            data["review_evidence"][key] = value
            self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])
        for task in ("event", None, "untrusted"):
            self.subject_fixture()
            self.status["review_task"] = task
            self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])

    def test_subject_projection_rejects_unbounded_or_private_claims(self):
        for claims in (
            None,
            [{}],
            [{"age_claim": "private synthetic", "head_center": None}],
            [{"age_claim": "baby", "head_center": None, "text": "private synthetic"}],
            [{"age_claim": "baby", "head_center": [True, 0]}],
            [{"age_claim": "baby", "head_center": None}] * 9,
        ):
            data = self.subject_fixture()
            data["subject_claims"] = claims
            self.assertIsNone(self.snapshot()["cameras"][0]["current_subjects"])
        data["subject_claims"] = []
        self.assertEqual([], self.snapshot()["cameras"][0]["current_subjects"]["age_claims"])


if __name__ == "__main__":
    unittest.main()
