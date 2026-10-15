import signal
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from scripts import live_desktop_backend as desktop
from tests import test_live_desktop_backend as fixtures


class DesktopReviewControlTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.LiveDesktopBackendTests()
        fixture.setUp()
        fixture.subject_fixture()
        self.fixture = fixture
        self.status = fixture.status
        self.status["worker_start_ticks"] = 123
        self.status["review_control"] = {
            "enabled": True,
            "pause_requested": False,
            "resident_unloaded": False,
            "timeout_seconds": 600,
            "remaining_seconds": 0,
            "suspension_count": 0,
        }

    def test_paused_or_draining_hides_even_fresh_legacy_claims(self):
        for phase in ("suspending", "suspended"):
            self.status["review_phase"] = phase
            result = self.fixture.snapshot()
            self.assertEqual("local_review_suspended", result["status_reason"])
            self.assertEqual("degraded", result["state"])
            self.assertIsNone(result["cameras"][0]["current_subjects"])
            self.assertFalse(result["baby_hint_summary"]["coverage_complete"])

    def test_ack_flag_blocks_claims_even_if_phase_still_analyzing(self):
        self.status["review_control"]["resident_unloaded"] = True
        self.assertIsNone(self.fixture.snapshot()["cameras"][0]["current_subjects"])

    def test_invalid_control_record_is_unknown_not_assumed_unpaused(self):
        self.status["review_control"]["resident_unloaded"] = "invalid"
        result = self.fixture.snapshot()
        self.assertEqual("local_review_unavailable", result["status_reason"])
        self.assertIsNone(result["cameras"][0]["current_subjects"])

    def test_request_only_signals_verified_worker_and_does_not_stop_any_unit(self):
        for action, signum in (
            ("suspend-review", signal.SIGUSR1),
            ("resume-review", signal.SIGUSR2),
        ):
            sender = Mock()
            runner = self.fixture.fake_runner(active=True)
            desktop.control(
                action,
                runner,
                status_reader=lambda: self.status,
                signal_sender=sender,
                boot_id="synthetic_boot",
                now=self.fixture.now,
                now_monotonic=self.fixture.now_monotonic,
            )
            sender.assert_called_once_with(42, 123, signum)
            self.assertTrue(all(call.args[0] == "show" for call in runner.call_args_list))

    def test_unsupported_stale_wrong_session_and_boolean_capability_never_signal(self):
        for change in (
            {"review_control": {"enabled": False}},
            {"review_control": {"enabled": 1}},
            {"worker_pid": 41},
            {"boot_id": "old_boot"},
            {"heartbeat_at": "2025-01-01T00:00:00+00:00"},
            {"heartbeat_monotonic_seconds": None},
            {"heartbeat_monotonic_seconds": 89},
            {"worker_start_ticks": True},
        ):
            status = {**deepcopy(self.status), **change}
            sender = Mock()
            with self.assertRaises(desktop.DesktopControlError):
                desktop.control(
                    "suspend-review",
                    self.fixture.fake_runner(active=True),
                    status_reader=lambda status=status: status,
                    signal_sender=sender,
                    boot_id="synthetic_boot",
                    now=self.fixture.now,
                    now_monotonic=self.fixture.now_monotonic,
                )
            sender.assert_not_called()

    def test_closed_capability_rejects_extra_text_and_nonfinite_fields(self):
        for update in (
            {"remaining_seconds": float("nan")},
            {"remaining_seconds": 601},
            {"timeout_seconds": True},
            {"pause_requested": 1},
            {"private_text": "fixture"},
        ):
            self.assertIsNone(
                desktop.review_control_snapshot({**self.status["review_control"], **update})
            )

    def test_pidfd_identity_mismatch_closes_pin_without_sending(self):
        with (
            patch.object(desktop.os, "pidfd_open", return_value=99) as pin,
            patch.object(desktop, "process_start_ticks", return_value=124),
            patch.object(desktop.signal, "pidfd_send_signal") as send,
            patch.object(desktop.os, "close") as close,
        ):
            with self.assertRaises(desktop.DesktopControlError):
                desktop._signal_review_worker(42, 123, signal.SIGUSR1)
            pin.assert_called_once_with(42)
            send.assert_not_called()
            close.assert_called_once_with(99)

    def test_pidfd_pins_main_process_and_is_closed_after_delivery(self):
        with (
            patch.object(desktop.os, "pidfd_open", return_value=99),
            patch.object(desktop, "process_start_ticks", return_value=123),
            patch.object(desktop.signal, "pidfd_send_signal") as send,
            patch.object(desktop.os, "close") as close,
        ):
            desktop._signal_review_worker(42, 123, signal.SIGUSR2)
            send.assert_called_once_with(99, signal.SIGUSR2)
            close.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
