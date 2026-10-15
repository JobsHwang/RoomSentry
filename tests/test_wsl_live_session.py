from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import hold_wsl_live_session as session


class FakeStop:
    def __init__(self, interrupt_after=100):
        self.calls = 0
        self.interrupt_after = interrupt_after

    def is_set(self):
        return self.calls >= self.interrupt_after

    def wait(self, seconds):
        if seconds != 2:
            raise AssertionError("unexpected wait interval")
        self.calls += 1


class WslLiveSessionTests(unittest.TestCase):
    def test_stopped_units_exit_after_grace_without_starting_anything(self):
        reader = Mock(return_value=("inactive", "failed"))
        result = session.hold_session(
            FakeStop(), read_states=reader, clock=Mock(side_effect=(0, 5, 15))
        )
        self.assertEqual("live_units_stopped", result)
        self.assertEqual(3, reader.call_count)

    def test_either_running_unit_holds_session_until_interrupt(self):
        for states in (("active", "inactive"), ("failed", "active"), ("activating", "reloading")):
            with self.subTest(states=states):
                result = session.hold_session(
                    FakeStop(3), read_states=lambda states=states: states, clock=lambda: 100
                )
                self.assertEqual("session_interrupted", result)

    def test_restart_clears_inactive_grace(self):
        reader = Mock(
            side_effect=(
                ("inactive", "inactive"),
                ("active", "inactive"),
                ("inactive", "inactive"),
                ("inactive", "inactive"),
            )
        )
        self.assertEqual(
            "live_units_stopped",
            session.hold_session(
                FakeStop(), read_states=reader, clock=Mock(side_effect=(0, 10, 20, 35))
            ),
        )

    def test_query_failure_is_not_treated_as_running(self):
        reader = Mock(side_effect=RuntimeError("private fixture detail"))
        with self.assertRaises(RuntimeError):
            session.hold_session(FakeStop(), read_states=reader)

    def test_unknown_or_missing_state_fails_closed(self):
        for states in (("active", "garbage"), ("active",)):
            with self.subTest(states=states), self.assertRaises(RuntimeError):
                session.hold_session(FakeStop(), read_states=lambda states=states: states)

    def test_systemctl_queries_only_fixed_units_and_do_not_mutate(self):
        with patch.object(
            session.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=b"active\n"),
        ) as run:
            self.assertEqual(("active", "active"), session.unit_states())
        self.assertEqual(2, run.call_count)
        for call, unit in zip(run.call_args_list, session.UNITS, strict=True):
            self.assertEqual(
                ["/usr/bin/systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
                call.args[0],
            )
            self.assertEqual(5, call.kwargs["timeout"])

    def test_nonzero_query_or_unknown_output_fails_closed(self):
        for result in (
            SimpleNamespace(returncode=1, stdout=b"active"),
            SimpleNamespace(returncode=0, stdout=b"private fixture detail"),
        ):
            with (
                self.subTest(result=result),
                patch.object(session.subprocess, "run", return_value=result),
                self.assertRaises(RuntimeError),
            ):
                session.unit_states()

    @unittest.skipUnless(os.name == "posix", "WSL session lock requires POSIX")
    def test_native_lock_is_exclusive_and_released_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            first = session.acquire_session_lock(root)
            try:
                self.assertIsNone(session.acquire_session_lock(root))
            finally:
                os.close(first)
            second = session.acquire_session_lock(root)
            os.close(second)
            self.assertEqual(
                0o600, (root / "baby-monitor-live-session.lock").stat().st_mode & 0o777
            )

    @unittest.skipUnless(os.name == "posix", "WSL session lock requires POSIX")
    def test_symlink_lock_is_rejected_and_target_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            target = root / "fixture"
            target.write_text("synthetic fixture")
            (root / "baby-monitor-live-session.lock").symlink_to(target)
            with self.assertRaises(OSError):
                session.acquire_session_lock(root)
            self.assertEqual("synthetic fixture", target.read_text())

    @unittest.skipUnless(os.name == "posix", "WSL session lock requires POSIX")
    def test_public_runtime_and_lock_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o755)
            with self.assertRaises(RuntimeError):
                session.acquire_session_lock(root)
            root.chmod(0o700)
            lock = root / "baby-monitor-live-session.lock"
            lock.touch(mode=0o644)
            with self.assertRaises(RuntimeError):
                session.acquire_session_lock(root)

    def test_main_failure_does_not_echo_exception_detail(self):
        with (
            patch.object(
                session, "acquire_session_lock", side_effect=RuntimeError("private fixture detail")
            ),
            patch("builtins.print") as output,
        ):
            self.assertEqual(1, session.main())
        self.assertEqual('{"status": "session_unavailable"}', output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
