from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from baby_monitor.adapters import vlm_execution as gate
from baby_monitor.adapters.vlm import CommandVLMAdapter, VLMError, VLMRequest
from baby_monitor.domain import CandidateEvent, EventMedia


@unittest.skipUnless(os.name == "posix", "POSIX flock is required")
class VLMExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.default_execution_root = gate._execution_root
        temporary_root = patch.object(gate, "_execution_root", return_value=self.root)
        temporary_root.start()
        self.addCleanup(temporary_root.stop)
        self.directory = self.root / "baby-monitor-vlm-execution"
        self.lock = self.directory / "gpu_0.lock"

    @staticmethod
    def _request():
        return VLMRequest(
            CandidateEvent("fixture", "cam_01", 0, 1, ()), EventMedia("", ("frame.jpg",))
        )

    @staticmethod
    def _adapter(code="raise SystemExit(3)"):
        return CommandVLMAdapter(
            name="synthetic_execution",
            command=(sys.executable, "-c", code, "{request}"),
            timeout_seconds=2,
        )

    def test_same_gpu_is_busy_but_other_gpu_is_independent_and_lock_inode_persists(self):
        with gate.local_vlm_execution("0"):
            original = self.lock.stat().st_ino
            self.assertEqual(0o700, self.directory.stat().st_mode & 0o777)
            self.assertEqual(0o600, self.lock.stat().st_mode & 0o777)
            for alias in ("0", "00", "000"):
                with (
                    self.subTest(alias=alias),
                    self.assertRaises(gate.VLMExecutionError) as raised,
                    gate.local_vlm_execution(alias),
                ):
                    self.fail("second holder must not enter")
                self.assertEqual("local_vlm_busy", raised.exception.code)
            with gate.local_vlm_execution("1"):
                self.assertTrue((self.directory / "gpu_1.lock").is_file())
        with gate.local_vlm_execution("0"):
            self.assertEqual(original, self.lock.stat().st_ino)
        self.assertTrue(self.lock.exists())

    def test_exception_releases_lock_and_preserves_body_exception(self):
        for failure in (ValueError("synthetic"), KeyboardInterrupt(), SystemExit(7)):
            with (
                self.subTest(error=type(failure).__name__),
                self.assertRaises(type(failure)) as raised,
                gate.local_vlm_execution("0"),
            ):
                raise failure
            self.assertIs(failure, raised.exception)
            with gate.local_vlm_execution("0"):
                pass

    def test_busy_adapter_does_not_start_model_gpu_sampler_or_request_directory(self):
        adapter = self._adapter()
        with (
            gate.local_vlm_execution("0"),
            patch("baby_monitor.adapters.vlm._GPUDeviceMemorySampler") as sampler,
            patch("baby_monitor.adapters.vlm.subprocess.Popen") as popen,
            patch("baby_monitor.adapters.vlm.tempfile.TemporaryDirectory") as directory,
            self.assertRaises(VLMError) as raised,
        ):
            adapter.analyze_with_metrics(self._request())
        self.assertEqual("local_vlm_busy", raised.exception.code)
        self.assertEqual("local VLM execution is busy", str(raised.exception))
        sampler.assert_not_called()
        popen.assert_not_called()
        directory.assert_not_called()

    def test_directory_symlink_wrong_permissions_and_nonownership_are_rejected(self):
        target = self.root / "target"
        target.mkdir(mode=0o700)
        self.directory.symlink_to(target, target_is_directory=True)
        with self.assertRaises(gate.VLMExecutionError), gate.local_vlm_execution("0"):
            self.fail("symlink directory accepted")
        self.directory.unlink()
        self.directory.mkdir(mode=0o755)
        with self.assertRaises(gate.VLMExecutionError), gate.local_vlm_execution("0"):
            self.fail("public directory accepted")
        self.assertEqual(0o755, self.directory.stat().st_mode & 0o777)
        # A different effective ownership expectation cannot reuse a real user's files.
        with (
            patch.object(gate.os, "getuid", return_value=os.getuid() + 1),
            self.assertRaises(gate.VLMExecutionError),
            gate.local_vlm_execution("0"),
        ):
            self.fail("non-owned directory accepted")

    def test_lock_symlink_hardlink_nonregular_and_wrong_permissions_are_rejected(self):
        self.directory.mkdir(mode=0o700)
        target = self.root / "target"
        target.write_bytes(b"synthetic sentinel")
        target.chmod(0o600)
        for kind in ("symlink", "hardlink", "fifo", "permissions"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    self.lock.symlink_to(target)
                elif kind == "hardlink":
                    os.link(target, self.lock)
                elif kind == "fifo":
                    os.mkfifo(self.lock, mode=0o600)
                else:
                    self.lock.write_bytes(b"")
                    self.lock.chmod(0o644)
                with (
                    self.assertRaises(gate.VLMExecutionError) as raised,
                    gate.local_vlm_execution("0"),
                ):
                    self.fail("unsafe lock accepted")
                self.assertEqual("local_vlm_lock_unsafe", raised.exception.code)
                self.assertNotIn(str(self.root), str(raised.exception))
                self.assertEqual(b"synthetic sentinel", target.read_bytes())
                self.lock.unlink()

    def test_replaced_lock_between_open_and_identity_check_fails_and_closes_descriptors(self):
        replacement = self.root / "replacement"
        replacement.write_bytes(b"")
        replacement.chmod(0o600)
        original_open = gate.os.open
        descriptors = []

        def racing_open(path, *args, **kwargs):
            descriptor = original_open(path, *args, **kwargs)
            descriptors.append(descriptor)
            if path == "gpu_0.lock":
                replacement.replace(self.lock)
            return descriptor

        with (
            patch.object(gate.os, "open", side_effect=racing_open),
            self.assertRaises(gate.VLMExecutionError),
            gate.local_vlm_execution("0"),
        ):
            self.fail("replacement accepted")
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        with gate.local_vlm_execution("0"):
            pass

    def test_different_tmpdirs_in_one_process_cannot_change_gate(self):
        first, second = self.root / "tmp_one", self.root / "tmp_two"
        first.mkdir(mode=0o700)
        second.mkdir(mode=0o700)
        with patch.dict(os.environ, {"TMPDIR": str(first), "HOME": str(first)}):
            expected = Path("/run/user") / str(os.getuid())
            self.assertEqual(expected, self.default_execution_root())
            with (
                gate.local_vlm_execution("0"),
                patch.dict(os.environ, {"TMPDIR": str(second), "HOME": str(second)}),
            ):
                self.assertEqual(expected, self.default_execution_root())
                with (
                    self.assertRaises(gate.VLMExecutionError) as raised,
                    gate.local_vlm_execution("0"),
                ):
                    self.fail("TMPDIR changed the execution gate")
                self.assertEqual("local_vlm_busy", raised.exception.code)
        self.assertEqual([], list(first.iterdir()))
        self.assertEqual([], list(second.iterdir()))

    def test_missing_public_or_symlink_user_runtime_fails_without_fallback(self):
        missing = self.root / "missing_runtime"
        public = self.root / "public_runtime"
        public.mkdir(mode=0o755)
        linked = self.root / "linked_runtime"
        linked.symlink_to(self.root, target_is_directory=True)
        for runtime in (missing, public, linked):
            with (
                self.subTest(runtime=runtime.name),
                patch.object(gate, "_execution_root", return_value=runtime),
                self.assertRaises(gate.VLMExecutionError) as raised,
                gate.local_vlm_execution("0"),
            ):
                self.fail("unsafe user runtime accepted")
            self.assertEqual("local_vlm_lock_unsafe", raised.exception.code)
            self.assertNotIn(str(self.root), str(raised.exception))
        self.assertFalse(missing.exists())
        self.assertFalse(self.directory.exists())
        self.assertEqual([], list(public.iterdir()))

    def test_independent_process_with_different_tmpdir_cannot_enter_until_release(self):
        code = """
import sys
from pathlib import Path
from baby_monitor.adapters import vlm_execution as gate
gate._execution_root = lambda: Path(sys.argv[1])
try:
    with gate.local_vlm_execution("0"):
        print("acquired")
except gate.VLMExecutionError as exc:
    print(exc.code)
    raise SystemExit(3)
"""
        command = (sys.executable, "-c", code, str(self.root))
        parent_tmp, child_tmp = self.root / "parent_tmp", self.root / "child_tmp"
        parent_tmp.mkdir(mode=0o700)
        child_tmp.mkdir(mode=0o700)
        environment = {
            "PATH": os.defpath,
            "TMPDIR": str(child_tmp),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        with patch.dict(os.environ, {"TMPDIR": str(parent_tmp)}), gate.local_vlm_execution("0"):
            blocked = subprocess.run(
                command, env=environment, capture_output=True, text=True, timeout=5
            )
        self.assertEqual(3, blocked.returncode, blocked.stderr)
        self.assertEqual("local_vlm_busy", blocked.stdout.strip())
        allowed = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=5
        )
        self.assertEqual(0, allowed.returncode, allowed.stderr)
        self.assertEqual("acquired", allowed.stdout.strip())

    def test_actual_short_runner_failure_releases_gate_for_next_command_adapter(self):
        with self.assertRaises(VLMError) as raised:
            self._adapter().analyze(self._request())
        self.assertEqual("runner_failed", raised.exception.code)
        result = {
            "event_type": "unknown",
            "subjects": [],
            "observations": ["合成图形"],
            "inferences": [],
            "risk_level": "none",
            "confidence": 0.1,
            "needs_remote_review": True,
            "summary_zh": "仅为合成测试。",
            "evidence_frames": ["frame.jpg"],
        }
        adapter = self._adapter("print(" + repr(json.dumps(result)) + ")")
        self.assertEqual("unknown", adapter.analyze(self._request()).event_type.value)
        with gate.local_vlm_execution("0"):
            pass

    def test_double_interrupt_waits_for_reaping_with_gate_held_then_preserves_original(self):
        adapter = self._adapter()
        first, second = KeyboardInterrupt(), KeyboardInterrupt()
        process = MagicMock()
        process.communicate.side_effect = (first, second)
        waits = []

        def wait():
            with self.assertRaises(gate.VLMExecutionError) as raised, gate.local_vlm_execution("0"):
                self.fail("gate released while runner is being reaped")
            self.assertEqual("local_vlm_busy", raised.exception.code)
            waits.append(1)
            if len(waits) == 1:
                raise KeyboardInterrupt()
            return 0

        process.wait.side_effect = wait
        with (
            patch("baby_monitor.adapters.vlm.subprocess.Popen", return_value=process),
            patch.object(adapter, "_kill_process_tree") as kill,
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            adapter.analyze(self._request())
        self.assertIs(first, raised.exception)
        self.assertEqual(2, len(waits))
        kill.assert_called_once_with(process)
        with gate.local_vlm_execution("0"):
            pass

    def test_timeout_waits_for_runner_exit_before_releasing_gate(self):
        adapter = self._adapter()
        process = MagicMock()
        process.communicate.side_effect = (subprocess.TimeoutExpired("synthetic", 2), ("", ""))

        def wait():
            with self.assertRaises(gate.VLMExecutionError), gate.local_vlm_execution("0"):
                self.fail("gate released before timeout cleanup")
            return 0

        process.wait.side_effect = wait
        with (
            patch("baby_monitor.adapters.vlm.subprocess.Popen", return_value=process),
            patch.object(adapter, "_kill_process_tree") as kill,
            self.assertRaises(VLMError) as raised,
        ):
            adapter.analyze(self._request())
        self.assertEqual("runner_timeout", raised.exception.code)
        kill.assert_called_once_with(process)
        process.wait.assert_called_once()
        with gate.local_vlm_execution("0"):
            pass


if __name__ == "__main__":
    unittest.main()
