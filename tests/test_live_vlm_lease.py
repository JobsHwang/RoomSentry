"""CPU-only fault tests for an inherited, process-tree-wide VLM lease."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters import vlm_execution as gate

_GRANDCHILD = r"""
import json, os, select, sys
lease, ready, release = map(int, sys.argv[1:])
os.fstat(lease)
os.write(ready, (json.dumps({'kind': 'grandchild', 'pid': os.getpid()}) + '\n').encode())
os.close(ready)
# Bounded even if the test is interrupted before learning our PID.
select.select([release], [], [], 15)
os.read(release, 1) if select.select([release], [], [], 0)[0] else None
os.close(lease)
os.close(release)
"""

_CHILD = r"""
import json, os, select, subprocess, sys
lease, ready, release, grand_release = map(int, sys.argv[1:5])
grandchild_code = sys.argv[5]
grandchild = subprocess.Popen(
    [sys.executable, '-c', grandchild_code, str(lease), str(ready), str(grand_release)],
    pass_fds=(lease, ready, grand_release),
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
os.close(grand_release)
os.write(ready, (json.dumps({'kind': 'child', 'pid': os.getpid(),
                           'grandchild_pid': grandchild.pid}) + '\n').encode())
try:
    select.select([release], [], [], 15)
    os.read(release, 1) if select.select([release], [], [], 0)[0] else None
    os.close(lease)
    os.write(ready, b'{"kind":"child_released"}\n')
    grandchild.wait(timeout=17)
    os.write(ready, b'{"kind":"child_finished"}\n')
finally:
    if grandchild.poll() is None:
        grandchild.kill()
        grandchild.wait(timeout=3)
    os.close(ready)
    os.close(release)
"""

_OWNER = r"""
import json, os, select, subprocess, sys
from pathlib import Path
from baby_monitor.adapters import vlm_execution as gate
root, ready, release, grand_release = sys.argv[1:5]
ready, release, grand_release = map(int, (ready, release, grand_release))
gate._execution_root = lambda: Path(root)
with gate.local_vlm_execution('0') as lease:
    child = subprocess.Popen(
        [sys.executable, '-c', sys.argv[5], str(lease), str(ready),
         str(release), str(grand_release), sys.argv[6]],
        pass_fds=(lease, ready, release, grand_release),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.write(ready, (json.dumps({'kind': 'owner', 'child_pid': child.pid}) + '\n').encode())
    os.close(ready)
    os.close(release)
    os.close(grand_release)
    # The test sends SIGKILL; this bound also prevents a forgotten owner process.
    child.wait(timeout=20)
"""


@unittest.skipUnless(os.name == "posix", "POSIX inherited flock leases are required")
class LiveVLMLeaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-lease-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        replacement = patch.object(gate, "_execution_root", return_value=self.root)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.descriptors = []
        self.processes = []
        self.child_pids = set()
        self.messages = []
        self.buffer = b""
        self.ready_read, self.ready_write = self._pipe()
        self.child_read, self.child_write = self._pipe()
        self.grand_read, self.grand_write = self._pipe()
        self.addCleanup(self._cleanup_tree)

    def _pipe(self):
        pair = os.pipe()
        self.descriptors.extend(pair)
        return pair

    def _release(self, descriptor):
        with suppress(OSError):
            os.write(descriptor, b"x")

    def _cleanup_tree(self):
        self._release(self.child_write)
        self._release(self.grand_write)
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        # Only exact PIDs reported by our synthetic subprocesses are targeted.
        for pid in self.child_pids:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        for descriptor in self.descriptors:
            with suppress(OSError):
                os.close(descriptor)

    def _message(self, kind):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for message in self.messages:
                if message["kind"] == kind:
                    return message
            ready, _, _ = select.select([self.ready_read], [], [], 0.05)
            if not ready:
                continue
            data = os.read(self.ready_read, 4096)
            self.assertTrue(data, "synthetic lease process closed before readiness")
            self.buffer += data
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                message = json.loads(line)
                self.messages.append(message)
                for key in ("pid", "child_pid", "grandchild_pid"):
                    if key in message:
                        self.child_pids.add(message[key])
        self.fail(f"synthetic lease process did not send {kind}")

    def _spawn_child(self, lease):
        self.assertIs(type(lease), int, "the gate must yield its inheritable lease descriptor")
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD,
                str(lease),
                str(self.ready_write),
                str(self.child_read),
                str(self.grand_read),
                _GRANDCHILD,
            ],
            pass_fds=(lease, self.ready_write, self.child_read, self.grand_read),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.processes.append(process)
        self._message("child")
        self._message("grandchild")
        return process

    def _assert_busy(self):
        with self.assertRaises(gate.VLMExecutionError) as caught, gate.local_vlm_execution("0"):
            self.fail("a descendant still holds the model lease")
        self.assertEqual("local_vlm_busy", caught.exception.code)

    def _assert_eventually_available(self):
        deadline = time.monotonic() + 5
        while True:
            try:
                with gate.local_vlm_execution("0") as descriptor:
                    self.assertIs(type(descriptor), int)
                    return
            except gate.VLMExecutionError as error:
                if error.code != "local_vlm_busy" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def test_normal_context_exit_keeps_child_and_grandchild_lease(self):
        with gate.local_vlm_execution("0") as lease:
            child = self._spawn_child(lease)
            self._assert_busy()
        with self.assertRaises(OSError):
            os.fstat(lease)
        self._assert_busy()
        self._release(self.child_write)
        self._message("child_released")
        self._assert_busy()  # Now only the model-like grandchild holds the lease.
        self._release(self.grand_write)
        self.assertEqual(0, child.wait(timeout=5))
        self.child_pids.clear()  # Child has reaped its own grandchild.
        self._assert_eventually_available()

    def test_killed_owner_cannot_unlock_live_model_grandchild(self):
        environment = {
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        owner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _OWNER,
                str(self.root),
                str(self.ready_write),
                str(self.child_read),
                str(self.grand_read),
                _CHILD,
                _GRANDCHILD,
            ],
            pass_fds=(self.ready_write, self.child_read, self.grand_read),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        self.processes.append(owner)
        self._message("owner")
        self._message("child")
        self._message("grandchild")
        owner.kill()  # Exact PID, deliberately simulating loss of the gate owner.
        self.assertEqual(-signal.SIGKILL, owner.wait(timeout=5))
        self._assert_busy()
        self._release(self.child_write)
        self._message("child_released")
        self._assert_busy()
        self._release(self.grand_write)
        self._assert_eventually_available()
        self._message("child_finished")
        self.child_pids.clear()  # Grandchild was reaped before the orphaned child exits.

    def test_single_process_exit_and_exception_still_release_lease(self):
        for fail in (False, True):
            with self.subTest(exception=fail):
                try:
                    with gate.local_vlm_execution("0") as descriptor:
                        self.assertIs(type(descriptor), int)
                        self.assertFalse(os.get_inheritable(descriptor))
                        self._assert_busy()
                        if fail:
                            raise ValueError("synthetic body failure")
                except ValueError:
                    if not fail:
                        raise
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
                self._assert_eventually_available()


if __name__ == "__main__":
    unittest.main()
