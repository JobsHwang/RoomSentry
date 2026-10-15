"""Keep a foreground WSL client attached while the existing live units run.

Launch from a hidden Windows wsl.exe process, NOT another systemd unit. This
does not start, restart, enable, or configure services. It exits after both live
units stop, on query failure, or on SIGTERM/SIGINT. No media or GPU access.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

UNITS = ("baby-monitor-yolo-bonsai-live.service", "xiaomi-stream-gateway.service")
_RUNNING = frozenset({"active", "activating", "reloading"})
_STATES = _RUNNING | {"inactive", "failed", "deactivating"}


def unit_states() -> tuple[str, str]:
    values = []
    for unit in UNITS:
        result = subprocess.run(
            ["/usr/bin/systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        value = result.stdout.strip().decode("ascii")
        if result.returncode != 0 or value not in _STATES:
            raise RuntimeError("live session unit query unavailable")
        values.append(value)
    return tuple(values)


def hold_session(
    stop: threading.Event,
    *,
    read_states: Callable[[], tuple[str, str]] = unit_states,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """A brief inactive grace allows an intentional service restart, not auto-recovery."""
    inactive_since = None
    while not stop.is_set():
        states = read_states()
        if len(states) != 2 or any(value not in _STATES for value in states):
            raise RuntimeError("live session unit query unavailable")
        now = clock()
        if any(value in _RUNNING for value in states):
            inactive_since = None
        elif inactive_since is None:
            inactive_since = now
        elif now - inactive_since >= 15:
            return "live_units_stopped"
        stop.wait(2)
    return "session_interrupted"


def acquire_session_lock(runtime_root: Path) -> int | None:
    """One keeper per user; never follow a symlink or delete an existing lock."""
    metadata = runtime_root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise RuntimeError("private live session runtime unavailable")
    descriptor = os.open(
        runtime_root / "baby-monitor-live-session.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise RuntimeError("private live session lock unavailable")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def main() -> int:
    descriptor = None
    stop = threading.Event()
    previous = {}
    try:
        descriptor = acquire_session_lock(Path("/run/user") / str(os.getuid()))
        if descriptor is None:
            print(json.dumps({"status": "session_already_held"}), flush=True)
            return 0
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda *_: stop.set())
        print(json.dumps({"status": "session_attached", "pid": os.getpid()}), flush=True)
        outcome = hold_session(stop)
        print(json.dumps({"status": outcome}), flush=True)
        return 0
    except Exception:
        # Never print raw subprocess errors, environment, paths or credentials.
        print(json.dumps({"status": "session_unavailable"}), flush=True)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if descriptor is not None:
            os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
