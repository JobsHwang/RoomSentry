"""Opt-in, bounded local-review suspension without stopping live detection.

Requests are atomic field assignments: POSIX signal handlers must not acquire
the monitor's locks or touch a model. Only its serialized review thread may
mark unloading complete, release the pause, and resume new evidence. This
does not replace the cross-process VLM execution lease.
"""

from __future__ import annotations

import math
import time
from pathlib import Path


def process_start_ticks(pid: int) -> int:
    """Read only Linux's process-start field, never argv/environment or secrets."""
    if type(pid) is not int or not 1 <= pid < 2**31:
        raise ValueError("invalid worker identity")
    with (Path("/proc") / str(pid) / "stat").open("rb") as stream:
        payload = stream.read(4097)
    if len(payload) > 4096:
        raise ValueError("invalid worker identity")
    # comm can contain spaces and ')'; fields after its last ')' begin at #3.
    fields = payload.rpartition(b")")[2].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise ValueError("invalid worker identity")
    return int(fields[19])


class LiveReviewSuspension:
    """One monitor-local lease; acknowledgement follows actual adapter.close()."""

    def __init__(self, *, timeout_seconds: int = 600, clock=time.monotonic):
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
            raise ValueError("invalid review suspension timeout")
        if not callable(clock):
            raise ValueError("invalid review suspension clock")
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self._deadline = 0.0
        self.unloaded = False
        self.resumed_after = 0.0
        self.suspension_count = 0

    def request_suspend(self) -> None:
        # A handler writes only this field. No locks, model calls or file I/O.
        self._deadline = self.clock() + self.timeout_seconds

    def request_resume(self) -> None:
        self._deadline = 0.0

    @property
    def requested(self) -> bool:
        return self.clock() < self._deadline

    @property
    def blocked(self) -> bool:
        # Expiry/resume does not admit work until the review thread clears it.
        return self.unloaded or self.requested

    def acknowledge_unloaded(self) -> None:
        if self.unloaded:
            raise ValueError("review already unloaded")
        self.unloaded = True
        self.suspension_count += 1

    def release(self) -> bool:
        if not self.unloaded:
            raise ValueError("review suspension not ready to resume")
        if self.requested:
            # A signal can renew the pause while the worker is flushing queues.
            return False
        self.resumed_after = self.clock()
        self.unloaded = False
        return True

    def snapshot(self) -> dict[str, object]:
        remaining = max(0.0, self._deadline - self.clock())
        if not math.isfinite(remaining):
            raise ValueError("invalid review suspension clock")
        return {
            "enabled": True,
            "pause_requested": remaining > 0,
            "resident_unloaded": self.unloaded,
            "timeout_seconds": self.timeout_seconds,
            "remaining_seconds": round(min(self.timeout_seconds, remaining), 3),
            "suspension_count": self.suspension_count,
        }
