"""Private, bounded publication lock shared by every detailed-report writer."""

from __future__ import annotations

import math
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ReportLockError(RuntimeError):
    pass


@contextmanager
def report_publication_lock(directory: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Serialize the complete read/render/publish operation, not only replace().

    The lock file remains in the private date directory: unlinking it would let
    another writer lock a different inode while an older owner is still active.
    """

    if type(timeout_seconds) not in {int, float} or not (
        math.isfinite(timeout_seconds) and 0 <= timeout_seconds <= 60
    ):
        raise ValueError("report lock timeout must be finite and between 0 and 60 seconds")
    try:
        import fcntl
    except ImportError as exc:
        raise ReportLockError("report publication requires local POSIX file locking") from exc

    path = directory / ".daily-report.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ReportLockError("report publication lock is unsafe")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReportLockError("report publication lock is busy") from None
                time.sleep(min(0.05, remaining))
        current = path.lstat()
        if (
            (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) != 0o600
        ):
            raise ReportLockError("report publication lock changed during acquisition")
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ReportLockError("report publication lock is unavailable") from exc
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    try:
        yield
    finally:
        os.close(descriptor)
