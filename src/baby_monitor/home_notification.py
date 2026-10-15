from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .notification import (
    FileNotificationSink,
    NotificationJobStatus,
    NotificationOutbox,
    NotificationQueueError,
    deliver_notification_once,
)


class HomeNotificationError(RuntimeError):
    """Path-free failure for a manually triggered local test notification."""


@dataclass(frozen=True, slots=True)
class HomeNotificationTestReceipt:
    status: str
    delivered_at: str

    def __post_init__(self) -> None:
        if self.status != "delivered":
            raise HomeNotificationError("local notification test failed")
        try:
            parsed = datetime.fromisoformat(self.delivered_at)
        except ValueError:
            raise HomeNotificationError("local notification test failed") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise HomeNotificationError("local notification test failed")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "delivered_at": self.delivered_at,
            "test_only": True,
            "local_file_written": True,
            "external_delivery_used": False,
            "event_authority": False,
            "risk_authority": False,
            "notification_rule_decision": False,
            "paths_disclosed": False,
        }


def _admit_database(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    parent = candidate.parent
    try:
        metadata = candidate.lstat()
        parent_metadata = parent.lstat()
        resolved = candidate.resolve(strict=True)
        resolved_parent = parent.resolve(strict=True)
    except OSError:
        raise HomeNotificationError("local notification database is unavailable") from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or resolved != candidate
        or parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
        or resolved_parent != parent
    ):
        raise HomeNotificationError("local notification database is unavailable")
    return resolved


def _admit_target(database: Path, target_root: str | Path) -> Path:
    target = Path(target_root).expanduser()
    expected = database.parent / "local-notifications"
    if not target.is_absolute() or target != expected:
        raise HomeNotificationError("local notification target is unavailable")
    try:
        target.mkdir(mode=0o700, exist_ok=True)
        metadata = target.lstat()
        resolved = target.resolve(strict=True)
    except OSError:
        raise HomeNotificationError("local notification target is unavailable") from None
    if (
        target.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != target
    ):
        raise HomeNotificationError("local notification target is unavailable")
    return resolved


def _verify_artifact(path: Path, *, target: Path) -> None:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise HomeNotificationError("local notification test failed") from None
    if (
        path.parent != target
        or path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or resolved != path
    ):
        raise HomeNotificationError("local notification test failed")


def run_home_local_notification_test(
    *,
    database_path: str | Path,
    target_root: str | Path,
    max_pending: int,
    lease_seconds: int,
    retry_delay_seconds: int,
    max_attempts: int,
    test_id: str | None = None,
    now: datetime | None = None,
) -> HomeNotificationTestReceipt:
    database = _admit_database(database_path)
    target = _admit_target(database, target_root)
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise HomeNotificationError("local notification test failed")
    opaque_test_id = test_id or secrets.token_hex(16)
    outbox = NotificationOutbox(database)
    try:
        enqueued = outbox.enqueue_local_test(
            test_id=opaque_test_id,
            sink_kind="file",
            sink_id="dashboard_test",
            max_active=max_pending,
            now=observed_at,
        )
        if enqueued.job.status is NotificationJobStatus.DELIVERED:
            delivered_notification_id = enqueued.job.notification_id
        else:
            attempt = deliver_notification_once(
                outbox=outbox,
                sink=FileNotificationSink(target),
                sink_id="dashboard_test",
                lease_seconds=lease_seconds,
                retry_delay_seconds=retry_delay_seconds,
                max_attempts=max_attempts,
                now=observed_at,
                notification_id=enqueued.job.notification_id,
            )
            if attempt.status != "delivered" or attempt.notification_id is None:
                raise HomeNotificationError("local notification test failed")
            delivered_notification_id = attempt.notification_id
    except (OSError, ValueError, NotificationQueueError):
        raise HomeNotificationError("local notification test failed") from None
    _verify_artifact(target / f"{delivered_notification_id}.json", target=target)
    return HomeNotificationTestReceipt(
        status="delivered",
        delivered_at=observed_at.astimezone(UTC).isoformat(),
    )
