from __future__ import annotations

import os
import re
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .daily_digest import DailyDigestError, DailyDigestPlan

_MAX_PLAN_BYTES = 2 * 1024 * 1024
_MAX_PLANS = 400
_PLAN_NAME = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<digest>[0-9a-f]{32})_digest_plan\.json\Z")
_ERROR = "daily digest plan store failed"


class DailyDigestStoreError(RuntimeError):
    """Fixed, path-free error for private digest plan persistence."""


@dataclass(frozen=True, slots=True)
class DailyDigestWriteReceipt:
    local_date: str
    source_event_count: int
    selected_event_count: int
    selected_duration_seconds: float
    target_met: bool
    reused_existing: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "stored",
            "local_date": self.local_date,
            "source_event_count": self.source_event_count,
            "selected_event_count": self.selected_event_count,
            "selected_duration_seconds": self.selected_duration_seconds,
            "target_met": self.target_met,
            "reused_existing": self.reused_existing,
            "plan_private": True,
            "media_read": False,
            "video_generated": False,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
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
        raise DailyDigestStoreError(_ERROR) from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size <= 0
        or resolved != candidate
        or parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or resolved_parent != parent
    ):
        raise DailyDigestStoreError(_ERROR)
    return resolved


def _admit_root(
    database: Path,
    output_root: str | Path,
    *,
    create: bool,
) -> Path | None:
    candidate = Path(output_root).expanduser()
    expected = database.parent / "daily-digests"
    if not candidate.is_absolute() or candidate != expected:
        raise DailyDigestStoreError(_ERROR)
    if not candidate.exists() and not create:
        return None
    try:
        if create:
            candidate.mkdir(mode=0o700, exist_ok=True)
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise DailyDigestStoreError(_ERROR) from None
    if (
        candidate.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != candidate
    ):
        raise DailyDigestStoreError(_ERROR)
    return resolved


def _verify_plan_file(path: Path, *, root: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise DailyDigestStoreError(_ERROR) from None
    if (
        path.parent != root
        or path.is_symlink()
        or _PLAN_NAME.fullmatch(path.name) is None
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not 0 < metadata.st_size <= _MAX_PLAN_BYTES
        or resolved != path
    ):
        raise DailyDigestStoreError(_ERROR)
    return metadata


def _read_plan(path: Path, *, root: Path) -> tuple[bytes, DailyDigestPlan]:
    expected = _verify_plan_file(path, root=root)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                expected.st_dev,
                expected.st_ino,
                expected.st_size,
            ):
                raise DailyDigestStoreError(_ERROR)
            payload = os.read(descriptor, _MAX_PLAN_BYTES + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise DailyDigestStoreError(_ERROR) from None
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        expected.st_dev,
        expected.st_ino,
        expected.st_size,
        expected.st_mtime_ns,
    ) or len(payload) != expected.st_size:
        raise DailyDigestStoreError(_ERROR)
    try:
        plan = DailyDigestPlan.from_json(payload.decode("utf-8"))
    except (UnicodeError, DailyDigestError):
        raise DailyDigestStoreError(_ERROR) from None
    matched = _PLAN_NAME.fullmatch(path.name)
    assert matched is not None
    if matched["date"] != plan.local_date or matched["digest"] != plan.digest_id:
        raise DailyDigestStoreError(_ERROR)
    if payload != (plan.to_json(indent=2) + "\n").encode("utf-8"):
        raise DailyDigestStoreError(_ERROR)
    return payload, plan


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise DailyDigestStoreError(_ERROR) from None


def write_daily_digest_plan(
    plan: DailyDigestPlan,
    *,
    database_path: str | Path,
    output_root: str | Path,
) -> DailyDigestWriteReceipt:
    if not isinstance(plan, DailyDigestPlan):
        raise DailyDigestStoreError(_ERROR)
    database = _admit_database(database_path)
    root = _admit_root(database, output_root, create=True)
    assert root is not None
    destination = root / f"{plan.local_date}_{plan.digest_id}_digest_plan.json"
    payload = (plan.to_json(indent=2) + "\n").encode("utf-8")
    if not 0 < len(payload) <= _MAX_PLAN_BYTES:
        raise DailyDigestStoreError(_ERROR)
    reused = False
    if destination.exists() or destination.is_symlink():
        existing, parsed = _read_plan(destination, root=root)
        if existing != payload or parsed != plan:
            raise DailyDigestStoreError(_ERROR)
        reused = True
    else:
        temporary = root / f".{destination.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError:
                existing, parsed = _read_plan(destination, root=root)
                if existing != payload or parsed != plan:
                    raise DailyDigestStoreError(_ERROR) from None
                reused = True
            _fsync_directory(root)
        except (OSError, DailyDigestStoreError):
            raise DailyDigestStoreError(_ERROR) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        _verify_plan_file(destination, root=root)
    return DailyDigestWriteReceipt(
        local_date=plan.local_date,
        source_event_count=plan.source_event_count,
        selected_event_count=plan.selected_event_count,
        selected_duration_seconds=plan.selected_duration_seconds,
        target_met=plan.target_met,
        reused_existing=reused,
    )


def inspect_daily_digest_plans(
    *,
    database_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    common = {
        "version": 1,
        "paths_disclosed": False,
        "event_ids_disclosed": False,
        "camera_ids_disclosed": False,
        "media_read": False,
        "video_generated": False,
        "identity_inference": False,
        "risk_inference": False,
        "notification_decision": False,
    }
    try:
        database = _admit_database(database_path)
        root = _admit_root(database, output_root, create=False)
        if root is None:
            return {
                **common,
                "status": "not_generated",
                "plan_count": 0,
                "available": False,
                "operational": True,
                "attention_required": False,
            }
        paths = sorted(root.iterdir())
        if len(paths) > _MAX_PLANS:
            raise DailyDigestStoreError(_ERROR)
        plans = [_read_plan(path, root=root)[1] for path in paths]
    except (OSError, DailyDigestStoreError):
        return {
            **common,
            "status": "unavailable",
            "plan_count": 0,
            "available": False,
            "operational": False,
            "attention_required": True,
        }
    if not plans:
        return {
            **common,
            "status": "not_generated",
            "plan_count": 0,
            "available": False,
            "operational": True,
            "attention_required": False,
        }
    latest = max(plans, key=lambda item: (item.local_date, item.digest_id))
    newest_mtime = max(path.stat().st_mtime for path in paths)
    return {
        **common,
        "status": "available",
        "plan_count": len(plans),
        "available": True,
        "latest_local_date": latest.local_date,
        "latest_source_event_count": latest.source_event_count,
        "latest_selected_event_count": latest.selected_event_count,
        "latest_selected_duration_seconds": latest.selected_duration_seconds,
        "latest_target_met": latest.target_met,
        "updated_at": datetime.fromtimestamp(newest_mtime, tz=UTC).isoformat(),
        "operational": True,
        "attention_required": False,
    }


def load_latest_daily_digest_plan(
    *,
    database_path: str | Path,
    output_root: str | Path,
    local_date: str,
) -> DailyDigestPlan:
    try:
        datetime.strptime(local_date, "%Y-%m-%d")
        database = _admit_database(database_path)
        root = _admit_root(database, output_root, create=False)
        if root is None:
            raise DailyDigestStoreError(_ERROR)
        paths = sorted(root.iterdir())
        if not paths or len(paths) > _MAX_PLANS:
            raise DailyDigestStoreError(_ERROR)
        matching: list[tuple[int, str, DailyDigestPlan]] = []
        for path in paths:
            metadata = _verify_plan_file(path, root=root)
            plan = _read_plan(path, root=root)[1]
            if plan.local_date == local_date:
                matching.append((metadata.st_mtime_ns, plan.digest_id, plan))
    except (OSError, ValueError, DailyDigestStoreError):
        raise DailyDigestStoreError(_ERROR) from None
    if not matching:
        raise DailyDigestStoreError(_ERROR)
    return max(matching, key=lambda item: (item[0], item[1]))[2]


def load_daily_digest_plan(
    *,
    database_path: str | Path,
    output_root: str | Path,
    local_date: str,
    digest_id: str,
) -> DailyDigestPlan:
    if (
        not isinstance(local_date, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", local_date) is None
        or not isinstance(digest_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", digest_id) is None
    ):
        raise DailyDigestStoreError(_ERROR)
    try:
        datetime.strptime(local_date, "%Y-%m-%d")
        database = _admit_database(database_path)
        root = _admit_root(database, output_root, create=False)
        if root is None:
            raise DailyDigestStoreError(_ERROR)
        path = root / f"{local_date}_{digest_id}_digest_plan.json"
        plan = _read_plan(path, root=root)[1]
    except (OSError, ValueError, DailyDigestStoreError):
        raise DailyDigestStoreError(_ERROR) from None
    if plan.local_date != local_date or plan.digest_id != digest_id:
        raise DailyDigestStoreError(_ERROR)
    return plan
