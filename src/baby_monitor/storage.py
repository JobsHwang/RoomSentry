from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from .config import StorageSettings

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_-]+")


def safe_component(value: str, *, fallback: str) -> str:
    normalized = _SAFE_COMPONENT.sub("_", value.strip()).strip("_")
    return normalized[:80] or fallback


@dataclass(frozen=True, slots=True)
class AllocatedMedia:
    clip_path: Path
    frame_paths: tuple[Path, ...]


def _harden_private_directory(path: Path) -> Path:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("private storage directory is unavailable") from exc
    try:
        if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid():
            raise ValueError("private storage directory is unsafe")
        os.fchmod(descriptor, 0o700)
        after = os.fstat(descriptor)
        path_metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("private storage directory cannot be secured") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if path.is_symlink() or (
        not stat.S_ISDIR(after.st_mode)
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o700
        or (path_metadata.st_dev, path_metadata.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise ValueError("private storage directory cannot be secured")
    return resolved


def _private_subdirectory(root: Path, *components: str) -> Path:
    resolved_root = root.resolve(strict=True)
    current = resolved_root
    for component in components:
        if not component or component in {".", ".."} or "/" in component:
            raise ValueError("private storage component is invalid")
        current = current / component
        try:
            current.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ValueError("private storage directory is unavailable") from exc
        resolved = _harden_private_directory(current)
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("private storage directory escaped its root")
        current = resolved
    return current


def secure_media_file(path: Path) -> None:
    """Tighten one newly written media artifact without following links."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("private media file is unavailable") from exc
    try:
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or before.st_size <= 0
        ):
            raise ValueError("private media file is unsafe")
        os.fchmod(descriptor, 0o600)
        after = os.fstat(descriptor)
        path_metadata = path.lstat()
    except OSError as exc:
        raise ValueError("private media file cannot be secured") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        path.is_symlink()
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o600
        or (after.st_dev, after.st_ino, after.st_size)
        != (before.st_dev, before.st_ino, before.st_size)
        or (path_metadata.st_dev, path_metadata.st_ino, path_metadata.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise ValueError("private media file cannot be secured")


class StorageResolver:
    def __init__(self, settings: StorageSettings):
        self.settings = settings

    def active_media_root(self) -> Path:
        nas_root = self.settings.nas_root
        if (
            self.settings.prefer_nas
            and nas_root is not None
            and nas_root.is_dir()
            and os.access(nas_root, os.W_OK)
        ):
            return nas_root.expanduser().resolve()
        local_root = self.settings.local_root.expanduser()
        local_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return _harden_private_directory(local_root)

    def allocate_media(
        self,
        *,
        occurred_at: datetime,
        camera_id: str,
        event_id: str,
        frame_count: int,
    ) -> AllocatedMedia:
        if occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if not 1 <= frame_count <= 8:
            raise ValueError("frame_count must be between 1 and 8")

        utc_time = occurred_at.astimezone(UTC)
        camera = safe_component(camera_id, fallback="camera")
        event = safe_component(event_id, fallback="event")
        directory = _private_subdirectory(
            self.active_media_root(),
            "events",
            utc_time.strftime("%Y"),
            utc_time.strftime("%m"),
            utc_time.strftime("%d"),
            camera,
        )
        stem = f"{utc_time.strftime('%Y%m%dT%H%M%S_%fZ')}_{camera}_{event}"
        clip_path = directory / f"{stem}.mp4"
        frame_paths = tuple(
            directory / f"{stem}_f{index:02d}.jpg" for index in range(1, frame_count + 1)
        )
        return AllocatedMedia(clip_path=clip_path, frame_paths=frame_paths)

    def report_paths(self, report_date: date) -> tuple[Path, Path]:
        directory = _private_subdirectory(
            self.active_media_root(),
            "reports",
            report_date.strftime("%Y"),
            report_date.strftime("%m"),
            report_date.strftime("%d"),
        )
        stem = report_date.isoformat() + "_daily_report"
        return directory / f"{stem}.md", directory / f"{stem}.html"
