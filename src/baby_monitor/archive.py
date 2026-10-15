from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

MAX_ARCHIVE_ITEMS = 64
MAX_ARCHIVE_FILE_BYTES = 64 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024**3
MAX_ARCHIVE_MANIFEST_BYTES = 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SAFE_PART = re.compile(r"[A-Za-z0-9_.-]{1,160}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArchiveError(RuntimeError):
    pass


class ArchiveUnavailableError(ArchiveError):
    pass


class ArchiveConflictError(ArchiveError):
    pass


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ArchiveError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArchiveError("archive relative_path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or not path.parts:
        raise ArchiveError("archive relative_path is invalid")
    if any(part in {".", ".."} or _SAFE_PART.fullmatch(part) is None for part in path.parts):
        raise ArchiveError("archive relative_path is invalid")
    return path


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArchiveError("archive manifest contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ArchiveError("archive manifest contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ArchiveError("archive manifest is invalid JSON") from exc


def _hash_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_FILE_BYTES:
                raise ArchiveError("archive source file exceeds the size limit")
            digest.update(chunk)
    except ArchiveError:
        raise
    except OSError as exc:
        raise ArchiveError("archive file could not be read") from exc
    return size, digest.hexdigest()


def _open_regular_read_only(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError("archive source file is missing or unsafe") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArchiveError("archive source file is missing or unsafe")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


@dataclass(frozen=True, slots=True)
class ArchiveItem:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path)
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_ARCHIVE_FILE_BYTES:
            raise ArchiveError("archive item size_bytes is invalid")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ArchiveError("archive item sha256 is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ArchiveItem:
        if not isinstance(value, dict) or set(value) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ArchiveError("archive item fields do not match v1")
        return cls(
            relative_path=value["relative_path"],
            size_bytes=value["size_bytes"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    archive_id: str
    created_at: datetime
    items: tuple[ArchiveItem, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ArchiveError("unsupported archive manifest version")
        if not isinstance(self.archive_id, str) or _SAFE_ID.fullmatch(self.archive_id) is None:
            raise ArchiveError("archive_id is invalid")
        object.__setattr__(self, "created_at", _aware(self.created_at, "created_at"))
        if not isinstance(self.items, tuple) or any(
            not isinstance(item, ArchiveItem) for item in self.items
        ):
            raise ArchiveError("archive manifest items are invalid")
        if not 1 <= len(self.items) <= MAX_ARCHIVE_ITEMS:
            raise ArchiveError("archive manifest must contain between 1 and 64 items")
        paths = [item.relative_path for item in self.items]
        if len(paths) != len(set(paths)) or paths != sorted(paths):
            raise ArchiveError("archive manifest item paths must be unique and sorted")
        if sum(item.size_bytes for item in self.items) > MAX_ARCHIVE_TOTAL_BYTES:
            raise ArchiveError("archive manifest exceeds the total size limit")

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "archive_id": self.archive_id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "items": [item.to_dict() for item in self.items],
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Any) -> ArchiveManifest:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "archive_id",
            "created_at",
            "items",
        }:
            raise ArchiveError("archive manifest fields do not match v1")
        if not isinstance(value["created_at"], str):
            raise ArchiveError("archive manifest created_at is invalid")
        try:
            created_at = datetime.fromisoformat(value["created_at"])
        except ValueError as exc:
            raise ArchiveError("archive manifest created_at is invalid") from exc
        if not isinstance(value["items"], list):
            raise ArchiveError("archive manifest items are invalid")
        return cls(
            version=value["version"],
            archive_id=value["archive_id"],
            created_at=created_at,
            items=tuple(ArchiveItem.from_dict(item) for item in value["items"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> ArchiveManifest:
        manifest_path = Path(path).expanduser()
        try:
            if (
                not manifest_path.is_absolute()
                or manifest_path.parent.resolve(strict=True) != manifest_path.parent
                or manifest_path.is_symlink()
                or not manifest_path.is_file()
            ):
                raise OSError
            if manifest_path.stat().st_size > MAX_ARCHIVE_MANIFEST_BYTES:
                raise ArchiveError("archive manifest exceeds the size limit")
            raw = manifest_path.read_text(encoding="utf-8")
        except ArchiveError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ArchiveError("archive manifest file is missing or invalid") from exc
        return cls.from_dict(_strict_json(raw))

    def write_new(self, path: str | Path) -> None:
        output = Path(path).expanduser()
        if not output.is_absolute():
            raise ArchiveError("archive manifest output must use an absolute path")
        parent = _validated_root(
            output.parent,
            field="archive manifest output directory",
            writable=True,
        )
        _publish_bytes_no_replace(
            parent / output.name,
            self.to_json_bytes(),
            conflict_message="archive manifest output already exists with different content",
        )


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    archive_id: str
    items: int
    total_bytes: int
    copied_items: int
    reused_items: int
    copied_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "archive_id": self.archive_id,
            "status": "archived",
            "items": self.items,
            "total_bytes": self.total_bytes,
            "copied_items": self.copied_items,
            "reused_items": self.reused_items,
            "copied_bytes": self.copied_bytes,
            "paths_disclosed": False,
        }


def _publish_bytes_no_replace(
    destination: Path,
    content: bytes,
    *,
    conflict_message: str,
) -> bool:
    if destination.is_symlink():
        raise ArchiveConflictError(conflict_message)
    if destination.exists():
        try:
            if (
                destination.is_file()
                and destination.stat().st_size == len(content)
                and destination.read_bytes() == content
            ):
                return False
        except OSError as exc:
            raise ArchiveUnavailableError("archive target is unavailable") from exc
        raise ArchiveConflictError(conflict_message)

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
            return True
        except FileExistsError:
            if destination.is_file() and not destination.is_symlink():
                try:
                    if (
                        destination.stat().st_size == len(content)
                        and destination.read_bytes() == content
                    ):
                        return False
                except OSError as exc:
                    raise ArchiveUnavailableError("archive target is unavailable") from exc
            raise ArchiveConflictError(conflict_message) from None
        except OSError as exc:
            raise ArchiveUnavailableError(
                "archive target does not support atomic no-replace publication"
            ) from exc
    except ArchiveError:
        raise
    except OSError as exc:
        raise ArchiveUnavailableError("archive target is unavailable") from exc
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _validated_root(path: str | Path, *, field: str, writable: bool) -> Path:
    root = Path(path).expanduser()
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ArchiveUnavailableError(f"{field} is missing or unsafe") from exc
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or resolved != root:
        raise ArchiveUnavailableError(f"{field} is missing or unsafe")
    if writable and not os.access(root, os.W_OK):
        raise ArchiveUnavailableError(f"{field} is not writable")
    return resolved


def _source_path(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArchiveError("archive source file is missing or unsafe")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise ArchiveError("archive source file is missing or unsafe") from exc
    if not resolved.is_relative_to(root):
        raise ArchiveError("archive source file is missing or unsafe")
    return resolved


def _source_item(root: Path, item: ArchiveItem) -> Path:
    return _source_path(root, _safe_relative_path(item.relative_path))


def _safe_target_parent(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parent.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise ArchiveUnavailableError("archive target is unavailable") from exc
        if current.is_symlink() or not current.is_dir():
            raise ArchiveConflictError("archive target contains an unsafe parent")
    return current


def _file_matches(path: Path, item: ArchiveItem) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with _open_regular_read_only(path) as handle:
            size, digest = _hash_stream(handle)
    except ArchiveError:
        return False
    return size == item.size_bytes and digest == item.sha256


def _copy_item_no_replace(source: Path, destination: Path, item: ArchiveItem) -> bool:
    if destination.exists() or destination.is_symlink():
        if _file_matches(destination, item):
            return False
        raise ArchiveConflictError("archive destination exists with different content")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        size = 0
        with (
            _open_regular_read_only(source) as source_handle,
            os.fdopen(descriptor, "wb") as target_handle,
        ):
            while chunk := source_handle.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_FILE_BYTES:
                    raise ArchiveError("archive source file exceeds the size limit")
                digest.update(chunk)
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if size != item.size_bytes or digest.hexdigest() != item.sha256:
            raise ArchiveError("archive source changed after the manifest was created")
        try:
            os.link(temporary, destination, follow_symlinks=False)
            return True
        except FileExistsError:
            if _file_matches(destination, item):
                return False
            raise ArchiveConflictError(
                "archive destination exists with different content"
            ) from None
        except OSError as exc:
            raise ArchiveUnavailableError(
                "archive target does not support atomic no-replace publication"
            ) from exc
    except ArchiveError:
        raise
    except OSError as exc:
        raise ArchiveUnavailableError("archive target is unavailable") from exc
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def build_archive_manifest(
    *,
    archive_id: str,
    source_root: str | Path,
    files: tuple[str | Path, ...],
    created_at: datetime | None = None,
) -> ArchiveManifest:
    root = _validated_root(source_root, field="archive source root", writable=False)
    if not 1 <= len(files) <= MAX_ARCHIVE_ITEMS:
        raise ArchiveError("archive plan must contain between 1 and 64 files")
    items: list[ArchiveItem] = []
    for raw_path in files:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ArchiveError("archive source files must use absolute paths")
        try:
            lexical_relative = path.relative_to(root)
        except ValueError as exc:
            raise ArchiveError("archive source file is outside the source root") from exc
        relative = _safe_relative_path(str(PurePosixPath(*lexical_relative.parts)))
        resolved = _source_path(root, relative)
        if resolved != path:
            raise ArchiveError("archive source file is outside the source root")
        with _open_regular_read_only(resolved) as handle:
            size, digest = _hash_stream(handle)
        items.append(ArchiveItem(str(relative), size, digest))
    items.sort(key=lambda item: item.relative_path)
    return ArchiveManifest(
        archive_id=archive_id,
        created_at=_aware(created_at or datetime.now(UTC), "created_at"),
        items=tuple(items),
    )


def validate_archive_sources(
    source_root: str | Path,
    manifest: ArchiveManifest,
) -> Path:
    root = _validated_root(source_root, field="archive source root", writable=False)
    for item in manifest.items:
        source = _source_item(root, item)
        with _open_regular_read_only(source) as handle:
            size, digest = _hash_stream(handle)
        if size != item.size_bytes or digest != item.sha256:
            raise ArchiveError("archive source changed after the manifest was created")
    return root


class FilesystemArchiveAdapter:
    def __init__(self, *, source_root: str | Path, target_root: str | Path):
        self.source_root = _validated_root(
            source_root,
            field="archive source root",
            writable=False,
        )
        self.target_root = _validated_root(
            target_root,
            field="archive target root",
            writable=True,
        )

    def archive(self, manifest: ArchiveManifest) -> ArchiveResult:
        copied_items = 0
        reused_items = 0
        copied_bytes = 0
        for item in manifest.items:
            source = _source_item(self.source_root, item)
            relative = _safe_relative_path(item.relative_path)
            parent = _safe_target_parent(self.target_root, relative)
            destination = parent / relative.name
            if _copy_item_no_replace(source, destination, item):
                copied_items += 1
                copied_bytes += item.size_bytes
            else:
                reused_items += 1

        manifest_parent = _safe_target_parent(
            self.target_root,
            PurePosixPath("archive-manifests", f"{manifest.archive_id}.json"),
        )
        _publish_bytes_no_replace(
            manifest_parent / f"{manifest.archive_id}.json",
            manifest.to_json_bytes(),
            conflict_message="archive completion manifest conflicts with existing content",
        )
        return ArchiveResult(
            archive_id=manifest.archive_id,
            items=len(manifest.items),
            total_bytes=manifest.total_bytes,
            copied_items=copied_items,
            reused_items=reused_items,
            copied_bytes=copied_bytes,
        )
