"""Private POSIX evidence byte bindings, not immutable model-input snapshots.

Each asset is hashed through a no-follow descriptor with identity and metadata
checked before and after reading. These endpoint checks cannot prove that no
transient modification occurred and was restored between observations. Nor do
they prove that a later model call held or consumed these same immutable bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import PurePosixPath

from .domain import EventMedia

MAX_BINDING_BYTES = 16 * 1024
MAX_CLIP_BYTES = 1024 * 1024 * 1024
MAX_FRAME_BYTES = 20 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ASSET_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_DIRECTORY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_uid")


class EvidenceBindingError(ValueError):
    """A path-free evidence binding validation or verification error."""


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise EvidenceBindingError("evidence binding payload is invalid") from None


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _check_sha(value: object) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EvidenceBindingError("evidence binding fingerprint is invalid")


def media_fingerprint(media: EventMedia) -> str:
    """Hash the exact canonical EventMedia payload without accessing its paths."""
    if not isinstance(media, EventMedia):
        raise EvidenceBindingError("evidence binding media is invalid")
    try:
        payload = media.to_dict()
        EventMedia.from_dict(payload)
        return _sha(payload)
    except (ValueError, TypeError, AttributeError, OverflowError, RecursionError, UnicodeError):
        raise EvidenceBindingError("evidence binding media is invalid") from None


@dataclass(frozen=True, slots=True)
class EvidenceAssetBinding:
    role: str
    index: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in ("clip", "frame") or type(self.index) is not int:
            raise EvidenceBindingError("evidence binding asset role or index is invalid")
        if (self.role == "clip" and self.index != 0) or (
            self.role == "frame" and not 0 <= self.index < 8
        ):
            raise EvidenceBindingError("evidence binding asset index is invalid")
        maximum = MAX_CLIP_BYTES if self.role == "clip" else MAX_FRAME_BYTES
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= maximum:
            raise EvidenceBindingError("evidence binding asset size is invalid")
        _check_sha(self.sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "index": self.index,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    media_sha256: str
    assets: tuple[EvidenceAssetBinding, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise EvidenceBindingError("evidence binding version is invalid")
        _check_sha(self.media_sha256)
        if not isinstance(self.assets, tuple) or not 2 <= len(self.assets) <= 9:
            raise EvidenceBindingError("evidence binding requires one clip and one to eight frames")
        for position, asset in enumerate(self.assets):
            if not isinstance(asset, EvidenceAssetBinding):
                raise EvidenceBindingError("evidence binding asset is invalid")
            expected = ("clip", 0) if position == 0 else ("frame", position - 1)
            if (asset.role, asset.index) != expected:
                raise EvidenceBindingError("evidence binding asset order is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "media_sha256": self.media_sha256,
            "assets": [asset.to_dict() for asset in self.assets],
        }

    def to_json(self) -> str:
        payload = _canonical(self.to_dict())
        if len(payload.encode("utf-8")) > MAX_BINDING_BYTES:
            raise EvidenceBindingError("evidence binding JSON exceeds its size limit")
        return payload

    @property
    def fingerprint(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> EvidenceBinding:
        if not isinstance(value, dict) or set(value) != {"version", "media_sha256", "assets"}:
            raise EvidenceBindingError("evidence binding fields are invalid")
        rows = value["assets"]
        if not isinstance(rows, list) or not 2 <= len(rows) <= 9:
            raise EvidenceBindingError("evidence binding assets are invalid")
        assets = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"role", "index", "size_bytes", "sha256"}:
                raise EvidenceBindingError("evidence binding asset fields are invalid")
            assets.append(EvidenceAssetBinding(**row))
        return cls(
            version=value["version"], media_sha256=value["media_sha256"], assets=tuple(assets)
        )

    @classmethod
    def from_json(cls, payload: str) -> EvidenceBinding:
        try:
            if (
                not isinstance(payload, str)
                or not 1 <= len(payload.encode("utf-8")) <= MAX_BINDING_BYTES
            ):
                raise EvidenceBindingError("evidence binding JSON size is invalid")

            def unique_object(pairs: list[tuple[str, object]]) -> dict:
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise EvidenceBindingError("evidence binding JSON has duplicate keys")
                    result[key] = value
                return result

            def reject_constant(_value: str) -> None:
                raise EvidenceBindingError("evidence binding JSON has nonfinite values")

            value = json.loads(
                payload, object_pairs_hook=unique_object, parse_constant=reject_constant
            )
            return cls.from_dict(value)
        except EvidenceBindingError:
            raise
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            raise EvidenceBindingError("evidence binding JSON is invalid") from None

    def verify(self, media: EventMedia) -> None:
        """Recapture endpoint bytes; do not imply immutable inputs held by a model."""
        if capture_evidence_binding(media) != self:
            raise EvidenceBindingError("evidence binding does not match current evidence")


def _same(left: os.stat_result, right: os.stat_result, fields: tuple[str, ...]) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _file_metadata(metadata: os.stat_result, maximum: int) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or not 1 <= metadata.st_size <= maximum
    ):
        raise EvidenceBindingError("evidence binding asset is not a bounded private owned file")


def _path_components(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not value.startswith("/")
        or value.startswith("//")
    ):
        raise EvidenceBindingError("evidence binding asset path is unsafe")
    if ".." in value.split("/"):
        raise EvidenceBindingError("evidence binding asset path is unsafe")
    path = PurePosixPath(value)
    if len(path.parts) < 2:
        raise EvidenceBindingError("evidence binding asset path is unsafe")
    return path.parts[1:]


def _capture_asset(parts: tuple[str, ...], *, role: str, index: int) -> EvidenceAssetBinding:
    maximum = MAX_CLIP_BYTES if role == "clip" else MAX_FRAME_BYTES
    descriptors = []
    anchors = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
        root = os.open("/", directory_flags)
        descriptors.append(root)
        parent = root
        for component in parts[:-1]:
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise EvidenceBindingError("evidence binding asset ancestor is unsafe")
            current = os.open(component, directory_flags, dir_fd=parent)
            descriptors.append(current)
            opened = os.fstat(current)
            if not _same(before, opened, _DIRECTORY_FIELDS):
                raise EvidenceBindingError("evidence binding asset ancestor changed")
            anchors.append((parent, component, current, opened))
            parent = current
        before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        _file_metadata(before, maximum)
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        _file_metadata(opened, maximum)
        if not _same(before, opened, _ASSET_FIELDS):
            raise EvidenceBindingError("evidence binding asset changed before reading")
        digest = hashlib.sha256()
        total = 0
        while total <= opened.st_size:
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, opened.st_size + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > opened.st_size:
                raise EvidenceBindingError("evidence binding asset changed while reading")
            digest.update(chunk)
        final = os.fstat(descriptor)
        final_path = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        _file_metadata(final, maximum)
        _file_metadata(final_path, maximum)
        if (
            total != opened.st_size
            or not _same(opened, final, _ASSET_FIELDS)
            or not _same(opened, final_path, _ASSET_FIELDS)
        ):
            raise EvidenceBindingError("evidence binding asset changed while reading")
        for parent_fd, component, current_fd, metadata in anchors:
            if not _same(metadata, os.fstat(current_fd), _DIRECTORY_FIELDS) or not _same(
                metadata,
                os.stat(component, dir_fd=parent_fd, follow_symlinks=False),
                _DIRECTORY_FIELDS,
            ):
                raise EvidenceBindingError("evidence binding asset ancestor changed")
        return EvidenceAssetBinding(
            role=role, index=index, size_bytes=total, sha256=digest.hexdigest()
        )
    except EvidenceBindingError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, NotImplementedError):
        raise EvidenceBindingError("evidence binding asset is missing or unsafe") from None
    finally:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)


def capture_evidence_binding(media: EventMedia) -> EvidenceBinding:
    """Hash clip then ordered frames using private, no-follow POSIX reads.

    System-owned ancestor directories are allowed but symlink ancestors are not.
    Asset files themselves must be owned by the current user, have one link, and
    grant no group/world permissions. This performs endpoint checks per asset;
    it is neither an atomic multi-asset snapshot nor a model-consumption proof.
    """
    if os.name != "posix" or any(
        not hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_NONBLOCK", "O_DIRECTORY")
    ):
        raise EvidenceBindingError("evidence binding requires private POSIX file access")
    fingerprint = media_fingerprint(media)
    if not 1 <= len(media.frame_paths) <= 8:
        raise EvidenceBindingError("evidence binding requires one to eight frames")
    paths = tuple(_path_components(path) for path in (media.clip_path, *media.frame_paths))
    if len(set(paths)) != len(paths):
        raise EvidenceBindingError("evidence binding asset paths must be unique")
    assets = tuple(
        _capture_asset(parts, role="clip" if position == 0 else "frame", index=max(0, position - 1))
        for position, parts in enumerate(paths)
    )
    return EvidenceBinding(media_sha256=fingerprint, assets=assets)
