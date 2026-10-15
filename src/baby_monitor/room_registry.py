from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ROOM_REGISTRY_BYTES = 65_536
MAX_ROOMS = 16
MAX_PHYSICAL_CAMERAS = 16
MAX_LOGICAL_STREAMS = 32

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class RoomRegistryError(RuntimeError):
    pass


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise RoomRegistryError(f"{field} is invalid")
    return value


def _safe_display_name(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not 1 <= len(value) <= 60
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise RoomRegistryError(f"{field} is invalid")
    return value


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-standard JSON number")
            ),
        )
    except (UnicodeError, ValueError) as exc:
        raise RoomRegistryError(f"{label} is invalid") from exc


def _require_private_linux_path(path: Path, *, label: str) -> Path:
    if os.name != "posix":
        return path
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RoomRegistryError(f"{label} is missing or unsafe") from exc
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "mnt" and parts[2].casefold() == "c":
        raise RoomRegistryError(f"{label} must remain on the private Linux filesystem")
    return resolved


def _require_git_external_posix_path(path: Path, *, label: str) -> None:
    resolved = _require_private_linux_path(path, label=label)
    if os.name != "posix":
        return
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise RoomRegistryError(f"{label} must remain outside the source worktree")


def _read_private_json(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
    require_git_external: bool = True,
) -> Any:
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise RoomRegistryError(f"{label} path must be absolute")
    if require_git_external:
        _require_git_external_posix_path(source, label=label)
    else:
        _require_private_linux_path(source, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = source.lstat()
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise RoomRegistryError(f"{label} is missing or unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or before.st_dev != metadata.st_dev
            or before.st_ino != metadata.st_ino
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= max_bytes
        ):
            raise RoomRegistryError(f"{label} is missing or unsafe")
        if os.name == "posix" and (
            metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RoomRegistryError(f"{label} permissions are unsafe")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) > max_bytes
            or after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
        ):
            raise RoomRegistryError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    return _strict_json(raw, label=label)


@dataclass(frozen=True, slots=True)
class RoomDefinition:
    room_id: str
    display_name: str

    def __post_init__(self) -> None:
        _safe_id(self.room_id, "room_id")
        _safe_display_name(self.display_name, "room display_name")

    @classmethod
    def from_dict(cls, value: Any) -> RoomDefinition:
        if not isinstance(value, dict) or set(value) != {"room_id", "display_name"}:
            raise RoomRegistryError("room definition fields do not match v1")
        return cls(room_id=value["room_id"], display_name=value["display_name"])

    def to_dict(self) -> dict[str, str]:
        return {"room_id": self.room_id, "display_name": self.display_name}


@dataclass(frozen=True, slots=True)
class PhysicalCameraDefinition:
    physical_camera_id: str
    display_name: str

    def __post_init__(self) -> None:
        _safe_id(self.physical_camera_id, "physical_camera_id")
        _safe_display_name(self.display_name, "physical camera display_name")

    @classmethod
    def from_dict(cls, value: Any) -> PhysicalCameraDefinition:
        expected = {"physical_camera_id", "display_name"}
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomRegistryError("physical camera fields do not match v2")
        return cls(
            physical_camera_id=value["physical_camera_id"],
            display_name=value["display_name"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "physical_camera_id": self.physical_camera_id,
            "display_name": self.display_name,
        }


@dataclass(frozen=True, slots=True)
class LogicalStreamDefinition:
    camera_id: str
    physical_camera_id: str
    room_id: str
    display_name: str

    def __post_init__(self) -> None:
        _safe_id(self.camera_id, "camera_id")
        _safe_id(self.physical_camera_id, "logical stream physical_camera_id")
        _safe_id(self.room_id, "logical stream room_id")
        _safe_display_name(self.display_name, "logical stream display_name")

    @classmethod
    def from_dict(cls, value: Any) -> LogicalStreamDefinition:
        expected = {"camera_id", "physical_camera_id", "room_id", "display_name"}
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomRegistryError("logical stream fields do not match v2")
        return cls(
            camera_id=value["camera_id"],
            physical_camera_id=value["physical_camera_id"],
            room_id=value["room_id"],
            display_name=value["display_name"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "camera_id": self.camera_id,
            "physical_camera_id": self.physical_camera_id,
            "room_id": self.room_id,
            "display_name": self.display_name,
        }


@dataclass(frozen=True, slots=True)
class RoomRegistry:
    rooms: tuple[RoomDefinition, ...]
    physical_cameras: tuple[PhysicalCameraDefinition, ...]
    logical_streams: tuple[LogicalStreamDefinition, ...]
    version: int = 2

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 2:
            raise RoomRegistryError("unsupported room registry version")
        if not isinstance(self.rooms, tuple) or not 1 <= len(self.rooms) <= MAX_ROOMS:
            raise RoomRegistryError("room count is invalid")
        if (
            not isinstance(self.physical_cameras, tuple)
            or not 1 <= len(self.physical_cameras) <= MAX_PHYSICAL_CAMERAS
        ):
            raise RoomRegistryError("physical camera count is invalid")
        if (
            not isinstance(self.logical_streams, tuple)
            or not 1 <= len(self.logical_streams) <= MAX_LOGICAL_STREAMS
        ):
            raise RoomRegistryError("logical stream count is invalid")
        if any(not isinstance(item, RoomDefinition) for item in self.rooms):
            raise RoomRegistryError("room definition is invalid")
        if any(not isinstance(item, PhysicalCameraDefinition) for item in self.physical_cameras):
            raise RoomRegistryError("physical camera definition is invalid")
        if any(not isinstance(item, LogicalStreamDefinition) for item in self.logical_streams):
            raise RoomRegistryError("logical stream definition is invalid")

        room_ids = tuple(item.room_id for item in self.rooms)
        physical_ids = tuple(item.physical_camera_id for item in self.physical_cameras)
        camera_ids = tuple(item.camera_id for item in self.logical_streams)
        for values, label in (
            (room_ids, "room_id"),
            (physical_ids, "physical_camera_id"),
            (camera_ids, "camera_id"),
        ):
            if len(values) != len(set(values)):
                raise RoomRegistryError(f"{label} values must be unique")

        known_rooms = set(room_ids)
        known_physical = set(physical_ids)
        if any(item.physical_camera_id not in known_physical for item in self.logical_streams):
            raise RoomRegistryError("logical stream references an unknown physical camera")
        if any(item.room_id not in known_rooms for item in self.logical_streams):
            raise RoomRegistryError("logical stream references an unknown room")
        referenced_rooms = {item.room_id for item in self.logical_streams}
        referenced_physical = {item.physical_camera_id for item in self.logical_streams}
        if referenced_physical != known_physical:
            raise RoomRegistryError("every physical camera must expose a logical stream")
        if referenced_rooms != known_rooms:
            raise RoomRegistryError("every room must contain a logical stream")

    @classmethod
    def from_dict(cls, value: Any) -> RoomRegistry:
        expected = {"version", "rooms", "physical_cameras", "logical_streams"}
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomRegistryError("room registry fields do not match a supported version")
        if not all(
            isinstance(value[field], list)
            for field in ("rooms", "physical_cameras", "logical_streams")
        ):
            raise RoomRegistryError("room registry collections are invalid")
        if value["version"] == 1:
            return cls._from_v1(value)
        if value["version"] != 2:
            raise RoomRegistryError("unsupported room registry version")
        return cls(
            version=value["version"],
            rooms=tuple(RoomDefinition.from_dict(item) for item in value["rooms"]),
            physical_cameras=tuple(
                PhysicalCameraDefinition.from_dict(item) for item in value["physical_cameras"]
            ),
            logical_streams=tuple(
                LogicalStreamDefinition.from_dict(item) for item in value["logical_streams"]
            ),
        )

    @classmethod
    def _from_v1(cls, value: dict[str, Any]) -> RoomRegistry:
        legacy_physical: list[tuple[str, str, str]] = []
        for item in value["physical_cameras"]:
            expected = {"physical_camera_id", "room_id", "display_name"}
            if not isinstance(item, dict) or set(item) != expected:
                raise RoomRegistryError("physical camera fields do not match v1")
            legacy_physical.append(
                (
                    _safe_id(item["physical_camera_id"], "physical_camera_id"),
                    _safe_id(item["room_id"], "physical camera room_id"),
                    _safe_display_name(item["display_name"], "physical camera display_name"),
                )
            )
        physical_ids = tuple(item[0] for item in legacy_physical)
        if len(physical_ids) != len(set(physical_ids)):
            raise RoomRegistryError("physical_camera_id values must be unique")
        rooms_by_physical = {item[0]: item[1] for item in legacy_physical}

        streams: list[LogicalStreamDefinition] = []
        for item in value["logical_streams"]:
            expected = {"camera_id", "physical_camera_id", "display_name"}
            if not isinstance(item, dict) or set(item) != expected:
                raise RoomRegistryError("logical stream fields do not match v1")
            physical_camera_id = _safe_id(
                item["physical_camera_id"],
                "logical stream physical_camera_id",
            )
            room_id = rooms_by_physical.get(physical_camera_id)
            if room_id is None:
                raise RoomRegistryError("logical stream references an unknown physical camera")
            streams.append(
                LogicalStreamDefinition(
                    camera_id=item["camera_id"],
                    physical_camera_id=physical_camera_id,
                    room_id=room_id,
                    display_name=item["display_name"],
                )
            )
        return cls(
            version=2,
            rooms=tuple(RoomDefinition.from_dict(item) for item in value["rooms"]),
            physical_cameras=tuple(
                PhysicalCameraDefinition(
                    physical_camera_id=physical_camera_id,
                    display_name=display_name,
                )
                for physical_camera_id, _room_id, display_name in legacy_physical
            ),
            logical_streams=tuple(streams),
        )

    @classmethod
    def read_private(cls, path: str | Path) -> RoomRegistry:
        return cls.from_dict(
            _read_private_json(
                path,
                label="room registry",
                max_bytes=MAX_ROOM_REGISTRY_BYTES,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "rooms": [item.to_dict() for item in self.rooms],
            "physical_cameras": [item.to_dict() for item in self.physical_cameras],
            "logical_streams": [item.to_dict() for item in self.logical_streams],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def room_for_stream(self, camera_id: str) -> RoomDefinition:
        _safe_id(camera_id, "camera_id")
        stream = next(
            (item for item in self.logical_streams if item.camera_id == camera_id),
            None,
        )
        if stream is None:
            raise RoomRegistryError("camera_id is not registered")
        return next(item for item in self.rooms if item.room_id == stream.room_id)

    def streams_for_room(self, room_id: str) -> tuple[LogicalStreamDefinition, ...]:
        _safe_id(room_id, "room_id")
        if not any(item.room_id == room_id for item in self.rooms):
            raise RoomRegistryError("room_id is not registered")
        return tuple(item for item in self.logical_streams if item.room_id == room_id)

    def summary(self) -> dict[str, object]:
        streams_per_camera = {
            item.physical_camera_id: sum(
                stream.physical_camera_id == item.physical_camera_id
                for stream in self.logical_streams
            )
            for item in self.physical_cameras
        }
        rooms_per_camera = {
            item.physical_camera_id: {
                stream.room_id
                for stream in self.logical_streams
                if stream.physical_camera_id == item.physical_camera_id
            }
            for item in self.physical_cameras
        }
        return {
            "version": 1,
            "room_count": len(self.rooms),
            "physical_camera_count": len(self.physical_cameras),
            "logical_stream_count": len(self.logical_streams),
            "multi_stream_physical_camera_count": sum(
                count > 1 for count in streams_per_camera.values()
            ),
            "multi_room_physical_camera_count": sum(
                len(room_ids) > 1 for room_ids in rooms_per_camera.values()
            ),
            "rooms": [
                {
                    "room_id": room.room_id,
                    "display_name": room.display_name,
                    "physical_camera_count": len(
                        {
                            stream.physical_camera_id
                            for stream in self.logical_streams
                            if stream.room_id == room.room_id
                        }
                    ),
                    "logical_stream_count": len(self.streams_for_room(room.room_id)),
                }
                for room in self.rooms
            ],
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }
