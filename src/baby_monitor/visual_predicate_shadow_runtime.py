from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baby_monitor.adapters.visual_predicate import (
    ConservativeVisualPredicateAdapter,
    ResidentCommandVisualPredicateAdapter,
    VisualPredicateAdapter,
    VisualPredicateFrame,
)
from baby_monitor.domain import CandidateEvent, Detection, SignalEvent
from baby_monitor.visual_predicate import VisualPredicate
from baby_monitor.visual_predicate_shadow import VisualPredicateShadowError

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_MAX_CANDIDATE_BYTES = 1024 * 1024
_MAX_FRAME_BYTES = 25 * 1024 * 1024
_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_DEFAULT_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"


class VisualPredicateShadowRuntimeError(VisualPredicateShadowError):
    """Path-free configuration or private input error."""


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowRuntimeSettings:
    enabled: bool = False
    backend: str = "conservative"
    command: tuple[str, ...] = ()
    predicates: tuple[VisualPredicate, ...] = (
        VisualPredicate.FACE_VISIBILITY,
        VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
    )
    hardware_signal_adapters: tuple[str, ...] = ("home_assistant",)
    min_person_confidence: float = 0.25
    model_id: str = "microsoft.Mage-VL"
    model_revision: str = _DEFAULT_REVISION
    prompt_version: str = "mage_atomic_visible_v1"
    max_jobs: int = 32
    idle_exit_seconds: float = 30.0
    max_runtime_seconds: float = 3600.0
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 180.0
    health_path: Path | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise VisualPredicateShadowRuntimeError("visual predicate shadow enabled is invalid")
        if self.backend not in {"conservative", "resident_command"}:
            raise VisualPredicateShadowRuntimeError("visual predicate shadow backend is invalid")
        if not isinstance(self.command, tuple) or any(
            not isinstance(part, str)
            or not part
            or any(character in part for character in ("\x00", "\r", "\n"))
            or "{request}" in part
            for part in self.command
        ):
            raise VisualPredicateShadowRuntimeError("visual predicate shadow command is invalid")
        if self.backend == "conservative" and self.command:
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow conservative backend cannot have a command"
            )
        if self.enabled and self.backend == "resident_command":
            if not self.command:
                raise VisualPredicateShadowRuntimeError(
                    "visual predicate shadow resident command is required"
                )
            executable = Path(self.command[0])
            if not executable.is_absolute() or not executable.is_file():
                raise VisualPredicateShadowRuntimeError(
                    "visual predicate shadow resident executable is unavailable"
                )
        if (
            not isinstance(self.predicates, tuple)
            or not self.predicates
            or len(set(self.predicates)) != len(self.predicates)
            or any(not isinstance(item, VisualPredicate) for item in self.predicates)
        ):
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow predicates are invalid"
            )
        if (
            not isinstance(self.hardware_signal_adapters, tuple)
            or len(set(self.hardware_signal_adapters)) != len(self.hardware_signal_adapters)
            or any(
                not isinstance(item, str) or _SAFE_ID.fullmatch(item) is None
                for item in self.hardware_signal_adapters
            )
        ):
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow signal adapters are invalid"
            )
        if (
            type(self.min_person_confidence) not in {int, float}
            or not math.isfinite(self.min_person_confidence)
            or not 0.0 <= float(self.min_person_confidence) <= 1.0
        ):
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow person threshold is invalid"
            )
        if (
            _SAFE_ID.fullmatch(self.model_id) is None
            or _GIT_SHA.fullmatch(self.model_revision) is None
            or _SAFE_ID.fullmatch(self.prompt_version) is None
        ):
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow model binding is invalid"
            )
        if type(self.max_jobs) is not int or not 1 <= self.max_jobs <= 256:
            raise VisualPredicateShadowRuntimeError("visual predicate shadow max jobs is invalid")
        for value, field_name, maximum in (
            (self.idle_exit_seconds, "idle timeout", 3600.0),
            (self.max_runtime_seconds, "maximum runtime", 86_400.0),
            (self.startup_timeout_seconds, "startup timeout", 3600.0),
            (self.request_timeout_seconds, "request timeout", 3600.0),
        ):
            if (
                type(value) not in {int, float}
                or not math.isfinite(value)
                or not 1.0 <= float(value) <= maximum
            ):
                raise VisualPredicateShadowRuntimeError(
                    f"visual predicate shadow {field_name} is invalid"
                )
        if self.health_path is not None and not self.health_path.is_absolute():
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow health path is invalid"
            )


def _as_bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    raise VisualPredicateShadowRuntimeError("visual predicate shadow enabled is invalid")


def _as_number(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise VisualPredicateShadowRuntimeError(f"visual predicate shadow {field} is invalid")
    return float(value)


def load_visual_predicate_shadow_runtime_settings(
    path: str | Path,
) -> VisualPredicateShadowRuntimeSettings:
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        raise VisualPredicateShadowRuntimeError(
            "visual predicate shadow configuration is unavailable"
        ) from None
    raw = data.get("visual_predicate_shadow", {})
    allowed = {
        "enabled",
        "backend",
        "command",
        "predicates",
        "hardware_signal_adapters",
        "min_person_confidence",
        "model_id",
        "model_revision",
        "prompt_version",
        "max_jobs",
        "idle_exit_seconds",
        "max_runtime_seconds",
        "startup_timeout_seconds",
        "request_timeout_seconds",
        "health_path",
    }
    if not isinstance(raw, dict) or not set(raw).issubset(allowed):
        raise VisualPredicateShadowRuntimeError(
            "visual predicate shadow configuration fields are invalid"
        )
    command = raw.get("command", [])
    for field_name in ("backend", "model_id", "model_revision", "prompt_version"):
        if field_name in raw and not isinstance(raw[field_name], str):
            raise VisualPredicateShadowRuntimeError(
                "visual predicate shadow configuration fields are invalid"
            )
    predicates = raw.get(
        "predicates",
        ["face_visibility", "covering_near_face_visible"],
    )
    adapters = raw.get("hardware_signal_adapters", ["home_assistant"])
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise VisualPredicateShadowRuntimeError("visual predicate shadow command is invalid")
    if not isinstance(predicates, list) or not all(isinstance(item, str) for item in predicates):
        raise VisualPredicateShadowRuntimeError("visual predicate shadow predicates are invalid")
    if not isinstance(adapters, list) or not all(isinstance(item, str) for item in adapters):
        raise VisualPredicateShadowRuntimeError(
            "visual predicate shadow signal adapters are invalid"
        )
    try:
        parsed_predicates = tuple(VisualPredicate(item) for item in predicates)
    except ValueError:
        raise VisualPredicateShadowRuntimeError(
            "visual predicate shadow predicates are invalid"
        ) from None
    health_value = raw.get("health_path", "")
    if not isinstance(health_value, str):
        raise VisualPredicateShadowRuntimeError("visual predicate shadow health path is invalid")
    return VisualPredicateShadowRuntimeSettings(
        enabled=_as_bool(raw.get("enabled", False)),
        backend=str(raw.get("backend", "conservative")),
        command=tuple(command),
        predicates=parsed_predicates,
        hardware_signal_adapters=tuple(adapters),
        min_person_confidence=_as_number(
            raw.get("min_person_confidence", 0.25), "person threshold"
        ),
        model_id=str(raw.get("model_id", "microsoft.Mage-VL")),
        model_revision=str(raw.get("model_revision", _DEFAULT_REVISION)),
        prompt_version=str(raw.get("prompt_version", "mage_atomic_visible_v1")),
        max_jobs=raw.get("max_jobs", 32),
        idle_exit_seconds=_as_number(raw.get("idle_exit_seconds", 30.0), "idle timeout"),
        max_runtime_seconds=_as_number(raw.get("max_runtime_seconds", 3600.0), "maximum runtime"),
        startup_timeout_seconds=_as_number(
            raw.get("startup_timeout_seconds", 180.0), "startup timeout"
        ),
        request_timeout_seconds=_as_number(
            raw.get("request_timeout_seconds", 180.0), "request timeout"
        ),
        health_path=(Path(health_value).expanduser() if health_value else None),
    )


def build_visual_predicate_shadow_adapter(
    settings: VisualPredicateShadowRuntimeSettings,
) -> VisualPredicateAdapter:
    if not isinstance(settings, VisualPredicateShadowRuntimeSettings) or not settings.enabled:
        raise VisualPredicateShadowRuntimeError("visual predicate shadow is disabled")
    if settings.backend == "conservative":
        return ConservativeVisualPredicateAdapter(
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            prompt_version=settings.prompt_version,
        )
    return ResidentCommandVisualPredicateAdapter(
        command=settings.command,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
        prompt_version=settings.prompt_version,
        max_jobs=settings.max_jobs,
        idle_exit_seconds=settings.idle_exit_seconds,
        max_runtime_seconds=settings.max_runtime_seconds,
        startup_timeout_seconds=settings.startup_timeout_seconds,
        request_timeout_seconds=settings.request_timeout_seconds,
        health_path=settings.health_path,
    )


def _read_private_file(path: Path, *, maximum_bytes: int, context: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise VisualPredicateShadowRuntimeError(
            f"visual predicate shadow {context} is unavailable"
        ) from None
    if not path.is_absolute() or path.is_symlink() or resolved != path:
        raise VisualPredicateShadowRuntimeError(f"visual predicate shadow {context} is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise VisualPredicateShadowRuntimeError(
            f"visual predicate shadow {context} is unavailable"
        ) from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= maximum_bytes
            or (
                os.name == "posix"
                and (metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077 != 0)
            )
        ):
            raise VisualPredicateShadowRuntimeError(f"visual predicate shadow {context} is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise VisualPredicateShadowRuntimeError(
                    f"visual predicate shadow {context} is unsafe"
                )
            chunks.append(chunk)
        if total != metadata.st_size:
            raise VisualPredicateShadowRuntimeError(f"visual predicate shadow {context} is unsafe")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_private_shadow_candidate(path: str | Path) -> CandidateEvent:
    payload = _read_private_file(
        Path(path).expanduser(),
        maximum_bytes=_MAX_CANDIDATE_BYTES,
        context="candidate",
    )
    try:

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError
                result[key] = item
            return result

        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(value, dict):
            raise ValueError
        allowed = {
            "event_id",
            "camera_id",
            "start_seconds",
            "end_seconds",
            "detections",
            "signals",
            "detector_id",
        }
        if not set(value).issubset(allowed) or not {
            "event_id",
            "camera_id",
            "start_seconds",
            "end_seconds",
            "detections",
        }.issubset(value):
            raise ValueError
        if (
            not isinstance(value["event_id"], str)
            or _SAFE_ID.fullmatch(value["event_id"]) is None
            or not isinstance(value["camera_id"], str)
            or _SAFE_ID.fullmatch(value["camera_id"]) is None
            or type(value["start_seconds"]) not in {int, float}
            or type(value["end_seconds"]) not in {int, float}
            or not math.isfinite(value["start_seconds"])
            or not math.isfinite(value["end_seconds"])
            or not 0.0 <= float(value["start_seconds"]) <= float(value["end_seconds"])
            or not isinstance(value["detections"], list)
            or not isinstance(value.get("signals", []), list)
        ):
            raise ValueError
        detections = tuple(
            Detection(
                timestamp_seconds=item["timestamp_seconds"],
                label=item["label"],
                confidence=item["confidence"],
                bbox_xyxy=tuple(item["bbox_xyxy"]),
            )
            for item in value["detections"]
            if isinstance(item, dict)
            and set(item) == {"timestamp_seconds", "label", "confidence", "bbox_xyxy"}
        )
        if len(detections) != len(value["detections"]):
            raise ValueError
        signals = tuple(SignalEvent.from_dict(item) for item in value.get("signals", []))
        if any(signal.camera_id != value["camera_id"] for signal in signals):
            raise ValueError
        return CandidateEvent(
            event_id=value["event_id"],
            camera_id=value["camera_id"],
            start_seconds=value["start_seconds"],
            end_seconds=value["end_seconds"],
            detections=detections,
            signals=signals,
            detector_id=value.get("detector_id"),
        )
    except (AttributeError, KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        raise VisualPredicateShadowRuntimeError(
            "visual predicate shadow candidate is invalid"
        ) from None


def admit_private_shadow_frame(specification: str) -> VisualPredicateFrame:
    if not isinstance(specification, str) or "=" not in specification:
        raise VisualPredicateShadowRuntimeError("visual predicate shadow frame is invalid")
    frame_id, raw_path = specification.split("=", 1)
    path = Path(raw_path).expanduser()
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if _SAFE_ID.fullmatch(frame_id) is None or media_type is None:
        raise VisualPredicateShadowRuntimeError("visual predicate shadow frame is invalid")
    payload = _read_private_file(path, maximum_bytes=_MAX_FRAME_BYTES, context="frame")
    return VisualPredicateFrame(
        frame_id=frame_id,
        sha256=hashlib.sha256(payload).hexdigest(),
        media_type=media_type,
        path=str(path),
    )
