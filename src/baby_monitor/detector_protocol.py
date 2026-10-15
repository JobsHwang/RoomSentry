from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .detector_sampling import DetectorSamplingError, DetectorSamplingLedger
from .domain import Detection
from .frame_transform import validate_image_size, validate_rotation_degrees


class DetectorProtocolError(ValueError):
    """A bounded detector subprocess request or result is not valid."""


MAX_DETECTOR_RESULT_BYTES = 8 * 1024 * 1024
MAX_DETECTOR_RESULT_DETECTIONS = 100_000
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DetectorProtocolError("detector JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DetectorProtocolError("detector JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise DetectorProtocolError("detector JSON is invalid") from exc


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DetectorProtocolError(f"detector {field} is invalid")
    return value


def _path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DetectorProtocolError(f"detector {field} is invalid")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise DetectorProtocolError(f"detector {field} is unsafe")
    return str(path)


def _labels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 1000:
        raise DetectorProtocolError("detector labels are invalid")
    labels: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or len(item) > 80
            or any(character in item for character in ("\x00", "\r", "\n"))
        ):
            raise DetectorProtocolError("detector labels are invalid")
        labels.append(item)
    if len(set(labels)) != len(labels):
        raise DetectorProtocolError("detector labels are invalid")
    return tuple(labels)


def _bounded_float(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise DetectorProtocolError(f"detector {field} is invalid")
    return float(value)


def _detection_to_dict(detection: Detection) -> dict[str, object]:
    return {
        "timestamp_seconds": detection.timestamp_seconds,
        "label": detection.label,
        "confidence": detection.confidence,
        "bbox_xyxy": list(detection.bbox_xyxy),
    }


def _detection_from_dict(value: Any) -> Detection:
    if not isinstance(value, dict) or set(value) != {
        "timestamp_seconds",
        "label",
        "confidence",
        "bbox_xyxy",
    }:
        raise DetectorProtocolError("detector detection fields do not match v1")
    bbox = value["bbox_xyxy"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise DetectorProtocolError("detector detection bbox is invalid")
    try:
        return Detection(
            timestamp_seconds=value["timestamp_seconds"],
            label=value["label"],
            confidence=value["confidence"],
            bbox_xyxy=tuple(bbox),
        )
    except (TypeError, ValueError) as exc:
        raise DetectorProtocolError("detector detection is invalid") from exc


@dataclass(frozen=True, slots=True)
class DetectorScanRequest:
    """Private request sent to one short-lived detector process."""

    camera_id: str
    detector_adapter: str
    source_path: str
    model_path: str
    manifest_path: str
    model_id: str
    model_revision: str
    model_sha256: str
    labels: tuple[str, ...]
    confidence: float
    sample_fps: float
    device: str
    image_size: int = 640
    rotation_degrees: int = 0
    version: int = 2

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 2:
            raise DetectorProtocolError("unsupported detector request version")
        _id(self.camera_id, "camera_id")
        _id(self.detector_adapter, "detector_adapter")
        _path(self.source_path, "source_path")
        _path(self.model_path, "model_path")
        _path(self.manifest_path, "manifest_path")
        _id(self.model_id, "model_id")
        _id(self.model_revision, "model_revision")
        if _SHA256.fullmatch(self.model_sha256) is None:
            raise DetectorProtocolError("detector model_sha256 is invalid")
        if self.labels != _labels(list(self.labels)):
            raise DetectorProtocolError("detector labels are invalid")
        _bounded_float(self.confidence, "confidence", minimum=0.0, maximum=1.0)
        _bounded_float(self.sample_fps, "sample_fps", minimum=0.01, maximum=10.0)
        if self.device != "cpu" and re.fullmatch(r"\d+", self.device) is None:
            raise DetectorProtocolError("detector device is invalid")
        try:
            validate_image_size(self.image_size)
            validate_rotation_degrees(self.rotation_degrees)
        except ValueError as exc:
            raise DetectorProtocolError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "source_path": self.source_path,
            "model_path": self.model_path,
            "manifest_path": self.manifest_path,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "labels": list(self.labels),
            "confidence": self.confidence,
            "sample_fps": self.sample_fps,
            "device": self.device,
            "image_size": self.image_size,
            "rotation_degrees": self.rotation_degrees,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Any) -> DetectorScanRequest:
        legacy_expected = {
            "version",
            "camera_id",
            "detector_adapter",
            "source_path",
            "model_path",
            "manifest_path",
            "model_id",
            "model_revision",
            "model_sha256",
            "labels",
            "confidence",
            "sample_fps",
            "device",
        }
        current_expected = legacy_expected | {"image_size", "rotation_degrees"}
        if not isinstance(value, dict):
            raise DetectorProtocolError("detector request fields do not match v2")
        if value.get("version") == 1 and set(value) == legacy_expected:
            value = {
                **value,
                "version": 2,
                "image_size": 640,
                "rotation_degrees": 0,
            }
        elif value.get("version") != 2 or set(value) != current_expected:
            raise DetectorProtocolError("detector request fields do not match v2")
        labels = _labels(value["labels"])
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            detector_adapter=value["detector_adapter"],
            source_path=value["source_path"],
            model_path=value["model_path"],
            manifest_path=value["manifest_path"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            model_sha256=value["model_sha256"],
            labels=labels,
            confidence=value["confidence"],
            sample_fps=value["sample_fps"],
            device=value["device"],
            image_size=value["image_size"],
            rotation_degrees=value["rotation_degrees"],
        )

    @classmethod
    def from_json(cls, raw: str) -> DetectorScanRequest:
        return cls.from_dict(_strict_json(raw))


@dataclass(frozen=True, slots=True)
class DetectorScanResult:
    """Path-free result returned by one detector subprocess."""

    camera_id: str
    detector_adapter: str
    model_id: str
    model_revision: str
    model_sha256: str
    detections: tuple[Detection, ...]
    subject_identity_inference: bool = False
    paths_disclosed: bool = False
    version: int = 1
    sampling: DetectorSamplingLedger | None = None

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in (1, 2):
            raise DetectorProtocolError("unsupported detector result version")
        _id(self.camera_id, "camera_id")
        _id(self.detector_adapter, "detector_adapter")
        _id(self.model_id, "model_id")
        _id(self.model_revision, "model_revision")
        if _SHA256.fullmatch(self.model_sha256) is None:
            raise DetectorProtocolError("detector model_sha256 is invalid")
        if (
            not isinstance(self.detections, tuple)
            or len(self.detections) > MAX_DETECTOR_RESULT_DETECTIONS
            or any(not isinstance(item, Detection) for item in self.detections)
        ):
            raise DetectorProtocolError("detector detections are invalid or too large")
        if any(
            current.timestamp_seconds < previous.timestamp_seconds
            for previous, current in zip(self.detections, self.detections[1:], strict=False)
        ):
            raise DetectorProtocolError("detector detections are not chronological")
        if type(self.subject_identity_inference) is not bool or self.subject_identity_inference:
            raise DetectorProtocolError("detector identity inference must be false")
        if type(self.paths_disclosed) is not bool or self.paths_disclosed:
            raise DetectorProtocolError("detector paths_disclosed must be false")
        if self.version == 1 and self.sampling is not None:
            raise DetectorProtocolError("v1 detector result cannot contain sampling evidence")
        if self.version == 2:
            if not isinstance(self.sampling, DetectorSamplingLedger):
                raise DetectorProtocolError("v2 detector result requires sampling evidence")
            try:
                self.sampling.validate_detections(self.detections)
            except DetectorSamplingError as exc:
                raise DetectorProtocolError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        value = {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "detections": [_detection_to_dict(item) for item in self.detections],
            "subject_identity_inference": False,
            "paths_disclosed": False,
        }
        if self.version == 2:
            value["sampling"] = self.sampling.to_dict()
        return value

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def at_confidence_threshold(self, minimum: float) -> DetectorScanResult:
        """Filter stored boxes while preserving/rebinding actual sampled frames.

        Zero retained boxes mean no evidence at this threshold, not an absent
        person. This is an offline filter, never a new model execution.
        """
        minimum = _bounded_float(minimum, "minimum confidence", minimum=0, maximum=1)
        kept = tuple(item for item in self.detections if item.confidence >= minimum)
        if self.sampling is None:
            return replace(self, detections=kept)
        samples = []
        offset = 0
        for sample in self.sampling.samples:
            count = sum(
                item.confidence >= minimum
                for item in self.detections[
                    sample.detection_start : sample.detection_start + sample.detection_count
                ]
            )
            samples.append(replace(sample, detection_start=offset, detection_count=count))
            offset += count
        return replace(
            self, detections=kept, sampling=replace(self.sampling, samples=tuple(samples))
        )

    @classmethod
    def read(cls, path: str | Path) -> DetectorScanResult:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_DETECTOR_RESULT_BYTES:
                raise DetectorProtocolError("detector result file is too large")
            raw = source.read_text(encoding="utf-8")
        except DetectorProtocolError:
            raise
        except (OSError, UnicodeError) as exc:
            raise DetectorProtocolError("detector result file is missing or unsafe") from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise DetectorProtocolError("detector result destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_DETECTOR_RESULT_BYTES:
            raise DetectorProtocolError("detector result file is too large")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            raise FileExistsError("detector result output already exists") from None
        except OSError as exc:
            raise DetectorProtocolError("detector result output could not be created") from exc
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise DetectorProtocolError("detector result output could not be written") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @classmethod
    def from_dict(cls, value: Any) -> DetectorScanResult:
        expected = {
            "version",
            "camera_id",
            "detector_adapter",
            "model_id",
            "model_revision",
            "model_sha256",
            "detections",
            "subject_identity_inference",
            "paths_disclosed",
        }
        if isinstance(value, dict) and value.get("version") == 2:
            expected.add("sampling")
        if not isinstance(value, dict) or set(value) != expected:
            raise DetectorProtocolError("detector result fields do not match the version")
        detections = value["detections"]
        if not isinstance(detections, list) or len(detections) > MAX_DETECTOR_RESULT_DETECTIONS:
            raise DetectorProtocolError("detector result detections are invalid")
        try:
            sampling = (
                DetectorSamplingLedger.from_dict(value["sampling"]) if "sampling" in value else None
            )
        except DetectorSamplingError as exc:
            raise DetectorProtocolError(str(exc)) from exc
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            detector_adapter=value["detector_adapter"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            model_sha256=value["model_sha256"],
            detections=tuple(_detection_from_dict(item) for item in detections),
            subject_identity_inference=value["subject_identity_inference"],
            paths_disclosed=value["paths_disclosed"],
            sampling=sampling,
        )

    @classmethod
    def from_json(cls, raw: str) -> DetectorScanResult:
        if len(raw.encode("utf-8")) > MAX_DETECTOR_RESULT_BYTES:
            raise DetectorProtocolError("detector result is too large")
        return cls.from_dict(_strict_json(raw))


@dataclass(frozen=True, slots=True)
class AdmittedDetectorScan:
    """Path-free detector result plus the exact admitted source and scan config."""

    source_admission_sha256: str
    labels: tuple[str, ...]
    confidence_threshold: float
    requested_sample_fps: float
    result: DetectorScanResult
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in (1, 2):
            raise DetectorProtocolError("unsupported admitted detector scan version")
        if (
            not isinstance(self.source_admission_sha256, str)
            or _SHA256.fullmatch(self.source_admission_sha256) is None
        ):
            raise DetectorProtocolError("admitted detector scan source admission hash is invalid")
        if self.labels != _labels(list(self.labels)):
            raise DetectorProtocolError("admitted detector scan labels are invalid")
        _bounded_float(
            self.confidence_threshold,
            "confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        _bounded_float(
            self.requested_sample_fps,
            "requested_sample_fps",
            minimum=0.01,
            maximum=10.0,
        )
        if not isinstance(self.result, DetectorScanResult):
            raise DetectorProtocolError("admitted detector scan result is invalid")
        if self.result.version != self.version:
            raise DetectorProtocolError("admitted scan and detector result versions must match")
        if (
            self.result.sampling is not None
            and self.result.sampling.requested_sample_fps != self.requested_sample_fps
        ):
            raise DetectorProtocolError(
                "admitted scan sampling rate conflicts with execution evidence"
            )
        if any(
            detection.label not in self.labels or detection.confidence < self.confidence_threshold
            for detection in self.result.detections
        ):
            raise DetectorProtocolError(
                "admitted detector scan detections conflict with scan config"
            )

    @classmethod
    def from_dict(cls, value: Any) -> AdmittedDetectorScan:
        expected = {
            "version",
            "source_admission_sha256",
            "labels",
            "confidence_threshold",
            "requested_sample_fps",
            "result",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise DetectorProtocolError("admitted detector scan fields do not match v1")
        return cls(
            version=value["version"],
            source_admission_sha256=value["source_admission_sha256"],
            labels=_labels(value["labels"]),
            confidence_threshold=value["confidence_threshold"],
            requested_sample_fps=value["requested_sample_fps"],
            result=DetectorScanResult.from_dict(value["result"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> AdmittedDetectorScan:
        if len(raw.encode("utf-8")) > MAX_DETECTOR_RESULT_BYTES:
            raise DetectorProtocolError("admitted detector scan is too large")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> AdmittedDetectorScan:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_DETECTOR_RESULT_BYTES:
                raise DetectorProtocolError("admitted detector scan is too large")
            raw = source.read_text(encoding="utf-8")
        except DetectorProtocolError:
            raise
        except (OSError, UnicodeError) as exc:
            raise DetectorProtocolError("admitted detector scan is missing or unsafe") from exc
        return cls.from_json(raw)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_admission_sha256": self.source_admission_sha256,
            "labels": list(self.labels),
            "confidence_threshold": self.confidence_threshold,
            "requested_sample_fps": self.requested_sample_fps,
            "result": self.result.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise DetectorProtocolError("admitted detector scan destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_DETECTOR_RESULT_BYTES:
            raise DetectorProtocolError("admitted detector scan is too large")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            raise FileExistsError("admitted detector scan output already exists") from None
        except OSError as exc:
            raise DetectorProtocolError(
                "admitted detector scan output could not be created"
            ) from exc
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise DetectorProtocolError(
                "admitted detector scan output could not be written"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
