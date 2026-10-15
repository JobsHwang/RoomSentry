from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .pose_protocol import COCO_PERSON_KEYPOINTS

MAX_YOLO_POSE_MANIFEST_BYTES = 1024 * 1024
MAX_YOLO_POSE_WEIGHT_BYTES = 4 * 1024 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_SAFE_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,79}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class YoloPoseModelError(RuntimeError):
    pass


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise YoloPoseModelError("YOLO pose manifest contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                YoloPoseModelError("YOLO pose manifest contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise YoloPoseModelError("YOLO pose manifest is not valid JSON") from exc


def _https_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise YoloPoseModelError(f"YOLO pose manifest {field} is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise YoloPoseModelError(f"YOLO pose manifest {field} must be a credential-free HTTPS URL")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise YoloPoseModelError("YOLO pose weight file could not be read") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class YoloPoseModelManifest:
    model_id: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    source_url: str
    license_spdx: str
    license_url: str
    class_name: str
    keypoint_names: tuple[str, ...]
    keypoint_dimensions: int
    pickle_checkpoint: bool
    subject_identity_inference: bool = False
    posture_classification: bool = False
    backend: str = "ultralytics_yolo_pose"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise YoloPoseModelError("unsupported YOLO pose manifest version")
        if self.backend != "ultralytics_yolo_pose":
            raise YoloPoseModelError("unsupported YOLO pose manifest backend")
        if _SAFE_ID.fullmatch(self.model_id) is None:
            raise YoloPoseModelError("YOLO pose manifest model_id is invalid")
        if _SAFE_ID.fullmatch(self.revision) is None:
            raise YoloPoseModelError("YOLO pose manifest revision is invalid")
        if _SAFE_FILENAME.fullmatch(self.filename) is None:
            raise YoloPoseModelError("YOLO pose manifest filename is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise YoloPoseModelError("YOLO pose manifest sha256 is invalid")
        if (
            type(self.size_bytes) is not int
            or not math.isfinite(float(self.size_bytes))
            or not 1 <= self.size_bytes <= MAX_YOLO_POSE_WEIGHT_BYTES
        ):
            raise YoloPoseModelError("YOLO pose manifest size_bytes is outside the allowed range")
        _https_url(self.source_url, "source_url")
        if _SAFE_LICENSE.fullmatch(self.license_spdx) is None:
            raise YoloPoseModelError("YOLO pose manifest license_spdx is invalid")
        _https_url(self.license_url, "license_url")
        if self.class_name != "person":
            raise YoloPoseModelError("YOLO pose manifest must retain the person class")
        if self.keypoint_names != COCO_PERSON_KEYPOINTS:
            raise YoloPoseModelError("YOLO pose manifest must use the ordered COCO-17 map")
        if type(self.keypoint_dimensions) is not int or self.keypoint_dimensions != 3:
            raise YoloPoseModelError("YOLO pose manifest requires x/y/conf keypoints")
        if type(self.pickle_checkpoint) is not bool or not self.pickle_checkpoint:
            raise YoloPoseModelError("YOLO pose manifest must disclose the pickle checkpoint")
        for value, field in (
            (self.subject_identity_inference, "subject_identity_inference"),
            (self.posture_classification, "posture_classification"),
        ):
            if type(value) is not bool or value:
                raise YoloPoseModelError(f"YOLO pose manifest {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> YoloPoseModelManifest:
        expected = {
            "version",
            "backend",
            "model_id",
            "revision",
            "filename",
            "sha256",
            "size_bytes",
            "source_url",
            "license_spdx",
            "license_url",
            "class_name",
            "keypoint_names",
            "keypoint_dimensions",
            "pickle_checkpoint",
            "subject_identity_inference",
            "posture_classification",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise YoloPoseModelError("YOLO pose manifest fields do not match v1")
        keypoint_names = value["keypoint_names"]
        if not isinstance(keypoint_names, list):
            raise YoloPoseModelError("YOLO pose keypoint_names must be an array")
        return cls(
            version=value["version"],
            backend=value["backend"],
            model_id=value["model_id"],
            revision=value["revision"],
            filename=value["filename"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
            source_url=value["source_url"],
            license_spdx=value["license_spdx"],
            license_url=value["license_url"],
            class_name=value["class_name"],
            keypoint_names=tuple(keypoint_names),
            keypoint_dimensions=value["keypoint_dimensions"],
            pickle_checkpoint=value["pickle_checkpoint"],
            subject_identity_inference=value["subject_identity_inference"],
            posture_classification=value["posture_classification"],
        )

    @classmethod
    def read(cls, path: str | Path) -> YoloPoseModelManifest:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_YOLO_POSE_MANIFEST_BYTES:
                raise YoloPoseModelError("YOLO pose manifest exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except YoloPoseModelError:
            raise
        except (OSError, UnicodeError) as exc:
            raise YoloPoseModelError("YOLO pose manifest is missing or unsafe") from exc
        return cls.from_dict(_strict_json(raw))

    def verify_weight(self, path: str | Path) -> YoloPoseModelVerification:
        weight = Path(path).expanduser()
        try:
            if weight.is_symlink() or not weight.is_file():
                raise OSError
            stat = weight.stat()
        except OSError as exc:
            raise YoloPoseModelError("YOLO pose weight file is missing or unsafe") from exc
        if weight.name != self.filename:
            raise YoloPoseModelError("YOLO pose weight filename does not match the manifest")
        if stat.st_size != self.size_bytes:
            raise YoloPoseModelError("YOLO pose weight size does not match the manifest")
        actual_sha256 = _file_sha256(weight)
        if actual_sha256 != self.sha256:
            raise YoloPoseModelError("YOLO pose weight sha256 does not match the manifest")
        return YoloPoseModelVerification(manifest=self, sha256=actual_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_url": self.source_url,
            "license_spdx": self.license_spdx,
            "license_url": self.license_url,
            "class_name": self.class_name,
            "keypoint_names": list(self.keypoint_names),
            "keypoint_dimensions": self.keypoint_dimensions,
            "pickle_checkpoint": self.pickle_checkpoint,
            "subject_identity_inference": False,
            "posture_classification": False,
        }


@dataclass(frozen=True, slots=True)
class YoloPoseModelVerification:
    manifest: YoloPoseModelManifest
    sha256: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.manifest.version,
            "backend": self.manifest.backend,
            "model_id": self.manifest.model_id,
            "revision": self.manifest.revision,
            "sha256": self.sha256,
            "size_bytes": self.manifest.size_bytes,
            "license_spdx": self.manifest.license_spdx,
            "class_name": self.manifest.class_name,
            "keypoint_names": list(self.manifest.keypoint_names),
            "keypoint_dimensions": self.manifest.keypoint_dimensions,
            "pickle_checkpoint": self.manifest.pickle_checkpoint,
            "subject_identity_inference": False,
            "posture_classification": False,
            "paths_disclosed": False,
            "model_loaded": False,
            "network_used": False,
        }
