from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_YOLO_MANIFEST_BYTES = 1024 * 1024
MAX_YOLO_WEIGHT_BYTES = 4 * 1024 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_SAFE_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,79}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class YoloModelError(RuntimeError):
    pass


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise YoloModelError("YOLO manifest contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                YoloModelError("YOLO manifest contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise YoloModelError("YOLO manifest is not valid JSON") from exc


def _https_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise YoloModelError(f"YOLO manifest {field} is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise YoloModelError(f"YOLO manifest {field} must be a credential-free HTTPS URL")
    return value


def _label(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 80
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise YoloModelError("YOLO manifest contains an invalid label")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise YoloModelError("YOLO weight file could not be read") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class YoloModelManifest:
    model_id: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    source_url: str
    license_spdx: str
    license_url: str
    labels: tuple[str, ...]
    subject_identity_inference: bool = False
    backend: str = "ultralytics_yolo"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise YoloModelError("unsupported YOLO manifest version")
        if self.backend != "ultralytics_yolo":
            raise YoloModelError("unsupported YOLO manifest backend")
        if _SAFE_ID.fullmatch(self.model_id) is None:
            raise YoloModelError("YOLO manifest model_id is invalid")
        if _SAFE_ID.fullmatch(self.revision) is None:
            raise YoloModelError("YOLO manifest revision is invalid")
        if _SAFE_FILENAME.fullmatch(self.filename) is None:
            raise YoloModelError("YOLO manifest filename is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise YoloModelError("YOLO manifest sha256 is invalid")
        if (
            type(self.size_bytes) is not int
            or not math.isfinite(float(self.size_bytes))
            or not 1 <= self.size_bytes <= MAX_YOLO_WEIGHT_BYTES
        ):
            raise YoloModelError("YOLO manifest size_bytes is outside the allowed range")
        _https_url(self.source_url, "source_url")
        if _SAFE_LICENSE.fullmatch(self.license_spdx) is None:
            raise YoloModelError("YOLO manifest license_spdx is invalid")
        _https_url(self.license_url, "license_url")
        if not isinstance(self.labels, tuple) or not self.labels:
            raise YoloModelError("YOLO manifest labels must be a non-empty array")
        if len(self.labels) > 1000:
            raise YoloModelError("YOLO manifest labels exceed the 1000-label limit")
        for value in self.labels:
            _label(value)
        if len(set(self.labels)) != len(self.labels):
            raise YoloModelError("YOLO manifest labels must be unique")
        if type(self.subject_identity_inference) is not bool:
            raise YoloModelError("YOLO manifest subject_identity_inference must be false")
        if self.subject_identity_inference:
            raise YoloModelError("YOLO object labels cannot authorize subject identity inference")

    @classmethod
    def from_dict(cls, value: Any) -> YoloModelManifest:
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
            "labels",
            "subject_identity_inference",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise YoloModelError("YOLO manifest fields do not match v1")
        labels = value["labels"]
        if not isinstance(labels, list):
            raise YoloModelError("YOLO manifest labels must be an array")
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
            labels=tuple(labels),
            subject_identity_inference=value["subject_identity_inference"],
        )

    @classmethod
    def read(cls, path: str | Path) -> YoloModelManifest:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_YOLO_MANIFEST_BYTES:
                raise YoloModelError("YOLO manifest exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except YoloModelError:
            raise
        except (OSError, UnicodeError) as exc:
            raise YoloModelError("YOLO manifest is missing or unsafe") from exc
        return cls.from_dict(_strict_json(raw))

    def verify_weight(self, path: str | Path) -> YoloModelVerification:
        weight = Path(path).expanduser()
        try:
            if weight.is_symlink() or not weight.is_file():
                raise OSError
            stat = weight.stat()
        except OSError as exc:
            raise YoloModelError("YOLO weight file is missing or unsafe") from exc
        if weight.name != self.filename:
            raise YoloModelError("YOLO weight filename does not match the manifest")
        if stat.st_size != self.size_bytes:
            raise YoloModelError("YOLO weight size does not match the manifest")
        actual_sha256 = file_sha256(weight)
        if actual_sha256 != self.sha256:
            raise YoloModelError("YOLO weight sha256 does not match the manifest")
        return YoloModelVerification(manifest=self, sha256=actual_sha256)

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
            "labels": list(self.labels),
            "subject_identity_inference": self.subject_identity_inference,
        }


@dataclass(frozen=True, slots=True)
class YoloModelVerification:
    manifest: YoloModelManifest
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
            "labels": list(self.manifest.labels),
            "subject_identity_inference": False,
            "paths_disclosed": False,
            "model_loaded": False,
            "network_used": False,
        }
