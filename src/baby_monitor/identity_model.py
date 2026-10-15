from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_FACE_MODEL_MANIFEST_BYTES = 1024 * 1024
MAX_FACE_MODEL_BYTES = 1024 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_SAFE_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,79}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_ROLES = frozenset({"detector", "embedding"})
_REVIEWED_FACE_MANIFEST_SHA256 = frozenset(
    {
        "f51d75b3662b8c14c0667f27702f15b9942fd0d46bf63762f50a596b655b0a65",
        "3e2f22c3861bcac0decba3eaddf4cf0f5b33f747a2e4b78e24d493ccbb93eceb",
    }
)


class FaceModelError(RuntimeError):
    pass


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise FaceModelError("face model manifest contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                FaceModelError("face model manifest contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise FaceModelError("face model manifest is not valid JSON") from exc


def _credential_free_https(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise FaceModelError(f"face model manifest {field} is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise FaceModelError(f"face model manifest {field} must be a credential-free HTTPS URL")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise FaceModelError("face model file could not be read") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FaceModelManifest:
    role: str
    model_id: str
    upstream_revision: str
    filename: str
    sha256: str
    size_bytes: int
    source_url: str
    license_spdx: str
    license_url: str
    input_width: int
    input_height: int
    input_color_space: str
    landmark_count: int
    embedding_dimension: int | None
    preprocessing_revision: str
    unsafe_pickle: bool
    automatic_download: bool
    identity_decision_authority: bool
    backend: str = "opencv_dnn"
    model_format: str = "ONNX"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise FaceModelError("unsupported face model manifest version")
        if self.backend != "opencv_dnn":
            raise FaceModelError("unsupported face model manifest backend")
        if self.model_format != "ONNX":
            raise FaceModelError("face model manifest must use ONNX")
        if self.role not in _ROLES:
            raise FaceModelError("face model manifest role is invalid")
        if _SAFE_ID.fullmatch(self.model_id) is None:
            raise FaceModelError("face model manifest model_id is invalid")
        if _GIT_SHA.fullmatch(self.upstream_revision) is None:
            raise FaceModelError("face model manifest upstream_revision is not pinned")
        if _SAFE_FILENAME.fullmatch(self.filename) is None:
            raise FaceModelError("face model manifest filename is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise FaceModelError("face model manifest sha256 is invalid")
        if (
            type(self.size_bytes) is not int
            or not math.isfinite(float(self.size_bytes))
            or not 1 <= self.size_bytes <= MAX_FACE_MODEL_BYTES
        ):
            raise FaceModelError("face model manifest size_bytes is outside the allowed range")
        _credential_free_https(self.source_url, "source_url")
        if _SAFE_LICENSE.fullmatch(self.license_spdx) is None:
            raise FaceModelError("face model manifest license_spdx is invalid")
        _credential_free_https(self.license_url, "license_url")
        for field, value in (
            ("input_width", self.input_width),
            ("input_height", self.input_height),
        ):
            if type(value) is not int or not 16 <= value <= 8192:
                raise FaceModelError(f"face model manifest {field} is invalid")
        if self.input_color_space != "BGR":
            raise FaceModelError("face model manifest input_color_space must be BGR")
        if type(self.landmark_count) is not int or self.landmark_count != 5:
            raise FaceModelError("face model manifest requires five facial landmarks")
        if self.role == "detector" and self.embedding_dimension is not None:
            raise FaceModelError("face detector manifest cannot declare an embedding dimension")
        if self.role == "embedding" and (
            type(self.embedding_dimension) is not int or not 1 <= self.embedding_dimension <= 4096
        ):
            raise FaceModelError("face embedding manifest dimension is invalid")
        if _SAFE_ID.fullmatch(self.preprocessing_revision) is None:
            raise FaceModelError("face model preprocessing_revision is invalid")
        for field, value in (
            ("unsafe_pickle", self.unsafe_pickle),
            ("automatic_download", self.automatic_download),
            ("identity_decision_authority", self.identity_decision_authority),
        ):
            if type(value) is not bool or value:
                raise FaceModelError(f"face model manifest {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> FaceModelManifest:
        expected = {
            "version",
            "backend",
            "model_format",
            "role",
            "model_id",
            "upstream_revision",
            "filename",
            "sha256",
            "size_bytes",
            "source_url",
            "license_spdx",
            "license_url",
            "input_width",
            "input_height",
            "input_color_space",
            "landmark_count",
            "embedding_dimension",
            "preprocessing_revision",
            "unsafe_pickle",
            "automatic_download",
            "identity_decision_authority",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise FaceModelError("face model manifest fields do not match v1")
        return cls(**value)

    @classmethod
    def read(cls, path: str | Path) -> FaceModelManifest:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_FACE_MODEL_MANIFEST_BYTES:
                raise FaceModelError("face model manifest exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except FaceModelError:
            raise
        except (OSError, UnicodeError) as exc:
            raise FaceModelError("face model manifest is missing or unsafe") from exc
        return cls.from_dict(_strict_json(raw))

    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def require_reviewed(self) -> str:
        digest = self.canonical_sha256()
        if digest not in _REVIEWED_FACE_MANIFEST_SHA256:
            raise FaceModelError("face model manifest is not in the built-in reviewed allowlist")
        return digest

    def verify_model(self, path: str | Path) -> FaceModelVerification:
        model = Path(path).expanduser()
        try:
            if model.is_symlink() or not model.is_file():
                raise OSError
            stat = model.stat()
        except OSError as exc:
            raise FaceModelError("face model file is missing or unsafe") from exc
        if model.name != self.filename:
            raise FaceModelError("face model filename does not match the manifest")
        if stat.st_size != self.size_bytes:
            raise FaceModelError("face model size does not match the manifest")
        actual_sha256 = _file_sha256(model)
        if actual_sha256 != self.sha256:
            raise FaceModelError("face model sha256 does not match the manifest")
        return FaceModelVerification(manifest=self, sha256=actual_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "backend": self.backend,
            "model_format": self.model_format,
            "role": self.role,
            "model_id": self.model_id,
            "upstream_revision": self.upstream_revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_url": self.source_url,
            "license_spdx": self.license_spdx,
            "license_url": self.license_url,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_color_space": self.input_color_space,
            "landmark_count": self.landmark_count,
            "embedding_dimension": self.embedding_dimension,
            "preprocessing_revision": self.preprocessing_revision,
            "unsafe_pickle": False,
            "automatic_download": False,
            "identity_decision_authority": False,
        }


@dataclass(frozen=True, slots=True)
class FaceModelVerification:
    manifest: FaceModelManifest
    sha256: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.manifest.version,
            "backend": self.manifest.backend,
            "model_format": self.manifest.model_format,
            "role": self.manifest.role,
            "model_id": self.manifest.model_id,
            "upstream_revision": self.manifest.upstream_revision,
            "sha256": self.sha256,
            "size_bytes": self.manifest.size_bytes,
            "license_spdx": self.manifest.license_spdx,
            "embedding_dimension": self.manifest.embedding_dimension,
            "preprocessing_revision": self.manifest.preprocessing_revision,
            "unsafe_pickle": False,
            "automatic_download": False,
            "identity_decision_authority": False,
            "paths_disclosed": False,
            "model_loaded": False,
            "network_used": False,
        }
