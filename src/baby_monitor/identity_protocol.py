from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .identity_domain import FaceObservation, FaceQualityState


class FaceEmbeddingProtocolError(ValueError):
    """A private face-embedding request or result is unsafe or malformed."""


class FaceEmbeddingMode(StrEnum):
    ASSESSMENT = "assessment"
    ENROLLMENT = "enrollment"


FACE_LANDMARK_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "right_mouth_corner",
    "left_mouth_corner",
)
MAX_FACE_PROTOCOL_BYTES = 16 * 1024 * 1024
MAX_FACE_REQUEST_BYTES = 2 * 1024 * 1024
MAX_FACE_IMAGES = 64
MAX_FACES_PER_IMAGE = 100
MAX_TOTAL_FACES = 1_000
MAX_FACE_IMAGE_BYTES = 64 * 1024 * 1024
FACE_EMBEDDING_DIMENSION = 128
FACE_EMBEDDING_PROTOCOL_VERSION = 2
OPENCV_FACE_RUNNER_REVISION = "opencv_yunet_sface_runner_v2"
FACE_QUALITY_POLICY_REVISION = "face_quality_aligned_laplacian_luma_five_point_v2"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUEST_NONCE = re.compile(r"[0-9a-f]{64}")

FACE_QUALITY_REJECTION_CODES = frozenset(
    {
        "bbox_out_of_frame",
        "face_too_small",
        "landmarks_out_of_frame",
        "eyes_too_close",
        "below_detector_threshold",
        "face_too_blurry",
        "underexposed",
        "overexposed",
        "excessive_roll",
        "excessive_yaw",
        "invalid_five_point_geometry",
    }
)


@dataclass(frozen=True, slots=True)
class FaceQualityPolicy:
    """Fixed pre-embedding quality policy included in the pipeline fingerprint."""

    revision: str = FACE_QUALITY_POLICY_REVISION
    aligned_crop_source: str = "opencv_sface_align_crop_bgr_v1"
    grayscale_conversion: str = "opencv_color_bgr2gray_v1"
    blur_metric: str = "variance_of_laplacian_v1"
    exposure_metric: str = "mean_luma_0_255_v1"
    laplacian_kernel_size: int = 3
    min_laplacian_variance: float = 100.0
    min_mean_luma: float = 40.0
    max_mean_luma: float = 215.0
    max_abs_roll_degrees: float = 20.0
    max_abs_nose_offset_eye_ratio: float = 0.35
    min_mouth_eye_ratio: float = 0.35
    max_mouth_eye_ratio: float = 1.35
    min_eye_mouth_vertical_ratio: float = 0.45
    max_eye_mouth_vertical_ratio: float = 2.2
    min_nose_vertical_fraction: float = 0.18
    max_nose_vertical_fraction: float = 0.82

    def to_fingerprint_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "aligned_crop_source": self.aligned_crop_source,
            "grayscale_conversion": self.grayscale_conversion,
            "blur_metric": self.blur_metric,
            "exposure_metric": self.exposure_metric,
            "laplacian_kernel_size": self.laplacian_kernel_size,
            "min_laplacian_variance": self.min_laplacian_variance,
            "min_mean_luma": self.min_mean_luma,
            "max_mean_luma": self.max_mean_luma,
            "max_abs_roll_degrees": self.max_abs_roll_degrees,
            "max_abs_nose_offset_eye_ratio": self.max_abs_nose_offset_eye_ratio,
            "min_mouth_eye_ratio": self.min_mouth_eye_ratio,
            "max_mouth_eye_ratio": self.max_mouth_eye_ratio,
            "min_eye_mouth_vertical_ratio": self.min_eye_mouth_vertical_ratio,
            "max_eye_mouth_vertical_ratio": self.max_eye_mouth_vertical_ratio,
            "min_nose_vertical_fraction": self.min_nose_vertical_fraction,
            "max_nose_vertical_fraction": self.max_nose_vertical_fraction,
        }


FACE_QUALITY_POLICY = FaceQualityPolicy()


def _strict_json(raw: str, *, maximum_bytes: int, context: str) -> Any:
    if not isinstance(raw, str):
        raise FaceEmbeddingProtocolError(f"{context} must be JSON text")
    if len(raw.encode("utf-8")) > maximum_bytes:
        raise FaceEmbeddingProtocolError(f"{context} exceeds the size limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise FaceEmbeddingProtocolError(f"{context} contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                FaceEmbeddingProtocolError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise FaceEmbeddingProtocolError(f"{context} is not valid JSON") from exc


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise FaceEmbeddingProtocolError(f"face embedding {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FaceEmbeddingProtocolError(f"face embedding {field} is invalid")
    return value


def _revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise FaceEmbeddingProtocolError(f"face embedding {field} is not pinned")
    return value


def _request_nonce(value: Any) -> str:
    if not isinstance(value, str) or _REQUEST_NONCE.fullmatch(value) is None:
        raise FaceEmbeddingProtocolError(
            "face embedding request_nonce must be a 256-bit lowercase hexadecimal value"
        )
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise FaceEmbeddingProtocolError(f"face embedding {field} is invalid")
    return float(value)


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise FaceEmbeddingProtocolError(f"face embedding {field} is invalid")
    return value


def _regular_file(
    value: Any,
    field: str,
    *,
    maximum_bytes: int | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or "\x00" in value
        or ".." in value.replace("\\", "/").split("/")
    ):
        raise FaceEmbeddingProtocolError(f"face embedding {field} is unsafe")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise FaceEmbeddingProtocolError(f"face embedding {field} is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except OSError as exc:
        raise FaceEmbeddingProtocolError(f"face embedding {field} is unsafe") from exc
    if resolved != candidate or not stat.S_ISREG(metadata.st_mode):
        raise FaceEmbeddingProtocolError(f"face embedding {field} is unsafe")
    if maximum_bytes is not None and not 0 < metadata.st_size <= maximum_bytes:
        raise FaceEmbeddingProtocolError(f"face embedding {field} exceeds the size limit")
    return str(candidate)


def file_sha256(path: str | Path, *, maximum_bytes: int, context: str) -> str:
    """Hash a revalidated regular local file without disclosing its path."""

    source = Path(_regular_file(str(path), context, maximum_bytes=maximum_bytes))
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise FaceEmbeddingProtocolError(f"{context} could not be hashed") from exc
    return digest.hexdigest()


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise FaceEmbeddingProtocolError("face embedding bbox must contain four numbers")
    parsed = tuple(_number(item, "bbox", minimum=0.0, maximum=1.0) for item in value)
    if parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
        raise FaceEmbeddingProtocolError("face embedding bbox must have positive area")
    return parsed


def _embedding(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != FACE_EMBEDDING_DIMENSION:
        raise FaceEmbeddingProtocolError("face embedding vector must contain 128 numbers")
    parsed = tuple(_number(item, "vector value", minimum=-1.0, maximum=1.0) for item in value)
    norm = math.hypot(*parsed)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        raise FaceEmbeddingProtocolError("face embedding vector must be L2-normalized")
    return parsed


def _private_staging_root(value: str | Path) -> Path:
    """Admit one explicit POSIX owner-only root for transient biometric IPC."""

    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise FaceEmbeddingProtocolError("face embedding staging requires a POSIX private root")
    text = str(value)
    if not text or len(text) > 4096 or "\x00" in text or ".." in text.replace("\\", "/").split("/"):
        raise FaceEmbeddingProtocolError("face embedding staging root is unsafe")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise FaceEmbeddingProtocolError("face embedding staging root is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except OSError as exc:
        raise FaceEmbeddingProtocolError("face embedding staging root is unsafe") from exc
    project_root = Path(__file__).resolve(strict=True).parents[2]
    if (
        resolved != candidate
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or resolved == project_root
        or resolved.parts[:3] == ("/", "mnt", "c")
    ):
        raise FaceEmbeddingProtocolError(
            "face embedding staging root must be a dedicated owner-only WSL directory"
        )
    return resolved


def _private_staging_child(path: str | Path, staging_root: Path) -> tuple[Path, str]:
    text = str(path)
    if not text or len(text) > 4096 or "\x00" in text or ".." in text.replace("\\", "/").split("/"):
        raise FaceEmbeddingProtocolError("face embedding staging file is unsafe")
    candidate = Path(text)
    if (
        not candidate.is_absolute()
        or candidate.name in {"", ".", ".."}
        or candidate.parent != staging_root
    ):
        raise FaceEmbeddingProtocolError("face embedding staging file is unsafe")
    try:
        if candidate.parent.resolve(strict=True) != staging_root:
            raise OSError
    except OSError as exc:
        raise FaceEmbeddingProtocolError("face embedding staging file is unsafe") from exc
    return candidate, candidate.name


def _open_private_staging_root(staging_root: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(staging_root, flags)
        opened = os.fstat(descriptor)
        current = staging_root.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise OSError
        return descriptor
    except OSError as exc:
        with suppress(UnboundLocalError, OSError):
            os.close(descriptor)
        raise FaceEmbeddingProtocolError(
            "face embedding staging root changed or is unsafe"
        ) from exc


def _read_private_staged_bytes(
    path: str | Path,
    *,
    staging_root: str | Path,
    maximum_bytes: int,
    context: str,
) -> bytes:
    root = _private_staging_root(staging_root)
    _candidate, name = _private_staging_child(path, root)
    root_descriptor = _open_private_staging_root(root)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        current = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= maximum_bytes
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise OSError
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise OSError
        return b"".join(chunks)
    except OSError as exc:
        raise FaceEmbeddingProtocolError(f"{context} is not an owner-only staged file") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_descriptor)


def _read_private_json_file(
    path: str | Path,
    *,
    staging_root: str | Path,
    maximum_bytes: int,
    context: str,
) -> Any:
    encoded = _read_private_staged_bytes(
        path,
        staging_root=staging_root,
        maximum_bytes=maximum_bytes,
        context=context,
    )
    try:
        raw = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise FaceEmbeddingProtocolError(f"{context} is not valid UTF-8") from exc
    return _strict_json(raw, maximum_bytes=maximum_bytes, context=context)


def _write_atomic_staged(
    payload: bytes,
    path: str | Path,
    *,
    staging_root: str | Path,
) -> None:
    root = _private_staging_root(staging_root)
    target, target_name = _private_staging_child(path, root)
    root_descriptor = _open_private_staging_root(root)
    descriptor = -1
    temporary_name = f".{target.name}.{secrets.token_hex(16)}.tmp"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary_name,
                target_name,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError("face embedding staged output already exists") from None
        os.unlink(temporary_name, dir_fd=root_descriptor)
        temporary_name = ""
        os.fsync(root_descriptor)
        metadata = os.stat(target_name, dir_fd=root_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise OSError
    except FileExistsError:
        raise
    except OSError as exc:
        raise FaceEmbeddingProtocolError(
            "face embedding staged output could not be written privately"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=root_descriptor)
        os.close(root_descriptor)


@dataclass(frozen=True, slots=True)
class FaceEmbeddingStagingRoot:
    """Explicitly approved owner-only root for one-shot request/result exchange."""

    root: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", str(_private_staging_root(self.root)))

    def read_request(self, path: str | Path) -> FaceEmbeddingRequest:
        return FaceEmbeddingRequest.from_dict(
            _read_private_json_file(
                path,
                staging_root=self.root,
                maximum_bytes=MAX_FACE_REQUEST_BYTES,
                context="face embedding request",
            )
        )

    def write_request(self, request: FaceEmbeddingRequest, path: str | Path) -> None:
        if not isinstance(request, FaceEmbeddingRequest):
            raise FaceEmbeddingProtocolError("face embedding request is invalid")
        _write_atomic_staged(
            f"{request.to_json()}\n".encode(),
            path,
            staging_root=self.root,
        )

    def read_batch(self, path: str | Path) -> FaceEmbeddingBatch:
        return FaceEmbeddingBatch.from_dict(
            _read_private_json_file(
                path,
                staging_root=self.root,
                maximum_bytes=MAX_FACE_PROTOCOL_BYTES,
                context="face embedding batch",
            )
        )

    def write_batch(self, batch: FaceEmbeddingBatch, path: str | Path) -> None:
        if not isinstance(batch, FaceEmbeddingBatch):
            raise FaceEmbeddingProtocolError("face embedding batch is invalid")
        _write_atomic_staged(
            f"{batch.to_json()}\n".encode(),
            path,
            staging_root=self.root,
        )

    def consume_request(self, request_sha256: str) -> None:
        digest = _sha256(request_sha256, "request_sha256")
        marker = Path(self.root) / f".consumed-{digest}.json"
        payload = json.dumps(
            {"request_sha256": digest, "state": "consumed"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        try:
            _write_atomic_staged(payload + b"\n", marker, staging_root=self.root)
        except FileExistsError:
            raise FaceEmbeddingProtocolError("face embedding result replay was rejected") from None


@dataclass(frozen=True, slots=True)
class FaceImageInput:
    """One explicitly admitted local image; the path never appears in results."""

    frame_ref: str
    source_path: str = field(repr=False)
    source_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.frame_ref, "frame_ref")
        object.__setattr__(
            self,
            "source_path",
            _regular_file(
                self.source_path,
                "image source_path",
                maximum_bytes=MAX_FACE_IMAGE_BYTES,
            ),
        )
        _sha256(self.source_sha256, "image source_sha256")

    @classmethod
    def from_dict(cls, value: Any) -> FaceImageInput:
        if not isinstance(value, dict) or set(value) != {
            "frame_ref",
            "source_path",
            "source_sha256",
        }:
            raise FaceEmbeddingProtocolError("face image input fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_ref": self.frame_ref,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class FaceModelReference:
    """Private reference to one separately reviewed and pinned local ONNX file."""

    role: str
    model_path: str = field(repr=False)
    manifest_path: str = field(repr=False)
    model_id: str
    upstream_revision: str
    model_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"detector", "embedding"}:
            raise FaceEmbeddingProtocolError("face model reference role is invalid")
        object.__setattr__(self, "model_path", _regular_file(self.model_path, "model_path"))
        object.__setattr__(
            self,
            "manifest_path",
            _regular_file(
                self.manifest_path,
                "manifest_path",
                maximum_bytes=1024 * 1024,
            ),
        )
        _identifier(self.model_id, "model_id")
        _revision(self.upstream_revision, "upstream_revision")
        _sha256(self.model_sha256, "model_sha256")

    @classmethod
    def from_dict(cls, value: Any) -> FaceModelReference:
        if not isinstance(value, dict) or set(value) != {
            "role",
            "model_path",
            "manifest_path",
            "model_id",
            "upstream_revision",
            "model_sha256",
        }:
            raise FaceEmbeddingProtocolError("face model reference fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "model_path": self.model_path,
            "manifest_path": self.manifest_path,
            "model_id": self.model_id,
            "upstream_revision": self.upstream_revision,
            "model_sha256": self.model_sha256,
        }


@dataclass(frozen=True, slots=True)
class FaceEmbeddingRequest:
    """Strict path-bearing manifest for one short-lived, network-free process."""

    request_id: str
    mode: FaceEmbeddingMode
    face_adapter: str
    images: tuple[FaceImageInput, ...] = field(repr=False)
    detector_model: FaceModelReference = field(repr=False)
    embedding_model: FaceModelReference = field(repr=False)
    detector_score_threshold: float
    detector_nms_threshold: float
    detector_top_k: int
    min_face_size_pixels: int
    min_eye_distance_pixels: float
    device: str = "cpu"
    request_nonce: str = field(default_factory=lambda: secrets.token_hex(32), repr=False)
    version: int = FACE_EMBEDDING_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != FACE_EMBEDDING_PROTOCOL_VERSION:
            raise FaceEmbeddingProtocolError("unsupported face embedding request version")
        _request_nonce(self.request_nonce)
        _identifier(self.request_id, "request_id")
        if not isinstance(self.mode, FaceEmbeddingMode):
            raise FaceEmbeddingProtocolError("face embedding mode is invalid")
        _identifier(self.face_adapter, "face_adapter")
        if (
            not isinstance(self.images, tuple)
            or not 1 <= len(self.images) <= MAX_FACE_IMAGES
            or any(not isinstance(item, FaceImageInput) for item in self.images)
        ):
            raise FaceEmbeddingProtocolError("face embedding images are invalid or too many")
        if len({item.frame_ref for item in self.images}) != len(self.images):
            raise FaceEmbeddingProtocolError("face embedding frame IDs must be unique")
        if len({item.source_path for item in self.images}) != len(self.images):
            raise FaceEmbeddingProtocolError("face embedding image paths must be unique")
        if self.mode is FaceEmbeddingMode.ENROLLMENT and len(self.images) != 1:
            raise FaceEmbeddingProtocolError("face enrollment requires exactly one image")
        if not isinstance(self.detector_model, FaceModelReference) or (
            self.detector_model.role != "detector"
        ):
            raise FaceEmbeddingProtocolError("face detector model reference is invalid")
        if not isinstance(self.embedding_model, FaceModelReference) or (
            self.embedding_model.role != "embedding"
        ):
            raise FaceEmbeddingProtocolError("face embedding model reference is invalid")
        object.__setattr__(
            self,
            "detector_score_threshold",
            _number(
                self.detector_score_threshold,
                "detector_score_threshold",
                minimum=0.01,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "detector_nms_threshold",
            _number(
                self.detector_nms_threshold,
                "detector_nms_threshold",
                minimum=0.01,
                maximum=1.0,
            ),
        )
        _integer(self.detector_top_k, "detector_top_k", minimum=1, maximum=5_000)
        _integer(
            self.min_face_size_pixels,
            "min_face_size_pixels",
            minimum=8,
            maximum=4096,
        )
        object.__setattr__(
            self,
            "min_eye_distance_pixels",
            _number(
                self.min_eye_distance_pixels,
                "min_eye_distance_pixels",
                minimum=1.0,
                maximum=2048.0,
            ),
        )
        if self.device != "cpu":
            raise FaceEmbeddingProtocolError("face embedding runner is CPU-only")

    @classmethod
    def from_dict(cls, value: Any) -> FaceEmbeddingRequest:
        expected = {
            "version",
            "request_id",
            "request_nonce",
            "mode",
            "face_adapter",
            "images",
            "detector_model",
            "embedding_model",
            "detector_score_threshold",
            "detector_nms_threshold",
            "detector_top_k",
            "min_face_size_pixels",
            "min_eye_distance_pixels",
            "device",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise FaceEmbeddingProtocolError("face embedding request fields do not match v2")
        images = value["images"]
        if not isinstance(images, list):
            raise FaceEmbeddingProtocolError("face embedding images must be an array")
        try:
            mode = FaceEmbeddingMode(value["mode"])
        except (TypeError, ValueError) as exc:
            raise FaceEmbeddingProtocolError("face embedding mode is invalid") from exc
        return cls(
            version=value["version"],
            request_id=value["request_id"],
            request_nonce=value["request_nonce"],
            mode=mode,
            face_adapter=value["face_adapter"],
            images=tuple(FaceImageInput.from_dict(item) for item in images),
            detector_model=FaceModelReference.from_dict(value["detector_model"]),
            embedding_model=FaceModelReference.from_dict(value["embedding_model"]),
            detector_score_threshold=value["detector_score_threshold"],
            detector_nms_threshold=value["detector_nms_threshold"],
            detector_top_k=value["detector_top_k"],
            min_face_size_pixels=value["min_face_size_pixels"],
            min_eye_distance_pixels=value["min_eye_distance_pixels"],
            device=value["device"],
        )

    @classmethod
    def from_json(cls, raw: str) -> FaceEmbeddingRequest:
        return cls.from_dict(
            _strict_json(
                raw,
                maximum_bytes=MAX_FACE_REQUEST_BYTES,
                context="face embedding request",
            )
        )

    @classmethod
    def read(cls, path: str | Path, *, staging_root: str | Path) -> FaceEmbeddingRequest:
        return FaceEmbeddingStagingRoot(str(staging_root)).read_request(path)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "request_nonce": self.request_nonce,
            "mode": self.mode.value,
            "face_adapter": self.face_adapter,
            "images": [item.to_dict() for item in self.images],
            "detector_model": self.detector_model.to_dict(),
            "embedding_model": self.embedding_model.to_dict(),
            "detector_score_threshold": self.detector_score_threshold,
            "detector_nms_threshold": self.detector_nms_threshold,
            "detector_top_k": self.detector_top_k,
            "min_face_size_pixels": self.min_face_size_pixels,
            "min_eye_distance_pixels": self.min_eye_distance_pixels,
            "device": self.device,
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > MAX_FACE_REQUEST_BYTES:
            raise FaceEmbeddingProtocolError("face embedding request exceeds the size limit")
        return encoded

    def canonical_sha256(self) -> str:
        """Bind every canonical request field without exposing path-bearing JSON."""

        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def write_atomic(self, path: str | Path, *, staging_root: str | Path) -> None:
        FaceEmbeddingStagingRoot(str(staging_root)).write_request(self, path)


def canonical_face_pipeline_fingerprint(
    request: FaceEmbeddingRequest,
    *,
    detector_manifest: Any | None = None,
    embedding_manifest: Any | None = None,
) -> str:
    """Recompute the admitted adapter pipeline fingerprint from trusted local manifests."""

    if not isinstance(request, FaceEmbeddingRequest):
        raise FaceEmbeddingProtocolError("face embedding request is invalid")
    if request.face_adapter != "opencv_yunet_sface_v1":
        raise FaceEmbeddingProtocolError("face embedding adapter is not supported")
    try:
        from .identity_model import FaceModelError, FaceModelManifest

        detector = (
            FaceModelManifest.read(request.detector_model.manifest_path)
            if detector_manifest is None
            else detector_manifest
        )
        embedding = (
            FaceModelManifest.read(request.embedding_model.manifest_path)
            if embedding_manifest is None
            else embedding_manifest
        )
        if not isinstance(detector, FaceModelManifest) or not isinstance(
            embedding, FaceModelManifest
        ):
            raise FaceModelError("face pipeline manifests are invalid")
        for manifest, reference in (
            (detector, request.detector_model),
            (embedding, request.embedding_model),
        ):
            manifest.require_reviewed()
            if (
                manifest.role != reference.role
                or manifest.model_id != reference.model_id
                or manifest.upstream_revision != reference.upstream_revision
                or manifest.sha256 != reference.model_sha256
            ):
                raise FaceModelError("face pipeline manifest does not match request")
            if detector_manifest is None and manifest is detector:
                manifest.verify_model(reference.model_path)
            if embedding_manifest is None and manifest is embedding:
                manifest.verify_model(reference.model_path)
    except (FaceModelError, OSError) as exc:
        raise FaceEmbeddingProtocolError(
            "face embedding pipeline fingerprint could not be recomputed"
        ) from exc

    payload = json.dumps(
        {
            "face_adapter": request.face_adapter,
            "runner_revision": OPENCV_FACE_RUNNER_REVISION,
            "quality_policy": FACE_QUALITY_POLICY.to_fingerprint_dict(),
            "detector_backend": detector.backend,
            "detector_model_sha256": detector.sha256,
            "detector_preprocessing_revision": detector.preprocessing_revision,
            "embedding_backend": embedding.backend,
            "embedding_model_sha256": embedding.sha256,
            "embedding_preprocessing_revision": embedding.preprocessing_revision,
            "embedding_dimension": embedding.embedding_dimension,
            "detector_score_threshold": request.detector_score_threshold,
            "detector_nms_threshold": request.detector_nms_threshold,
            "detector_top_k": request.detector_top_k,
            "min_face_size_pixels": request.min_face_size_pixels,
            "min_eye_distance_pixels": request.min_eye_distance_pixels,
            "device": request.device,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FaceLandmark:
    name: str
    x: float | None
    y: float | None
    in_frame: bool

    def __post_init__(self) -> None:
        if self.name not in FACE_LANDMARK_NAMES:
            raise FaceEmbeddingProtocolError("face landmark name is invalid")
        if type(self.in_frame) is not bool:
            raise FaceEmbeddingProtocolError("face landmark in_frame must be a boolean")
        if self.in_frame:
            if self.x is None or self.y is None:
                raise FaceEmbeddingProtocolError("in-frame face landmark coordinates are required")
            object.__setattr__(self, "x", _number(self.x, "landmark x", minimum=0.0, maximum=1.0))
            object.__setattr__(self, "y", _number(self.y, "landmark y", minimum=0.0, maximum=1.0))
        elif self.x is not None or self.y is not None:
            raise FaceEmbeddingProtocolError("out-of-frame face landmark coordinates must be null")

    @classmethod
    def from_dict(cls, value: Any) -> FaceLandmark:
        if not isinstance(value, dict) or set(value) != {"name", "x", "y", "in_frame"}:
            raise FaceEmbeddingProtocolError("face landmark fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "x": self.x, "y": self.y, "in_frame": self.in_frame}


@dataclass(frozen=True, slots=True)
class FaceEmbeddingObservation:
    """Visible detector/quality facts plus a private embedding; never an identity claim."""

    face_id: str
    frame_ref: str
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    landmarks: tuple[FaceLandmark, ...]
    quality_state: FaceQualityState
    rejection_codes: tuple[str, ...]
    embedding: tuple[float, ...] | None = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.face_id, "face_id")
        _identifier(self.frame_ref, "frame_ref")
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy))
        object.__setattr__(
            self,
            "detector_confidence",
            _number(self.detector_confidence, "detector_confidence", minimum=0.0, maximum=1.0),
        )
        if (
            not isinstance(self.landmarks, tuple)
            or len(self.landmarks) != len(FACE_LANDMARK_NAMES)
            or any(not isinstance(item, FaceLandmark) for item in self.landmarks)
            or tuple(item.name for item in self.landmarks) != FACE_LANDMARK_NAMES
        ):
            raise FaceEmbeddingProtocolError(
                "face landmarks must contain the ordered five-point set"
            )
        if not isinstance(self.quality_state, FaceQualityState):
            raise FaceEmbeddingProtocolError("face quality_state is invalid")
        if not isinstance(self.rejection_codes, tuple) or any(
            code not in FACE_QUALITY_REJECTION_CODES for code in self.rejection_codes
        ):
            raise FaceEmbeddingProtocolError("face rejection_codes are not approved")
        if len(set(self.rejection_codes)) != len(self.rejection_codes):
            raise FaceEmbeddingProtocolError("face rejection_codes must be unique")
        if self.quality_state is FaceQualityState.ACCEPTED:
            if self.rejection_codes or self.embedding is None:
                raise FaceEmbeddingProtocolError(
                    "accepted face requires one embedding and no rejection codes"
                )
            object.__setattr__(self, "embedding", _embedding(self.embedding))
        elif not self.rejection_codes or self.embedding is not None:
            raise FaceEmbeddingProtocolError(
                "rejected face requires rejection codes and no embedding"
            )
        FaceObservation(
            face_id=self.face_id,
            frame_ref=self.frame_ref,
            bbox_xyxy=self.bbox_xyxy,
            detector_confidence=self.detector_confidence,
            quality_state=self.quality_state,
            rejection_codes=self.rejection_codes,
        )

    @classmethod
    def from_dict(cls, value: Any) -> FaceEmbeddingObservation:
        expected = {
            "face_id",
            "frame_ref",
            "bbox_xyxy",
            "detector_confidence",
            "landmarks",
            "quality_state",
            "rejection_codes",
            "embedding",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise FaceEmbeddingProtocolError("face embedding observation fields do not match v1")
        landmarks = value["landmarks"]
        rejection_codes = value["rejection_codes"]
        if not isinstance(landmarks, list) or not isinstance(rejection_codes, list):
            raise FaceEmbeddingProtocolError("face landmarks and rejection_codes must be arrays")
        embedding = value["embedding"]
        if embedding is not None and not isinstance(embedding, list):
            raise FaceEmbeddingProtocolError("face embedding vector must be an array or null")
        try:
            quality_state = FaceQualityState(value["quality_state"])
        except (TypeError, ValueError) as exc:
            raise FaceEmbeddingProtocolError("face quality_state is invalid") from exc
        return cls(
            face_id=value["face_id"],
            frame_ref=value["frame_ref"],
            bbox_xyxy=_bbox(value["bbox_xyxy"]),
            detector_confidence=value["detector_confidence"],
            landmarks=tuple(FaceLandmark.from_dict(item) for item in landmarks),
            quality_state=quality_state,
            rejection_codes=tuple(rejection_codes),
            embedding=None if embedding is None else tuple(embedding),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "frame_ref": self.frame_ref,
            "bbox_xyxy": list(self.bbox_xyxy),
            "detector_confidence": self.detector_confidence,
            "landmarks": [item.to_dict() for item in self.landmarks],
            "quality_state": self.quality_state.value,
            "rejection_codes": list(self.rejection_codes),
            "embedding": None if self.embedding is None else list(self.embedding),
        }

    @property
    def observation(self) -> FaceObservation:
        return FaceObservation(
            face_id=self.face_id,
            frame_ref=self.frame_ref,
            bbox_xyxy=self.bbox_xyxy,
            detector_confidence=self.detector_confidence,
            quality_state=self.quality_state,
            rejection_codes=self.rejection_codes,
        )


@dataclass(frozen=True, slots=True)
class FaceImageEmbeddingResult:
    frame_ref: str
    source_sha256: str
    width: int
    height: int
    faces: tuple[FaceEmbeddingObservation, ...]

    def __post_init__(self) -> None:
        _identifier(self.frame_ref, "result frame_ref")
        _sha256(self.source_sha256, "result source_sha256")
        _integer(self.width, "result width", minimum=1, maximum=65535)
        _integer(self.height, "result height", minimum=1, maximum=65535)
        if (
            not isinstance(self.faces, tuple)
            or len(self.faces) > MAX_FACES_PER_IMAGE
            or any(not isinstance(item, FaceEmbeddingObservation) for item in self.faces)
        ):
            raise FaceEmbeddingProtocolError("face image result faces are invalid or too many")
        if len({item.face_id for item in self.faces}) != len(self.faces):
            raise FaceEmbeddingProtocolError("face IDs must be unique within one image")
        if any(item.frame_ref != self.frame_ref for item in self.faces):
            raise FaceEmbeddingProtocolError("face frame_ref does not match its image")

    @classmethod
    def from_dict(cls, value: Any) -> FaceImageEmbeddingResult:
        if not isinstance(value, dict) or set(value) != {
            "frame_ref",
            "source_sha256",
            "width",
            "height",
            "faces",
        }:
            raise FaceEmbeddingProtocolError("face image result fields do not match v1")
        faces = value["faces"]
        if not isinstance(faces, list):
            raise FaceEmbeddingProtocolError("face image result faces must be an array")
        return cls(
            frame_ref=value["frame_ref"],
            source_sha256=value["source_sha256"],
            width=value["width"],
            height=value["height"],
            faces=tuple(FaceEmbeddingObservation.from_dict(item) for item in faces),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_ref": self.frame_ref,
            "source_sha256": self.source_sha256,
            "width": self.width,
            "height": self.height,
            "faces": [item.to_dict() for item in self.faces],
        }


@dataclass(frozen=True, slots=True)
class FaceEmbeddingBatch:
    """Path-free, private local output; identity matching is intentionally separate."""

    request_id: str
    request_sha256: str
    mode: FaceEmbeddingMode
    face_adapter: str
    detector_model_id: str
    detector_model_revision: str
    detector_model_sha256: str
    embedding_model_id: str
    embedding_model_revision: str
    embedding_model_sha256: str
    model_fingerprint: str
    images: tuple[FaceImageEmbeddingResult, ...] = field(repr=False)
    identity_inference: bool = False
    paths_disclosed: bool = False
    network_used: bool = False
    version: int = FACE_EMBEDDING_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != FACE_EMBEDDING_PROTOCOL_VERSION:
            raise FaceEmbeddingProtocolError("unsupported face embedding batch version")
        _identifier(self.request_id, "result request_id")
        _sha256(self.request_sha256, "result request_sha256")
        if not isinstance(self.mode, FaceEmbeddingMode):
            raise FaceEmbeddingProtocolError("face embedding result mode is invalid")
        _identifier(self.face_adapter, "result face_adapter")
        _identifier(self.detector_model_id, "result detector_model_id")
        _revision(self.detector_model_revision, "result detector_model_revision")
        _sha256(self.detector_model_sha256, "result detector_model_sha256")
        _identifier(self.embedding_model_id, "result embedding_model_id")
        _revision(self.embedding_model_revision, "result embedding_model_revision")
        _sha256(self.embedding_model_sha256, "result embedding_model_sha256")
        _sha256(self.model_fingerprint, "result model_fingerprint")
        if (
            not isinstance(self.images, tuple)
            or not 1 <= len(self.images) <= MAX_FACE_IMAGES
            or any(not isinstance(item, FaceImageEmbeddingResult) for item in self.images)
        ):
            raise FaceEmbeddingProtocolError("face embedding result images are invalid")
        if len({item.frame_ref for item in self.images}) != len(self.images):
            raise FaceEmbeddingProtocolError("face embedding result frame IDs must be unique")
        faces = tuple(face for image in self.images for face in image.faces)
        if len(faces) > MAX_TOTAL_FACES:
            raise FaceEmbeddingProtocolError("face embedding result has too many faces")
        if len({face.face_id for face in faces}) != len(faces):
            raise FaceEmbeddingProtocolError("face embedding result face IDs must be unique")
        accepted_count = sum(face.quality_state is FaceQualityState.ACCEPTED for face in faces)
        if self.mode is FaceEmbeddingMode.ENROLLMENT and accepted_count != 1:
            raise FaceEmbeddingProtocolError(
                "face enrollment requires exactly one quality-accepted face"
            )
        for value, field_name in (
            (self.identity_inference, "identity_inference"),
            (self.paths_disclosed, "paths_disclosed"),
            (self.network_used, "network_used"),
        ):
            if type(value) is not bool or value:
                raise FaceEmbeddingProtocolError(f"face embedding {field_name} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> FaceEmbeddingBatch:
        expected = {
            "version",
            "request_id",
            "request_sha256",
            "mode",
            "face_adapter",
            "detector_model_id",
            "detector_model_revision",
            "detector_model_sha256",
            "embedding_model_id",
            "embedding_model_revision",
            "embedding_model_sha256",
            "model_fingerprint",
            "images",
            "identity_inference",
            "paths_disclosed",
            "network_used",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise FaceEmbeddingProtocolError("face embedding batch fields do not match v2")
        images = value["images"]
        if not isinstance(images, list):
            raise FaceEmbeddingProtocolError("face embedding result images must be an array")
        try:
            mode = FaceEmbeddingMode(value["mode"])
        except (TypeError, ValueError) as exc:
            raise FaceEmbeddingProtocolError("face embedding result mode is invalid") from exc
        return cls(
            version=value["version"],
            request_id=value["request_id"],
            request_sha256=value["request_sha256"],
            mode=mode,
            face_adapter=value["face_adapter"],
            detector_model_id=value["detector_model_id"],
            detector_model_revision=value["detector_model_revision"],
            detector_model_sha256=value["detector_model_sha256"],
            embedding_model_id=value["embedding_model_id"],
            embedding_model_revision=value["embedding_model_revision"],
            embedding_model_sha256=value["embedding_model_sha256"],
            model_fingerprint=value["model_fingerprint"],
            images=tuple(FaceImageEmbeddingResult.from_dict(item) for item in images),
            identity_inference=value["identity_inference"],
            paths_disclosed=value["paths_disclosed"],
            network_used=value["network_used"],
        )

    @classmethod
    def from_json(cls, raw: str) -> FaceEmbeddingBatch:
        return cls.from_dict(
            _strict_json(
                raw,
                maximum_bytes=MAX_FACE_PROTOCOL_BYTES,
                context="face embedding batch",
            )
        )

    @classmethod
    def read(cls, path: str | Path, *, staging_root: str | Path) -> FaceEmbeddingBatch:
        return FaceEmbeddingStagingRoot(str(staging_root)).read_batch(path)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "mode": self.mode.value,
            "face_adapter": self.face_adapter,
            "detector_model_id": self.detector_model_id,
            "detector_model_revision": self.detector_model_revision,
            "detector_model_sha256": self.detector_model_sha256,
            "embedding_model_id": self.embedding_model_id,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_model_sha256": self.embedding_model_sha256,
            "model_fingerprint": self.model_fingerprint,
            "images": [item.to_dict() for item in self.images],
            "identity_inference": False,
            "paths_disclosed": False,
            "network_used": False,
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > MAX_FACE_PROTOCOL_BYTES:
            raise FaceEmbeddingProtocolError("face embedding batch exceeds the size limit")
        return encoded

    def write_atomic(self, path: str | Path, *, staging_root: str | Path) -> None:
        FaceEmbeddingStagingRoot(str(staging_root)).write_batch(self, path)

    def validate_against_request(
        self,
        request: FaceEmbeddingRequest,
        *,
        replay_guard: FaceEmbeddingStagingRoot,
    ) -> None:
        """Recompute all bindings and atomically consume this one-shot request."""

        if not isinstance(request, FaceEmbeddingRequest):
            raise FaceEmbeddingProtocolError("face embedding request is invalid")
        if not isinstance(replay_guard, FaceEmbeddingStagingRoot):
            raise FaceEmbeddingProtocolError("face embedding replay guard is required")
        if self.request_sha256 != request.canonical_sha256():
            raise FaceEmbeddingProtocolError(
                "face embedding result does not match the canonical request"
            )
        if (
            self.request_id != request.request_id
            or self.mode is not request.mode
            or self.face_adapter != request.face_adapter
        ):
            raise FaceEmbeddingProtocolError(
                "face embedding result request identity does not match"
            )
        if (
            self.detector_model_id != request.detector_model.model_id
            or self.detector_model_revision != request.detector_model.upstream_revision
            or self.detector_model_sha256 != request.detector_model.model_sha256
            or self.embedding_model_id != request.embedding_model.model_id
            or self.embedding_model_revision != request.embedding_model.upstream_revision
            or self.embedding_model_sha256 != request.embedding_model.model_sha256
        ):
            raise FaceEmbeddingProtocolError("face embedding result model binding does not match")
        requested_images = tuple((image.frame_ref, image.source_sha256) for image in request.images)
        returned_images = tuple((image.frame_ref, image.source_sha256) for image in self.images)
        if returned_images != requested_images:
            raise FaceEmbeddingProtocolError(
                "face embedding result image order or source hash does not match"
            )
        expected_fingerprint = canonical_face_pipeline_fingerprint(request)
        if self.model_fingerprint != expected_fingerprint:
            raise FaceEmbeddingProtocolError(
                "face embedding result pipeline fingerprint does not match"
            )
        replay_guard.consume_request(self.request_sha256)

    @property
    def face_count(self) -> int:
        return sum(len(image.faces) for image in self.images)

    @property
    def accepted_face_count(self) -> int:
        return sum(
            face.quality_state is FaceQualityState.ACCEPTED
            for image in self.images
            for face in image.faces
        )
