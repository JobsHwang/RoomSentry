from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .identity_domain import DeclaredKind
from .identity_enrollment_contracts import (
    _SHA256,
    EnrollmentImage,
    IdentityEnrollmentError,
    ValidatedEnrollmentEmbedding,
    _safe_path_text,
)
from .identity_protocol import (
    FACE_EMBEDDING_DIMENSION,
    MAX_FACE_IMAGE_BYTES,
    FaceEmbeddingMode,
)
from .identity_service import IdentitySettings, authorize_enrollment
from .identity_store import IdentityEnrollmentSummary

_MANIFEST_VERSION = 1
_MIN_ENROLLMENT_IMAGES = 3
_MAX_ENROLLMENT_IMAGES = 12
_MAX_MANIFEST_BYTES = 1024 * 1024
_PROFILE_ID = re.compile(r"profile_[0-9a-f]{32}")
_TEMPLATE_ID = re.compile(r"template_[0-9a-f]{32}")
# OpenCV SFace's same-identity cosine threshold is reused only as a fixed
# enrollment outlier veto. Passing this gate does not prove identity, prove a
# same-subject cohort, or establish baby-recognition accuracy; an operator must
# still confirm that every enrollment image shows the intended same subject.
ENROLLMENT_COHORT_MIN_PAIRWISE_COSINE = 0.363
ENROLLMENT_POLICY_REVISION = "sface-cosine-0.363-cohort-outlier-v1"


def _strict_json(raw: str) -> Any:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
        raise IdentityEnrollmentError("identity enrollment manifest exceeds the size limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise IdentityEnrollmentError(
                    "identity enrollment manifest contains duplicate JSON keys"
                )
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                IdentityEnrollmentError(
                    "identity enrollment manifest contains a non-standard number"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise IdentityEnrollmentError("identity enrollment manifest is not valid JSON") from exc


def _private_manifest_root(value: str | Path) -> Path:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise IdentityEnrollmentError("identity enrollment requires a private POSIX manifest root")
    text = _safe_path_text(value, "manifest root")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise IdentityEnrollmentError("identity enrollment manifest root is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
        project_root = Path(__file__).resolve(strict=True).parents[2]
        home_root = Path.home().resolve(strict=True)
    except OSError as exc:
        raise IdentityEnrollmentError("identity enrollment manifest root is unsafe") from exc
    if (
        resolved != candidate
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or resolved in {Path("/"), home_root, project_root}
        or resolved.parts[:3] == ("/", "mnt", "c")
    ):
        raise IdentityEnrollmentError(
            "identity enrollment manifest root must be a dedicated owner-only WSL directory"
        )
    return resolved


def _read_private_manifest(
    path: str | Path,
    *,
    private_root: str | Path,
) -> IdentityEnrollmentManifest:
    root = _private_manifest_root(private_root)
    text = _safe_path_text(path, "manifest file")
    candidate = Path(text)
    if (
        not candidate.is_absolute()
        or candidate.parent != root
        or candidate.suffix.lower() != ".json"
        or candidate.name in {"", ".", ".."}
    ):
        raise IdentityEnrollmentError("identity enrollment manifest file is unsafe")

    root_descriptor = -1
    descriptor = -1
    try:
        root_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_root = os.fstat(root_descriptor)
        current_root = root.lstat()
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or stat.S_IMODE(opened_root.st_mode) != 0o700
            or opened_root.st_uid != os.geteuid()
            or (opened_root.st_dev, opened_root.st_ino)
            != (current_root.st_dev, current_root.st_ino)
        ):
            raise OSError
        descriptor = os.open(
            candidate.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        metadata = os.fstat(descriptor)
        current = os.stat(candidate.name, dir_fd=root_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= _MAX_MANIFEST_BYTES
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise OSError
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(
            descriptor,
            min(64 * 1024, _MAX_MANIFEST_BYTES + 1 - total),
        ):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_MANIFEST_BYTES:
                raise OSError
    except OSError as exc:
        raise IdentityEnrollmentError(
            "identity enrollment manifest is not an owner-only private file"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)
    try:
        raw = b"".join(chunks).decode("utf-8")
    except UnicodeError as exc:
        raise IdentityEnrollmentError("identity enrollment manifest is not valid UTF-8") from exc
    return IdentityEnrollmentManifest.from_dict(_strict_json(raw))


def _verify_source_image(image: EnrollmentImage) -> None:
    candidate = Path(image.source_path)
    descriptor = -1
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        current = candidate.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= MAX_FACE_IMAGE_BYTES
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise OSError
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    except OSError as exc:
        raise IdentityEnrollmentError(
            "identity enrollment source image could not be verified"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if digest.hexdigest() != image.source_sha256:
        raise IdentityEnrollmentError("identity enrollment source image hash does not match")


@dataclass(frozen=True, slots=True)
class IdentityEnrollmentManifest:
    declared_kind: DeclaredKind
    images: tuple[EnrollmentImage, ...] = field(repr=False)
    version: int = _MANIFEST_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != _MANIFEST_VERSION:
            raise IdentityEnrollmentError("unsupported identity enrollment manifest version")
        if (
            not isinstance(self.declared_kind, DeclaredKind)
            or self.declared_kind is DeclaredKind.UNKNOWN
        ):
            raise IdentityEnrollmentError("identity enrollment declared_kind is invalid")
        if (
            not isinstance(self.images, tuple)
            or not _MIN_ENROLLMENT_IMAGES <= len(self.images) <= _MAX_ENROLLMENT_IMAGES
            or any(not isinstance(item, EnrollmentImage) for item in self.images)
        ):
            raise IdentityEnrollmentError("identity enrollment requires between 3 and 12 images")
        if len({item.frame_ref for item in self.images}) != len(self.images):
            raise IdentityEnrollmentError("identity enrollment frame_ref values must be unique")
        if len({item.source_sha256 for item in self.images}) != len(self.images):
            raise IdentityEnrollmentError("identity enrollment source_sha256 values must be unique")
        if len({item.source_path for item in self.images}) != len(self.images):
            raise IdentityEnrollmentError("identity enrollment source paths must be unique")

    @classmethod
    def from_dict(cls, value: Any) -> IdentityEnrollmentManifest:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "declared_kind",
            "images",
        }:
            raise IdentityEnrollmentError("identity enrollment manifest fields do not match v1")
        images = value["images"]
        if not isinstance(images, list):
            raise IdentityEnrollmentError("identity enrollment images must be an array")
        if not _MIN_ENROLLMENT_IMAGES <= len(images) <= _MAX_ENROLLMENT_IMAGES:
            raise IdentityEnrollmentError("identity enrollment requires between 3 and 12 images")
        try:
            declared_kind = DeclaredKind(value["declared_kind"])
        except (TypeError, ValueError) as exc:
            raise IdentityEnrollmentError("identity enrollment declared_kind is invalid") from exc
        return cls(
            version=value["version"],
            declared_kind=declared_kind,
            images=tuple(EnrollmentImage.from_dict(item) for item in images),
        )


class EnrollmentEmbeddingExtractor(Protocol):
    """Injected adapter boundary; no subprocess or concrete model is selected here."""

    def extract_validated_enrollment(
        self,
        image: EnrollmentImage,
    ) -> ValidatedEnrollmentEmbedding: ...


class AtomicEnrollmentStore(Protocol):
    def enroll_profile(
        self,
        *,
        declared_kind: DeclaredKind,
        model_fingerprint: str,
        embeddings: Sequence[Sequence[float]],
    ) -> IdentityEnrollmentSummary: ...


class AtomicEnrollmentStoreFactory(Protocol):
    def __call__(self) -> AtomicEnrollmentStore: ...


def _validate_enrollment_cohort(embeddings: Sequence[Sequence[float]]) -> None:
    """Apply a fixed pairwise outlier veto without making an identity claim."""

    if not _MIN_ENROLLMENT_IMAGES <= len(embeddings) <= _MAX_ENROLLMENT_IMAGES:
        raise IdentityEnrollmentError("identity enrollment cohort size is invalid")
    normalized: list[tuple[float, ...]] = []
    for embedding in embeddings:
        if (
            not isinstance(embedding, tuple)
            or len(embedding) != FACE_EMBEDDING_DIMENSION
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in embedding
            )
        ):
            raise IdentityEnrollmentError(
                "identity enrollment cohort failed the fixed pairwise outlier veto"
            )
        vector = tuple(float(item) for item in embedding)
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or not math.isclose(
            norm,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise IdentityEnrollmentError(
                "identity enrollment cohort failed the fixed pairwise outlier veto"
            )
        normalized.append(vector)
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            try:
                cosine = math.fsum(a * b for a, b in zip(left, right, strict=True))
            except (OverflowError, ValueError):
                raise IdentityEnrollmentError(
                    "identity enrollment cohort failed the fixed pairwise outlier veto"
                ) from None
            if not math.isfinite(cosine) or cosine < ENROLLMENT_COHORT_MIN_PAIRWISE_COSINE:
                raise IdentityEnrollmentError(
                    "identity enrollment cohort failed the fixed pairwise outlier veto"
                )


@dataclass(frozen=True, slots=True)
class IdentityEnrollmentResult:
    profile_id: str
    template_ids: tuple[str, ...]
    declared_kind: DeclaredKind
    model_fingerprint: str
    embedding_dimension: int
    source_image_count: int
    source_images_persisted: bool = field(default=False, init=False)
    network_used: bool = field(default=False, init=False)
    identity_inference: bool = field(default=False, init=False)
    paths_disclosed: bool = field(default=False, init=False)
    cohort_consistency_checked: bool = field(default=True, init=False)
    enrollment_policy_revision: str = field(default=ENROLLMENT_POLICY_REVISION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or _PROFILE_ID.fullmatch(self.profile_id) is None:
            raise IdentityEnrollmentError("identity enrollment profile_id is invalid")
        if (
            not isinstance(self.template_ids, tuple)
            or len(self.template_ids) != self.source_image_count
            or len(set(self.template_ids)) != len(self.template_ids)
            or any(
                not isinstance(item, str) or _TEMPLATE_ID.fullmatch(item) is None
                for item in self.template_ids
            )
        ):
            raise IdentityEnrollmentError("identity enrollment template_ids are invalid")
        if (
            not isinstance(self.declared_kind, DeclaredKind)
            or self.declared_kind is DeclaredKind.UNKNOWN
        ):
            raise IdentityEnrollmentError("identity enrollment declared_kind is invalid")
        if (
            not isinstance(self.model_fingerprint, str)
            or _SHA256.fullmatch(self.model_fingerprint) is None
        ):
            raise IdentityEnrollmentError("identity enrollment model_fingerprint is invalid")
        if (
            type(self.embedding_dimension) is not int
            or self.embedding_dimension != FACE_EMBEDDING_DIMENSION
        ):
            raise IdentityEnrollmentError("identity enrollment embedding dimension is invalid")
        if (
            type(self.source_image_count) is not int
            or not _MIN_ENROLLMENT_IMAGES <= self.source_image_count <= _MAX_ENROLLMENT_IMAGES
        ):
            raise IdentityEnrollmentError("identity enrollment source image count is invalid")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "template_ids": list(self.template_ids),
            "declared_kind": self.declared_kind.value,
            "model_fingerprint": self.model_fingerprint,
            "embedding_dimension": self.embedding_dimension,
            "source_image_count": self.source_image_count,
            "source_images_persisted": False,
            "network_used": False,
            "identity_inference": False,
            "paths_disclosed": False,
            "atomic_store_write": True,
            "cohort_consistency_checked": True,
            "enrollment_policy_revision": ENROLLMENT_POLICY_REVISION,
        }


def enroll_identity_from_manifest(
    *,
    settings: IdentitySettings,
    one_shot_confirmation: bool,
    manifest_path: str | Path,
    manifest_root: str | Path,
    extractor: EnrollmentEmbeddingExtractor,
    store_factory: AtomicEnrollmentStoreFactory,
) -> IdentityEnrollmentResult:
    """Authorize, validate every sample, then perform exactly one atomic store write."""

    authorize_enrollment(
        settings,
        one_shot_confirmation=one_shot_confirmation,
    )
    manifest = _read_private_manifest(manifest_path, private_root=manifest_root)

    embeddings: list[tuple[float, ...]] = []
    pipeline_fingerprint: str | None = None
    for ordinal, image in enumerate(manifest.images, start=1):
        try:
            _verify_source_image(image)
        except Exception:
            raise IdentityEnrollmentError(
                f"identity enrollment source verification failed for sample {ordinal}"
            ) from None
        try:
            proof = extractor.extract_validated_enrollment(image)
            if not isinstance(proof, ValidatedEnrollmentEmbedding):
                raise IdentityEnrollmentError(
                    "identity enrollment extractor did not return a binding proof"
                )
            if (
                proof.mode is not FaceEmbeddingMode.ENROLLMENT
                or not proof.request_binding_validated
                or proof.frame_ref != image.frame_ref
                or proof.source_sha256 != image.source_sha256
                or proof.embedding_dimension != FACE_EMBEDDING_DIMENSION
            ):
                raise IdentityEnrollmentError(
                    "identity enrollment extraction does not match its source"
                )
        except Exception:
            raise IdentityEnrollmentError(
                f"identity enrollment extraction failed for sample {ordinal}"
            ) from None
        if pipeline_fingerprint is None:
            pipeline_fingerprint = proof.model_fingerprint
        elif proof.model_fingerprint != pipeline_fingerprint:
            raise IdentityEnrollmentError(
                "identity enrollment pipeline fingerprint changed between samples"
            )
        embeddings.append(proof.embedding)

    if pipeline_fingerprint is None:
        raise IdentityEnrollmentError("identity enrollment has no validated embeddings")
    _validate_enrollment_cohort(embeddings)
    try:
        store = store_factory()
        stored = store.enroll_profile(
            declared_kind=manifest.declared_kind,
            model_fingerprint=pipeline_fingerprint,
            embeddings=tuple(embeddings),
        )
    except Exception:
        raise IdentityEnrollmentError("identity enrollment atomic store write failed") from None
    if (
        not isinstance(stored, IdentityEnrollmentSummary)
        or stored.declared_kind is not manifest.declared_kind
        or stored.model_fingerprint != pipeline_fingerprint
        or stored.embedding_dimension != FACE_EMBEDDING_DIMENSION
        or len(stored.template_ids) != len(embeddings)
    ):
        raise IdentityEnrollmentError("identity enrollment store returned an inconsistent summary")
    return IdentityEnrollmentResult(
        profile_id=stored.profile_id,
        template_ids=stored.template_ids,
        declared_kind=stored.declared_kind,
        model_fingerprint=stored.model_fingerprint,
        embedding_dimension=stored.embedding_dimension,
        source_image_count=len(embeddings),
    )
