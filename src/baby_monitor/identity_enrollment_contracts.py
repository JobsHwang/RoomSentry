"""Shared enrollment image and request-binding proof contracts.

Adapters can construct admitted images and consume one bound embedding without
depending on enrollment authorization, manifest orchestration, or profile stores.
"""

from __future__ import annotations

import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .identity_domain import FaceQualityState
from .identity_protocol import (
    FACE_EMBEDDING_DIMENSION,
    MAX_FACE_IMAGE_BYTES,
    FaceEmbeddingBatch,
    FaceEmbeddingMode,
    FaceEmbeddingRequest,
    FaceEmbeddingStagingRoot,
)


class IdentityEnrollmentError(ValueError):
    """A local biometric enrollment could not be completed safely."""


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROOF_TOKEN = object()


def _safe_path_text(value: str | Path, context: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError:
        raise IdentityEnrollmentError(f"identity enrollment {context} is unsafe") from None
    if (
        not isinstance(text, str)
        or not text
        or len(text) > 4096
        or "\x00" in text
        or ".." in text.replace("\\", "/").split("/")
    ):
        raise IdentityEnrollmentError(f"identity enrollment {context} is unsafe")
    return text


def _local_image_path(value: object) -> str:
    if not isinstance(value, str):
        raise IdentityEnrollmentError("identity enrollment image path is unsafe")
    text = _safe_path_text(value, "image path")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise IdentityEnrollmentError("identity enrollment image path is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except OSError as exc:
        raise IdentityEnrollmentError("identity enrollment image path is unsafe") from exc
    if (
        resolved != candidate
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or not 0 < metadata.st_size <= MAX_FACE_IMAGE_BYTES
    ):
        raise IdentityEnrollmentError("identity enrollment image path is unsafe")
    return str(candidate)


@dataclass(frozen=True, slots=True)
class EnrollmentImage:
    frame_ref: str
    source_path: str = field(repr=False)
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame_ref, str) or _SAFE_ID.fullmatch(self.frame_ref) is None:
            raise IdentityEnrollmentError("identity enrollment frame_ref is invalid")
        object.__setattr__(self, "source_path", _local_image_path(self.source_path))
        if not isinstance(self.source_sha256, str) or _SHA256.fullmatch(self.source_sha256) is None:
            raise IdentityEnrollmentError("identity enrollment source_sha256 is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> EnrollmentImage:
        if not isinstance(value, dict) or set(value) != {
            "frame_ref",
            "source_path",
            "source_sha256",
        }:
            raise IdentityEnrollmentError("identity enrollment image fields do not match v1")
        return cls(**value)


class ValidatedEnrollmentEmbedding:
    """Path-free one-shot proof produced only after full request binding validation."""

    __slots__ = (
        "_embedding",
        "_frame_ref",
        "_mode",
        "_model_fingerprint",
        "_source_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        frame_ref: str,
        source_sha256: str,
        model_fingerprint: str,
        embedding: tuple[float, ...],
    ) -> None:
        if token is not _PROOF_TOKEN:
            raise IdentityEnrollmentError(
                "validated enrollment embeddings require a consumed request binding"
            )
        object.__setattr__(self, "_mode", FaceEmbeddingMode.ENROLLMENT)
        object.__setattr__(self, "_frame_ref", frame_ref)
        object.__setattr__(self, "_source_sha256", source_sha256)
        object.__setattr__(self, "_model_fingerprint", model_fingerprint)
        object.__setattr__(self, "_embedding", embedding)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("validated enrollment embeddings are immutable")

    @classmethod
    def consume_bound_batch(
        cls,
        request: FaceEmbeddingRequest,
        batch: FaceEmbeddingBatch,
        *,
        replay_guard: FaceEmbeddingStagingRoot,
    ) -> ValidatedEnrollmentEmbedding:
        """Consume a fully bound one-image enrollment result into a path-free proof."""

        if not isinstance(request, FaceEmbeddingRequest) or not isinstance(
            batch, FaceEmbeddingBatch
        ):
            raise IdentityEnrollmentError("identity enrollment binding objects are invalid")
        if not isinstance(replay_guard, FaceEmbeddingStagingRoot):
            raise IdentityEnrollmentError("identity enrollment replay guard is invalid")
        try:
            batch.validate_against_request(request, replay_guard=replay_guard)
        except Exception:
            raise IdentityEnrollmentError(
                "identity enrollment request binding validation failed"
            ) from None
        if (
            request.mode is not FaceEmbeddingMode.ENROLLMENT
            or batch.mode is not FaceEmbeddingMode.ENROLLMENT
            or len(request.images) != 1
            or len(batch.images) != 1
            or batch.identity_inference
            or batch.paths_disclosed
            or batch.network_used
        ):
            raise IdentityEnrollmentError(
                "identity enrollment result is not a local enrollment result"
            )
        image = batch.images[0]
        if (
            batch.face_count != 1
            or len(image.faces) != 1
            or batch.accepted_face_count != 1
            or image.faces[0].quality_state is not FaceQualityState.ACCEPTED
        ):
            raise IdentityEnrollmentError(
                "identity enrollment result requires exactly one face and it must be accepted"
            )
        embedding = image.faces[0].embedding
        if (
            not isinstance(embedding, tuple)
            or len(embedding) != FACE_EMBEDDING_DIMENSION
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in embedding
            )
            or not math.isclose(
                math.hypot(*(float(item) for item in embedding)),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise IdentityEnrollmentError(
                "identity enrollment embedding must be a normalized 128D vector"
            )
        return cls(
            _PROOF_TOKEN,
            frame_ref=image.frame_ref,
            source_sha256=image.source_sha256,
            model_fingerprint=batch.model_fingerprint,
            embedding=embedding,
        )

    @property
    def mode(self) -> FaceEmbeddingMode:
        return self._mode

    @property
    def frame_ref(self) -> str:
        return self._frame_ref

    @property
    def source_sha256(self) -> str:
        return self._source_sha256

    @property
    def model_fingerprint(self) -> str:
        return self._model_fingerprint

    @property
    def embedding(self) -> tuple[float, ...]:
        return self._embedding

    @property
    def embedding_dimension(self) -> int:
        return len(self._embedding)

    @property
    def request_binding_validated(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            "ValidatedEnrollmentEmbedding("
            "mode='enrollment', embedding_dimension=128, "
            "request_binding_validated=True)"
        )
