#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import secrets
import stat
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.identity_domain import FaceQualityState
from baby_monitor.identity_model import FaceModelManifest
from baby_monitor.identity_protocol import (
    MAX_FACE_IMAGES,
    FaceEmbeddingMode,
    FaceEmbeddingRequest,
    FaceImageInput,
    FaceModelReference,
)
from baby_monitor.identity_review import (
    MAX_REVIEW_CANDIDATES,
    IdentityReviewError,
    ReviewBundle,
    ReviewCandidate,
    admit_private_review_root,
    build_review_html,
    new_review_id,
    sha256_bytes,
    write_private_file,
    write_private_json,
)
from baby_monitor.local_runners.opencv_face import run_request


@dataclass(frozen=True, slots=True)
class Sample:
    ordinal: int
    frame_ref: str
    path: Path = field(repr=False)
    sha256: str


@dataclass(frozen=True, slots=True)
class AcceptedSample:
    sample: Sample
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    embedding: tuple[float, ...] = field(repr=False)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a private, local-only household identity enrollment review.",
    )
    parser.add_argument("--buffer-root", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--detector-model", required=True)
    parser.add_argument("--detector-manifest", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--max-segments", type=int, default=192)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--samples-per-segment", type=int, default=3)
    parser.add_argument("--detector-score-threshold", type=float, default=0.6)
    parser.add_argument(
        "--segment-selection",
        choices=("evenly-spaced", "recent"),
        default="evenly-spaced",
    )
    parser.add_argument("--minimum-segment-age-seconds", type=float, default=45.0)
    parser.add_argument("--confirm-household-biometric-capture", action="store_true")
    return parser.parse_args(argv)


def _existing_directory(value: str, *, private: bool) -> Path:
    try:
        candidate = Path(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise OSError
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
    except (OSError, TypeError, ValueError):
        raise IdentityReviewError("identity review directory is unsafe") from None
    if (
        resolved != candidate
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or (private and stat.S_IMODE(metadata.st_mode) != 0o700)
        or resolved.parts[:3] == ("/", "mnt", "c")
    ):
        raise IdentityReviewError("identity review directory is unsafe")
    return resolved


def _new_run_root(parent: Path, review_id: str) -> Path:
    admitted_parent = admit_private_review_root(parent)
    root = admitted_parent / review_id
    try:
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
    except OSError as exc:
        raise IdentityReviewError("identity review run directory could not be created") from exc
    return admit_private_review_root(root)


def _regular_complete_segments(
    root: Path,
    *,
    minimum_age_seconds: float,
) -> tuple[Path, ...]:
    now = time.time()
    accepted: list[tuple[int, str, Path]] = []
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise IdentityReviewError("identity review buffer could not be enumerated") from exc
    for path in entries:
        if path.suffix.lower() != ".mkv":
            continue
        try:
            metadata = path.lstat()
        except OSError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_uid != os.geteuid()
            or now - metadata.st_mtime < minimum_age_seconds
        ):
            continue
        accepted.append((metadata.st_mtime_ns, path.name, path))
    accepted.sort()
    return tuple(item[2] for item in accepted)


def _evenly_spaced(values: tuple[Path, ...], limit: int) -> tuple[Path, ...]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indexes = tuple(round(index * (len(values) - 1) / (limit - 1)) for index in range(limit))
    return tuple(values[index] for index in indexes)


def _write_sample(root: Path, ordinal: int, payload: bytes) -> Sample:
    frame_ref = f"sample_{ordinal:04d}"
    path = root / f"{frame_ref}.jpg"
    write_private_file(path, payload, maximum_bytes=64 * 1024 * 1024)
    return Sample(
        ordinal=ordinal,
        frame_ref=frame_ref,
        path=path,
        sha256=sha256_bytes(payload),
    )


def _extract_samples(
    segments: tuple[Path, ...],
    *,
    root: Path,
    cv2: object,
    samples_per_segment: int,
) -> tuple[Sample, ...]:
    samples: list[Sample] = []
    for segment in segments:
        capture = cv2.VideoCapture(str(segment))
        try:
            if not capture.isOpened():
                continue
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for sample_index in range(1, samples_per_segment + 1):
                if frame_count > samples_per_segment + 1:
                    position = round(sample_index * (frame_count - 1) / (samples_per_segment + 1))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(1, position))
                elif sample_index > 1:
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    if sample_index != 1:
                        continue
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                shape = getattr(frame, "shape", ())
                if len(shape) != 3 or shape[2] != 3 or shape[0] <= 0 or shape[1] <= 0:
                    continue
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
                if not encoded_ok:
                    continue
                payload = bytes(encoded)
                if not payload.startswith(b"\xff\xd8"):
                    continue
                samples.append(_write_sample(root, len(samples) + 1, payload))
        finally:
            capture.release()
    return tuple(samples)


def _model_reference(
    *,
    role: str,
    model_path: str,
    manifest_path: str,
) -> FaceModelReference:
    model = Path(model_path).resolve(strict=True)
    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest = FaceModelManifest.read(manifest_file)
    manifest.require_reviewed()
    if manifest.role != role:
        raise IdentityReviewError("identity review model role is invalid")
    verification = manifest.verify_model(model)
    return FaceModelReference(
        role=role,
        model_path=str(model),
        manifest_path=str(manifest_file),
        model_id=manifest.model_id,
        upstream_revision=manifest.upstream_revision,
        model_sha256=verification.sha256,
    )


def _assess_samples(
    samples: tuple[Sample, ...],
    *,
    detector: FaceModelReference,
    embedding: FaceModelReference,
    detector_score_threshold: float,
) -> tuple[AcceptedSample, ...]:
    accepted: list[AcceptedSample] = []
    for start in range(0, len(samples), MAX_FACE_IMAGES):
        chunk = samples[start : start + MAX_FACE_IMAGES]
        request = FaceEmbeddingRequest(
            request_id=f"review_{start // MAX_FACE_IMAGES + 1:04d}",
            mode=FaceEmbeddingMode.ASSESSMENT,
            face_adapter="opencv_yunet_sface_v1",
            images=tuple(
                FaceImageInput(
                    frame_ref=sample.frame_ref,
                    source_path=str(sample.path),
                    source_sha256=sample.sha256,
                )
                for sample in chunk
            ),
            detector_model=detector,
            embedding_model=embedding,
            detector_score_threshold=detector_score_threshold,
            detector_nms_threshold=0.3,
            detector_top_k=100,
            min_face_size_pixels=40,
            min_eye_distance_pixels=12.0,
        )
        batch = run_request(request)
        if len(batch.images) != len(chunk):
            raise IdentityReviewError("identity review face result count is invalid")
        for sample, result in zip(chunk, batch.images, strict=True):
            if result.frame_ref != sample.frame_ref or len(result.faces) != 1:
                continue
            face = result.faces[0]
            if face.quality_state is not FaceQualityState.ACCEPTED or face.embedding is None:
                continue
            accepted.append(
                AcceptedSample(
                    sample=sample,
                    bbox_xyxy=face.bbox_xyxy,
                    detector_confidence=face.detector_confidence,
                    embedding=face.embedding,
                )
            )
    return tuple(accepted)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    value = sum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, value))


def _deduplicate(values: tuple[AcceptedSample, ...]) -> tuple[AcceptedSample, ...]:
    """Drop near-identical frames only; this is not an identity classifier."""

    kept: list[AcceptedSample] = []
    for value in values:
        if any(_cosine(value.embedding, other.embedding) >= 0.9995 for other in kept):
            continue
        kept.append(value)
    return tuple(kept)


def _select_evenly(values: tuple[AcceptedSample, ...], limit: int) -> tuple[AcceptedSample, ...]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return (values[len(values) // 2],)
    indexes = tuple(round(index * (len(values) - 1) / (limit - 1)) for index in range(limit))
    return tuple(values[index] for index in indexes)


def _persist_candidates(
    values: tuple[AcceptedSample, ...],
    *,
    root: Path,
    cv2: object,
    numpy: object,
) -> tuple[tuple[ReviewCandidate, ...], dict[str, bytes]]:
    candidates: list[ReviewCandidate] = []
    previews: dict[str, bytes] = {}
    for accepted in values:
        source = accepted.sample.path.read_bytes()
        decoded = cv2.imdecode(numpy.frombuffer(source, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise IdentityReviewError("identity review source JPEG could not be decoded")
        height, width = decoded.shape[:2]
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())
        luma_stddev = float(gray.std())
        if not all(math.isfinite(value) for value in (mean_luma, luma_stddev)):
            raise IdentityReviewError("identity review luma metrics are invalid")
        if mean_luma < 5.0 or luma_stddev < 3.0:
            continue
        candidate_id = f"candidate_{secrets.token_hex(8)}"
        source_filename = f"{candidate_id}.source.jpg"
        preview_filename = f"{candidate_id}.preview.jpg"
        write_private_file(
            root / source_filename,
            source,
            maximum_bytes=64 * 1024 * 1024,
        )
        x1, y1, x2, y2 = accepted.bbox_xyxy
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width - 1, round(x2 * width)))
        bottom = max(top + 1, min(height - 1, round(y2 * height)))
        cv2.rectangle(decoded, (left, top), (right, bottom), (32, 220, 90), 4)
        cv2.putText(
            decoded,
            f"{len(candidates) + 1:02d}",
            (left, max(28, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (32, 220, 90),
            2,
            cv2.LINE_AA,
        )
        preview_ok, preview_encoded = cv2.imencode(
            ".jpg",
            decoded,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not preview_ok:
            raise IdentityReviewError("identity review preview could not be encoded")
        preview = bytes(preview_encoded)
        write_private_file(
            root / preview_filename,
            preview,
            maximum_bytes=64 * 1024 * 1024,
        )
        candidate = ReviewCandidate(
            candidate_id=candidate_id,
            ordinal=len(candidates) + 1,
            source_filename=source_filename,
            source_sha256=sha256_bytes(source),
            preview_filename=preview_filename,
            preview_sha256=sha256_bytes(preview),
            width=int(width),
            height=int(height),
            bbox_xyxy=tuple(float(item) for item in accepted.bbox_xyxy),
            detector_confidence=float(accepted.detector_confidence),
            mean_luma=round(mean_luma, 4),
            luma_stddev=round(luma_stddev, 4),
        )
        candidates.append(candidate)
        previews[candidate_id] = preview
    return tuple(candidates), previews


def _cleanup_samples(samples: tuple[Sample, ...], root: Path) -> None:
    for sample in samples:
        if sample.path.parent != root or not sample.path.name.startswith("sample_"):
            raise IdentityReviewError("identity review sample cleanup boundary is unsafe")
        try:
            sample.path.unlink(missing_ok=True)
        except OSError as exc:
            raise IdentityReviewError("identity review temporary sample cleanup failed") from exc


def prepare(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if not args.confirm_household_biometric_capture:
        raise IdentityReviewError("explicit household biometric capture confirmation is required")
    if not 6 <= args.max_segments <= 512:
        raise IdentityReviewError("identity review max segments is invalid")
    if not 6 <= args.max_candidates <= MAX_REVIEW_CANDIDATES:
        raise IdentityReviewError("identity review max candidates is invalid")
    if not 1 <= args.samples_per_segment <= 3:
        raise IdentityReviewError("identity review samples per segment is invalid")
    if not 0.5 <= args.detector_score_threshold <= 1.0:
        raise IdentityReviewError("identity review detector score threshold is invalid")
    if not 10.0 <= args.minimum_segment_age_seconds <= 3600.0:
        raise IdentityReviewError("identity review minimum segment age is invalid")

    import cv2
    import numpy

    buffer_root = _existing_directory(args.buffer_root, private=False)
    output_parent = _existing_directory(args.output_parent, private=True)
    review_id = new_review_id()
    root = _new_run_root(output_parent, review_id)
    segments = _regular_complete_segments(
        buffer_root,
        minimum_age_seconds=args.minimum_segment_age_seconds,
    )
    if args.segment_selection == "recent":
        selected_segments = segments[-args.max_segments :]
    else:
        selected_segments = _evenly_spaced(segments, args.max_segments)
    if len(selected_segments) < 6:
        raise IdentityReviewError("identity review has too few complete local buffer segments")
    samples = _extract_samples(
        selected_segments,
        root=root,
        cv2=cv2,
        samples_per_segment=args.samples_per_segment,
    )
    if len(samples) < 6:
        raise IdentityReviewError("identity review could not decode enough local samples")
    detector = _model_reference(
        role="detector",
        model_path=args.detector_model,
        manifest_path=args.detector_manifest,
    )
    embedding = _model_reference(
        role="embedding",
        model_path=args.embedding_model,
        manifest_path=args.embedding_manifest,
    )
    accepted = _assess_samples(
        samples,
        detector=detector,
        embedding=embedding,
        detector_score_threshold=args.detector_score_threshold,
    )
    diverse = _deduplicate(accepted)
    selected = _select_evenly(diverse, args.max_candidates)
    if len(selected) < 6:
        raise IdentityReviewError(
            "identity review found fewer than six usable single-face images "
            f"(accepted={len(accepted)}, diverse={len(diverse)})"
        )
    candidates, previews = _persist_candidates(
        selected,
        root=root,
        cv2=cv2,
        numpy=numpy,
    )
    if len(candidates) < 6:
        raise IdentityReviewError("identity review found fewer than six visible candidate images")
    bundle = ReviewBundle(
        review_id=review_id,
        created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        candidates=candidates,
    )
    write_private_json(
        root / "candidates.json",
        bundle.to_private_dict(),
        maximum_bytes=4 * 1024 * 1024,
    )
    write_private_file(
        root / "review.local.html",
        build_review_html(bundle, previews),
        maximum_bytes=32 * 1024 * 1024,
    )
    _cleanup_samples(samples, root)
    print(
        "identity_review_prepared "
        f"segments={len(selected_segments)} decoded={len(samples)} "
        f"quality_accepted={len(accepted)} candidates={len(candidates)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return prepare(argv)
    except Exception as exc:
        if isinstance(exc, IdentityReviewError):
            message = str(exc)
        else:
            message = "identity review preparation failed safely"
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
