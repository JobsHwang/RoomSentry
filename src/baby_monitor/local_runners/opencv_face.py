from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from baby_monitor.identity_domain import FaceQualityState
from baby_monitor.identity_model import FaceModelManifest
from baby_monitor.identity_protocol import (
    FACE_EMBEDDING_DIMENSION,
    FACE_LANDMARK_NAMES,
    FACE_QUALITY_POLICY,
    MAX_FACE_IMAGE_BYTES,
    MAX_FACES_PER_IMAGE,
    MAX_TOTAL_FACES,
    FaceEmbeddingBatch,
    FaceEmbeddingObservation,
    FaceEmbeddingProtocolError,
    FaceEmbeddingRequest,
    FaceEmbeddingStagingRoot,
    FaceImageEmbeddingResult,
    FaceImageInput,
    FaceLandmark,
    FaceModelReference,
    canonical_face_pipeline_fingerprint,
)

MAX_DECODED_IMAGE_PIXELS = 50_000_000
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(MAX_DECODED_IMAGE_PIXELS)


def _read_bounded_regular_bytes(
    path: str,
    *,
    maximum_bytes: int,
    context: str,
) -> bytes:
    """Read one regular file through one no-follow fd into bounded immutable bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise OSError
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise OSError
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or after.st_size != total:
            raise OSError
        return b"".join(chunks)
    except OSError as exc:
        raise FaceEmbeddingProtocolError(f"{context} is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _admit_model(reference: FaceModelReference) -> tuple[FaceModelManifest, bytes]:
    manifest = FaceModelManifest.read(reference.manifest_path)
    manifest.require_reviewed()
    if (
        manifest.role != reference.role
        or manifest.model_id != reference.model_id
        or manifest.upstream_revision != reference.upstream_revision
        or manifest.sha256 != reference.model_sha256
    ):
        raise FaceEmbeddingProtocolError(
            "face runner request does not match the reviewed model manifest"
        )
    if Path(reference.model_path).name != manifest.filename:
        raise FaceEmbeddingProtocolError("face runner model filename does not match manifest")
    payload = _read_bounded_regular_bytes(
        reference.model_path,
        maximum_bytes=manifest.size_bytes,
        context="face runner model",
    )
    if len(payload) != manifest.size_bytes or _payload_sha256(payload) != manifest.sha256:
        raise FaceEmbeddingProtocolError("face runner model bytes do not match manifest")
    return manifest, payload


def _admit_images(images: tuple[FaceImageInput, ...]) -> None:
    for image in images:
        payload = _read_bounded_regular_bytes(
            image.source_path,
            maximum_bytes=MAX_FACE_IMAGE_BYTES,
            context="face image source",
        )
        if _payload_sha256(payload) != image.source_sha256:
            raise FaceEmbeddingProtocolError("face image source hash does not match request")


def _read_admitted_image_bytes(image: FaceImageInput) -> bytes:
    payload = _read_bounded_regular_bytes(
        image.source_path,
        maximum_bytes=MAX_FACE_IMAGE_BYTES,
        context="face image source",
    )
    if _payload_sha256(payload) != image.source_sha256:
        raise FaceEmbeddingProtocolError("face image source changed after admission")
    return payload


def _write_private_model(root: Path, filename: str, payload: bytes) -> str:
    root_descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            filename,
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
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise OSError
    except OSError as exc:
        raise FaceEmbeddingProtocolError("private face model staging failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_descriptor)
    return str(root / filename)


@contextmanager
def _materialized_models(
    detector_manifest: FaceModelManifest,
    detector_payload: bytes,
    embedding_manifest: FaceModelManifest,
    embedding_payload: bytes,
) -> Iterator[tuple[str, str]]:
    if detector_manifest.filename == embedding_manifest.filename:
        raise FaceEmbeddingProtocolError("face model filenames must be distinct")
    try:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-face-models-") as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            metadata = root.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise OSError
            detector_path = _write_private_model(
                root,
                detector_manifest.filename,
                detector_payload,
            )
            embedding_path = _write_private_model(
                root,
                embedding_manifest.filename,
                embedding_payload,
            )
            yield detector_path, embedding_path
    except FaceEmbeddingProtocolError:
        raise
    except OSError as exc:
        raise FaceEmbeddingProtocolError("private face model workspace failed") from exc


def _image_shape(image: object) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) not in {2, 3}:
        raise FaceEmbeddingProtocolError("face runner decoded an invalid image")
    try:
        height = int(shape[0])
        width = int(shape[1])
    except (TypeError, ValueError) as exc:
        raise FaceEmbeddingProtocolError("face runner decoded an invalid image") from exc
    if (
        height <= 0
        or width <= 0
        or height > 65_535
        or width > 65_535
        or height * width > MAX_DECODED_IMAGE_PIXELS
    ):
        raise FaceEmbeddingProtocolError("face runner decoded image dimensions are unsafe")
    if len(shape) == 3 and shape[2] != 3:
        raise FaceEmbeddingProtocolError("face runner requires a three-channel BGR image")
    return width, height


def _row_values(row: object) -> tuple[float, ...]:
    try:
        raw = row.tolist() if hasattr(row, "tolist") else row
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        if not isinstance(raw, (list, tuple)) or len(raw) != 15:
            raise TypeError
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise FaceEmbeddingProtocolError("face runner returned a malformed YuNet row") from exc
    if any(not math.isfinite(item) for item in values):
        raise FaceEmbeddingProtocolError("face runner returned nonfinite YuNet values")
    if not 0.0 <= values[14] <= 1.0:
        raise FaceEmbeddingProtocolError("face runner returned an invalid detector score")
    return values


def _face_rows(faces: object) -> tuple[tuple[object, tuple[float, ...]], ...]:
    if faces is None:
        return ()
    try:
        count = len(faces)  # type: ignore[arg-type]
    except TypeError as exc:
        raise FaceEmbeddingProtocolError("face runner returned malformed detections") from exc
    if count > MAX_FACES_PER_IMAGE:
        raise FaceEmbeddingProtocolError("face runner returned too many faces")
    try:
        return tuple((faces[index], _row_values(faces[index])) for index in range(count))  # type: ignore[index]
    except (IndexError, KeyError) as exc:
        raise FaceEmbeddingProtocolError("face runner returned malformed detections") from exc


def _landmark(
    name: str,
    x: float,
    y: float,
    *,
    width: int,
    height: int,
) -> FaceLandmark:
    in_frame = 0.0 <= x < width and 0.0 <= y < height
    return FaceLandmark(
        name=name,
        x=x / width if in_frame else None,
        y=y / height if in_frame else None,
        in_frame=in_frame,
    )


def _normalized_embedding(feature: object) -> tuple[float, ...]:
    try:
        if hasattr(feature, "reshape"):
            raw = feature.reshape(-1).tolist()  # type: ignore[union-attr]
        elif hasattr(feature, "tolist"):
            raw = feature.tolist()  # type: ignore[union-attr]
        else:
            raw = feature
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        if not isinstance(raw, (list, tuple)) or len(raw) != FACE_EMBEDDING_DIMENSION:
            raise TypeError
        values = tuple(float(item) for item in raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise FaceEmbeddingProtocolError("SFace returned an invalid embedding shape") from exc
    if any(not math.isfinite(item) for item in values):
        raise FaceEmbeddingProtocolError("SFace returned a nonfinite embedding")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm <= 0.0:
        raise FaceEmbeddingProtocolError("SFace returned a zero-norm embedding")
    normalized = tuple(item / norm for item in values)
    if abs(math.hypot(*normalized) - 1.0) > 1e-9:
        raise FaceEmbeddingProtocolError("SFace embedding normalization failed")
    return normalized


def _pose_quality_rejection_codes(
    coordinates: tuple[tuple[float, float], ...],
    *,
    eye_distance: float,
) -> tuple[str, ...]:
    """Return only coarse five-point pose/geometry vetoes, never identity claims."""

    right_eye, left_eye, nose, right_mouth, left_mouth = coordinates
    eye_dx = left_eye[0] - right_eye[0]
    eye_dy = left_eye[1] - right_eye[1]
    mouth_dx = left_mouth[0] - right_mouth[0]
    reasons: list[str] = []

    roll_degrees = abs(math.degrees(math.atan2(eye_dy, eye_dx)))
    if roll_degrees > FACE_QUALITY_POLICY.max_abs_roll_degrees:
        reasons.append("excessive_roll")

    eye_midpoint = (
        (right_eye[0] + left_eye[0]) / 2.0,
        (right_eye[1] + left_eye[1]) / 2.0,
    )
    nose_offset_ratio = abs(nose[0] - eye_midpoint[0]) / eye_distance
    if nose_offset_ratio > FACE_QUALITY_POLICY.max_abs_nose_offset_eye_ratio:
        reasons.append("excessive_yaw")

    mouth_distance = math.dist(right_mouth, left_mouth)
    mouth_eye_ratio = mouth_distance / eye_distance
    mouth_midpoint = (
        (right_mouth[0] + left_mouth[0]) / 2.0,
        (right_mouth[1] + left_mouth[1]) / 2.0,
    )
    eye_mouth_vertical = mouth_midpoint[1] - eye_midpoint[1]
    geometry_invalid = (
        eye_dx <= 0.0
        or mouth_dx <= 0.0
        or not FACE_QUALITY_POLICY.min_mouth_eye_ratio
        <= mouth_eye_ratio
        <= FACE_QUALITY_POLICY.max_mouth_eye_ratio
        or not FACE_QUALITY_POLICY.min_eye_mouth_vertical_ratio
        <= eye_mouth_vertical / eye_distance
        <= FACE_QUALITY_POLICY.max_eye_mouth_vertical_ratio
    )
    if eye_mouth_vertical <= 0.0:
        geometry_invalid = True
    else:
        nose_vertical_fraction = (nose[1] - eye_midpoint[1]) / eye_mouth_vertical
        if (
            not FACE_QUALITY_POLICY.min_nose_vertical_fraction
            <= nose_vertical_fraction
            <= (FACE_QUALITY_POLICY.max_nose_vertical_fraction)
        ):
            geometry_invalid = True
    if geometry_invalid:
        reasons.append("invalid_five_point_geometry")
    return tuple(reasons)


def _aligned_pixel_quality_rejection_codes(
    aligned: object,
    *,
    cv2_module: object,
    numpy_module: object,
) -> tuple[str, ...]:
    """Measure blur/exposure on an in-memory SFace-aligned crop without retaining it."""

    width, height = _image_shape(aligned)
    try:
        gray = cv2_module.cvtColor(  # type: ignore[attr-defined]
            aligned,
            cv2_module.COLOR_BGR2GRAY,  # type: ignore[attr-defined]
        )
        if _image_shape(gray) != (width, height):
            raise TypeError
        gray_array = numpy_module.asarray(gray)  # type: ignore[attr-defined]
        mean_luma = float(numpy_module.mean(gray_array))  # type: ignore[attr-defined]
        laplacian = cv2_module.Laplacian(  # type: ignore[attr-defined]
            gray,
            cv2_module.CV_64F,  # type: ignore[attr-defined]
            ksize=FACE_QUALITY_POLICY.laplacian_kernel_size,
        )
        if _image_shape(laplacian) != (width, height):
            raise TypeError
        laplacian_variance = float(
            numpy_module.var(numpy_module.asarray(laplacian))  # type: ignore[attr-defined]
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise FaceEmbeddingProtocolError("face quality measurement failed") from exc
    if (
        not math.isfinite(mean_luma)
        or not 0.0 <= mean_luma <= 255.0
        or not math.isfinite(laplacian_variance)
        or laplacian_variance < 0.0
    ):
        raise FaceEmbeddingProtocolError("face quality measurement was invalid")

    reasons: list[str] = []
    if laplacian_variance < FACE_QUALITY_POLICY.min_laplacian_variance:
        reasons.append("face_too_blurry")
    if mean_luma < FACE_QUALITY_POLICY.min_mean_luma:
        reasons.append("underexposed")
    elif mean_luma > FACE_QUALITY_POLICY.max_mean_luma:
        reasons.append("overexposed")
    return tuple(reasons)


def _observation(
    *,
    image: object,
    raw_row: object,
    row: tuple[float, ...],
    image_index: int,
    face_index: int,
    frame_ref: str,
    width: int,
    height: int,
    request: FaceEmbeddingRequest,
    recognizer: object,
    cv2_module: object,
    numpy_module: object,
) -> FaceEmbeddingObservation:
    x, y, box_width, box_height = row[:4]
    if box_width <= 0.0 or box_height <= 0.0:
        raise FaceEmbeddingProtocolError("face runner returned a non-positive face box")
    clipped_x1 = min(float(width), max(0.0, x))
    clipped_y1 = min(float(height), max(0.0, y))
    clipped_x2 = min(float(width), max(0.0, x + box_width))
    clipped_y2 = min(float(height), max(0.0, y + box_height))
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        raise FaceEmbeddingProtocolError("face runner returned a face outside the image")

    coordinates = tuple((row[4 + index * 2], row[5 + index * 2]) for index in range(5))
    landmarks = tuple(
        _landmark(name, point_x, point_y, width=width, height=height)
        for name, (point_x, point_y) in zip(
            FACE_LANDMARK_NAMES,
            coordinates,
            strict=True,
        )
    )
    eye_distance = math.dist(coordinates[0], coordinates[1])
    reasons: list[str] = []
    if x < 0.0 or y < 0.0 or x + box_width > width or y + box_height > height:
        reasons.append("bbox_out_of_frame")
    if (
        clipped_x2 - clipped_x1 < request.min_face_size_pixels
        or clipped_y2 - clipped_y1 < request.min_face_size_pixels
    ):
        reasons.append("face_too_small")
    if any(not landmark.in_frame for landmark in landmarks):
        reasons.append("landmarks_out_of_frame")
    if eye_distance < request.min_eye_distance_pixels:
        reasons.append("eyes_too_close")
    if row[14] < request.detector_score_threshold:
        reasons.append("below_detector_threshold")

    if not reasons:
        reasons.extend(_pose_quality_rejection_codes(coordinates, eye_distance=eye_distance))

    embedding: tuple[float, ...] | None = None
    aligned: object | None = None
    if not reasons:
        try:
            aligned = recognizer.alignCrop(image, raw_row)  # type: ignore[attr-defined]
        except Exception as exc:
            raise FaceEmbeddingProtocolError("SFace alignment failed") from exc
        reasons.extend(
            _aligned_pixel_quality_rejection_codes(
                aligned,
                cv2_module=cv2_module,
                numpy_module=numpy_module,
            )
        )
    if not reasons and aligned is not None:
        try:
            feature = recognizer.feature(aligned)  # type: ignore[attr-defined]
        except Exception as exc:
            raise FaceEmbeddingProtocolError("SFace embedding failed") from exc
        embedding = _normalized_embedding(feature)
    quality_state = FaceQualityState.REJECTED if reasons else FaceQualityState.ACCEPTED

    return FaceEmbeddingObservation(
        face_id=f"f{image_index:04d}_{face_index:04d}",
        frame_ref=frame_ref,
        bbox_xyxy=(
            clipped_x1 / width,
            clipped_y1 / height,
            clipped_x2 / width,
            clipped_y2 / height,
        ),
        detector_confidence=row[14],
        landmarks=landmarks,
        quality_state=quality_state,
        rejection_codes=tuple(reasons),
        embedding=embedding,
    )


def run_request(request: FaceEmbeddingRequest) -> FaceEmbeddingBatch:
    """Run admitted bytes locally; an outer process memory limit remains required."""

    detector_manifest, detector_payload = _admit_model(request.detector_model)
    embedding_manifest, embedding_payload = _admit_model(request.embedding_model)
    if detector_manifest.embedding_dimension is not None:
        raise FaceEmbeddingProtocolError("YuNet detector manifest has an embedding dimension")
    if embedding_manifest.embedding_dimension != FACE_EMBEDDING_DIMENSION:
        raise FaceEmbeddingProtocolError("SFace manifest must declare 128 dimensions")
    _admit_images(request.images)

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FaceEmbeddingProtocolError("OpenCV/NumPy face runtime dependency is missing") from exc

    with _materialized_models(
        detector_manifest,
        detector_payload,
        embedding_manifest,
        embedding_payload,
    ) as (detector_path, embedding_path):
        try:
            detector = cv2.FaceDetectorYN.create(
                detector_path,
                "",
                (detector_manifest.input_width, detector_manifest.input_height),
                request.detector_score_threshold,
                request.detector_nms_threshold,
                request.detector_top_k,
            )
            recognizer = cv2.FaceRecognizerSF.create(embedding_path, "")
        except Exception as exc:
            raise FaceEmbeddingProtocolError("OpenCV face models could not be loaded") from exc

        image_results: list[FaceImageEmbeddingResult] = []
        total_faces = 0
        for image_index, admitted_image in enumerate(request.images, start=1):
            encoded_bytes = _read_admitted_image_bytes(admitted_image)
            try:
                encoded = numpy.frombuffer(encoded_bytes, dtype=numpy.uint8)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            except Exception as exc:
                raise FaceEmbeddingProtocolError(
                    "OpenCV could not decode the admitted image bytes"
                ) from exc
            if decoded is None:
                raise FaceEmbeddingProtocolError("OpenCV could not decode the admitted image bytes")
            width, height = _image_shape(decoded)
            try:
                detector.setInputSize((width, height))
                detected = detector.detect(decoded)
            except Exception as exc:
                raise FaceEmbeddingProtocolError("YuNet face detection failed") from exc
            if not isinstance(detected, (list, tuple)) or len(detected) != 2:
                raise FaceEmbeddingProtocolError("YuNet returned an invalid detection response")
            rows = _face_rows(detected[1])
            total_faces += len(rows)
            if total_faces > MAX_TOTAL_FACES:
                raise FaceEmbeddingProtocolError("face runner returned too many total faces")
            faces = tuple(
                _observation(
                    image=decoded,
                    raw_row=raw_row,
                    row=row,
                    image_index=image_index,
                    face_index=face_index,
                    frame_ref=admitted_image.frame_ref,
                    width=width,
                    height=height,
                    request=request,
                    recognizer=recognizer,
                    cv2_module=cv2,
                    numpy_module=numpy,
                )
                for face_index, (raw_row, row) in enumerate(rows, start=1)
            )
            image_results.append(
                FaceImageEmbeddingResult(
                    frame_ref=admitted_image.frame_ref,
                    source_sha256=admitted_image.source_sha256,
                    width=width,
                    height=height,
                    faces=faces,
                )
            )

        return FaceEmbeddingBatch(
            request_id=request.request_id,
            request_sha256=request.canonical_sha256(),
            mode=request.mode,
            face_adapter=request.face_adapter,
            detector_model_id=detector_manifest.model_id,
            detector_model_revision=detector_manifest.upstream_revision,
            detector_model_sha256=detector_manifest.sha256,
            embedding_model_id=embedding_manifest.model_id,
            embedding_model_revision=embedding_manifest.upstream_revision,
            embedding_model_sha256=embedding_manifest.sha256,
            model_fingerprint=canonical_face_pipeline_fingerprint(
                request,
                detector_manifest=detector_manifest,
                embedding_manifest=embedding_manifest,
            ),
            images=tuple(image_results),
        )


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 3:
        print("face_embedding_runner_failed", file=sys.stderr)
        return 2
    staging_root, request_path, output_path = values
    try:
        staging = FaceEmbeddingStagingRoot(staging_root)
        request = staging.read_request(Path(request_path))
        result = run_request(request)
        staging.write_batch(result, Path(output_path))
    except (OSError, RuntimeError, ValueError):
        print("face_embedding_runner_failed", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "image_count": len(result.images),
                "face_count": result.face_count,
                "accepted_face_count": result.accepted_face_count,
                "identity_inference": False,
                "paths_disclosed": False,
                "network_used": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
