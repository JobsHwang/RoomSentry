from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from baby_monitor.identity_domain import FaceQualityState
from baby_monitor.identity_protocol import (
    FACE_LANDMARK_NAMES,
    FaceEmbeddingBatch,
    FaceEmbeddingMode,
    FaceEmbeddingObservation,
    FaceEmbeddingProtocolError,
    FaceEmbeddingRequest,
    FaceEmbeddingStagingRoot,
    FaceImageEmbeddingResult,
    FaceImageInput,
    FaceLandmark,
    FaceModelReference,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _files(root: Path) -> dict[str, Path]:
    values = {
        "image": root / "public-synthetic.jpg",
        "detector": root / "detector.onnx",
        "detector_manifest": root / "detector.json",
        "embedding": root / "embedding.onnx",
        "embedding_manifest": root / "embedding.json",
    }
    for name, path in values.items():
        path.write_bytes(f"synthetic-{name}".encode())
    return values


def _reference(
    files: dict[str, Path],
    *,
    role: str,
) -> FaceModelReference:
    return FaceModelReference(
        role=role,
        model_path=str(files[role].resolve()),
        manifest_path=str(files[f"{role}_manifest"].resolve()),
        model_id=f"fixture.{role}",
        upstream_revision=("1" if role == "detector" else "2") * 40,
        model_sha256=("a" if role == "detector" else "b") * 64,
    )


def _request(
    root: Path,
    *,
    mode: FaceEmbeddingMode = FaceEmbeddingMode.ASSESSMENT,
) -> FaceEmbeddingRequest:
    files = _files(root)
    image_content = files["image"].read_bytes()
    return FaceEmbeddingRequest(
        request_id="request_01",
        mode=mode,
        face_adapter="opencv_yunet_sface_v1",
        images=(
            FaceImageInput(
                frame_ref="frame_01",
                source_path=str(files["image"].resolve()),
                source_sha256=_sha(image_content),
            ),
        ),
        detector_model=_reference(files, role="detector"),
        embedding_model=_reference(files, role="embedding"),
        detector_score_threshold=0.8,
        detector_nms_threshold=0.3,
        detector_top_k=100,
        min_face_size_pixels=40,
        min_eye_distance_pixels=12.0,
    )


def _landmarks() -> tuple[FaceLandmark, ...]:
    return tuple(
        FaceLandmark(name=name, x=0.5, y=0.5, in_frame=True) for name in FACE_LANDMARK_NAMES
    )


def _accepted(face_id: str = "face_01") -> FaceEmbeddingObservation:
    return FaceEmbeddingObservation(
        face_id=face_id,
        frame_ref="frame_01",
        bbox_xyxy=(0.1, 0.2, 0.8, 0.9),
        detector_confidence=0.95,
        landmarks=_landmarks(),
        quality_state=FaceQualityState.ACCEPTED,
        rejection_codes=(),
        embedding=(1.0,) + (0.0,) * 127,
    )


def _batch(
    *,
    mode: FaceEmbeddingMode = FaceEmbeddingMode.ASSESSMENT,
    faces: tuple[FaceEmbeddingObservation, ...] = (),
    request_sha256: str = "e" * 64,
    source_sha256: str = "d" * 64,
    model_fingerprint: str = "c" * 64,
) -> FaceEmbeddingBatch:
    return FaceEmbeddingBatch(
        request_id="request_01",
        request_sha256=request_sha256,
        mode=mode,
        face_adapter="opencv_yunet_sface_v1",
        detector_model_id="fixture.detector",
        detector_model_revision="1" * 40,
        detector_model_sha256="a" * 64,
        embedding_model_id="fixture.embedding",
        embedding_model_revision="2" * 40,
        embedding_model_sha256="b" * 64,
        model_fingerprint=model_fingerprint,
        images=(
            FaceImageEmbeddingResult(
                frame_ref="frame_01",
                source_sha256=source_sha256,
                width=640,
                height=480,
                faces=faces,
            ),
        ),
    )


class FaceEmbeddingProtocolTests(unittest.TestCase):
    def test_private_request_is_strict_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory).resolve())
            self.assertEqual(request, FaceEmbeddingRequest.from_json(request.to_json()))

            payload = request.to_dict()
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "fields do not match"):
                FaceEmbeddingRequest.from_dict({**payload, "extra": True})
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "duplicate JSON keys"):
                FaceEmbeddingRequest.from_json('{"version":2,"version":2}')

            duplicate = FaceImageInput(
                frame_ref=request.images[0].frame_ref,
                source_path=request.images[0].source_path,
                source_sha256=request.images[0].source_sha256,
            )
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "frame IDs"):
                replace(request, images=(request.images[0], duplicate))

    def test_input_rejects_traversal_symlink_and_nonregular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "image.jpg"
            source.write_bytes(b"fixture")
            traversal = f"{root}/child/../image.jpg"
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "unsafe"):
                FaceImageInput(
                    frame_ref="frame_01",
                    source_path=traversal,
                    source_sha256=_sha(b"fixture"),
                )
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "unsafe"):
                FaceImageInput(
                    frame_ref="frame_01",
                    source_path=str(root),
                    source_sha256=_sha(b"fixture"),
                )
            if os.name == "posix":
                link = root / "image-link.jpg"
                link.symlink_to(source)
                with self.assertRaisesRegex(FaceEmbeddingProtocolError, "unsafe"):
                    FaceImageInput(
                        frame_ref="frame_01",
                        source_path=str(link),
                        source_sha256=_sha(b"fixture"),
                    )

    def test_assessment_allows_zero_or_many_faces_but_enrollment_requires_one(self) -> None:
        self.assertEqual(0, _batch().face_count)
        self.assertEqual(2, _batch(faces=(_accepted("face_01"), _accepted("face_02"))).face_count)
        enrollment = _batch(
            mode=FaceEmbeddingMode.ENROLLMENT,
            faces=(_accepted(),),
        )
        self.assertEqual(1, enrollment.accepted_face_count)

        with self.assertRaisesRegex(FaceEmbeddingProtocolError, "exactly one"):
            _batch(mode=FaceEmbeddingMode.ENROLLMENT)
        with self.assertRaisesRegex(FaceEmbeddingProtocolError, "exactly one"):
            _batch(
                mode=FaceEmbeddingMode.ENROLLMENT,
                faces=(_accepted("face_01"), _accepted("face_02")),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root, mode=FaceEmbeddingMode.ENROLLMENT)
            second = root / "second.jpg"
            second.write_bytes(b"second")
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "exactly one image"):
                replace(
                    request,
                    images=(
                        request.images[0],
                        FaceImageInput(
                            frame_ref="frame_02",
                            source_path=str(second),
                            source_sha256=_sha(b"second"),
                        ),
                    ),
                )

    def test_embedding_is_finite_normalized_and_never_an_identity_claim(self) -> None:
        accepted = _accepted()
        self.assertAlmostEqual(1.0, sum(item * item for item in accepted.embedding or ()))
        self.assertEqual(accepted.face_id, accepted.observation.face_id)

        with self.assertRaisesRegex(FaceEmbeddingProtocolError, "L2-normalized"):
            FaceEmbeddingObservation(
                face_id="face_01",
                frame_ref="frame_01",
                bbox_xyxy=(0.1, 0.2, 0.8, 0.9),
                detector_confidence=0.95,
                landmarks=_landmarks(),
                quality_state=FaceQualityState.ACCEPTED,
                rejection_codes=(),
                embedding=(0.5,) + (0.0,) * 127,
            )
        with self.assertRaisesRegex(FaceEmbeddingProtocolError, "embedding and no rejection"):
            FaceEmbeddingObservation(
                face_id="face_01",
                frame_ref="frame_01",
                bbox_xyxy=(0.1, 0.2, 0.8, 0.9),
                detector_confidence=0.95,
                landmarks=_landmarks(),
                quality_state=FaceQualityState.ACCEPTED,
                rejection_codes=("face_too_small",),
                embedding=(1.0,) + (0.0,) * 127,
            )
        with self.assertRaisesRegex(FaceEmbeddingProtocolError, "not approved"):
            FaceEmbeddingObservation(
                face_id="face_01",
                frame_ref="frame_01",
                bbox_xyxy=(0.1, 0.2, 0.8, 0.9),
                detector_confidence=0.95,
                landmarks=_landmarks(),
                quality_state=FaceQualityState.REJECTED,
                rejection_codes=("operator_supplied_detail",),
                embedding=None,
            )

        payload = _batch(faces=(accepted,)).to_dict()
        payload["identity_inference"] = True
        with self.assertRaisesRegex(FaceEmbeddingProtocolError, "identity_inference"):
            FaceEmbeddingBatch.from_dict(payload)

    def test_batch_binding_recomputes_request_and_rejects_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root)
            staging_path = root / "private-staging"
            staging_path.mkdir(mode=0o700)
            staging = FaceEmbeddingStagingRoot(str(staging_path))
            batch = _batch(
                request_sha256=request.canonical_sha256(),
                source_sha256=request.images[0].source_sha256,
            )

            tampered = (
                replace(batch, request_sha256="f" * 64),
                replace(batch, request_id="request_02"),
                replace(batch, face_adapter="different_adapter_v1"),
                replace(batch, detector_model_sha256="f" * 64),
                replace(
                    batch,
                    images=(replace(batch.images[0], source_sha256="f" * 64),),
                ),
                replace(batch, model_fingerprint="f" * 64),
            )
            with patch(
                "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
                return_value="c" * 64,
            ):
                for candidate in tampered:
                    with (
                        self.subTest(candidate=candidate),
                        self.assertRaises(FaceEmbeddingProtocolError),
                    ):
                        candidate.validate_against_request(
                            request,
                            replay_guard=staging,
                        )

                batch.validate_against_request(request, replay_guard=staging)
                with self.assertRaisesRegex(FaceEmbeddingProtocolError, "replay"):
                    batch.validate_against_request(request, replay_guard=staging)

                renewed = replace(
                    request,
                    request_nonce=hashlib.sha256(b"renewed request").hexdigest(),
                )
                with self.assertRaisesRegex(FaceEmbeddingProtocolError, "canonical"):
                    batch.validate_against_request(renewed, replay_guard=staging)

            second_path = root / "second.jpg"
            second_path.write_bytes(b"second")
            second = FaceImageInput(
                frame_ref="frame_02",
                source_path=str(second_path),
                source_sha256=_sha(b"second"),
            )
            two_image_request = replace(
                request,
                request_nonce=hashlib.sha256(b"two image request").hexdigest(),
                images=(request.images[0], second),
            )
            first_result = replace(
                batch.images[0],
                source_sha256=request.images[0].source_sha256,
            )
            second_result = FaceImageEmbeddingResult(
                frame_ref=second.frame_ref,
                source_sha256=second.source_sha256,
                width=320,
                height=240,
                faces=(),
            )
            reversed_batch = replace(
                batch,
                request_sha256=two_image_request.canonical_sha256(),
                images=(second_result, first_result),
            )
            with (
                patch(
                    "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
                    return_value="c" * 64,
                ),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "order"),
            ):
                reversed_batch.validate_against_request(
                    two_image_request,
                    replay_guard=staging,
                )

            self.assertNotIn(request.request_nonce, batch.to_json())
            self.assertNotIn(request.images[0].source_path, batch.to_json())

    @unittest.skipUnless(os.name == "posix", "private staging requires POSIX/WSL")
    def test_private_staging_rejects_weak_permissions_links_and_wrong_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root)
            staging_path = root / "private-staging"
            staging_path.mkdir(mode=0o700)
            staging = FaceEmbeddingStagingRoot(str(staging_path))
            request_path = staging_path / "request.json"
            request.write_atomic(request_path, staging_root=staging.root)
            self.assertEqual(0o600, request_path.stat().st_mode & 0o777)
            self.assertEqual(request, staging.read_request(request_path))

            request_path.chmod(0o640)
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "owner-only"):
                staging.read_request(request_path)
            request_path.chmod(0o600)

            hardlink = staging_path / "request-hardlink.json"
            os.link(request_path, hardlink)
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "owner-only"):
                staging.read_request(request_path)
            hardlink.unlink()

            symlink = staging_path / "request-symlink.json"
            symlink.symlink_to(request_path)
            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "owner-only"):
                staging.read_request(symlink)

            with self.assertRaisesRegex(FaceEmbeddingProtocolError, "staging file"):
                request.write_atomic(root / "outside.json", staging_root=staging.root)

            staging_path.chmod(0o750)
            try:
                with self.assertRaisesRegex(FaceEmbeddingProtocolError, "owner-only"):
                    FaceEmbeddingStagingRoot(str(staging_path))
            finally:
                staging_path.chmod(0o700)

            repository_root = Path(__file__).resolve().parents[1]
            with self.assertRaises(FaceEmbeddingProtocolError):
                FaceEmbeddingStagingRoot(str(repository_root))

    def test_batch_schema_round_trip_and_atomic_private_output(self) -> None:
        batch = _batch(faces=(_accepted(),))
        restored = FaceEmbeddingBatch.from_json(batch.to_json())
        self.assertEqual(batch, restored)
        self.assertNotIn("source_path", batch.to_json())
        self.assertNotIn("model_path", batch.to_json())
        self.assertNotIn("profile_id", batch.to_json())

        schema = json.loads(
            Path("schemas/face-embedding-batch-v2.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(batch.to_dict()))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            staging_path = root / "private-staging"
            staging_path.mkdir(mode=0o700)
            staging = FaceEmbeddingStagingRoot(str(staging_path))
            output = staging_path / "private-face-batch.json"
            batch.write_atomic(output, staging_root=staging.root)
            self.assertEqual(batch, FaceEmbeddingBatch.read(output, staging_root=staging.root))
            if os.name == "posix":
                self.assertEqual(0o600, output.stat().st_mode & 0o777)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                batch.write_atomic(output, staging_root=staging.root)


if __name__ == "__main__":
    unittest.main()
