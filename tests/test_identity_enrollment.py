from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baby_monitor.identity_domain import DeclaredKind, FaceQualityState
from baby_monitor.identity_enrollment import (
    ENROLLMENT_COHORT_MIN_PAIRWISE_COSINE,
    ENROLLMENT_POLICY_REVISION,
    IdentityEnrollmentError,
    ValidatedEnrollmentEmbedding,
    enroll_identity_from_manifest,
)
from baby_monitor.identity_protocol import (
    FACE_LANDMARK_NAMES,
    FaceEmbeddingBatch,
    FaceEmbeddingMode,
    FaceEmbeddingObservation,
    FaceEmbeddingRequest,
    FaceEmbeddingStagingRoot,
    FaceImageEmbeddingResult,
    FaceImageInput,
    FaceLandmark,
    FaceModelReference,
)
from baby_monitor.identity_service import IdentityAuthorizationError, IdentitySettings
from baby_monitor.identity_store import IdentityEnrollmentSummary


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _authorized_settings() -> IdentitySettings:
    return IdentitySettings(
        enabled=True,
        biometric_enrollment_consent=True,
    )


def _write_manifest(
    parent: Path,
    *,
    count: int,
    duplicate_hash: bool = False,
    duplicate_frame: bool = False,
    extra_manifest_field: bool = False,
) -> tuple[Path, Path, list[dict[str, str]]]:
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent.chmod(0o700)
    manifest_root = parent / "private-manifest"
    manifest_root.mkdir(mode=0o700)
    manifest_root.chmod(0o700)
    image_root = parent / "source-images"
    image_root.mkdir(mode=0o700)
    images: list[dict[str, str]] = []
    first_payload = b"enrollment-image-0"
    for index in range(count):
        payload = (
            first_payload
            if duplicate_hash and index == 1
            else (f"enrollment-image-{index}".encode())
        )
        source = (image_root / f"sample-{index:02d}.jpg").resolve()
        source.write_bytes(payload)
        images.append(
            {
                "frame_ref": "frame_00" if duplicate_frame and index == 1 else f"frame_{index:02d}",
                "source_path": str(source),
                "source_sha256": _sha(payload),
            }
        )
    manifest: dict[str, object] = {
        "version": 1,
        "declared_kind": "baby",
        "images": images,
    }
    if extra_manifest_field:
        manifest["name"] = "forbidden-field"
    manifest_path = (manifest_root / "enrollment.json").resolve()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    return manifest_root.resolve(), manifest_path, images


class _SpyStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def enroll_profile(
        self,
        *,
        declared_kind: DeclaredKind,
        model_fingerprint: str,
        embeddings: tuple[tuple[float, ...], ...],
    ) -> IdentityEnrollmentSummary:
        self.calls.append(
            {
                "declared_kind": declared_kind,
                "model_fingerprint": model_fingerprint,
                "embeddings": embeddings,
            }
        )
        return IdentityEnrollmentSummary(
            profile_id=f"profile_{1:032x}",
            template_ids=tuple(f"template_{index + 1:032x}" for index in range(len(embeddings))),
            declared_kind=declared_kind,
            model_fingerprint=model_fingerprint,
            embedding_dimension=128,
        )


class _BoundExtractor:
    """Test adapter that uses the real request validator and consumed marker."""

    def __init__(
        self,
        parent: Path,
        *,
        fingerprints: tuple[str, ...] | None = None,
        fail_at: int | None = None,
        embeddings: tuple[tuple[float, ...], ...] | None = None,
        extra_rejected_face_at: int | None = None,
    ) -> None:
        model_root = parent / "models"
        model_root.mkdir(mode=0o700)
        detector_path = (model_root / "detector.onnx").resolve()
        embedding_path = (model_root / "embedding.onnx").resolve()
        detector_manifest = (model_root / "detector.json").resolve()
        embedding_manifest = (model_root / "embedding.json").resolve()
        for path, payload in (
            (detector_path, b"detector"),
            (embedding_path, b"embedding"),
            (detector_manifest, b"{}"),
            (embedding_manifest, b"{}"),
        ):
            path.write_bytes(payload)
        self.detector = FaceModelReference(
            role="detector",
            model_path=str(detector_path),
            manifest_path=str(detector_manifest),
            model_id="fixture.detector",
            upstream_revision="1" * 40,
            model_sha256="a" * 64,
        )
        self.embedding = FaceModelReference(
            role="embedding",
            model_path=str(embedding_path),
            manifest_path=str(embedding_manifest),
            model_id="fixture.embedding",
            upstream_revision="2" * 40,
            model_sha256="b" * 64,
        )
        staging_root = parent / "private-ipc"
        staging_root.mkdir(mode=0o700)
        staging_root.chmod(0o700)
        self.replay_guard = FaceEmbeddingStagingRoot(str(staging_root.resolve()))
        self.fingerprints = fingerprints
        self.fail_at = fail_at
        self.embeddings = embeddings
        self.extra_rejected_face_at = extra_rejected_face_at
        self.calls: list[str] = []
        self.proofs: list[ValidatedEnrollmentEmbedding] = []

    def extract_validated_enrollment(self, image: object) -> ValidatedEnrollmentEmbedding:
        self.calls.append(image.frame_ref)
        call_number = len(self.calls)
        if self.fail_at == call_number:
            raise RuntimeError("synthetic adapter failure with a private path")
        fingerprint = (
            self.fingerprints[call_number - 1] if self.fingerprints is not None else "c" * 64
        )
        request = FaceEmbeddingRequest(
            request_id=f"request_{call_number:02d}",
            mode=FaceEmbeddingMode.ENROLLMENT,
            face_adapter="opencv_yunet_sface_v1",
            images=(
                FaceImageInput(
                    frame_ref=image.frame_ref,
                    source_path=image.source_path,
                    source_sha256=image.source_sha256,
                ),
            ),
            detector_model=self.detector,
            embedding_model=self.embedding,
            detector_score_threshold=0.8,
            detector_nms_threshold=0.3,
            detector_top_k=100,
            min_face_size_pixels=40,
            min_eye_distance_pixels=12.0,
        )
        embedding = (
            self.embeddings[call_number - 1]
            if self.embeddings is not None
            else tuple(1.0 if index == 0 else 0.0 for index in range(128))
        )
        landmarks = tuple(
            FaceLandmark(name=name, x=0.5, y=0.5, in_frame=True) for name in FACE_LANDMARK_NAMES
        )
        face = FaceEmbeddingObservation(
            face_id=f"face_{call_number:02d}",
            frame_ref=image.frame_ref,
            bbox_xyxy=(0.1, 0.1, 0.9, 0.9),
            detector_confidence=0.99,
            landmarks=landmarks,
            quality_state=FaceQualityState.ACCEPTED,
            rejection_codes=(),
            embedding=embedding,
        )
        faces: tuple[FaceEmbeddingObservation, ...] = (face,)
        if self.extra_rejected_face_at == call_number:
            rejected_face = FaceEmbeddingObservation(
                face_id=f"rejected_face_{call_number:02d}",
                frame_ref=image.frame_ref,
                bbox_xyxy=(0.2, 0.2, 0.3, 0.3),
                detector_confidence=0.5,
                landmarks=landmarks,
                quality_state=FaceQualityState.REJECTED,
                rejection_codes=("face_too_small",),
                embedding=None,
            )
            faces = (face, rejected_face)
        batch = FaceEmbeddingBatch(
            request_id=request.request_id,
            request_sha256=request.canonical_sha256(),
            mode=FaceEmbeddingMode.ENROLLMENT,
            face_adapter=request.face_adapter,
            detector_model_id=self.detector.model_id,
            detector_model_revision=self.detector.upstream_revision,
            detector_model_sha256=self.detector.model_sha256,
            embedding_model_id=self.embedding.model_id,
            embedding_model_revision=self.embedding.upstream_revision,
            embedding_model_sha256=self.embedding.model_sha256,
            model_fingerprint=fingerprint,
            images=(
                FaceImageEmbeddingResult(
                    frame_ref=image.frame_ref,
                    source_sha256=image.source_sha256,
                    width=640,
                    height=480,
                    faces=faces,
                ),
            ),
        )
        with mock.patch(
            "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
            return_value=fingerprint,
        ):
            proof = ValidatedEnrollmentEmbedding.consume_bound_batch(
                request,
                batch,
                replay_guard=self.replay_guard,
            )
        self.proofs.append(proof)
        return proof


@unittest.skipUnless(os.name == "posix", "private enrollment manifests require POSIX/WSL")
class IdentityEnrollmentCoordinatorTests(unittest.TestCase):
    def test_authorization_failure_happens_before_manifest_media_or_store_access(self) -> None:
        extractor = mock.Mock()
        store_factory = mock.Mock()
        with (
            mock.patch("baby_monitor.identity_enrollment._read_private_manifest") as reader,
            self.assertRaises(IdentityAuthorizationError),
        ):
            enroll_identity_from_manifest(
                settings=IdentitySettings(),
                one_shot_confirmation=True,
                manifest_path=mock.sentinel.manifest_path,
                manifest_root=mock.sentinel.manifest_root,
                extractor=extractor,
                store_factory=store_factory,
            )
        reader.assert_not_called()
        extractor.extract_validated_enrollment.assert_not_called()
        store_factory.assert_not_called()

    def test_manifest_rejects_two_or_thirteen_samples_before_extraction(self) -> None:
        for count in (2, 13):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory).resolve() / f"case-{count}"
                root, manifest, _images = _write_manifest(parent, count=count)
                extractor = mock.Mock()
                store_factory = mock.Mock()
                with self.assertRaisesRegex(IdentityEnrollmentError, "between 3 and 12"):
                    enroll_identity_from_manifest(
                        settings=_authorized_settings(),
                        one_shot_confirmation=True,
                        manifest_path=manifest,
                        manifest_root=root,
                        extractor=extractor,
                        store_factory=store_factory,
                    )
                extractor.extract_validated_enrollment.assert_not_called()
                store_factory.assert_not_called()

    def test_manifest_rejects_duplicate_hash_frame_and_name_field(self) -> None:
        cases = (
            ({"duplicate_hash": True}, "source_sha256 values must be unique"),
            ({"duplicate_frame": True}, "frame_ref values must be unique"),
            ({"extra_manifest_field": True}, "fields do not match"),
        )
        for options, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root, manifest, _images = _write_manifest(
                    Path(directory).resolve(),
                    count=3,
                    **options,
                )
                store_factory = mock.Mock()
                with self.assertRaisesRegex(IdentityEnrollmentError, message):
                    enroll_identity_from_manifest(
                        settings=_authorized_settings(),
                        one_shot_confirmation=True,
                        manifest_path=manifest,
                        manifest_root=root,
                        extractor=mock.Mock(),
                        store_factory=store_factory,
                    )
                store_factory.assert_not_called()

    def test_private_manifest_requires_root_0700_file_0600_and_single_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, _images = _write_manifest(parent, count=3)
            store_factory = mock.Mock()

            root.chmod(0o750)
            try:
                with self.assertRaisesRegex(IdentityEnrollmentError, "owner-only WSL"):
                    enroll_identity_from_manifest(
                        settings=_authorized_settings(),
                        one_shot_confirmation=True,
                        manifest_path=manifest,
                        manifest_root=root,
                        extractor=mock.Mock(),
                        store_factory=store_factory,
                    )
            finally:
                root.chmod(0o700)

            manifest.chmod(0o640)
            try:
                with self.assertRaisesRegex(IdentityEnrollmentError, "owner-only private"):
                    enroll_identity_from_manifest(
                        settings=_authorized_settings(),
                        one_shot_confirmation=True,
                        manifest_path=manifest,
                        manifest_root=root,
                        extractor=mock.Mock(),
                        store_factory=store_factory,
                    )
            finally:
                manifest.chmod(0o600)

            alias = root / "manifest-hardlink.json"
            os.link(manifest, alias)
            with self.assertRaisesRegex(IdentityEnrollmentError, "owner-only private"):
                enroll_identity_from_manifest(
                    settings=_authorized_settings(),
                    one_shot_confirmation=True,
                    manifest_path=manifest,
                    manifest_root=root,
                    extractor=mock.Mock(),
                    store_factory=store_factory,
                )
            alias.unlink()

            symlink = root / "manifest-symlink.json"
            symlink.symlink_to(manifest)
            with self.assertRaisesRegex(IdentityEnrollmentError, "owner-only private"):
                enroll_identity_from_manifest(
                    settings=_authorized_settings(),
                    one_shot_confirmation=True,
                    manifest_path=symlink,
                    manifest_root=root,
                    extractor=mock.Mock(),
                    store_factory=store_factory,
                )
            symlink.unlink()
            store_factory.assert_not_called()

    def test_second_or_third_sample_failure_leaves_store_untouched(self) -> None:
        for fail_at in (2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory).resolve()
                root, manifest, images = _write_manifest(parent, count=4)
                extractor = _BoundExtractor(parent, fail_at=fail_at)
                store = _SpyStore()
                store_factory = mock.Mock(return_value=store)
                with self.assertRaisesRegex(
                    IdentityEnrollmentError,
                    f"extraction failed for sample {fail_at}",
                ):
                    enroll_identity_from_manifest(
                        settings=_authorized_settings(),
                        one_shot_confirmation=True,
                        manifest_path=manifest,
                        manifest_root=root,
                        extractor=extractor,
                        store_factory=store_factory,
                    )
                self.assertEqual(
                    [item["frame_ref"] for item in images[:fail_at]],
                    extractor.calls,
                )
                store_factory.assert_not_called()
                self.assertEqual([], store.calls)

    def test_pipeline_fingerprint_mismatch_leaves_store_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, _images = _write_manifest(parent, count=3)
            extractor = _BoundExtractor(
                parent,
                fingerprints=("c" * 64, "c" * 64, "d" * 64),
            )
            store = _SpyStore()
            store_factory = mock.Mock(return_value=store)
            with self.assertRaisesRegex(IdentityEnrollmentError, "fingerprint changed"):
                enroll_identity_from_manifest(
                    settings=_authorized_settings(),
                    one_shot_confirmation=True,
                    manifest_path=manifest,
                    manifest_root=root,
                    extractor=extractor,
                    store_factory=store_factory,
                )
            store_factory.assert_not_called()
            self.assertEqual([], store.calls)

    def test_source_hash_mismatch_fails_before_extractor_or_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, images = _write_manifest(parent, count=3)
            Path(images[0]["source_path"]).write_bytes(b"changed-after-manifest")
            extractor = mock.Mock()
            store_factory = mock.Mock()
            with self.assertRaisesRegex(IdentityEnrollmentError, "sample 1"):
                enroll_identity_from_manifest(
                    settings=_authorized_settings(),
                    one_shot_confirmation=True,
                    manifest_path=manifest,
                    manifest_root=root,
                    extractor=extractor,
                    store_factory=store_factory,
                )
            extractor.extract_validated_enrollment.assert_not_called()
            store_factory.assert_not_called()

    def test_one_accepted_and_one_rejected_face_fails_before_store_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, _images = _write_manifest(parent, count=3)
            extractor = _BoundExtractor(parent, extra_rejected_face_at=1)
            store_factory = mock.Mock()

            with self.assertRaisesRegex(IdentityEnrollmentError, "sample 1"):
                enroll_identity_from_manifest(
                    settings=_authorized_settings(),
                    one_shot_confirmation=True,
                    manifest_path=manifest,
                    manifest_root=root,
                    extractor=extractor,
                    store_factory=store_factory,
                )

            self.assertEqual(["frame_00"], extractor.calls)
            self.assertEqual(
                1,
                len(tuple(Path(extractor.replay_guard.root).glob(".consumed-*"))),
            )
            store_factory.assert_not_called()

    def test_orthogonal_cohort_outlier_fails_before_store_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, _images = _write_manifest(parent, count=3)
            same = tuple(1.0 if index == 0 else 0.0 for index in range(128))
            orthogonal = tuple(1.0 if index == 1 else 0.0 for index in range(128))
            extractor = _BoundExtractor(
                parent,
                embeddings=(same, same, orthogonal),
            )
            store = _SpyStore()
            store_factory = mock.Mock(return_value=store)

            with self.assertRaisesRegex(IdentityEnrollmentError, "pairwise outlier veto"):
                enroll_identity_from_manifest(
                    settings=_authorized_settings(),
                    one_shot_confirmation=True,
                    manifest_path=manifest,
                    manifest_root=root,
                    extractor=extractor,
                    store_factory=store_factory,
                )

            self.assertEqual(["frame_00", "frame_01", "frame_02"], extractor.calls)
            store_factory.assert_not_called()
            self.assertEqual([], store.calls)

    def test_cohort_cosine_boundary_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, _images = _write_manifest(parent, count=3)
            minimum = ENROLLMENT_COHORT_MIN_PAIRWISE_COSINE
            same = tuple(1.0 if index == 0 else 0.0 for index in range(128))
            boundary = (
                minimum,
                math.sqrt(1.0 - minimum * minimum),
                *(0.0 for _index in range(126)),
            )
            self.assertEqual(
                minimum,
                math.fsum(left * right for left, right in zip(same, boundary, strict=True)),
            )
            extractor = _BoundExtractor(
                parent,
                embeddings=(same, boundary, same),
            )
            store = _SpyStore()
            store_factory = mock.Mock(return_value=store)

            result = enroll_identity_from_manifest(
                settings=_authorized_settings(),
                one_shot_confirmation=True,
                manifest_path=manifest,
                manifest_root=root,
                extractor=extractor,
                store_factory=store_factory,
            )

            store_factory.assert_called_once_with()
            self.assertEqual(1, len(store.calls))
            public = result.to_public_dict()
            self.assertTrue(public["cohort_consistency_checked"])
            self.assertEqual(
                ENROLLMENT_POLICY_REVISION,
                public["enrollment_policy_revision"],
            )
            self.assertNotIn("pairwise_cosine", public)
            self.assertNotIn("cohort_scores", public)

    def test_success_validates_in_order_then_performs_one_path_free_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root, manifest, images = _write_manifest(parent, count=3)
            extractor = _BoundExtractor(parent)
            store = _SpyStore()
            store_factory = mock.Mock(return_value=store)
            result = enroll_identity_from_manifest(
                settings=_authorized_settings(),
                one_shot_confirmation=True,
                manifest_path=manifest,
                manifest_root=root,
                extractor=extractor,
                store_factory=store_factory,
            )

            store_factory.assert_called_once_with()
            self.assertEqual(
                [item["frame_ref"] for item in images],
                extractor.calls,
            )
            self.assertEqual(1, len(store.calls))
            stored_embeddings = store.calls[0]["embeddings"]
            self.assertEqual(3, len(stored_embeddings))
            self.assertEqual((stored_embeddings[0],) * 3, stored_embeddings)
            self.assertTrue(all(len(item) == 128 for item in stored_embeddings))
            self.assertTrue(all(math.isclose(math.hypot(*item), 1.0) for item in stored_embeddings))
            self.assertEqual(3, len(tuple(Path(extractor.replay_guard.root).glob(".consumed-*"))))

            public = result.to_public_dict()
            self.assertFalse(public["source_images_persisted"])
            self.assertFalse(public["network_used"])
            self.assertFalse(public["identity_inference"])
            self.assertFalse(public["paths_disclosed"])
            self.assertTrue(public["atomic_store_write"])
            self.assertTrue(public["cohort_consistency_checked"])
            self.assertEqual(ENROLLMENT_POLICY_REVISION, public["enrollment_policy_revision"])
            serialized = json.dumps(public, sort_keys=True)
            serialized += repr(result)
            serialized += repr(extractor.proofs[0])
            self.assertNotIn(str(parent), serialized)
            self.assertNotIn("source_path", serialized)
            self.assertNotIn(images[0]["source_sha256"], serialized)
            self.assertNotIn("(1.0, 0.0", serialized)


if __name__ == "__main__":
    unittest.main()
