from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.identity_domain import FaceQualityState
from baby_monitor.identity_model import FaceModelManifest
from baby_monitor.identity_protocol import (
    FACE_QUALITY_POLICY,
    FaceEmbeddingBatch,
    FaceEmbeddingMode,
    FaceEmbeddingProtocolError,
    FaceEmbeddingRequest,
    FaceEmbeddingStagingRoot,
    FaceImageInput,
    FaceModelReference,
    canonical_face_pipeline_fingerprint,
)
from baby_monitor.local_runners import opencv_face as opencv_face_runner
from baby_monitor.local_runners.opencv_face import main, run_request


def _manifest_payload(content: bytes, *, role: str, filename: str) -> dict[str, object]:
    return {
        "version": 1,
        "backend": "opencv_dnn",
        "model_format": "ONNX",
        "role": role,
        "model_id": f"fixture.face.{role}",
        "upstream_revision": ("1" if role == "detector" else "2") * 40,
        "filename": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "source_url": f"https://example.invalid/models/{filename}",
        "license_spdx": "LicenseRef-Fixture-Only",
        "license_url": "https://example.invalid/licenses/fixture",
        "input_width": 320 if role == "detector" else 112,
        "input_height": 320 if role == "detector" else 112,
        "input_color_space": "BGR",
        "landmark_count": 5,
        "embedding_dimension": None if role == "detector" else 128,
        "preprocessing_revision": f"fixture_{role}_bgr_v1",
        "unsafe_pickle": False,
        "automatic_download": False,
        "identity_decision_authority": False,
    }


def _write_model(root: Path, *, role: str) -> tuple[Path, Path, FaceModelManifest]:
    filename = f"fixture-{role}.onnx"
    content = f"synthetic-{role}-onnx".encode()
    model = root / filename
    model.write_bytes(content)
    manifest_path = root / f"fixture-{role}.manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_payload(content, role=role, filename=filename)),
        encoding="utf-8",
    )
    return model, manifest_path, FaceModelManifest.read(manifest_path)


def _request(
    root: Path,
    *,
    mode: FaceEmbeddingMode = FaceEmbeddingMode.ASSESSMENT,
) -> FaceEmbeddingRequest:
    detector_path, detector_manifest_path, detector = _write_model(root, role="detector")
    embedding_path, embedding_manifest_path, embedding = _write_model(root, role="embedding")
    source_content = b"synthetic-public-image"
    source = root / "public-synthetic-image.jpg"
    source.write_bytes(source_content)
    return FaceEmbeddingRequest(
        request_id="request_01",
        mode=mode,
        face_adapter="opencv_yunet_sface_v1",
        images=(
            FaceImageInput(
                frame_ref="frame_01",
                source_path=str(source.resolve()),
                source_sha256=hashlib.sha256(source_content).hexdigest(),
            ),
        ),
        detector_model=FaceModelReference(
            role="detector",
            model_path=str(detector_path.resolve()),
            manifest_path=str(detector_manifest_path.resolve()),
            model_id=detector.model_id,
            upstream_revision=detector.upstream_revision,
            model_sha256=detector.sha256,
        ),
        embedding_model=FaceModelReference(
            role="embedding",
            model_path=str(embedding_path.resolve()),
            manifest_path=str(embedding_manifest_path.resolve()),
            model_id=embedding.model_id,
            upstream_revision=embedding.upstream_revision,
            model_sha256=embedding.sha256,
        ),
        detector_score_threshold=0.8,
        detector_nms_threshold=0.3,
        detector_top_k=100,
        min_face_size_pixels=40,
        min_eye_distance_pixels=12.0,
    )


def _allow_fixture_models(request: FaceEmbeddingRequest):
    digests = frozenset(
        {
            FaceModelManifest.read(request.detector_model.manifest_path).canonical_sha256(),
            FaceModelManifest.read(request.embedding_model.manifest_path).canonical_sha256(),
        }
    )
    return patch("baby_monitor.identity_model._REVIEWED_FACE_MANIFEST_SHA256", digests)


def _accepted_row(x: float = 100.0) -> list[float]:
    return [
        x,
        80.0,
        220.0,
        260.0,
        x + 50.0,
        150.0,
        x + 150.0,
        150.0,
        x + 100.0,
        210.0,
        x + 65.0,
        280.0,
        x + 135.0,
        280.0,
        0.95,
    ]


def _rejected_row() -> list[float]:
    return [
        10.0,
        10.0,
        20.0,
        20.0,
        15.0,
        15.0,
        17.0,
        15.0,
        16.0,
        18.0,
        15.0,
        25.0,
        17.0,
        25.0,
        0.9,
    ]


class _FakeImage:
    shape = (480, 640, 3)


class _FakeAlignedCrop:
    shape = (112, 112, 3)

    def __init__(self, *, mean_luma: float, laplacian_variance: float):
        self.mean_luma = mean_luma
        self.laplacian_variance = laplacian_variance


class _FakeGray:
    shape = (112, 112)

    def __init__(self, crop: _FakeAlignedCrop):
        self.mean_luma = crop.mean_luma
        self.laplacian_variance = crop.laplacian_variance


class _FakeLaplacian:
    shape = (112, 112)

    def __init__(self, variance: float):
        self.variance = variance


class _FakeEncoded:
    def __init__(self, payload: bytes):
        self.payload = payload


class _FakeDetector:
    def __init__(self, rows: list[list[float]]):
        self.rows = rows
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, value: tuple[int, int]) -> None:
        self.input_size = value

    def detect(self, _image: object) -> tuple[int, list[list[float]]]:
        return 1, self.rows


class _FakeRecognizer:
    def __init__(
        self,
        state: dict[str, object],
        *,
        mean_luma: float,
        laplacian_variance: float,
    ):
        self.state = state
        self.mean_luma = mean_luma
        self.laplacian_variance = laplacian_variance

    def alignCrop(self, _image: object, _row: object) -> object:
        self.state["align_count"] = int(self.state.get("align_count", 0)) + 1
        return _FakeAlignedCrop(
            mean_luma=self.mean_luma,
            laplacian_variance=self.laplacian_variance,
        )

    def feature(self, _aligned: object) -> list[float]:
        self.state["feature_count"] = int(self.state.get("feature_count", 0)) + 1
        return [3.0] + [0.0] * 127


def _fake_runtime(
    rows: list[list[float]],
    *,
    mean_luma: float = 128.0,
    laplacian_variance: float = 200.0,
    on_frombuffer: object | None = None,
    fail_recognizer: bool = False,
) -> tuple[object, object, dict[str, object]]:
    state: dict[str, object] = {
        "detector_created": False,
        "recognizer_created": False,
        "align_count": 0,
        "feature_count": 0,
        "imread_count": 0,
        "decode_count": 0,
    }

    def record_model(role: str, value: object) -> None:
        path = Path(str(value))
        state[f"{role}_model_path"] = str(path)
        state[f"{role}_model_bytes"] = path.read_bytes()
        state[f"{role}_model_mode"] = path.stat().st_mode & 0o777
        state[f"{role}_directory_mode"] = path.parent.stat().st_mode & 0o777

    def create_detector(*args: object) -> _FakeDetector:
        if len(args) != 6:
            raise AssertionError("unexpected FaceDetectorYN.create arguments")
        record_model("detector", args[0])
        state["detector_created"] = True
        detector = _FakeDetector(rows)
        state["detector"] = detector
        return detector

    def create_recognizer(*args: object) -> _FakeRecognizer:
        if len(args) != 2:
            raise AssertionError("unexpected FaceRecognizerSF.create arguments")
        record_model("embedding", args[0])
        if fail_recognizer:
            raise RuntimeError("fixture recognizer load failed")
        state["recognizer_created"] = True
        return _FakeRecognizer(
            state,
            mean_luma=mean_luma,
            laplacian_variance=laplacian_variance,
        )

    def forbidden_imread(_path: object, _mode: object) -> object:
        state["imread_count"] = int(state["imread_count"]) + 1
        raise AssertionError("imread must not be used")

    def imdecode(encoded: _FakeEncoded, _mode: object) -> _FakeImage:
        state["decode_count"] = int(state["decode_count"]) + 1
        state["decoded_bytes"] = encoded.payload
        return _FakeImage()

    uint8_token = object()

    def frombuffer(payload: bytes, *, dtype: object) -> _FakeEncoded:
        if dtype is not uint8_token:
            raise AssertionError("unexpected NumPy dtype")
        if callable(on_frombuffer):
            on_frombuffer()
        state["buffer_bytes"] = payload
        return _FakeEncoded(payload)

    module = SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=create_detector),
        FaceRecognizerSF=SimpleNamespace(create=create_recognizer),
        IMREAD_COLOR=1,
        COLOR_BGR2GRAY=6,
        CV_64F=6,
        imread=forbidden_imread,
        imdecode=imdecode,
        cvtColor=lambda crop, _code: _FakeGray(crop),
        Laplacian=lambda gray, _depth, *, ksize: (
            _FakeLaplacian(gray.laplacian_variance)
            if ksize == 3
            else (_ for _ in ()).throw(AssertionError("unexpected Laplacian kernel"))
        ),
    )
    numpy_module = SimpleNamespace(
        uint8=uint8_token,
        frombuffer=frombuffer,
        asarray=lambda value: value,
        mean=lambda value: value.mean_luma,
        var=lambda value: value.variance,
    )
    return module, numpy_module, state


class OpenCvFaceRunnerTests(unittest.TestCase):
    def test_assessment_emits_visible_facts_and_normalized_128d_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory).resolve())
            fake_cv2, fake_numpy, state = _fake_runtime([_accepted_row(), _rejected_row()])
            with (
                _allow_fixture_models(request),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
            ):
                result = run_request(request)
            self.assertEqual(request.canonical_sha256(), result.request_sha256)

            self.assertEqual(2, result.face_count)
            self.assertEqual(1, result.accepted_face_count)
            accepted, rejected = result.images[0].faces
            self.assertEqual(FaceQualityState.ACCEPTED, accepted.quality_state)
            self.assertEqual(128, len(accepted.embedding or ()))
            self.assertAlmostEqual(
                1.0,
                sum(value * value for value in accepted.embedding or ()),
            )
            self.assertEqual(5, len(accepted.landmarks))
            self.assertEqual(FaceQualityState.REJECTED, rejected.quality_state)
            self.assertIsNone(rejected.embedding)
            self.assertIn("face_too_small", rejected.rejection_codes)
            self.assertIn("eyes_too_close", rejected.rejection_codes)
            self.assertEqual(1, state["feature_count"])
            self.assertEqual((640, 480), state["detector"].input_size)  # type: ignore[union-attr]
            self.assertFalse(result.identity_inference)
            self.assertFalse(result.paths_disclosed)
            self.assertFalse(result.network_used)

    def test_aligned_crop_quality_rejections_never_call_feature(self) -> None:
        cases = (
            (
                "face_too_blurry",
                128.0,
                FACE_QUALITY_POLICY.min_laplacian_variance - 0.1,
            ),
            (
                "underexposed",
                FACE_QUALITY_POLICY.min_mean_luma - 0.1,
                FACE_QUALITY_POLICY.min_laplacian_variance + 1.0,
            ),
            (
                "overexposed",
                FACE_QUALITY_POLICY.max_mean_luma + 0.1,
                FACE_QUALITY_POLICY.min_laplacian_variance + 1.0,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory).resolve())
            for expected, mean_luma, laplacian_variance in cases:
                with self.subTest(expected=expected):
                    fake_cv2, fake_numpy, state = _fake_runtime(
                        [_accepted_row()],
                        mean_luma=mean_luma,
                        laplacian_variance=laplacian_variance,
                    )
                    with (
                        _allow_fixture_models(request),
                        patch.dict(
                            sys.modules,
                            {"cv2": fake_cv2, "numpy": fake_numpy},
                        ),
                    ):
                        result = run_request(request)
                    face = result.images[0].faces[0]
                    self.assertEqual(FaceQualityState.REJECTED, face.quality_state)
                    self.assertEqual((expected,), face.rejection_codes)
                    self.assertIsNone(face.embedding)
                    self.assertEqual(1, state["align_count"])
                    self.assertEqual(0, state["feature_count"])

    def test_pose_and_landmark_geometry_rejections_never_align_or_embed(self) -> None:
        roll = _accepted_row()
        roll[7] = roll[5] + 50.0
        yaw = _accepted_row()
        yaw[8] = yaw[4] + 95.0
        geometry = _accepted_row()
        geometry[10] = geometry[4] + 45.0
        geometry[12] = geometry[4] + 55.0
        cases = (
            ("excessive_roll", roll),
            ("excessive_yaw", yaw),
            ("invalid_five_point_geometry", geometry),
        )
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory).resolve())
            for expected, row in cases:
                with self.subTest(expected=expected):
                    fake_cv2, fake_numpy, state = _fake_runtime([row])
                    with (
                        _allow_fixture_models(request),
                        patch.dict(
                            sys.modules,
                            {"cv2": fake_cv2, "numpy": fake_numpy},
                        ),
                    ):
                        result = run_request(request)
                    face = result.images[0].faces[0]
                    self.assertEqual(FaceQualityState.REJECTED, face.quality_state)
                    self.assertIn(expected, face.rejection_codes)
                    self.assertIsNone(face.embedding)
                    self.assertEqual(0, state["align_count"])
                    self.assertEqual(0, state["feature_count"])

    def test_quality_policy_parameters_change_pipeline_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory).resolve())
            with _allow_fixture_models(request):
                baseline = canonical_face_pipeline_fingerprint(request)
                changed_policy = replace(
                    FACE_QUALITY_POLICY,
                    min_laplacian_variance=(FACE_QUALITY_POLICY.min_laplacian_variance + 1.0),
                )
                with patch(
                    "baby_monitor.identity_protocol.FACE_QUALITY_POLICY",
                    changed_policy,
                ):
                    changed = canonical_face_pipeline_fingerprint(request)
            self.assertNotEqual(baseline, changed)

    def test_assessment_accepts_zero_faces_and_enrollment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            assessment = _request(root)
            fake_cv2, fake_numpy, _ = _fake_runtime([])
            with (
                _allow_fixture_models(assessment),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
            ):
                result = run_request(assessment)
            self.assertEqual(0, result.face_count)

            enrollment = replace(assessment, mode=FaceEmbeddingMode.ENROLLMENT)
            fake_cv2, fake_numpy, _ = _fake_runtime([])
            with (
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
                _allow_fixture_models(enrollment),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "exactly one"),
            ):
                run_request(enrollment)

            fake_cv2, fake_numpy, state = _fake_runtime(
                [_accepted_row()],
                laplacian_variance=FACE_QUALITY_POLICY.min_laplacian_variance - 0.1,
            )
            with (
                patch.dict(
                    sys.modules,
                    {"cv2": fake_cv2, "numpy": fake_numpy},
                ),
                _allow_fixture_models(enrollment),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "exactly one"),
            ):
                run_request(enrollment)
            self.assertEqual(1, state["align_count"])
            self.assertEqual(0, state["feature_count"])

            fake_cv2, fake_numpy, _ = _fake_runtime([_accepted_row(), _accepted_row(340.0)])
            with (
                _allow_fixture_models(enrollment),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "exactly one"),
            ):
                run_request(enrollment)

    def test_image_path_exchange_after_read_does_not_change_decoded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root)
            source = Path(request.images[0].source_path)
            admitted_bytes = source.read_bytes()

            def exchange_path_contents() -> None:
                source.write_bytes(b"replacement-after-secure-read")

            fake_cv2, fake_numpy, state = _fake_runtime(
                [],
                on_frombuffer=exchange_path_contents,
            )
            with (
                _allow_fixture_models(request),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
            ):
                result = run_request(request)

            self.assertEqual(0, result.face_count)
            self.assertEqual(admitted_bytes, state["buffer_bytes"])
            self.assertEqual(admitted_bytes, state["decoded_bytes"])
            self.assertEqual(1, state["decode_count"])
            self.assertEqual(0, state["imread_count"])
            self.assertEqual(b"replacement-after-secure-read", source.read_bytes())

    def test_models_use_exact_private_copies_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root)
            detector_source = Path(request.detector_model.model_path)
            embedding_source = Path(request.embedding_model.model_path)
            detector_bytes = detector_source.read_bytes()
            embedding_bytes = embedding_source.read_bytes()
            real_admit_images = opencv_face_runner._admit_images

            def admit_images_then_exchange_models(
                images: tuple[FaceImageInput, ...],
            ) -> None:
                real_admit_images(images)
                detector_source.write_bytes(b"replaced-detector-after-admission")
                embedding_source.write_bytes(b"replaced-embedding-after-admission")

            fake_cv2, fake_numpy, state = _fake_runtime([])
            with (
                _allow_fixture_models(request),
                patch.object(
                    opencv_face_runner,
                    "_admit_images",
                    side_effect=admit_images_then_exchange_models,
                ),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
            ):
                run_request(request)

            self.assertEqual(detector_bytes, state["detector_model_bytes"])
            self.assertEqual(embedding_bytes, state["embedding_model_bytes"])
            detector_path = Path(str(state["detector_model_path"]))
            embedding_path = Path(str(state["embedding_model_path"]))
            self.assertNotEqual(detector_source, detector_path)
            self.assertNotEqual(embedding_source, embedding_path)
            if os.name == "posix":
                self.assertEqual(0o600, state["detector_model_mode"])
                self.assertEqual(0o600, state["embedding_model_mode"])
                self.assertEqual(0o700, state["detector_directory_mode"])
                self.assertEqual(0o700, state["embedding_directory_mode"])
            self.assertFalse(detector_path.exists())
            self.assertFalse(embedding_path.exists())
            self.assertFalse(detector_path.parent.exists())

    def test_private_model_workspace_is_cleaned_after_load_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory).resolve())
            fake_cv2, fake_numpy, state = _fake_runtime([], fail_recognizer=True)
            with (
                _allow_fixture_models(request),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "could not be loaded"),
            ):
                run_request(request)

            detector_path = Path(str(state["detector_model_path"]))
            embedding_path = Path(str(state["embedding_model_path"]))
            self.assertFalse(detector_path.exists())
            self.assertFalse(embedding_path.exists())
            self.assertFalse(detector_path.parent.exists())

    def test_model_and_image_hashes_are_verified_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root)
            fake_cv2, fake_numpy, state = _fake_runtime([_accepted_row()])
            mismatched_model = replace(request.detector_model, model_sha256="0" * 64)
            with (
                _allow_fixture_models(request),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "reviewed"),
            ):
                run_request(replace(request, detector_model=mismatched_model))
            self.assertFalse(state["detector_created"])
            self.assertFalse(state["recognizer_created"])

            fake_cv2, fake_numpy, state = _fake_runtime([_accepted_row()])
            bad_image = replace(request.images[0], source_sha256="f" * 64)
            with (
                _allow_fixture_models(request),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
                self.assertRaisesRegex(FaceEmbeddingProtocolError, "hash"),
            ):
                run_request(replace(request, images=(bad_image,)))
            self.assertFalse(state["detector_created"])
            self.assertFalse(state["recognizer_created"])

    def test_main_writes_atomic_private_file_and_stdout_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = _request(root)
            staging_path = root / "private-staging"
            staging_path.mkdir(mode=0o700)
            staging = FaceEmbeddingStagingRoot(str(staging_path))
            request_path = staging_path / "face-request.json"
            request.write_atomic(request_path, staging_root=staging.root)
            output_path = staging_path / "private-face-output.json"
            fake_cv2, fake_numpy, _ = _fake_runtime([_accepted_row()])
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                _allow_fixture_models(request),
                patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main([staging.root, str(request_path), str(output_path)])

            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            summary = json.loads(stdout.getvalue())
            self.assertEqual(1, summary["accepted_face_count"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertNotIn("embedding", stdout.getvalue())
            result = FaceEmbeddingBatch.read(output_path, staging_root=staging.root)
            with _allow_fixture_models(request):
                result.validate_against_request(request, replay_guard=staging)
                with self.assertRaisesRegex(FaceEmbeddingProtocolError, "replay"):
                    result.validate_against_request(request, replay_guard=staging)
            self.assertEqual(128, len(result.images[0].faces[0].embedding or ()))
            if os.name == "posix":
                self.assertEqual(0o600, output_path.stat().st_mode & 0o777)


if __name__ == "__main__":
    unittest.main()
