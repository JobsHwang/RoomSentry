from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from baby_monitor.adapters.identity import (
    MAX_FACE_COMMAND_CAPTURE_BYTES,
    CommandFaceEmbeddingAdapter,
    FaceCommandAdapterError,
    FaceEnrollmentCommandConfig,
)
from baby_monitor.identity_domain import FaceQualityState
from baby_monitor.identity_enrollment import EnrollmentImage, ValidatedEnrollmentEmbedding
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


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _request(root: Path, *, seed: str = "request-one") -> FaceEmbeddingRequest:
    image = root / "public-synthetic.jpg"
    detector = root / "detector.onnx"
    detector_manifest = root / "detector.json"
    embedding = root / "embedding.onnx"
    embedding_manifest = root / "embedding.json"
    image.write_bytes(b"synthetic-public-image")
    detector.write_bytes(b"synthetic-detector")
    detector_manifest.write_bytes(b"synthetic-detector-manifest")
    embedding.write_bytes(b"synthetic-embedding")
    embedding_manifest.write_bytes(b"synthetic-embedding-manifest")
    return FaceEmbeddingRequest(
        request_id="request_01",
        request_nonce=_sha(seed.encode()),
        mode=FaceEmbeddingMode.ASSESSMENT,
        face_adapter="opencv_yunet_sface_v1",
        images=(
            FaceImageInput(
                frame_ref="frame_01",
                source_path=str(image),
                source_sha256=_sha(image.read_bytes()),
            ),
        ),
        detector_model=FaceModelReference(
            role="detector",
            model_path=str(detector),
            manifest_path=str(detector_manifest),
            model_id="fixture.detector",
            upstream_revision="1" * 40,
            model_sha256="a" * 64,
        ),
        embedding_model=FaceModelReference(
            role="embedding",
            model_path=str(embedding),
            manifest_path=str(embedding_manifest),
            model_id="fixture.embedding",
            upstream_revision="2" * 40,
            model_sha256="b" * 64,
        ),
        detector_score_threshold=0.8,
        detector_nms_threshold=0.3,
        detector_top_k=100,
        min_face_size_pixels=40,
        min_eye_distance_pixels=12.0,
    )


def _batch(request: FaceEmbeddingRequest) -> FaceEmbeddingBatch:
    return FaceEmbeddingBatch(
        request_id=request.request_id,
        request_sha256=request.canonical_sha256(),
        mode=request.mode,
        face_adapter=request.face_adapter,
        detector_model_id=request.detector_model.model_id,
        detector_model_revision=request.detector_model.upstream_revision,
        detector_model_sha256=request.detector_model.model_sha256,
        embedding_model_id=request.embedding_model.model_id,
        embedding_model_revision=request.embedding_model.upstream_revision,
        embedding_model_sha256=request.embedding_model.model_sha256,
        model_fingerprint="c" * 64,
        images=(
            FaceImageEmbeddingResult(
                frame_ref=request.images[0].frame_ref,
                source_sha256=request.images[0].source_sha256,
                width=640,
                height=480,
                faces=(),
            ),
        ),
    )


def _enrollment_batch(request: FaceEmbeddingRequest) -> FaceEmbeddingBatch:
    landmarks = tuple(
        FaceLandmark(name=name, x=0.5, y=0.5, in_frame=True) for name in FACE_LANDMARK_NAMES
    )
    face = FaceEmbeddingObservation(
        face_id="face_01",
        frame_ref=request.images[0].frame_ref,
        bbox_xyxy=(0.1, 0.1, 0.9, 0.9),
        detector_confidence=0.99,
        landmarks=landmarks,
        quality_state=FaceQualityState.ACCEPTED,
        rejection_codes=(),
        embedding=(1.0,) + (0.0,) * 127,
    )
    return FaceEmbeddingBatch(
        request_id=request.request_id,
        request_sha256=request.canonical_sha256(),
        mode=FaceEmbeddingMode.ENROLLMENT,
        face_adapter=request.face_adapter,
        detector_model_id=request.detector_model.model_id,
        detector_model_revision=request.detector_model.upstream_revision,
        detector_model_sha256=request.detector_model.model_sha256,
        embedding_model_id=request.embedding_model.model_id,
        embedding_model_revision=request.embedding_model.upstream_revision,
        embedding_model_sha256=request.embedding_model.model_sha256,
        model_fingerprint="c" * 64,
        images=(
            FaceImageEmbeddingResult(
                frame_ref=request.images[0].frame_ref,
                source_sha256=request.images[0].source_sha256,
                width=640,
                height=480,
                faces=(face,),
            ),
        ),
    )


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b'{"status":"ok"}\n',
        stderr: bytes = b"",
        returncode: int | None = 0,
    ):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.pid = 999_999
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise subprocess.TimeoutExpired("private-face-runner", 2.0)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _terminate_fake(process: _FakeProcess) -> None:
    process.kill()


def _factory(
    request: FaceEmbeddingRequest,
    batch: FaceEmbeddingBatch,
    *,
    stdout: bytes = b'{"status":"ok"}\n',
    stderr: bytes = b"",
    returncode: int | None = 0,
    write_result: bool = True,
    write_extra: bool = False,
    preconsume: bool = False,
    leak_dynamic_path: bool = False,
    calls: list[tuple[list[str], dict[str, Any]]] | None = None,
):
    def launch(command: list[str], **kwargs: Any) -> _FakeProcess:
        if calls is not None:
            calls.append((list(command), dict(kwargs)))
        staging = FaceEmbeddingStagingRoot(command[-3])
        if write_result:
            staging.write_batch(batch, Path(command[-1]))
        if write_extra:
            extra = Path(staging.root) / "unexpected-output.json"
            extra.write_bytes(b"unexpected")
            extra.chmod(0o600)
        if preconsume:
            staging.consume_request(request.canonical_sha256())
        actual_stderr = command[-2].encode() if leak_dynamic_path else stderr
        return _FakeProcess(
            stdout=stdout,
            stderr=actual_stderr,
            returncode=returncode,
        )

    return launch


class CommandFaceEmbeddingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        if os.name != "posix":
            self.skipTest("strict face staging is intentionally POSIX/WSL-only")

    @staticmethod
    def _layout(root: Path) -> tuple[Path, Path, Path]:
        executable = root / "fixed-face-runner"
        executable.write_bytes(b"not executed; process factory is injected")
        executable.chmod(0o700)
        parent = root / "private-face-command"
        parent.mkdir(mode=0o700)
        sentinel = parent / "parent-sentinel.txt"
        sentinel.write_bytes(b"preserve")
        sentinel.chmod(0o600)
        return executable, parent, sentinel

    def test_enrollment_extracts_one_consumed_path_free_proof_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, sentinel = self._layout(root)
            base_request = _request(root)
            source = base_request.images[0]
            image = EnrollmentImage(
                frame_ref=source.frame_ref,
                source_path=source.source_path,
                source_sha256=source.source_sha256,
            )
            staged_requests: list[FaceEmbeddingRequest] = []
            marker_counts: list[int] = []

            def launch(command: list[str], **_kwargs: Any) -> _FakeProcess:
                staging = FaceEmbeddingStagingRoot(command[-3])
                request = staging.read_request(Path(command[-2]))
                staged_requests.append(request)
                staging.write_batch(_enrollment_batch(request), Path(command[-1]))
                return _FakeProcess()

            enrollment = FaceEnrollmentCommandConfig(
                detector_model=base_request.detector_model,
                embedding_model=base_request.embedding_model,
            )
            adapter = CommandFaceEmbeddingAdapter(
                command=(str(executable),),
                staging_parent=parent,
                enrollment=enrollment,
                _process_factory=launch,
                _process_terminator=_terminate_fake,
            )
            original_consume = ValidatedEnrollmentEmbedding.consume_bound_batch

            def consume_once(
                request: FaceEmbeddingRequest,
                batch: FaceEmbeddingBatch,
                *,
                replay_guard: FaceEmbeddingStagingRoot,
            ) -> ValidatedEnrollmentEmbedding:
                proof = original_consume(
                    request,
                    batch,
                    replay_guard=replay_guard,
                )
                marker_counts.append(len(tuple(Path(replay_guard.root).glob(".consumed-*.json"))))
                return proof

            with (
                patch(
                    "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
                    return_value="c" * 64,
                ),
                patch.object(
                    ValidatedEnrollmentEmbedding,
                    "consume_bound_batch",
                    side_effect=consume_once,
                ) as consumed,
            ):
                proof = adapter.extract_validated_enrollment(image)

            self.assertEqual(1, consumed.call_count)
            self.assertEqual([1], marker_counts)
            self.assertEqual(1, len(staged_requests))
            self.assertIs(staged_requests[0].mode, FaceEmbeddingMode.ENROLLMENT)
            self.assertEqual(source.frame_ref, proof.frame_ref)
            self.assertEqual(source.source_sha256, proof.source_sha256)
            self.assertEqual(128, proof.embedding_dimension)
            self.assertTrue(proof.request_binding_validated)
            self.assertEqual({sentinel}, set(parent.iterdir()))
            self.assertNotIn(str(root), repr(proof))
            self.assertNotIn("1.0", repr(proof))

            unconfigured = CommandFaceEmbeddingAdapter(
                command=(str(executable),),
                staging_parent=parent,
                _process_factory=lambda *_args, **_kwargs: self.fail("process started"),
                _process_terminator=_terminate_fake,
            )
            with self.assertRaises(FaceCommandAdapterError) as caught:
                unconfigured.extract_validated_enrollment(image)
            self.assertEqual("invalid_configuration", caught.exception.code)
            self.assertEqual({sentinel}, set(parent.iterdir()))

    def test_success_uses_fixed_argv_minimal_env_validates_and_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, sentinel = self._layout(root)
            request = _request(root)
            batch = _batch(request)
            calls: list[tuple[list[str], dict[str, Any]]] = []
            adapter = CommandFaceEmbeddingAdapter(
                command=(str(executable), "--offline"),
                staging_parent=parent,
                _process_factory=_factory(request, batch, calls=calls),
                _process_terminator=_terminate_fake,
            )
            with (
                patch(
                    "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
                    return_value="c" * 64,
                ),
                patch.dict(
                    os.environ,
                    {"LD_LIBRARY_PATH": "/tmp/unreviewed-library-injection"},
                ),
            ):
                result = adapter.embed(request)

            self.assertEqual(batch, result)
            self.assertEqual({sentinel}, set(parent.iterdir()))
            self.assertEqual(1, len(calls))
            command, options = calls[0]
            self.assertEqual(str(executable.resolve()), command[0])
            self.assertEqual("--offline", command[1])
            self.assertEqual(5, len(command))
            self.assertEqual(command[-3], options["cwd"])
            self.assertFalse(options["shell"])
            self.assertTrue(options["close_fds"])
            self.assertTrue(options["start_new_session"])
            self.assertEqual(0o077, options["umask"])
            self.assertEqual("1", options["env"]["TRANSFORMERS_OFFLINE"])
            self.assertNotIn("HTTP_PROXY", options["env"])
            self.assertNotIn("HOME", options["env"])
            self.assertNotIn("LD_LIBRARY_PATH", options["env"])
            self.assertIn("IPAddressDeny=any", CommandFaceEmbeddingAdapter.__doc__ or "")
            self.assertIn("not an OS network", CommandFaceEmbeddingAdapter.__doc__ or "")

    def test_same_request_is_one_shot_even_after_private_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, sentinel = self._layout(root)
            request = _request(root)
            batch = _batch(request)
            calls: list[tuple[list[str], dict[str, Any]]] = []
            adapter = CommandFaceEmbeddingAdapter(
                command=(str(executable),),
                staging_parent=parent,
                _process_factory=_factory(request, batch, calls=calls),
                _process_terminator=_terminate_fake,
            )
            with patch(
                "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
                return_value="c" * 64,
            ):
                adapter.embed(request)
            with self.assertRaises(FaceCommandAdapterError) as caught:
                adapter.embed(request)

            self.assertEqual("replay_rejected", caught.exception.code)
            self.assertEqual(
                "local face command request replay was rejected",
                str(caught.exception),
            )
            self.assertEqual(1, len(calls))
            self.assertEqual({sentinel}, set(parent.iterdir()))

    def test_process_and_result_failures_are_fixed_path_free_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, sentinel = self._layout(root)
            base_request = _request(root)
            cases = (
                ("nonzero", "process_failed", {"returncode": 7}),
                ("missing", "result_count_invalid", {"write_result": False}),
                ("multiple", "result_count_invalid", {"write_extra": True}),
                ("preconsumed", "result_count_invalid", {"preconsume": True}),
            )
            for seed, expected_code, options in cases:
                with self.subTest(seed=seed):
                    request = replace(base_request, request_nonce=_sha(seed.encode()))
                    batch = _batch(request)
                    adapter = CommandFaceEmbeddingAdapter(
                        command=(str(executable),),
                        staging_parent=parent,
                        _process_factory=_factory(request, batch, **options),
                        _process_terminator=_terminate_fake,
                    )
                    with (
                        patch(
                            "baby_monitor.identity_protocol.canonical_face_pipeline_fingerprint",
                            return_value="c" * 64,
                        ),
                        self.assertRaises(FaceCommandAdapterError) as caught,
                    ):
                        adapter.embed(request)
                    self.assertEqual(expected_code, caught.exception.code)
                    self.assertNotIn(str(root), str(caught.exception))
                    self.assertNotIn("embedding", str(caught.exception).casefold())
                    self.assertEqual({sentinel}, set(parent.iterdir()))

            tampered_request = replace(
                base_request,
                request_nonce=_sha(b"tampered"),
            )
            tampered_batch = replace(_batch(tampered_request), request_sha256="f" * 64)
            adapter = CommandFaceEmbeddingAdapter(
                command=(str(executable),),
                staging_parent=parent,
                _process_factory=_factory(tampered_request, tampered_batch),
                _process_terminator=_terminate_fake,
            )
            with self.assertRaises(FaceCommandAdapterError) as caught:
                adapter.embed(tampered_request)
            self.assertEqual("result_invalid", caught.exception.code)
            self.assertEqual({sentinel}, set(parent.iterdir()))

    def test_timeout_oversize_and_disclosures_fail_closed_without_echo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, sentinel = self._layout(root)
            base_request = _request(root)
            cases = (
                (
                    "timeout",
                    "timeout",
                    {"returncode": None},
                    {"timeout_seconds": 0.01},
                ),
                (
                    "oversize",
                    "output_too_large",
                    {"stdout": b"x" * (MAX_FACE_COMMAND_CAPTURE_BYTES + 1)},
                    {},
                ),
                (
                    "embedding-key",
                    "output_disclosure",
                    {"stdout": b'{"embedding":[1,0]}'},
                    {},
                ),
                (
                    "private-path",
                    "output_disclosure",
                    {"leak_dynamic_path": True},
                    {},
                ),
            )
            for seed, expected_code, factory_options, adapter_options in cases:
                with self.subTest(seed=seed):
                    request = replace(base_request, request_nonce=_sha(seed.encode()))
                    batch = _batch(request)
                    adapter = CommandFaceEmbeddingAdapter(
                        command=(str(executable),),
                        staging_parent=parent,
                        _process_factory=_factory(
                            request,
                            batch,
                            **factory_options,
                        ),
                        _process_terminator=_terminate_fake,
                        **adapter_options,
                    )
                    with self.assertRaises(FaceCommandAdapterError) as caught:
                        adapter.embed(request)
                    self.assertEqual(expected_code, caught.exception.code)
                    self.assertNotIn(str(root), str(caught.exception))
                    self.assertNotIn("[1,0]", str(caught.exception))
                    self.assertEqual({sentinel}, set(parent.iterdir()))

    def test_start_failure_is_sanitized_and_cleanup_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, sentinel = self._layout(root)
            request = _request(root)

            def fail_start(*_args: Any, **_kwargs: Any) -> _FakeProcess:
                raise OSError(f"secret path: {root}")

            adapter = CommandFaceEmbeddingAdapter(
                command=(str(executable),),
                staging_parent=parent,
                _process_factory=fail_start,
                _process_terminator=_terminate_fake,
            )
            with self.assertRaises(FaceCommandAdapterError) as caught:
                adapter.embed(request)
            self.assertEqual("start_failed", caught.exception.code)
            self.assertEqual("local face command could not start", str(caught.exception))
            self.assertNotIn(str(root), str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)
            self.assertEqual({sentinel}, set(parent.iterdir()))

    def test_configuration_rejects_shell_placeholders_controls_and_weak_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable, parent, _sentinel = self._layout(root)
            invalid_commands = (
                (),
                "not-a-sequence",
                ("relative-runner",),
                (str(executable), "{request}"),
                (str(executable), "bad\nargument"),
                (str(Path("/bin/sh").resolve()),),
            )
            for command in invalid_commands:
                with (
                    self.subTest(command=command),
                    self.assertRaises(FaceCommandAdapterError) as caught,
                ):
                    CommandFaceEmbeddingAdapter(
                        command=command,  # type: ignore[arg-type]
                        staging_parent=parent,
                    )
                self.assertEqual("invalid_configuration", caught.exception.code)
            executable.chmod(0o722)
            try:
                with self.assertRaises(FaceCommandAdapterError) as caught:
                    CommandFaceEmbeddingAdapter(
                        command=(str(executable),),
                        staging_parent=parent,
                    )
                self.assertEqual("invalid_configuration", caught.exception.code)
            finally:
                executable.chmod(0o700)

            parent.chmod(0o750)
            try:
                with self.assertRaises(FaceCommandAdapterError) as caught:
                    CommandFaceEmbeddingAdapter(
                        command=(str(executable),),
                        staging_parent=parent,
                    )
                self.assertEqual("invalid_configuration", caught.exception.code)
            finally:
                parent.chmod(0o700)

            for options in (
                {"timeout_seconds": 0.0},
                {"max_captured_bytes": 100},
            ):
                with self.assertRaises(FaceCommandAdapterError) as caught:
                    CommandFaceEmbeddingAdapter(
                        command=(str(executable),),
                        staging_parent=parent,
                        **options,
                    )
                self.assertEqual("invalid_configuration", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
