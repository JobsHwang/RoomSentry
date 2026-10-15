from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.local_runners.mage_vl_4b import OFFICIAL_REVISION
from baby_monitor.local_runners.mage_vl_predicate import (
    ATOMIC_MODEL_ID,
    PROMPT_VERSION,
    MagePredicateRequest,
    MagePredicateRunnerError,
    build_atomic_prompt,
    load_predicate_request,
    main,
    normalize_model_answer,
)
from baby_monitor.visual_predicate import VisualPredicateAnswer


def _private_root() -> tempfile.TemporaryDirectory[str]:
    directory = tempfile.TemporaryDirectory()
    Path(directory.name).chmod(0o700)
    return directory


def _write_request(root: Path, *, duplicate: bool = False) -> Path:
    frame = root / "frame_00.png"
    payload = b"synthetic-public-frame"
    frame.write_bytes(payload)
    frame.chmod(0o600)
    value = {
        "version": 1,
        "event_id": "public_event_01",
        "predicate": "posture_transition_visible",
        "frames": [
            {
                "frame_id": "t00_full",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "media_type": "image/png",
                "path": str(frame),
            }
        ],
        "model_id": ATOMIC_MODEL_ID,
        "model_revision": OFFICIAL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "result_contract": "visual_predicate_result_v1",
    }
    request = root / "request.json"
    raw = json.dumps(value, separators=(",", ":"))
    if duplicate:
        raw = raw.replace('"version":1', '"version":1,"version":1')
    request.write_text(raw, encoding="utf-8")
    request.chmod(0o600)
    return request


class MageVLPredicateRunnerTests(unittest.TestCase):
    def test_request_admission_is_strict_private_and_hash_bound(self) -> None:
        with _private_root() as directory:
            root = Path(directory)
            request = load_predicate_request(_write_request(root))
            self.assertEqual(("t00_full",), request.frame_ids)
            self.assertNotIn(str(root), repr(request))

            path = root / "request.json"
            path.chmod(0o644)
            with self.assertRaisesRegex(MagePredicateRunnerError, "private file"):
                load_predicate_request(path)

    def test_request_rejects_duplicate_key_hash_mismatch_and_outside_path(self) -> None:
        with _private_root() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(MagePredicateRunnerError, "duplicate"):
                load_predicate_request(_write_request(root, duplicate=True))

        with _private_root() as directory:
            root = Path(directory)
            path = _write_request(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["frames"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(MagePredicateRunnerError, "admission"):
                load_predicate_request(path)

        with _private_root() as directory, _private_root() as outside_directory:
            root = Path(directory)
            path = _write_request(root)
            outside = Path(outside_directory) / "outside.png"
            outside.write_bytes(b"outside")
            outside.chmod(0o600)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["frames"][0]["path"] = str(outside)
            value["frames"][0]["sha256"] = hashlib.sha256(b"outside").hexdigest()
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(MagePredicateRunnerError, "private file"):
                load_predicate_request(path)

    def test_prompt_explains_full_crop_and_forbids_identity_risk(self) -> None:
        with _private_root() as directory:
            request = load_predicate_request(_write_request(Path(directory)))
        system, user = build_atomic_prompt(request)
        self.assertIn("exactly one", system)
        self.assertIn("Never infer identity", system)
        self.assertIn("Detector crops", system)
        self.assertIn("same time index", user)
        self.assertIn("yes, no, unknown", user)

    def test_normalizes_only_exact_answer_and_constructs_bound_result(self) -> None:
        with _private_root() as directory:
            request = load_predicate_request(_write_request(Path(directory)))
        yes = normalize_model_answer(request, "yes")
        self.assertIs(yes.answer, VisualPredicateAnswer.YES)
        self.assertEqual(("t00_full",), yes.evidence_frames)
        self.assertEqual(0.5, yes.confidence)
        self.assertFalse(yes.subject_identity_inference)
        self.assertFalse(yes.risk_inference)
        self.assertFalse(yes.notification_decision)

        invalid = normalize_model_answer(request, "The answer is yes.")
        self.assertIs(invalid.answer, VisualPredicateAnswer.UNKNOWN)
        self.assertEqual((), invalid.evidence_frames)
        self.assertEqual(0.0, invalid.confidence)

    def test_face_answer_family_rejects_yes_as_unknown(self) -> None:
        with _private_root() as directory:
            root = Path(directory)
            path = _write_request(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["predicate"] = "face_visibility"
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)
            request = load_predicate_request(path)
        self.assertIs(
            normalize_model_answer(request, "yes").answer,
            VisualPredicateAnswer.UNKNOWN,
        )
        self.assertIs(
            normalize_model_answer(request, "clear").answer,
            VisualPredicateAnswer.CLEAR,
        )

    def test_main_refuses_confirmation_before_request_or_model_access(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with (
            patch(
                "baby_monitor.local_runners.mage_vl_predicate.TrustedModelCodeManifest.read"
            ) as manifest_read,
            patch(
                "baby_monitor.local_runners.mage_vl_predicate.load_predicate_request"
            ) as request_read,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = main(
                [
                    "--request",
                    "/missing/request.json",
                    "--model-path",
                    "/missing/model",
                    "--code-manifest",
                    "/missing/manifest.json",
                ]
            )
        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertEqual("runner_failed", json.loads(stderr.getvalue())["error_code"])
        manifest_read.assert_not_called()
        request_read.assert_not_called()

    def test_main_prints_only_bound_result_with_injected_session(self) -> None:
        with _private_root() as directory:
            root = Path(directory)
            request_path = _write_request(root)
            request = load_predicate_request(request_path)
            expected = normalize_model_answer(request, "yes")
            stdout = StringIO()
            stderr = StringIO()

            class FakeManifest:
                revision = OFFICIAL_REVISION

                def verify(self, _root: Path) -> None:
                    return None

            class FakeSession:
                def __init__(self, _settings: object):
                    pass

                def __enter__(self):
                    return self

                def assess(self, _request: MagePredicateRequest):
                    print("untrusted stdout")
                    print("untrusted stderr", file=sys.stderr)
                    return expected

                def __exit__(self, *_args: object) -> None:
                    return None

            class FakeCache:
                def __enter__(self):
                    return None

                def __exit__(self, *_args: object) -> None:
                    return None

            class FakeSettings:
                def __init__(self, **_kwargs: object):
                    pass

            with (
                patch(
                    "baby_monitor.local_runners.mage_vl_predicate.TrustedModelCodeManifest.read",
                    return_value=FakeManifest(),
                ),
                patch(
                    "baby_monitor.local_runners.mage_vl_predicate.MagePredicateSession",
                    FakeSession,
                ),
                patch(
                    "baby_monitor.local_runners.mage_vl_predicate.isolated_transformers_module_cache",
                    return_value=FakeCache(),
                ),
                patch(
                    "baby_monitor.local_runners.mage_vl_predicate.MageVLSettings",
                    FakeSettings,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "--request",
                        str(request_path),
                        "--model-path",
                        "/model",
                        "--code-manifest",
                        "/manifest.json",
                        "--confirm-reviewed-remote-code",
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            self.assertEqual(expected, type(expected).from_json(stdout.getvalue()))

    def test_source_has_no_network_or_download_path(self) -> None:
        source = Path("src/baby_monitor/local_runners/mage_vl_predicate.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("snapshot_download", "http://", "https://", "requests."):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', source)


if __name__ == "__main__":
    unittest.main()
