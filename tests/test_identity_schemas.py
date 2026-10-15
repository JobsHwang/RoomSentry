from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_IDENTITY_SCHEMA = _REPO_ROOT / "schemas/identity-assessment-v1.schema.json"
_EMBEDDING_SCHEMA = _REPO_ROOT / "schemas/face-embedding-batch-v2.schema.json"
_MODEL_SCHEMA = _REPO_ROOT / "schemas/face-model-manifest-v1.schema.json"
_VALIDATOR_PROBE = (
    "from jsonschema import Draft202012Validator; "
    "assert Draft202012Validator.META_SCHEMA['$id'].endswith('/draft/2020-12/schema')"
)
_VALIDATOR_SCRIPT = """
import json
import sys

from jsonschema import Draft202012Validator, FormatChecker

with open(sys.argv[1], encoding="utf-8") as handle:
    schema = json.load(handle)
Draft202012Validator.check_schema(schema)
instance = json.load(sys.stdin)
errors = sorted(
    Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance),
    key=lambda error: tuple(str(part) for part in error.absolute_path),
)
if errors:
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        print(f"{location}: {error.message}", file=sys.stderr)
    raise SystemExit(1)
"""


def _find_validator_python() -> str | None:
    candidates = [shutil.which("python3"), shutil.which("python"), sys.executable]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        probe = subprocess.run(
            [candidate, "-c", _VALIDATOR_PROBE],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return candidate
    return None


_VALIDATOR_PYTHON = _find_validator_python()


def _identity_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "assessment_id": "assessment_01",
        "event_id": "event_01",
        "camera_id": "cam_01",
        "observations": [
            {
                "face_id": "face_01",
                "frame_ref": "frame_01",
                "bbox_xyxy": [0.1, 0.2, 0.6, 0.8],
                "detector_confidence": 0.95,
                "quality_state": "accepted",
                "rejection_codes": [],
            }
        ],
        "inferences": [
            {
                "face_id": "face_01",
                "outcome": "matched",
                "profile_id": "profile_01",
                "declared_kind": "baby",
                "match_score": 0.93,
                "decision_threshold": 0.8,
                "runner_up_margin": 0.2,
                "runner_up_margin_threshold": 0.1,
                "reason_code": "threshold_and_margin_passed",
                "support_frame_refs": ["frame_01"],
            }
        ],
        "model_fingerprint": "a" * 64,
        "embedding_dimension": 128,
        "matcher_revision": "cosine_v1",
        "threshold_revision": "shadow_v1",
        "match_threshold": 0.8,
        "runner_up_margin_threshold": 0.1,
        "cross_camera_linking": False,
        "tracker_identity_propagation": False,
        "paths_disclosed": False,
    }


def _landmarks() -> list[dict[str, Any]]:
    return [
        {"name": name, "x": 0.5, "y": 0.5, "in_frame": True}
        for name in (
            "right_eye",
            "left_eye",
            "nose_tip",
            "right_mouth_corner",
            "left_mouth_corner",
        )
    ]


def _face(face_id: str = "face_01", *, accepted: bool = True) -> dict[str, Any]:
    return {
        "face_id": face_id,
        "frame_ref": "frame_01",
        "bbox_xyxy": [0.1, 0.2, 0.6, 0.8],
        "detector_confidence": 0.95,
        "landmarks": _landmarks(),
        "quality_state": "accepted" if accepted else "rejected",
        "rejection_codes": [] if accepted else ["face_too_small"],
        "embedding": ([1.0] + [0.0] * 127) if accepted else None,
    }


def _image(frame_ref: str, faces: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "frame_ref": frame_ref,
        "source_sha256": "d" * 64,
        "width": 640,
        "height": 480,
        "faces": copy.deepcopy(faces),
    }
    for face in result["faces"]:
        face["frame_ref"] = frame_ref
    return result


def _embedding_payload(
    *,
    mode: str = "assessment",
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": 2,
        "request_id": "request_01",
        "request_sha256": "e" * 64,
        "mode": mode,
        "face_adapter": "opencv_yunet_sface_v1",
        "detector_model_id": "fixture.detector",
        "detector_model_revision": "1" * 40,
        "detector_model_sha256": "a" * 64,
        "embedding_model_id": "fixture.embedding",
        "embedding_model_revision": "2" * 40,
        "embedding_model_sha256": "b" * 64,
        "model_fingerprint": "c" * 64,
        "images": images if images is not None else [_image("frame_01", [_face()])],
        "identity_inference": False,
        "paths_disclosed": False,
        "network_used": False,
    }


@unittest.skipUnless(
    _VALIDATOR_PYTHON,
    "a Python jsonschema Draft 2020-12 validator is required for schema tests",
)
class IdentitySchemaTests(unittest.TestCase):
    def assert_schema_valid(self, schema: Path, payload: dict[str, Any]) -> None:
        result = subprocess.run(
            [_VALIDATOR_PYTHON, "-c", _VALIDATOR_SCRIPT, str(schema)],
            input=json.dumps(payload, allow_nan=False),
            check=False,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def assert_schema_invalid(self, schema: Path, payload: dict[str, Any]) -> None:
        result = subprocess.run(
            [_VALIDATOR_PYTHON, "-c", _VALIDATOR_SCRIPT, str(schema)],
            input=json.dumps(payload, allow_nan=False),
            check=False,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        self.assertNotEqual(0, result.returncode, "payload unexpectedly passed the schema")

    def test_schemas_are_valid_draft_2020_12_and_accept_reviewed_examples(self) -> None:
        self.assert_schema_valid(_IDENTITY_SCHEMA, _identity_payload())
        self.assert_schema_valid(_EMBEDDING_SCHEMA, _embedding_payload())
        for path in (
            _REPO_ROOT / "config/opencv-yunet-2023mar.reviewed.example.json",
            _REPO_ROOT / "config/opencv-sface-2021dec.reviewed.example.json",
        ):
            self.assert_schema_valid(
                _MODEL_SCHEMA,
                json.loads(path.read_text(encoding="utf-8")),
            )

    def test_face_quality_state_controls_rejection_codes(self) -> None:
        accepted_with_rejection = _identity_payload()
        accepted_with_rejection["observations"][0]["rejection_codes"] = ["blurred"]
        self.assert_schema_invalid(_IDENTITY_SCHEMA, accepted_with_rejection)

        rejected_without_reason = _identity_payload()
        rejected_without_reason["observations"][0]["quality_state"] = "rejected"
        self.assert_schema_invalid(_IDENTITY_SCHEMA, rejected_without_reason)

    def test_identity_outcomes_reject_inconsistent_field_combinations(self) -> None:
        matched_cases = []
        for field, value in (
            ("profile_id", None),
            ("declared_kind", "unknown"),
            ("match_score", None),
            ("reason_code", "no_templates"),
        ):
            payload = _identity_payload()
            payload["inferences"][0][field] = value
            matched_cases.append(payload)

        unknown_with_profile = _identity_payload()
        unknown_with_profile["inferences"][0].update(
            outcome="unknown",
            declared_kind="unknown",
            reason_code="no_templates",
            match_score=None,
            runner_up_margin=None,
        )
        unknown_with_score = copy.deepcopy(unknown_with_profile)
        unknown_with_score["inferences"][0]["profile_id"] = None
        unknown_with_score["inferences"][0]["match_score"] = 0.2

        ambiguous_with_profile = _identity_payload()
        ambiguous_with_profile["inferences"][0].update(
            outcome="ambiguous",
            declared_kind="unknown",
            reason_code="insufficient_runner_up_margin",
        )

        quality_with_score = _identity_payload()
        quality_with_score["inferences"][0].update(
            outcome="quality_rejected",
            profile_id=None,
            declared_kind="unknown",
            reason_code="quality_rejected",
            runner_up_margin=None,
        )

        for payload in (
            *matched_cases,
            unknown_with_profile,
            unknown_with_score,
            ambiguous_with_profile,
            quality_with_score,
        ):
            with self.subTest(payload=payload["inferences"][0]):
                self.assert_schema_invalid(_IDENTITY_SCHEMA, payload)

    def test_all_identity_outcomes_have_a_valid_shape(self) -> None:
        unknown = _identity_payload()
        unknown["inferences"][0].update(
            outcome="unknown",
            profile_id=None,
            declared_kind="unknown",
            match_score=None,
            runner_up_margin=None,
            reason_code="single_profile_uncalibrated",
        )
        below = copy.deepcopy(unknown)
        below["inferences"][0].update(
            match_score=0.4,
            reason_code="below_match_threshold",
        )
        ambiguous = copy.deepcopy(unknown)
        ambiguous["inferences"][0].update(
            outcome="ambiguous",
            match_score=0.9,
            runner_up_margin=0.01,
            reason_code="insufficient_runner_up_margin",
        )
        quality = copy.deepcopy(unknown)
        quality["observations"][0].update(
            quality_state="rejected",
            rejection_codes=["blurred"],
        )
        quality["inferences"][0].update(
            outcome="quality_rejected",
            reason_code="quality_rejected",
        )
        for payload in (unknown, below, ambiguous, quality):
            with self.subTest(outcome=payload["inferences"][0]["outcome"]):
                self.assert_schema_valid(_IDENTITY_SCHEMA, payload)

    def test_embedding_quality_state_controls_embedding_and_rejection(self) -> None:
        accepted_without_embedding = _embedding_payload()
        accepted_without_embedding["images"][0]["faces"][0]["embedding"] = None
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, accepted_without_embedding)

        accepted_with_rejection = _embedding_payload()
        accepted_with_rejection["images"][0]["faces"][0]["rejection_codes"] = ["blurred"]
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, accepted_with_rejection)

        rejected_with_embedding = _embedding_payload(
            images=[_image("frame_01", [_face(accepted=False)])]
        )
        rejected_with_embedding["images"][0]["faces"][0]["embedding"] = [1.0] + [0.0] * 127
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, rejected_with_embedding)

        rejected_without_reason = _embedding_payload(
            images=[_image("frame_01", [_face(accepted=False)])]
        )
        rejected_without_reason["images"][0]["faces"][0]["rejection_codes"] = []
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, rejected_without_reason)

    def test_landmark_presence_and_order_are_strict(self) -> None:
        out_of_frame_with_coordinates = _embedding_payload()
        out_of_frame_with_coordinates["images"][0]["faces"][0]["landmarks"][0].update(
            in_frame=False
        )
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, out_of_frame_with_coordinates)

        in_frame_without_coordinates = _embedding_payload()
        in_frame_without_coordinates["images"][0]["faces"][0]["landmarks"][0].update(
            x=None,
            y=None,
        )
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, in_frame_without_coordinates)

        wrong_order = _embedding_payload()
        landmarks = wrong_order["images"][0]["faces"][0]["landmarks"]
        landmarks[0], landmarks[1] = landmarks[1], landmarks[0]
        self.assert_schema_invalid(_EMBEDDING_SCHEMA, wrong_order)

    def test_enrollment_requires_exactly_one_accepted_face_globally(self) -> None:
        accepted = _face("face_01")
        rejected = _face("face_02", accepted=False)
        valid = _embedding_payload(
            mode="enrollment",
            images=[
                _image("frame_01", [accepted, rejected]),
                _image("frame_02", []),
            ],
        )
        self.assert_schema_valid(_EMBEDDING_SCHEMA, valid)

        zero = _embedding_payload(
            mode="enrollment",
            images=[_image("frame_01", [_face(accepted=False)])],
        )
        same_image_two = _embedding_payload(
            mode="enrollment",
            images=[_image("frame_01", [_face("face_01"), _face("face_02")])],
        )
        different_images_two = _embedding_payload(
            mode="enrollment",
            images=[
                _image("frame_01", [_face("face_01")]),
                _image("frame_02", [_face("face_02")]),
            ],
        )
        for payload in (zero, same_image_two, different_images_two):
            self.assert_schema_invalid(_EMBEDDING_SCHEMA, payload)

    def test_model_urls_reject_userinfo_but_allow_at_sign_in_path(self) -> None:
        payload = json.loads(
            (_REPO_ROOT / "config/opencv-sface-2021dec.reviewed.example.json").read_text(
                encoding="utf-8"
            )
        )
        for field in ("source_url", "license_url"):
            credentialed = copy.deepcopy(payload)
            # release-audit: allow=fixture-credential
            credentialed[field] = "https://user:secret@example.invalid/model.onnx"
            self.assert_schema_invalid(_MODEL_SCHEMA, credentialed)

        path_at_sign = copy.deepcopy(payload)
        path_at_sign["license_url"] = "https://example.invalid/licenses/user@example"
        self.assert_schema_valid(_MODEL_SCHEMA, path_at_sign)

    def test_bbox_arithmetic_boundaries_are_explicitly_delegated(self) -> None:
        for schema_path in (_IDENTITY_SCHEMA, _EMBEDDING_SCHEMA):
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            comment = schema["$defs"]["bbox"]["$comment"]
            self.assertIn("x_min < x_max", comment)
            self.assertIn("canonical parser", comment)


if __name__ == "__main__":
    unittest.main()
