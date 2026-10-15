from __future__ import annotations

import json
import unittest
from pathlib import Path

from baby_monitor.identity_domain import (
    DeclaredKind,
    FaceObservation,
    FaceQualityState,
    IdentityAssessment,
    IdentityDomainError,
    IdentityInference,
    IdentityOutcome,
    IdentityReason,
)


def _observation(
    face_id: str = "face_01",
    frame_ref: str = "frame_01",
    *,
    rejected: bool = False,
) -> FaceObservation:
    return FaceObservation(
        face_id=face_id,
        frame_ref=frame_ref,
        bbox_xyxy=(0.1, 0.2, 0.4, 0.8),
        detector_confidence=0.91,
        quality_state=(FaceQualityState.REJECTED if rejected else FaceQualityState.ACCEPTED),
        rejection_codes=("face_too_small",) if rejected else (),
    )


def _matched(face_id: str = "face_01", frame_ref: str = "frame_01") -> IdentityInference:
    return IdentityInference(
        face_id=face_id,
        outcome=IdentityOutcome.MATCHED,
        profile_id="profile_01",
        declared_kind=DeclaredKind.BABY,
        match_score=0.93,
        decision_threshold=0.8,
        runner_up_margin=0.22,
        runner_up_margin_threshold=0.1,
        reason_code=IdentityReason.THRESHOLD_AND_MARGIN_PASSED,
        support_frame_refs=(frame_ref,),
    )


def _assessment() -> IdentityAssessment:
    return IdentityAssessment(
        assessment_id="assessment_01",
        event_id="event_01",
        camera_id="cam_01",
        observations=(_observation(),),
        inferences=(_matched(),),
        model_fingerprint="a" * 64,
        embedding_dimension=128,
        matcher_revision="cosine_v1",
        threshold_revision="household_shadow_v1",
        match_threshold=0.8,
        runner_up_margin_threshold=0.1,
    )


class IdentityDomainTests(unittest.TestCase):
    def test_assessment_round_trips_and_matches_public_schema_shape(self) -> None:
        assessment = _assessment()
        restored = IdentityAssessment.from_json(assessment.to_json())

        self.assertEqual(assessment, restored)
        schema = json.loads(
            Path("schemas/identity-assessment-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(assessment.to_dict()))
        self.assertFalse(assessment.cross_camera_linking)
        self.assertFalse(assessment.tracker_identity_propagation)
        self.assertFalse(assessment.paths_disclosed)
        inference = assessment.to_dict()["inferences"][0]
        self.assertIn("match_score", inference)
        self.assertNotIn("confidence", inference)
        observation = assessment.to_dict()["observations"][0]
        self.assertFalse({"profile_id", "declared_kind", "match_score"} & set(observation))

    def test_json_and_every_nested_object_reject_extra_or_duplicate_fields(self) -> None:
        payload = _assessment().to_dict()
        with self.assertRaisesRegex(IdentityDomainError, "fields do not match"):
            IdentityAssessment.from_dict({**payload, "extra": True})

        payload = json.loads(_assessment().to_json())
        payload["observations"][0]["identity_guess"] = "profile_01"
        with self.assertRaisesRegex(IdentityDomainError, "observation fields"):
            IdentityAssessment.from_dict(payload)

        payload = json.loads(_assessment().to_json())
        payload["inferences"][0]["identity_confidence"] = 0.99
        with self.assertRaisesRegex(IdentityDomainError, "inference fields"):
            IdentityAssessment.from_dict(payload)

        with self.assertRaisesRegex(IdentityDomainError, "duplicate keys"):
            IdentityAssessment.from_json('{"version":1,"version":1}')
        with self.assertRaisesRegex(IdentityDomainError, "non-standard number"):
            IdentityAssessment.from_json('{"version":NaN}')

    def test_face_observation_contains_facts_only_and_quality_is_consistent(self) -> None:
        rejected = _observation(rejected=True)
        self.assertEqual(("face_too_small",), rejected.rejection_codes)

        with self.assertRaisesRegex(IdentityDomainError, "cannot have rejection"):
            FaceObservation(
                face_id="face_01",
                frame_ref="frame_01",
                bbox_xyxy=(0.1, 0.1, 0.4, 0.4),
                detector_confidence=0.8,
                quality_state=FaceQualityState.ACCEPTED,
                rejection_codes=("blurred",),
            )
        with self.assertRaisesRegex(IdentityDomainError, "requires a rejection"):
            FaceObservation(
                face_id="face_01",
                frame_ref="frame_01",
                bbox_xyxy=(0.1, 0.1, 0.4, 0.4),
                detector_confidence=0.8,
                quality_state=FaceQualityState.REJECTED,
            )
        with self.assertRaisesRegex(IdentityDomainError, "positive normalized area"):
            FaceObservation(
                face_id="face_01",
                frame_ref="frame_01",
                bbox_xyxy=(0.4, 0.1, 0.4, 0.4),
                detector_confidence=0.8,
                quality_state=FaceQualityState.ACCEPTED,
            )

    def test_inference_outcome_invariants_fail_closed(self) -> None:
        with self.assertRaisesRegex(IdentityDomainError, "below threshold"):
            IdentityInference(
                face_id="face_01",
                outcome=IdentityOutcome.MATCHED,
                profile_id="profile_01",
                declared_kind=DeclaredKind.BABY,
                match_score=0.7,
                decision_threshold=0.8,
                runner_up_margin=0.2,
                runner_up_margin_threshold=0.1,
                reason_code=IdentityReason.THRESHOLD_AND_MARGIN_PASSED,
                support_frame_refs=("frame_01",),
            )
        with self.assertRaisesRegex(IdentityDomainError, "ambiguous"):
            IdentityInference(
                face_id="face_01",
                outcome=IdentityOutcome.AMBIGUOUS,
                profile_id="profile_01",
                declared_kind=DeclaredKind.BABY,
                match_score=0.9,
                decision_threshold=0.8,
                runner_up_margin=0.01,
                runner_up_margin_threshold=0.1,
                reason_code=IdentityReason.INSUFFICIENT_RUNNER_UP_MARGIN,
                support_frame_refs=("frame_01",),
            )
        with self.assertRaisesRegex(IdentityDomainError, "cannot expose a score"):
            IdentityInference(
                face_id="face_01",
                outcome=IdentityOutcome.UNKNOWN,
                profile_id=None,
                declared_kind=DeclaredKind.UNKNOWN,
                match_score=0.1,
                decision_threshold=0.8,
                runner_up_margin=None,
                runner_up_margin_threshold=0.1,
                reason_code=IdentityReason.MODEL_FINGERPRINT_MISMATCH,
                support_frame_refs=("frame_01",),
            )

    def test_assessment_enforces_per_face_pairing_and_disabled_linking(self) -> None:
        kwargs = {
            "assessment_id": "assessment_01",
            "event_id": "event_01",
            "camera_id": "cam_01",
            "observations": (_observation(),),
            "inferences": (_matched(),),
            "model_fingerprint": "a" * 64,
            "embedding_dimension": 128,
            "matcher_revision": "cosine_v1",
            "threshold_revision": "shadow_v1",
            "match_threshold": 0.8,
            "runner_up_margin_threshold": 0.1,
        }
        for field in (
            "cross_camera_linking",
            "tracker_identity_propagation",
            "paths_disclosed",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(IdentityDomainError, "must be false"),
            ):
                IdentityAssessment(**kwargs, **{field: True})

        wrong_frame = _matched(frame_ref="frame_other")
        with self.assertRaisesRegex(IdentityDomainError, "support frame"):
            IdentityAssessment(**{**kwargs, "inferences": (wrong_frame,)})

        rejected = _observation(rejected=True)
        with self.assertRaisesRegex(IdentityDomainError, "quality fact"):
            IdentityAssessment(**{**kwargs, "observations": (rejected,)})


if __name__ == "__main__":
    unittest.main()
