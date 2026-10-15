from __future__ import annotations

import unittest

from baby_monitor.identity_domain import (
    DeclaredKind,
    FaceObservation,
    FaceQualityState,
    IdentityOutcome,
    IdentityReason,
)
from baby_monitor.identity_matching import (
    FaceEmbedding,
    IdentityMatchingError,
    IdentityMatchPolicy,
    IdentityTemplate,
    assess_identities,
)

MODEL = "a" * 64
OTHER_MODEL = "b" * 64


def _observation(
    face_id: str = "face_01",
    frame_ref: str = "frame_01",
    *,
    rejected: bool = False,
) -> FaceObservation:
    return FaceObservation(
        face_id=face_id,
        frame_ref=frame_ref,
        bbox_xyxy=(0.1, 0.1, 0.5, 0.7),
        detector_confidence=0.9,
        quality_state=(FaceQualityState.REJECTED if rejected else FaceQualityState.ACCEPTED),
        rejection_codes=("blurred",) if rejected else (),
    )


def _policy(*, threshold: float = 0.8, margin: float = 0.1) -> IdentityMatchPolicy:
    return IdentityMatchPolicy(
        model_fingerprint=MODEL,
        embedding_dimension=3,
        match_threshold=threshold,
        runner_up_margin_threshold=margin,
        matcher_revision="cosine_v1",
        threshold_revision="synthetic_v1",
    )


def _template(
    template_id: str,
    profile_id: str,
    embedding: tuple[float, ...],
    *,
    kind: DeclaredKind = DeclaredKind.BABY,
    fingerprint: str = MODEL,
) -> IdentityTemplate:
    return IdentityTemplate(
        template_id=template_id,
        profile_id=profile_id,
        declared_kind=kind,
        model_fingerprint=fingerprint,
        embedding=embedding,
    )


def _assess(
    *,
    observations: tuple[FaceObservation, ...] = (_observation(),),
    embeddings: tuple[FaceEmbedding, ...] | None = None,
    templates: tuple[IdentityTemplate, ...],
    policy: IdentityMatchPolicy | None = None,
):
    if embeddings is None:
        embeddings = (FaceEmbedding("face_01", MODEL, (1.0, 0.0, 0.0)),)
    return assess_identities(
        assessment_id="assessment_01",
        event_id="event_01",
        camera_id="cam_01",
        observations=observations,
        embeddings=embeddings,
        templates=templates,
        policy=policy or _policy(),
    )


class IdentityMatchingTests(unittest.TestCase):
    def test_match_requires_both_top_score_and_distinct_profile_margin(self) -> None:
        matched = _assess(
            templates=(
                _template("template_01", "profile_baby", (1.0, 0.0, 0.0)),
                _template(
                    "template_02",
                    "profile_adult",
                    (0.0, 1.0, 0.0),
                    kind=DeclaredKind.ADULT,
                ),
            )
        ).inferences[0]
        self.assertEqual(IdentityOutcome.MATCHED, matched.outcome)
        self.assertEqual("profile_baby", matched.profile_id)
        self.assertEqual(DeclaredKind.BABY, matched.declared_kind)
        self.assertAlmostEqual(1.0, matched.match_score or 0.0)
        self.assertAlmostEqual(1.0, matched.runner_up_margin or 0.0)

        ambiguous = _assess(
            templates=(
                _template("template_01", "profile_baby", (1.0, 0.0, 0.0)),
                _template(
                    "template_02",
                    "profile_adult",
                    (0.995, 0.1, 0.0),
                    kind=DeclaredKind.ADULT,
                ),
            )
        ).inferences[0]
        self.assertEqual(IdentityOutcome.AMBIGUOUS, ambiguous.outcome)
        self.assertIsNone(ambiguous.profile_id)
        self.assertEqual(
            IdentityReason.INSUFFICIENT_RUNNER_UP_MARGIN,
            ambiguous.reason_code,
        )

        below = _assess(
            embeddings=(FaceEmbedding("face_01", MODEL, (0.0, 0.0, 1.0)),),
            templates=(
                _template("template_01", "profile_baby", (1.0, 0.0, 0.0)),
                _template(
                    "template_02",
                    "profile_adult",
                    (0.0, 1.0, 0.0),
                    kind=DeclaredKind.ADULT,
                ),
            ),
        ).inferences[0]
        self.assertEqual(IdentityOutcome.UNKNOWN, below.outcome)
        self.assertEqual(IdentityReason.BELOW_MATCH_THRESHOLD, below.reason_code)
        self.assertEqual(0.0, below.match_score)

    def test_four_profile_household_matches_child_and_rejects_ambiguous_adult(
        self,
    ) -> None:
        templates = (
            _template("template_child", "profile_child", (1.0, 0.0, 0.0)),
            _template(
                "template_adult_01",
                "profile_adult_01",
                (0.0, 1.0, 0.0),
                kind=DeclaredKind.ADULT,
            ),
            _template(
                "template_adult_02",
                "profile_adult_02",
                (0.0, 0.995, 0.1),
                kind=DeclaredKind.ADULT,
            ),
            _template(
                "template_adult_03",
                "profile_adult_03",
                (-1.0, 0.0, 0.0),
                kind=DeclaredKind.ADULT,
            ),
        )

        child = _assess(templates=templates).inferences[0]
        self.assertEqual(IdentityOutcome.MATCHED, child.outcome)
        self.assertEqual("profile_child", child.profile_id)
        self.assertEqual(DeclaredKind.BABY, child.declared_kind)

        ambiguous_adult = _assess(
            embeddings=(FaceEmbedding("face_01", MODEL, (0.0, 1.0, 0.0)),),
            templates=templates,
        ).inferences[0]
        self.assertEqual(IdentityOutcome.AMBIGUOUS, ambiguous_adult.outcome)
        self.assertIsNone(ambiguous_adult.profile_id)
        self.assertEqual(
            IdentityReason.INSUFFICIENT_RUNNER_UP_MARGIN,
            ambiguous_adult.reason_code,
        )

    def test_single_profile_without_negative_calibration_fails_closed(self) -> None:
        inference = _assess(
            templates=(
                _template("template_01", "profile_baby", (1.0, 0.0, 0.0)),
                _template("template_02", "profile_baby", (0.9, 0.1, 0.0)),
            )
        ).inferences[0]

        self.assertEqual(IdentityOutcome.UNKNOWN, inference.outcome)
        self.assertEqual(IdentityReason.SINGLE_PROFILE_UNCALIBRATED, inference.reason_code)
        self.assertIsNone(inference.profile_id)
        self.assertIsNone(inference.runner_up_margin)

    def test_model_nonfinite_dimension_and_norm_errors_fail_closed(self) -> None:
        cases = (
            (
                "query_model",
                (FaceEmbedding("face_01", OTHER_MODEL, (1.0, 0.0, 0.0)),),
                (_template("template_01", "profile_baby", (1.0, 0.0, 0.0)),),
                IdentityReason.MODEL_FINGERPRINT_MISMATCH,
            ),
            (
                "template_model",
                (FaceEmbedding("face_01", MODEL, (1.0, 0.0, 0.0)),),
                (
                    _template(
                        "template_01",
                        "profile_baby",
                        (1.0, 0.0, 0.0),
                        fingerprint=OTHER_MODEL,
                    ),
                ),
                IdentityReason.MODEL_FINGERPRINT_MISMATCH,
            ),
            (
                "query_dimension",
                (FaceEmbedding("face_01", MODEL, (1.0, 0.0)),),
                (_template("template_01", "profile_baby", (1.0, 0.0, 0.0)),),
                IdentityReason.EMBEDDING_DIMENSION_MISMATCH,
            ),
            (
                "template_dimension",
                (FaceEmbedding("face_01", MODEL, (1.0, 0.0, 0.0)),),
                (_template("template_01", "profile_baby", (1.0, 0.0)),),
                IdentityReason.EMBEDDING_DIMENSION_MISMATCH,
            ),
            (
                "query_nonfinite",
                (FaceEmbedding("face_01", MODEL, (float("nan"), 0.0, 0.0)),),
                (_template("template_01", "profile_baby", (1.0, 0.0, 0.0)),),
                IdentityReason.NONFINITE_EMBEDDING,
            ),
            (
                "template_nonfinite",
                (FaceEmbedding("face_01", MODEL, (1.0, 0.0, 0.0)),),
                (_template("template_01", "profile_baby", (float("inf"), 0.0, 0.0)),),
                IdentityReason.NONFINITE_EMBEDDING,
            ),
            (
                "zero_norm",
                (FaceEmbedding("face_01", MODEL, (0.0, 0.0, 0.0)),),
                (_template("template_01", "profile_baby", (1.0, 0.0, 0.0)),),
                IdentityReason.INVALID_EMBEDDING_NORM,
            ),
        )
        for label, embeddings, templates, expected_reason in cases:
            with self.subTest(label=label):
                inference = _assess(embeddings=embeddings, templates=templates).inferences[0]
                self.assertEqual(IdentityOutcome.UNKNOWN, inference.outcome)
                self.assertEqual(expected_reason, inference.reason_code)
                self.assertIsNone(inference.profile_id)
                self.assertIsNone(inference.match_score)

    def test_each_face_is_independent_and_tracker_propagation_remains_disabled(self) -> None:
        observations = (
            _observation("face_01", "frame_01"),
            _observation("face_02", "frame_01", rejected=True),
            _observation("face_03", "frame_02"),
        )
        assessment = _assess(
            observations=observations,
            embeddings=(
                FaceEmbedding("face_01", MODEL, (1.0, 0.0, 0.0)),
                FaceEmbedding("face_02", MODEL, (1.0, 0.0, 0.0)),
            ),
            templates=(
                _template("template_01", "profile_baby", (1.0, 0.0, 0.0)),
                _template(
                    "template_02",
                    "profile_adult",
                    (0.0, 1.0, 0.0),
                    kind=DeclaredKind.ADULT,
                ),
            ),
        )

        self.assertEqual(
            (
                IdentityOutcome.MATCHED,
                IdentityOutcome.QUALITY_REJECTED,
                IdentityOutcome.UNKNOWN,
            ),
            tuple(item.outcome for item in assessment.inferences),
        )
        self.assertEqual(
            IdentityReason.MISSING_EMBEDDING,
            assessment.inferences[2].reason_code,
        )
        self.assertEqual(
            (("frame_01",), ("frame_01",), ("frame_02",)),
            tuple(item.support_frame_refs for item in assessment.inferences),
        )
        self.assertFalse(assessment.cross_camera_linking)
        self.assertFalse(assessment.tracker_identity_propagation)
        self.assertNotIn("tracking_id", assessment.to_json())

    def test_no_templates_is_unknown_and_unobserved_embedding_is_rejected(self) -> None:
        inference = _assess(templates=()).inferences[0]
        self.assertEqual(IdentityOutcome.UNKNOWN, inference.outcome)
        self.assertEqual(IdentityReason.NO_TEMPLATES, inference.reason_code)

        with self.assertRaisesRegex(IdentityMatchingError, "no face observation"):
            _assess(
                embeddings=(FaceEmbedding("face_other", MODEL, (1.0, 0.0, 0.0)),),
                templates=(),
            )


if __name__ == "__main__":
    unittest.main()
