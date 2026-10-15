from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from baby_monitor.household_ground_truth import (
    HouseholdEpisodeAnnotation,
    HouseholdEpisodeRange,
    HouseholdGroundTruthError,
    evaluate_household_trigger_coverage,
)
from baby_monitor.household_sampled_coverage import evaluate_household_sampled_coverage
from tests.test_household_ground_truth import _annotation, _media_admission
from tests.test_household_sampled_coverage import _sampled_scan

V1_FIELDS = {
    "annotation_id",
    "media_admission_sha256",
    "camera_id",
    "source_fps",
    "source_frame_count",
    "source_duration_seconds",
    "scenario_tags",
    "ranges",
    "review_state",
    "reviewers",
    "version",
    "human_verified",
    "model_generated_labels",
    "source_media_included",
    "paths_disclosed",
    "subject_identity_inference",
    "annotation_sha256",
}


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class HouseholdAnnotationTimebaseTests(unittest.TestCase):
    REFERENCE_SHA256 = "c" * 64

    def setUp(self) -> None:
        self.admission = _media_admission()

    def elapsed(self, *, review_state="approved"):
        return replace(
            _annotation(self.admission, review_state=review_state),
            version=2,
            timebase="elapsed_source_pts",
            source_fps=None,
            source_frame_count=None,
            source_duration_seconds=7.0,
            ranges=(
                HouseholdEpisodeRange(0, 2, "empty_room"),
                HouseholdEpisodeRange(2, 3, "person_present"),
                HouseholdEpisodeRange(5, 7, "danger_event"),
            ),
        )

    def binding(self, annotation, *, offset=0.0):
        return {
            "version": 1,
            "annotation_sha256": annotation.fingerprint,
            "media_admission_sha256": self.admission.fingerprint,
            "reference_sha256": self.REFERENCE_SHA256,
            "source_pts_at_annotation_zero_seconds": offset,
            "alignment_confirmed": True,
        }

    def test_v1_json_shape_and_prechange_fixture_hashes_remain_exact(self):
        expected = {
            "pending": "a6990d2482d561117355e38a1c670ced5b33bb08271b056ca1ad4019674ae326",
            "approved": "ee54a8918d270d5c677e814f6e6b2733583f22707e59192b6d8b4d524ba74f3e",
        }
        for state, fingerprint in expected.items():
            with self.subTest(state=state):
                annotation = _annotation(self.admission, review_state=state)
                self.assertEqual("nominal_frame_grid", annotation.timebase)
                self.assertEqual(fingerprint, annotation.fingerprint)
                self.assertEqual(V1_FIELDS, set(annotation.to_dict()))
                self.assertNotIn("timebase", json.loads(annotation.to_json()))
                self.assertEqual(
                    annotation, HouseholdEpisodeAnnotation.from_json(annotation.to_json())
                )

    def test_v2_elapsed_round_trip_omits_nominal_fields_and_keeps_explicit_duration(self):
        annotation = self.elapsed()
        value = annotation.to_dict()
        self.assertEqual(
            (V1_FIELDS - {"source_fps", "source_frame_count"}) | {"timebase"}, set(value)
        )
        self.assertEqual("elapsed_source_pts", value["timebase"])
        self.assertEqual(7.0, value["source_duration_seconds"])
        self.assertIsNone(annotation.source_fps)
        self.assertIsNone(annotation.source_frame_count)
        restored = HouseholdEpisodeAnnotation.from_json(annotation.to_json())
        self.assertEqual(annotation, restored)
        self.assertIsNone(restored.source_fps)
        self.assertIsNone(restored.source_frame_count)
        self.assertEqual(
            annotation.fingerprint, _canonical_hash(annotation.to_dict(include_fingerprint=False))
        )
        with tempfile.TemporaryDirectory(prefix="baby-monitor-annotation-timebase-") as directory:
            target = Path(directory) / "synthetic.json"
            annotation.write_new(target)
            self.assertEqual(annotation, HouseholdEpisodeAnnotation.read(target))

    def test_v1_retains_nominal_duration_constraint_and_disallows_elapsed_timebase(self):
        annotation = _annotation(self.admission)
        for changes in (
            {"source_duration_seconds": 7.0},
            {"source_fps": None},
            {"source_frame_count": None},
            {"timebase": "elapsed_source_pts"},
        ):
            with self.subTest(fields=tuple(changes)), self.assertRaises(HouseholdGroundTruthError):
                replace(annotation, **changes)

    def test_v2_rejects_nominal_metadata_mixed_timebases_and_wrong_wire_shape(self):
        annotation = self.elapsed()
        for changes in (
            {"source_fps": 25.0},
            {"source_frame_count": 100},
            {"source_fps": 25.0, "source_frame_count": 100},
            {"timebase": "nominal_frame_grid"},
            {"timebase": "unknown"},
        ):
            with self.subTest(fields=tuple(changes)), self.assertRaises(HouseholdGroundTruthError):
                replace(annotation, **changes)
        value = annotation.to_dict()
        malformed_values = (
            value | {"source_fps": None},
            value | {"source_frame_count": None},
            {key: item for key, item in value.items() if key != "timebase"},
            _annotation(self.admission).to_dict() | {"timebase": "nominal_frame_grid"},
        )
        for index, malformed in enumerate(malformed_values):
            with self.subTest(wire_case=index), self.assertRaises(HouseholdGroundTruthError):
                HouseholdEpisodeAnnotation.from_dict(malformed)

    def test_v2_approval_privacy_range_and_hash_guards_remain_strict(self):
        annotation = self.elapsed()
        for changes in (
            {"human_verified": False},
            {"reviewers": ()},
            {"model_generated_labels": True},
            {"source_media_included": True},
            {"paths_disclosed": True},
            {"subject_identity_inference": True},
            {"ranges": (HouseholdEpisodeRange(6, 8, "person_present"),)},
            {
                "ranges": (
                    HouseholdEpisodeRange(0, 3, "empty_room"),
                    HouseholdEpisodeRange(2, 4, "person_present"),
                )
            },
        ):
            with self.subTest(fields=tuple(changes)), self.assertRaises(HouseholdGroundTruthError):
                replace(annotation, **changes)
        tampered = annotation.to_dict() | {"source_duration_seconds": 8.0}
        with self.assertRaisesRegex(HouseholdGroundTruthError, "fingerprint"):
            HouseholdEpisodeAnnotation.from_dict(tampered)

    def test_v2_pending_round_trip_never_promotes_to_human_approved(self):
        pending = self.elapsed(review_state="pending")
        restored = HouseholdEpisodeAnnotation.from_json(pending.to_json())
        self.assertEqual("pending", restored.review_state)
        self.assertFalse(restored.human_verified)
        self.assertEqual((), restored.reviewers)
        scan = _sampled_scan(self.admission, ((5.5, (("person", 0.9),), "container_pts"),))
        with self.assertRaises(ValueError):
            evaluate_household_sampled_coverage(
                restored,
                scan,
                media_admission=self.admission,
                timeline_binding=self.binding(restored),
                reference_sha256=self.REFERENCE_SHA256,
                confidence_threshold=0.5,
            )

    def test_v2_end_boundary_is_strict_while_v1_retains_legacy_tolerance(self):
        legacy = replace(
            _annotation(self.admission),
            ranges=(HouseholdEpisodeRange(0.0, 4.0 + 5e-10, "person_present"),),
        )
        self.assertGreater(legacy.ranges[-1].end_seconds, legacy.source_duration_seconds)
        with self.assertRaises(HouseholdGroundTruthError):
            replace(
                self.elapsed(), ranges=(HouseholdEpisodeRange(0.0, 7.0 + 5e-10, "person_present"),)
            )

    def test_wire_arrays_require_lists_before_any_fingerprint_check(self):
        for annotation in (_annotation(self.admission), self.elapsed()):
            for field in ("ranges", "scenario_tags", "reviewers"):
                for invalid in (None, "private/path", ()):
                    with (
                        self.subTest(version=annotation.version, field=field, kind=type(invalid)),
                        self.assertRaisesRegex(HouseholdGroundTruthError, "arrays") as error,
                    ):
                        HouseholdEpisodeAnnotation.from_dict(
                            annotation.to_dict() | {field: invalid}
                        )
                    self.assertNotIn("private/path", str(error.exception))

    def test_wire_review_state_bad_types_fail_before_fingerprint_validation(self):
        for annotation in (_annotation(self.admission), self.elapsed()):
            for invalid in ([], {}, None, 0):
                with (
                    self.subTest(version=annotation.version, kind=type(invalid)),
                    self.assertRaisesRegex(HouseholdGroundTruthError, "review state") as error,
                ):
                    HouseholdEpisodeAnnotation.from_dict(
                        annotation.to_dict() | {"review_state": invalid}
                    )
                self.assertNotIn("fingerprint", str(error.exception))

    def test_boolean_review_flags_and_huge_numeric_values_fail_with_domain_errors(self):
        for annotation in (
            _annotation(self.admission, review_state="pending"),
            self.elapsed(review_state="pending"),
        ):
            with (
                self.subTest(version=annotation.version),
                self.assertRaises(HouseholdGroundTruthError),
            ):
                replace(annotation, human_verified=0)
        with self.assertRaises(HouseholdGroundTruthError):
            replace(self.elapsed(), source_duration_seconds=10**1000)
        with self.assertRaises(HouseholdGroundTruthError):
            HouseholdEpisodeRange(0.0, 10**1000, "person_present")

    def test_actual_vfr_duration_retains_late_positive_window_without_nominal_grid(self):
        annotation = self.elapsed()
        scan = _sampled_scan(
            self.admission,
            (
                (11.0, (), "container_pts"),
                (12.5, (("person", 0.9),), "container_pts"),
                (15.5, (("person", 0.9),), "container_pts"),
                (16.99, (), "container_pts"),
                (17.0, (("person", 0.9),), "container_pts"),
            ),
        )
        result = evaluate_household_sampled_coverage(
            annotation,
            scan,
            media_admission=self.admission,
            timeline_binding=self.binding(annotation, offset=10.0),
            reference_sha256=self.REFERENCE_SHA256,
            confidence_threshold=0.5,
        )
        self.assertEqual(2, result["positive_interval_count"])
        self.assertEqual(2, result["detected_positive_interval_count"])
        self.assertEqual(3, result["positive_sample_count"])
        self.assertEqual(1, result["outside_source_sample_count"])
        late = result["intervals"][-1]
        self.assertEqual((5.0, 7.0), (late["start_seconds"], late["end_seconds"]))
        self.assertEqual(2, late["actual_sample_count"])
        self.assertEqual("person_detected", late["state"])
        self.assertEqual(0.5, late["first_person_detection_delay_seconds"])
        self.assertFalse(result["danger_recognition_evaluated"])
        self.assertFalse(result["source_exhaustion_verified"])

    def test_v2_nominal_evaluation_is_explicitly_rejected(self):
        annotation = self.elapsed()
        scan = _sampled_scan(self.admission, ((0.0, (), "container_pts"),))
        with self.assertRaises(HouseholdGroundTruthError):
            evaluate_household_trigger_coverage(
                annotation,
                scan,
                media_admission=self.admission,
                thresholds=(0.5,),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )

    def test_v2_actual_evaluation_still_requires_explicit_matching_origin_binding(self):
        annotation = self.elapsed()
        scan = _sampled_scan(self.admission, ((15.5, (("person", 0.9),), "container_pts"),))
        for binding in (
            None,
            {},
            self.binding(annotation, offset=10.0) | {"alignment_confirmed": False},
            self.binding(annotation, offset=10.0)
            | {"annotation_sha256": _annotation(self.admission).fingerprint},
        ):
            with self.subTest(binding_type=type(binding)), self.assertRaises(ValueError):
                evaluate_household_sampled_coverage(
                    annotation,
                    scan,
                    media_admission=self.admission,
                    timeline_binding=binding,
                    reference_sha256=self.REFERENCE_SHA256,
                    confidence_threshold=0.5,
                )


if __name__ == "__main__":
    unittest.main()
