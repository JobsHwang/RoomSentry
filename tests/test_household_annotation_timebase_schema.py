from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from baby_monitor.household_ground_truth import (
    HOUSEHOLD_RANGE_KINDS,
    HOUSEHOLD_SCENARIO_TAGS,
    MAX_HOUSEHOLD_ANNOTATION_RANGES,
    HouseholdEpisodeAnnotation,
    HouseholdEpisodeRange,
)

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

_SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


def _schema() -> dict[str, object]:
    return json.loads(
        (_SCHEMAS / "household-episode-annotation-v2.schema.json").read_text(encoding="utf-8")
    )


def _payload(*, approved: bool = False) -> dict[str, object]:
    return {
        "version": 2,
        "annotation_id": "annotation_synthetic",
        "media_admission_sha256": "a" * 64,
        "camera_id": "camera_01",
        "timebase": "elapsed_source_pts",
        "source_duration_seconds": 55.559044,
        "scenario_tags": ["daytime", "partial_occlusion"],
        "ranges": [{"start_seconds": 0, "end_seconds": 55.559044, "kind": "person_present"}],
        "review_state": "approved" if approved else "pending",
        "reviewers": ["reviewer_opaque01"] if approved else [],
        "human_verified": approved,
        "model_generated_labels": False,
        "source_media_included": False,
        "paths_disclosed": False,
        "subject_identity_inference": False,
        # Shape fixture only: canonical digest authenticity belongs to core.
        "annotation_sha256": "b" * 64,
    }


class HouseholdAnnotationTimebaseSchemaStructureTests(unittest.TestCase):
    def test_v2_fields_replace_only_the_nominal_frame_grid(self) -> None:
        old = json.loads(
            (_SCHEMAS / "household-episode-annotation-v1.schema.json").read_text(encoding="utf-8")
        )
        schema = _schema()
        expected = set(old["required"]) - {"source_fps", "source_frame_count"} | {"timebase"}
        self.assertEqual(expected, set(schema["required"]))
        self.assertEqual(expected, set(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(2, schema["properties"]["version"]["const"])
        self.assertEqual("elapsed_source_pts", schema["properties"]["timebase"]["const"])
        self.assertEqual(1, old["properties"]["version"]["const"])
        self.assertNotIn("timebase", old["properties"])

    def test_time_bounds_range_limits_and_enums_match_core_contract(self) -> None:
        schema = _schema()
        duration = schema["properties"]["source_duration_seconds"]
        self.assertEqual((0.01, 86400), (duration["minimum"], duration["maximum"]))
        ranges = schema["properties"]["ranges"]
        self.assertEqual(
            (1, MAX_HOUSEHOLD_ANNOTATION_RANGES), (ranges["minItems"], ranges["maxItems"])
        )
        range_properties = schema["$defs"]["range"]["properties"]
        for field in ("start_seconds", "end_seconds"):
            self.assertEqual(
                (0, 86400), (range_properties[field]["minimum"], range_properties[field]["maximum"])
            )
        self.assertEqual(HOUSEHOLD_RANGE_KINDS, tuple(range_properties["kind"]["enum"]))
        self.assertEqual(
            HOUSEHOLD_SCENARIO_TAGS,
            tuple(schema["properties"]["scenario_tags"]["items"]["enum"]),
        )

    def test_approval_requires_reviewers_and_preserves_privacy_constants(self) -> None:
        schema = _schema()
        reviewers = schema["properties"]["reviewers"]
        self.assertEqual(8, reviewers["maxItems"])
        self.assertTrue(reviewers["uniqueItems"])
        rules = {
            rule["if"]["properties"]["review_state"]["const"]: rule["then"]["properties"]
            for rule in schema["allOf"]
        }
        self.assertEqual({"minItems": 1}, rules["approved"]["reviewers"])
        self.assertIs(True, rules["approved"]["human_verified"]["const"])
        self.assertEqual({"maxItems": 0}, rules["pending"]["reviewers"])
        self.assertIs(False, rules["pending"]["human_verified"]["const"])
        for field in (
            "model_generated_labels",
            "source_media_included",
            "paths_disclosed",
            "subject_identity_inference",
        ):
            self.assertIs(False, schema["properties"][field]["const"])
            self.assertEqual("boolean", schema["properties"][field]["type"])

    def test_core_only_checks_are_explicit_not_claimed_by_schema(self) -> None:
        comment = _schema()["$comment"]
        for wording in (
            "Core validation",
            "non-overlapping",
            "SHA256",
            "origin",
            "source completeness",
        ):
            self.assertIn(wording, comment)


@unittest.skipUnless(
    Draft202012Validator, "optional jsonschema Draft 2020-12 validator unavailable"
)
class HouseholdAnnotationTimebaseSchemaValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        schema = _schema()
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)

    def assert_valid(self, payload: dict[str, object]) -> None:
        self.assertEqual([], list(self.validator.iter_errors(payload)))

    def assert_invalid(self, payload: dict[str, object]) -> None:
        self.assertTrue(list(self.validator.iter_errors(payload)), "invalid payload passed schema")

    def test_pending_and_approved_elapsed_pts_shapes_pass_without_fps(self) -> None:
        self.assert_valid(_payload())
        self.assert_valid(_payload(approved=True))
        for duration in (0.01, 86400):
            payload = _payload()
            payload["source_duration_seconds"] = duration
            payload["ranges"][0]["end_seconds"] = duration
            self.assert_valid(payload)

    def test_actual_core_v2_serialization_matches_the_schema(self) -> None:
        for approved in (False, True):
            annotation = HouseholdEpisodeAnnotation(
                annotation_id="annotation_synthetic",
                media_admission_sha256="a" * 64,
                camera_id="camera_01",
                source_fps=None,
                source_frame_count=None,
                source_duration_seconds=55.559044,
                scenario_tags=("daytime",),
                ranges=(HouseholdEpisodeRange(0, 55.559044, "person_present"),),
                review_state="approved" if approved else "pending",
                reviewers=("reviewer_opaque01",) if approved else (),
                human_verified=approved,
                version=2,
                timebase="elapsed_source_pts",
            )
            payload = annotation.to_dict()
            self.assert_valid(payload)
            self.assertEqual(annotation, HouseholdEpisodeAnnotation.from_dict(payload))
            self.assertNotIn("source_fps", payload)
            self.assertNotIn("source_frame_count", payload)

    def test_mixed_legacy_unknown_missing_and_wrong_timebase_fields_fail(self) -> None:
        for changes in (
            {"source_fps": 25},
            {"source_frame_count": 1200},
            {"private_path": "/private"},
            {"version": 1},
            {"version": True},
            {"timebase": "nominal_fps"},
            {"timebase": None},
        ):
            with self.subTest(changes=changes):
                self.assert_invalid(_payload() | changes)
        for field in _payload():
            payload = _payload()
            del payload[field]
            with self.subTest(missing=field):
                self.assert_invalid(payload)

    def test_review_state_and_anonymous_reviewer_constraints_fail_closed(self) -> None:
        for approved, changes in (
            (True, {"reviewers": []}),
            (True, {"human_verified": False}),
            (False, {"reviewers": ["reviewer_x"]}),
            (False, {"human_verified": True}),
            (True, {"reviewers": ["reviewer_x", "reviewer_x"]}),
            (True, {"reviewers": [f"reviewer_{index}" for index in range(9)]}),
            (True, {"reviewers": ["not_anonymous"]}),
            (True, {"reviewers": ["reviewer_x\n"]}),
            (True, {"reviewers": ["reviewer_" + "x" * 65]}),
            (True, {"human_verified": 1}),
            (False, {"review_state": "unknown"}),
        ):
            with self.subTest(approved=approved, changes=changes):
                self.assert_invalid(_payload(approved=approved) | changes)
        valid = _payload(approved=True)
        valid["reviewers"] = [f"reviewer_{index}" for index in range(8)]
        self.assert_valid(valid)

    def test_numeric_bounds_hashes_identifiers_and_privacy_are_strict(self) -> None:
        for field, values in {
            "source_duration_seconds": (0, 0.009, 86400.001, True, "1"),
            "annotation_id": ("", "x" * 129, "id\n", "/private"),
            "camera_id": ("id\n", "/private"),
            "annotation_sha256": ("a" * 63, "A" * 64, "a" * 64 + "\n"),
            "media_admission_sha256": ("a" * 65, "a" * 64 + "\n"),
            "model_generated_labels": (True, 0),
            "source_media_included": (True, 0),
            "paths_disclosed": (True, 0),
            "subject_identity_inference": (True, 0),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assert_invalid(_payload() | {field: value})

    def test_ranges_are_bounded_closed_objects_and_tags_are_controlled(self) -> None:
        for changes in (
            {"start_seconds": -1},
            {"end_seconds": 86400.01},
            {"start_seconds": True},
            {"kind": "sleeping"},
            {"private_note": "not allowed"},
        ):
            payload = _payload()
            payload["ranges"][0].update(changes)
            self.assert_invalid(payload)
        for ranges in ([], [_payload()["ranges"][0]] * 257):
            self.assert_invalid(_payload() | {"ranges": ranges})
        for tags in (["daytime", "daytime"], ["private_name"]):
            self.assert_invalid(_payload() | {"scenario_tags": tags})

    def test_schema_alone_does_not_claim_cross_field_or_hash_verification(self) -> None:
        payload = _payload()
        payload["ranges"] = [
            {"start_seconds": 4, "end_seconds": 5, "kind": "person_present"},
            {"start_seconds": 2, "end_seconds": 2, "kind": "empty_room"},
        ]
        payload["annotation_sha256"] = "0" * 64
        self.assert_valid(copy.deepcopy(payload))


if __name__ == "__main__":
    unittest.main()
