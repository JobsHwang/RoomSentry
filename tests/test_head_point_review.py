from __future__ import annotations

import copy
import json
import unittest

from baby_monitor.head_point_review import (
    HeadPointReviewError,
    diagnose_point_coverage,
    fingerprint,
    parse_json,
    validate_manifest,
    validate_review,
)


def manifest():
    return {
        "version": 1,
        "scope": "frame_visible_head_points",
        "review_id": "review_opaque",
        "selection_purpose": "development_counterexamples",
        "blinding_scope": "page_hides_models_and_previous_answers",
        "items": [
            {
                "item_id": "frame_01",
                "camera_id": "camera_04",
                "frame_sha256": "a" * 64,
                "frame_received_at": "2026-10-03T00:17:23+00:00",
                "width": 848,
                "height": 480,
            }
        ],
    }


def review(m=None):
    return {
        "version": 1,
        "scope": "frame_visible_head_points",
        "manifest_sha256": fingerprint(m or manifest()),
        "reviewer_id": "reviewer_synthetic",
        "provenance": "synthetic",
        "answers": [
            {
                "item_id": "frame_01",
                "frame_sha256": "a" * 64,
                "decision": "marked",
                "points": [
                    {"point_id": "head_01", "point_xy": [0.25, 0.5], "role": "baby"},
                    {"point_id": "head_02", "point_xy": [0.75, 0.5], "role": "adult"},
                ],
                "inventory_complete": True,
                "reviewed": True,
            }
        ],
    }


class HeadPointReviewTests(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest()
        self.review = review()

    def predictions(self, boxes):
        return [{"item_id": "frame_01", "frame_sha256": "a" * 64, "boxes": boxes}]

    def diagnose(self, boxes, r=None):
        return diagnose_point_coverage(self.manifest, r or self.review, self.predictions(boxes))

    def test_json_roundtrip_and_stable_fingerprint(self):
        raw = json.dumps(self.manifest).encode()
        self.assertEqual(self.manifest, validate_manifest(parse_json(raw)))
        self.assertEqual(
            fingerprint(self.manifest), fingerprint(dict(reversed(list(self.manifest.items()))))
        )
        self.assertEqual(
            self.review,
            validate_review(parse_json(json.dumps(self.review).encode()), self.manifest),
        )

    def test_strict_json_rejects_duplicates_nonfinite_depth_bytes_and_size(self):
        for raw in (
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b'{"a":Infinity}',
            b"\xff",
            b"[" * 2000 + b"0" + b"]" * 2000,
            b"",
            b" " * (128 * 1024 + 1),
            "[]",
        ):
            with self.subTest(raw_type=type(raw)), self.assertRaises(HeadPointReviewError):
                parse_json(raw)

    def test_manifest_rejects_promoted_or_extra_fields(self):
        for change in (
            {"version": True},
            {"version": 2},
            {"scope": "identity"},
            {"selection_purpose": "holdout"},
            {"blinding_scope": "full_blinding"},
            {"review_id": "../private_name"},
            {"private_path": "private_name"},
            {"items": []},
        ):
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                validate_manifest(self.manifest | change)

    def test_manifest_item_bounds_types_hash_and_timestamp(self):
        for change in (
            {"width": True},
            {"width": 4097},
            {"height": 0},
            {"frame_sha256": "A" * 64},
            {"camera_id": "camera/private_name"},
            {"frame_received_at": "2026-10-03T00:00:00"},
            {"frame_received_at": "2026-10-03T08:00:00+08:00"},
            {"frame_received_at": []},
            {"extra": 1},
        ):
            value = copy.deepcopy(self.manifest)
            value["items"][0].update(change)
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                validate_manifest(value)

    def test_manifest_duplicate_frames_and_items_are_rejected(self):
        for change in ({}, {"item_id": "frame_02"}, {"frame_sha256": "b" * 64}):
            value = copy.deepcopy(self.manifest)
            value["items"].append(value["items"][0] | change)
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                validate_manifest(value)

    def test_review_requires_complete_ordered_bound_answers(self):
        for change in (
            {"version": True},
            {"scope": "live"},
            {"manifest_sha256": "b" * 64},
            {"reviewer_id": "personal_name"},
            {"provenance": "model"},
            {"answers": []},
            {"deployment_approved": True},
        ):
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                validate_review(self.review | change, self.manifest)
        for change in (
            {"item_id": "frame_02"},
            {"frame_sha256": "b" * 64},
            {"reviewed": 1},
            {"reviewed": False},
            {"inventory_complete": 1},
            {"decision": "empty_room"},
            {"points": []},
        ):
            value = copy.deepcopy(self.review)
            value["answers"][0].update(change)
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                validate_review(value, self.manifest)

    def test_review_role_point_coordinates_ids_and_duplicates_rejected(self):
        for change in (
            {"point_id": "head_02"},
            {"role": "parent"},
            {"role": []},
            {"point_xy": [True, 0.5]},
            {"point_xy": [float("nan"), 0.5]},
            {"point_xy": [float("inf"), 0.5]},
            {"point_xy": [10**400, 0.5]},
            {"point_xy": [-0.01, 0.5]},
            {"point_xy": [0.2, 1.01]},
            {"point_xy": [0.2]},
            {"point_xy": (0.2, 0.5)},
            {"identity": "private_name"},
        ):
            value = copy.deepcopy(self.review)
            value["answers"][0]["points"][0].update(change)
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                validate_review(value, self.manifest)
        value = copy.deepcopy(self.review)
        value["answers"][0]["points"][1]["point_xy"] = [0.25, 0.5]
        with self.assertRaises(HeadPointReviewError):
            validate_review(value, self.manifest)

    def test_unknown_is_not_zero_and_cannot_claim_inventory(self):
        value = copy.deepcopy(self.review)
        value["answers"][0].update(decision="unknown", points=[], inventory_complete=False)
        report = self.diagnose([[0, 0, 1, 1]], value)
        self.assertIsNone(report["rows"][0]["candidates_without_marked_point"])
        value["answers"][0]["inventory_complete"] = True
        with self.assertRaises(HeadPointReviewError):
            validate_review(value, self.manifest)

    def test_no_visible_head_is_not_whole_frame_person_absence(self):
        value = copy.deepcopy(self.review)
        value["answers"][0].update(decision="no_visible_head", points=[])
        report = self.diagnose([[0, 0, 1, 1]], value)
        self.assertEqual(1, report["rows"][0]["candidates_without_marked_point"])
        self.assertFalse(report["whole_frame_person_absence_established"])

    def test_distinct_candidates_map_unique_points_without_age_recognition(self):
        report = self.diagnose([[0.1, 0.3, 0.4, 0.7], [0.6, 0.3, 0.9, 0.7]])
        self.assertEqual(
            ["unique_point_containment"] * 2, [p["state"] for p in report["rows"][0]["points"]]
        )
        self.assertFalse(report["age_recognition_evaluated"])

    def test_one_mixed_box_does_not_localize_two_people(self):
        report = self.diagnose([[0, 0, 1, 1]])
        self.assertEqual(
            ["ambiguous_point_containment"] * 2, [p["state"] for p in report["rows"][0]["points"]]
        )

    def test_duplicate_boxes_are_ambiguous_not_extra_people(self):
        report = self.diagnose([[0.1, 0.3, 0.4, 0.7]] * 2)
        self.assertEqual(
            ["ambiguous_point_containment", "point_not_covered"],
            [p["state"] for p in report["rows"][0]["points"]],
        )

    def test_zero_box_rows_preserve_misses_but_missing_rows_fail(self):
        self.assertEqual(
            ["point_not_covered"] * 2, [p["state"] for p in self.diagnose([])["rows"][0]["points"]]
        )
        with self.assertRaises(HeadPointReviewError):
            diagnose_point_coverage(self.manifest, self.review, [])

    def test_incomplete_inventory_does_not_count_unmatched_candidates(self):
        value = copy.deepcopy(self.review)
        value["answers"][0]["inventory_complete"] = False
        report = self.diagnose([[0, 0, 0.1, 0.1]], value)
        self.assertIsNone(report["rows"][0]["candidates_without_marked_point"])

    def test_prediction_binding_coordinates_and_shape_fail_closed(self):
        for change in (
            {"item_id": "other"},
            {"frame_sha256": "b" * 64},
            {"private_path": "name"},
            {"boxes": [[0, 0, 0, 1]]},
            {"boxes": [[0, 0, 1, True]]},
            {"boxes": [[0, 0, 1, float("inf")]]},
            {"boxes": [[0, 0, 1, 1]] * 129},
        ):
            value = self.predictions([])
            value[0].update(change)
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                diagnose_point_coverage(self.manifest, self.review, value)

    def test_human_provenance_is_declared_not_authenticated_or_live(self):
        self.review["provenance"] = "human_user_review"
        report = self.diagnose([])
        self.assertTrue(report["human_provenance_declared"])
        for key in (
            "reviewer_authenticated",
            "formal_box_accuracy",
            "deployment_approved",
            "identity_or_live_state_authorized",
            "temporal_ground_truth",
        ):
            self.assertFalse(report[key])

    def test_error_messages_do_not_echo_paths_or_names(self):
        value = self.manifest | {"review_id": "private_name/private_path"}
        with self.assertRaises(HeadPointReviewError) as caught:
            validate_manifest(value)
        self.assertNotIn("private_name", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
