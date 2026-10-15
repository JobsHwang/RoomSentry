from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from baby_monitor.head_point_review import fingerprint
from scripts.evaluate_subject_layers import bind_records, evaluate, main, run
from tests.test_live_people import BOX, OTHER, claims
from tests.test_subject_layer_diagnostic import record


def inputs(source=None, points=None, *, width=1000, height=500):
    source = record() if source is None else source
    manifest = {
        "version": 1,
        "scope": "frame_visible_head_points",
        "review_id": "review_synthetic",
        "selection_purpose": "development_counterexamples",
        "blinding_scope": "page_hides_models_and_previous_answers",
        "items": [
            {
                "item_id": "frame_01",
                "camera_id": source["camera_id"],
                "frame_sha256": source["subject_result"]["input_sha256"],
                "frame_received_at": source["people_trace"]["frames"][0]["received_at"],
                "width": width,
                "height": height,
            }
        ],
    }
    review = {
        "version": 1,
        "scope": manifest["scope"],
        "manifest_sha256": fingerprint(manifest),
        "reviewer_id": "reviewer_synthetic",
        "provenance": "synthetic",
        "answers": [
            {
                "item_id": "frame_01",
                "frame_sha256": manifest["items"][0]["frame_sha256"],
                "decision": "marked",
                "points": points
                or [
                    {
                        "point_id": "head_01",
                        "point_xy": [0.25, 0.3],
                        "role": "baby",
                    }
                ],
                "inventory_complete": True,
                "reviewed": True,
            }
        ],
    }
    return manifest, review, [source]


class SubjectLayerAcceptanceTests(unittest.TestCase):
    def evaluate(self, values, radius=20):
        return evaluate(*values, maximum_head_distance_pixels=radius)

    def test_visible_baby_reference_separates_correct_role_from_later_gap(self):
        values = inputs()
        prior = copy.deepcopy(values)
        result = self.evaluate(values)
        point = result["rows"][0]["points"][0]
        self.assertEqual("unique_point_containment", point["detector_state"])
        self.assertEqual("agrees_with_reference", point["bonsai_age_comparison"])
        self.assertEqual(1, point["anonymous_source_box_path"]["first_missing"]["input_index"])
        self.assertEqual(1, result["counts"]["reference_baby_heads"])
        self.assertEqual(1, result["counts"]["tracking_source_boxes_with_gap"])
        self.assertEqual(values, prior)
        self.assertEqual("synthetic", result["reference_provenance_declared"])
        self.assertIsNone(result["formal_person_box_recall"])
        self.assertIsNone(result["temporal_id_switches"])
        self.assertFalse(result["deployment_approved"])
        self.assertFalse(result["posture_reference_available"])

    def test_source_box_path_does_not_depend_on_bonsai_claim(self):
        values = inputs(record(value={"people": [], "uncertainties": []}))
        point = self.evaluate(values)["rows"][0]["points"][0]
        self.assertEqual("no_nearby_located_claim", point["bonsai_correspondence"])
        self.assertIsNotNone(point["anonymous_source_box_path"]["first_missing"])

    def test_zero_boxes_are_a_detector_point_miss_even_when_model_age_agrees(self):
        result = self.evaluate(inputs(record(boxes=((),))))
        point = result["rows"][0]["points"][0]
        self.assertEqual("point_not_covered", point["detector_state"])
        self.assertEqual("agrees_with_reference", point["bonsai_age_comparison"])
        self.assertIsNone(point["anonymous_source_box_path"])
        self.assertEqual(1, result["counts"]["baby_detector_point_not_covered"])

    def test_two_reference_heads_in_one_box_remain_detector_ambiguity(self):
        value = claims()
        value["people"] += claims("adult", head=(300, 600))["people"]
        points = [
            {"point_id": "head_01", "point_xy": [0.25, 0.3], "role": "baby"},
            {"point_id": "head_02", "point_xy": [0.3, 0.6], "role": "adult"},
        ]
        result = self.evaluate(inputs(record(boxes=((BOX,),), value=value), points))
        self.assertEqual(2, result["counts"]["detector_ambiguous_point_containment"])
        self.assertEqual(2, result["counts"]["bonsai_age_agrees_with_reference"])
        self.assertTrue(
            all(p["anonymous_source_box_path"] is None for p in result["rows"][0]["points"])
        )
        self.assertEqual(2, result["counts"]["detector_multiple_reference_heads_in_one_box"])

    def test_overlapping_boxes_do_not_pick_a_box_for_tracking(self):
        result = self.evaluate(inputs(record(boxes=((BOX, BOX),))))
        point = result["rows"][0]["points"][0]
        self.assertEqual("ambiguous_point_containment", point["detector_state"])
        self.assertEqual([0, 1], point["detector_box_indices"])
        self.assertEqual("overlapping_boxes_at_reference_head", point["detector_reason"])
        self.assertIsNone(point["anonymous_source_box_path"])

    def test_one_to_one_matching_refuses_nearest_choice_in_a_crowd(self):
        value = claims(head=(252, 300))
        value["people"] += claims("adult", head=(258, 300))["people"]
        result = self.evaluate(inputs(record(boxes=((BOX,),), value=value)))
        point = result["rows"][0]["points"][0]
        self.assertEqual("ambiguous_head_correspondence", point["bonsai_correspondence"])
        self.assertIsNone(point["bonsai_role"])
        self.assertEqual(0, result["counts"]["bonsai_age_agrees_with_reference"])

    def test_one_model_head_close_to_two_references_remains_unresolved(self):
        points = [
            {"point_id": "head_01", "point_xy": [0.25, 0.3], "role": "baby"},
            {"point_id": "head_02", "point_xy": [0.26, 0.3], "role": "adult"},
        ]
        result = self.evaluate(inputs(record(boxes=((BOX,),)), points))
        self.assertEqual(2, result["counts"]["bonsai_ambiguous_head_correspondence"])

    def test_age_mismatch_unknown_and_unlocated_claim_have_different_denominators(self):
        for role, head, expected in (
            ("adult", (250, 300), "disagrees_with_reference"),
            ("unknown", (250, 300), "model_age_unknown"),
            ("baby", None, "unresolved_correspondence"),
        ):
            with self.subTest(role=role, head=head):
                result = self.evaluate(inputs(record(value=claims(role, head=head))))
                self.assertEqual(expected, result["rows"][0]["points"][0]["bonsai_age_comparison"])

    def test_pixel_radius_uses_both_actual_image_dimensions(self):
        horizontal = inputs(record(value=claims(head=(280, 300))))
        vertical = inputs(record(value=claims(head=(250, 330))))
        self.assertEqual(
            "no_nearby_located_claim",
            self.evaluate(horizontal)["rows"][0]["points"][0]["bonsai_correspondence"],
        )
        self.assertAlmostEqual(
            15, self.evaluate(vertical)["rows"][0]["points"][0]["bonsai_head_distance_pixels"]
        )

    def test_unknown_or_incomplete_review_does_not_invent_absent_people(self):
        values = inputs(record(boxes=((BOX, OTHER),)))
        values[1]["answers"][0]["inventory_complete"] = False
        result = self.evaluate(values)
        self.assertEqual(1, result["counts"]["reference_heads"])
        self.assertEqual(0, result["counts"]["complete_visible_head_inventories"])
        values[1]["answers"][0].update(decision="unknown", points=[])
        result = self.evaluate(values)
        self.assertEqual(0, result["counts"]["reference_heads"])
        self.assertEqual(1, result["counts"]["unknown_frames"])
        self.assertIsNone(result["formal_person_box_precision"])

    def test_wrong_image_camera_time_duplicate_or_missing_records_reject(self):
        for change in ("hash", "camera", "time", "duplicate", "missing", "malformed"):
            values = inputs()
            if change == "hash":
                values[2][0]["subject_result"]["input_sha256"] = "c" * 64
            elif change == "camera":
                values[0]["items"][0]["camera_id"] = "camera_03"
            elif change == "time":
                values[0]["items"][0]["frame_received_at"] = "2026-10-13T00:00:00+00:00"
            elif change == "duplicate":
                values[2].append(copy.deepcopy(values[2][0]))
            elif change == "missing":
                values[2].clear()
            else:
                values[2].append({"subject_result": None})
            with self.subTest(change=change), self.assertRaises(ValueError):
                bind_records(values[0], values[2])

    def test_partial_trace_or_changed_actual_audit_rejects(self):
        for change in ("partial", "audit"):
            values = inputs()
            trace = values[2][0]["people_trace"]
            if change == "partial":
                trace["omitted_middle_frames"] = 1
            else:
                trace["frames"][0]["association_audit"]["extra"] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.evaluate(values)

    def test_radius_must_be_fixed_explicit_finite_and_bounded(self):
        for radius in (True, 0, 65, float("nan"), float("inf"), 10**400):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.evaluate(inputs(), radius)

    def test_model_prose_and_paths_are_not_exposed(self):
        values = inputs()
        values[2][0]["subject_result"]["claims"]["people"][0]["visual_evidence"] = "private prose"
        result = self.evaluate(values)
        self.assertNotIn("private prose", json.dumps(result))
        self.assertNotIn("synthetic.png", json.dumps(result))

    def test_assistant_visual_review_cannot_enter_human_acceptance_lane(self):
        values = inputs()
        values[1]["provenance"] = "current_assistant_visual_review"
        prior = copy.deepcopy(values)
        with self.assertRaises(ValueError):
            self.evaluate(values)
        self.assertEqual(values, prior)

    def test_cli_preflight_and_explicit_answers_use_private_new_outputs(self):
        values = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            blind = root / "blind-review"
            blind.mkdir(mode=0o700)
            for path, value in (
                (blind / "manifest.json", values[0]),
                (root / "source-records.json", {"records": values[2]}),
                (root / "answers.json", values[1]),
                (
                    blind / "package.json",
                    {
                        "synthetic": True,
                        "manifest_sha256": fingerprint(values[0]),
                        "item_count": 1,
                    },
                ),
                (
                    root / "image-bindings.json",
                    {
                        "items": [
                            {
                                "item_id": "frame_01",
                                "source_record_sha256": values[2][0]["record_sha256"],
                                "png_sha256": values[0]["items"][0]["frame_sha256"],
                                "bgr_sha256": values[2][0]["raw_bgr_sha256"],
                                "dimensions": [1000, 500],
                                "exact_png_bgr_verified": True,
                            }
                        ]
                    },
                ),
            ):
                path.write_text(json.dumps(value))
                path.chmod(0o600)
            self.assertEqual(
                "awaiting_independent_answers",
                run(
                    root,
                    root / "preflight",
                    maximum_head_distance_pixels=20,
                )["status"],
            )
            self.assertEqual(
                "evaluated_supplied_references",
                run(
                    root,
                    root / "result",
                    answers=root / "answers.json",
                    maximum_head_distance_pixels=20,
                )["status"],
            )
            with self.assertRaises(ValueError):
                run(root, root / "result", maximum_head_distance_pixels=20)
            self.assertEqual(0o600, (root / "result" / "report.json").stat().st_mode & 0o777)
            (root / "answers.json").write_text('{"version":1,"version":1}')
            self.assertEqual(
                2,
                main(
                    [
                        "--cohort",
                        str(root),
                        "--answers",
                        str(root / "answers.json"),
                        "--output",
                        str(root / "rejected"),
                        "--maximum-head-distance-pixels",
                        "20",
                    ]
                ),
            )
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
