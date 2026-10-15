from __future__ import annotations

import copy
import unittest

from baby_monitor.live_people import LivePeople
from baby_monitor.local_runners.subject_state_probe import PROMPT_VERSION
from scripts.subject_layer_diagnostic import STRATA, diagnose, select_development_cases
from tests.test_live_people import BOX, OTHER, claims, observation


def record(boxes=((BOX,), (), (BOX,)), value=None, camera="camera_04"):
    people = LivePeople((camera,), association_diagnostics=True)
    people.session = "synthetic"
    rows = [observation(i, camera=camera, boxes=b, elapsed=i / 10) for i, b in enumerate(boxes)]
    for row in rows:
        people.observe(row)
    trace = people.review_trace(rows[0])
    return {
        "record_sha256": "a" * 64,
        "camera_id": camera,
        "detector_view": "original",
        "frame_generation": rows[0].frame.generation,
        "frame_sequence": rows[0].frame.sequence,
        "raw_bgr_sha256": rows[0].frame.sha256,
        "selected_at": rows[0].frame.received_at.isoformat(),
        "review_duration_ms": 1000,
        "people_trace": trace,
        "subject_result": {
            "version": 2,
            "prompt_revision": PROMPT_VERSION,
            "semantics": "unverified_model_claims",
            "frame_id": "synthetic.png",
            "input_sha256": "b" * 64,
            "coordinate_space": "submitted_image_thousandths",
            "claims": claims() if value is None else value,
        },
    }


class SubjectLayerDiagnosticTests(unittest.TestCase):
    def test_zero_then_return_is_mechanical_not_reidentified_baby(self):
        source = record()
        before = copy.deepcopy(source)
        report = diagnose(source)
        path = report["bonsai"]["claims"][0]["anonymous_origin_path"]
        self.assertEqual(1, path["first_missing"]["input_index"])
        self.assertEqual(0, path["first_missing"]["box_count"])
        self.assertEqual(2, path["first_same_id_return"]["input_index"])
        self.assertEqual(2, path["first_epoch_change"]["input_index"])
        self.assertIn(STRATA[4], report["strata"])
        self.assertTrue(report["tracking"]["actual_audits_reproduced"])
        self.assertEqual(source, before)
        self.assertIsNone(report["tracking"]["id_switches"])
        self.assertFalse(report["accuracy_measured"])
        self.assertFalse(report["deployment_approved"])
        self.assertFalse(report["pixels_verified_by_this_function"])

    def test_missing_origin_with_other_box_is_not_zero_frame(self):
        report = diagnose(record(boxes=((BOX,), (OTHER,))))
        path = report["bonsai"]["claims"][0]["anonymous_origin_path"]
        self.assertEqual(1, path["first_missing"]["box_count"])
        self.assertEqual(0, report["tracking"]["zero_box_inputs"])
        self.assertIsNone(path["first_same_id_return"])

    def test_missing_source_box_is_separate_from_tracking(self):
        report = diagnose(record(boxes=((), (BOX,))))
        claim = report["bonsai"]["claims"][0]
        self.assertEqual("no_detector_box_at_head", claim["spatial_diagnostic"])
        self.assertIsNone(claim["anonymous_origin_path"])
        self.assertIn(STRATA[0], report["strata"])
        self.assertIsNone(report["detector"]["false_negatives"])

    def test_outside_nonempty_and_null_point_are_distinct(self):
        outside = diagnose(record(value=claims(head=(800, 300))))
        null = diagnose(record(value=claims(head=None)))
        self.assertIn(STRATA[1], outside["strata"])
        self.assertEqual("head_not_located", null["bonsai"]["claims"][0]["spatial_diagnostic"])
        self.assertEqual([], null["strata"])

    def test_shared_box_is_not_two_independent_people(self):
        value = claims()
        value["people"].extend(claims("adult", head=(300, 300))["people"])
        report = diagnose(record(value=value))
        self.assertEqual(
            ["multiple_heads_in_one_box"] * 2,
            [c["spatial_diagnostic"] for c in report["bonsai"]["claims"]],
        )
        self.assertIn(STRATA[2], report["strata"])
        self.assertNotIn(STRATA[4], report["strata"])

    def test_overlap_does_not_force_nearest_box(self):
        report = diagnose(record(boxes=((BOX, (0.15, 0.15, 0.5, 0.9)),)))
        claim = report["bonsai"]["claims"][0]
        self.assertEqual("overlapping_detector_boxes", claim["spatial_diagnostic"])
        self.assertEqual([0, 1], claim["source_box_indices"])
        self.assertIsNone(claim["anonymous_origin_path"])

    def test_empty_model_claims_are_not_empty_room_reference(self):
        report = diagnose(
            record(boxes=((),), value={"people": [], "uncertainties": []}, camera="camera_01")
        )
        self.assertEqual([], report["bonsai"]["claims"])
        self.assertIsNone(report["detector"]["false_positives"])
        self.assertEqual(["camera_01_control"], report["strata"])

    def test_full_actual_audit_must_match_not_just_ids(self):
        source = record()
        source["people_trace"]["frames"][0]["association_audit"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            diagnose(source)

    def test_partial_changed_or_different_lane_fails_closed(self):
        for mutate in (
            lambda r: r.update(detector_view="rotation_180"),
            lambda r: r.update(frame_sequence=999),
            lambda r: r.update(camera_id="different"),
            lambda r: r.update(raw_bgr_sha256="c" * 64),
            lambda r: r.update(review_duration_ms=True),
            lambda r: r.update(review_duration_ms=float("nan")),
            lambda r: r.update(record_sha256="private/path"),
            lambda r: r.update(selected_at="2026-10-13"),
            lambda r: r["people_trace"].update(omitted_middle_frames=1),
            lambda r: r["people_trace"].update(status="review_frame_unavailable"),
            lambda r: r["people_trace"].pop("association_checkpoint"),
            lambda r: r["people_trace"]["frames"][0]["detections"][0].update(epoch=True),
            lambda r: r["subject_result"].update(semantics="human_truth"),
        ):
            source = record()
            mutate(source)
            with self.subTest(), self.assertRaises(ValueError):
                diagnose(source)

    def test_no_model_prose_leaks_to_diagnostic(self):
        source = record()
        source["subject_result"]["claims"]["people"][0]["visual_evidence"] = "private prose"
        report = diagnose(source)
        self.assertNotIn("private prose", str(report))
        self.assertNotIn("track_", str(report))

    def test_stable_fixed_strata_do_not_turn_controls_into_negatives(self):
        reports = [diagnose(record(boxes=((),), camera="camera_01")), diagnose(record())]
        reports[0]["png_sha256"] = "c" * 64
        a = select_development_cases(reports)
        self.assertEqual(a, select_development_cases(reports[::-1]))
        self.assertEqual(2, len(a["items"]))
        self.assertEqual(6, len(a["missing_strata"]))
        self.assertFalse(a["negative_ground_truth_inferred"])
        self.assertFalse(a["independent_holdout"])

    def test_same_image_not_duplicated_for_multiple_strata(self):
        reports = [diagnose(record()), diagnose(record(boxes=((),), camera="camera_01"))]
        selection = select_development_cases(reports)
        self.assertEqual(1, len(selection["items"]))
        self.assertEqual(1, selection["unique_image_count"])

    def test_selection_rejects_empty_unbounded_and_wrong_revision(self):
        valid = diagnose(record())
        for reports in ([], [valid] * 129, [valid | {"revision": "other"}]):
            with self.subTest(), self.assertRaises(ValueError):
                select_development_cases(reports)


if __name__ == "__main__":
    unittest.main()
