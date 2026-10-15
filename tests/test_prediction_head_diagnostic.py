from __future__ import annotations

import copy
import hashlib
import json
import unittest

from scripts.prediction_head_diagnostic import diagnose_prediction_heads


def _box(confidence=0.8):
    return {"confidence": confidence, "bbox_xyxy": (0.1, 0.1, 0.4, 0.8)}


class PredictionHeadDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.labels = tuple(
            {
                "review_id": f"review_{index}",
                "frame_sha256": str(index + 1) * 64,
                "presence": presence,
            }
            for index, presence in enumerate(("present", "absent", "unknown"))
        )
        self.left = self.branch("end2end", ((_box(0.4), _box(0.8)), (), (_box(0.8),)))
        self.right = self.branch("one_to_many", ((), (_box(0.8),), (_box(0.4),)))

    def branch(self, branch_id, boxes, *, model_id="model_a", model_sha="a" * 64):
        return {
            "branch_id": branch_id,
            "labels_source_sha256": "d" * 64,
            "source_confidence": 0.1,
            "rows": tuple(
                {
                    "model_id": model_id,
                    "model_sha256": model_sha,
                    "review_id": label["review_id"],
                    "frame_sha256": label["frame_sha256"],
                    "boxes": row_boxes,
                }
                for label, row_boxes in zip(self.labels, boxes, strict=True)
            ),
        }

    def diagnose(self, *, left=None, right=None, labels=None, thresholds=(0.1, 0.5)):
        return diagnose_prediction_heads(
            left if left is not None else self.left,
            right if right is not None else self.right,
            weak_labels=labels if labels is not None else self.labels,
            label_source_sha256="d" * 64,
            thresholds=thresholds,
        )

    def test_threshold_counts_preserve_weak_presence_denominators(self):
        result = self.diagnose()
        low, high = result["models"][0]["points"]
        first = low["branches"][0]
        second = high["branches"][1]
        self.assertEqual(3, first["stored_raw_box_count"])
        self.assertEqual(3, first["box_count_at_threshold"])
        self.assertEqual(2, high["branches"][0]["box_count_at_threshold"])
        self.assertEqual(1, first["weak_present_frame_count"])
        self.assertEqual(1, first["weak_absent_frame_count"])
        self.assertEqual(1, first["weak_unknown_frame_count"])
        self.assertEqual(0, first["present_no_box_count"])
        self.assertEqual(1, second["present_no_box_count"])
        self.assertEqual(1, second["absent_any_box_count"])
        self.assertEqual(1, second["unknown_no_box_count"])
        self.assertEqual(2, first["frames"][0]["box_count_at_threshold"])
        self.assertFalse(result["separate_person_coverage_verified"])

    def test_multiple_models_are_compared_separately_not_matched_across_models(self):
        left = copy.deepcopy(self.left)
        right = copy.deepcopy(self.right)
        left["rows"] += self.branch("unused", ((), (), ()), model_id="model_b", model_sha="b" * 64)[
            "rows"
        ]
        right["rows"] += self.branch(
            "unused", ((_box(),), (), ()), model_id="model_b", model_sha="b" * 64
        )["rows"]
        result = self.diagnose(left=left, right=right)
        self.assertEqual(["model_a", "model_b"], [model["model_id"] for model in result["models"]])
        self.assertEqual(
            0, result["models"][1]["points"][0]["branches"][0]["box_count_at_threshold"]
        )
        self.assertEqual(
            1, result["models"][1]["points"][0]["branches"][1]["box_count_at_threshold"]
        )
        self.assertEqual(3, result["fixed_review_frame_count"])

    def test_row_order_does_not_change_matching_or_counts(self):
        right = copy.deepcopy(self.right)
        right["rows"] = tuple(reversed(right["rows"]))
        self.assertEqual(self.diagnose()["models"], self.diagnose(right=right)["models"])

    def test_missing_duplicate_extra_and_cross_model_frames_fail_closed(self):
        cases = []
        for rows in (self.right["rows"][:-1], self.right["rows"] + (self.right["rows"][0],)):
            cases.append(self.right | {"rows": rows})
        different = copy.deepcopy(self.right)
        different["rows"][0]["model_id"] = "other_model"
        cases.append(different)
        for index, right in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.diagnose(right=right)

    def test_frame_weight_and_label_file_hash_bindings_must_match(self):
        for field, value in (("frame_sha256", "e" * 64), ("model_sha256", "e" * 64)):
            right = copy.deepcopy(self.right)
            right["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.diagnose(right=right)
        with self.assertRaises(ValueError):
            self.diagnose(right=self.right | {"labels_source_sha256": "e" * 64})
        with self.assertRaises(ValueError):
            self.diagnose(right=self.right | {"branch_id": self.left["branch_id"]})
        for field in ("model_id", "model_sha256"):
            right = copy.deepcopy(self.right)
            for row in right["rows"]:
                row[field] = "other_model" if field == "model_id" else "e" * 64
            with self.subTest(whole_branch_field=field), self.assertRaises(ValueError):
                self.diagnose(right=right)

    def test_thresholds_cannot_recover_unstored_scores(self):
        for levels in ((0.09,), (0.5, 0.5), (True,), (float("nan"),), (), (0.5,) * 33, [0.5]):
            with self.subTest(levels_type=type(levels)), self.assertRaises(ValueError):
                self.diagnose(thresholds=levels)
        with self.assertRaises(ValueError):
            self.diagnose(left=self.left | {"source_confidence": 0.5})
        # Equality at the requested threshold is included.
        self.assertEqual(
            3,
            self.diagnose(thresholds=(0.4,))["models"][0]["points"][0]["branches"][0][
                "box_count_at_threshold"
            ],
        )

    def test_invalid_boxes_and_finite_number_bounds_are_rejected(self):
        for box in (
            _box(True),
            _box(float("nan")),
            _box(0.01),
            _box(10**1000),
            {"confidence": 0.8, "bbox_xyxy": (0.1, 0.1, 0.1, 0.8)},
            {"confidence": 0.8, "bbox_xyxy": (-0.1, 0.1, 0.4, 0.8)},
            {"confidence": 0.8, "bbox_xyxy": (0.1, 0.1, 0.4, float("inf"))},
        ):
            left = copy.deepcopy(self.left)
            left["rows"][0]["boxes"] = (box,)
            with self.subTest(box_type=type(box)), self.assertRaises(ValueError):
                self.diagnose(left=left)

    def test_bounds_apply_to_total_boxes_rows_and_labels(self):
        left = copy.deepcopy(self.left)
        left["rows"][0]["boxes"] = (_box(),) * 1000
        with self.assertRaises(ValueError):
            self.diagnose(left=left)
        with self.assertRaises(ValueError):
            self.diagnose(left=self.left | {"rows": self.left["rows"] * 11})
        with self.assertRaises(ValueError):
            self.diagnose(labels=self.labels * 11)
        left["rows"][0]["boxes"] = (_box(),) * 999
        self.assertEqual(
            1000,
            self.diagnose(left=left)["models"][0]["points"][0]["branches"][0][
                "stored_raw_box_count"
            ],
        )

    def test_repeated_weak_frame_or_person_count_labels_are_not_accepted(self):
        with self.assertRaises(ValueError):
            self.diagnose(labels=self.labels + (self.labels[0],))
        labels = copy.deepcopy(self.labels)
        labels[1]["frame_sha256"] = labels[0]["frame_sha256"]
        with self.assertRaises(ValueError):
            self.diagnose(labels=labels)
        labels = copy.deepcopy(self.labels)
        labels[0]["visible_person_count"] = 2
        with self.assertRaises(ValueError):
            self.diagnose(labels=labels)

    def test_inputs_unchanged_hash_bound_and_no_ground_truth_or_winner_claim(self):
        before = copy.deepcopy((self.left, self.right, self.labels))
        result = self.diagnose()
        self.assertEqual(before, (self.left, self.right, self.labels))
        payload = {key: value for key, value in result.items() if key != "report_sha256"}
        expected = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        self.assertEqual(expected, result["report_sha256"])
        for name in (
            "ground_truth",
            "accuracy_evaluated",
            "baby_identity_evaluated",
            "box_localization_evaluated",
            "separate_person_coverage_verified",
            "deployment_approved",
            "winner_selected",
            "source_exhaustion_verified",
            "label_source_bytes_verified_by_helper",
            "media_bytes_verified_by_helper",
            "model_weights_verified_by_helper",
            "paths_disclosed",
        ):
            self.assertIs(False, result[name])


if __name__ == "__main__":
    unittest.main()
