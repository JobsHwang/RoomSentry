"""Synthetic-only data separation and annotation export checks."""

import copy
import unittest

from scripts.household_finetune_dataset import person_labels, split_items, training_labels


def item(n, day, **extra):
    return dict(
        item_id=f"frame_{n}",
        camera_id="camera_02",
        captured_at=f"2026-10-{day:02d}T12:00:00+00:00",
        png_sha256=f"{n:064x}",
        bgr_sha256=f"{n + 100:064x}",
        **extra,
    )


def annotation(row, people=None):
    return dict(
        png_sha256=row["png_sha256"],
        bgr_sha256=row["bgr_sha256"],
        inventory_complete=True,
        visually_reviewed=True,
        uncertain_regions=[],
        provenance="human_visual_review",
        human_verified=True,
        ground_truth=True,
        reviewer_id="reviewer_fixture",
        negative_kind=None,
        people=people
        if people is not None
        else [
            dict(
                role="baby",
                bbox_xyxy=[0.2, 0.3, 0.4, 0.6],
                box_scope="visible_body",
                occlusion="partial",
            )
        ],
    )


class FineTunePreparationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [item(1, 12), item(2, 12), item(3, 12), item(4, 12), item(5, 11), item(6, 13)]
        self.fixed = [r["png_sha256"] for r in self.rows[:3]]

    def test_comparison_day_fully_excluded_and_latest_day_validation(self):
        original = copy.deepcopy(self.rows)
        plan = split_items(self.rows, self.fixed)
        self.assertEqual(
            [r["split"] for r in plan["items"]],
            ["comparison"] * 3 + ["quarantine", "train", "validation"],
        )
        self.assertEqual(self.rows, original)
        self.assertFalse(plan["training_ready"])
        self.assertTrue(all(r["annotation_status"] == "pending" for r in plan["items"]))

    def test_timezone_is_normalized(self):
        self.rows[3]["captured_at"] = "2026-10-13T00:01:00+08:00"
        self.assertEqual(split_items(self.rows, self.fixed)["items"][3]["split"], "quarantine")

    def test_identical_decoded_pixels_cannot_leak(self):
        self.rows[4]["bgr_sha256"] = self.rows[0]["bgr_sha256"]
        self.assertEqual(split_items(self.rows, self.fixed)["items"][4]["split"], "quarantine")

    def test_independent_negatives_are_not_training(self):
        result = split_items(self.rows, self.fixed, [self.rows[4]["png_sha256"]])
        self.assertEqual(result["items"][4]["split"], "independent_negative")
        self.assertFalse(result["items"][4]["negative_confirmed"])

    def test_duplicate_or_missing_sources_rejected(self):
        with self.assertRaises(ValueError):
            split_items(self.rows + [dict(self.rows[0], item_id="other")], self.fixed)
        with self.assertRaises(ValueError):
            split_items(self.rows[1:], self.fixed)

    def test_naive_clock_rejected(self):
        self.rows[0]["captured_at"] = "2026-10-12T12:00:00"
        with self.assertRaises(ValueError):
            split_items(self.rows, self.fixed)

    def test_no_training_when_only_comparison_and_validation(self):
        result = split_items(self.rows[:4] + self.rows[-1:], self.fixed)
        self.assertNotIn("train", result["split_counts"])

    def test_export_person_box_without_role_class_or_identity(self):
        row = dict(self.rows[4], split="train")
        self.assertEqual(
            person_labels(row, annotation(row)),
            "0 0.3000000000 0.4500000000 0.2000000000 0.3000000000\n",
        )

    def test_assistant_requires_explicit_training_and_cannot_evaluate(self):
        row = dict(self.rows[4], split="train")
        label = annotation(row)
        label.update(provenance="assistant_visual_review", human_verified=False, ground_truth=False)
        with self.assertRaises(ValueError):
            person_labels(row, label)
        self.assertTrue(person_labels(row, label, allow_assistant_training=True))
        for split in ("validation", "comparison", "independent_negative"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                person_labels(dict(row, split=split), label, allow_assistant_training=True)

    def test_model_output_and_forged_human_labels_rejected(self):
        row = dict(self.rows[4], split="train")
        label = annotation(row)
        for source in (
            "yolo_prediction",
            "bonsai_prediction",
            "pending",
            "assistant_visual_review",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                person_labels(row, dict(label, provenance=source), allow_assistant_training=True)

    def test_negative_must_be_explicit_and_visually_complete(self):
        row = dict(self.rows[4], split="train")
        label = annotation(row, [])
        with self.assertRaises(ValueError):
            person_labels(row, label)
        label["negative_kind"] = "no_visible_person"
        self.assertEqual(person_labels(row, label), "")
        for field, value in (
            ("inventory_complete", False),
            ("visually_reviewed", False),
            ("uncertain_regions", [[0, 0, 1, 1]]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                person_labels(row, dict(label, **{field: value}))

    def test_adult_only_is_not_empty_training_label(self):
        row = dict(self.rows[4], split="train")
        label = annotation(row)
        label["negative_kind"] = "adult_only"
        with self.assertRaises(ValueError):
            person_labels(row, label)
        label["people"][0]["role"] = "adult"
        self.assertTrue(person_labels(row, label))

    def test_invalid_or_duplicate_boxes_and_binding_rejected(self):
        row = dict(self.rows[4], split="train")
        label = annotation(row)
        for box in ([0.4, 0.3, 0.2, 0.6], [0, 0, float("nan"), 1], [True, 0, 1, 1], [0, 0, 2, 1]):
            trial = copy.deepcopy(label)
            trial["people"][0]["bbox_xyxy"] = box
            with self.subTest(box=box), self.assertRaises(ValueError):
                person_labels(row, trial)
        with self.assertRaises(ValueError):
            person_labels(row, dict(label, png_sha256="f" * 64))
        with self.assertRaises(ValueError):
            person_labels(row, dict(label, people=label["people"] * 2))

    def test_comparison_and_quarantine_are_never_training_by_accident(self):
        for split in ("comparison", "validation", "independent_negative", "quarantine"):
            row = dict(self.rows[4], split=split)
            with self.subTest(split=split), self.assertRaises(ValueError):
                training_labels(row, annotation(row), training_authorized=True)
        row = dict(self.rows[4], split="train")
        with self.assertRaises(ValueError):
            training_labels(row, annotation(row), training_authorized=False)
        self.assertTrue(training_labels(row, annotation(row), training_authorized=True))


if __name__ == "__main__":
    unittest.main()
