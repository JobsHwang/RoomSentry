from __future__ import annotations

import copy
import math
import unittest

from scripts.roi_review_geometry import MAX_DIMENSION, MAX_PIXELS, map_crop_box, review_regions


class RoiReviewGeometryTests(unittest.TestCase):
    def mapped(self, bbox=(0, 0, 1, 1), *, region="left", width=100, height=50, **kwargs):
        selected = next(r for r in review_regions(width, height) if r["region_id"] == region)
        return map_crop_box(
            {"bbox_xyxy": bbox, "confidence": 0.5},
            region=selected,
            source_width=width,
            source_height=height,
            **kwargs,
        )

    def test_regions_even_width(self):
        self.assertEqual(
            review_regions(100, 50),
            [
                {"region_id": "full", "pixel_xyxy": [0, 0, 100, 50]},
                {"region_id": "left", "pixel_xyxy": [0, 0, 60, 50]},
                {"region_id": "right", "pixel_xyxy": [40, 0, 100, 50]},
            ],
        )

    def test_odd_width_rounds_outward_and_has_equal_widths(self):
        for width in range(1, 100):
            with self.subTest(width=width):
                full, left, right = review_regions(width, 17)
                self.assertEqual(left["pixel_xyxy"][2], width - right["pixel_xyxy"][0])
                self.assertGreater(left["pixel_xyxy"][2], right["pixel_xyxy"][0])
                self.assertEqual(len({r["region_id"] for r in (full, left, right)}), 3)
        self.assertEqual(review_regions(101, 17)[1]["pixel_xyxy"], [0, 0, 61, 17])

    def test_dimensions_reject_bool_float_nan_and_huge_values(self):
        for invalid in (True, False, 0, -1, 1.0, "100", None, math.nan, math.inf, 10**1000):
            for width, height in ((invalid, 50), (100, invalid)):
                with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                    review_regions(width, height)
        with self.assertRaises(ValueError):
            review_regions(MAX_DIMENSION, MAX_DIMENSION)
        self.assertEqual(MAX_DIMENSION * 4096, MAX_PIXELS)
        self.assertEqual(review_regions(MAX_DIMENSION, 4096)[0]["pixel_xyxy"][2], MAX_DIMENSION)

    def test_full_image_mapping_preserves_box_and_ignores_outer_edges(self):
        result = self.mapped(region="full")
        self.assertEqual(result["bbox_xyxy"], [0, 0, 1, 1])
        self.assertFalse(result["truncated_at_internal_boundary"])
        self.assertEqual(result["internal_boundary_edges"], [])

    def test_left_and_right_mapping(self):
        self.assertEqual(self.mapped()["bbox_xyxy"], [0, 0, 0.6, 1])
        self.assertEqual(self.mapped(region="right")["bbox_xyxy"], [0.4, 0, 1, 1])
        self.assertEqual(self.mapped((0.25, 0.2, 0.75, 0.8))["bbox_xyxy"], [0.15, 0.2, 0.45, 0.8])

    def test_internal_seams_only(self):
        self.assertEqual(self.mapped()["internal_boundary_edges"], ["right"])
        self.assertEqual(self.mapped(region="right")["internal_boundary_edges"], ["left"])
        self.assertFalse(self.mapped((0, 0, 0.5, 1))["truncated_at_internal_boundary"])
        self.assertFalse(
            self.mapped((0.5, 0, 1, 1), region="right")["truncated_at_internal_boundary"]
        )

    def test_exact_margin_is_inclusive_without_cancellation(self):
        self.assertTrue(self.mapped((0, 0, 0.98, 1))["truncated_at_internal_boundary"])
        self.assertFalse(
            self.mapped((0, 0, math.nextafter(0.98, 0), 1))["truncated_at_internal_boundary"]
        )
        self.assertTrue(
            self.mapped((0.02, 0, 1, 1), region="right")["truncated_at_internal_boundary"]
        )
        self.assertFalse(
            self.mapped((math.nextafter(0.02, 1), 0, 1, 1), region="right")[
                "truncated_at_internal_boundary"
            ]
        )

    def test_zero_margin_marks_only_exact_seam(self):
        self.assertTrue(self.mapped(boundary_margin=0)["truncated_at_internal_boundary"])
        self.assertFalse(
            self.mapped((0, 0, math.nextafter(1, 0), 1), boundary_margin=0)[
                "truncated_at_internal_boundary"
            ]
        )

    def test_tiny_source_has_no_internal_seam(self):
        for region in ("full", "left", "right"):
            self.assertFalse(
                self.mapped(region=region, width=1, height=1)["truncated_at_internal_boundary"]
            )

    def test_mapping_uses_integer_crop_bounds_not_nominal_ratio(self):
        self.assertEqual(self.mapped(width=101)["bbox_xyxy"][2], 61 / 101)
        self.assertEqual(self.mapped(region="right", width=101)["bbox_xyxy"][0], 40 / 101)

    def test_invalid_boxes_rejected(self):
        for bbox in (
            None,
            [],
            [0, 0, 1],
            [0, 0, 1, 1, 1],
            [True, 0, 1, 1],
            [0, 0, math.nan, 1],
            [0, 0, math.inf, 1],
            [-0.1, 0, 1, 1],
            [0, 0, 1.1, 1],
            [0.5, 0, 0.5, 1],
            [0.6, 0, 0.5, 1],
            [0, 0.5, 1, 0.5],
            [0, 0, 1e-200, 1e-200],
        ):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                self.mapped(bbox)

    def test_mapping_rejects_representational_collapse(self):
        # Valid crop area collapses after adding the right crop's source offset.
        with self.assertRaises(ValueError):
            self.mapped((0, 0, 1e-100, 1), region="right")

    def test_invalid_margins_rejected(self):
        for margin in (True, math.nan, math.inf, -0.01, 1.01, "0.02"):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                self.mapped(boundary_margin=margin)

    def test_forged_region_rejected(self):
        for region in (
            {},
            {"region_id": "left", "pixel_xyxy": [0, 0, 59, 50]},
            {"region_id": "left", "pixel_xyxy": [False, 0, 60, 50]},
            {"region_id": "left", "pixel_xyxy": [0.0, 0, 60, 50]},
            {"region_id": "full", "pixel_xyxy": [0, 0, 60, 50]},
            {"region_id": "left", "pixel_xyxy": [0, 0, 60, 50], "extra": 1},
        ):
            with self.subTest(region=region), self.assertRaises(ValueError):
                map_crop_box(
                    {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.1},
                    region=region,
                    source_width=100,
                    source_height=50,
                )

    def test_metadata_preserved_deepcopied_and_not_relabelled(self):
        box = {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0, "keypoints": [[0.2, 0.3, 0.4]]}
        region = review_regions(100, 50)[1]
        original = copy.deepcopy(box)
        result = map_crop_box(box, region=region, source_width=100, source_height=50)
        self.assertEqual(box, original)
        self.assertEqual(result["keypoints"], box["keypoints"])
        result["keypoints"][0][0] = 0.7
        result["region"]["pixel_xyxy"][0] = 1
        self.assertEqual(box, original)
        self.assertEqual(region, review_regions(100, 50)[1])
        self.assertNotIn("true_person_count", result)
        self.assertNotIn("presence", result)

    def test_invalid_confidence_missing_fields_and_reserved_metadata(self):
        cases = [{}, {"bbox_xyxy": [0, 0, 1, 1]}]
        cases += [{"bbox_xyxy": [0, 0, 1, 1], "confidence": v} for v in (True, math.nan, -1, 2)]
        cases += [
            {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5, "region": {}},
            {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5, "nested": [math.nan]},
            {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5, "nested": object()},
        ]
        for box in cases:
            with self.subTest(box=box), self.assertRaises(ValueError):
                map_crop_box(
                    box, region=review_regions(100, 50)[1], source_width=100, source_height=50
                )

    def test_metadata_structural_limits(self):
        nested = []
        nested.append(nested)
        for metadata in (nested, [None] * 10_001):
            with self.subTest(cyclic=metadata is nested), self.assertRaises(ValueError):
                map_crop_box(
                    {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5, "extra": metadata},
                    region=review_regions(100, 50)[1],
                    source_width=100,
                    source_height=50,
                )


if __name__ == "__main__":
    unittest.main()
