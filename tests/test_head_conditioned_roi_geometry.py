from __future__ import annotations

import copy
import math
import unittest

from scripts.head_conditioned_roi_geometry import head_review_regions, map_head_roi_box


class HeadConditionedROIGeometryTests(unittest.TestCase):
    def test_fixed_pixel_sizes_and_integer_source_bounds(self):
        regions = head_review_regions([500, 500], 849, 481)
        self.assertEqual([328, 144, 520, 336], regions[0]["pixel_xyxy"])
        self.assertEqual([280, 96, 568, 384], regions[1]["pixel_xyxy"])
        self.assertEqual([232, 48, 616, 432], regions[2]["pixel_xyxy"])

    def test_edge_clamping_keeps_square_size_and_declared_target(self):
        for head in ([0, 0], [1000, 1000], [0, 1000], [1000, 0]):
            for region in head_review_regions(head, 848, 480):
                x1, y1, x2, y2 = region["pixel_xyxy"]
                self.assertEqual(region["side_pixels"], x2 - x1)
                self.assertEqual(x2 - x1, y2 - y1)
                self.assertTrue(x1 <= head[0] * 848 / 1000 <= x2)
                self.assertTrue(y1 <= head[1] * 480 / 1000 <= y2)

    def test_tiny_sources_have_explicit_coincident_regions(self):
        regions = head_review_regions([500, 500], 1, 2)
        self.assertEqual(3, len({r["region_id"] for r in regions}))
        self.assertEqual([[0, 1, 1, 2]] * 3, [r["pixel_xyxy"] for r in regions])

    def test_source_point_and_results_are_detached(self):
        head = [500, 500]
        regions = head_review_regions(head, 848, 480)
        regions[0]["head_center"][0] = 0
        self.assertEqual([500, 500], head)
        self.assertEqual([500, 500], regions[1]["head_center"])

    def test_full_local_box_maps_to_exact_integer_roi(self):
        for region in head_review_regions([500, 500], 848, 480):
            mapped = map_head_roi_box(
                [0, 0, 1, 1], region=region, source_width=848, source_height=480
            )
            self.assertEqual(
                [p / d for p, d in zip(region["pixel_xyxy"], (848, 480, 848, 480), strict=True)],
                mapped["bbox_xyxy"],
            )
            self.assertEqual(["left", "top", "right", "bottom"], mapped["internal_boundary_edges"])

    def test_rotation_maps_coordinates_without_rotating_head_reference(self):
        region = head_review_regions([500, 500], 848, 480)[0]
        got = map_head_roi_box(
            [0.1, 0.2, 0.4, 0.6], region=region, source_width=848, source_height=480, rotation=180
        )
        for actual, expected in zip(got["crop_bbox_xyxy"], [0.6, 0.4, 0.9, 0.8], strict=True):
            self.assertAlmostEqual(expected, actual)

    def test_source_edges_not_marked_as_internal_seams(self):
        region = head_review_regions([0, 0], 848, 480)[0]
        got = map_head_roi_box([0, 0, 1, 1], region=region, source_width=848, source_height=480)
        self.assertEqual(["right", "bottom"], got["internal_boundary_edges"])

    def test_no_boundary_flag_for_interior_box(self):
        region = head_review_regions([500, 500], 848, 480)[0]
        got = map_head_roi_box(
            [0.1, 0.1, 0.9, 0.9], region=region, source_width=848, source_height=480
        )
        self.assertEqual([], got["internal_boundary_edges"])

    def test_invalid_points_and_dimensions_are_rejected(self):
        for point in (None, [1], [True, 500], [0.5, 500], [-1, 500], [1001, 500]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                head_review_regions(point, 848, 480)
        for width, height in ((0, 480), (848, True), (848.0, 480), (65536, 65536)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                head_review_regions([500, 500], width, height)

    def test_region_tampering_and_extra_fields_are_rejected(self):
        valid = head_review_regions([500, 500], 848, 480)[0]
        for key, value in (
            ("side_pixels", 193),
            ("pixel_xyxy", [0, 0, 192, 192]),
            ("head_center", [0, 0]),
            ("extra", False),
        ):
            bad = copy.deepcopy(valid)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                map_head_roi_box([0, 0, 1, 1], region=bad, source_width=848, source_height=480)

    def test_invalid_box_and_rotation_fail_closed(self):
        region = head_review_regions([500, 500], 848, 480)[0]
        for box in ([0, 0, 0, 1], [0, 0, float("nan"), 1], [0, 0, True, 1], [-0.1, 0, 1, 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                map_head_roi_box(box, region=region, source_width=848, source_height=480)
        for rotation in (True, 90, 360, 180.0):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                map_head_roi_box(
                    [0, 0, 1, 1],
                    region=region,
                    source_width=848,
                    source_height=480,
                    rotation=rotation,
                )

    def test_region_coordinates_and_declared_size_require_exact_integer_types(self):
        valid = head_review_regions([0, 0], 848, 480)[0]
        for key, value in (
            ("side_pixels", 192.0),
            ("pixel_xyxy", [False, 0, 192, 192]),
            ("pixel_xyxy", [0.0, 0.0, 192.0, 192.0]),
        ):
            bad = copy.deepcopy(valid)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                map_head_roi_box([0, 0, 1, 1], region=bad, source_width=848, source_height=480)

    def test_inverse_rotation_and_source_mapping_reject_representational_collapse(self):
        region = head_review_regions([500, 500], 65536, 4096)[0]
        for box, rotation in (
            ([0.5, 0.1, math.nextafter(0.5, 1), 0.8], 0),
            ([0.0, 0.1, 1e-20, 0.8], 180),
        ):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                map_head_roi_box(
                    box, region=region, source_width=65536, source_height=4096, rotation=rotation
                )


if __name__ == "__main__":
    unittest.main()
