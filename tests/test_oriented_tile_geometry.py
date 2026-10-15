from __future__ import annotations

import copy
import math
import unittest

from scripts.oriented_tile_geometry import map_oriented_tile_box, map_oriented_tile_point
from scripts.roi_review_geometry import (
    MAX_DIMENSION,
    MAX_PIXELS,
    map_crop_box,
    map_tile_box,
    review_regions,
    tile_review_regions,
)


class OrientedTileGeometryTests(unittest.TestCase):
    def options(self, *, width=101, height=51, index=4, rotation=180):
        return {
            "region": tile_review_regions(width, height)[index],
            "source_width": width,
            "source_height": height,
            "rotation_degrees": rotation,
        }

    def mapped(self, bbox=(0.1, 0.2, 0.7, 0.9), confidence=0.35, **kwargs):
        return map_oriented_tile_box(
            {"bbox_xyxy": bbox, "confidence": confidence}, **self.options(**kwargs)
        )

    def test_default_zero_rotation_matches_existing_tile_mapping(self):
        box = {"bbox_xyxy": [0.1, 0.2, 0.7, 0.9], "confidence": 0.35}
        for region in tile_review_regions(101, 51):
            options = {"region": region, "source_width": 101, "source_height": 51}
            expected = map_tile_box(box, **options)
            result = map_oriented_tile_box(box, **options)
            self.assertEqual({k: result[k] for k in expected}, expected)
            self.assertEqual(result["rotation_degrees"], 0)
            self.assertEqual(result["rotated_tile_bbox_xyxy"], box["bbox_xyxy"])

    def test_rotation_happens_inside_tile_before_source_offset(self):
        result = self.mapped((0, 0, 0.25, 0.5))
        self.assertEqual(result["crop_bbox_xyxy"], [0.75, 0.5, 1, 1])
        self.assertEqual(result["bbox_xyxy"], [(40 + 0.75 * 61) / 101, 35.5 / 51, 1, 1])
        self.assertEqual(result["rotated_tile_bbox_xyxy"], [0, 0, 0.25, 0.5])
        self.assertEqual(result["coordinate_system"], "normalized source frame xyxy")

    def test_odd_dimensions_use_exact_integer_crop_not_nominal_ratio(self):
        self.assertEqual(self.mapped((0, 0, 1, 1))["bbox_xyxy"], [40 / 101, 20 / 51, 1, 1])
        point = map_oriented_tile_point(1, 1, **self.options())
        self.assertEqual(point, (40 / 101, 20 / 51))

    def test_box_corners_and_points_agree_for_every_tile_and_rotation(self):
        bbox = (0.1, 0.2, 0.7, 0.9)
        for rotation in (0, 180):
            for index in range(5):
                options = self.options(rotation=rotation, index=index)
                points = [
                    map_oriented_tile_point(x, y, **options)
                    for x in (bbox[0], bbox[2])
                    for y in (bbox[1], bbox[3])
                ]
                mapped = map_oriented_tile_box({"bbox_xyxy": bbox, "confidence": 0.5}, **options)
                self.assertEqual(
                    mapped["bbox_xyxy"],
                    [
                        min(x for x, _ in points),
                        min(y for _, y in points),
                        max(x for x, _ in points),
                        max(y for _, y in points),
                    ],
                )

    def test_full_source_rotates_without_internal_seams(self):
        result = self.mapped((0, 0, 0.25, 0.5), index=0)
        self.assertEqual(result["bbox_xyxy"], [0.75, 0.5, 1, 1])
        self.assertFalse(result["truncated_at_internal_boundary"])
        self.assertEqual(result["internal_boundary_edges"], [])

    def test_rotation_180_seams_are_named_in_original_tile_direction(self):
        result = self.mapped((0.5, 0.5, 1, 1))
        self.assertEqual(result["internal_boundary_edges"], ["left", "top"])
        result = self.mapped((0, 0, 0.5, 0.5), index=1)
        self.assertEqual(result["internal_boundary_edges"], ["right", "bottom"])

    def test_original_outer_edges_never_become_internal_seams_after_rotation(self):
        self.assertFalse(self.mapped((0, 0, 0.5, 0.5))["truncated_at_internal_boundary"])
        self.assertFalse(self.mapped((0.5, 0.5, 1, 1), index=1)["truncated_at_internal_boundary"])

    def test_rotation_180_upper_seam_margin_is_inclusive_without_cancellation(self):
        exact = self.mapped((0.5, 0.2, 0.98, 0.8))
        inside = self.mapped((0.5, 0.2, math.nextafter(0.98, 0), 0.8))
        self.assertEqual(exact["internal_boundary_edges"], ["left"])
        self.assertEqual(inside["internal_boundary_edges"], [])

    def test_rotation_180_lower_seam_margin_is_inclusive_without_cancellation(self):
        exact = self.mapped((0.02, 0.2, 0.5, 0.8), index=1)
        inside = self.mapped((math.nextafter(0.02, 1), 0.2, 0.5, 0.8), index=1)
        self.assertEqual(exact["internal_boundary_edges"], ["right"])
        self.assertEqual(inside["internal_boundary_edges"], [])

    def test_tiny_dimensions_and_duplicate_geometries_remain_valid(self):
        for width, height in ((1, 1), (1, 3), (3, 1), (2, 2)):
            for rotation in (0, 180):
                for index in range(5):
                    result = self.mapped(
                        (0, 0, 1, 1), width=width, height=height, index=index, rotation=rotation
                    )
                    self.assertEqual(result["region"], tile_review_regions(width, height)[index])
                    if width == height == 1:
                        self.assertEqual(result["bbox_xyxy"], [0, 0, 1, 1])
                        self.assertEqual(result["internal_boundary_edges"], [])

    def test_maximum_metadata_dimensions_do_not_allocate_pixels(self):
        self.assertEqual(MAX_DIMENSION * 4096, MAX_PIXELS)
        options = self.options(width=MAX_DIMENSION, height=4096)
        self.assertEqual(map_oriented_tile_point(0, 0, **options), (1, 1))
        result = map_oriented_tile_box({"bbox_xyxy": [0, 0, 1, 1], "confidence": 1}, **options)
        self.assertEqual(result["bbox_xyxy"][2:], [1, 1])

    def test_confidence_boundaries_are_not_adjusted(self):
        for confidence in (0, 0.1, 0.35, 1):
            self.assertEqual(self.mapped(confidence=confidence)["confidence"], confidence)

    def test_inputs_and_output_coordinate_lists_are_independent(self):
        box = {"bbox_xyxy": [0.1, 0.2, 0.7, 0.9], "confidence": 0.35}
        options = self.options()
        original = copy.deepcopy((box, options))
        result = map_oriented_tile_box(box, **options)
        result["rotated_tile_bbox_xyxy"][0] = 0
        result["crop_bbox_xyxy"][0] = 0
        result["region"]["pixel_xyxy"][0] = 0
        self.assertEqual((box, options), original)

    def test_invalid_dimensions_fail_for_both_interfaces(self):
        for value in (True, False, None, 0, -1, 1.0, "101", math.nan, math.inf, 10**400):
            for field in ("source_width", "source_height"):
                options = {**self.options(), field: value}
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        map_oriented_tile_point(0.5, 0.5, **options)
                    with self.assertRaises(ValueError):
                        map_oriented_tile_box(
                            {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5}, **options
                        )
        options = {**self.options(), "source_width": MAX_DIMENSION, "source_height": MAX_DIMENSION}
        with self.assertRaises(ValueError):
            map_oriented_tile_point(0.5, 0.5, **options)

    def test_only_strict_zero_or_180_rotation_is_accepted(self):
        class IntSubclass(int):
            pass

        for value in (
            True,
            False,
            None,
            0.0,
            180.0,
            "180",
            90,
            270,
            -180,
            360,
            math.nan,
            IntSubclass(0),
        ):
            options = {**self.options(), "rotation_degrees": value}
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    map_oriented_tile_point(0, 0, **options)
                with self.assertRaises(ValueError):
                    map_oriented_tile_box({"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5}, **options)

    def test_forged_region_is_rejected_by_both_interfaces(self):
        for region in (
            {},
            {"region_id": "bottom_right", "pixel_xyxy": [40, 20, 100, 51]},
            {"region_id": "full", "pixel_xyxy": [40, 20, 101, 51]},
            {"region_id": "bottom_right", "pixel_xyxy": [40.0, 20, 101, 51]},
            {"region_id": "top_left", "pixel_xyxy": [False, 0, 61, 31]},
            {"region_id": "bottom_right", "pixel_xyxy": (40, 20, 101, 51)},
            {**tile_review_regions(101, 51)[4], "extra": True},
        ):
            options = {**self.options(), "region": region}
            with self.subTest(region=region):
                with self.assertRaises(ValueError):
                    map_oriented_tile_point(0, 0, **options)
                with self.assertRaises(ValueError):
                    map_oriented_tile_box({"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5}, **options)

    def test_points_reject_invalid_coordinates_before_inverse_rotation(self):
        for value in (True, False, None, "0.5", -0.1, 1.1, math.nan, math.inf, 10**400):
            for point in ((value, 0.5), (0.5, value)):
                with self.subTest(point=point), self.assertRaises(ValueError):
                    map_oriented_tile_point(*point, **self.options())

    def test_invalid_bboxes_are_rejected_before_mapping(self):
        for bbox in (
            None,
            [],
            [0, 0, 1],
            [0, 0, 1, 1, 1],
            [True, 0, 1, 1],
            [0, 0, "1", 1],
            [0, 0, math.nan, 1],
            [0, 0, math.inf, 1],
            [0, 0, 10**400, 1],
            [-0.1, 0, 1, 1],
            [0, 0, 1.1, 1],
            [0.5, 0, 0.5, 1],
            [0.7, 0, 0.5, 1],
            [0, 0.5, 1, 0.5],
            [0, 0, 1e-200, 1e-200],
        ):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                self.mapped(bbox)

    def test_rotation_or_offset_representational_collapse_is_rejected(self):
        for rotation in (0, 180):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                self.mapped((0, 0, 1e-100, 1), rotation=rotation)

    def test_confidence_is_strict_finite_and_bounded(self):
        for value in (True, False, None, "0.5", -0.1, 1.1, math.nan, math.inf, 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.mapped(confidence=value)

    def test_extra_fields_and_untransformed_keypoints_are_rejected(self):
        for extra in (
            {"keypoints": [[0.1, 0.2, 0.9]]},
            {"label": "person"},
            {"rotation_degrees": 0},
            {"region": {}},
            {"confidence_extra": 0.9},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                map_oriented_tile_box(
                    {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5, **extra}, **self.options()
                )
        for box in ({}, {"bbox_xyxy": [0, 0, 1, 1]}, None, []):
            with self.assertRaises(ValueError):
                map_oriented_tile_box(box, **self.options())

    def test_old_interfaces_and_region_sets_are_unchanged(self):
        box = {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5}
        left = review_regions(101, 51)[1]
        with self.assertRaises(ValueError):
            map_oriented_tile_box(box, region=left, source_width=101, source_height=51)
        with self.assertRaises(ValueError):
            map_crop_box(
                box, region=tile_review_regions(101, 51)[1], source_width=101, source_height=51
            )
        self.assertEqual(
            [r["region_id"] for r in review_regions(101, 51)], ["full", "left", "right"]
        )
        self.assertEqual(
            map_tile_box(
                box, **{k: v for k, v in self.options().items() if k != "rotation_degrees"}
            )["bbox_xyxy"],
            [40 / 101, 20 / 51, 1, 1],
        )

    def test_output_contains_no_person_count_identity_or_presence_claim(self):
        result = self.mapped()
        for field in (
            "presence",
            "true_person_count",
            "baby_identity",
            "ground_truth",
            "deployment_approved",
            "keypoints",
            "subjects",
        ):
            self.assertNotIn(field, result)


if __name__ == "__main__":
    unittest.main()
