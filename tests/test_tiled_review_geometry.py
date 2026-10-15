from __future__ import annotations

import unittest

from scripts.roi_review_geometry import map_crop_box, map_tile_box, tile_review_regions


class TiledReviewGeometryTests(unittest.TestCase):
    def test_fixed_order_and_outward_rounding(self):
        self.assertEqual(
            tile_review_regions(101, 51),
            [
                {"region_id": "full", "pixel_xyxy": [0, 0, 101, 51]},
                {"region_id": "top_left", "pixel_xyxy": [0, 0, 61, 31]},
                {"region_id": "top_right", "pixel_xyxy": [40, 0, 101, 31]},
                {"region_id": "bottom_left", "pixel_xyxy": [0, 20, 61, 51]},
                {"region_id": "bottom_right", "pixel_xyxy": [40, 20, 101, 51]},
            ],
        )

    def test_tiny_and_odd_images_have_full_coverage_and_equal_tile_sizes(self):
        for width in range(1, 20):
            for height in range(1, 20):
                regions = tile_review_regions(width, height)
                self.assertEqual(len({r["region_id"] for r in regions}), 5)
                covered = set()
                sizes = set()
                for region in regions[1:]:
                    x1, y1, x2, y2 = region["pixel_xyxy"]
                    sizes.add((x2 - x1, y2 - y1))
                    covered.update((x, y) for x in range(x1, x2) for y in range(y1, y2))
                self.assertEqual(len(sizes), 1)
                self.assertEqual(len(covered), width * height)

    def test_two_axis_mapping_and_internal_edges(self):
        region = tile_review_regions(101, 51)[4]
        result = map_tile_box(
            {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5},
            region=region,
            source_width=101,
            source_height=51,
        )
        self.assertEqual(result["bbox_xyxy"], [40 / 101, 20 / 51, 1, 1])
        self.assertEqual(result["internal_boundary_edges"], ["left", "top"])
        self.assertEqual(result["confidence"], 0.5)
        self.assertNotIn("presence", result)

    def test_all_tile_centers_map_from_real_pixel_bounds(self):
        for region in tile_review_regions(101, 51):
            x1, y1, x2, y2 = region["pixel_xyxy"]
            result = map_tile_box(
                {"bbox_xyxy": [0.25, 0.25, 0.75, 0.75], "confidence": 0.2},
                region=region,
                source_width=101,
                source_height=51,
            )
            self.assertEqual(
                result["bbox_xyxy"],
                [
                    (x1 + 0.25 * (x2 - x1)) / 101,
                    (y1 + 0.25 * (y2 - y1)) / 51,
                    (x1 + 0.75 * (x2 - x1)) / 101,
                    (y1 + 0.75 * (y2 - y1)) / 51,
                ],
            )
            self.assertFalse(result["truncated_at_internal_boundary"])

    def test_old_mapping_does_not_silently_accept_tiles(self):
        with self.assertRaises(ValueError):
            map_crop_box(
                {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5},
                region=tile_review_regions(100, 50)[1],
                source_width=100,
                source_height=50,
            )

    def test_forged_tile_rejected(self):
        region = tile_review_regions(100, 50)[1]
        region["pixel_xyxy"][2] -= 1
        with self.assertRaises(ValueError):
            map_tile_box(
                {"bbox_xyxy": [0, 0, 1, 1], "confidence": 0.5},
                region=region,
                source_width=100,
                source_height=50,
            )

    def test_invalid_dimensions_fail_closed(self):
        for value in (True, 0, -1, 1.0, float("nan"), 10**20):
            with self.assertRaises(ValueError):
                tile_review_regions(value, 50)


if __name__ == "__main__":
    unittest.main()
