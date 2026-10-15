import unittest

from baby_monitor.local_runners.head_probe import decode_head_rows


class HeadProbeTests(unittest.TestCase):
    def decode(self, scores=None, boxes=None, **kwargs):
        return decode_head_rows(
            [[0.8]] if scores is None else scores,
            [[0, 0, 48, 64, 240, 320]] if boxes is None else boxes,
            input_width=640,
            input_height=480,
            **kwargs,
        )

    def test_direct_resize_maps_by_input_not_original_pixels(self):
        box = self.decode()[0]
        self.assertEqual([0.1, 0.1, 0.5, 0.5], box["bbox_xyxy"])
        self.assertEqual("head", box["kind"])
        self.assertNotIn("age_group", box)

    def test_rotation_inverse(self):
        for rotation, expected in (
            (0, [0.1, 0.1, 0.5, 0.5]),
            (90, [0.1, 0.5, 0.5, 0.9]),
            (180, [0.5, 0.5, 0.9, 0.9]),
            (270, [0.5, 0.1, 0.9, 0.5]),
        ):
            with self.subTest(rotation=rotation):
                self.assertEqual(expected, self.decode(rotation_degrees=rotation)[0]["bbox_xyxy"])

    def test_clip_before_inverse(self):
        self.assertEqual(
            [0.0, 0.0, 1.0, 1.0], self.decode(boxes=[[0, 0, -2, -3, 500, 700]])[0]["bbox_xyxy"]
        )

    def test_empty_valid(self):
        self.assertEqual([], self.decode(scores=[], boxes=[]))

    def test_threshold_boundary_and_outside(self):
        self.assertEqual(1, len(self.decode(scores=[[0.35]])))
        self.assertEqual([], self.decode(scores=[[0.349]]))
        self.assertEqual([], self.decode(boxes=[[0, 0, 1, 650, 20, 680]]))

    def test_bad_scores(self):
        for score in (True, -0.1, 1.1, float("nan"), float("inf"), "0.8"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.decode(scores=[[score]])

    def test_bad_boxes_even_below_threshold(self):
        for box in (
            [1, 0, 1, 1, 4, 4],
            [0, 1, 1, 1, 4, 4],
            [0, 0, 4, 1, 1, 4],
            [0, 0, 1, 1, 4, 1],
            [0, 0, 1, 1, 4, float("nan")],
            [False, 0, 1, 1, 4, 4],
            [],
        ):
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.decode(scores=[[0.01]], boxes=[box])

    def test_mismatch_and_limit(self):
        for scores, boxes in (
            ([], [[0] * 6]),
            ([[0.8, 0.9]], [[0] * 6]),
            ([[0.8]] * 4097, [[0, 0, 1, 1, 4, 4]] * 4097),
        ):
            with self.assertRaises(ValueError):
                self.decode(scores=scores, boxes=boxes)

    def test_invalid_configuration(self):
        for threshold in (True, float("nan"), -1, 2):
            with self.assertRaises(ValueError):
                self.decode(confidence=threshold)
        with self.assertRaises(ValueError):
            self.decode(rotation_degrees=45)
        with self.assertRaises(ValueError):
            decode_head_rows([], [], input_width=True, input_height=480)


if __name__ == "__main__":
    unittest.main()
