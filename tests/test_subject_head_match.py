import unittest

from baby_monitor.local_runners.subject_head_match import match_head_centers


class SubjectHeadMatchTests(unittest.TestCase):
    def match(self, centers, heads, **kwargs):
        return match_head_centers(
            probe_input_sha256="a" * 64,
            head_input_sha256=kwargs.get("digest", "a" * 64),
            centers=centers,
            heads=heads,
        )

    def test_two_separate_people(self):
        value = self.match([[200, 300], [800, 300]], [[0.1, 0.1, 0.4, 0.5], [0.6, 0.1, 0.9, 0.5]])
        self.assertEqual(
            [
                {"status": "unique_spatial_match", "head_indices": [0]},
                {"status": "unique_spatial_match", "head_indices": [1]},
            ],
            value["subjects"],
        )
        self.assertEqual("spatial_agreement_only", value["semantics"])

    def test_two_people_in_one_box_stay_ambiguous(self):
        value = self.match([[200, 300], [800, 300]], [[0, 0, 1, 1]])
        self.assertTrue(all(r["status"] == "ambiguous_head_match" for r in value["subjects"]))

    def test_one_point_in_two_boxes_stays_ambiguous(self):
        value = self.match([[500, 500]], [[0, 0, 1, 1], [0.3, 0.3, 0.7, 0.7]])
        self.assertEqual("ambiguous_head_match", value["subjects"][0]["status"])

    def test_null_and_unmatched_remain_distinct(self):
        value = self.match([None, [900, 900]], [[0, 0, 0.1, 0.1]])
        self.assertEqual(
            ["head_not_located", "no_head_match"], [r["status"] for r in value["subjects"]]
        )
        self.assertEqual([0], value["unmatched_head_indices"])

    def test_no_detector_heads_does_not_reject_vlm_person(self):
        self.assertEqual("no_head_match", self.match([[100, 100]], [])["subjects"][0]["status"])

    def test_mismatched_images_fail(self):
        for digest in ("b" * 64, "bad", None):
            with self.assertRaises(ValueError):
                self.match([], [], digest=digest)

    def test_invalid_points(self):
        for point in ([True, 1], [1.0, 2], [-1, 2], [2, 1001], [1], {}, "0,0"):
            with self.assertRaises(ValueError):
                self.match([point], [])

    def test_invalid_boxes(self):
        for box in (
            [0, 0, 0, 1],
            [0, 0, 1, float("nan")],
            [False, 0, 1, 1],
            [-0.1, 0, 1, 1],
            [0, 0, 2, 1],
            [],
        ):
            with self.assertRaises(ValueError):
                self.match([], [box])

    def test_limits(self):
        with self.assertRaises(ValueError):
            self.match([None] * 9, [])
        with self.assertRaises(ValueError):
            self.match([], [[0, 0, 1, 1]] * 4097)


if __name__ == "__main__":
    unittest.main()
