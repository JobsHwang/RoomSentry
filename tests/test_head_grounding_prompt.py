import unittest

from baby_monitor.local_runners.head_grounding_prompt import build_prompt
from baby_monitor.local_runners.subject_probe import PROMPT


class HeadGroundingPromptTests(unittest.TestCase):
    def build(self, heads, source="a" * 64, target="a" * 64):
        return build_prompt(frame_sha256=source, head_input_sha256=target, heads=heads)

    def test_outward_rounding_preserves_small_edge_candidates(self):
        prompt = self.build([[0.0001, 0.9991, 0.0002, 1.0]])
        self.assertIn("[[0,999,1,1000]]", prompt)
        self.assertTrue(prompt.startswith(PROMPT))

    def test_empty_detector_output_keeps_full_scene_task(self):
        prompt = self.build([])
        self.assertIn("[]", prompt)
        self.assertIn("候选之外的人也要报告", prompt)

    def test_different_images_and_invalid_fingerprints_rejected(self):
        for source, target in (("a" * 64, "b" * 64), ("bad", "bad")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.build([], source, target)

    def test_labels_scores_and_private_text_cannot_enter_prompt(self):
        for value in (
            [{"bbox_xyxy": [0, 0, 1, 1], "age": "baby"}],
            [[0, 0, 1, 1, "private"]],
            "private",
            [[False, 0, 1, 1]],
            [[0, 0, float("nan"), 1]],
            [[0, 0, 2, 1]],
            [[1, 0, 0, 1]],
            [[0, 0, 1, 1]] * 9,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(value)
