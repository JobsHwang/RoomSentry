import unittest

from scripts.audit_alternate_review_gate import audit


class AlternateGateAuditTests(unittest.TestCase):
    def test_presence_keeps_weaker_same_frame_candidate(self):
        rows = [
            {"confidences": [0.6, 0.36], "truth_presence": True},
            {"confidences": [0.38], "truth_presence": False},
            {"confidences": [], "truth_presence": False},
        ]
        result = audit(rows, 0.4)
        group = result["groups"]["cuda:review"]
        self.assertEqual(group["raw_boxes"], 3)
        self.assertEqual(group["eligible_frame_boxes"], 2)
        self.assertEqual(group["true_eligible"], 1)
        self.assertEqual(group["false_eligible"], 0)
        self.assertEqual(group["withheld_nonempty"], 1)
        self.assertEqual(rows[0]["confidences"], [0.6, 0.36])

    def test_scale_and_cuda_shapes(self):
        result = audit(
            [
                {"shape": [384, 640], "rotation": 180, "boxes": [{"confidence": 0.4}]},
                {"hints": None},
            ],
            0.4,
        )
        self.assertEqual(result["groups"]["[384, 640]:180"]["eligible"], 1)
        self.assertEqual(result["groups"]["cuda:review"]["eligible"], 0)
        self.assertEqual(result["withheld"], [])
