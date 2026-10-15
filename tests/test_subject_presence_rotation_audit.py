import unittest

from scripts.audit_subject_presence_rotation import score


class SubjectPresenceRotationAuditTests(unittest.TestCase):
    def matrix(self):
        frames = [
            {"sha256": "a" * 64, "human_presence": "present"},
            {"sha256": "b" * 64, "human_presence": "absent"},
        ]
        records = []
        for index in range(2):
            for rotation in (0, 180):
                people = (
                    []
                    if rotation
                    else [
                        {"age_group": "unknown", "head_center": None, "visual_evidence": "visible"}
                    ]
                )
                records.append(
                    {
                        "index": index,
                        "rotation": rotation,
                        "source_sha256": frames[index]["sha256"],
                        "claims": {"people": people, "uncertainties": []},
                    }
                )
        return frames, records

    def test_all_four_outcomes(self):
        result = score(*self.matrix())
        self.assertEqual(
            result["metrics"]["0"], {"tp": 1, "fn": 0, "fp": 1, "tn": 0, "error_indices": [1]}
        )
        self.assertEqual(
            result["metrics"]["180"], {"tp": 0, "fn": 1, "fp": 0, "tn": 1, "error_indices": [0]}
        )
        self.assertFalse(result["age_accuracy_verified"])

    def test_missing_and_duplicate_not_scored(self):
        frames, records = self.matrix()
        for changed in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                score(frames, changed)

    def test_binding_and_label_mismatch(self):
        frames, records = self.matrix()
        records[0]["source_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            score(frames, records)
        frames, records = self.matrix()
        frames[0]["human_presence"] = True
        with self.assertRaises(ValueError):
            score(frames, records)

    def test_failed_model_output_not_negative(self):
        frames, records = self.matrix()
        records[0]["claims"] = None
        with self.assertRaises(ValueError):
            score(frames, records)

    def test_boolean_indices_and_rotation_rejected(self):
        for field in ("index", "rotation"):
            frames, records = self.matrix()
            records[0][field] = False
            with self.assertRaises(ValueError):
                score(frames, records)
