import json
import unittest
from copy import deepcopy
from datetime import UTC, datetime

from baby_monitor.subject_diagnostic import build, validate


class SubjectDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 5, 12, tzinfo=UTC)
        self.value = dict(
            version=1,
            scope="historical_model_claims",
            camera_id="camera_04",
            rows=[
                dict(
                    received_at="2026-10-05T08:00:00+00:00",
                    subjects=[dict(age_claim="baby", spatial_status="unique_spatial_match")],
                )
            ],
        )

    def test_history_age_does_not_become_current_state(self):
        result = validate(self.value, now=self.now)
        self.assertEqual(14400, result["rows"][0]["age_seconds"])
        self.assertNotIn("baby_location", result)
        self.assertEqual("historical_model_claims", result["scope"])

    def test_extra_private_fields_rejected(self):
        for target in (self.value, self.value["rows"][0], self.value["rows"][0]["subjects"][0]):
            target["raw_text"] = "private"
            with self.assertRaises(ValueError):
                validate(self.value, now=self.now)
            del target["raw_text"]

    def test_invalid_enums_and_version(self):
        for key, value in (("version", True), ("scope", "live"), ("camera_id", "bedroom-private")):
            item = deepcopy(self.value)
            item[key] = value
            with self.assertRaises(ValueError):
                validate(item, now=self.now)
        for key, value in (("age_claim", "mother"), ("spatial_status", "verified_identity")):
            item = deepcopy(self.value)
            item["rows"][0]["subjects"][0][key] = value
            with self.assertRaises(ValueError):
                validate(item, now=self.now)

    def test_future_naive_invalid_times(self):
        for stamp in ("2026-10-06T00:00:00+00:00", "2026-10-05T08:00:00", "bad"):
            self.value["rows"][0]["received_at"] = stamp
            with self.assertRaises(ValueError):
                validate(self.value, now=self.now)

    def test_empty_subject_list_preserved(self):
        self.value["rows"][0]["subjects"] = []
        self.assertEqual([], validate(self.value, now=self.now)["rows"][0]["subjects"])

    def test_row_and_subject_limits(self):
        self.value["rows"] *= 9
        with self.assertRaises(ValueError):
            validate(self.value, now=self.now)
        self.value["rows"] = self.value["rows"][:1]
        self.value["rows"][0]["subjects"] *= 9
        with self.assertRaises(ValueError):
            validate(self.value, now=self.now)

    def inputs(self):
        capture = {"frames": [{"sha256": "a" * 64, "received_at": "2026-01-01T00:00:00+00:00"}]}
        probe = {
            "records": [
                {
                    "index": 0,
                    "source_sha256": "a" * 64,
                    "result": {
                        "version": 1,
                        "semantics": "unverified_model_claims",
                        "coordinate_space": "submitted_image_thousandths",
                        "input_sha256": "a" * 64,
                        "claims": {
                            "people": [
                                {
                                    "age_group": "baby",
                                    "head_center": [500, 500],
                                    "visual_evidence": "private model text not exported",
                                }
                            ],
                            "uncertainties": ["private"],
                        },
                    },
                }
            ]
        }
        heads = {
            "records": [
                {
                    "index": 0,
                    "rotation": 180,
                    "source_sha256": "a" * 64,
                    "heads": [{"bbox_xyxy": [0, 0, 1, 1]}],
                }
            ]
        }
        return capture, {"camera_id": "camera_04"}, probe, heads

    def test_export_contains_no_model_prose(self):
        value = build(*self.inputs())
        self.assertNotIn("private", json.dumps(value))
        self.assertEqual("unique_spatial_match", value["rows"][0]["subjects"][0]["spatial_status"])

    def test_mismatch_and_duplicates_rejected(self):
        values = self.inputs()
        values[2]["records"][0]["source_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            build(*values)
        values = self.inputs()
        values[3]["records"] *= 2
        with self.assertRaises(ValueError):
            build(*values)

    def test_wrong_coordinate_space_rejected(self):
        values = self.inputs()
        values[2]["records"][0]["result"]["coordinate_space"] = "other_crop"
        with self.assertRaises(ValueError):
            build(*values)


if __name__ == "__main__":
    unittest.main()
