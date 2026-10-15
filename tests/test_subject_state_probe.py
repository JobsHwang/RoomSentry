import copy
import json
import unittest
from unittest.mock import patch

from baby_monitor.local_runners import bonsai_27b as runner
from baby_monitor.local_runners import subject_state_probe as probe
from tests import test_subject_probe as subject_fixtures
from tests.test_live_people import claims


def response(value, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(value)}}]}


class SubjectStateProbeTests(unittest.TestCase):
    def test_complete_visible_state_and_exact_binding(self):
        result = {
            "version": 2,
            "prompt_revision": probe.PROMPT_VERSION,
            "semantics": "unverified_model_claims",
            "frame_id": "frame.png",
            "input_sha256": "a" * 64,
            "coordinate_space": "submitted_image_thousandths",
            "claims": claims(),
        }
        self.assertEqual(
            result, probe.validate_bound_result(result, frame_id="frame.png", input_sha256="a" * 64)
        )
        with self.assertRaises(ValueError):
            probe.validate_bound_result(result, frame_id="other.png", input_sha256="a" * 64)

    def test_sleep_identity_extra_fields_and_missing_posture_are_rejected(self):
        for field, value in (
            ("visible_posture", "sleeping"),
            ("visibility", "probably"),
            ("identity", "parent"),
            ("head_center", [True, 2]),
        ):
            body = claims()
            body["people"][0][field] = value
            with self.assertRaises(ValueError):
                probe.parse_response(response(body))
        body = claims()
        del body["people"][0]["visible_posture"]
        with self.assertRaises(ValueError):
            probe.parse_response(response(body))

    def test_truncation_duplicate_keys_and_nonfinite_values_fail(self):
        with self.assertRaises(ValueError):
            probe.parse_response(response(claims(), finish="length"))
        for raw in (
            '{"people":[],"people":[],"uncertainties":[]}',
            '{"people":[],"uncertainties":[NaN]}',
        ):
            value = response({})
            value["choices"][0]["message"]["content"] = raw
            with self.assertRaises(ValueError):
                probe.parse_response(value)

    def test_schema_has_required_posture_and_is_fresh(self):
        before = copy.deepcopy(probe.schema())
        changed = probe.schema()
        changed["properties"]["people"]["items"]["required"].clear()
        self.assertEqual(before, probe.schema())
        self.assertIn("visible_posture", before["properties"]["people"]["items"]["required"])


class SubjectStateRunnerTests(unittest.TestCase):
    setUp = subject_fixtures.SubjectProbeRunnerTests.setUp

    def test_independent_state_payload_and_bound_result_preserves_event_contract(self):
        baseline = runner.build_payload(self.settings, self.request)
        with patch.object(runner, "_http", return_value=response(claims())) as http:
            result = self.session.probe_subject_state(self.request)
        payload = http.call_args.args[1]
        self.assertNotIn("private_question", json.dumps(payload))
        self.assertNotIn("private_hint", json.dumps(payload))
        self.assertEqual(probe.PROMPT, payload["messages"][0]["content"][1]["text"])
        self.assertEqual(probe.PROMPT_VERSION, result["prompt_revision"])
        self.assertEqual("lying", result["claims"]["people"][0]["visible_posture"])
        self.assertEqual(baseline, runner.build_payload(self.settings, self.request))

    def test_state_truncation_is_rejected(self):
        with (
            patch.object(runner, "_http", return_value=response(claims(), "length")),
            self.assertRaises(subject_fixtures.LocalVLMRunnerError),
        ):
            self.session.probe_subject_state(self.request)
