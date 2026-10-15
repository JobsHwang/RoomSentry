from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from baby_monitor.local_runners import bonsai_27b as runner
from baby_monitor.local_runners import scene_probe as probe
from baby_monitor.local_runners.common import LocalVLMRunnerError
from tests import test_subject_probe as subject_fixtures


def claims(person="yes", baby="unknown", adult="unknown"):
    return dict(
        person_visible=person,
        baby_visible=baby,
        adult_visible=adult,
        visual_evidence="可见局部身体，年龄不清",
    )


def envelope(value=None, raw=None):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": raw
                    if raw is not None
                    else json.dumps(value or claims(), ensure_ascii=False)
                },
            }
        ]
    }


class SceneProbeTests(unittest.TestCase):
    def test_visible_person_does_not_force_age(self):
        self.assertEqual(claims(), probe.parse_response(envelope()))

    def test_visible_baby_adult_and_no_are_separate(self):
        for value in (
            claims(baby="yes"),
            claims(adult="yes"),
            claims("no", "no", "no"),
            claims("unknown", "unknown", "unknown"),
        ):
            with self.subTest(value=value):
                self.assertEqual(value, probe.parse_response(envelope(value)))

    def test_presence_contradictions_are_rejected_not_repaired(self):
        for value in (
            claims("no", "yes", "no"),
            claims("unknown", "no", "yes"),
            claims("no", "unknown", "no"),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.parse_response(envelope(value))

    def test_no_geometry_identity_sleep_or_confidence(self):
        for field in ("head_center", "identity", "sleeping", "confidence", "baby_location"):
            value = claims()
            value[field] = "unsupported"
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.parse_response(envelope(value))

    def test_invalid_answers_and_text(self):
        for field, item in (
            ("baby_visible", True),
            ("adult_visible", []),
            ("person_visible", "absent"),
            ("visual_evidence", " "),
            ("visual_evidence", "x" * 241),
            ("visual_evidence", "\ud800"),
        ):
            value = claims()
            value[field] = item
            with self.subTest(field=field, item=repr(item)), self.assertRaises(ValueError):
                probe.parse_response(envelope(value))

    def test_duplicate_nonfinite_fences_and_overflow(self):
        for raw in (
            '{"person_visible":"yes","person_visible":"no"}',
            "NaN",
            "```json\n{}\n```",
            "[" * 1500,
            "x" * 32769,
            "\ud800",
        ):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                probe.parse_response(envelope(raw=raw))

    def test_malformed_and_incomplete_responses(self):
        for value in (
            None,
            {},
            [],
            {"choices": [None]},
            {"choices": [2]},
            {"choices": [{"finish_reason": "stop", "message": None}]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.parse_response(value)
        for field, item in (("finish_reason", "length"), ("finish_reason", "tool_calls")):
            value = envelope()
            value["choices"][0][field] = item
            with self.assertRaises(ValueError):
                probe.parse_response(value)
        for field in ("tool_calls", "refusal"):
            value = envelope()
            value["choices"][0]["message"][field] = "unsupported"
            with self.assertRaises(ValueError):
                probe.parse_response(value)

    def test_schema_values_are_fresh(self):
        value = probe.schema()
        value["properties"]["baby_visible"]["enum"].clear()
        self.assertEqual(list(probe.ANSWERS), probe.schema()["properties"]["baby_visible"]["enum"])


class SceneProbeRunnerTests(unittest.TestCase):
    setUp = subject_fixtures.SubjectProbeRunnerTests.setUp

    def test_scene_payload_hides_upstream_hints_without_changing_defaults(self):
        baseline = runner.build_payload(self.settings, self.request)
        subject = runner.build_subject_probe_payload(self.settings, self.request)
        value = runner.build_scene_probe_payload(self.settings, self.request)
        self.assertEqual(probe.PROMPT, value["messages"][0]["content"][1]["text"])
        self.assertEqual(probe.schema(), value["response_format"]["json_schema"]["schema"])
        self.assertEqual(baseline["messages"][0]["content"][0], value["messages"][0]["content"][0])
        self.assertNotIn("private_hint", json.dumps(value))
        self.assertNotIn("private_question", json.dumps(value))
        self.assertFalse(value["cache_prompt"])
        self.assertEqual(baseline, runner.build_payload(self.settings, self.request))
        self.assertEqual(subject, runner.build_subject_probe_payload(self.settings, self.request))

    def test_exact_submitted_bytes_are_bound(self):
        digest = hashlib.sha256(self.frame.read_bytes()).hexdigest()

        def answer(*_args):
            self.frame.write_bytes(b"changed after submission")
            return envelope(claims(baby="yes"))

        with patch.object(runner, "_http", side_effect=answer):
            value = self.session.probe_scene(self.request)
        self.assertEqual(digest, value["input_sha256"])
        self.assertEqual(probe.PROMPT_VERSION, value["prompt_revision"])
        self.assertEqual("unverified_model_claims", value["semantics"])
        self.assertNotIn("baby_location", value)

    def test_multiframe_and_dead_model_do_not_call_http(self):
        with patch.object(runner, "_http") as http:
            with self.assertRaises(LocalVLMRunnerError):
                self.session.probe_scene(
                    replace(self.request, frame_paths=(self.frame, self.frame))
                )
            self.session.process.poll.return_value = 1
            with self.assertRaises(LocalVLMRunnerError):
                self.session.probe_scene(self.request)
            http.assert_not_called()

    def test_invalid_or_truncated_response_not_converted_to_absence(self):
        for response in (
            envelope(raw="private_bad_output"),
            {"choices": [{"finish_reason": "length"}]},
        ):
            with (
                patch.object(runner, "_http", return_value=response),
                self.assertRaisesRegex(LocalVLMRunnerError, "^invalid scene probe response$"),
            ):
                self.session.probe_scene(self.request)

    def test_post_call_exit_rejected(self):
        self.session.process.poll.side_effect = [None, 1]
        with (
            patch.object(runner, "_http", return_value=envelope()),
            self.assertRaises(LocalVLMRunnerError),
        ):
            self.session.probe_scene(self.request)


if __name__ == "__main__":
    unittest.main()
