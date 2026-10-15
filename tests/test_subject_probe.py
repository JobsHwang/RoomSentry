from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor.domain import EventMedia, VLMEvent
from baby_monitor.local_runners import bonsai_27b as runner
from baby_monitor.local_runners import subject_probe as probe
from baby_monitor.local_runners.common import LocalVLMRequest, LocalVLMRunnerError


def claims(center=None, age="unknown"):
    return {
        "people": [{"head_center": center, "age_group": age, "visual_evidence": "局部可见"}],
        "uncertainties": ["单帧不能判断状态"],
    }


def envelope(value=None, raw=None):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": raw if raw is not None else json.dumps(value or claims())},
            }
        ]
    }


class SubjectProbeTests(unittest.TestCase):
    def test_unknown_head_and_role_remain_unknown(self):
        self.assertEqual(claims(), probe.parse_response(envelope()))

    def test_empty_people_is_allowed_not_proof_of_absence(self):
        value = {"people": [], "uncertainties": []}
        self.assertEqual(value, probe.parse_response(envelope(value)))

    def test_roles_and_integer_boundary_coordinates(self):
        for age in ("baby", "adult", "unknown"):
            value = claims([0, 1000], age)
            self.assertEqual(value, probe.parse_response(envelope(value)))

    def test_invalid_centers(self):
        for center in ([True, 1], [0.5, 2], [-1, 1], [1001, 2], [0], {}, "0,0"):
            with self.subTest(center=center), self.assertRaises(ValueError):
                probe.parse_response(envelope(claims(center)))

    def test_no_identity_sleep_confidence_or_additional_fields(self):
        for extra in ("identity", "sleeping", "confidence", "track_id"):
            value = claims()
            value["people"][0][extra] = "not permitted"
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                probe.parse_response(envelope(value))

    def test_limits_and_invalid_fields(self):
        variants = [claims(age="person"), claims(age=[]), {"people": [], "uncertainties": {}}]
        for field, value in (("visual_evidence", ""), ("visual_evidence", "x" * 241)):
            item = claims()
            item["people"][0][field] = value
            variants.append(item)
        variants.extend(
            [
                {"people": claims()["people"] * 9, "uncertainties": []},
                {"people": [], "uncertainties": ["x"] * 9},
                {"people": [], "uncertainties": ["x" * 241]},
                {"people": [], "uncertainties": [None]},
            ]
        )
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.parse_response(envelope(value))

    def test_strict_json(self):
        for raw in (
            '{"people":[],"people":[],"uncertainties":[]}',
            "NaN",
            '{"people":[],"uncertainties":[],"x":Infinity}',
            "```json\n{}\n```",
            "[" * 1500,
            "x" * 32769,
            "\ud800",
        ):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                probe.parse_response(envelope(raw=raw))

    def test_malformed_envelopes_do_not_raise_attribute_errors(self):
        for value in (
            None,
            [],
            {},
            {"choices": [None]},
            {"choices": [2]},
            {"choices": [{"finish_reason": "stop", "message": None}]},
            {"choices": [{"finish_reason": "stop", "message": []}]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.parse_response(value)

    def test_truncated_tool_call_and_refusal_rejected(self):
        for reason in ("length", "tool_calls", None, "error"):
            value = envelope()
            value["choices"][0]["finish_reason"] = reason
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                probe.parse_response(value)
        for field in ("tool_calls", "refusal"):
            value = envelope()
            value["choices"][0]["message"][field] = "present"
            with self.assertRaises(ValueError):
                probe.parse_response(value)

    def test_schema_returns_independent_values(self):
        schema = probe.schema()
        schema["properties"].clear()
        self.assertIn("people", probe.schema()["properties"])


class SubjectProbeRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.frame = root / "frame.png"
        self.frame.write_bytes(
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 32, 32)
        )
        (root / "manifest.json").write_text("{}")
        self.settings = runner.BonsaiSettings(root, root, root / "manifest.json", (root,))
        self.request = LocalVLMRequest(
            candidate={"detections": [{"kind": "baby"}], "signals": ["private_hint"]},
            evidence=EventMedia("", (str(self.frame),)),
            frame_paths=(self.frame,),
            question="private_question",
            output_schema=VLMEvent.json_schema(),
        )
        self.session = runner.BonsaiSession(self.settings)
        self.session.process = Mock()
        self.session.process.poll.return_value = None

    def test_explicit_probe_no_labels_question_or_default_change(self):
        baseline = runner.build_payload(self.settings, self.request)
        payload = runner.build_subject_probe_payload(self.settings, self.request)
        self.assertEqual(probe.PROMPT, payload["messages"][0]["content"][1]["text"])
        self.assertNotIn("private_hint", json.dumps(payload))
        self.assertNotIn("private_question", json.dumps(payload))
        self.assertEqual(
            baseline["messages"][0]["content"][0], payload["messages"][0]["content"][0]
        )
        self.assertFalse(payload["cache_prompt"])
        self.assertEqual(baseline, runner.build_payload(self.settings, self.request))

    def test_multiframe_rejected_before_http(self):
        request = replace(self.request, frame_paths=(self.frame, self.frame))
        with patch.object(runner, "_http") as http, self.assertRaises(LocalVLMRunnerError):
            self.session.probe_subjects(request)
        http.assert_not_called()

    def test_exact_sent_bytes_binding_survives_source_change(self):
        expected = hashlib.sha256(self.frame.read_bytes()).hexdigest()

        def answer(*_args):
            self.frame.write_bytes(b"changed after payload")
            return envelope(claims([500, 500], "baby"))

        with patch.object(runner, "_http", side_effect=answer) as http:
            result = self.session.probe_subjects(self.request)
        http.assert_called_once()
        self.assertEqual(expected, result["input_sha256"])
        self.assertEqual("unverified_model_claims", result["semantics"])
        self.assertEqual("submitted_image_thousandths", result["coordinate_space"])
        self.assertNotIn("confidence", json.dumps(result))
        self.assertNotIn("baby_location", result)

    def test_dead_model_no_call(self):
        self.session.process.poll.return_value = 1
        with patch.object(runner, "_http") as http, self.assertRaises(LocalVLMRunnerError):
            self.session.probe_subjects(self.request)
        http.assert_not_called()

    def test_bad_response_not_returned_as_claim(self):
        with (
            patch.object(runner, "_http", return_value=envelope(raw="private_invalid")),
            self.assertRaisesRegex(LocalVLMRunnerError, "^invalid subject probe response$"),
        ):
            self.session.probe_subjects(self.request)

    def test_post_call_server_exit_rejected(self):
        self.session.process.poll.side_effect = [None, 1]
        with (
            patch.object(runner, "_http", return_value=envelope()),
            self.assertRaises(LocalVLMRunnerError),
        ):
            self.session.probe_subjects(self.request)


if __name__ == "__main__":
    unittest.main()
