import copy
import json
import unittest
from pathlib import Path

from baby_monitor.local_runners.frame_alias import ALIAS, alias_payload, restore_event


class FrameAliasTests(unittest.TestCase):
    source = "20261005T123456Z_camera_04_0123456789abcdef.png"

    def payload(self):
        field = {"items": {"type": "string", "enum": [self.source]}}
        return {
            "messages": [
                {
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:unchanged"}},
                        {
                            "type": "text",
                            "text": "question\n帧 ID（输入顺序）：" + json.dumps([self.source]),
                        },
                    ]
                }
            ],
            "response_format": {
                "json_schema": {
                    "schema": {
                        "properties": {
                            "evidence_frames": copy.deepcopy(field),
                            "subjects": {
                                "items": {"properties": {"evidence_frames": copy.deepcopy(field)}}
                            },
                        }
                    }
                }
            },
            "max_tokens": 768,
        }

    def test_only_wire_references_change(self):
        payload = self.payload()
        before = copy.deepcopy(payload)
        result = alias_payload(payload, self.source)
        self.assertEqual(payload, before)
        self.assertEqual(result["max_tokens"], 768)
        self.assertEqual(result["messages"][0]["content"][0], payload["messages"][0]["content"][0])
        self.assertIn(ALIAS, result["messages"][0]["content"][1]["text"])
        self.assertNotIn(self.source, result["messages"][0]["content"][1]["text"])

    def test_wrong_binding_rejected(self):
        with self.assertRaises(ValueError):
            alias_payload(self.payload(), "wrong.png")
        value = self.payload()
        value["response_format"]["json_schema"]["schema"]["properties"]["evidence_frames"]["items"][
            "enum"
        ] = ["other.png"]
        with self.assertRaises(ValueError):
            alias_payload(value, self.source)

    def test_invalid_names_and_collision(self):
        for source in (ALIAS, "../secret", "a/b.png", "..", "", True):
            with self.assertRaises(ValueError):
                alias_payload(self.payload(), source)

    def event(self):
        value = json.loads(Path("tests/fixtures/valid_vlm_event.json").read_text())
        value["evidence_frames"] = [ALIAS]
        for subject in value["subjects"]:
            subject["evidence_frames"] = [ALIAS]
        return value

    def test_restore_only_evidence_not_observations(self):
        value = self.event()
        value["observations"] = [ALIAS]
        result = restore_event(json.dumps(value), self.source).to_dict()
        self.assertEqual(result["observations"], [ALIAS])
        self.assertEqual(result["evidence_frames"], [self.source])
        self.assertEqual(result["subjects"][0]["evidence_frames"], [self.source])

    def test_unexpected_output_reference_rejected(self):
        value = self.event()
        value["evidence_frames"] = [self.source]
        with self.assertRaises(ValueError):
            restore_event(json.dumps(value), self.source)
