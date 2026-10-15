import json
import unittest
from copy import deepcopy
from datetime import timedelta

from baby_monitor.live_people_view import supervision_view
from tests import test_live_people as people_fixtures
from tests.test_live_people import BASE


class PeopleViewTests(unittest.TestCase):
    def setUp(self):
        fixture = people_fixtures.LivePeopleTests()
        fixture.setUp()
        self.value = fixture.establish()
        self.sources = {
            "a": {"phase": "receiving", "generation": 0},
            "b": {"phase": "receiving", "generation": 0},
        }

    def view(self, *, value=None, age=0, now=1):
        return supervision_view(
            self.value if value is None else value,
            camera_sources=self.sources,
            heartbeat_age=age,
            now=BASE + timedelta(seconds=now),
        )

    def test_current_role_room_and_visible_posture(self):
        value = self.view()
        self.assertEqual("current_model_estimate", value["status"])
        self.assertEqual("room_a", value["current"]["room_id"])
        self.assertEqual("lying", value["current"]["visible_posture"])

    def test_transport_age_expires_current_detection(self):
        value = self.view(age=3, now=4)
        self.assertEqual("last_seen", value["status"])
        self.assertIsNone(value["current"])

    def test_utc_alone_can_expire_role(self):
        self.assertIsNone(self.view(now=33)["current"])

    def test_disconnected_or_new_generation_cannot_show_old_role(self):
        for source in (
            {"phase": "disconnected", "generation": 0},
            {"phase": "receiving", "generation": 1},
        ):
            self.sources["a"] = source
            self.assertIsNone(self.view()["current"])

    def test_closed_view_cannot_leak_model_prose_paths_or_credentials(self):
        value = deepcopy(self.value)
        value["secret"] = "synthetic-private-token"
        value["cameras"]["a"]["people"][0]["model_prose"] = "synthetic-private-token"
        value["cameras"]["a"]["people"][0]["evidence_ref"] = "/private/frame.png"
        serialized = json.dumps(self.view(value=value))
        self.assertNotIn("synthetic-private-token", serialized)
        self.assertNotIn("/private", serialized)

    def test_invalid_marker_sleep_claim_and_foreign_camera_fields(self):
        for field, invalid in (
            ("version", True),
            ("accuracy_verified", True),
            ("sleep_established", True),
            ("semantics", "facts"),
        ):
            value = deepcopy(self.value)
            value[field] = invalid
            self.assertIsNone(self.view(value=value))
        value = deepcopy(self.value)
        value["cameras"]["a"]["people"][0]["visible_posture"] = "sleeping"
        self.assertEqual("unknown", self.view(value=value)["current"]["visible_posture"])
