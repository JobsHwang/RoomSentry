from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.domain import EventType, RiskLevel
from baby_monitor.event_policy import (
    HouseholdEventPolicy,
    PolicyEventInput,
    PolicyEventRule,
)
from baby_monitor.event_policy_runtime import (
    EventPolicyDecisionStore,
    EventPolicyRuntimeError,
)
from baby_monitor.notification import ReviewStatus


def _rule(
    *,
    cooldown_seconds: float = 0.0,
    aggregation_window_seconds: float = 0.0,
    aggregation_max_events: int = 1,
) -> PolicyEventRule:
    return PolicyEventRule(
        rule_id="unknown_local",
        event_types=frozenset({"unknown"}),
        risk_levels=frozenset({"none"}),
        review_statuses=frozenset({"final"}),
        sink_ids=("shadow_local",),
        local_only=True,
        cameras=frozenset(),
        cooldown_seconds=cooldown_seconds,
        quiet_hours_suppress=False,
        quiet_hours_start="22:00",
        quiet_hours_end="07:00",
        aggregation_window_seconds=aggregation_window_seconds,
        aggregation_max_events=aggregation_max_events,
    )


def _policy(rule: PolicyEventRule, *, enabled: bool = True) -> HouseholdEventPolicy:
    return HouseholdEventPolicy(
        package_id="runtime_test",
        timezone="Asia/Shanghai",
        rules=(rule,),
        enabled=enabled,
    )


def _event(
    event_id: str,
    occurred_at: datetime,
    *,
    camera_id: str = "camera_01",
) -> PolicyEventInput:
    return PolicyEventInput(
        event_id=event_id,
        camera_id=camera_id,
        occurred_at=occurred_at,
        review_status=ReviewStatus.FINAL,
        event_type=EventType.UNKNOWN,
        risk_level=RiskLevel.NONE,
    )


class EventPolicyRuntimeTests(unittest.TestCase):
    def test_persistent_history_applies_cooldown_and_retry_reuses_original_decision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            policy = _policy(_rule(cooldown_seconds=300.0))
            started = datetime(2026, 8, 29, tzinfo=UTC)

            first = EventPolicyDecisionStore(database).decide_and_record(
                policy=policy,
                event=_event("event_01", started),
                now=started,
            )
            second = EventPolicyDecisionStore(database).decide_and_record(
                policy=policy,
                event=_event("event_02", started + timedelta(seconds=60)),
                now=started + timedelta(seconds=60),
            )
            third = EventPolicyDecisionStore(database).decide_and_record(
                policy=policy,
                event=_event("event_03", started + timedelta(seconds=301)),
                now=started + timedelta(seconds=301),
            )

            self.assertEqual("route", first.record.decision.action)
            self.assertEqual("cooldown", second.record.decision.reason)
            self.assertEqual("route", third.record.decision.action)
            self.assertFalse(first.reused_existing)

            disabled = _policy(_rule(cooldown_seconds=300.0), enabled=False)
            retried = EventPolicyDecisionStore(database).decide_and_record(
                policy=disabled,
                event=_event("event_01", started),
                now=started + timedelta(hours=1),
            )
            self.assertTrue(retried.reused_existing)
            self.assertEqual("route", retried.record.decision.action)
            self.assertEqual(policy.fingerprint, retried.record.policy_sha256)

            public = retried.to_public_dict()
            self.assertTrue(public["routing_only"])
            self.assertFalse(public["facts_modified"])
            self.assertFalse(public["notification_delivery_authority"])
            self.assertFalse(public["paths_disclosed"])

    def test_persistent_history_applies_sliding_aggregation_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            policy = _policy(
                _rule(
                    aggregation_window_seconds=600.0,
                    aggregation_max_events=2,
                )
            )
            started = datetime(2026, 8, 29, tzinfo=UTC)
            reasons = []
            for index, seconds in enumerate((0, 60, 120, 601), start=1):
                result = EventPolicyDecisionStore(database).decide_and_record(
                    policy=policy,
                    event=_event(f"event_{index:02d}", started + timedelta(seconds=seconds)),
                )
                reasons.append(result.record.decision.reason)
            self.assertEqual(
                ["routed", "routed", "aggregation_cap", "routed"],
                reasons,
            )

    def test_disabled_policy_is_persisted_as_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            policy = _policy(_rule(), enabled=False)
            result = EventPolicyDecisionStore(database).decide_and_record(
                policy=policy,
                event=_event("event_disabled", datetime(2026, 8, 29, tzinfo=UTC)),
            )
            self.assertEqual("suppress", result.record.decision.action)
            self.assertEqual("policy_disabled", result.record.decision.reason)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    1,
                    connection.execute("SELECT COUNT(*) FROM event_policy_decisions").fetchone()[0],
                )
                self.assertIsNone(
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE name = 'notification_jobs'"
                    ).fetchone()
                )

    def test_out_of_order_event_fails_closed_without_mutating_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            store = EventPolicyDecisionStore(database)
            policy = _policy(_rule(cooldown_seconds=300.0))
            started = datetime(2026, 8, 29, tzinfo=UTC)
            store.decide_and_record(
                policy=policy,
                event=_event("event_later", started + timedelta(seconds=60)),
            )

            with self.assertRaisesRegex(EventPolicyRuntimeError, "chronological"):
                store.decide_and_record(
                    policy=policy,
                    event=_event("event_earlier", started),
                )
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    1,
                    connection.execute("SELECT COUNT(*) FROM event_policy_decisions").fetchone()[0],
                )

    def test_independent_cameras_do_not_impose_a_global_arrival_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            store = EventPolicyDecisionStore(database)
            policy = _policy(_rule(cooldown_seconds=300.0))
            started = datetime(2026, 8, 29, tzinfo=UTC)

            later = store.decide_and_record(
                policy=policy,
                event=_event(
                    "event_camera_02_later",
                    started + timedelta(seconds=60),
                    camera_id="camera_02",
                ),
            )
            earlier = store.decide_and_record(
                policy=policy,
                event=_event(
                    "event_camera_01_earlier",
                    started,
                    camera_id="camera_01",
                ),
            )

            self.assertEqual("route", later.record.decision.action)
            self.assertEqual("route", earlier.record.decision.action)
            self.assertEqual("routed", earlier.record.decision.reason)

    def test_tampered_record_is_rejected_without_disclosing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            store = EventPolicyDecisionStore(database)
            policy = _policy(_rule())
            event = _event("event_tampered", datetime(2026, 8, 29, tzinfo=UTC))
            store.decide_and_record(policy=policy, event=event)
            with sqlite3.connect(database) as connection, connection:
                connection.execute(
                    "UPDATE event_policy_decisions SET occurred_at = ?",
                    ("2026-08-29T08:00:00+08:00",),
                )

            with self.assertRaises(EventPolicyRuntimeError):
                store.decide_and_record(policy=policy, event=event)

            with sqlite3.connect(database) as connection, connection:
                connection.execute(
                    """
                    UPDATE event_policy_decisions
                    SET occurred_at = ?, sink_ids_json = ?
                    """,
                    (event.occurred_at.isoformat(), '["private_sink"]'),
                )

            with self.assertRaises(EventPolicyRuntimeError) as raised:
                store.decide_and_record(policy=policy, event=event)
            self.assertNotIn("private", str(raised.exception).lower())


if __name__ == "__main__":
    unittest.main()
