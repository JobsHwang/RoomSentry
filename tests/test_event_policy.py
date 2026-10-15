from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.domain import EventType, RiskLevel
from baby_monitor.event_policy import (
    SUPPRESSION_REASONS,
    EventPolicyDecision,
    EventPolicyError,
    HouseholdEventPolicy,
    PolicyEventInput,
    PolicyEventRule,
    evaluate_event_policy,
    policy_event_from_notification_message,
    policy_event_from_stored_event,
)
from baby_monitor.home_dashboard.rendering import render_home_dashboard
from baby_monitor.notification import ReviewStatus

BASE = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)  # 10:00 Asia/Shanghai


def _event(
    index: int,
    offset_seconds: float,
    *,
    camera: str = "camera_01",
    status: ReviewStatus = ReviewStatus.FINAL,
    event_type: EventType = EventType.POSSIBLE_RISK,
    risk: RiskLevel = RiskLevel.HIGH,
) -> PolicyEventInput:
    return PolicyEventInput(
        event_id=f"evt_{index:03d}",
        camera_id=camera,
        occurred_at=BASE + timedelta(seconds=offset_seconds),
        review_status=status,
        event_type=event_type,
        risk_level=risk,
    )


def _rule(**overrides: object) -> PolicyEventRule:
    values: dict[str, object] = {
        "rule_id": "risk_high",
        "event_types": frozenset({"possible_risk"}),
        "risk_levels": frozenset({"high"}),
        "review_statuses": frozenset({"final"}),
        "sink_ids": ("shadow_local",),
    }
    values.update(overrides)
    return PolicyEventRule(**values)


def _policy(*rules: PolicyEventRule, enabled: bool = True) -> HouseholdEventPolicy:
    return HouseholdEventPolicy(
        package_id="test_package",
        timezone="Asia/Shanghai",
        rules=rules,
        enabled=enabled,
    )


class HouseholdEventPolicyTests(unittest.TestCase):
    def test_valid_policy_round_trip_and_defaults(self) -> None:
        policy = _policy(_rule())
        disabled_default = HouseholdEventPolicy(
            package_id="x", timezone="Asia/Shanghai", rules=(_rule(),)
        )
        self.assertFalse(disabled_default.enabled)
        self.assertTrue(policy.enabled)
        restored = HouseholdEventPolicy.from_json(policy.to_json())
        self.assertEqual(policy, restored)
        schema = json.loads(
            Path("schemas/household-event-policy-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(policy.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy.write_new(path)
            self.assertEqual(policy, HouseholdEventPolicy.read(path))
            with self.assertRaises(FileExistsError):
                policy.write_new(path)

    def test_rejects_duplicate_rules_bad_enums_and_bad_timezone(self) -> None:
        with self.assertRaises(EventPolicyError):
            _policy(_rule(), _rule())
        with self.assertRaises(EventPolicyError):
            _rule(event_types=frozenset({"naps"}))
        with self.assertRaises(EventPolicyError):
            _rule(risk_levels=frozenset({"extreme"}))
        with self.assertRaises(EventPolicyError):
            _rule(review_statuses=frozenset({"draft"}))
        with self.assertRaises(EventPolicyError):
            HouseholdEventPolicy(package_id="x", timezone="Not/AZone", rules=(_rule(),))
        with self.assertRaises(EventPolicyError):
            _rule(quiet_hours_start="24:00")
        with self.assertRaises(EventPolicyError):
            _rule(aggregation_max_events=0)
        with self.assertRaises(EventPolicyError):
            _rule(sink_ids=())

    def test_rejects_tampered_fingerprint_and_duplicate_keys(self) -> None:
        policy = _policy(_rule())
        tampered = json.loads(policy.to_json())
        tampered["enabled"] = False
        with self.assertRaises(EventPolicyError):
            HouseholdEventPolicy.from_dict(tampered)
        with self.assertRaises(EventPolicyError):
            HouseholdEventPolicy.from_json('{"version": 1, "version": 1}')


class EventPolicyEvaluationTests(unittest.TestCase):
    def test_cooldown_anchors_only_on_routed_events(self) -> None:
        policy = _policy(_rule(cooldown_seconds=300.0))
        result = evaluate_event_policy(
            policy,
            (_event(0, 0), _event(1, 60), _event(2, 120), _event(3, 400)),
        )
        self.assertEqual(
            ["routed", "cooldown", "cooldown", "routed"],
            [item.reason for item in result.decisions],
        )

    def test_aggregation_caps_and_recovers_after_window(self) -> None:
        policy = _policy(_rule(aggregation_window_seconds=600.0, aggregation_max_events=2))
        result = evaluate_event_policy(
            policy,
            (_event(0, 0), _event(1, 60), _event(2, 120), _event(3, 5_000)),
        )
        self.assertEqual(
            ["routed", "routed", "aggregation_cap", "routed"],
            [item.reason for item in result.decisions],
        )

    def test_quiet_hours_wrap_overnight_and_policy_timezone(self) -> None:
        policy = _policy(_rule(quiet_hours_suppress=True))
        night = PolicyEventInput(
            event_id="night",
            camera_id="camera_01",
            occurred_at=datetime(2026, 8, 29, 15, 0, tzinfo=UTC),  # 23:00 local
            review_status=ReviewStatus.FINAL,
            event_type=EventType.POSSIBLE_RISK,
            risk_level=RiskLevel.HIGH,
        )
        result = evaluate_event_policy(policy, (night, _event(0, 0)))
        self.assertEqual(
            ["routed", "quiet_hours"],
            [item.reason for item in result.decisions],
        )

    def test_disabled_policy_and_no_match_suppress_without_routing(self) -> None:
        disabled = _policy(_rule(), enabled=False)
        result = evaluate_event_policy(disabled, (_event(0, 0),))
        self.assertEqual("policy_disabled", result.decisions[0].reason)
        self.assertEqual(0, result.routed_count)

        unmatched = _policy(_rule())
        result = evaluate_event_policy(unmatched, (_event(0, 0, event_type=EventType.UNKNOWN),))
        self.assertEqual("no_rule_match", result.decisions[0].reason)

    def test_camera_filter_and_first_match_wins(self) -> None:
        specific = _rule(rule_id="camera_only", cameras=frozenset({"camera_02"}))
        general = _rule(rule_id="any_camera")
        policy = _policy(specific, general)
        result = evaluate_event_policy(
            policy, (_event(0, 0, camera="camera_01"), _event(1, 0, camera="camera_02"))
        )
        self.assertEqual(
            [("any_camera", "evt_000"), ("camera_only", "evt_001")],
            [(item.rule_id, item.event_id) for item in result.decisions],
        )

    def test_events_are_sorted_chronologically(self) -> None:
        policy = _policy(_rule())
        result = evaluate_event_policy(policy, (_event(1, 60), _event(0, 0)))
        self.assertEqual(
            ["evt_000", "evt_001"],
            [item.event_id for item in result.decisions],
        )

    def test_decision_and_accounting_contracts(self) -> None:
        with self.assertRaises(EventPolicyError):
            EventPolicyDecision(event_id="x", action="route", rule_id=None, reason="routed")
        with self.assertRaises(EventPolicyError):
            EventPolicyDecision(
                event_id="x",
                action="suppress",
                rule_id=None,
                reason="no_rule_match",
                sink_ids=("s",),
            )
        policy = _policy(_rule())
        result = evaluate_event_policy(
            policy,
            (_event(0, 0), _event(1, 0, event_type=EventType.UNKNOWN)),
        )
        self.assertEqual(2, result.event_count)
        self.assertEqual(
            result.routed_count + sum(result.suppressed_counts.values()),
            result.event_count,
        )
        self.assertTrue(all(reason in SUPPRESSION_REASONS for reason in result.suppressed_counts))
        payload = result.to_dict()
        self.assertTrue(payload["routing_only"])
        self.assertFalse(payload["facts_modified"])
        self.assertFalse(payload["notification_delivery_authority"])

    def test_notification_message_conversion(self) -> None:
        message = type(
            "Message",
            (),
            {
                "event_id": "evt_x",
                "camera_id": "camera_01",
                "occurred_at": BASE,
                "review_status": ReviewStatus.FINAL,
                "event_type": EventType.MOVEMENT,
                "risk_level": RiskLevel.LOW,
            },
        )
        converted = policy_event_from_notification_message(message)
        self.assertEqual("evt_x", converted.event_id)
        self.assertEqual(EventType.MOVEMENT, converted.event_type)
        self.assertEqual(RiskLevel.LOW, converted.risk_level)

        stored = type(
            "Stored",
            (),
            {
                "event_id": "evt_stored",
                "camera_id": "camera_01",
                "occurred_at": BASE,
                "result": type(
                    "Result",
                    (),
                    {
                        "needs_remote_review": False,
                        "risk_level": RiskLevel.HIGH,
                        "event_type": EventType.POSSIBLE_RISK,
                    },
                ),
                "decision": type(
                    "Decision",
                    (),
                    {"needs_remote_review": False, "risk_level": RiskLevel.HIGH},
                ),
            },
        )
        stored_policy_event = policy_event_from_stored_event(stored)
        self.assertEqual(RiskLevel.HIGH, stored_policy_event.risk_level)
        stored.decision.risk_level = RiskLevel.LOW
        with self.assertRaisesRegex(EventPolicyError, "DecisionRecord"):
            policy_event_from_stored_event(stored)


class EventPolicyCLITests(unittest.TestCase):
    def test_cli_validate_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _policy(_rule(cooldown_seconds=60.0), enabled=True)
            policy_path = root / "policy.json"
            policy.write_new(policy_path)
            events = [
                _event(0, 0).to_dict(),
                _event(1, 30).to_dict(),
                _event(2, 0, event_type=EventType.UNKNOWN).to_dict(),
            ]
            events_path = root / "events.json"
            events_path.write_text(json.dumps(events), encoding="utf-8")
            output_path = root / "evaluation.json"

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["validate-event-policy", str(policy_path)])
            self.assertEqual(0, code)
            summary = json.loads(stdout.getvalue())
            self.assertEqual("test_package", summary["package_id"])
            self.assertEqual(1, summary["rule_count"])

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "evaluate-event-policy",
                        "--policy",
                        str(policy_path),
                        "--events",
                        str(events_path),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(0, code)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(3, payload["event_count"])
            self.assertEqual(1, payload["routed_count"])
            self.assertEqual(1, payload["suppressed_counts"]["cooldown"])
            self.assertEqual(1, payload["suppressed_counts"]["no_rule_match"])
            self.assertEqual(payload, json.loads(output_path.read_text(encoding="utf-8")))
            self.assertNotIn(str(root), stdout.getvalue())


class EventPolicyDashboardCardTests(unittest.TestCase):
    def test_card_renders_configured_unconfigured_and_invalid(self) -> None:
        policy = _policy(_rule(quiet_hours_suppress=True), enabled=False)
        html = render_home_dashboard(
            {
                "version": 1,
                "event_policy": {**policy.to_public_summary(), "status": "configured"},
            }
        ).decode("utf-8")
        self.assertIn("事件包路由策略", html)
        self.assertIn("1 条规则 · 未启用", html)
        self.assertIn("夜间静默", html)

        unconfigured = render_home_dashboard({"version": 1}).decode("utf-8")
        self.assertIn("未配置", unconfigured)

        invalid = render_home_dashboard(
            {"version": 1, "event_policy": {"status": "invalid"}}
        ).decode("utf-8")
        self.assertIn("策略文件无效", invalid)


if __name__ == "__main__":
    unittest.main()
