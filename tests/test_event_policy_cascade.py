from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.domain import (
    DecisionRecord,
    EventType,
    FinalSource,
    ReviewStatus,
    ReviewTier,
    RiskLevel,
)
from baby_monitor.event_policy import HouseholdEventPolicy, PolicyEventRule
from baby_monitor.event_policy_cascade import (
    EventPolicyCascadeCase,
    EventPolicyCascadeError,
    EventPolicyCascadeReport,
    evaluate_event_policy_cascade,
    event_policy_cascade_cases_from_json,
)

BASE = datetime(2026, 8, 29, 2, 0, tzinfo=UTC)


def _decision(
    *,
    review_tier: ReviewTier = ReviewTier.LOCAL_PRIMARY,
    remote_allowed: bool = False,
    remote_used: bool = False,
    final_source: FinalSource = FinalSource.LOCAL_PRIMARY,
    contradictions: tuple[str, ...] = (),
) -> DecisionRecord:
    return DecisionRecord(
        risk_level=RiskLevel.HIGH,
        review_tier=review_tier,
        review_reasons=("elevated_risk",) if review_tier is ReviewTier.REMOTE else (),
        contradictions=contradictions,
        remote_allowed=remote_allowed,
        remote_used=remote_used,
        final_source=final_source,
    )


def _case(
    index: int,
    offset_seconds: int,
    *,
    decision: DecisionRecord | None = None,
    result_needs_remote_review: bool = False,
) -> EventPolicyCascadeCase:
    return EventPolicyCascadeCase(
        event_id=f"cascade_{index:02d}",
        camera_id="camera_01",
        occurred_at=BASE + timedelta(seconds=offset_seconds),
        event_type=EventType.POSSIBLE_RISK,
        result_needs_remote_review=result_needs_remote_review,
        decision=decision or _decision(),
    )


def _policy(*, enabled: bool = True) -> HouseholdEventPolicy:
    return HouseholdEventPolicy(
        package_id="cascade_test",
        timezone="Asia/Shanghai",
        enabled=enabled,
        rules=(
            PolicyEventRule(
                rule_id="high_risk",
                event_types=frozenset({"possible_risk"}),
                risk_levels=frozenset({"high"}),
                review_statuses=frozenset({"final", "pending_review"}),
                sink_ids=("shadow_local",),
                cooldown_seconds=300.0,
            ),
        ),
    )


def _report_fingerprint(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("report_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class EventPolicyCascadeTests(unittest.TestCase):
    def test_case_derives_risk_and_review_status_from_completed_decision(self) -> None:
        final = _case(1, 0)
        pending_decision = _case(
            2,
            60,
            decision=_decision(review_tier=ReviewTier.REMOTE),
        )
        pending_result = _case(3, 120, result_needs_remote_review=True)

        self.assertEqual(RiskLevel.HIGH, final.to_policy_event().risk_level)
        self.assertEqual(ReviewStatus.FINAL, final.review_status)
        self.assertEqual(ReviewStatus.PENDING_REVIEW, pending_decision.review_status)
        self.assertEqual(ReviewStatus.PENDING_REVIEW, pending_result.review_status)
        self.assertEqual(final, EventPolicyCascadeCase.from_dict(final.to_dict()))

    def test_replay_cross_tabs_decision_state_and_policy_routing(self) -> None:
        remote_pending = _decision(review_tier=ReviewTier.REMOTE)
        remote_complete = _decision(
            review_tier=ReviewTier.REMOTE,
            remote_allowed=True,
            remote_used=True,
            final_source=FinalSource.REMOTE,
            contradictions=("risk_level_disagreement",),
        )
        report = evaluate_event_policy_cascade(
            _policy(),
            (
                _case(3, 400, decision=remote_complete),
                _case(1, 0),
                _case(2, 60, decision=remote_pending),
            ),
        )

        self.assertEqual(
            ["cascade_01", "cascade_02", "cascade_03"],
            [item.case.event_id for item in report.items],
        )
        self.assertEqual(
            ["routed", "cooldown", "routed"],
            [item.routing.reason for item in report.items],
        )
        self.assertEqual(3, report.case_count)
        self.assertEqual(2, report.routed_count)
        self.assertEqual({"cooldown": 1}, report.suppressed_counts)
        self.assertEqual(3, sum(outcome.count for outcome in report.outcomes))
        self.assertTrue(any(outcome.has_contradictions for outcome in report.outcomes))
        self.assertTrue(report.offline_replay_only)
        self.assertFalse(report.facts_modified)
        self.assertFalse(report.risk_inference)
        self.assertFalse(report.review_decision_authority)
        self.assertFalse(report.notification_delivery_authority)
        self.assertFalse(report.delivery_performed)
        self.assertFalse(report.runtime_state_read)
        self.assertFalse(report.source_media_read)
        self.assertFalse(report.source_media_included)
        self.assertFalse(report.notification_outbox_written)
        self.assertFalse(report.network_used)
        self.assertFalse(report.subject_identity_inference)
        self.assertFalse(report.paths_disclosed)

        disabled = evaluate_event_policy_cascade(
            _policy(enabled=False),
            (_case(1, 0),),
        )
        self.assertEqual(0, disabled.routed_count)
        self.assertEqual({"policy_disabled": 1}, disabled.suppressed_counts)

    def test_report_round_trip_and_nested_tamper_refusal(self) -> None:
        report = evaluate_event_policy_cascade(_policy(), (_case(1, 0),))
        restored = EventPolicyCascadeReport.from_json(report.to_json())
        self.assertEqual(report, restored)

        tampered = report.to_dict()
        tampered["items"][0]["case"]["decision"]["risk_level"] = "low"
        tampered["report_sha256"] = _report_fingerprint(tampered)
        with self.assertRaisesRegex(EventPolicyCascadeError, "DecisionRecord digest"):
            EventPolicyCascadeReport.from_dict(tampered)

        wrong_sink_shape = report.to_dict()
        wrong_sink_shape["items"][0]["routing"]["sink_ids"] = "shadow_local"
        wrong_sink_shape["report_sha256"] = _report_fingerprint(wrong_sink_shape)
        with self.assertRaisesRegex(EventPolicyCascadeError, "sink_ids"):
            EventPolicyCascadeReport.from_dict(wrong_sink_shape)

        with self.assertRaises(EventPolicyCascadeError):
            EventPolicyCascadeReport.from_json('{"version":1,"version":1}')

    def test_input_is_strict_bounded_and_requires_unique_event_ids(self) -> None:
        fixture = Path("tests/fixtures/event_policy_cascade_cases.json").read_text(encoding="utf-8")
        cases = event_policy_cascade_cases_from_json(fixture)
        self.assertEqual(2, len(cases))
        self.assertEqual(ReviewStatus.PENDING_REVIEW, cases[1].review_status)

        duplicate = {"version": 1, "cases": [cases[0].to_dict(), cases[0].to_dict()]}
        with self.assertRaisesRegex(EventPolicyCascadeError, "unique"):
            event_policy_cascade_cases_from_json(json.dumps(duplicate))
        with self.assertRaisesRegex(EventPolicyCascadeError, "duplicate keys"):
            event_policy_cascade_cases_from_json('{"version":1,"version":1,"cases":[]}')

        naive = cases[0].to_dict()
        naive["occurred_at"] = "2026-08-29T02:00:00"
        with self.assertRaisesRegex(EventPolicyCascadeError, "timezone-aware"):
            event_policy_cascade_cases_from_json(json.dumps({"version": 1, "cases": [naive]}))

    def test_public_schemas_match_runtime_fields(self) -> None:
        report = evaluate_event_policy_cascade(_policy(), (_case(1, 0),))
        input_schema = json.loads(
            Path("schemas/event-policy-cascade-input-v1.schema.json").read_text(encoding="utf-8")
        )
        report_schema = json.loads(
            Path("schemas/event-policy-cascade-report-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(input_schema["$defs"]["case"]["required"]),
            set(_case(1, 0).to_dict()),
        )
        self.assertEqual(set(report_schema["required"]), set(report.to_dict()))

    def test_cli_evaluates_and_validates_without_disclosing_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-policy-cascade-") as directory:
            root = Path(directory)
            policy_path = root / "policy.json"
            _policy().write_new(policy_path)
            cases_path = root / "cases.json"
            cases = (_case(1, 0), _case(2, 60, decision=_decision(review_tier=ReviewTier.REMOTE)))
            cases_path.write_text(
                json.dumps({"version": 1, "cases": [case.to_dict() for case in cases]}),
                encoding="utf-8",
            )
            output_path = root / "report.json"

            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "evaluate-event-policy-cascade",
                            "--policy",
                            str(policy_path),
                            "--cases",
                            str(cases_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(2, payload["case_count"])
            self.assertEqual(payload, EventPolicyCascadeReport.read(output_path).to_dict())
            self.assertNotIn(str(root), stdout.getvalue())

            validate_stdout = StringIO()
            with redirect_stdout(validate_stdout):
                self.assertEqual(
                    0,
                    main(["validate-event-policy-cascade", str(output_path)]),
                )
            summary = json.loads(validate_stdout.getvalue())
            self.assertTrue(summary["valid"])
            self.assertTrue(summary["offline_replay_only"])
            self.assertFalse(summary["delivery_performed"])
            self.assertNotIn(str(root), validate_stdout.getvalue())

            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(
                    2,
                    main(
                        [
                            "evaluate-event-policy-cascade",
                            "--policy",
                            str(policy_path),
                            "--cases",
                            str(cases_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                )
            self.assertNotIn(str(root), stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
