"""Household event-policy validation and offline routing evaluation commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baby_monitor.cli_support import _json_print
from baby_monitor.event_policy import (
    EventPolicyError,
    HouseholdEventPolicy,
    PolicyEventInput,
    evaluate_event_policy,
)
from baby_monitor.event_policy_cascade import (
    EventPolicyCascadeReport,
    evaluate_event_policy_cascade,
    read_event_policy_cascade_cases,
)

MAX_POLICY_EVENTS_FILE_BYTES = 1024 * 1024


def command_validate_event_policy(args: argparse.Namespace) -> int:
    policy = HouseholdEventPolicy.read(args.path)
    _json_print(policy.to_public_summary())
    return 0


def _load_policy_events(path: str) -> tuple[PolicyEventInput, ...]:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if source.stat().st_size > MAX_POLICY_EVENTS_FILE_BYTES:
            raise EventPolicyError("policy events file exceeds the 1 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EventPolicyError("policy events file is missing or unsafe") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EventPolicyError("policy events file is not valid JSON") from exc
    if isinstance(payload, dict) and set(payload) == {"events"}:
        payload = payload["events"]
    if not isinstance(payload, list) or not 1 <= len(payload) <= 1000:
        raise EventPolicyError("policy events file must be a non-empty JSON array")
    return tuple(PolicyEventInput.from_dict(item) for item in payload)


def command_evaluate_event_policy(args: argparse.Namespace) -> int:
    policy = HouseholdEventPolicy.read(args.policy)
    events = _load_policy_events(args.events)
    evaluation = evaluate_event_policy(policy, events)
    payload = evaluation.to_dict()
    if args.output:
        target = Path(args.output).expanduser()
        if target.exists():
            raise FileExistsError("event policy evaluation output already exists")
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _json_print(payload)
    return 0


def command_evaluate_event_policy_cascade(args: argparse.Namespace) -> int:
    policy = HouseholdEventPolicy.read(args.policy)
    cases = read_event_policy_cascade_cases(args.cases)
    report = evaluate_event_policy_cascade(policy, cases)
    if args.output:
        report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_event_policy_cascade(args: argparse.Namespace) -> int:
    report = EventPolicyCascadeReport.read(args.path)
    _json_print(
        {
            "valid": True,
            "version": report.version,
            "policy_sha256": report.policy_sha256,
            "package_id": report.package_id,
            "policy_enabled": report.policy_enabled,
            "input_sha256": report.input_sha256,
            "case_count": report.case_count,
            "routed_count": report.routed_count,
            "suppressed_counts": report.suppressed_counts,
            "outcome_count": len(report.outcomes),
            "report_sha256": report.fingerprint,
            "offline_replay_only": report.offline_replay_only,
            "risk_inference": report.risk_inference,
            "review_decision_authority": report.review_decision_authority,
            "notification_delivery_authority": (report.notification_delivery_authority),
            "delivery_performed": report.delivery_performed,
            "runtime_state_read": report.runtime_state_read,
            "source_media_read": report.source_media_read,
            "source_media_included": report.source_media_included,
            "notification_outbox_written": report.notification_outbox_written,
            "network_used": report.network_used,
            "subject_identity_inference": report.subject_identity_inference,
            "paths_disclosed": report.paths_disclosed,
        }
    )
    return 0
