"""Pure offline replay from ``DecisionRecord`` into event-policy routing.

This module evaluates an already-decided event stream.  It derives the
``risk_level`` and review state used by :mod:`baby_monitor.event_policy` from
strict ``DecisionRecord`` inputs, then records the resulting route/suppress
decisions in a fingerprinted report.  It never reads media or a runtime
database, mutates event facts, enqueues notifications, or grants delivery
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import DecisionRecord, EventType, FinalSource, ReviewStatus, ReviewTier
from .event_policy import (
    SUPPRESSION_REASONS,
    EventPolicyDecision,
    EventPolicyError,
    HouseholdEventPolicy,
    PolicyEventInput,
    evaluate_event_policy,
)

MAX_EVENT_POLICY_CASCADE_INPUT_BYTES = 4 * 1024 * 1024
MAX_EVENT_POLICY_CASCADE_REPORT_BYTES = 8 * 1024 * 1024
MAX_EVENT_POLICY_CASCADE_CASES = 1_000

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROUTE_REASONS = ("routed", *SUPPRESSION_REASONS)


class EventPolicyCascadeError(ValueError):
    """An offline DecisionRecord-to-routing replay is invalid or unsafe."""


def _strict_json(raw: str, *, label: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EventPolicyCascadeError(f"{label} JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                EventPolicyCascadeError(f"{label} JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise EventPolicyCascadeError(f"{label} JSON is invalid") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _safe_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise EventPolicyCascadeError(f"event policy cascade {field} is invalid")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EventPolicyCascadeError(f"event policy cascade {field} is invalid")
    return value


def _count(value: Any, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= MAX_EVENT_POLICY_CASCADE_CASES:
        raise EventPolicyCascadeError(f"event policy cascade {field} is invalid")
    return value


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EventPolicyCascadeError(f"event policy cascade {field} must be timezone-aware")
    return value.astimezone(UTC)


def _decision_sha256(decision: DecisionRecord) -> str:
    return _canonical_sha256(decision.to_dict())


@dataclass(frozen=True, slots=True)
class EventPolicyCascadeCase:
    """Minimal completed-event facts plus the authoritative DecisionRecord."""

    event_id: str
    camera_id: str
    occurred_at: datetime
    event_type: EventType
    result_needs_remote_review: bool
    decision: DecisionRecord
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise EventPolicyCascadeError("unsupported event policy cascade case version")
        _safe_id(self.event_id, field="case event_id")
        _safe_id(self.camera_id, field="case camera_id")
        object.__setattr__(
            self,
            "occurred_at",
            _utc(self.occurred_at, field="case occurred_at"),
        )
        if not isinstance(self.event_type, EventType):
            raise EventPolicyCascadeError("event policy cascade case event_type is invalid")
        if type(self.result_needs_remote_review) is not bool:
            raise EventPolicyCascadeError(
                "event policy cascade result_needs_remote_review must be a boolean"
            )
        if not isinstance(self.decision, DecisionRecord):
            raise EventPolicyCascadeError("event policy cascade DecisionRecord is invalid")
        self.to_policy_event()

    @property
    def review_status(self) -> ReviewStatus:
        if self.result_needs_remote_review or self.decision.needs_remote_review:
            return ReviewStatus.PENDING_REVIEW
        return ReviewStatus.FINAL

    def to_policy_event(self) -> PolicyEventInput:
        """Derive the exact routing input without changing decided risk."""

        return PolicyEventInput(
            event_id=self.event_id,
            camera_id=self.camera_id,
            occurred_at=self.occurred_at,
            review_status=self.review_status,
            event_type=self.event_type,
            risk_level=self.decision.risk_level,
        )

    @classmethod
    def from_dict(cls, value: Any) -> EventPolicyCascadeCase:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise EventPolicyCascadeError("event policy cascade case fields do not match v1")
        occurred_at = value["occurred_at"]
        try:
            parsed_at = datetime.fromisoformat(occurred_at)
            event_type = EventType(value["event_type"])
            decision = DecisionRecord.from_dict(value["decision"])
        except (TypeError, ValueError) as exc:
            raise EventPolicyCascadeError("event policy cascade case is invalid") from exc
        return cls(
            version=value["version"],
            event_id=value["event_id"],
            camera_id=value["camera_id"],
            occurred_at=parsed_at,
            event_type=event_type,
            result_needs_remote_review=value["result_needs_remote_review"],
            decision=decision,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "occurred_at": self.occurred_at.isoformat(),
            "event_type": self.event_type.value,
            "result_needs_remote_review": self.result_needs_remote_review,
            "decision": self.decision.to_dict(),
        }


def event_policy_cascade_cases_from_json(raw: str) -> tuple[EventPolicyCascadeCase, ...]:
    """Parse one strict, bounded replay input document."""

    if len(raw.encode("utf-8")) > MAX_EVENT_POLICY_CASCADE_INPUT_BYTES:
        raise EventPolicyCascadeError("event policy cascade input exceeds the 4 MiB limit")
    value = _strict_json(raw, label="event policy cascade input")
    if not isinstance(value, dict) or set(value) != {"version", "cases"}:
        raise EventPolicyCascadeError("event policy cascade input fields do not match v1")
    if type(value["version"]) is not int or value["version"] != 1:
        raise EventPolicyCascadeError("unsupported event policy cascade input version")
    cases = value["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_EVENT_POLICY_CASCADE_CASES:
        raise EventPolicyCascadeError("event policy cascade input cases are invalid")
    parsed = tuple(EventPolicyCascadeCase.from_dict(item) for item in cases)
    if len({case.event_id for case in parsed}) != len(parsed):
        raise EventPolicyCascadeError("event policy cascade event IDs must be unique")
    return parsed


def read_event_policy_cascade_cases(path: str | Path) -> tuple[EventPolicyCascadeCase, ...]:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if source.stat().st_size > MAX_EVENT_POLICY_CASCADE_INPUT_BYTES:
            raise EventPolicyCascadeError("event policy cascade input exceeds the 4 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except EventPolicyCascadeError:
        raise
    except (OSError, UnicodeError) as exc:
        raise EventPolicyCascadeError("event policy cascade input is missing or unsafe") from exc
    return event_policy_cascade_cases_from_json(raw)


def _routing_from_dict(value: Any) -> EventPolicyDecision:
    fields = set(EventPolicyDecision.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != fields:
        raise EventPolicyCascadeError("event policy cascade routing fields do not match v1")
    sink_ids = value["sink_ids"]
    if not isinstance(sink_ids, list):
        raise EventPolicyCascadeError("event policy cascade routing sink_ids are invalid")
    try:
        return EventPolicyDecision(
            event_id=value["event_id"],
            action=value["action"],
            rule_id=value["rule_id"],
            reason=value["reason"],
            sink_ids=tuple(sink_ids),
            local_only=value["local_only"],
        )
    except (EventPolicyError, TypeError, ValueError) as exc:
        raise EventPolicyCascadeError("event policy cascade routing is invalid") from exc


@dataclass(frozen=True, slots=True)
class EventPolicyCascadeItem:
    """One DecisionRecord and the route/suppress decision it produced."""

    case: EventPolicyCascadeCase
    routing: EventPolicyDecision

    def __post_init__(self) -> None:
        if not isinstance(self.case, EventPolicyCascadeCase):
            raise EventPolicyCascadeError("event policy cascade item case is invalid")
        if not isinstance(self.routing, EventPolicyDecision):
            raise EventPolicyCascadeError("event policy cascade item routing is invalid")
        if self.routing.event_id != self.case.event_id:
            raise EventPolicyCascadeError("event policy cascade item binding is invalid")

    @property
    def decision_sha256(self) -> str:
        return _decision_sha256(self.case.decision)

    @classmethod
    def from_dict(cls, value: Any) -> EventPolicyCascadeItem:
        if not isinstance(value, dict) or set(value) != {
            "case",
            "decision_sha256",
            "routing",
        }:
            raise EventPolicyCascadeError("event policy cascade item fields do not match v1")
        item = cls(
            case=EventPolicyCascadeCase.from_dict(value["case"]),
            routing=_routing_from_dict(value["routing"]),
        )
        if value["decision_sha256"] != item.decision_sha256:
            raise EventPolicyCascadeError("event policy cascade DecisionRecord digest is invalid")
        return item

    def to_dict(self) -> dict[str, object]:
        return {
            "case": self.case.to_dict(),
            "decision_sha256": self.decision_sha256,
            "routing": self.routing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EventPolicyCascadeOutcome:
    """One exact DecisionRecord-state × routing-state cross-tab row."""

    review_status: ReviewStatus
    review_tier: ReviewTier
    final_source: FinalSource
    remote_used: bool
    has_contradictions: bool
    action: str
    reason: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.review_status, ReviewStatus):
            raise EventPolicyCascadeError("event policy cascade outcome review_status is invalid")
        if not isinstance(self.review_tier, ReviewTier):
            raise EventPolicyCascadeError("event policy cascade outcome review_tier is invalid")
        if not isinstance(self.final_source, FinalSource):
            raise EventPolicyCascadeError("event policy cascade outcome final_source is invalid")
        if type(self.remote_used) is not bool or type(self.has_contradictions) is not bool:
            raise EventPolicyCascadeError("event policy cascade outcome flags are invalid")
        if self.action not in {"route", "suppress"} or self.reason not in _ROUTE_REASONS:
            raise EventPolicyCascadeError("event policy cascade outcome routing is invalid")
        if (self.action == "route") != (self.reason == "routed"):
            raise EventPolicyCascadeError("event policy cascade outcome routing is inconsistent")
        _count(self.count, field="outcome count", positive=True)

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            self.review_status.value,
            self.review_tier.value,
            self.final_source.value,
            self.remote_used,
            self.has_contradictions,
            self.action,
            self.reason,
        )

    @classmethod
    def from_dict(cls, value: Any) -> EventPolicyCascadeOutcome:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise EventPolicyCascadeError("event policy cascade outcome fields do not match v1")
        try:
            return cls(
                review_status=ReviewStatus(value["review_status"]),
                review_tier=ReviewTier(value["review_tier"]),
                final_source=FinalSource(value["final_source"]),
                remote_used=value["remote_used"],
                has_contradictions=value["has_contradictions"],
                action=value["action"],
                reason=value["reason"],
                count=value["count"],
            )
        except (TypeError, ValueError) as exc:
            raise EventPolicyCascadeError("event policy cascade outcome is invalid") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "review_status": self.review_status.value,
            "review_tier": self.review_tier.value,
            "final_source": self.final_source.value,
            "remote_used": self.remote_used,
            "has_contradictions": self.has_contradictions,
            "action": self.action,
            "reason": self.reason,
            "count": self.count,
        }


def _ordered_cases(
    cases: tuple[EventPolicyCascadeCase, ...],
) -> tuple[EventPolicyCascadeCase, ...]:
    if (
        not isinstance(cases, tuple)
        or not 1 <= len(cases) <= MAX_EVENT_POLICY_CASCADE_CASES
        or any(not isinstance(case, EventPolicyCascadeCase) for case in cases)
    ):
        raise EventPolicyCascadeError("event policy cascade cases are invalid")
    if len({case.event_id for case in cases}) != len(cases):
        raise EventPolicyCascadeError("event policy cascade event IDs must be unique")
    return tuple(sorted(cases, key=lambda case: (case.occurred_at, case.event_id)))


def _input_sha256(cases: tuple[EventPolicyCascadeCase, ...]) -> str:
    ordered = _ordered_cases(cases)
    return _canonical_sha256({"version": 1, "cases": [case.to_dict() for case in ordered]})


def _outcomes(
    items: tuple[EventPolicyCascadeItem, ...],
) -> tuple[EventPolicyCascadeOutcome, ...]:
    counts: Counter[tuple[object, ...]] = Counter()
    for item in items:
        decision = item.case.decision
        counts[
            (
                item.case.review_status,
                decision.review_tier,
                decision.final_source,
                decision.remote_used,
                bool(decision.contradictions),
                item.routing.action,
                item.routing.reason,
            )
        ] += 1
    outcomes = tuple(
        EventPolicyCascadeOutcome(
            review_status=key[0],
            review_tier=key[1],
            final_source=key[2],
            remote_used=key[3],
            has_contradictions=key[4],
            action=key[5],
            reason=key[6],
            count=count,
        )
        for key, count in counts.items()
    )
    return tuple(sorted(outcomes, key=lambda outcome: outcome.sort_key))


@dataclass(frozen=True, slots=True)
class EventPolicyCascadeReport:
    """Fingerprint-bound, self-contained offline cascade replay report."""

    policy_sha256: str
    package_id: str
    policy_enabled: bool
    input_sha256: str
    case_count: int
    routed_count: int
    suppressed_counts: dict[str, int]
    outcomes: tuple[EventPolicyCascadeOutcome, ...]
    items: tuple[EventPolicyCascadeItem, ...]
    version: int = 1
    offline_replay_only: bool = True
    facts_modified: bool = False
    risk_inference: bool = False
    review_decision_authority: bool = False
    notification_delivery_authority: bool = False
    delivery_performed: bool = False
    runtime_state_read: bool = False
    source_media_read: bool = False
    source_media_included: bool = False
    notification_outbox_written: bool = False
    network_used: bool = False
    subject_identity_inference: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise EventPolicyCascadeError("unsupported event policy cascade report version")
        _sha256(self.policy_sha256, field="policy_sha256")
        _safe_id(self.package_id, field="package_id")
        if type(self.policy_enabled) is not bool:
            raise EventPolicyCascadeError("event policy cascade policy_enabled is invalid")
        _sha256(self.input_sha256, field="input_sha256")
        _count(self.case_count, field="case_count", positive=True)
        _count(self.routed_count, field="routed_count")
        if (
            not isinstance(self.items, tuple)
            or len(self.items) != self.case_count
            or any(not isinstance(item, EventPolicyCascadeItem) for item in self.items)
        ):
            raise EventPolicyCascadeError("event policy cascade report items are invalid")
        ordered = tuple(
            sorted(self.items, key=lambda item: (item.case.occurred_at, item.case.event_id))
        )
        if self.items != ordered or len({item.case.event_id for item in self.items}) != len(
            self.items
        ):
            raise EventPolicyCascadeError(
                "event policy cascade report items must be unique and chronological"
            )
        cases = tuple(item.case for item in self.items)
        if self.input_sha256 != _input_sha256(cases):
            raise EventPolicyCascadeError("event policy cascade input digest is invalid")
        routed_count = sum(item.routing.action == "route" for item in self.items)
        suppressed = Counter(
            item.routing.reason for item in self.items if item.routing.action == "suppress"
        )
        expected_suppressed = dict(sorted(suppressed.items()))
        if (
            self.routed_count != routed_count
            or not isinstance(self.suppressed_counts, dict)
            or self.suppressed_counts != expected_suppressed
            or self.routed_count + sum(self.suppressed_counts.values()) != self.case_count
        ):
            raise EventPolicyCascadeError(
                "event policy cascade report routing accounting is inconsistent"
            )
        if any(
            reason not in SUPPRESSION_REASONS or type(count) is not int or count <= 0
            for reason, count in self.suppressed_counts.items()
        ):
            raise EventPolicyCascadeError("event policy cascade suppression accounting is invalid")
        if (
            not self.policy_enabled
            and (
                self.routed_count != 0
                or self.suppressed_counts != {"policy_disabled": self.case_count}
            )
        ) or (self.policy_enabled and "policy_disabled" in self.suppressed_counts):
            raise EventPolicyCascadeError(
                "event policy cascade policy state and routing are inconsistent"
            )
        if (
            not isinstance(self.outcomes, tuple)
            or any(not isinstance(outcome, EventPolicyCascadeOutcome) for outcome in self.outcomes)
            or self.outcomes != _outcomes(self.items)
            or sum(outcome.count for outcome in self.outcomes) != self.case_count
        ):
            raise EventPolicyCascadeError("event policy cascade outcome accounting is inconsistent")
        for field, expected in (
            ("offline_replay_only", True),
            ("facts_modified", False),
            ("risk_inference", False),
            ("review_decision_authority", False),
            ("notification_delivery_authority", False),
            ("delivery_performed", False),
            ("runtime_state_read", False),
            ("source_media_read", False),
            ("source_media_included", False),
            ("notification_outbox_written", False),
            ("network_used", False),
            ("subject_identity_inference", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise EventPolicyCascadeError(
                    f"event policy cascade report {field} must be {str(expected).lower()}"
                )

    @classmethod
    def from_dict(cls, value: Any) -> EventPolicyCascadeReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise EventPolicyCascadeError("event policy cascade report fields do not match v1")
        report_sha256 = value["report_sha256"]
        try:
            report = cls(
                **{
                    key: (
                        tuple(EventPolicyCascadeOutcome.from_dict(item) for item in field_value)
                        if key == "outcomes" and isinstance(field_value, list)
                        else tuple(EventPolicyCascadeItem.from_dict(item) for item in field_value)
                        if key == "items" and isinstance(field_value, list)
                        else field_value
                    )
                    for key, field_value in value.items()
                    if key != "report_sha256"
                }
            )
        except EventPolicyCascadeError:
            raise
        except (TypeError, ValueError) as exc:
            raise EventPolicyCascadeError("event policy cascade report is invalid") from exc
        if report_sha256 != report.fingerprint:
            raise EventPolicyCascadeError("event policy cascade report fingerprint is invalid")
        return report

    @classmethod
    def from_json(cls, raw: str) -> EventPolicyCascadeReport:
        if len(raw.encode("utf-8")) > MAX_EVENT_POLICY_CASCADE_REPORT_BYTES:
            raise EventPolicyCascadeError("event policy cascade report exceeds the 8 MiB limit")
        return cls.from_dict(_strict_json(raw, label="event policy cascade report"))

    @classmethod
    def read(cls, path: str | Path) -> EventPolicyCascadeReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_EVENT_POLICY_CASCADE_REPORT_BYTES:
                raise EventPolicyCascadeError("event policy cascade report exceeds the 8 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except EventPolicyCascadeError:
            raise
        except (OSError, UnicodeError) as exc:
            raise EventPolicyCascadeError(
                "event policy cascade report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [outcome.to_dict() for outcome in self.outcomes]
                if field == "outcomes"
                else [item.to_dict() for item in self.items]
                if field == "items"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["report_sha256"] = self.fingerprint
        return value

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise EventPolicyCascadeError("event policy cascade report destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_EVENT_POLICY_CASCADE_REPORT_BYTES:
            raise EventPolicyCascadeError("event policy cascade report exceeds the 8 MiB limit")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(target, flags, 0o600)
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise FileExistsError("event policy cascade report output already exists") from None
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise EventPolicyCascadeError(
                "event policy cascade report output could not be written"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def to_public_dict(self) -> dict[str, object]:
        return self.to_dict()


def evaluate_event_policy_cascade(
    policy: HouseholdEventPolicy,
    cases: tuple[EventPolicyCascadeCase, ...],
) -> EventPolicyCascadeReport:
    """Replay DecisionRecord-bound cases through the pure routing policy."""

    if not isinstance(policy, HouseholdEventPolicy):
        raise EventPolicyCascadeError("event policy cascade policy is invalid")
    ordered = _ordered_cases(cases)
    try:
        evaluation = evaluate_event_policy(
            policy,
            tuple(case.to_policy_event() for case in ordered),
        )
    except EventPolicyError as exc:
        raise EventPolicyCascadeError("event policy cascade evaluation failed") from exc
    decisions = {decision.event_id: decision for decision in evaluation.decisions}
    if len(decisions) != len(ordered):
        raise EventPolicyCascadeError("event policy cascade routing output is incomplete")
    items = tuple(
        EventPolicyCascadeItem(case=case, routing=decisions[case.event_id]) for case in ordered
    )
    return EventPolicyCascadeReport(
        policy_sha256=policy.fingerprint,
        package_id=policy.package_id,
        policy_enabled=policy.enabled,
        input_sha256=_input_sha256(ordered),
        case_count=len(items),
        routed_count=evaluation.routed_count,
        suppressed_counts=dict(sorted(evaluation.suppressed_counts.items())),
        outcomes=_outcomes(items),
        items=items,
    )
