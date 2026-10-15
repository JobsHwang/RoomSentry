"""Declarative household event-package routing policy and its rule engine.

This module is the deterministic rule layer between completed events and the
existing notification outbox gates:

```text
candidate trigger -> analysis -> completed event (facts fixed)
                                   |
                                   v
                    HouseholdEventPolicy (declarative JSON)
                                   |
                                   v
                 EventPolicyEvaluation (route / suppress + reason)
                                   |
                                   v
              existing notification outbox + transport gates
```

Hard boundaries (mirrored in the public schema and docs/rules.md):

- The policy never modifies event facts, ``risk_level``, or ``review_status``;
  it only routes already-decided events.
- A routed decision grants no delivery authority: external delivery still
  requires the standing double opt-in (``notification.enabled`` plus
  ``external_delivery_enabled`` and D-012 credentials).
- Rules default to ``local_only`` sinks and the whole policy defaults to
  ``enabled=false``.
- Everything is path-free, anonymous, and free of media or credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .domain import EventType, ReviewStatus, RiskLevel

MAX_EVENT_POLICY_BYTES = 1024 * 1024
MAX_EVENT_POLICY_RULES = 32
MAX_POLICY_EVENTS_PER_EVALUATION = 1_000

_EVENT_TYPES = tuple(member.value for member in EventType)
_RISK_LEVELS = tuple(member.value for member in RiskLevel)
_REVIEW_STATUSES = tuple(member.value for member in ReviewStatus)
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SINK_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_CLOCK_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")

SUPPRESSION_REASONS = (
    "policy_disabled",
    "no_rule_match",
    "quiet_hours",
    "cooldown",
    "aggregation_cap",
)


class EventPolicyError(ValueError):
    """A household event policy or its evaluation input is unsafe."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EventPolicyError("event policy JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                EventPolicyError("event policy JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise EventPolicyError("event policy JSON is invalid") from exc


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


def _enum_set(value: Any, field: str, allowed: tuple[str, ...]) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 16
        or any(item not in allowed for item in value)
        or len(set(value)) != len(value)
    ):
        raise EventPolicyError(f"event policy {field} is invalid")
    return frozenset(value)


def _bounded_seconds(value: Any, field: str, *, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= maximum
    ):
        raise EventPolicyError(f"event policy {field} is invalid")
    return float(value)


def _safe_id(value: Any, field: str, *, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EventPolicyError(f"event policy {field} is invalid")
    return value


def _clock_time(value: Any, field: str) -> time:
    if not isinstance(value, str) or _CLOCK_TIME.fullmatch(value) is None:
        raise EventPolicyError(f"event policy {field} is invalid")
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour=hour, minute=minute)


@dataclass(frozen=True, slots=True)
class PolicyEventInput:
    """One completed event summarized for routing evaluation."""

    event_id: str
    camera_id: str
    occurred_at: datetime
    review_status: ReviewStatus
    event_type: EventType
    risk_level: RiskLevel

    def __post_init__(self) -> None:
        for field, value in (("event_id", self.event_id), ("camera_id", self.camera_id)):
            _safe_id(value, f"event {field}", pattern=_SAFE_ID)
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise EventPolicyError("event occurred_at must be timezone-aware")
        if not isinstance(self.review_status, ReviewStatus):
            raise EventPolicyError("event review_status is invalid")
        if not isinstance(self.event_type, EventType):
            raise EventPolicyError("event event_type is invalid")
        if not isinstance(self.risk_level, RiskLevel):
            raise EventPolicyError("event risk_level is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> PolicyEventInput:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise EventPolicyError("policy event fields do not match v1")
        occurred_at = value["occurred_at"]
        if not isinstance(occurred_at, datetime):
            try:
                occurred_at = datetime.fromisoformat(occurred_at)
            except (TypeError, ValueError) as exc:
                raise EventPolicyError("policy event occurred_at is invalid") from exc
        return cls(
            event_id=value["event_id"],
            camera_id=value["camera_id"],
            occurred_at=occurred_at,
            review_status=ReviewStatus(value["review_status"]),
            event_type=EventType(value["event_type"]),
            risk_level=RiskLevel(value["risk_level"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "occurred_at": self.occurred_at.isoformat(),
            "review_status": self.review_status.value,
            "event_type": self.event_type.value,
            "risk_level": self.risk_level.value,
        }


@dataclass(frozen=True, slots=True)
class PolicyEventRule:
    """One declarative routing rule: match conditions plus routing guards."""

    rule_id: str
    event_types: frozenset[str]
    risk_levels: frozenset[str]
    review_statuses: frozenset[str]
    sink_ids: tuple[str, ...]
    local_only: bool = True
    cameras: frozenset[str] = frozenset()
    cooldown_seconds: float = 0.0
    quiet_hours_suppress: bool = False
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "07:00"
    aggregation_window_seconds: float = 0.0
    aggregation_max_events: int = 1

    def __post_init__(self) -> None:
        _safe_id(self.rule_id, "rule_id", pattern=_SAFE_ID)
        object.__setattr__(
            self,
            "event_types",
            _enum_set(sorted(self.event_types), "rule event_types", _EVENT_TYPES),
        )
        object.__setattr__(
            self,
            "risk_levels",
            _enum_set(sorted(self.risk_levels), "rule risk_levels", _RISK_LEVELS),
        )
        object.__setattr__(
            self,
            "review_statuses",
            _enum_set(sorted(self.review_statuses), "rule review_statuses", _REVIEW_STATUSES),
        )
        if (
            not isinstance(self.sink_ids, tuple)
            or not 1 <= len(self.sink_ids) <= 4
            or any(
                _safe_id(sink, "rule sink_id", pattern=_SINK_ID) != sink for sink in self.sink_ids
            )
            or len(set(self.sink_ids)) != len(self.sink_ids)
        ):
            raise EventPolicyError("event policy rule sink_ids are invalid")
        if type(self.local_only) is not bool:
            raise EventPolicyError("event policy rule local_only must be a boolean")
        if (
            not isinstance(self.cameras, frozenset)
            or len(self.cameras) > 8
            or any(
                _safe_id(camera, "rule camera", pattern=_SAFE_ID) != camera
                for camera in self.cameras
            )
        ):
            raise EventPolicyError("event policy rule cameras are invalid")
        object.__setattr__(
            self,
            "cooldown_seconds",
            _bounded_seconds(self.cooldown_seconds, "rule cooldown_seconds", maximum=86_400.0),
        )
        if type(self.quiet_hours_suppress) is not bool:
            raise EventPolicyError("event policy rule quiet_hours_suppress must be a boolean")
        object.__setattr__(self, "quiet_hours_start", self.quiet_hours_start)
        object.__setattr__(self, "quiet_hours_end", self.quiet_hours_end)
        _clock_time(self.quiet_hours_start, "rule quiet_hours_start")
        _clock_time(self.quiet_hours_end, "rule quiet_hours_end")
        object.__setattr__(
            self,
            "aggregation_window_seconds",
            _bounded_seconds(
                self.aggregation_window_seconds,
                "rule aggregation_window_seconds",
                maximum=86_400.0,
            ),
        )
        if (
            type(self.aggregation_max_events) is not int
            or not 1 <= self.aggregation_max_events <= 100
        ):
            raise EventPolicyError("event policy rule aggregation_max_events is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> PolicyEventRule:
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != expected:
            raise EventPolicyError("event policy rule fields do not match v1")
        return cls(
            rule_id=value["rule_id"],
            event_types=frozenset(value["event_types"]),
            risk_levels=frozenset(value["risk_levels"]),
            review_statuses=frozenset(value["review_statuses"]),
            sink_ids=tuple(value["sink_ids"]),
            local_only=value["local_only"],
            cameras=frozenset(value["cameras"]),
            cooldown_seconds=value["cooldown_seconds"],
            quiet_hours_suppress=value["quiet_hours_suppress"],
            quiet_hours_start=value["quiet_hours_start"],
            quiet_hours_end=value["quiet_hours_end"],
            aggregation_window_seconds=value["aggregation_window_seconds"],
            aggregation_max_events=value["aggregation_max_events"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "event_types": sorted(self.event_types),
            "risk_levels": sorted(self.risk_levels),
            "review_statuses": sorted(self.review_statuses),
            "sink_ids": list(self.sink_ids),
            "local_only": self.local_only,
            "cameras": sorted(self.cameras),
            "cooldown_seconds": self.cooldown_seconds,
            "quiet_hours_suppress": self.quiet_hours_suppress,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "aggregation_window_seconds": self.aggregation_window_seconds,
            "aggregation_max_events": self.aggregation_max_events,
        }

    def matches(self, event: PolicyEventInput) -> bool:
        if event.event_type.value not in self.event_types:
            return False
        if event.risk_level.value not in self.risk_levels:
            return False
        if event.review_status.value not in self.review_statuses:
            return False
        return not (self.cameras and event.camera_id not in self.cameras)

    def in_quiet_hours(self, event: PolicyEventInput, timezone: ZoneInfo) -> bool:
        if not self.quiet_hours_suppress:
            return False
        local = event.occurred_at.astimezone(timezone).time()
        start = _clock_time(self.quiet_hours_start, "rule quiet_hours_start")
        end = _clock_time(self.quiet_hours_end, "rule quiet_hours_end")
        if start <= end:
            return start <= local < end
        return local >= start or local < end


@dataclass(frozen=True, slots=True)
class HouseholdEventPolicy:
    """A versioned, declarative household event package.

    Policies route already-decided events to local sinks under cooldown,
    quiet-hour, and aggregation guards. They carry no risk or delivery
    authority: the standing notification gates stay in charge.
    """

    package_id: str
    timezone: str
    rules: tuple[PolicyEventRule, ...]
    version: int = 1
    enabled: bool = False
    risk_inference: bool = False
    notification_delivery_authority: bool = False
    subject_identity_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise EventPolicyError("unsupported household event policy version")
        _safe_id(self.package_id, "package_id", pattern=_SAFE_ID)
        if not isinstance(self.timezone, str):
            raise EventPolicyError("event policy timezone is invalid")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise EventPolicyError("event policy timezone is invalid") from exc
        if (
            not isinstance(self.rules, tuple)
            or not 1 <= len(self.rules) <= MAX_EVENT_POLICY_RULES
            or any(not isinstance(rule, PolicyEventRule) for rule in self.rules)
        ):
            raise EventPolicyError("event policy rules are invalid")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise EventPolicyError("event policy rule ids must be unique")
        if type(self.enabled) is not bool:
            raise EventPolicyError("event policy enabled must be a boolean")
        for field, expected in (
            ("risk_inference", False),
            ("notification_delivery_authority", False),
            ("subject_identity_inference", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise EventPolicyError(f"event policy {field} must be {str(expected).lower()}")

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdEventPolicy:
        fields = {*cls.__dataclass_fields__, "policy_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise EventPolicyError("household event policy fields do not match v1")
        policy_sha256 = value["policy_sha256"]
        policy = cls(
            **{
                key: (
                    tuple(PolicyEventRule.from_dict(item) for item in value[key])
                    if key == "rules"
                    else value[key]
                )
                for key in value
                if key != "policy_sha256"
            }
        )
        if policy_sha256 != policy.fingerprint:
            raise EventPolicyError("household event policy fingerprint is invalid")
        return policy

    @classmethod
    def from_json(cls, raw: str) -> HouseholdEventPolicy:
        if len(raw.encode("utf-8")) > MAX_EVENT_POLICY_BYTES:
            raise EventPolicyError("household event policy exceeds the 1 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdEventPolicy:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_EVENT_POLICY_BYTES:
                raise EventPolicyError("household event policy exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except EventPolicyError:
            raise
        except (OSError, UnicodeError) as exc:
            raise EventPolicyError("household event policy is missing or unsafe") from exc
        return cls.from_json(raw)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [rule.to_dict() for rule in self.rules]
                if field == "rules"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["policy_sha256"] = self.fingerprint
        return value

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise EventPolicyError("household event policy destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_EVENT_POLICY_BYTES:
            raise EventPolicyError("household event policy exceeds the 1 MiB limit")
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
            raise FileExistsError("household event policy output already exists") from None
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise EventPolicyError("household event policy output could not be written") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def to_public_summary(self) -> dict[str, object]:
        """A path-free card summary for dashboards and CLI previews."""

        return {
            "version": 1,
            "package_id": self.package_id,
            "timezone": self.timezone,
            "enabled": self.enabled,
            "rule_count": len(self.rules),
            "rules": [rule.to_dict() for rule in self.rules],
            "risk_inference": False,
            "notification_delivery_authority": False,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class EventPolicyDecision:
    event_id: str
    action: str  # route | suppress
    rule_id: str | None
    reason: str  # routed | suppressed (see SUPPRESSION_REASONS)
    sink_ids: tuple[str, ...] = ()
    local_only: bool = True

    def __post_init__(self) -> None:
        _safe_id(self.event_id, "decision event_id", pattern=_SAFE_ID)
        if self.action not in {"route", "suppress"}:
            raise EventPolicyError("event policy decision action is invalid")
        if self.rule_id is not None:
            _safe_id(self.rule_id, "decision rule_id", pattern=_SAFE_ID)
        if type(self.local_only) is not bool:
            raise EventPolicyError("event policy decision local_only must be a boolean")
        if (
            not isinstance(self.sink_ids, tuple)
            or len(self.sink_ids) > 4
            or any(
                _safe_id(sink_id, "decision sink_id", pattern=_SINK_ID) != sink_id
                for sink_id in self.sink_ids
            )
            or len(set(self.sink_ids)) != len(self.sink_ids)
        ):
            raise EventPolicyError("event policy decision sink_ids are invalid")
        if self.action == "route":
            if self.rule_id is None or not self.sink_ids:
                raise EventPolicyError("routed decision requires a rule and sinks")
            if self.reason != "routed":
                raise EventPolicyError("routed decision reason is invalid")
        else:
            if self.reason not in SUPPRESSION_REASONS:
                raise EventPolicyError("suppressed decision reason is invalid")
            if self.reason == "no_rule_match" and self.rule_id is not None:
                raise EventPolicyError("no-match decision cannot bind a rule")
            if self.reason == "policy_disabled" and self.rule_id is not None:
                raise EventPolicyError("disabled-policy decision cannot bind a rule")
            if self.sink_ids:
                raise EventPolicyError("suppressed decision cannot route sinks")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "action": self.action,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "sink_ids": list(self.sink_ids),
            "local_only": self.local_only,
        }


@dataclass(frozen=True, slots=True)
class EventPolicyEvaluation:
    policy_sha256: str
    package_id: str
    policy_enabled: bool
    event_count: int
    routed_count: int
    suppressed_counts: dict[str, int]
    decisions: tuple[EventPolicyDecision, ...]
    version: int = 1
    routing_only: bool = True
    facts_modified: bool = False
    risk_inference: bool = False
    notification_delivery_authority: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise EventPolicyError("unsupported event policy evaluation version")
        if re.fullmatch(r"[0-9a-f]{64}", self.policy_sha256) is None:
            raise EventPolicyError("event policy evaluation fingerprint is invalid")
        if self.event_count != len(self.decisions):
            raise EventPolicyError("event policy evaluation accounting is inconsistent")
        routed = sum(1 for item in self.decisions if item.action == "route")
        if routed != self.routed_count:
            raise EventPolicyError("event policy evaluation routing count is inconsistent")
        for reason, count in self.suppressed_counts.items():
            if reason not in SUPPRESSION_REASONS or count < 0:
                raise EventPolicyError("event policy suppression reason is invalid")
        if sum(self.suppressed_counts.values()) + self.routed_count != self.event_count:
            raise EventPolicyError("event policy evaluation accounting is inconsistent")
        for field, expected in (
            ("routing_only", True),
            ("facts_modified", False),
            ("risk_inference", False),
            ("notification_delivery_authority", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise EventPolicyError(
                    f"event policy evaluation {field} must be {str(expected).lower()}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policy_sha256": self.policy_sha256,
            "package_id": self.package_id,
            "policy_enabled": self.policy_enabled,
            "event_count": self.event_count,
            "routed_count": self.routed_count,
            "suppressed_counts": dict(sorted(self.suppressed_counts.items())),
            "decisions": [item.to_dict() for item in self.decisions],
            "routing_only": self.routing_only,
            "facts_modified": self.facts_modified,
            "risk_inference": self.risk_inference,
            "notification_delivery_authority": self.notification_delivery_authority,
            "source_media_included": self.source_media_included,
            "paths_disclosed": self.paths_disclosed,
        }


def evaluate_event_policy(
    policy: HouseholdEventPolicy,
    events: tuple[PolicyEventInput, ...],
) -> EventPolicyEvaluation:
    """Route already-decided events through the declarative policy.

    Events are processed in chronological order. The first matching rule
    wins; cooldown and aggregation budgets are tracked per (rule, camera)
    and quiet hours are evaluated in the policy timezone. A disabled policy
    suppresses everything with ``policy_disabled`` — never silently routes.
    """

    if not isinstance(policy, HouseholdEventPolicy):
        raise EventPolicyError("event policy is invalid")
    if (
        not isinstance(events, tuple)
        or not 1 <= len(events) <= MAX_POLICY_EVENTS_PER_EVALUATION
        or any(not isinstance(event, PolicyEventInput) for event in events)
    ):
        raise EventPolicyError("policy events are invalid")
    ordered = tuple(sorted(events, key=lambda event: (event.occurred_at, event.event_id)))
    if any(
        current.occurred_at < previous.occurred_at
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise EventPolicyError("policy events are not chronologically sortable")

    timezone = ZoneInfo(policy.timezone)
    last_routed_at: dict[tuple[str, str], datetime] = {}
    routed_history: dict[tuple[str, str], list[datetime]] = {}
    decisions: list[EventPolicyDecision] = []

    for event in ordered:
        if not policy.enabled:
            decisions.append(
                EventPolicyDecision(
                    event_id=event.event_id,
                    action="suppress",
                    rule_id=None,
                    reason="policy_disabled",
                )
            )
            continue
        matched = next((rule for rule in policy.rules if rule.matches(event)), None)
        if matched is None:
            decisions.append(
                EventPolicyDecision(
                    event_id=event.event_id,
                    action="suppress",
                    rule_id=None,
                    reason="no_rule_match",
                )
            )
            continue
        key = (matched.rule_id, event.camera_id)
        if matched.in_quiet_hours(event, timezone):
            decisions.append(
                EventPolicyDecision(
                    event_id=event.event_id,
                    action="suppress",
                    rule_id=matched.rule_id,
                    reason="quiet_hours",
                )
            )
            continue
        previous = last_routed_at.get(key)
        if (
            previous is not None
            and matched.cooldown_seconds > 0.0
            and (event.occurred_at - previous).total_seconds() < matched.cooldown_seconds
        ):
            decisions.append(
                EventPolicyDecision(
                    event_id=event.event_id,
                    action="suppress",
                    rule_id=matched.rule_id,
                    reason="cooldown",
                )
            )
            continue
        if matched.aggregation_window_seconds > 0.0:
            window_start = event.occurred_at.timestamp() - matched.aggregation_window_seconds
            history = [
                stamp for stamp in routed_history.get(key, []) if stamp.timestamp() >= window_start
            ]
            if len(history) >= matched.aggregation_max_events:
                decisions.append(
                    EventPolicyDecision(
                        event_id=event.event_id,
                        action="suppress",
                        rule_id=matched.rule_id,
                        reason="aggregation_cap",
                    )
                )
                routed_history[key] = history
                continue
            history.append(event.occurred_at)
            routed_history[key] = history
        last_routed_at[key] = event.occurred_at
        decisions.append(
            EventPolicyDecision(
                event_id=event.event_id,
                action="route",
                rule_id=matched.rule_id,
                reason="routed",
                sink_ids=matched.sink_ids,
                local_only=matched.local_only,
            )
        )

    suppressed_counts: dict[str, int] = {reason: 0 for reason in SUPPRESSION_REASONS}
    for decision in decisions:
        if decision.action == "suppress":
            suppressed_counts[decision.reason] += 1
    routed_count = sum(1 for decision in decisions if decision.action == "route")
    return EventPolicyEvaluation(
        policy_sha256=policy.fingerprint,
        package_id=policy.package_id,
        policy_enabled=policy.enabled,
        event_count=len(decisions),
        routed_count=routed_count,
        suppressed_counts={k: v for k, v in suppressed_counts.items() if v > 0},
        decisions=tuple(decisions),
    )


def policy_event_from_notification_message(message: Any) -> PolicyEventInput:
    """Summarize a :class:`NotificationMessage` for routing evaluation."""

    return PolicyEventInput(
        event_id=message.event_id,
        camera_id=message.camera_id,
        occurred_at=message.occurred_at,
        review_status=message.review_status,
        event_type=message.event_type,
        risk_level=message.risk_level,
    )


def policy_event_from_stored_event(event: Any) -> PolicyEventInput:
    """Summarize a completed stored event before choosing a notification sink."""

    if event.result.risk_level is not event.decision.risk_level:
        raise EventPolicyError("stored event risk and DecisionRecord disagree")
    pending = event.result.needs_remote_review or event.decision.needs_remote_review
    return PolicyEventInput(
        event_id=event.event_id,
        camera_id=event.camera_id,
        occurred_at=event.occurred_at,
        review_status=(ReviewStatus.PENDING_REVIEW if pending else ReviewStatus.FINAL),
        event_type=event.result.event_type,
        risk_level=event.decision.risk_level,
    )
