from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .domain import RiskLevel

_SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SAFE_CODE = re.compile(r"[a-z0-9_.-]{1,64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SELECTOR_ID = "deterministic_facts_v1"
_SELECTOR_REVISION = "2026-08-19"
_REASON_ORDER = (
    "event_recorded",
    "human_keep",
    "person_fact",
    "motion_fact",
    "hardware_signal",
    "zone_signal",
    "risk_field",
    "coverage_complete",
)


class DailyDigestError(ValueError):
    pass


def _finite_unit(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise DailyDigestError(f"daily digest {field} is invalid")
    return float(value)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DailyDigestError("daily digest timestamp is invalid")
    return value.astimezone(UTC)


def _timezone(name: str) -> ZoneInfo:
    if not isinstance(name, str) or not 1 <= len(name) <= 80:
        raise DailyDigestError("daily digest timezone is invalid")
    try:
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError):
        raise DailyDigestError("daily digest timezone is invalid") from None


@dataclass(frozen=True, slots=True)
class DailyDigestConfig:
    target_duration_seconds: float = 60.0
    max_items: int = 12
    min_item_seconds: float = 2.0
    max_item_seconds: float = 8.0
    per_camera_quota: int = 4
    per_hour_quota: int = 2
    min_same_camera_gap_seconds: float = 120.0
    selector_id: str = _SELECTOR_ID
    revision: str = _SELECTOR_REVISION

    def __post_init__(self) -> None:
        for value, field, minimum, maximum in (
            (self.target_duration_seconds, "target duration", 2.0, 600.0),
            (self.min_item_seconds, "minimum item duration", 1.0, 30.0),
            (self.max_item_seconds, "maximum item duration", 1.0, 60.0),
            (self.min_same_camera_gap_seconds, "same-camera gap", 0.0, 86_400.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise DailyDigestError(f"daily digest {field} is invalid")
        if self.max_item_seconds < self.min_item_seconds:
            raise DailyDigestError("daily digest item duration range is invalid")
        for value, field, maximum in (
            (self.max_items, "max_items", 60),
            (self.per_camera_quota, "per_camera_quota", 60),
            (self.per_hour_quota, "per_hour_quota", 24),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise DailyDigestError(f"daily digest {field} is invalid")
        if self.selector_id != _SELECTOR_ID or self.revision != _SELECTOR_REVISION:
            raise DailyDigestError("daily digest selector provenance is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "selector_id": self.selector_id,
            "revision": self.revision,
            "target_duration_seconds": float(self.target_duration_seconds),
            "max_items": self.max_items,
            "min_item_seconds": float(self.min_item_seconds),
            "max_item_seconds": float(self.max_item_seconds),
            "per_camera_quota": self.per_camera_quota,
            "per_hour_quota": self.per_hour_quota,
            "min_same_camera_gap_seconds": float(self.min_same_camera_gap_seconds),
        }

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class DailyDigestCandidate:
    event_id: str
    camera_id: str
    occurred_at: datetime
    source_binding_sha256: str
    media_duration_seconds: float
    fact_offsets_seconds: tuple[float, ...] = ()
    person_confidence: float = 0.0
    motion_score: float = 0.0
    hardware_signal: bool = False
    zone_signal: bool = False
    risk_level: RiskLevel = RiskLevel.NONE
    human_keep: bool = False
    anonymous_track_key: str | None = None
    coverage_gap_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_id, str)
            or not isinstance(self.camera_id, str)
            or _SAFE_ID.fullmatch(self.event_id) is None
            or _SAFE_ID.fullmatch(self.camera_id) is None
        ):
            raise DailyDigestError("daily digest candidate identity is invalid")
        _aware_utc(self.occurred_at)
        if (
            not isinstance(self.source_binding_sha256, str)
            or _SHA256.fullmatch(self.source_binding_sha256) is None
        ):
            raise DailyDigestError("daily digest source binding is invalid")
        if (
            isinstance(self.media_duration_seconds, bool)
            or not isinstance(self.media_duration_seconds, (int, float))
            or not math.isfinite(float(self.media_duration_seconds))
            or not 0.1 <= float(self.media_duration_seconds) <= 3_600.0
        ):
            raise DailyDigestError("daily digest media duration is invalid")
        if not isinstance(self.fact_offsets_seconds, tuple) or len(self.fact_offsets_seconds) > 128:
            raise DailyDigestError("daily digest fact offsets are invalid")
        normalized_offsets: list[float] = []
        for value in self.fact_offsets_seconds:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= float(self.media_duration_seconds)
            ):
                raise DailyDigestError("daily digest fact offsets are invalid")
            normalized_offsets.append(float(value))
        if normalized_offsets != sorted(set(normalized_offsets)):
            raise DailyDigestError("daily digest fact offsets must be sorted and unique")
        _finite_unit(self.person_confidence, field="person confidence")
        _finite_unit(self.motion_score, field="motion score")
        if any(
            type(value) is not bool
            for value in (self.hardware_signal, self.zone_signal, self.human_keep)
        ):
            raise DailyDigestError("daily digest fact flags are invalid")
        if not isinstance(self.risk_level, RiskLevel):
            raise DailyDigestError("daily digest risk field is invalid")
        if self.anonymous_track_key is not None and (
            _SAFE_ID.fullmatch(self.anonymous_track_key) is None
        ):
            raise DailyDigestError("daily digest track key is invalid")
        if (
            not isinstance(self.coverage_gap_reasons, tuple)
            or len(self.coverage_gap_reasons) > 64
            or len(set(self.coverage_gap_reasons)) != len(self.coverage_gap_reasons)
            or any(_SAFE_CODE.fullmatch(value) is None for value in self.coverage_gap_reasons)
        ):
            raise DailyDigestError("daily digest coverage gaps are invalid")


@dataclass(frozen=True, slots=True)
class DailyDigestItem:
    event_id: str
    camera_id: str
    occurred_at: datetime
    source_binding_sha256: str
    clip_start_seconds: float
    clip_end_seconds: float
    selection_score: float
    score_components: tuple[tuple[str, float], ...]
    selection_reasons: tuple[str, ...]
    rank: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_id, str)
            or not isinstance(self.camera_id, str)
            or _SAFE_ID.fullmatch(self.event_id) is None
            or _SAFE_ID.fullmatch(self.camera_id) is None
        ):
            raise DailyDigestError("daily digest item identity is invalid")
        _aware_utc(self.occurred_at)
        if (
            not isinstance(self.source_binding_sha256, str)
            or _SHA256.fullmatch(self.source_binding_sha256) is None
        ):
            raise DailyDigestError("daily digest item source binding is invalid")
        for value, field, minimum, maximum in (
            (self.clip_start_seconds, "clip start", 0.0, 3_600.0),
            (self.clip_end_seconds, "clip end", 0.0, 3_600.0),
            (self.selection_score, "selection score", 0.0, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise DailyDigestError(f"daily digest item {field} is invalid")
        if self.clip_end_seconds <= self.clip_start_seconds:
            raise DailyDigestError("daily digest item clip window is invalid")
        if (
            not isinstance(self.score_components, tuple)
            or tuple(name for name, _value in self.score_components) != _REASON_ORDER
        ):
            raise DailyDigestError("daily digest score components are invalid")
        for name, value in self.score_components:
            if _SAFE_CODE.fullmatch(name) is None:
                raise DailyDigestError("daily digest score components are invalid")
            _finite_unit(value, field="score component")
        expected_score = round(
            min(1.0, sum(value for _name, value in self.score_components)),
            6,
        )
        if not math.isclose(self.selection_score, expected_score, abs_tol=1e-6):
            raise DailyDigestError("daily digest selection score is inconsistent")
        expected_reasons = tuple(name for name, value in self.score_components if value > 0.0)
        if self.selection_reasons != expected_reasons:
            raise DailyDigestError("daily digest selection reasons are inconsistent")
        if type(self.rank) is not int or not 1 <= self.rank <= 60:
            raise DailyDigestError("daily digest item rank is invalid")

    @property
    def duration_seconds(self) -> float:
        return round(self.clip_end_seconds - self.clip_start_seconds, 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "source_binding_sha256": self.source_binding_sha256,
            "clip_start_seconds": self.clip_start_seconds,
            "clip_end_seconds": self.clip_end_seconds,
            "duration_seconds": self.duration_seconds,
            "selection_score": self.selection_score,
            "score_components": dict(self.score_components),
            "selection_reasons": list(self.selection_reasons),
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class DailyDigestCoverage:
    source_by_camera: tuple[tuple[str, int], ...]
    selected_by_camera: tuple[tuple[str, int], ...]
    source_by_hour: tuple[tuple[str, int], ...]
    selected_by_hour: tuple[tuple[str, int], ...]
    gap_reason_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for values, field in (
            (self.source_by_camera, "source camera coverage"),
            (self.selected_by_camera, "selected camera coverage"),
            (self.source_by_hour, "source hour coverage"),
            (self.selected_by_hour, "selected hour coverage"),
            (self.gap_reason_counts, "gap coverage"),
        ):
            if (
                not isinstance(values, tuple)
                or tuple(sorted(values)) != values
                or len({key for key, _count in values}) != len(values)
                or any(
                    not isinstance(key, str)
                    or _SAFE_ID.fullmatch(key) is None
                    or type(count) is not int
                    or not 1 <= count <= 10_000
                    for key, count in values
                )
            ):
                raise DailyDigestError(f"daily digest {field} is invalid")
        for values in (self.source_by_hour, self.selected_by_hour):
            if any(re.fullmatch(r"(?:[01][0-9]|2[0-3])", key) is None for key, _ in values):
                raise DailyDigestError("daily digest hour coverage is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_by_camera": dict(self.source_by_camera),
            "selected_by_camera": dict(self.selected_by_camera),
            "source_by_hour": dict(self.source_by_hour),
            "selected_by_hour": dict(self.selected_by_hour),
            "gap_reason_counts": dict(self.gap_reason_counts),
        }


@dataclass(frozen=True, slots=True)
class DailyDigestPlan:
    digest_id: str
    local_date: str
    timezone: str
    selector_id: str
    revision: str
    config_sha256: str
    target_duration_seconds: float
    source_event_count: int
    selected_event_count: int
    selected_duration_seconds: float
    target_met: bool
    coverage: DailyDigestCoverage
    items: tuple[DailyDigestItem, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise DailyDigestError("daily digest plan version is invalid")
        if (
            not isinstance(self.digest_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", self.digest_id) is None
        ):
            raise DailyDigestError("daily digest plan ID is invalid")
        try:
            parsed_date = date.fromisoformat(self.local_date)
        except (TypeError, ValueError):
            raise DailyDigestError("daily digest plan date is invalid") from None
        timezone = _timezone(self.timezone)
        if self.selector_id != _SELECTOR_ID or self.revision != _SELECTOR_REVISION:
            raise DailyDigestError("daily digest plan provenance is invalid")
        if not isinstance(self.config_sha256, str) or _SHA256.fullmatch(self.config_sha256) is None:
            raise DailyDigestError("daily digest plan config binding is invalid")
        for value, field in (
            (self.target_duration_seconds, "target duration"),
            (self.selected_duration_seconds, "selected duration"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 600.0
            ):
                raise DailyDigestError(f"daily digest plan {field} is invalid")
        if self.target_duration_seconds < 2.0:
            raise DailyDigestError("daily digest plan target duration is invalid")
        if type(self.target_met) is not bool:
            raise DailyDigestError("daily digest plan target status is invalid")
        if not isinstance(self.coverage, DailyDigestCoverage) or not isinstance(self.items, tuple):
            raise DailyDigestError("daily digest plan content is invalid")
        if (
            type(self.source_event_count) is not int
            or not 0 <= self.source_event_count <= 10_000
            or type(self.selected_event_count) is not int
            or not 0 <= self.selected_event_count <= 60
            or self.selected_event_count > self.source_event_count
            or self.selected_event_count != len(self.items)
        ):
            raise DailyDigestError("daily digest plan counts are inconsistent")
        if tuple(item.rank for item in self.items) != tuple(range(1, len(self.items) + 1)) or len(
            {item.event_id for item in self.items}
        ) != len(self.items):
            raise DailyDigestError("daily digest plan item ordering is invalid")
        expected_duration = round(sum(item.duration_seconds for item in self.items), 3)
        if (
            not math.isclose(
                self.selected_duration_seconds,
                expected_duration,
                abs_tol=1e-6,
            )
            or self.selected_duration_seconds > self.target_duration_seconds
        ):
            raise DailyDigestError("daily digest plan duration is inconsistent")
        if self.target_met != (self.selected_duration_seconds >= self.target_duration_seconds):
            raise DailyDigestError("daily digest plan target status is inconsistent")
        if sum(dict(self.coverage.source_by_camera).values()) != self.source_event_count:
            raise DailyDigestError("daily digest plan source coverage is inconsistent")
        if sum(dict(self.coverage.selected_by_camera).values()) != self.selected_event_count:
            raise DailyDigestError("daily digest plan selected coverage is inconsistent")
        for item in self.items:
            if item.occurred_at.astimezone(timezone).date() != parsed_date:
                raise DailyDigestError("daily digest plan item date is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "digest_id": self.digest_id,
            "local_date": self.local_date,
            "timezone": self.timezone,
            "selector_id": self.selector_id,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "target_duration_seconds": self.target_duration_seconds,
            "source_event_count": self.source_event_count,
            "selected_event_count": self.selected_event_count,
            "selected_duration_seconds": self.selected_duration_seconds,
            "target_met": self.target_met,
            "coverage": self.coverage.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "facts_only": True,
            "identity_inference": False,
            "risk_inference": False,
            "paths_disclosed": False,
            "notification_decision": False,
            "media_read": False,
            "video_generated": False,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, data: object) -> DailyDigestPlan:
        """Read stored selection provenance without applying today's scoring.

        Historical components, including the legacy coverage bonus, stay intact
        for auditability. Reading them is not a new coverage verification; new
        plans alone receive the current scoring rules.
        """
        expected = {
            "version",
            "digest_id",
            "local_date",
            "timezone",
            "selector_id",
            "revision",
            "config_sha256",
            "target_duration_seconds",
            "source_event_count",
            "selected_event_count",
            "selected_duration_seconds",
            "target_met",
            "coverage",
            "items",
            "facts_only",
            "identity_inference",
            "risk_inference",
            "paths_disclosed",
            "notification_decision",
            "media_read",
            "video_generated",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise DailyDigestError("daily digest plan fields are invalid")
        if data["facts_only"] is not True or any(
            data[field] is not False
            for field in (
                "identity_inference",
                "risk_inference",
                "paths_disclosed",
                "notification_decision",
                "media_read",
                "video_generated",
            )
        ):
            raise DailyDigestError("daily digest plan authority flags are invalid")
        coverage_data = data["coverage"]
        coverage_fields = {
            "source_by_camera",
            "selected_by_camera",
            "source_by_hour",
            "selected_by_hour",
            "gap_reason_counts",
        }
        if not isinstance(coverage_data, dict) or set(coverage_data) != coverage_fields:
            raise DailyDigestError("daily digest coverage fields are invalid")

        def count_items(value: object) -> tuple[tuple[str, int], ...]:
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or type(count) is not int for key, count in value.items()
            ):
                raise DailyDigestError("daily digest coverage values are invalid")
            return tuple(sorted(value.items()))

        coverage = DailyDigestCoverage(
            source_by_camera=count_items(coverage_data["source_by_camera"]),
            selected_by_camera=count_items(coverage_data["selected_by_camera"]),
            source_by_hour=count_items(coverage_data["source_by_hour"]),
            selected_by_hour=count_items(coverage_data["selected_by_hour"]),
            gap_reason_counts=count_items(coverage_data["gap_reason_counts"]),
        )
        raw_items = data["items"]
        if not isinstance(raw_items, list) or len(raw_items) > 60:
            raise DailyDigestError("daily digest plan items are invalid")
        items: list[DailyDigestItem] = []
        item_fields = {
            "event_id",
            "camera_id",
            "occurred_at",
            "source_binding_sha256",
            "clip_start_seconds",
            "clip_end_seconds",
            "duration_seconds",
            "selection_score",
            "score_components",
            "selection_reasons",
            "rank",
        }
        for raw_item in raw_items:
            if not isinstance(raw_item, dict) or set(raw_item) != item_fields:
                raise DailyDigestError("daily digest item fields are invalid")
            raw_components = raw_item["score_components"]
            raw_reasons = raw_item["selection_reasons"]
            if (
                not isinstance(raw_components, dict)
                or set(raw_components) != set(_REASON_ORDER)
                or not isinstance(raw_reasons, list)
                or any(not isinstance(reason, str) for reason in raw_reasons)
            ):
                raise DailyDigestError("daily digest item score fields are invalid")
            try:
                occurred_at = datetime.fromisoformat(raw_item["occurred_at"])
            except (TypeError, ValueError):
                raise DailyDigestError("daily digest item timestamp is invalid") from None
            item = DailyDigestItem(
                event_id=raw_item["event_id"],
                camera_id=raw_item["camera_id"],
                occurred_at=occurred_at,
                source_binding_sha256=raw_item["source_binding_sha256"],
                clip_start_seconds=raw_item["clip_start_seconds"],
                clip_end_seconds=raw_item["clip_end_seconds"],
                selection_score=raw_item["selection_score"],
                score_components=tuple((name, raw_components[name]) for name in _REASON_ORDER),
                selection_reasons=tuple(raw_reasons),
                rank=raw_item["rank"],
            )
            if not math.isclose(
                item.duration_seconds,
                raw_item["duration_seconds"],
                abs_tol=1e-6,
            ):
                raise DailyDigestError("daily digest item duration is inconsistent")
            items.append(item)
        return cls(
            version=data["version"],
            digest_id=data["digest_id"],
            local_date=data["local_date"],
            timezone=data["timezone"],
            selector_id=data["selector_id"],
            revision=data["revision"],
            config_sha256=data["config_sha256"],
            target_duration_seconds=data["target_duration_seconds"],
            source_event_count=data["source_event_count"],
            selected_event_count=data["selected_event_count"],
            selected_duration_seconds=data["selected_duration_seconds"],
            target_met=data["target_met"],
            coverage=coverage,
            items=tuple(items),
        )

    @classmethod
    def from_json(cls, payload: str) -> DailyDigestPlan:
        if not isinstance(payload, str) or not 1 <= len(payload.encode("utf-8")) <= 2 * 1024 * 1024:
            raise DailyDigestError("daily digest plan JSON is invalid")

        def reject_constant(_value: str) -> None:
            raise ValueError

        def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError
                result[key] = value
            return result

        try:
            data = json.loads(
                payload,
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raise DailyDigestError("daily digest plan JSON is invalid") from None
        return cls.from_dict(data)


def _score(candidate: DailyDigestCandidate) -> tuple[float, tuple[tuple[str, float], ...]]:
    components = (
        ("event_recorded", 0.05),
        ("human_keep", 0.35 if candidate.human_keep else 0.0),
        ("person_fact", round(0.25 * float(candidate.person_confidence), 6)),
        ("motion_fact", round(0.25 * float(candidate.motion_score), 6)),
        ("hardware_signal", 0.15 if candidate.hardware_signal else 0.0),
        ("zone_signal", 0.10 if candidate.zone_signal else 0.0),
        ("risk_field", round(0.15 * candidate.risk_level.rank / 3.0, 6)),
        # v1 has no affirmative continuous-coverage certification. An empty gap
        # list means no recorded gaps, not proof of complete observation. Retain
        # the legacy component key for stored-plan compatibility, without bonus.
        ("coverage_complete", 0.0),
    )
    return round(min(1.0, sum(value for _key, value in components)), 6), components


def _clip_window(
    candidate: DailyDigestCandidate,
    *,
    desired_seconds: float,
) -> tuple[float, float]:
    duration = min(float(candidate.media_duration_seconds), desired_seconds)
    center = (
        sum(candidate.fact_offsets_seconds) / len(candidate.fact_offsets_seconds)
        if candidate.fact_offsets_seconds
        else float(candidate.media_duration_seconds) / 2.0
    )
    start = max(0.0, center - duration / 2.0)
    end = min(float(candidate.media_duration_seconds), start + duration)
    start = max(0.0, end - duration)
    return round(start, 3), round(end, 3)


def _hour_key(value: datetime, timezone: ZoneInfo) -> str:
    return value.astimezone(timezone).strftime("%H")


def _coverage(
    candidates: tuple[DailyDigestCandidate, ...],
    selected: tuple[DailyDigestItem, ...],
    *,
    timezone: ZoneInfo,
) -> DailyDigestCoverage:
    source_camera = Counter(item.camera_id for item in candidates)
    selected_camera = Counter(item.camera_id for item in selected)
    source_hour = Counter(_hour_key(item.occurred_at, timezone) for item in candidates)
    selected_hour = Counter(_hour_key(item.occurred_at, timezone) for item in selected)
    gap_reasons = Counter(
        reason for candidate in candidates for reason in candidate.coverage_gap_reasons
    )
    return DailyDigestCoverage(
        source_by_camera=tuple(sorted(source_camera.items())),
        selected_by_camera=tuple(sorted(selected_camera.items())),
        source_by_hour=tuple(sorted(source_hour.items())),
        selected_by_hour=tuple(sorted(selected_hour.items())),
        gap_reason_counts=tuple(sorted(gap_reasons.items())),
    )


def plan_daily_digest(
    candidates: tuple[DailyDigestCandidate, ...],
    *,
    local_date: date,
    timezone_name: str,
    config: DailyDigestConfig | None = None,
) -> DailyDigestPlan:
    if (
        not isinstance(candidates, tuple)
        or len(candidates) > 10_000
        or any(not isinstance(item, DailyDigestCandidate) for item in candidates)
    ):
        raise DailyDigestError("daily digest candidates are invalid")
    if not isinstance(local_date, date) or isinstance(local_date, datetime):
        raise DailyDigestError("daily digest local date is invalid")
    if config is None:
        config = DailyDigestConfig()
    if not isinstance(config, DailyDigestConfig):
        raise DailyDigestError("daily digest config is invalid")
    timezone = _timezone(timezone_name)
    local_start = datetime.combine(local_date, time.min, tzinfo=timezone)
    start = local_start.astimezone(UTC)
    end = (local_start + timedelta(days=1)).astimezone(UTC)
    identities: set[str] = set()
    for candidate in candidates:
        occurred_at = _aware_utc(candidate.occurred_at)
        if not start <= occurred_at < end:
            raise DailyDigestError("daily digest candidate is outside the local date")
        if candidate.event_id in identities:
            raise DailyDigestError("daily digest candidate event IDs must be unique")
        identities.add(candidate.event_id)

    ranked: list[
        tuple[float, datetime, str, DailyDigestCandidate, tuple[tuple[str, float], ...]]
    ] = []
    for candidate in candidates:
        score, components = _score(candidate)
        ranked.append(
            (
                score,
                candidate.occurred_at.astimezone(UTC),
                candidate.event_id,
                candidate,
                components,
            )
        )
    ranked.sort(key=lambda value: (-value[0], value[1], value[2]))

    camera_counts: Counter[str] = Counter()
    hour_counts: Counter[str] = Counter()
    selected_times: dict[str, list[datetime]] = defaultdict(list)
    selected_tracks: set[str] = set()
    items: list[DailyDigestItem] = []
    selected_duration = 0.0
    for score, occurred_at, _event_id, candidate, components in ranked:
        if len(items) >= config.max_items:
            break
        remaining = float(config.target_duration_seconds) - selected_duration
        if remaining < float(config.min_item_seconds):
            break
        hour = _hour_key(occurred_at, timezone)
        if camera_counts[candidate.camera_id] >= config.per_camera_quota:
            continue
        if hour_counts[hour] >= config.per_hour_quota:
            continue
        if candidate.anonymous_track_key and candidate.anonymous_track_key in selected_tracks:
            continue
        if any(
            abs((occurred_at - previous).total_seconds())
            < float(config.min_same_camera_gap_seconds)
            for previous in selected_times[candidate.camera_id]
        ):
            continue
        desired = min(float(config.max_item_seconds), remaining)
        if min(float(candidate.media_duration_seconds), desired) < float(config.min_item_seconds):
            continue
        clip_start, clip_end = _clip_window(candidate, desired_seconds=desired)
        reasons = tuple(name for name in _REASON_ORDER if dict(components).get(name, 0.0) > 0.0)
        item = DailyDigestItem(
            event_id=candidate.event_id,
            camera_id=candidate.camera_id,
            occurred_at=occurred_at,
            source_binding_sha256=candidate.source_binding_sha256,
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
            selection_score=score,
            score_components=components,
            selection_reasons=reasons,
            rank=len(items) + 1,
        )
        items.append(item)
        selected_duration = round(selected_duration + item.duration_seconds, 3)
        camera_counts[candidate.camera_id] += 1
        hour_counts[hour] += 1
        selected_times[candidate.camera_id].append(occurred_at)
        if candidate.anonymous_track_key:
            selected_tracks.add(candidate.anonymous_track_key)

    item_tuple = tuple(items)
    coverage = _coverage(candidates, item_tuple, timezone=timezone)
    plan_payload = {
        "local_date": local_date.isoformat(),
        "timezone": timezone_name,
        "config_sha256": config.config_sha256,
        "source_bindings": [candidate.source_binding_sha256 for candidate in candidates],
        "items": [item.to_dict() for item in item_tuple],
    }
    return DailyDigestPlan(
        digest_id=_canonical_sha256(plan_payload)[:32],
        local_date=local_date.isoformat(),
        timezone=timezone_name,
        selector_id=config.selector_id,
        revision=config.revision,
        config_sha256=config.config_sha256,
        target_duration_seconds=float(config.target_duration_seconds),
        source_event_count=len(candidates),
        selected_event_count=len(item_tuple),
        selected_duration_seconds=selected_duration,
        target_met=selected_duration >= float(config.target_duration_seconds),
        coverage=coverage,
        items=item_tuple,
    )
