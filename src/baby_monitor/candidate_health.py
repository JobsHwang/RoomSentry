from __future__ import annotations

import json
import math
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .detector_sampling import MAX_DECODED_FRAMES, MAX_DETECTOR_SAMPLES, DetectorSamplingLedger


class CandidateHealthError(ValueError):
    """Path-free failure for candidate worker health persistence."""


class CandidateHealthState(StrEnum):
    STARTING = "starting"
    SCANNING = "scanning"
    IDLE = "idle"
    FAILED = "failed"
    STOPPED = "stopped"


class CandidateHealthFreshness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    CLOCK_SKEW = "clock_skew"


_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SAFE_ERROR = re.compile(r"[a-z0-9_]{1,80}")
_MAX_HEALTH_BYTES = 128 * 1024
_SCAN_DURATION_BUCKETS = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1_800.0,
    3_600.0,
    604_800.0,
)
_LOCK_WAIT_BUCKETS = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1_800.0,
    3_600.0,
)


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CandidateHealthError(f"candidate health {field} is invalid")
    return value.astimezone(UTC)


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise CandidateHealthError(f"candidate health {field} is invalid")
    try:
        return _utc(datetime.fromisoformat(value), field)
    except ValueError:
        raise CandidateHealthError(f"candidate health {field} is invalid") from None


def _optional_time(value: Any, field: str) -> datetime | None:
    return None if value is None else _parse_time(value, field)


def _counter(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise CandidateHealthError(f"candidate health {field} is invalid")
    return value


def _optional_duration(value: Any, field: str, *, maximum: float) -> float | None:
    if value is None:
        return None
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= maximum
    ):
        raise CandidateHealthError(f"candidate health {field} is invalid")
    return float(value)


def _histogram_counts(value: Any, field: str, *, size: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise CandidateHealthError(f"candidate health {field} is invalid")
    return tuple(_counter(item, field) for item in value)


def _add_histogram_sample(
    counts: tuple[int, ...],
    bounds: tuple[float, ...],
    value: float,
) -> tuple[int, ...]:
    updated = list(counts)
    for index, upper_bound in enumerate(bounds):
        if value <= upper_bound:
            updated[index] += 1
            return tuple(updated)
    raise CandidateHealthError("candidate health duration is invalid")


def _p95_upper_bound(
    counts: tuple[int, ...],
    bounds: tuple[float, ...],
) -> float | None:
    total = sum(counts)
    if total == 0:
        return None
    threshold = math.ceil(total * 0.95)
    cumulative = 0
    for count, upper_bound in zip(counts, bounds, strict=True):
        cumulative += count
        if cumulative >= threshold:
            return upper_bound
    raise CandidateHealthError("candidate health histogram is inconsistent")


def _histogram_max_upper_bound(
    counts: tuple[int, ...],
    bounds: tuple[float, ...],
) -> float | None:
    for count, upper_bound in reversed(tuple(zip(counts, bounds, strict=True))):
        if count:
            return upper_bound
    return None


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CandidateHealthError("candidate health JSON is invalid")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CandidateHealthError("candidate health JSON is invalid")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise CandidateHealthError("candidate health JSON is invalid") from None


def _private_parent(path: Path) -> Path:
    if os.name != "posix" or not path.is_absolute() or path.name in {"", ".", ".."}:
        raise CandidateHealthError("candidate health path is unsafe")
    parent = path.parent
    try:
        metadata = parent.lstat()
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o777 != 0o700
            or parent.resolve(strict=True) != parent
        ):
            raise CandidateHealthError("candidate health path is unsafe")
    except OSError:
        raise CandidateHealthError("candidate health path is unsafe") from None
    return parent


@dataclass(frozen=True, slots=True)
class CandidateHealthLifetime:
    updated_at: datetime
    sessions_started: int = 0
    sessions_stopped: int = 0
    sessions_failed: int = 0
    unclean_restarts: int = 0
    scans_succeeded: int = 0
    scans_failed: int = 0
    candidate_groups_emitted: int = 0
    candidate_groups_suppressed: int = 0
    scan_duration_sample_count: int = 0
    scan_duration_max_seconds: float | None = None
    scan_duration_histogram: tuple[int, ...] = (0,) * len(_SCAN_DURATION_BUCKETS)
    execution_lock_wait_sample_count: int = 0
    execution_lock_wait_max_seconds: float | None = None
    execution_lock_wait_histogram: tuple[int, ...] = (0,) * len(_LOCK_WAIT_BUCKETS)

    def __post_init__(self) -> None:
        _utc(self.updated_at, "lifetime.updated_at")
        counters = {
            field: _counter(getattr(self, field), f"lifetime.{field}")
            for field in (
                "sessions_started",
                "sessions_stopped",
                "sessions_failed",
                "unclean_restarts",
                "scans_succeeded",
                "scans_failed",
                "candidate_groups_emitted",
                "candidate_groups_suppressed",
                "scan_duration_sample_count",
                "execution_lock_wait_sample_count",
            )
        }
        if (
            counters["sessions_stopped"]
            + counters["sessions_failed"]
            + counters["unclean_restarts"]
            > counters["sessions_started"]
        ):
            raise CandidateHealthError("candidate health lifetime sessions are inconsistent")
        if counters["unclean_restarts"] >= max(1, counters["sessions_started"]):
            raise CandidateHealthError("candidate health lifetime restarts are inconsistent")
        duration_counts = _histogram_counts(
            self.scan_duration_histogram,
            "lifetime.scan_duration_histogram",
            size=len(_SCAN_DURATION_BUCKETS),
        )
        wait_counts = _histogram_counts(
            self.execution_lock_wait_histogram,
            "lifetime.execution_lock_wait_histogram",
            size=len(_LOCK_WAIT_BUCKETS),
        )
        if sum(duration_counts) != counters["scan_duration_sample_count"]:
            raise CandidateHealthError("candidate health duration histogram is inconsistent")
        if sum(wait_counts) != counters["execution_lock_wait_sample_count"]:
            raise CandidateHealthError("candidate health wait histogram is inconsistent")
        duration_max = _optional_duration(
            self.scan_duration_max_seconds,
            "lifetime.scan_duration_max_seconds",
            maximum=604_800.0,
        )
        wait_max = _optional_duration(
            self.execution_lock_wait_max_seconds,
            "lifetime.execution_lock_wait_max_seconds",
            maximum=3_600.0,
        )
        if (counters["scan_duration_sample_count"] == 0) != (duration_max is None):
            raise CandidateHealthError("candidate health duration maximum is inconsistent")
        if (counters["execution_lock_wait_sample_count"] == 0) != (wait_max is None):
            raise CandidateHealthError("candidate health wait maximum is inconsistent")

    @classmethod
    def begin_session(
        cls,
        previous: CandidateWorkerHealth | None,
        *,
        updated_at: datetime,
    ) -> CandidateHealthLifetime:
        if previous is not None and previous.lifetime is not None:
            base = previous.lifetime
            if previous.version < 3:
                base = replace(
                    base,
                    candidate_groups_emitted=(
                        base.candidate_groups_emitted + previous.emitted_events
                    ),
                )
        elif previous is not None:
            duration_counts = (0,) * len(_SCAN_DURATION_BUCKETS)
            wait_counts = (0,) * len(_LOCK_WAIT_BUCKETS)
            duration_samples = 0
            wait_samples = 0
            duration_max = None
            wait_max = None
            if previous.last_scan_duration_seconds is not None:
                duration_max = previous.last_scan_duration_seconds
                duration_counts = _add_histogram_sample(
                    duration_counts,
                    _SCAN_DURATION_BUCKETS,
                    duration_max,
                )
                duration_samples = 1
            if previous.last_execution_lock_wait_seconds is not None:
                wait_max = previous.last_execution_lock_wait_seconds
                wait_counts = _add_histogram_sample(
                    wait_counts,
                    _LOCK_WAIT_BUCKETS,
                    wait_max,
                )
                wait_samples = 1
            base = cls(
                updated_at=previous.checked_at,
                sessions_started=1,
                sessions_stopped=int(previous.state is CandidateHealthState.STOPPED),
                sessions_failed=int(previous.state is CandidateHealthState.FAILED),
                scans_succeeded=previous.scans_succeeded,
                scans_failed=previous.scans_failed,
                candidate_groups_emitted=previous.emitted_events,
                scan_duration_sample_count=duration_samples,
                scan_duration_max_seconds=duration_max,
                scan_duration_histogram=duration_counts,
                execution_lock_wait_sample_count=wait_samples,
                execution_lock_wait_max_seconds=wait_max,
                execution_lock_wait_histogram=wait_counts,
            )
        else:
            base = cls(updated_at=updated_at)
        return cls(
            updated_at=updated_at,
            sessions_started=base.sessions_started + 1,
            sessions_stopped=base.sessions_stopped,
            sessions_failed=base.sessions_failed,
            unclean_restarts=(
                base.unclean_restarts + int(previous is not None and previous.worker_running)
            ),
            scans_succeeded=base.scans_succeeded,
            scans_failed=base.scans_failed,
            candidate_groups_emitted=base.candidate_groups_emitted,
            candidate_groups_suppressed=base.candidate_groups_suppressed,
            scan_duration_sample_count=base.scan_duration_sample_count,
            scan_duration_max_seconds=base.scan_duration_max_seconds,
            scan_duration_histogram=base.scan_duration_histogram,
            execution_lock_wait_sample_count=base.execution_lock_wait_sample_count,
            execution_lock_wait_max_seconds=base.execution_lock_wait_max_seconds,
            execution_lock_wait_histogram=base.execution_lock_wait_histogram,
        )

    def record_scan(
        self,
        *,
        updated_at: datetime,
        succeeded: bool,
        duration_seconds: float,
        execution_lock_wait_seconds: float | None,
    ) -> CandidateHealthLifetime:
        duration = _optional_duration(
            duration_seconds,
            "lifetime.scan_duration_seconds",
            maximum=604_800.0,
        )
        if duration is None or type(succeeded) is not bool:
            raise CandidateHealthError("candidate health lifetime scan is invalid")
        wait = _optional_duration(
            execution_lock_wait_seconds,
            "lifetime.execution_lock_wait_seconds",
            maximum=3_600.0,
        )
        duration_counts = _add_histogram_sample(
            self.scan_duration_histogram,
            _SCAN_DURATION_BUCKETS,
            duration,
        )
        wait_counts = self.execution_lock_wait_histogram
        if wait is not None:
            wait_counts = _add_histogram_sample(wait_counts, _LOCK_WAIT_BUCKETS, wait)
        return CandidateHealthLifetime(
            updated_at=updated_at,
            sessions_started=self.sessions_started,
            sessions_stopped=self.sessions_stopped,
            sessions_failed=self.sessions_failed,
            unclean_restarts=self.unclean_restarts,
            scans_succeeded=self.scans_succeeded + int(succeeded),
            scans_failed=self.scans_failed + int(not succeeded),
            candidate_groups_emitted=self.candidate_groups_emitted,
            candidate_groups_suppressed=self.candidate_groups_suppressed,
            scan_duration_sample_count=self.scan_duration_sample_count + 1,
            scan_duration_max_seconds=max(self.scan_duration_max_seconds or 0.0, duration),
            scan_duration_histogram=duration_counts,
            execution_lock_wait_sample_count=(
                self.execution_lock_wait_sample_count + int(wait is not None)
            ),
            execution_lock_wait_max_seconds=(
                max(self.execution_lock_wait_max_seconds or 0.0, wait)
                if wait is not None
                else self.execution_lock_wait_max_seconds
            ),
            execution_lock_wait_histogram=wait_counts,
        )

    def record_candidate_groups(
        self,
        *,
        updated_at: datetime,
        emitted: int,
        suppressed: int,
    ) -> CandidateHealthLifetime:
        emitted = _counter(emitted, "lifetime.candidate_groups_emitted_delta")
        suppressed = _counter(
            suppressed,
            "lifetime.candidate_groups_suppressed_delta",
        )
        return replace(
            self,
            updated_at=_utc(updated_at, "lifetime.updated_at"),
            candidate_groups_emitted=self.candidate_groups_emitted + emitted,
            candidate_groups_suppressed=(self.candidate_groups_suppressed + suppressed),
        )

    def finish_session(
        self,
        *,
        updated_at: datetime,
        failed: bool,
    ) -> CandidateHealthLifetime:
        if type(failed) is not bool:
            raise CandidateHealthError("candidate health lifetime session is invalid")
        return CandidateHealthLifetime(
            updated_at=updated_at,
            sessions_started=self.sessions_started,
            sessions_stopped=self.sessions_stopped + int(not failed),
            sessions_failed=self.sessions_failed + int(failed),
            unclean_restarts=self.unclean_restarts,
            scans_succeeded=self.scans_succeeded,
            scans_failed=self.scans_failed,
            candidate_groups_emitted=self.candidate_groups_emitted,
            candidate_groups_suppressed=self.candidate_groups_suppressed,
            scan_duration_sample_count=self.scan_duration_sample_count,
            scan_duration_max_seconds=self.scan_duration_max_seconds,
            scan_duration_histogram=self.scan_duration_histogram,
            execution_lock_wait_sample_count=self.execution_lock_wait_sample_count,
            execution_lock_wait_max_seconds=self.execution_lock_wait_max_seconds,
            execution_lock_wait_histogram=self.execution_lock_wait_histogram,
        )

    def to_dict(
        self,
        *,
        include_candidate_groups: bool = True,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "updated_at": self.updated_at.astimezone(UTC).isoformat(),
            "sessions_started": self.sessions_started,
            "sessions_stopped": self.sessions_stopped,
            "sessions_failed": self.sessions_failed,
            "unclean_restarts": self.unclean_restarts,
            "scans_succeeded": self.scans_succeeded,
            "scans_failed": self.scans_failed,
            "scan_duration_sample_count": self.scan_duration_sample_count,
            "scan_duration_max_seconds": self.scan_duration_max_seconds,
            "scan_duration_histogram": list(self.scan_duration_histogram),
            "execution_lock_wait_sample_count": self.execution_lock_wait_sample_count,
            "execution_lock_wait_max_seconds": self.execution_lock_wait_max_seconds,
            "execution_lock_wait_histogram": list(self.execution_lock_wait_histogram),
        }
        if include_candidate_groups:
            payload["candidate_groups_emitted"] = self.candidate_groups_emitted
            payload["candidate_groups_suppressed"] = self.candidate_groups_suppressed
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> CandidateHealthLifetime:
        legacy = {
            "updated_at",
            "sessions_started",
            "sessions_stopped",
            "sessions_failed",
            "unclean_restarts",
            "scans_succeeded",
            "scans_failed",
            "scan_duration_sample_count",
            "scan_duration_max_seconds",
            "scan_duration_histogram",
            "execution_lock_wait_sample_count",
            "execution_lock_wait_max_seconds",
            "execution_lock_wait_histogram",
        }
        candidate_fields = {
            "candidate_groups_emitted",
            "candidate_groups_suppressed",
        }
        if not isinstance(value, dict) or frozenset(value) not in {
            frozenset(legacy),
            frozenset(legacy | candidate_fields),
        }:
            raise CandidateHealthError("candidate health lifetime fields are invalid")
        has_candidate_groups = candidate_fields <= set(value)
        return cls(
            updated_at=_parse_time(value["updated_at"], "lifetime.updated_at"),
            sessions_started=value["sessions_started"],
            sessions_stopped=value["sessions_stopped"],
            sessions_failed=value["sessions_failed"],
            unclean_restarts=value["unclean_restarts"],
            scans_succeeded=value["scans_succeeded"],
            scans_failed=value["scans_failed"],
            candidate_groups_emitted=(
                value["candidate_groups_emitted"] if has_candidate_groups else 0
            ),
            candidate_groups_suppressed=(
                value["candidate_groups_suppressed"] if has_candidate_groups else 0
            ),
            scan_duration_sample_count=value["scan_duration_sample_count"],
            scan_duration_max_seconds=value["scan_duration_max_seconds"],
            scan_duration_histogram=_histogram_counts(
                value["scan_duration_histogram"],
                "lifetime.scan_duration_histogram",
                size=len(_SCAN_DURATION_BUCKETS),
            ),
            execution_lock_wait_sample_count=value["execution_lock_wait_sample_count"],
            execution_lock_wait_max_seconds=value["execution_lock_wait_max_seconds"],
            execution_lock_wait_histogram=_histogram_counts(
                value["execution_lock_wait_histogram"],
                "lifetime.execution_lock_wait_histogram",
                size=len(_LOCK_WAIT_BUCKETS),
            ),
        )

    def summary(self) -> dict[str, object]:
        duration_p95 = _p95_upper_bound(
            self.scan_duration_histogram,
            _SCAN_DURATION_BUCKETS,
        )
        wait_p95 = _p95_upper_bound(
            self.execution_lock_wait_histogram,
            _LOCK_WAIT_BUCKETS,
        )
        if duration_p95 is not None and self.scan_duration_max_seconds is not None:
            duration_p95 = min(duration_p95, self.scan_duration_max_seconds)
        if wait_p95 is not None and self.execution_lock_wait_max_seconds is not None:
            wait_p95 = min(wait_p95, self.execution_lock_wait_max_seconds)
        return {
            "sessions_started": self.sessions_started,
            "sessions_stopped": self.sessions_stopped,
            "sessions_failed": self.sessions_failed,
            "unclean_restarts": self.unclean_restarts,
            "scans_succeeded": self.scans_succeeded,
            "scans_failed": self.scans_failed,
            "candidate_groups_emitted": self.candidate_groups_emitted,
            "candidate_groups_suppressed": self.candidate_groups_suppressed,
            "scan_duration_sample_count": self.scan_duration_sample_count,
            "scan_duration_max_seconds": self.scan_duration_max_seconds,
            "scan_duration_p95_upper_bound_seconds": duration_p95,
            "execution_lock_wait_sample_count": self.execution_lock_wait_sample_count,
            "execution_lock_wait_max_seconds": self.execution_lock_wait_max_seconds,
            "execution_lock_wait_p95_upper_bound_seconds": wait_p95,
            "approximate_percentile": True,
            "media_read": False,
            "paths_disclosed": False,
        }

    def delta_summary(self, baseline: CandidateHealthLifetime) -> dict[str, object]:
        if not isinstance(baseline, CandidateHealthLifetime):
            raise CandidateHealthError("candidate health baseline is invalid")
        counter_fields = (
            "sessions_started",
            "sessions_stopped",
            "sessions_failed",
            "unclean_restarts",
            "scans_succeeded",
            "scans_failed",
            "candidate_groups_emitted",
            "candidate_groups_suppressed",
            "scan_duration_sample_count",
            "execution_lock_wait_sample_count",
        )
        deltas: dict[str, int] = {}
        for field in counter_fields:
            current = getattr(self, field)
            previous = getattr(baseline, field)
            if current < previous:
                raise CandidateHealthError("candidate health lifetime moved backwards")
            deltas[field] = current - previous
        duration_histogram = tuple(
            current - previous
            for current, previous in zip(
                self.scan_duration_histogram,
                baseline.scan_duration_histogram,
                strict=True,
            )
        )
        wait_histogram = tuple(
            current - previous
            for current, previous in zip(
                self.execution_lock_wait_histogram,
                baseline.execution_lock_wait_histogram,
                strict=True,
            )
        )
        if any(value < 0 for value in (*duration_histogram, *wait_histogram)):
            raise CandidateHealthError("candidate health histogram moved backwards")
        if sum(duration_histogram) != deltas["scan_duration_sample_count"]:
            raise CandidateHealthError("candidate health duration delta is inconsistent")
        if sum(wait_histogram) != deltas["execution_lock_wait_sample_count"]:
            raise CandidateHealthError("candidate health wait delta is inconsistent")
        return {
            **deltas,
            "scan_duration_p95_upper_bound_seconds": _p95_upper_bound(
                duration_histogram,
                _SCAN_DURATION_BUCKETS,
            ),
            "scan_duration_max_upper_bound_seconds": _histogram_max_upper_bound(
                duration_histogram,
                _SCAN_DURATION_BUCKETS,
            ),
            "execution_lock_wait_p95_upper_bound_seconds": _p95_upper_bound(
                wait_histogram,
                _LOCK_WAIT_BUCKETS,
            ),
            "execution_lock_wait_max_upper_bound_seconds": _histogram_max_upper_bound(
                wait_histogram,
                _LOCK_WAIT_BUCKETS,
            ),
            "approximate_percentile": True,
            "media_read": False,
            "paths_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class CandidateSamplingHealth:
    """Path-free latest-scan evidence, never presence or complete-source coverage."""

    sampled_frame_count: int
    zero_detection_sample_count: int
    decoded_frame_count: int
    timestamp_fallback_frame_count: int
    max_adjacent_container_pts_gap_seconds: float | None
    container_pts_regression_count: int
    end_reason: str = "read_returned_false"
    source_exhaustion_verified: bool = False

    def __post_init__(self) -> None:
        for field in (
            "sampled_frame_count",
            "zero_detection_sample_count",
            "decoded_frame_count",
            "timestamp_fallback_frame_count",
            "container_pts_regression_count",
        ):
            _counter(getattr(self, field), f"last_sampling.{field}")
        if not 1 <= self.sampled_frame_count <= MAX_DETECTOR_SAMPLES:
            raise CandidateHealthError("candidate health sampling count is invalid")
        if not self.sampled_frame_count <= self.decoded_frame_count <= MAX_DECODED_FRAMES:
            raise CandidateHealthError("candidate health decoded count is inconsistent")
        if self.zero_detection_sample_count > self.sampled_frame_count:
            raise CandidateHealthError("candidate health zero detection count is inconsistent")
        if self.timestamp_fallback_frame_count > self.decoded_frame_count:
            raise CandidateHealthError("candidate health fallback count is inconsistent")
        pts_count = self.decoded_frame_count - self.timestamp_fallback_frame_count
        if self.container_pts_regression_count > max(0, pts_count - 1):
            raise CandidateHealthError("candidate health PTS regression count is inconsistent")
        gap = self.max_adjacent_container_pts_gap_seconds
        if gap is not None:
            try:
                valid = type(gap) in {int, float} and math.isfinite(gap) and gap >= 0
            except (OverflowError, TypeError, ValueError):
                valid = False
            if not valid or pts_count < 2:
                raise CandidateHealthError("candidate health PTS gap is invalid")
        if self.end_reason != "read_returned_false":
            raise CandidateHealthError("candidate health sampling end reason is invalid")
        if self.source_exhaustion_verified is not False:
            raise CandidateHealthError("candidate health sampling cannot verify source exhaustion")

    @classmethod
    def from_ledger(cls, ledger: DetectorSamplingLedger) -> CandidateSamplingHealth:
        if not isinstance(ledger, DetectorSamplingLedger):
            raise CandidateHealthError("candidate health sampling ledger is invalid")
        return cls(
            sampled_frame_count=len(ledger.samples),
            zero_detection_sample_count=sum(
                sample.detection_count == 0 for sample in ledger.samples
            ),
            decoded_frame_count=ledger.decoded_frame_count,
            timestamp_fallback_frame_count=ledger.timestamp_fallback_frame_count,
            max_adjacent_container_pts_gap_seconds=ledger.max_adjacent_container_pts_gap_seconds,
            container_pts_regression_count=ledger.container_pts_regression_count,
            end_reason=ledger.end_reason,
            source_exhaustion_verified=ledger.source_exhaustion_verified,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sampled_frame_count": self.sampled_frame_count,
            "zero_detection_sample_count": self.zero_detection_sample_count,
            "decoded_frame_count": self.decoded_frame_count,
            "timestamp_fallback_frame_count": self.timestamp_fallback_frame_count,
            "max_adjacent_container_pts_gap_seconds": self.max_adjacent_container_pts_gap_seconds,
            "container_pts_regression_count": self.container_pts_regression_count,
            "end_reason": self.end_reason,
            "source_exhaustion_verified": self.source_exhaustion_verified,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateSamplingHealth:
        expected = {
            "sampled_frame_count",
            "zero_detection_sample_count",
            "decoded_frame_count",
            "timestamp_fallback_frame_count",
            "max_adjacent_container_pts_gap_seconds",
            "container_pts_regression_count",
            "end_reason",
            "source_exhaustion_verified",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CandidateHealthError("candidate health sampling fields are invalid")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CandidateWorkerHealth:
    camera_id: str
    scanner_identity: str
    detector_identity: str
    checked_at: datetime
    state: CandidateHealthState
    worker_running: bool
    scans_started: int = 0
    scans_succeeded: int = 0
    scans_failed: int = 0
    segments_scanned: int = 0
    detections: int = 0
    emitted_events: int = 0
    suppressed_candidate_groups: int = 0
    queue_full_events: int = 0
    last_scan_started_at: datetime | None = None
    last_scan_completed_at: datetime | None = None
    last_scan_duration_seconds: float | None = None
    last_execution_lock_wait_seconds: float | None = None
    last_error_code: str | None = None
    lifetime: CandidateHealthLifetime | None = None
    version: int = 1
    last_sampling: CandidateSamplingHealth | None = None

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in {1, 2, 3, 4}:
            raise CandidateHealthError("unsupported candidate health version")
        if _SAFE_ID.fullmatch(self.camera_id) is None:
            raise CandidateHealthError("candidate health camera_id is invalid")
        if _SAFE_ID.fullmatch(self.scanner_identity) is None:
            raise CandidateHealthError("candidate health scanner_identity is invalid")
        if _SAFE_ID.fullmatch(self.detector_identity) is None:
            raise CandidateHealthError("candidate health detector_identity is invalid")
        checked = _utc(self.checked_at, "checked_at")
        if not isinstance(self.state, CandidateHealthState):
            raise CandidateHealthError("candidate health state is invalid")
        if type(self.worker_running) is not bool:
            raise CandidateHealthError("candidate health worker_running is invalid")
        expected_running = self.state in {
            CandidateHealthState.STARTING,
            CandidateHealthState.SCANNING,
            CandidateHealthState.IDLE,
        }
        if self.worker_running is not expected_running:
            raise CandidateHealthError("candidate health state is inconsistent")
        counters = {
            field: _counter(getattr(self, field), field)
            for field in (
                "scans_started",
                "scans_succeeded",
                "scans_failed",
                "segments_scanned",
                "detections",
                "emitted_events",
                "suppressed_candidate_groups",
                "queue_full_events",
            )
        }
        if counters["scans_succeeded"] + counters["scans_failed"] > counters["scans_started"]:
            raise CandidateHealthError("candidate health scan counts are inconsistent")
        started = (
            _utc(self.last_scan_started_at, "last_scan_started_at")
            if self.last_scan_started_at is not None
            else None
        )
        completed = (
            _utc(self.last_scan_completed_at, "last_scan_completed_at")
            if self.last_scan_completed_at is not None
            else None
        )
        if (counters["scans_started"] == 0) != (started is None):
            raise CandidateHealthError("candidate health scan timestamps are inconsistent")
        if (counters["scans_succeeded"] == 0) != (completed is None):
            raise CandidateHealthError("candidate health scan timestamps are inconsistent")
        if started is not None and started > checked:
            raise CandidateHealthError("candidate health scan timestamps are inconsistent")
        if completed is not None and (completed > checked or started is None):
            raise CandidateHealthError("candidate health scan timestamps are inconsistent")
        _optional_duration(
            self.last_scan_duration_seconds,
            "last_scan_duration_seconds",
            maximum=604_800.0,
        )
        _optional_duration(
            self.last_execution_lock_wait_seconds,
            "last_execution_lock_wait_seconds",
            maximum=3_600.0,
        )
        if (completed is None) != (self.last_scan_duration_seconds is None):
            raise CandidateHealthError("candidate health duration is inconsistent")
        if completed is None and self.last_execution_lock_wait_seconds is not None:
            raise CandidateHealthError("candidate health lock wait is inconsistent")
        if self.state is CandidateHealthState.FAILED:
            if (
                not isinstance(self.last_error_code, str)
                or _SAFE_ERROR.fullmatch(self.last_error_code) is None
            ):
                raise CandidateHealthError("candidate health error code is invalid")
        elif self.last_error_code is not None:
            raise CandidateHealthError("candidate health error code is inconsistent")
        if self.version < 3 and self.suppressed_candidate_groups != 0:
            raise CandidateHealthError("legacy candidate health suppression count is inconsistent")
        if self.version == 1 and self.lifetime is not None:
            raise CandidateHealthError("candidate health v1 lifetime is inconsistent")
        if self.last_sampling is not None:
            if self.version < 4 or not isinstance(self.last_sampling, CandidateSamplingHealth):
                raise CandidateHealthError("candidate health last sampling is invalid")
            if completed is None or self.state is CandidateHealthState.FAILED:
                raise CandidateHealthError("candidate health last sampling is inconsistent")
        if self.version in {2, 3, 4}:
            if not isinstance(self.lifetime, CandidateHealthLifetime):
                raise CandidateHealthError(f"candidate health v{self.version} lifetime is invalid")
            if self.lifetime.updated_at.astimezone(UTC) > checked:
                raise CandidateHealthError("candidate health lifetime timestamp is invalid")
            if self.version >= 3 and (
                self.lifetime.candidate_groups_emitted < self.emitted_events
                or self.lifetime.candidate_groups_suppressed < self.suppressed_candidate_groups
            ):
                raise CandidateHealthError("candidate health group counts are inconsistent")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "camera_id": self.camera_id,
            "scanner_identity": self.scanner_identity,
            "detector_identity": self.detector_identity,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "state": self.state.value,
            "worker_running": self.worker_running,
            "scans_started": self.scans_started,
            "scans_succeeded": self.scans_succeeded,
            "scans_failed": self.scans_failed,
            "segments_scanned": self.segments_scanned,
            "detections": self.detections,
            "emitted_events": self.emitted_events,
            "queue_full_events": self.queue_full_events,
            "last_scan_started_at": (
                self.last_scan_started_at.astimezone(UTC).isoformat()
                if self.last_scan_started_at is not None
                else None
            ),
            "last_scan_completed_at": (
                self.last_scan_completed_at.astimezone(UTC).isoformat()
                if self.last_scan_completed_at is not None
                else None
            ),
            "last_scan_duration_seconds": self.last_scan_duration_seconds,
            "last_execution_lock_wait_seconds": self.last_execution_lock_wait_seconds,
            "last_error_code": self.last_error_code,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }
        if self.version >= 3:
            payload["suppressed_candidate_groups"] = self.suppressed_candidate_groups
        if self.version in {2, 3, 4}:
            assert self.lifetime is not None
            payload["lifetime"] = self.lifetime.to_dict(
                include_candidate_groups=self.version >= 3,
            )
        if self.version == 4:
            payload["last_sampling"] = (
                self.last_sampling.to_dict() if self.last_sampling is not None else None
            )
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> CandidateWorkerHealth:
        expected_v1 = {
            "version",
            "camera_id",
            "scanner_identity",
            "detector_identity",
            "checked_at",
            "state",
            "worker_running",
            "scans_started",
            "scans_succeeded",
            "scans_failed",
            "segments_scanned",
            "detections",
            "emitted_events",
            "queue_full_events",
            "last_scan_started_at",
            "last_scan_completed_at",
            "last_scan_duration_seconds",
            "last_execution_lock_wait_seconds",
            "last_error_code",
            "paths_disclosed",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        }
        if not isinstance(value, dict):
            raise CandidateHealthError("candidate health fields are invalid")
        version = value.get("version")
        if type(version) is not int:
            raise CandidateHealthError("candidate health fields do not match version")
        if version == 1:
            expected = expected_v1
        elif version == 2:
            expected = expected_v1 | {"lifetime"}
        elif version in {3, 4}:
            expected = expected_v1 | {
                "lifetime",
                "suppressed_candidate_groups",
            }
            if version == 4:
                expected.add("last_sampling")
        else:
            expected = set()
        if type(version) is not int or version not in {1, 2, 3, 4} or set(value) != expected:
            raise CandidateHealthError("candidate health fields do not match version")
        if any(
            value[field] is not False
            for field in (
                "paths_disclosed",
                "identity_inference",
                "risk_assessment",
                "notification_authority",
            )
        ):
            raise CandidateHealthError("candidate health privacy flags are invalid")
        try:
            state = CandidateHealthState(value["state"])
        except (TypeError, ValueError):
            raise CandidateHealthError("candidate health state is invalid") from None
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            scanner_identity=value["scanner_identity"],
            detector_identity=value["detector_identity"],
            checked_at=_parse_time(value["checked_at"], "checked_at"),
            state=state,
            worker_running=value["worker_running"],
            scans_started=value["scans_started"],
            scans_succeeded=value["scans_succeeded"],
            scans_failed=value["scans_failed"],
            segments_scanned=value["segments_scanned"],
            detections=value["detections"],
            emitted_events=value["emitted_events"],
            suppressed_candidate_groups=(
                value["suppressed_candidate_groups"] if version >= 3 else 0
            ),
            queue_full_events=value["queue_full_events"],
            last_scan_started_at=_optional_time(
                value["last_scan_started_at"], "last_scan_started_at"
            ),
            last_scan_completed_at=_optional_time(
                value["last_scan_completed_at"], "last_scan_completed_at"
            ),
            last_scan_duration_seconds=_optional_duration(
                value["last_scan_duration_seconds"],
                "last_scan_duration_seconds",
                maximum=604_800.0,
            ),
            last_execution_lock_wait_seconds=_optional_duration(
                value["last_execution_lock_wait_seconds"],
                "last_execution_lock_wait_seconds",
                maximum=3_600.0,
            ),
            last_error_code=value["last_error_code"],
            lifetime=(
                CandidateHealthLifetime.from_dict(value["lifetime"])
                if version in {2, 3, 4}
                else None
            ),
            last_sampling=(
                CandidateSamplingHealth.from_dict(value["last_sampling"])
                if version == 4 and value["last_sampling"] is not None
                else None
            ),
        )

    def write_atomic_private(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = _private_parent(target)
        if target.exists() or target.is_symlink():
            try:
                metadata = target.lstat()
                if (
                    target.is_symlink()
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or metadata.st_mode & 0o777 != 0o600
                ):
                    raise CandidateHealthError("candidate health path is unsafe")
            except OSError:
                raise CandidateHealthError("candidate health path is unsafe") from None
        encoded = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, target)
            parent_descriptor = os.open(
                parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except (OSError, CandidateHealthError):
            raise CandidateHealthError("candidate health persistence failed") from None
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    @classmethod
    def read_private(cls, path: str | Path) -> CandidateWorkerHealth:
        target = Path(path).expanduser()
        _private_parent(target)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags)
        except OSError:
            raise CandidateHealthError("candidate health file is unavailable") from None
        try:
            metadata = os.fstat(descriptor)
            path_metadata = target.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o777 != 0o600
                or metadata.st_size > _MAX_HEALTH_BYTES
                or (metadata.st_dev, metadata.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                raise CandidateHealthError("candidate health file is unsafe")
            raw = bytearray()
            while len(raw) <= _MAX_HEALTH_BYTES:
                chunk = os.read(descriptor, min(65_536, _MAX_HEALTH_BYTES + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
            if len(raw) > _MAX_HEALTH_BYTES:
                raise CandidateHealthError("candidate health file is unsafe")
        except (OSError, CandidateHealthError):
            raise CandidateHealthError("candidate health file is unsafe") from None
        finally:
            os.close(descriptor)
        try:
            return cls.from_dict(_strict_json(raw.decode("utf-8")))
        except (CandidateHealthError, UnicodeDecodeError):
            raise CandidateHealthError("candidate health file is invalid") from None


@dataclass(frozen=True, slots=True)
class CandidateHealthAssessment:
    health: CandidateWorkerHealth
    checked_at: datetime
    freshness: CandidateHealthFreshness
    age_seconds: float
    operational: bool
    attention_required: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "freshness": self.freshness.value,
            "age_seconds": self.age_seconds,
            "operational": self.operational,
            "attention_required": self.attention_required,
            "reason_codes": list(self.reason_codes),
            "health": self.health.to_dict(),
            "paths_disclosed": False,
        }


def assess_candidate_health(
    health: CandidateWorkerHealth,
    *,
    checked_at: datetime,
    stale_after_seconds: float,
    future_tolerance_seconds: float = 5.0,
) -> CandidateHealthAssessment:
    now = _utc(checked_at, "assessment.checked_at")
    if (
        type(stale_after_seconds) not in {int, float}
        or not math.isfinite(float(stale_after_seconds))
        or not 1.0 <= float(stale_after_seconds) <= 604_800.0
    ):
        raise CandidateHealthError("candidate health stale threshold is invalid")
    if (
        type(future_tolerance_seconds) not in {int, float}
        or not math.isfinite(float(future_tolerance_seconds))
        or not 0.0 <= float(future_tolerance_seconds) <= 3_600.0
    ):
        raise CandidateHealthError("candidate health future tolerance is invalid")
    age = (now - health.checked_at.astimezone(UTC)).total_seconds()
    reasons: list[str] = []
    if age < -float(future_tolerance_seconds):
        freshness = CandidateHealthFreshness.CLOCK_SKEW
        reasons.append("candidate_health_clock_skew")
    elif age > float(stale_after_seconds):
        freshness = CandidateHealthFreshness.STALE
        reasons.append("candidate_health_stale")
    else:
        freshness = CandidateHealthFreshness.FRESH
    if health.state is CandidateHealthState.FAILED:
        reasons.append("candidate_worker_failed")
    elif not health.worker_running:
        reasons.append("candidate_worker_not_running")
    operational = not reasons
    return CandidateHealthAssessment(
        health=health,
        checked_at=now,
        freshness=freshness,
        age_seconds=round(max(0.0, age), 3),
        operational=operational,
        attention_required=not operational,
        reason_codes=tuple(reasons),
    )
