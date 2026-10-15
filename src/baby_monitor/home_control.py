from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .analysis_queue import AnalysisQueue, AnalysisQueueError
from .candidate_health import (
    CandidateHealthError,
    CandidateHealthFreshness,
    CandidateHealthState,
    CandidateWorkerHealth,
    assess_candidate_health,
)
from .candidate_soak import CandidateSoakError, inspect_candidate_soak
from .daily_digest_export import inspect_daily_digest_exports
from .daily_digest_store import inspect_daily_digest_plans
from .home_daily_report import inspect_home_daily_report
from .home_event_digest import HomeEventDigestError, read_home_event_digest
from .notification import NotificationOutbox, NotificationQueueError
from .report_queue import ReportQueueError, ReportRefreshQueue
from .signal_outbox import SignalOutbox, SignalOutboxError
from .visual_predicate_review import (
    VISUAL_PREDICATE_REVIEW_POLICY_VERSION,
    VisualPredicateReviewError,
    VisualPredicateReviewState,
    assess_visual_predicate_review,
)
from .visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
    VisualPredicateShadowQueueError,
)


class HomeControlError(RuntimeError):
    """Raised when a fixed local home-stack action cannot be completed safely."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HomeUnit:
    component: str
    unit: str
    group: str


SystemctlRunner = Callable[[Sequence[str], float], CommandResult]

_SYSTEMCTL = "/usr/bin/systemctl"
_MAX_COMMAND_OUTPUT = 16 * 1024
_SAFE_ATOM = re.compile(r"[A-Za-z0-9_.:@-]{1,160}\Z")
_SHOW_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "UnitFileState")

_MEDIA_GATEWAY = HomeUnit(
    component="xiaomi_stream_gateway",
    unit="xiaomi-stream-gateway.service",
    group="base",
)
_HA_SIGNALS = HomeUnit(
    component="home_assistant_signals",
    unit="baby-monitor-ha-signals.service",
    group="base",
)
_EVIDENCE_BUFFER = HomeUnit(
    component="local_evidence_buffer",
    unit="baby-monitor-home_shadow-camera_01-buffer.service",
    group="base",
)
_EVIDENCE_BUFFER_CAMERA_02 = HomeUnit(
    component="local_evidence_buffer_camera_02",
    unit="baby-monitor-home_shadow-camera_02-buffer.service",
    group="base",
)
_EVIDENCE_BUFFER_CAMERA_03 = HomeUnit(
    component="local_evidence_buffer_camera_03",
    unit="baby-monitor-home_shadow-camera_03-buffer.service",
    group="base",
)
_EVIDENCE_BUFFER_CAMERA_04 = HomeUnit(
    component="local_evidence_buffer_camera_04",
    unit="baby-monitor-home_shadow-camera_04-buffer.service",
    group="base",
)
_HA_SIGNAL_DISPATCH = HomeUnit(
    component="ha_signal_buffer_dispatch",
    unit="baby-monitor-ha-signal-dispatch.service",
    group="base",
)
_SHADOW_STACK = HomeUnit(
    component="bounded_home_shadow",
    unit="baby-monitor-home_shadow.target",
    group="shadow",
)
_CANDIDATE_SOAK = HomeUnit(
    component="candidate_soak_24h",
    unit="baby-monitor-candidate-soak.target",
    group="soak",
)
_CANDIDATE_SOAK_STOP = HomeUnit(
    component="candidate_soak_stop",
    unit="baby-monitor-candidate-soak-stop.service",
    group="internal",
)
_MAGE_SHADOW_ONCE = HomeUnit(
    component="int8_mage_shadow_once",
    unit="baby-monitor-mage-shadow-once.service",
    group="manual_once",
)
_DASHBOARD = HomeUnit(
    component="local_dashboard",
    unit="baby-monitor-dashboard.service",
    group="interface",
)

HOME_UNITS = (
    _MEDIA_GATEWAY,
    _HA_SIGNALS,
    _EVIDENCE_BUFFER,
    _EVIDENCE_BUFFER_CAMERA_02,
    _EVIDENCE_BUFFER_CAMERA_03,
    _EVIDENCE_BUFFER_CAMERA_04,
    _HA_SIGNAL_DISPATCH,
    _SHADOW_STACK,
    _CANDIDATE_SOAK,
    _MAGE_SHADOW_ONCE,
    _DASHBOARD,
)
BASE_UNITS = (
    _MEDIA_GATEWAY,
    _HA_SIGNALS,
    _EVIDENCE_BUFFER,
    _EVIDENCE_BUFFER_CAMERA_02,
    _EVIDENCE_BUFFER_CAMERA_03,
    _EVIDENCE_BUFFER_CAMERA_04,
    _HA_SIGNAL_DISPATCH,
)
SHADOW_UNITS = (_SHADOW_STACK,)
SOAK_UNITS = (_CANDIDATE_SOAK,)
MANUAL_ONCE_UNITS = (_MAGE_SHADOW_ONCE,)
HOME_CONTROL_ACTIONS = (
    "status",
    "start",
    "stop",
    "shadow-start",
    "shadow-stop",
    "soak-start",
    "soak-stop",
    "int8-shadow-once",
    "dashboard-start",
    "dashboard-stop",
)


def _default_runner(argv: Sequence[str], timeout_seconds: float) -> CommandResult:
    try:
        completed = subprocess.run(
            tuple(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HomeControlError("home control command unavailable") from exc
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _bounded_result(result: CommandResult) -> CommandResult:
    if not isinstance(result.returncode, int):
        raise HomeControlError("home control command returned an invalid result")
    if not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
        raise HomeControlError("home control command returned an invalid result")
    if len(result.stdout.encode("utf-8")) > _MAX_COMMAND_OUTPUT:
        raise HomeControlError("home control command output exceeded the limit")
    if len(result.stderr.encode("utf-8")) > _MAX_COMMAND_OUTPUT:
        raise HomeControlError("home control command output exceeded the limit")
    if "\x00" in result.stdout or "\x00" in result.stderr:
        raise HomeControlError("home control command returned an invalid result")
    return result


def _run(
    argv: Sequence[str],
    *,
    runner: SystemctlRunner,
    timeout_seconds: float = 30.0,
) -> CommandResult:
    return _bounded_result(runner(tuple(argv), timeout_seconds))


def _parse_show_output(output: str, *, expected_unit: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in output.splitlines():
        if not line or "=" not in line:
            raise HomeControlError("home control status returned an invalid result")
        key, value = line.split("=", 1)
        if key not in _SHOW_PROPERTIES or key in properties:
            raise HomeControlError("home control status returned an invalid result")
        if not _SAFE_ATOM.fullmatch(value):
            raise HomeControlError("home control status returned an invalid result")
        properties[key] = value
    if set(properties) != set(_SHOW_PROPERTIES):
        raise HomeControlError("home control status returned an incomplete result")
    if properties["Id"] != expected_unit:
        raise HomeControlError("home control status returned the wrong unit")
    return properties


def _unavailable_status(spec: HomeUnit) -> dict[str, object]:
    return {
        "component": spec.component,
        "unit": spec.unit,
        "group": spec.group,
        "installed": False,
        "enabled": False,
        "active": False,
        "state": "unavailable",
        "attention_required": True,
    }


def _unit_status(spec: HomeUnit, *, runner: SystemctlRunner) -> dict[str, object]:
    argv = (
        _SYSTEMCTL,
        "--user",
        "show",
        spec.unit,
        *(f"--property={name}" for name in _SHOW_PROPERTIES),
        "--no-pager",
    )
    try:
        result = _run(argv, runner=runner, timeout_seconds=10.0)
        if result.returncode != 0:
            return _unavailable_status(spec)
        properties = _parse_show_output(result.stdout, expected_unit=spec.unit)
    except HomeControlError:
        return _unavailable_status(spec)

    installed = properties["LoadState"] == "loaded"
    enabled = properties["UnitFileState"] in {"enabled", "enabled-runtime"}
    active = properties["ActiveState"] == "active"
    if not installed:
        state = "not_installed"
    elif active:
        state = properties["SubState"]
    elif properties["ActiveState"] == "failed":
        state = "failed"
    else:
        state = "inactive"
    return {
        "component": spec.component,
        "unit": spec.unit,
        "group": spec.group,
        "installed": installed,
        "enabled": enabled,
        "active": active,
        "state": state,
        "attention_required": not installed or properties["ActiveState"] == "failed",
    }


def _ha_outbox_status(database_path: str | Path | None) -> dict[str, object]:
    if database_path is None:
        return {
            "status": "not_configured",
            "operational": True,
            "attention_required": False,
            "paths_disclosed": False,
            "payloads_disclosed": False,
        }
    try:
        payload = SignalOutbox(database_path).snapshot_read_only().to_public_dict()
    except (OSError, SignalOutboxError):
        return {
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
            "paths_disclosed": False,
            "payloads_disclosed": False,
        }
    return {
        **payload,
        "operational": True,
        "attention_required": bool(payload["failed"]),
    }


def _analysis_queue_status(database_path: str | Path | None) -> dict[str, object]:
    if database_path is None:
        return {
            "version": 1,
            "status": "not_configured",
            "operational": True,
            "attention_required": False,
            "paths_disclosed": False,
            "media_read": False,
        }
    try:
        payload = AnalysisQueue(database_path).snapshot_read_only().to_dict()
    except (OSError, AnalysisQueueError, ValueError):
        return {
            "version": 1,
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
            "paths_disclosed": False,
            "media_read": False,
        }
    return {
        "version": 1,
        "status": "available",
        **payload,
        "operational": True,
        "attention_required": bool(payload["failed"] or payload["missing_media_files"]),
        "paths_disclosed": False,
        "media_read": False,
    }


def _candidate_health_status(
    health_paths: Sequence[str | Path] | None,
    *,
    stale_after_seconds: float,
) -> dict[str, object]:
    common = {
        "version": 1,
        "media_read": False,
        "paths_disclosed": False,
        "identity_inference": False,
        "risk_assessment": False,
        "notification_authority": False,
    }
    if not health_paths:
        return {
            **common,
            "status": "not_configured",
            "operational": True,
            "attention_required": False,
            "cameras": [],
        }
    paths = tuple(Path(value).expanduser() for value in health_paths)
    if len(paths) > 16 or len(paths) != len(set(paths)):
        return {
            **common,
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
            "cameras": [],
        }
    if all(not path.exists() and not path.is_symlink() for path in paths):
        return {
            **common,
            "status": "not_initialized",
            "operational": True,
            "attention_required": False,
            "cameras": [],
        }
    checked_at = datetime.now(UTC)
    try:
        assessments = tuple(
            assess_candidate_health(
                CandidateWorkerHealth.read_private(path),
                checked_at=checked_at,
                stale_after_seconds=stale_after_seconds,
            )
            for path in paths
        )
    except (CandidateHealthError, OSError, ValueError):
        return {
            **common,
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
            "cameras": [],
        }
    camera_ids = [item.health.camera_id for item in assessments]
    if len(camera_ids) != len(set(camera_ids)):
        return {
            **common,
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
            "cameras": [],
        }
    operational = all(item.operational for item in assessments)
    attention = any(
        item.health.state is CandidateHealthState.FAILED
        or item.freshness is CandidateHealthFreshness.CLOCK_SKEW
        for item in assessments
    )
    cameras: list[dict[str, object]] = []
    for item in assessments:
        camera = item.to_dict()
        camera["history_summary"] = (
            item.health.lifetime.summary() if item.health.lifetime is not None else None
        )
        cameras.append(camera)
    return {
        **common,
        "status": "available",
        "operational": operational,
        "attention_required": attention,
        "cameras": cameras,
    }


def _candidate_soak_status(ledger_path: str | Path | None) -> dict[str, object]:
    if ledger_path is None:
        return {
            "version": 1,
            "status": "not_configured",
            "session_count": 0,
            "latest": None,
            "attention_required": False,
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }
    try:
        return inspect_candidate_soak(
            ledger_path=ledger_path,
            checked_at=datetime.now(UTC),
        )
    except (CandidateSoakError, OSError, ValueError):
        return {
            "version": 1,
            "status": "unavailable",
            "session_count": 0,
            "latest": None,
            "attention_required": True,
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }


def _visual_predicate_shadow_status(
    database_path: str | Path | None,
) -> dict[str, object]:
    empty = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "active": 0,
        "ready_pending": 0,
        "oldest_pending_age_seconds": None,
        "failed_error_counts": {},
        "recent_answers": [],
        "review_policy_version": VISUAL_PREDICATE_REVIEW_POLICY_VERSION,
        "recent_assessed_count": 0,
        "recent_review_recommended": 0,
        "recent_unknown": 0,
        "recent_no_rule_match": 0,
    }
    if database_path is None:
        return {
            "version": 1,
            "status": "not_configured",
            **empty,
            "operational": True,
            "attention_required": False,
            "paths_disclosed": False,
            "media_read": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
    try:
        queue = VisualPredicateShadowAttemptQueue(database_path)
        if not queue.is_initialized_read_only():
            return {
                "version": 1,
                "status": "not_initialized",
                **empty,
                "operational": True,
                "attention_required": False,
                "paths_disclosed": False,
                "media_read": False,
                "subject_identity_inference": False,
                "risk_inference": False,
                "notification_decision": False,
            }
        payload = queue.snapshot(read_only=True).to_dict()
        recent_answers: list[dict[str, object]] = []
        review_counts = {
            VisualPredicateReviewState.REVIEW_RECOMMENDED: 0,
            VisualPredicateReviewState.UNKNOWN: 0,
            VisualPredicateReviewState.NO_RULE_MATCH: 0,
        }
        for observation in queue.recent_observations_read_only(limit=6):
            if not observation.results:
                continue
            recommendation = assess_visual_predicate_review(observation)
            review_counts[recommendation.state] += 1
            recent_answers.append(
                {
                    "recorded_at": (
                        observation.recorded_at.astimezone().isoformat()
                        if observation.recorded_at is not None
                        else None
                    ),
                    "hardware_signal_gate": bool(observation.attempt.gate.hardware_signal_count),
                    "person_detection_gate": bool(observation.attempt.gate.person_detection_count),
                    "answers": [
                        {
                            "predicate": result.predicate.value,
                            "answer": result.answer.value,
                            "confidence": result.confidence,
                        }
                        for result in observation.results
                    ],
                    "review_state": recommendation.state.value,
                    "review_reasons": [item.value for item in recommendation.reasons],
                    "manual_review_recommended": (recommendation.manual_review_recommended),
                }
            )
    except (
        OSError,
        VisualPredicateReviewError,
        VisualPredicateShadowQueueError,
        ValueError,
    ):
        return {
            "version": 1,
            "status": "unavailable",
            **empty,
            "operational": False,
            "attention_required": True,
            "paths_disclosed": False,
            "media_read": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
    return {
        "version": 1,
        "status": "available",
        **payload,
        "recent_answers": recent_answers,
        "review_policy_version": VISUAL_PREDICATE_REVIEW_POLICY_VERSION,
        "recent_assessed_count": sum(review_counts.values()),
        "recent_review_recommended": review_counts[VisualPredicateReviewState.REVIEW_RECOMMENDED],
        "recent_unknown": review_counts[VisualPredicateReviewState.UNKNOWN],
        "recent_no_rule_match": review_counts[VisualPredicateReviewState.NO_RULE_MATCH],
        "operational": True,
        "attention_required": bool(payload["failed"]),
        "paths_disclosed": False,
        "media_read": False,
        "subject_identity_inference": False,
        "risk_inference": False,
        "notification_decision": False,
    }


def _event_digest_status(database_path: str | Path | None) -> dict[str, object]:
    if database_path is None:
        return {
            "version": 1,
            "status": "not_configured",
            "operational": True,
            "attention_required": False,
            "media_read": False,
            "paths_disclosed": False,
        }
    try:
        payload = read_home_event_digest(database_path).to_public_dict()
    except (OSError, HomeEventDigestError, ValueError):
        return {
            "version": 1,
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
            "media_read": False,
            "paths_disclosed": False,
        }
    return {
        **payload,
        "operational": True,
        "attention_required": False,
    }


def _notification_queue_status(
    database_path: str | Path | None,
    *,
    enabled: bool,
    external_delivery_enabled: bool,
    max_pending_age_seconds: int,
) -> dict[str, object]:
    empty = {
        "pending": 0,
        "running": 0,
        "delivered": 0,
        "failed": 0,
        "active": 0,
        "ready_pending": 0,
        "oldest_pending_age_seconds": None,
        "failed_error_counts": {},
    }
    common = {
        "version": 1,
        "enabled": enabled,
        "external_delivery_enabled": external_delivery_enabled,
        "local_test_available": enabled,
        "media_included": False,
        "paths_disclosed": False,
    }
    if database_path is None:
        return {
            **common,
            "status": "not_configured",
            **empty,
            "operational": True,
            "attention_required": False,
        }
    try:
        outbox = NotificationOutbox(database_path)
        if not outbox.is_initialized_read_only():
            return {
                **common,
                "status": "not_initialized",
                **empty,
                "operational": True,
                "attention_required": False,
            }
        payload = outbox.snapshot_read_only().to_dict()
    except (OSError, NotificationQueueError, ValueError):
        return {
            **common,
            "status": "unavailable",
            **empty,
            "operational": False,
            "attention_required": True,
        }
    oldest = payload["oldest_pending_age_seconds"]
    too_old = isinstance(oldest, (int, float)) and oldest > max_pending_age_seconds
    return {
        **common,
        "status": "available",
        **payload,
        "operational": True,
        "attention_required": bool(payload["failed"] or too_old),
    }


def _report_refresh_status(database_path: str | Path | None) -> dict[str, object]:
    """Inspect optional detailed-report work, never generate or retry on a read."""
    common: dict[str, object] = {
        "version": 1,
        "counts": None,
        "ready": None,
        "expired_leases": None,
        "dirty_jobs": None,
        "oldest_pending_age_seconds": None,
        "max_pending_age_seconds": 3600,
        "media_read": False,
        "paths_disclosed": False,
        "job_ids_disclosed": False,
        "worker_liveness_verified": False,
    }
    if database_path is None:
        return {
            **common,
            "status": "not_configured",
            "operational": True,
            "attention_required": False,
        }
    try:
        database = Path(database_path).expanduser()
        if not database.is_file():
            raise ReportQueueError("report queue database is unavailable")
        snapshot = ReportRefreshQueue(database).snapshot_read_only()
    except (OSError, ValueError, ReportQueueError):
        return {
            **common,
            "status": "unavailable",
            "operational": False,
            "attention_required": True,
        }
    if not snapshot["table_present"]:
        return {
            **common,
            "status": "not_initialized",
            "operational": True,
            "attention_required": False,
        }
    counts = snapshot["counts"]
    age = snapshot["oldest_pending_age_seconds"]
    attention = bool(
        counts["failed"] or snapshot["expired_leases"] or (age is not None and age > 3600)
    )
    return {
        **common,
        "status": "available",
        "counts": counts,
        "ready": snapshot["ready"],
        "expired_leases": snapshot["expired_leases"],
        "dirty_jobs": snapshot["dirty_jobs"],
        "oldest_pending_age_seconds": age,
        "operational": True,
        "attention_required": attention,
    }


def _daily_report_status(
    database_path: str | Path | None,
    output_root: str | Path | None,
    *,
    timezone_name: str,
) -> dict[str, object]:
    if database_path is None or output_root is None:
        return {
            "version": 1,
            "status": "not_configured",
            "local_date": None,
            "event_count": 0,
            "available": False,
            "operational": True,
            "attention_required": False,
            "media_read": False,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "summaries_disclosed": False,
            "risk_authority": False,
            "notification_decision": False,
        }
    return inspect_home_daily_report(
        database_path=database_path,
        output_root=output_root,
        timezone_name=timezone_name,
    )


def _daily_digest_status(database_path: str | Path | None) -> dict[str, object]:
    if database_path is None:
        return {
            "version": 1,
            "status": "not_configured",
            "plan_available": False,
            "video_available": False,
            "operational": True,
            "attention_required": False,
            "media_read": False,
            "paths_disclosed": False,
        }
    database = Path(database_path).expanduser()
    plan = inspect_daily_digest_plans(
        database_path=database,
        output_root=database.parent / "daily-digests",
    )
    export = inspect_daily_digest_exports(
        database_path=database,
        output_root=database.parent / "daily-digest-exports",
    )
    video_available = export.get("video_available") is True
    plan_available = plan.get("available") is True
    operational = plan.get("operational") is True and export.get("operational") is True
    return {
        "version": 1,
        "status": "available"
        if video_available
        else "planned"
        if plan_available
        else "not_generated"
        if operational
        else "unavailable",
        "plan_available": plan_available,
        "video_available": video_available,
        "latest_local_date": (
            export.get("latest_local_date") if video_available else plan.get("latest_local_date")
        ),
        "latest_selected_event_count": (
            export.get("latest_selected_event_count")
            if video_available
            else plan.get("latest_selected_event_count")
        ),
        "latest_duration_seconds": (
            export.get("latest_output_duration_seconds")
            if video_available
            else plan.get("latest_selected_duration_seconds")
        ),
        "operational": operational,
        "attention_required": not operational,
        "media_read": False,
        "paths_disclosed": False,
        "event_ids_disclosed": False,
        "camera_ids_disclosed": False,
        "identity_inference": False,
        "risk_inference": False,
        "notification_decision": False,
    }


def home_stack_status(
    *,
    runner: SystemctlRunner = _default_runner,
    ha_outbox_database: str | Path | None = None,
    analysis_database: str | Path | None = None,
    candidate_health_paths: Sequence[str | Path] | None = None,
    candidate_health_stale_after_seconds: float = 120.0,
    candidate_soak_ledger: str | Path | None = None,
    notification_enabled: bool = False,
    external_notification_delivery_enabled: bool = False,
    notification_max_pending_age_seconds: int = 3600,
    daily_report_output: str | Path | None = None,
    timezone_name: str = "Asia/Shanghai",
) -> dict[str, object]:
    components = tuple(_unit_status(spec, runner=runner) for spec in HOME_UNITS)
    by_name = {str(item["component"]): item for item in components}
    ha_outbox = _ha_outbox_status(ha_outbox_database)
    analysis_queue = _analysis_queue_status(analysis_database)
    candidate_health = _candidate_health_status(
        candidate_health_paths,
        stale_after_seconds=candidate_health_stale_after_seconds,
    )
    candidate_soak = _candidate_soak_status(candidate_soak_ledger)
    visual_predicate_shadow = _visual_predicate_shadow_status(analysis_database)
    event_digest = _event_digest_status(analysis_database)
    notification_queue = _notification_queue_status(
        analysis_database,
        enabled=notification_enabled,
        external_delivery_enabled=external_notification_delivery_enabled,
        max_pending_age_seconds=notification_max_pending_age_seconds,
    )
    daily_report = _daily_report_status(
        analysis_database,
        daily_report_output,
        timezone_name=timezone_name,
    )
    daily_digest = _daily_digest_status(analysis_database)
    base_operational = (
        all(bool(by_name[spec.component]["active"]) for spec in BASE_UNITS)
        and bool(ha_outbox["operational"])
        and bool(analysis_queue["operational"])
    )
    base_autostart_enabled = all(bool(by_name[spec.component]["enabled"]) for spec in BASE_UNITS)
    manual_start_only = all(not bool(by_name[spec.component]["enabled"]) for spec in BASE_UNITS)
    return {
        "version": 1,
        "base_operational": base_operational,
        "base_autostart_enabled": base_autostart_enabled,
        "manual_start_only": manual_start_only,
        "shadow_active": bool(by_name[_SHADOW_STACK.component]["active"]),
        "candidate_soak_active": bool(by_name[_CANDIDATE_SOAK.component]["active"]),
        "int8_shadow_active": bool(by_name[_MAGE_SHADOW_ONCE.component]["active"]),
        "components": list(components),
        "ha_signal_outbox": ha_outbox,
        "analysis_queue": analysis_queue,
        "candidate_health": candidate_health,
        "candidate_soak": candidate_soak,
        "visual_predicate_shadow": visual_predicate_shadow,
        "notification_queue": notification_queue,
        "event_digest": event_digest,
        "daily_report": daily_report,
        "report_refresh_queue": _report_refresh_status(analysis_database),
        "daily_digest": daily_digest,
        "dashboard": (
            "available" if bool(by_name[_DASHBOARD.component]["installed"]) else "not_installed"
        ),
        "web_review": "local_shadow_review",
        "remote_upload_enabled": False,
        "paths_disclosed": False,
        "secrets_disclosed": False,
    }


def _perform_action(
    verb: str,
    units: Sequence[HomeUnit],
    *,
    runner: SystemctlRunner,
    no_block: bool = False,
) -> None:
    if verb not in {"start", "stop"} or not units:
        raise HomeControlError("home control action is invalid")
    options = ("--no-block",) if no_block else ()
    argv = (
        _SYSTEMCTL,
        "--user",
        "--no-pager",
        *options,
        verb,
        *(spec.unit for spec in units),
    )
    result = _run(argv, runner=runner)
    if result.returncode != 0:
        raise HomeControlError("home control action failed")


def control_home_stack(
    action: str,
    *,
    runner: SystemctlRunner = _default_runner,
    ha_outbox_database: str | Path | None = None,
    analysis_database: str | Path | None = None,
    candidate_health_paths: Sequence[str | Path] | None = None,
    candidate_health_stale_after_seconds: float = 120.0,
    candidate_soak_ledger: str | Path | None = None,
    notification_enabled: bool = False,
    external_notification_delivery_enabled: bool = False,
    notification_max_pending_age_seconds: int = 3600,
    daily_report_output: str | Path | None = None,
    timezone_name: str = "Asia/Shanghai",
    confirm_buffer_eviction: bool = False,
) -> dict[str, object]:
    if action not in HOME_CONTROL_ACTIONS:
        raise HomeControlError("home control action is invalid")
    if type(confirm_buffer_eviction) is not bool:
        raise HomeControlError("buffer confirmation is invalid")
    if action in {"start", "shadow-start", "soak-start", "int8-shadow-once"} and not (
        confirm_buffer_eviction
    ):
        raise HomeControlError("manual buffer start requires one-shot confirmation")

    if action == "start":
        _perform_action("start", BASE_UNITS, runner=runner)
    elif action == "stop":
        _perform_action("start", (_CANDIDATE_SOAK_STOP,), runner=runner)
        _perform_action(
            "stop",
            (*MANUAL_ONCE_UNITS, *SHADOW_UNITS, *reversed(BASE_UNITS)),
            runner=runner,
        )
    elif action == "shadow-start":
        _perform_action("start", (_CANDIDATE_SOAK_STOP,), runner=runner)
        _perform_action("start", BASE_UNITS, runner=runner)
        _perform_action("start", SHADOW_UNITS, runner=runner)
    elif action == "shadow-stop":
        _perform_action("stop", (*MANUAL_ONCE_UNITS, *SHADOW_UNITS), runner=runner)
    elif action == "soak-start":
        _perform_action(
            "stop",
            (*MANUAL_ONCE_UNITS, *SHADOW_UNITS, *reversed(BASE_UNITS)),
            runner=runner,
        )
        _perform_action("start", SOAK_UNITS, runner=runner)
    elif action == "soak-stop":
        _perform_action("start", (_CANDIDATE_SOAK_STOP,), runner=runner)
    elif action == "int8-shadow-once":
        _perform_action("start", (_CANDIDATE_SOAK_STOP,), runner=runner)
        _perform_action("start", BASE_UNITS, runner=runner)
        _perform_action("start", MANUAL_ONCE_UNITS, runner=runner, no_block=True)
    elif action == "dashboard-start":
        _perform_action("start", (_DASHBOARD,), runner=runner)
    elif action == "dashboard-stop":
        _perform_action("stop", (_DASHBOARD,), runner=runner)

    status = home_stack_status(
        runner=runner,
        ha_outbox_database=ha_outbox_database,
        analysis_database=analysis_database,
        candidate_health_paths=candidate_health_paths,
        candidate_health_stale_after_seconds=candidate_health_stale_after_seconds,
        candidate_soak_ledger=candidate_soak_ledger,
        notification_enabled=notification_enabled,
        external_notification_delivery_enabled=external_notification_delivery_enabled,
        notification_max_pending_age_seconds=notification_max_pending_age_seconds,
        daily_report_output=daily_report_output,
        timezone_name=timezone_name,
    )
    return {
        **status,
        "action": action,
        "action_accepted": True,
        "persistent_enablement_changed": False,
    }
