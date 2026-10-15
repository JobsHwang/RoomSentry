"""Resident visual-predicate health, shadow queue, and calibration report commands."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from baby_monitor.cli_support import (  # noqa: F401
    _absolute_runner_executable,
    _aware_datetime,
    _configured_remote_adapter,
    _database_from_path,
    _enqueue_buffered_candidate,
    _json_print,
    _local_adapter,
    _remote_adapter,
    _review_policy,
    _secondary_adapter,
    _storage_override,
    _yolo_detector,
)

from ..adapters.visual_predicate import (
    ResidentCommandVisualPredicateAdapter,
    ResidentVisualPredicateHealth,
)
from ..analysis_worker import (
    WorkerAttempt,
    WorkerAttemptState,
    WorkerLoopSettings,
    WorkerStopReason,
    run_analysis_worker,
)
from ..config import load_settings
from ..home_shadow_review import HomeShadowReviewRepository
from ..visual_predicate_health import assess_resident_visual_predicate_health
from ..visual_predicate_review import build_visual_predicate_review_report
from ..visual_predicate_shadow import (
    VisualPredicateShadowStore,
    run_visual_predicate_shadow,
    select_visual_predicate_shadow,
)
from ..visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
    VisualPredicateShadowQueueFullError,
)
from ..visual_predicate_shadow_runtime import (
    admit_private_shadow_frame,
    build_visual_predicate_shadow_adapter,
    load_private_shadow_candidate,
    load_visual_predicate_shadow_runtime_settings,
)
from ..visual_predicate_shadow_worker import (
    VisualPredicateShadowWorkerStatus,
    close_visual_predicate_shadow_adapter,
    run_visual_predicate_shadow_attempt_once,
)


def _shadow_worker_attempt(result) -> WorkerAttempt:
    states = {
        VisualPredicateShadowWorkerStatus.IDLE: WorkerAttemptState.IDLE,
        VisualPredicateShadowWorkerStatus.COMPLETED: WorkerAttemptState.COMPLETED,
        VisualPredicateShadowWorkerStatus.RETRY_PENDING: WorkerAttemptState.RETRY_PENDING,
        VisualPredicateShadowWorkerStatus.FAILED: WorkerAttemptState.FAILED,
    }
    return WorkerAttempt(
        states[result.status],
        event_id=result.event_id,
        error_code=result.error_code,
    )


def command_visual_predicate_health_status(args: argparse.Namespace) -> int:
    health = ResidentVisualPredicateHealth.read_private(args.health)
    assessment = assess_resident_visual_predicate_health(
        health,
        stale_after_seconds=args.stale_after_seconds,
        future_tolerance_seconds=args.future_tolerance_seconds,
    )
    _json_print(
        {
            "version": 1,
            "health": health.to_dict(),
            "assessment": assessment.to_dict(),
            "health_path_disclosed": False,
            "media_metadata_included": False,
        }
    )
    return 0 if assessment.operational else 1


def command_visual_predicate_shadow_run(args: argparse.Namespace) -> int:
    runtime = load_visual_predicate_shadow_runtime_settings(args.config)
    if not runtime.enabled:
        raise ValueError("visual predicate shadow is disabled")
    candidate = load_private_shadow_candidate(args.candidate)
    if (
        args.person_detector_id is not None
        and candidate.detector_id is not None
        and args.person_detector_id != candidate.detector_id
    ):
        raise ValueError("visual predicate shadow detector provenance conflicts")
    person_detector_id = args.person_detector_id or candidate.detector_id
    gate = select_visual_predicate_shadow(
        candidate,
        predicates=runtime.predicates,
        person_detector_id=person_detector_id,
        hardware_signal_source_adapters=runtime.hardware_signal_adapters,
        periodic_sample_due=args.periodic_sample_due,
        min_person_confidence=runtime.min_person_confidence,
    )
    if not gate.selected:
        _json_print(
            {
                "version": 1,
                "gate": gate.to_dict(),
                "results": [],
                "persisted": False,
                "reused_existing": False,
                "subject_identity_inference": False,
                "risk_inference": False,
                "notification_decision": False,
                "paths_disclosed": False,
                "media_metadata_included": False,
                "os_network_isolation_enforced": False,
            }
        )
        return 0

    frames = tuple(admit_private_shadow_frame(value) for value in args.frame)
    if not 1 <= len(frames) <= 12 or len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("visual predicate shadow frames are invalid")

    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    store = VisualPredicateShadowStore(database_path)
    adapter = build_visual_predicate_shadow_adapter(runtime)
    try:
        summary = run_visual_predicate_shadow(
            gate=gate,
            frames=frames,
            adapter=adapter,
            store_factory=lambda: store,
        )
    finally:
        if isinstance(adapter, ResidentCommandVisualPredicateAdapter):
            adapter.close()
    payload = summary.to_dict()
    payload["os_network_isolation_enforced"] = False
    _json_print(payload)
    return 0


def command_visual_predicate_shadow_queue_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    snapshot = VisualPredicateShadowAttemptQueue(database_path).snapshot(read_only=True)
    _json_print(
        {
            "version": 1,
            "queue": snapshot.to_dict(),
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
    )
    return 0


def command_visual_predicate_shadow_report(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    observations = VisualPredicateShadowAttemptQueue(database_path).recent_observations_read_only(
        limit=args.limit
    )
    _json_print(
        {
            "version": 1,
            "observation_count": len(observations),
            "observations": [item.to_dict() for item in observations],
            "paths_disclosed": False,
            "media_read": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
    )
    return 0


def command_visual_predicate_review_report(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    observations = VisualPredicateShadowAttemptQueue(database_path).recent_observations_read_only(
        limit=args.limit
    )
    report = build_visual_predicate_review_report(observations)
    _json_print(report.to_dict())
    return 0


def command_visual_predicate_human_calibration_report(
    args: argparse.Namespace,
) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    repository = HomeShadowReviewRepository(
        database_path,
        token_secret=secrets.token_bytes(32),
    )
    _json_print(repository.calibration_report(limit=args.limit))
    return 0


def command_visual_predicate_shadow_retry(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    try:
        attempt = VisualPredicateShadowAttemptQueue(database_path).retry_failed(
            args.event_id,
            max_active=args.max_active,
        )
    except VisualPredicateShadowQueueFullError:
        raise ValueError(
            "visual predicate shadow attempt queue is full; failed event remains tracked"
        ) from None
    _json_print(
        {
            "version": 1,
            "operation": "retry",
            "attempt": attempt.to_public_dict(),
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
    )
    return 0


def command_visual_predicate_shadow_run_next(args: argparse.Namespace) -> int:
    runtime = load_visual_predicate_shadow_runtime_settings(args.config)
    if not runtime.enabled:
        raise ValueError("visual predicate shadow is disabled")
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    adapter = build_visual_predicate_shadow_adapter(runtime)
    try:
        result = run_visual_predicate_shadow_attempt_once(
            database_path=database_path,
            runtime=runtime,
            adapter=adapter,
            max_active=args.max_active,
            lease_seconds=args.lease_seconds,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    finally:
        close_visual_predicate_shadow_adapter(adapter)
    payload = result.to_dict()
    payload["os_network_isolation_enforced"] = False
    _json_print(payload)
    return (
        0
        if result.status
        in {
            VisualPredicateShadowWorkerStatus.IDLE,
            VisualPredicateShadowWorkerStatus.COMPLETED,
        }
        else 2
    )


def command_visual_predicate_shadow_worker(args: argparse.Namespace) -> int:
    runtime = load_visual_predicate_shadow_runtime_settings(args.config)
    if not runtime.enabled:
        raise ValueError("visual predicate shadow is disabled")
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    max_jobs = args.max_jobs if args.max_jobs is not None else runtime.max_jobs
    if runtime.backend == "resident_command" and (max_jobs == 0 or max_jobs > runtime.max_jobs):
        raise ValueError("visual predicate shadow worker exceeds the resident job limit")
    loop_settings = WorkerLoopSettings(
        max_jobs=max_jobs,
        idle_exit_seconds=(
            args.idle_exit_seconds
            if args.idle_exit_seconds is not None
            else runtime.idle_exit_seconds
        ),
        max_runtime_seconds=(
            args.max_runtime_seconds
            if args.max_runtime_seconds is not None
            else runtime.max_runtime_seconds
        ),
        poll_seconds=args.poll_seconds,
    )
    adapter = build_visual_predicate_shadow_adapter(runtime)

    def run_once() -> WorkerAttempt:
        result = run_visual_predicate_shadow_attempt_once(
            database_path=database_path,
            runtime=runtime,
            adapter=adapter,
            max_active=args.max_active,
            lease_seconds=args.lease_seconds,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        return _shadow_worker_attempt(result)

    try:
        summary = run_analysis_worker(
            settings=loop_settings,
            run_once=run_once,
            queue_snapshot=lambda: (
                VisualPredicateShadowAttemptQueue(database_path).snapshot().to_dict()
            ),
        )
    finally:
        close_visual_predicate_shadow_adapter(adapter)
    _json_print(
        {
            "version": 1,
            "status": "stopped",
            "worker": summary.to_dict(),
            "os_network_isolation_enforced": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "paths_disclosed": False,
        }
    )
    if summary.stop_reason is WorkerStopReason.INTERRUPTED:
        return 130
    return 2 if summary.failed_jobs else 0
