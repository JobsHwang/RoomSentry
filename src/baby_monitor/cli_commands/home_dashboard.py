"""Systemd render, home control, daily report, digest, and dashboard commands."""

from __future__ import annotations

import argparse
import secrets
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

from ..candidate_health import (
    CandidateWorkerHealth,
)
from ..candidate_presence import (
    candidate_health_gap_sample,
    merge_stream_presence_batches,
)
from ..config import load_settings
from ..daily_digest import DailyDigestConfig, plan_daily_digest
from ..daily_digest_export import (
    DailyDigestCatalog,
    export_daily_digest,
    inspect_daily_digest_exports,
)
from ..daily_digest_source import (
    latest_daily_digest_local_date,
    load_daily_digest_candidates,
)
from ..daily_digest_store import (
    inspect_daily_digest_plans,
    load_latest_daily_digest_plan,
    write_daily_digest_plan,
)
from ..event_policy import EventPolicyError, HouseholdEventPolicy
from ..home_control import control_home_stack
from ..home_daily_report import (
    generate_home_daily_report,
    read_home_daily_report_html,
)
from ..home_dashboard import HomeDashboardApplication, serve_home_dashboard
from ..home_notification import run_home_local_notification_test
from ..home_shadow_review import HomeShadowReviewRepository
from ..room_presence import StreamPresenceBatch, evaluate_room_presence
from ..room_registry import RoomRegistry
from ..systemd_deployment import RuntimeDeploymentProfile, render_systemd_user_units


def command_systemd_render(args: argparse.Namespace) -> int:
    profile = RuntimeDeploymentProfile.read(args.profile)
    rendered = render_systemd_user_units(profile, args.output_dir)
    _json_print(rendered.to_public_dict())
    return 0


def command_home_control(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    daily_report_output = (
        Path(args.analysis_database).expanduser().parent / "daily-reports"
        if args.analysis_database
        else None
    )
    _json_print(
        control_home_stack(
            args.action,
            ha_outbox_database=args.ha_outbox_database,
            analysis_database=args.analysis_database,
            candidate_health_paths=args.candidate_health,
            candidate_health_stale_after_seconds=(args.candidate_health_stale_after_seconds),
            candidate_soak_ledger=args.candidate_soak_ledger,
            notification_enabled=settings.notification.enabled,
            external_notification_delivery_enabled=(
                settings.notification.external_delivery_enabled
            ),
            notification_max_pending_age_seconds=(settings.notification.max_pending_age_seconds),
            daily_report_output=daily_report_output,
            timezone_name=settings.timezone,
            confirm_buffer_eviction=args.confirm_buffer_eviction,
        )
    )
    return 0


def command_home_daily_report(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database or settings.storage.database).expanduser()
    receipt = generate_home_daily_report(
        database_path=database_path,
        output_root=database_path.parent / "daily-reports",
        timezone_name=settings.timezone,
    )
    _json_print(receipt.to_public_dict())
    return 0


def command_daily_digest_plan(args: argparse.Namespace) -> int:
    if not secrets.compare_digest(args.confirmation.upper(), "PLAN"):
        raise ValueError("daily digest planning requires PLAN confirmation")
    settings = load_settings(args.config)
    database_path = Path(args.database or settings.storage.database).expanduser()
    storage_root = Path(args.storage_root or settings.storage.local_root).expanduser()
    local_date = (
        date.fromisoformat(args.date)
        if args.date
        else datetime.now(ZoneInfo(settings.timezone)).date()
    )
    admission = load_daily_digest_candidates(
        database_path=database_path,
        storage_root=storage_root,
        local_date=local_date,
        timezone_name=settings.timezone,
    )
    plan = plan_daily_digest(
        admission.candidates,
        local_date=local_date,
        timezone_name=settings.timezone,
        config=DailyDigestConfig(target_duration_seconds=args.target_duration_seconds),
    )
    receipt = write_daily_digest_plan(
        plan,
        database_path=database_path,
        output_root=database_path.parent / "daily-digests",
    )
    _json_print(
        {
            "version": 1,
            "status": "stored",
            "admission": admission.to_public_dict(),
            "plan": receipt.to_public_dict(),
            "facts_only": True,
            "media_content_decoded": False,
            "video_generated": False,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
    )
    return 0


def command_daily_digest_plan_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database or settings.storage.database).expanduser()
    _json_print(
        inspect_daily_digest_plans(
            database_path=database_path,
            output_root=database_path.parent / "daily-digests",
        )
    )
    return 0


def command_daily_digest_export(args: argparse.Namespace) -> int:
    if not secrets.compare_digest(args.confirmation.upper(), "EXPORT"):
        raise ValueError("daily digest export requires EXPORT confirmation")
    settings = load_settings(args.config)
    database_path = Path(args.database or settings.storage.database).expanduser()
    local_date = (
        date.fromisoformat(args.date)
        if args.date
        else datetime.now(ZoneInfo(settings.timezone)).date()
    )
    plan = load_latest_daily_digest_plan(
        database_path=database_path,
        output_root=database_path.parent / "daily-digests",
        local_date=local_date.isoformat(),
    )
    admission = load_daily_digest_candidates(
        database_path=database_path,
        storage_root=Path(args.storage_root or settings.storage.local_root).expanduser(),
        local_date=local_date,
        timezone_name=settings.timezone,
    )
    receipt = export_daily_digest(
        plan,
        admission,
        database_path=database_path,
        output_root=database_path.parent / "daily-digest-exports",
        storage_root=Path(args.storage_root or settings.storage.local_root).expanduser(),
    )
    _json_print(receipt.to_public_dict())
    return 0


def command_daily_digest_export_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database or settings.storage.database).expanduser()
    _json_print(
        inspect_daily_digest_exports(
            database_path=database_path,
            output_root=database_path.parent / "daily-digest-exports",
        )
    )
    return 0


def command_home_dashboard(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    review_secret = secrets.token_bytes(32)
    daily_digest_secret = secrets.token_bytes(32)
    daily_report_output = (
        Path(args.analysis_database).expanduser().parent / "daily-reports"
        if args.analysis_database
        else None
    )
    daily_digest_output = (
        Path(args.analysis_database).expanduser().parent / "daily-digest-exports"
        if args.analysis_database
        else None
    )

    def room_presence_status() -> dict[str, object]:
        if not args.room_registry:
            return {
                "version": 1,
                "status": "not_configured",
                "rooms": [],
                "media_read": False,
                "paths_disclosed": False,
                "identity_inference": False,
                "risk_assessment": False,
                "notification_authority": False,
            }
        try:
            registry = RoomRegistry.read_private(args.room_registry)
            batches = tuple(
                StreamPresenceBatch.read_private(path)
                for path in (args.room_presence_samples or ())
            )
            gap_samples = tuple(
                sample
                for sample in (
                    candidate_health_gap_sample(CandidateWorkerHealth.read_private(path))
                    for path in (args.room_presence_health or ())
                )
                if sample is not None
            )
            batch = merge_stream_presence_batches(
                batches,
                extra_samples=gap_samples,
            )
            report = evaluate_room_presence(
                registry=registry,
                batch=batch,
                checked_at=datetime.now(UTC),
            )
        except (OSError, ValueError):
            return {
                "version": 1,
                "status": "unavailable",
                "rooms": [],
                "media_read": False,
                "paths_disclosed": False,
                "identity_inference": False,
                "risk_assessment": False,
                "notification_authority": False,
            }
        payload = report.to_dict()
        payload["status"] = "available"
        return payload

    def provide_status() -> dict[str, object]:
        payload = control_home_stack(
            "status",
            ha_outbox_database=args.ha_outbox_database,
            analysis_database=args.analysis_database,
            candidate_health_paths=args.candidate_health,
            candidate_health_stale_after_seconds=(args.candidate_health_stale_after_seconds),
            candidate_soak_ledger=args.candidate_soak_ledger,
            notification_enabled=settings.notification.enabled,
            external_notification_delivery_enabled=(
                settings.notification.external_delivery_enabled
            ),
            notification_max_pending_age_seconds=(settings.notification.max_pending_age_seconds),
            daily_report_output=daily_report_output,
            timezone_name=settings.timezone,
        )
        payload["room_presence"] = room_presence_status()
        payload["event_policy"] = event_policy_status()
        return payload

    def event_policy_status() -> dict[str, object]:
        if not args.event_policy:
            return {
                "version": 1,
                "status": "not_configured",
                "risk_inference": False,
                "notification_delivery_authority": False,
                "paths_disclosed": False,
            }
        try:
            policy = HouseholdEventPolicy.read(args.event_policy)
        except EventPolicyError:
            return {
                "version": 1,
                "status": "invalid",
                "risk_inference": False,
                "notification_delivery_authority": False,
                "paths_disclosed": False,
            }
        summary = policy.to_public_summary()
        summary["status"] = "configured"
        summary["policy_sha256"] = policy.fingerprint
        return summary

    def review_repository() -> HomeShadowReviewRepository:
        if not args.analysis_database:
            raise ValueError("shadow review database is not configured")
        return HomeShadowReviewRepository(
            args.analysis_database,
            token_secret=review_secret,
        )

    def provide_reviews(limit: int) -> tuple[dict[str, object], ...]:
        return tuple(item.to_public_dict() for item in review_repository().list_items(limit=limit))

    def provide_review_calibration() -> dict[str, object]:
        return review_repository().calibration_report(limit=1_000)

    def provide_review_media(
        review_token: str,
        media_kind: str,
        index: int,
    ) -> tuple[str, bytes]:
        media = review_repository().read_media(
            review_token,
            media_kind=media_kind,
            index=index,
        )
        return media.content_type, media.payload

    def record_review_label(
        review_token: str,
        predicate: str,
        human_answer: str,
    ) -> dict[str, object]:
        return (
            review_repository()
            .record_label(
                review_token,
                predicate=predicate,
                human_answer=human_answer,
            )
            .to_public_dict()
        )

    review_callbacks = (
        {
            "review_provider": provide_reviews,
            "review_media_provider": provide_review_media,
            "review_label_recorder": record_review_label,
            "review_calibration_provider": provide_review_calibration,
        }
        if args.analysis_database
        else {}
    )

    def run_notification_test() -> dict[str, object]:
        if not args.analysis_database or not settings.notification.enabled:
            raise ValueError("local notification test is disabled")
        database_path = Path(args.analysis_database).expanduser()
        return run_home_local_notification_test(
            database_path=database_path,
            target_root=database_path.parent / "local-notifications",
            max_pending=settings.notification.max_pending,
            lease_seconds=settings.notification.lease_seconds,
            retry_delay_seconds=settings.notification.retry_delay_seconds,
            max_attempts=settings.notification.max_attempts,
        ).to_public_dict()

    def generate_daily_report() -> dict[str, object]:
        if not args.analysis_database or daily_report_output is None:
            raise ValueError("home daily report is disabled")
        return generate_home_daily_report(
            database_path=args.analysis_database,
            output_root=daily_report_output,
            timezone_name=settings.timezone,
        ).to_public_dict()

    def provide_daily_report() -> bytes:
        if not args.analysis_database or daily_report_output is None:
            raise ValueError("home daily report is disabled")
        return read_home_daily_report_html(
            database_path=args.analysis_database,
            output_root=daily_report_output,
            timezone_name=settings.timezone,
        )

    def daily_digest_catalog() -> DailyDigestCatalog:
        if not args.analysis_database or daily_digest_output is None:
            raise ValueError("daily digest catalog is disabled")
        database_path = Path(args.analysis_database).expanduser()
        return DailyDigestCatalog(
            database_path=database_path,
            plan_root=database_path.parent / "daily-digests",
            export_root=daily_digest_output,
            token_secret=daily_digest_secret,
        )

    def provide_daily_digest_catalog(
        limit: int,
    ) -> tuple[dict[str, object], ...]:
        return daily_digest_catalog().list_public(limit=limit)

    def provide_daily_digest_video(review_token: str) -> bytes:
        return daily_digest_catalog().read_video(review_token)

    def generate_daily_digest() -> dict[str, object]:
        if not args.analysis_database or daily_digest_output is None:
            raise ValueError("daily digest generation is disabled")
        database_path = Path(args.analysis_database).expanduser()
        local_date = latest_daily_digest_local_date(
            database_path=database_path,
            timezone_name=settings.timezone,
        )
        admission = load_daily_digest_candidates(
            database_path=database_path,
            storage_root=settings.storage.local_root.expanduser(),
            local_date=local_date,
            timezone_name=settings.timezone,
        )
        plan = plan_daily_digest(
            admission.candidates,
            local_date=local_date,
            timezone_name=settings.timezone,
            config=DailyDigestConfig(),
        )
        write_daily_digest_plan(
            plan,
            database_path=database_path,
            output_root=database_path.parent / "daily-digests",
        )
        return export_daily_digest(
            plan,
            admission,
            database_path=database_path,
            output_root=daily_digest_output,
            storage_root=settings.storage.local_root.expanduser(),
        ).to_public_dict()

    application = HomeDashboardApplication(
        status_provider=provide_status,
        action_executor=lambda action, confirmed: control_home_stack(
            action,
            ha_outbox_database=args.ha_outbox_database,
            analysis_database=args.analysis_database,
            candidate_health_paths=args.candidate_health,
            candidate_health_stale_after_seconds=(args.candidate_health_stale_after_seconds),
            candidate_soak_ledger=args.candidate_soak_ledger,
            notification_enabled=settings.notification.enabled,
            external_notification_delivery_enabled=(
                settings.notification.external_delivery_enabled
            ),
            notification_max_pending_age_seconds=(settings.notification.max_pending_age_seconds),
            daily_report_output=daily_report_output,
            timezone_name=settings.timezone,
            confirm_buffer_eviction=confirmed,
        ),
        notification_test_executor=(
            run_notification_test
            if args.analysis_database and settings.notification.enabled
            else None
        ),
        daily_report_executor=(generate_daily_report if args.analysis_database else None),
        daily_report_provider=(provide_daily_report if args.analysis_database else None),
        daily_digest_catalog_provider=(
            provide_daily_digest_catalog if args.analysis_database else None
        ),
        daily_digest_video_provider=(
            provide_daily_digest_video if args.analysis_database else None
        ),
        daily_digest_executor=(generate_daily_digest if args.analysis_database else None),
        **review_callbacks,
    )

    def ready(port: int) -> None:
        _json_print(
            {
                "version": 1,
                "status": "ready",
                "url": f"http://127.0.0.1:{port}/",
                "bind": "loopback",
                "automatic_monitoring_start": False,
                "shadow_review_enabled": bool(args.analysis_database),
                "local_notification_test_enabled": bool(
                    args.analysis_database and settings.notification.enabled
                ),
                "external_notification_delivery_enabled": (
                    settings.notification.external_delivery_enabled
                ),
                "local_daily_report_enabled": bool(args.analysis_database),
                "local_daily_digest_enabled": bool(args.analysis_database),
                "remote_used": False,
                "paths_disclosed": False,
            }
        )
        sys.stdout.flush()

    summary = serve_home_dashboard(
        application,
        port=args.port,
        max_runtime_seconds=args.max_runtime_seconds,
        ready=ready,
    )
    _json_print(summary)
    return 0
