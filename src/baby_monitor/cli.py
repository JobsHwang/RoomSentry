"""Command registration for the baby-monitor CLI.

Handler implementations live in :mod:`baby_monitor.cli_commands`;
shared helpers live in :mod:`baby_monitor.cli_support`. This module keeps
the full argument surface and ``main`` entry point in one readable place.
"""

from __future__ import annotations

import argparse
import sys

from baby_monitor.cli_commands.admissions import (  # noqa: F401
    command_validate_source_admission,
    command_verify_source_asset,
)
from baby_monitor.cli_commands.analysis import (  # noqa: F401
    _execute_analysis_run_once,
    command_analysis_gpu_status,
    command_analysis_queue_status,
    command_analysis_retry,
    command_analysis_run_once,
    command_analysis_worker,
    command_runtime_status,
)
from baby_monitor.cli_commands.archive import (  # noqa: F401
    _archive_run_once,
    _enqueue_archive_manifest,
    command_archive_copy,
    command_archive_enqueue,
    command_archive_enqueue_event,
    command_archive_enqueue_report,
    command_archive_plan,
    command_archive_queue_status,
    command_archive_retry,
    command_archive_run_once,
    command_archive_worker,
    command_retention_scan,
)
from baby_monitor.cli_commands.candidate_runtime import (  # noqa: F401
    _candidate_detector,
    _candidate_scanner_identity,
    command_candidate_health_status,
    command_candidate_presence_health_init,
    command_candidate_soak_finish,
    command_candidate_soak_start,
    command_candidate_soak_status,
    command_candidate_worker,
    command_room_presence_evaluate,
    command_room_registry_status,
    command_runtime_trial_finish,
    command_runtime_trial_start,
    command_runtime_trial_status,
)
from baby_monitor.cli_commands.detection_eval import (  # noqa: F401
    command_combine_household_model_temporal_diagnostic_batches,
    command_compare_household_model_diagnostic_batches,
    command_compare_household_model_presence_diagnostics,
    command_compare_household_trigger_coverage,
    command_evaluate_candidate_start_gate_sweep,
    command_evaluate_detector_box_quality,
    command_evaluate_detector_threshold_sweep,
    command_evaluate_household_model_presence_diagnostic,
    command_evaluate_household_model_temporal_diagnostic,
    command_evaluate_household_sampled_coverage,
    command_evaluate_household_trigger_coverage,
    command_prepare_detector_tracking_batch,
    command_prepare_detector_tracking_evaluation,
    command_select_candidate_start_gate_operating_point,
    command_select_detector_operating_point,
    command_select_household_model_presence_working_point,
    command_select_household_model_temporal_batch_working_point,
    command_select_household_model_temporal_working_point,
    command_select_household_trigger_operating_point,
    command_validate_candidate_start_gate_sweep,
    command_validate_detector_box_quality,
    command_validate_detector_scan,
    command_validate_detector_threshold_sweep,
    command_validate_household_episode_annotation,
    command_validate_household_model_diagnostic_drift,
    command_validate_household_model_presence_diagnostic,
    command_validate_household_model_presence_review,
    command_validate_household_model_temporal_batch_diagnostic,
    command_validate_household_model_temporal_diagnostic,
    command_validate_household_trigger_coverage,
    command_verify_yolo_model,
    command_verify_yolo_pose_model,
    command_yolo_scan_file,
)
from baby_monitor.cli_commands.event_policy import (  # noqa: F401
    command_evaluate_event_policy,
    command_evaluate_event_policy_cascade,
    command_validate_event_policy,
    command_validate_event_policy_cascade,
)
from baby_monitor.cli_commands.home_assistant import (  # noqa: F401
    command_authorize_ha,
    command_capture_ha_token,
    command_discover_ha_signals,
    command_ha_signal_dispatch_once,
    command_ha_signal_dispatch_worker,
    command_ha_signal_outbox_ack,
    command_ha_signal_outbox_claim,
    command_ha_signal_outbox_fail,
    command_ha_signal_outbox_retry,
    command_ha_signal_outbox_status,
    command_listen_ha_signals,
    command_normalize_ha_signals,
    command_store_ha_token,
    command_watch_ha_signals,
)
from baby_monitor.cli_commands.home_dashboard import (  # noqa: F401
    command_daily_digest_export,
    command_daily_digest_export_status,
    command_daily_digest_plan,
    command_daily_digest_plan_status,
    command_home_control,
    command_home_daily_report,
    command_home_dashboard,
    command_systemd_render,
)
from baby_monitor.cli_commands.identity import (  # noqa: F401
    command_identity_consent_status,
    command_identity_delete_profile,
    command_identity_deletion_preflight,
    command_identity_enroll,
    command_identity_enrollment_preflight,
    command_identity_profiles,
)
from baby_monitor.cli_commands.local_review import (
    command_local_review_retry,
    command_local_review_status,
)
from baby_monitor.cli_commands.media_pipeline import (  # noqa: F401
    _local_adapter,
    _pipeline,
    _remote_adapter,
    _review_policy,
    _secondary_adapter,
    _yolo_detector,
    command_buffer_extract,
    command_enqueue_buffer_event,
    command_init_db,
    command_list_events,
    command_media_health_status,
    command_process_buffer_event,
    command_process_file,
    command_report,
    command_rtsp_buffer,
    command_rtsp_soak,
    command_smoke_demo,
)
from baby_monitor.cli_commands.notification import (  # noqa: F401
    _notification_sink_from_args,
    _require_notification_sink_enabled,
    _require_notifications_enabled,
    command_notification_enqueue_event,
    command_notification_local_test,
    command_notification_queue_status,
    command_notification_retry,
    command_notification_run_once,
    command_notification_transport_preview,
    command_notification_worker,
    command_validate_notification_json,
)
from baby_monitor.cli_commands.pose import (  # noqa: F401
    _pose_quality_public_summary,
    command_evaluate_pose_quality,
    command_pose_scan_admitted_file,
    command_pose_scan_file,
    command_validate_admitted_pose_scan,
    command_validate_pose_frames,
    command_validate_pose_quality,
)
from baby_monitor.cli_commands.pose_fusion import (  # noqa: F401
    command_attest_household_pose_fusion_temporal_holdout,
    command_combine_household_pose_fusion_diagnostic_batches,
    command_evaluate_household_human_pose_fusion_audit,
    command_evaluate_household_pose_fusion_diagnostic,
    command_import_household_human_presence_audit,
    command_validate_household_human_pose_fusion_audit,
    command_validate_household_human_presence_audit,
    command_validate_household_pose_fusion_batch_diagnostic,
    command_validate_household_pose_fusion_diagnostic,
    command_validate_household_pose_fusion_profile,
    command_validate_household_pose_fusion_temporal_holdout,
)
from baby_monitor.cli_commands.recent_observations import command_recent_observations
from baby_monitor.cli_commands.release import (  # noqa: F401
    command_release_audit,
    command_release_bundle,
    command_release_sbom,
)
from baby_monitor.cli_commands.remote_review import (  # noqa: F401
    _configured_remote_adapter,
    command_recognition_audit_status,
    command_remote_audit_status,
    command_remote_mock_server,
    command_remote_preview,
)
from baby_monitor.cli_commands.report_refresh import (  # noqa: F401
    command_report_refresh_request,
    command_report_refresh_retry,
    command_report_refresh_run_once,
    command_report_refresh_status,
    command_report_refresh_worker,
)
from baby_monitor.cli_commands.schema import (  # noqa: F401
    command_schema,
)
from baby_monitor.cli_commands.signal_events import (  # noqa: F401
    _one_signal,
    command_enqueue_signal_event,
    command_validate_decision_json,
    command_validate_signal_json,
)
from baby_monitor.cli_commands.tracking import (  # noqa: F401
    _motion_tracker_state_for_batch,
    _temporal_state_for_config,
    _tracker_disagreement_inputs,
    _tracker_state_for_batch,
    command_compare_tracker_predictions,
    command_evaluate_tracker_association,
    command_evaluate_tracker_disagreement,
    command_import_caviar_tracking_ground_truth,
    command_prepare_tracker_disagreement_review,
    command_prepare_tracker_review,
    command_temporal_zone_ack,
    command_temporal_zone_pending,
    command_temporal_zone_run,
    command_track_iou_ack,
    command_track_iou_pending,
    command_track_iou_predict,
    command_track_iou_run,
    command_track_iou_zone_run,
    command_track_motion_ack,
    command_track_motion_pending,
    command_track_motion_predict,
    command_track_motion_run,
    command_track_motion_zone_run,
    command_validate_detection_frames,
    command_validate_tracked_observations,
    command_validate_tracker_association,
    command_validate_tracker_comparison,
    command_validate_tracker_disagreement,
    command_validate_tracker_disagreement_review,
    command_validate_tracker_predictions,
    command_validate_tracker_review,
)
from baby_monitor.cli_commands.visual_predicate import (  # noqa: F401
    _shadow_worker_attempt,
    command_visual_predicate_health_status,
    command_visual_predicate_human_calibration_report,
    command_visual_predicate_review_report,
    command_visual_predicate_shadow_queue_status,
    command_visual_predicate_shadow_report,
    command_visual_predicate_shadow_retry,
    command_visual_predicate_shadow_run,
    command_visual_predicate_shadow_run_next,
    command_visual_predicate_shadow_worker,
)
from baby_monitor.cli_commands.vlm_eval import (  # noqa: F401
    command_ab_run,
    command_validate_ab_dataset,
    command_validate_json,
)
from baby_monitor.cli_support import _absolute_runner_executable

from .home_control import HOME_CONTROL_ACTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baby-monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="initialize the SQLite event database")
    init_db.add_argument("--config", default="config/default.toml")
    init_db.add_argument("--database")
    init_db.set_defaults(handler=command_init_db)

    identity_status = subparsers.add_parser(
        "identity-consent-status",
        help="show path-free identity feature and consent gates without reading media",
    )
    identity_status.add_argument("--config", default="config/default.toml")
    identity_status.set_defaults(handler=command_identity_consent_status)

    identity_enrollment = subparsers.add_parser(
        "identity-enrollment-preflight",
        help=(
            "authorize one enrollment attempt without opening media, models, "
            "event history, or identity storage"
        ),
    )
    identity_enrollment.add_argument("--config", default="config/default.toml")
    identity_enrollment.add_argument(
        "--confirm-biometric-enrollment",
        action="store_true",
        help=(
            "one-shot confirmation; persistent identity.enabled and "
            "identity.biometric_enrollment_consent must also be true"
        ),
    )
    identity_enrollment.set_defaults(handler=command_identity_enrollment_preflight)

    identity_deletion = subparsers.add_parser(
        "identity-deletion-preflight",
        help=(
            "authorize deletion of one opaque identity profile without "
            "opening or mutating identity storage"
        ),
    )
    identity_deletion.add_argument("--profile-id", required=True)
    identity_deletion.add_argument(
        "--confirm-identity-profile-deletion",
        action="store_true",
        help=(
            "independent one-shot deletion confirmation; remains available "
            "when identity and enrollment consent are disabled"
        ),
    )
    identity_deletion.set_defaults(handler=command_identity_deletion_preflight)

    identity_enroll = subparsers.add_parser(
        "identity-enroll",
        help=(
            "enroll one local identity profile from a private 3-12 image manifest; "
            "household execution requires systemd IPAddressDeny=any"
        ),
    )
    identity_enroll.add_argument("--config", default="config/default.toml")
    identity_enroll.add_argument("--manifest", required=True)
    identity_enroll.add_argument("--manifest-root", required=True)
    identity_enroll.add_argument("--store-root", required=True)
    identity_enroll.add_argument("--staging-parent", required=True)
    identity_enroll.add_argument(
        "--runner-executable",
        required=True,
        type=_absolute_runner_executable,
        help=(
            "absolute Python executable; argv is fixed to "
            "'-m baby_monitor.local_runners.opencv_face'"
        ),
    )
    identity_enroll.add_argument("--detector-model", required=True)
    identity_enroll.add_argument("--detector-manifest", required=True)
    identity_enroll.add_argument("--embedding-model", required=True)
    identity_enroll.add_argument("--embedding-manifest", required=True)
    identity_enroll.add_argument("--timeout-seconds", type=float, default=120.0)
    identity_enroll.add_argument(
        "--detector-score-threshold",
        type=float,
        default=0.8,
    )
    identity_enroll.add_argument(
        "--detector-nms-threshold",
        type=float,
        default=0.3,
    )
    identity_enroll.add_argument("--detector-top-k", type=int, default=100)
    identity_enroll.add_argument("--min-face-size-pixels", type=int, default=40)
    identity_enroll.add_argument(
        "--min-eye-distance-pixels",
        type=float,
        default=12.0,
    )
    identity_enroll.add_argument(
        "--confirm-biometric-enrollment",
        action="store_true",
        help=(
            "one-shot confirmation; persistent identity.enabled and "
            "identity.biometric_enrollment_consent must both be true"
        ),
    )
    identity_enroll.set_defaults(handler=command_identity_enroll)

    identity_profiles = subparsers.add_parser(
        "identity-profiles",
        help="list opaque local identity profiles without loading or exporting embeddings",
    )
    identity_profiles.add_argument("--store-root", required=True)
    identity_profiles.set_defaults(handler=command_identity_profiles)

    identity_delete = subparsers.add_parser(
        "identity-delete-profile",
        help="delete exactly one opaque local identity profile and its templates",
    )
    identity_delete.add_argument("--store-root", required=True)
    identity_delete.add_argument("--profile-id", required=True)
    identity_delete.add_argument(
        "--confirm-identity-profile-deletion",
        action="store_true",
        help=(
            "independent one-shot confirmation; deletion remains available "
            "after identity enrollment is disabled"
        ),
    )
    identity_delete.set_defaults(handler=command_identity_delete_profile)

    remote_preview = subparsers.add_parser(
        "remote-preview",
        help="show the path-free remote target and selected frame budget without uploading",
    )
    remote_preview.add_argument("--config", default="config/default.toml")
    remote_preview.add_argument(
        "--frame",
        required=True,
        action="append",
        help="local evidence image; repeat to preview deterministic selection",
    )
    remote_preview.set_defaults(handler=command_remote_preview)

    remote_audit = subparsers.add_parser(
        "remote-audit-status",
        help="show media-free remote review attempts from SQLite",
    )
    remote_audit.add_argument("--config", default="config/default.toml")
    remote_audit.add_argument("--database")
    remote_audit.add_argument("--limit", type=int, default=20)
    remote_audit.set_defaults(handler=command_remote_audit_status)

    recognition_audit = subparsers.add_parser(
        "recognition-audit-status",
        help="show path-free local, secondary, and remote recognition attempts",
    )
    recognition_audit.add_argument("--config", default="config/default.toml")
    recognition_audit.add_argument("--database")
    recognition_audit.add_argument("--limit", type=int, default=20)
    recognition_audit.set_defaults(handler=command_recognition_audit_status)

    recent_observations = subparsers.add_parser(
        "recent-observations",
        help="read local model descriptions per mapped camera; not live baby location or state",
    )
    recent_observations.add_argument("--database", required=True)
    recent_observations.add_argument("--registry", required=True)
    recent_observations.add_argument("--recent-seconds", type=int, default=120)
    recent_observations.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="text shows Chinese history/unknown labels; json preserves the structured output",
    )
    recent_observations.set_defaults(handler=command_recent_observations)

    local_retry = subparsers.add_parser(
        "local-review-retry",
        help=("retry one historically bound event using only the configured local secondary model"),
    )
    local_retry.add_argument("--config", default="config/default.toml")
    local_retry.add_argument("--database")
    local_retry.add_argument("--event-id", required=True)
    local_retry.add_argument("--request-id", required=True)
    local_retry.set_defaults(handler=command_local_review_retry)

    local_status = subparsers.add_parser(
        "local-review-status",
        help="read path-free local review retry attempts without running models",
    )
    local_status.add_argument("--config", default="config/default.toml")
    local_status.add_argument("--database")
    local_status.add_argument("--limit", type=int, default=20)
    local_status.set_defaults(handler=command_local_review_status)

    remote_mock = subparsers.add_parser(
        "remote-mock-server",
        help="serve a bounded OpenAI-compatible fixture on an IP loopback address only",
    )
    remote_mock.add_argument("--host", default="127.0.0.1")
    remote_mock.add_argument("--port", type=int, default=18081)
    remote_mock.add_argument("--max-requests", type=int, default=1)
    remote_mock.set_defaults(handler=command_remote_mock_server)

    notification_validate = subparsers.add_parser(
        "validate-notification-json",
        help="validate one strict path-free NotificationMessage v1 JSON file",
    )
    notification_validate.add_argument("path")
    notification_validate.set_defaults(handler=command_validate_notification_json)

    notification_enqueue = subparsers.add_parser(
        "notification-enqueue-event",
        help="enqueue one stored final or pending-review event for a local sink",
    )
    notification_enqueue.add_argument("--config", default="config/default.toml")
    notification_enqueue.add_argument("--database")
    notification_enqueue.add_argument("--event-id", required=True)
    notification_enqueue.add_argument(
        "--sink", choices=["stdout", "file", "smtp", "webhook"], required=True
    )
    notification_enqueue.add_argument("--sink-id", required=True)
    notification_enqueue.set_defaults(handler=command_notification_enqueue_event)

    notification_local_test = subparsers.add_parser(
        "notification-local-test",
        help="write one explicit test-only notification through the local file outbox",
    )
    notification_local_test.add_argument("--config", default="config/default.toml")
    notification_local_test.add_argument("--database")
    notification_local_test.add_argument("--output-dir", required=True)
    notification_local_test.add_argument("--confirmation", required=True)
    notification_local_test.set_defaults(handler=command_notification_local_test)

    notification_run = subparsers.add_parser(
        "notification-run-once",
        help="deliver at most one queued notification to a configured sink",
    )
    notification_run.add_argument("--config", default="config/default.toml")
    notification_run.add_argument("--database")
    notification_run.add_argument(
        "--sink", choices=["stdout", "file", "smtp", "webhook"], required=True
    )
    notification_run.add_argument("--sink-id", required=True)
    notification_run.add_argument("--output-dir")
    notification_run.add_argument("--profile")
    notification_run.set_defaults(handler=command_notification_run_once)

    notification_worker = subparsers.add_parser(
        "notification-worker",
        help="run a bounded notification delivery loop for one configured sink",
    )
    notification_worker.add_argument("--config", default="config/default.toml")
    notification_worker.add_argument("--database")
    notification_worker.add_argument(
        "--sink", choices=["stdout", "file", "smtp", "webhook"], required=True
    )
    notification_worker.add_argument("--sink-id", required=True)
    notification_worker.add_argument("--output-dir")
    notification_worker.add_argument("--profile")
    notification_worker.add_argument("--max-jobs", type=int, default=0)
    notification_worker.add_argument("--idle-exit-seconds", type=float, default=30.0)
    notification_worker.add_argument("--max-runtime-seconds", type=float, default=3600.0)
    notification_worker.add_argument("--poll-seconds", type=float, default=2.0)
    notification_worker.set_defaults(handler=command_notification_worker)

    notification_preview = subparsers.add_parser(
        "notification-transport-preview",
        help="inspect a private SMTP/webhook profile without using the network or exposing secrets",
    )
    notification_preview.add_argument("--profile", required=True)
    notification_preview.set_defaults(handler=command_notification_transport_preview)

    notification_status = subparsers.add_parser(
        "notification-queue-status",
        help="show the path-free local notification outbox without delivering",
    )
    notification_status.add_argument("--config", default="config/default.toml")
    notification_status.add_argument("--database")
    notification_status.set_defaults(handler=command_notification_queue_status)

    notification_retry = subparsers.add_parser(
        "notification-retry",
        help="explicitly reset one failed local notification job",
    )
    notification_retry.add_argument("--config", default="config/default.toml")
    notification_retry.add_argument("--database")
    notification_retry.add_argument("--notification-id", required=True)
    notification_retry.set_defaults(handler=command_notification_retry)

    validate = subparsers.add_parser("validate-json", help="validate one strict VLM JSON file")
    validate.add_argument("path")
    validate.set_defaults(handler=command_validate_json)

    validate_signal = subparsers.add_parser(
        "validate-signal-json",
        help="validate one strict vendor-neutral SignalEvent v1 JSON object",
    )
    validate_signal.add_argument("path")
    validate_signal.set_defaults(handler=command_validate_signal_json)

    normalize_ha = subparsers.add_parser(
        "normalize-ha-signals",
        help="normalize an offline HA WebSocket event capture without connecting to HA",
    )
    normalize_ha.add_argument("--mapping", required=True)
    normalize_ha.add_argument("--events", required=True)
    normalize_ha.add_argument("--max-events", type=int, default=1_000)
    normalize_ha.set_defaults(handler=command_normalize_ha_signals)

    discover_ha = subparsers.add_parser(
        "discover-ha-signals",
        help=(
            "read a bounded HA entity inventory over a LAN-only WebSocket and "
            "write private event/binary-sensor candidates"
        ),
    )
    discover_ha.add_argument("--url", required=True)
    discover_ha.add_argument(
        "--credentials-file",
        "--token-file",
        dest="token_file",
        required=True,
    )
    discover_ha.add_argument("--output", required=True)
    discover_ha.add_argument("--timeout-seconds", type=float, default=10.0)
    discover_ha.add_argument("--confirm-private-output", action="store_true")
    discover_ha.set_defaults(handler=command_discover_ha_signals)

    authorize_ha = subparsers.add_parser(
        "authorize-ha",
        help=(
            "authorize this LAN-only client in a browser and store refreshable "
            "HA credentials in a new 0600 file without disclosing tokens"
        ),
    )
    authorize_ha.add_argument("--url", required=True)
    authorize_ha.add_argument("--credentials-output", required=True)
    authorize_ha.add_argument("--callback-port", type=int, default=0)
    authorize_ha.add_argument("--authorization-request-output")
    authorize_ha.add_argument("--timeout-seconds", type=float, default=180.0)
    authorize_ha.set_defaults(handler=command_authorize_ha)

    store_ha_token = subparsers.add_parser(
        "store-ha-token",
        help=(
            "store one HA access value from a hidden prompt in a new 0600 file "
            "without command-line or stdout disclosure"
        ),
    )
    store_ha_token.add_argument("--output", required=True)
    store_ha_token.set_defaults(handler=command_store_ha_token)

    capture_ha_token = subparsers.add_parser(
        "capture-ha-token",
        help=(
            "receive one HA access value through a loopback-only browser form "
            "and store it in a new 0600 file without disclosing it"
        ),
    )
    capture_ha_token.add_argument("--output", required=True)
    capture_ha_token.add_argument("--callback-port", type=int, default=0)
    capture_ha_token.add_argument("--capture-request-output")
    capture_ha_token.add_argument("--timeout-seconds", type=float, default=600.0)
    capture_ha_token.set_defaults(handler=command_capture_ha_token)

    listen_ha = subparsers.add_parser(
        "listen-ha-signals",
        help=(
            "subscribe to bounded LAN-only HA state_changed events and emit "
            "only explicitly mapped vendor-neutral signals"
        ),
    )
    listen_ha.add_argument("--url", required=True)
    listen_ha.add_argument(
        "--credentials-file",
        "--token-file",
        dest="token_file",
        required=True,
    )
    listen_ha.add_argument("--mapping", required=True)
    listen_ha.add_argument("--duration-seconds", type=float, default=60.0)
    listen_ha.add_argument("--max-messages", type=int, default=1_000)
    listen_ha.add_argument("--max-signals", type=int, default=100)
    listen_ha.add_argument("--timeout-seconds", type=float, default=10.0)
    listen_ha.set_defaults(handler=command_listen_ha_signals)

    watch_ha = subparsers.add_parser(
        "watch-ha-signals",
        help=(
            "run a bounded LAN-only HA state_changed watcher with sanitized "
            "health counters and exponential reconnect"
        ),
    )
    watch_ha.add_argument("--url", required=True)
    watch_ha.add_argument(
        "--credentials-file",
        "--token-file",
        dest="token_file",
        required=True,
    )
    watch_ha.add_argument("--mapping", required=True)
    watch_ha.add_argument("--max-runtime-seconds", type=float, default=3600.0)
    watch_ha.add_argument("--poll-seconds", type=float, default=15.0)
    watch_ha.add_argument(
        "--reconnect-initial-seconds",
        type=float,
        default=1.0,
    )
    watch_ha.add_argument("--reconnect-max-seconds", type=float, default=30.0)
    watch_ha.add_argument("--max-connection-attempts", type=int, default=100)
    watch_ha.add_argument("--max-messages", type=int, default=100_000)
    watch_ha.add_argument("--max-signals", type=int, default=1_000)
    watch_ha.add_argument("--timeout-seconds", type=float, default=10.0)
    watch_ha.add_argument("--stream-json-lines", action="store_true")
    watch_ha.add_argument(
        "--outbox-database",
        help=(
            "persist every normalized signal to this local SQLite outbox before "
            "streaming or acknowledging receipt"
        ),
    )
    watch_ha.add_argument("--outbox-max-active", type=int, default=10_000)
    watch_ha.add_argument("--outbox-max-completed", type=int, default=100_000)
    watch_ha.add_argument(
        "--outbox-camera-id",
        action="append",
        default=[],
        help=(
            "persist only signals whose anonymous camera_id exactly matches this value; "
            "repeat for multiple locally routable sources"
        ),
    )
    watch_ha.set_defaults(handler=command_watch_ha_signals)

    ha_outbox_status = subparsers.add_parser(
        "ha-signal-outbox-status",
        help="read HA signal outbox health without disclosing payloads or paths",
    )
    ha_outbox_status.add_argument("--database", required=True)
    ha_outbox_status.set_defaults(handler=command_ha_signal_outbox_status)

    ha_outbox_claim = subparsers.add_parser(
        "ha-signal-outbox-claim",
        help="lease at most one persisted normalized HA signal",
    )
    ha_outbox_claim.add_argument("--database", required=True)
    ha_outbox_claim.add_argument("--lease-seconds", type=int, default=60)
    ha_outbox_claim.add_argument("--max-attempts", type=int, default=5)
    ha_outbox_claim.set_defaults(handler=command_ha_signal_outbox_claim)

    ha_outbox_ack = subparsers.add_parser(
        "ha-signal-outbox-ack",
        help="acknowledge one leased HA signal and scrub its persisted payload",
    )
    ha_outbox_ack.add_argument("--database", required=True)
    ha_outbox_ack.add_argument("--outbox-id", required=True)
    ha_outbox_ack.add_argument("--lease-token", required=True)
    ha_outbox_ack.set_defaults(handler=command_ha_signal_outbox_ack)

    ha_outbox_fail = subparsers.add_parser(
        "ha-signal-outbox-fail",
        help="record a bounded downstream failure for one leased HA signal",
    )
    ha_outbox_fail.add_argument("--database", required=True)
    ha_outbox_fail.add_argument("--outbox-id", required=True)
    ha_outbox_fail.add_argument("--lease-token", required=True)
    ha_outbox_fail.add_argument("--error-code", required=True)
    ha_outbox_fail.add_argument("--retry-delay-seconds", type=int, default=5)
    ha_outbox_fail.add_argument("--max-attempts", type=int, default=5)
    ha_outbox_fail.set_defaults(handler=command_ha_signal_outbox_fail)

    ha_outbox_retry = subparsers.add_parser(
        "ha-signal-outbox-retry",
        help="explicitly return one failed HA signal to the bounded pending queue",
    )
    ha_outbox_retry.add_argument("--database", required=True)
    ha_outbox_retry.add_argument("--outbox-id", required=True)
    ha_outbox_retry.add_argument("--max-active", type=int, default=10_000)
    ha_outbox_retry.set_defaults(handler=command_ha_signal_outbox_retry)

    ha_dispatch = subparsers.add_parser(
        "ha-signal-dispatch-once",
        help=(
            "lease at most one HA signal and enqueue bounded local buffer evidence "
            "without loading a model"
        ),
    )
    ha_dispatch.add_argument("--outbox-database", required=True)
    ha_dispatch.add_argument("--route-map", required=True)
    ha_dispatch.add_argument("--config", default="config/default.toml")
    ha_dispatch.add_argument("--staging-root", required=True)
    ha_dispatch.add_argument("--analysis-database")
    ha_dispatch.add_argument("--output-root")
    ha_dispatch.add_argument("--lease-seconds", type=int, default=240)
    ha_dispatch.add_argument("--max-attempts", type=int, default=5)
    ha_dispatch.add_argument("--retry-delay-seconds", type=int, default=30)
    ha_dispatch.add_argument("--command-timeout-seconds", type=float, default=180.0)
    ha_dispatch.set_defaults(handler=command_ha_signal_dispatch_once)

    ha_dispatch_worker = subparsers.add_parser(
        "ha-signal-dispatch-worker",
        help=(
            "consume the local HA signal outbox sequentially and enqueue bounded "
            "buffer evidence without loading a model"
        ),
    )
    ha_dispatch_worker.add_argument("--outbox-database", required=True)
    ha_dispatch_worker.add_argument("--route-map", required=True)
    ha_dispatch_worker.add_argument("--config", default="config/default.toml")
    ha_dispatch_worker.add_argument("--staging-root", required=True)
    ha_dispatch_worker.add_argument("--analysis-database")
    ha_dispatch_worker.add_argument("--output-root")
    ha_dispatch_worker.add_argument("--lease-seconds", type=int, default=240)
    ha_dispatch_worker.add_argument("--max-attempts", type=int, default=5)
    ha_dispatch_worker.add_argument("--retry-delay-seconds", type=int, default=30)
    ha_dispatch_worker.add_argument("--command-timeout-seconds", type=float, default=180.0)
    ha_dispatch_worker.add_argument("--max-jobs", type=int)
    ha_dispatch_worker.add_argument("--max-runtime-seconds", type=float, default=86_400.0)
    ha_dispatch_worker.add_argument("--poll-seconds", type=float, default=2.0)
    ha_dispatch_worker.set_defaults(handler=command_ha_signal_dispatch_worker)

    validate_tracked = subparsers.add_parser(
        "validate-tracked-observations",
        help="validate strict path-free TrackedObjectObservation v1 JSON",
    )
    validate_tracked.add_argument("path")
    validate_tracked.set_defaults(handler=command_validate_tracked_observations)

    validate_detection_frames = subparsers.add_parser(
        "validate-detection-frames",
        help="validate a strict path-free DetectionFrameBatch v1 JSON file",
    )
    validate_detection_frames.add_argument("path")
    validate_detection_frames.set_defaults(handler=command_validate_detection_frames)

    validate_pose_frames = subparsers.add_parser(
        "validate-pose-frames",
        help="validate a strict path-free PoseFrameBatch v1 JSON file",
    )
    validate_pose_frames.add_argument("path")
    validate_pose_frames.set_defaults(handler=command_validate_pose_frames)

    validate_admitted_pose_scan = subparsers.add_parser(
        "validate-admitted-pose-scan",
        help="validate a strict path-free AdmittedPoseScan v1 JSON file",
    )
    validate_admitted_pose_scan.add_argument("path")
    validate_admitted_pose_scan.set_defaults(handler=command_validate_admitted_pose_scan)

    validate_pose_quality = subparsers.add_parser(
        "validate-pose-quality",
        help="validate a strict path-free PoseQualityReport v1 JSON file",
    )
    validate_pose_quality.add_argument("path")
    validate_pose_quality.set_defaults(handler=command_validate_pose_quality)

    validate_decision = subparsers.add_parser(
        "validate-decision-json",
        help="validate one strict DecisionRecord v1 JSON object",
    )
    validate_decision.add_argument("path")
    validate_decision.set_defaults(handler=command_validate_decision_json)

    smoke = subparsers.add_parser("smoke-demo", help="run a no-video/no-AI plumbing demo")
    smoke.add_argument("--output-root", default="./data/smoke")
    smoke.set_defaults(handler=command_smoke_demo)

    verify_yolo = subparsers.add_parser(
        "verify-yolo-model",
        help="verify a reviewed local YOLO weight without importing or running the model",
    )
    verify_yolo.add_argument("--manifest", required=True)
    verify_yolo.add_argument("--weights", required=True)
    verify_yolo.set_defaults(handler=command_verify_yolo_model)

    yolo_scan = subparsers.add_parser(
        "yolo-scan-file",
        help=("scan one admitted bounded local video with a reviewed offline YOLO detector"),
    )
    yolo_scan.add_argument("--source", required=True)
    yolo_scan.add_argument("--source-admission", required=True)
    yolo_scan.add_argument("--camera-id", required=True)
    yolo_scan.add_argument("--manifest", required=True)
    yolo_scan.add_argument("--weights", required=True)
    yolo_scan.add_argument("--runner-executable", required=True)
    yolo_scan.add_argument("--label", action="append")
    yolo_scan.add_argument("--device", default="0")
    yolo_scan.add_argument("--confidence", type=float, default=0.35)
    yolo_scan.add_argument("--sample-fps", type=float, default=1.0)
    yolo_scan.add_argument("--image-size", type=int, default=640)
    yolo_scan.add_argument("--rotation-degrees", type=int, choices=(0, 90, 180, 270), default=0)
    yolo_scan.add_argument("--timeout-seconds", type=float, default=600.0)
    yolo_scan.add_argument("--output", required=True)
    yolo_scan.set_defaults(handler=command_yolo_scan_file)

    validate_detector_scan = subparsers.add_parser(
        "validate-detector-scan",
        help="validate a strict path-free AdmittedDetectorScan v1 JSON file",
    )
    validate_detector_scan.add_argument("path")
    validate_detector_scan.set_defaults(handler=command_validate_detector_scan)

    detector_quality = subparsers.add_parser(
        "evaluate-detector-box-quality",
        help=(
            "evaluate one admitted person detector scan against aligned "
            "publisher-human boxes, including empty sampled frames"
        ),
    )
    detector_quality.add_argument("--ground-truth-frames", required=True)
    detector_quality.add_argument("--ground-truth-review", required=True)
    detector_quality.add_argument("--ground-truth-admission", required=True)
    detector_quality.add_argument("--media-admission", required=True)
    detector_quality.add_argument("--media-asset", required=True)
    detector_quality.add_argument("--detector-scan", required=True)
    detector_quality.add_argument("--source-fps", required=True, type=float)
    detector_quality.add_argument(
        "--source-frame-count",
        required=True,
        type=int,
    )
    detector_quality.add_argument(
        "--annotation-frame-offset",
        type=int,
        default=0,
    )
    detector_quality.add_argument("--iou-threshold", type=float, default=0.5)
    detector_quality.add_argument(
        "--confirm-frame-alignment",
        action="store_true",
        help=(
            "confirm that local decode metadata and visual spot checks establish "
            "the declared source-to-annotation frame alignment"
        ),
    )
    detector_quality.add_argument("--output", required=True)
    detector_quality.set_defaults(handler=command_evaluate_detector_box_quality)

    validate_detector_quality = subparsers.add_parser(
        "validate-detector-box-quality",
        help="validate a strict path-free DetectorBoxQualityReport v1 JSON file",
    )
    validate_detector_quality.add_argument("path")
    validate_detector_quality.set_defaults(handler=command_validate_detector_box_quality)

    detector_threshold_sweep = subparsers.add_parser(
        "evaluate-detector-threshold-sweep",
        help=(
            "derive a strict confidence-threshold quality curve from one admitted "
            "low-threshold person scan without rerunning the model"
        ),
    )
    detector_threshold_sweep.add_argument("--ground-truth-frames", required=True)
    detector_threshold_sweep.add_argument("--ground-truth-review", required=True)
    detector_threshold_sweep.add_argument(
        "--ground-truth-admission",
        required=True,
    )
    detector_threshold_sweep.add_argument("--media-admission", required=True)
    detector_threshold_sweep.add_argument("--media-asset", required=True)
    detector_threshold_sweep.add_argument("--detector-scan", required=True)
    detector_threshold_sweep.add_argument("--source-fps", required=True, type=float)
    detector_threshold_sweep.add_argument(
        "--source-frame-count",
        required=True,
        type=int,
    )
    detector_threshold_sweep.add_argument(
        "--annotation-frame-offset",
        type=int,
        default=0,
    )
    detector_threshold_sweep.add_argument(
        "--threshold",
        required=True,
        action="append",
        type=float,
    )
    detector_threshold_sweep.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
    )
    detector_threshold_sweep.add_argument(
        "--confirm-frame-alignment",
        action="store_true",
    )
    detector_threshold_sweep.add_argument("--output", required=True)
    detector_threshold_sweep.set_defaults(handler=command_evaluate_detector_threshold_sweep)

    detector_operating_point = subparsers.add_parser(
        "select-detector-operating-point",
        help=(
            "select a dataset-specific point from a strict threshold sweep "
            "under explicit precision, recall, and longest-miss constraints"
        ),
    )
    detector_operating_point.add_argument("path")
    detector_operating_point.add_argument(
        "--objective",
        required=True,
        choices=("recall_first", "balanced_f1"),
    )
    detector_operating_point.add_argument(
        "--minimum-precision",
        required=True,
        type=float,
    )
    detector_operating_point.add_argument(
        "--minimum-recall",
        required=True,
        type=float,
    )
    detector_operating_point.add_argument(
        "--maximum-longest-miss-seconds",
        required=True,
        type=float,
    )
    detector_operating_point.set_defaults(handler=command_select_detector_operating_point)

    validate_detector_threshold_sweep = subparsers.add_parser(
        "validate-detector-threshold-sweep",
        help="validate a strict path-free DetectorThresholdSweepReport v1 JSON file",
    )
    validate_detector_threshold_sweep.add_argument("path")
    validate_detector_threshold_sweep.set_defaults(
        handler=command_validate_detector_threshold_sweep
    )

    candidate_start_gate_sweep = subparsers.add_parser(
        "evaluate-candidate-start-gate-sweep",
        help=(
            "measure candidate trigger coverage across start-confidence gates "
            "from one admitted low-threshold person scan"
        ),
    )
    candidate_start_gate_sweep.add_argument(
        "--ground-truth-frames",
        required=True,
    )
    candidate_start_gate_sweep.add_argument(
        "--ground-truth-review",
        required=True,
    )
    candidate_start_gate_sweep.add_argument(
        "--ground-truth-admission",
        required=True,
    )
    candidate_start_gate_sweep.add_argument("--media-admission", required=True)
    candidate_start_gate_sweep.add_argument("--media-asset", required=True)
    candidate_start_gate_sweep.add_argument("--detector-scan", required=True)
    candidate_start_gate_sweep.add_argument(
        "--source-fps",
        required=True,
        type=float,
    )
    candidate_start_gate_sweep.add_argument(
        "--source-frame-count",
        required=True,
        type=int,
    )
    candidate_start_gate_sweep.add_argument(
        "--annotation-frame-offset",
        type=int,
        default=0,
    )
    candidate_start_gate_sweep.add_argument(
        "--start-threshold",
        required=True,
        action="append",
        type=float,
    )
    candidate_start_gate_sweep.add_argument(
        "--candidate-gap-seconds",
        required=True,
        type=float,
    )
    candidate_start_gate_sweep.add_argument(
        "--max-event-span-seconds",
        required=True,
        type=float,
    )
    candidate_start_gate_sweep.add_argument(
        "--validation-iou-threshold",
        type=float,
        default=0.5,
    )
    candidate_start_gate_sweep.add_argument(
        "--confirm-frame-alignment",
        action="store_true",
    )
    candidate_start_gate_sweep.add_argument("--output", required=True)
    candidate_start_gate_sweep.set_defaults(handler=command_evaluate_candidate_start_gate_sweep)

    validate_candidate_start_gate_sweep = subparsers.add_parser(
        "validate-candidate-start-gate-sweep",
        help=("validate a strict path-free CandidateStartGateSweepReport v1 JSON file"),
    )
    validate_candidate_start_gate_sweep.add_argument("path")
    validate_candidate_start_gate_sweep.set_defaults(
        handler=command_validate_candidate_start_gate_sweep
    )

    select_candidate_start_gate = subparsers.add_parser(
        "select-candidate-start-gate-operating-point",
        help=(
            "select one dataset-specific candidate start threshold under "
            "explicit event-coverage and negative-activation constraints"
        ),
    )
    select_candidate_start_gate.add_argument("path")
    select_candidate_start_gate.add_argument(
        "--minimum-ground-truth-episode-recall",
        required=True,
        type=float,
    )
    select_candidate_start_gate.add_argument(
        "--minimum-positive-sample-recall",
        required=True,
        type=float,
    )
    select_candidate_start_gate.add_argument(
        "--maximum-detection-delay-seconds",
        required=True,
        type=float,
    )
    select_candidate_start_gate.add_argument(
        "--maximum-longest-uncovered-positive-seconds",
        required=True,
        type=float,
    )
    select_candidate_start_gate.add_argument(
        "--maximum-negative-false-activation-rate",
        type=float,
    )
    select_candidate_start_gate.add_argument(
        "--require-negative-samples",
        action="store_true",
    )
    select_candidate_start_gate.set_defaults(
        handler=command_select_candidate_start_gate_operating_point
    )

    validate_household_annotation = subparsers.add_parser(
        "validate-household-episode-annotation",
        help=(
            "validate path-free household annotation v1 nominal-grid or v2 elapsed-source-PTS "
            "JSON without upgrading pending labels to human-verified ground truth"
        ),
    )
    validate_household_annotation.add_argument("path")
    validate_household_annotation.set_defaults(
        handler=command_validate_household_episode_annotation
    )

    household_trigger_coverage = subparsers.add_parser(
        "evaluate-household-trigger-coverage",
        help=(
            "measure nominal-grid person-trigger coverage from approved household annotation v1; "
            "elapsed-source-PTS annotation v2 requires evaluate-household-sampled-coverage"
        ),
    )
    household_trigger_coverage.add_argument("--annotation", required=True)
    household_trigger_coverage.add_argument("--media-admission", required=True)
    household_trigger_coverage.add_argument("--media-asset")
    household_trigger_coverage.add_argument("--detector-scan", required=True)
    household_trigger_coverage.add_argument(
        "--start-threshold",
        required=True,
        action="append",
        type=float,
    )
    household_trigger_coverage.add_argument(
        "--candidate-gap-seconds",
        required=True,
        type=float,
    )
    household_trigger_coverage.add_argument(
        "--max-event-span-seconds",
        required=True,
        type=float,
    )
    household_trigger_coverage.add_argument("--output", required=True)
    household_trigger_coverage.set_defaults(handler=command_evaluate_household_trigger_coverage)

    household_sampled_coverage = subparsers.add_parser(
        "evaluate-household-sampled-coverage",
        help=(
            "measure actual-PTS person detection coverage using approved household annotation "
            "v1 or elapsed-source-PTS v2 and an explicit hashed timeline binding; "
            "no candidate grouping or danger recognition"
        ),
    )
    for flag in ("--annotation", "--detector-scan", "--media-admission", "--media-asset"):
        household_sampled_coverage.add_argument(flag, required=True)
    household_sampled_coverage.add_argument(
        "--timeline-binding",
        required=True,
        help=(
            "strict JSON sidecar binding annotation, media, reference hashes "
            "and annotation-zero PTS"
        ),
    )
    household_sampled_coverage.add_argument(
        "--timeline-reference",
        required=True,
        help="opaque regular local alignment reference, at most 8 MiB; only its bytes are hashed",
    )
    household_sampled_coverage.add_argument("--confidence-threshold", required=True, type=float)
    household_sampled_coverage.add_argument("--output", required=True)
    household_sampled_coverage.set_defaults(handler=command_evaluate_household_sampled_coverage)

    validate_household_coverage = subparsers.add_parser(
        "validate-household-trigger-coverage",
        help=("validate a strict path-free HouseholdTriggerCoverageReport v1 JSON file"),
    )
    validate_household_coverage.add_argument("path")
    validate_household_coverage.set_defaults(handler=command_validate_household_trigger_coverage)

    validate_event_policy = subparsers.add_parser(
        "validate-event-policy",
        help=(
            "validate a strict path-free HouseholdEventPolicy v1 household "
            "event-package routing policy"
        ),
    )
    validate_event_policy.add_argument("path")
    validate_event_policy.set_defaults(handler=command_validate_event_policy)

    evaluate_event_policy_parser = subparsers.add_parser(
        "evaluate-event-policy",
        help=(
            "replay completed events through a household event policy and "
            "report route/suppress decisions without delivering anything"
        ),
    )
    evaluate_event_policy_parser.add_argument("--policy", required=True)
    evaluate_event_policy_parser.add_argument("--events", required=True)
    evaluate_event_policy_parser.add_argument("--output")
    evaluate_event_policy_parser.set_defaults(handler=command_evaluate_event_policy)

    evaluate_event_policy_cascade_parser = subparsers.add_parser(
        "evaluate-event-policy-cascade",
        help=(
            "replay DecisionRecord-bound completed events through a household "
            "event policy without reading media or delivering anything"
        ),
    )
    evaluate_event_policy_cascade_parser.add_argument("--policy", required=True)
    evaluate_event_policy_cascade_parser.add_argument("--cases", required=True)
    evaluate_event_policy_cascade_parser.add_argument("--output")
    evaluate_event_policy_cascade_parser.set_defaults(handler=command_evaluate_event_policy_cascade)

    validate_event_policy_cascade = subparsers.add_parser(
        "validate-event-policy-cascade",
        help="validate a strict path-free EventPolicyCascadeReport v1 JSON file",
    )
    validate_event_policy_cascade.add_argument("path")
    validate_event_policy_cascade.set_defaults(handler=command_validate_event_policy_cascade)

    compare_household_trigger = subparsers.add_parser(
        "compare-household-trigger-coverage",
        help=(
            "compare two or more household trigger coverage reports that "
            "share the same annotation, media, sample grid, and thresholds"
        ),
    )
    compare_household_trigger.add_argument("paths", nargs="+")
    compare_household_trigger.set_defaults(handler=command_compare_household_trigger_coverage)

    select_household_trigger = subparsers.add_parser(
        "select-household-trigger-operating-point",
        help=(
            "select one dataset-specific household trigger threshold under "
            "mandatory danger-episode recall and explicit coverage constraints"
        ),
    )
    select_household_trigger.add_argument("path")
    select_household_trigger.add_argument(
        "--minimum-danger-episode-recall",
        required=True,
        type=float,
    )
    select_household_trigger.add_argument(
        "--minimum-person-episode-recall",
        required=True,
        type=float,
    )
    select_household_trigger.add_argument(
        "--minimum-positive-sample-recall",
        required=True,
        type=float,
    )
    select_household_trigger.add_argument(
        "--maximum-detection-delay-seconds",
        required=True,
        type=float,
    )
    select_household_trigger.add_argument(
        "--maximum-longest-uncovered-positive-seconds",
        required=True,
        type=float,
    )
    select_household_trigger.add_argument(
        "--maximum-negative-false-activation-rate",
        type=float,
    )
    select_household_trigger.add_argument(
        "--require-negative-samples",
        action="store_true",
    )
    select_household_trigger.set_defaults(handler=command_select_household_trigger_operating_point)

    validate_household_model_review = subparsers.add_parser(
        "validate-household-model-presence-review",
        help=(
            "validate a strict path-free model-generated household visible-person presence review"
        ),
    )
    validate_household_model_review.add_argument("path")
    validate_household_model_review.set_defaults(
        handler=command_validate_household_model_presence_review
    )

    household_model_diagnostic = subparsers.add_parser(
        "evaluate-household-model-presence-diagnostic",
        help=(
            "diagnose person-detector confidence thresholds against a bound "
            "model-generated household presence review without rerunning a model"
        ),
    )
    household_model_diagnostic.add_argument("--review", required=True)
    household_model_diagnostic.add_argument(
        "--media-admission",
        required=True,
        action="append",
    )
    household_model_diagnostic.add_argument(
        "--detector-scan",
        required=True,
        action="append",
    )
    household_model_diagnostic.add_argument(
        "--threshold",
        required=True,
        action="append",
        type=float,
    )
    household_model_diagnostic.add_argument(
        "--bin-seconds",
        required=True,
        type=float,
    )
    household_model_diagnostic.add_argument("--output", required=True)
    household_model_diagnostic.set_defaults(
        handler=command_evaluate_household_model_presence_diagnostic
    )

    validate_household_model_diagnostic = subparsers.add_parser(
        "validate-household-model-presence-diagnostic",
        help=("validate a strict path-free HouseholdModelDiagnosticReport v1 JSON file"),
    )
    validate_household_model_diagnostic.add_argument("path")
    validate_household_model_diagnostic.set_defaults(
        handler=command_validate_household_model_presence_diagnostic
    )

    compare_household_model_diagnostic = subparsers.add_parser(
        "compare-household-model-presence-diagnostics",
        help=(
            "compare two or more diagnostic reports that share one model-generated "
            "review, source set, bin grid, and threshold grid without selecting a winner"
        ),
    )
    compare_household_model_diagnostic.add_argument("paths", nargs="+")
    compare_household_model_diagnostic.set_defaults(
        handler=command_compare_household_model_presence_diagnostics
    )

    compare_household_model_batches = subparsers.add_parser(
        "compare-household-model-diagnostic-batches",
        help=(
            "compare one detector across two disjoint model-reviewed household batches "
            "under explicit diagnostic-only drift tolerances"
        ),
    )
    compare_household_model_batches.add_argument("--baseline", required=True)
    compare_household_model_batches.add_argument("--current", required=True)
    compare_household_model_batches.add_argument(
        "--maximum-precision-drop", required=True, type=float
    )
    compare_household_model_batches.add_argument("--maximum-recall-drop", required=True, type=float)
    compare_household_model_batches.add_argument(
        "--maximum-specificity-drop", required=True, type=float
    )
    compare_household_model_batches.add_argument(
        "--maximum-partial-visibility-recall-drop", required=True, type=float
    )
    compare_household_model_batches.add_argument(
        "--allow-no-partial-visibility-samples", action="store_true"
    )
    compare_household_model_batches.add_argument("--output", required=True)
    compare_household_model_batches.set_defaults(
        handler=command_compare_household_model_diagnostic_batches
    )

    validate_household_model_drift = subparsers.add_parser(
        "validate-household-model-diagnostic-drift",
        help="validate a strict path-free HouseholdModelDiagnosticDriftReport v1 JSON file",
    )
    validate_household_model_drift.add_argument("path")
    validate_household_model_drift.set_defaults(
        handler=command_validate_household_model_diagnostic_drift
    )

    select_household_model_diagnostic = subparsers.add_parser(
        "select-household-model-presence-working-point",
        help=(
            "select one diagnostic-only next shadow working point under explicit "
            "model-generated presence constraints"
        ),
    )
    select_household_model_diagnostic.add_argument("path")
    select_household_model_diagnostic.add_argument(
        "--minimum-precision",
        required=True,
        type=float,
    )
    select_household_model_diagnostic.add_argument(
        "--minimum-recall",
        required=True,
        type=float,
    )
    select_household_model_diagnostic.add_argument(
        "--minimum-partial-visibility-recall",
        required=True,
        type=float,
    )
    select_household_model_diagnostic.add_argument(
        "--allow-no-partial-visibility-samples",
        action="store_true",
    )
    select_household_model_diagnostic.set_defaults(
        handler=command_select_household_model_presence_working_point
    )

    household_model_temporal_diagnostic = subparsers.add_parser(
        "evaluate-household-model-temporal-diagnostic",
        help=(
            "diagnose causal detector holds by anonymous camera and clear/partial "
            "visibility using only a saved model review and detector scans"
        ),
    )
    household_model_temporal_diagnostic.add_argument("--review", required=True)
    household_model_temporal_diagnostic.add_argument(
        "--media-admission",
        required=True,
        action="append",
    )
    household_model_temporal_diagnostic.add_argument(
        "--detector-scan",
        required=True,
        action="append",
    )
    household_model_temporal_diagnostic.add_argument(
        "--threshold",
        required=True,
        action="append",
        type=float,
    )
    household_model_temporal_diagnostic.add_argument(
        "--hold-seconds",
        required=True,
        action="append",
        type=float,
    )
    household_model_temporal_diagnostic.add_argument(
        "--bin-seconds",
        required=True,
        type=float,
    )
    household_model_temporal_diagnostic.add_argument("--output", required=True)
    household_model_temporal_diagnostic.set_defaults(
        handler=command_evaluate_household_model_temporal_diagnostic
    )

    validate_household_model_temporal = subparsers.add_parser(
        "validate-household-model-temporal-diagnostic",
        help=("validate a strict path-free HouseholdModelTemporalDiagnosticReport v1 JSON file"),
    )
    validate_household_model_temporal.add_argument("path")
    validate_household_model_temporal.set_defaults(
        handler=command_validate_household_model_temporal_diagnostic
    )

    select_household_model_temporal = subparsers.add_parser(
        "select-household-model-temporal-working-point",
        help=(
            "select one diagnostic-only next shadow temporal point under aggregate, "
            "visibility, per-negative-camera, and hold constraints"
        ),
    )
    select_household_model_temporal.add_argument("path")
    select_household_model_temporal.add_argument("--minimum-precision", required=True, type=float)
    select_household_model_temporal.add_argument("--minimum-recall", required=True, type=float)
    select_household_model_temporal.add_argument(
        "--minimum-positive-camera-recall", required=True, type=float
    )
    select_household_model_temporal.add_argument(
        "--minimum-clear-visibility-recall", required=True, type=float
    )
    select_household_model_temporal.add_argument(
        "--minimum-partial-visibility-recall", required=True, type=float
    )
    select_household_model_temporal.add_argument(
        "--maximum-negative-camera-false-positive-rate",
        required=True,
        type=float,
    )
    select_household_model_temporal.add_argument(
        "--maximum-hold-seconds", required=True, type=float
    )
    select_household_model_temporal.add_argument(
        "--require-clear-visibility-samples", action="store_true"
    )
    select_household_model_temporal.add_argument(
        "--allow-no-partial-visibility-samples", action="store_true"
    )
    select_household_model_temporal.add_argument(
        "--allow-no-negative-camera-samples", action="store_true"
    )
    select_household_model_temporal.add_argument(
        "--allow-single-label-camera-strata", action="store_true"
    )
    select_household_model_temporal.set_defaults(
        handler=command_select_household_model_temporal_working_point
    )

    combine_household_model_temporal_batches = subparsers.add_parser(
        "combine-household-model-temporal-diagnostic-batches",
        help=(
            "combine two or more disjoint same-model temporal diagnostic reports "
            "into one camera-stratified diagnostic-only aggregate"
        ),
    )
    combine_household_model_temporal_batches.add_argument("paths", nargs="+")
    combine_household_model_temporal_batches.add_argument("--output", required=True)
    combine_household_model_temporal_batches.set_defaults(
        handler=command_combine_household_model_temporal_diagnostic_batches
    )

    validate_household_model_temporal_batch = subparsers.add_parser(
        "validate-household-model-temporal-batch-diagnostic",
        help=(
            "validate a strict path-free HouseholdModelTemporalBatchDiagnosticReport v1 JSON file"
        ),
    )
    validate_household_model_temporal_batch.add_argument("path")
    validate_household_model_temporal_batch.set_defaults(
        handler=command_validate_household_model_temporal_batch_diagnostic
    )

    select_household_model_temporal_batch = subparsers.add_parser(
        "select-household-model-temporal-batch-working-point",
        help=(
            "select one diagnostic-only cross-batch temporal shadow point under "
            "aggregate, visibility, per-camera, and hold constraints"
        ),
    )
    select_household_model_temporal_batch.add_argument("path")
    select_household_model_temporal_batch.add_argument(
        "--minimum-precision", required=True, type=float
    )
    select_household_model_temporal_batch.add_argument(
        "--minimum-recall", required=True, type=float
    )
    select_household_model_temporal_batch.add_argument(
        "--minimum-positive-camera-recall", required=True, type=float
    )
    select_household_model_temporal_batch.add_argument(
        "--minimum-clear-visibility-recall", required=True, type=float
    )
    select_household_model_temporal_batch.add_argument(
        "--minimum-partial-visibility-recall", required=True, type=float
    )
    select_household_model_temporal_batch.add_argument(
        "--maximum-negative-camera-false-positive-rate",
        required=True,
        type=float,
    )
    select_household_model_temporal_batch.add_argument(
        "--maximum-hold-seconds", required=True, type=float
    )
    select_household_model_temporal_batch.add_argument(
        "--require-clear-visibility-samples", action="store_true"
    )
    select_household_model_temporal_batch.add_argument(
        "--allow-no-partial-visibility-samples", action="store_true"
    )
    select_household_model_temporal_batch.add_argument(
        "--allow-no-negative-camera-samples", action="store_true"
    )
    select_household_model_temporal_batch.add_argument(
        "--allow-single-label-camera-strata", action="store_true"
    )
    select_household_model_temporal_batch.set_defaults(
        handler=command_select_household_model_temporal_batch_working_point
    )

    detector_tracking = subparsers.add_parser(
        "prepare-detector-tracking-evaluation",
        help=(
            "convert an exact detector quality run into sampled detector frames "
            "with publisher-subject matches and explicit false positives"
        ),
    )
    detector_tracking.add_argument("--ground-truth-frames", required=True)
    detector_tracking.add_argument("--ground-truth-review", required=True)
    detector_tracking.add_argument("--ground-truth-admission", required=True)
    detector_tracking.add_argument("--media-admission", required=True)
    detector_tracking.add_argument("--media-asset", required=True)
    detector_tracking.add_argument("--detector-scan", required=True)
    detector_tracking.add_argument("--quality-report", required=True)
    detector_tracking.add_argument("--frames-output", required=True)
    detector_tracking.add_argument("--review-output", required=True)
    detector_tracking.set_defaults(handler=command_prepare_detector_tracking_evaluation)

    detector_tracking_batch = subparsers.add_parser(
        "prepare-detector-tracking-batch",
        help=(
            "convert an admitted detector scan into nominal-grid or actual container-PTS "
            "tracker frames, including empty samples, without claiming ground truth"
        ),
    )
    detector_tracking_batch.add_argument("--media-admission", required=True)
    detector_tracking_batch.add_argument("--media-asset", required=True)
    detector_tracking_batch.add_argument("--detector-scan", required=True)
    detector_tracking_batch.add_argument(
        "--timebase",
        choices=("nominal", "actual_pts"),
        default="nominal",
        help="nominal source grid (default) or only actual v2 container-PTS samples",
    )
    detector_tracking_batch.add_argument(
        "--source-fps",
        type=float,
        help="required for nominal timebase; forbidden for actual_pts",
    )
    detector_tracking_batch.add_argument(
        "--source-frame-count",
        type=int,
        help="required for nominal timebase; forbidden for actual_pts",
    )
    detector_tracking_batch.add_argument(
        "--timeline-start",
        required=True,
        help=(
            "timezone-aware ISO-8601 local ordering anchor, not an inferred capture timestamp; "
            "actual_pts anchors PTS zero and does not subtract the first sample timestamp"
        ),
    )
    detector_tracking_batch.add_argument("--output", required=True)
    detector_tracking_batch.set_defaults(handler=command_prepare_detector_tracking_batch)

    verify_yolo_pose = subparsers.add_parser(
        "verify-yolo-pose-model",
        help=("verify a reviewed local YOLO pose weight without importing or running the model"),
    )
    verify_yolo_pose.add_argument("--manifest", required=True)
    verify_yolo_pose.add_argument("--weights", required=True)
    verify_yolo_pose.set_defaults(handler=command_verify_yolo_pose_model)

    pose_scan = subparsers.add_parser(
        "pose-scan-file",
        help="scan one bounded local video with a reviewed offline YOLO pose model",
    )
    pose_scan.add_argument("--source", required=True)
    pose_scan.add_argument("--camera-id", required=True)
    pose_scan.add_argument("--manifest", required=True)
    pose_scan.add_argument("--weights", required=True)
    pose_scan.add_argument("--runner-executable", required=True)
    pose_scan.add_argument("--output", required=True)
    pose_scan.add_argument("--device", default="cpu")
    pose_scan.add_argument("--confidence", type=float, default=0.35)
    pose_scan.add_argument("--sample-fps", type=float, default=1.0)
    pose_scan.add_argument("--max-frames", type=int, default=1_000)
    pose_scan.add_argument("--image-size", type=int, default=640)
    pose_scan.add_argument("--rotation-degrees", type=int, choices=(0, 90, 180, 270), default=0)
    pose_scan.add_argument("--timeout-seconds", type=float, default=600.0)
    pose_scan.set_defaults(handler=command_pose_scan_file)

    admitted_pose_scan = subparsers.add_parser(
        "pose-scan-admitted-file",
        help=("scan one admitted bounded local video with a reviewed offline YOLO pose model"),
    )
    admitted_pose_scan.add_argument("--source", required=True)
    admitted_pose_scan.add_argument("--source-admission", required=True)
    admitted_pose_scan.add_argument("--camera-id", required=True)
    admitted_pose_scan.add_argument("--manifest", required=True)
    admitted_pose_scan.add_argument("--weights", required=True)
    admitted_pose_scan.add_argument("--runner-executable", required=True)
    admitted_pose_scan.add_argument("--output", required=True)
    admitted_pose_scan.add_argument("--device", default="cpu")
    admitted_pose_scan.add_argument("--confidence", type=float, default=0.35)
    admitted_pose_scan.add_argument("--sample-fps", type=float, default=1.0)
    admitted_pose_scan.add_argument("--max-frames", type=int, default=1_000)
    admitted_pose_scan.add_argument("--image-size", type=int, default=640)
    admitted_pose_scan.add_argument(
        "--rotation-degrees",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
    )
    admitted_pose_scan.add_argument("--timeout-seconds", type=float, default=600.0)
    admitted_pose_scan.set_defaults(handler=command_pose_scan_admitted_file)

    pose_quality = subparsers.add_parser(
        "evaluate-pose-quality",
        help=(
            "measure keypoint coverage and adjacent-frame coordinate stability without "
            "identity or posture inference"
        ),
    )
    pose_quality.add_argument("--pose-frames", required=True)
    pose_quality.add_argument("--output", required=True)
    pose_quality.add_argument("--keypoint-confidence", type=float, default=0.5)
    pose_quality.add_argument("--min-usable-keypoints", type=int, default=8)
    pose_quality.add_argument("--min-pair-bbox-iou", type=float, default=0.2)
    pose_quality.add_argument("--min-shared-keypoints", type=int, default=8)
    pose_quality.add_argument("--image-change-threshold", type=float, default=0.05)
    pose_quality.add_argument("--shape-change-threshold", type=float, default=0.05)
    pose_quality.set_defaults(handler=command_evaluate_pose_quality)

    validate_pose_fusion_profile = subparsers.add_parser(
        "validate-household-pose-fusion-profile",
        help="validate one strict evaluation-only per-camera detector/pose profile",
    )
    validate_pose_fusion_profile.add_argument("path")
    validate_pose_fusion_profile.set_defaults(
        handler=command_validate_household_pose_fusion_profile
    )

    evaluate_pose_fusion = subparsers.add_parser(
        "evaluate-household-pose-fusion-diagnostic",
        help=(
            "replay one explicit per-camera detector/pose OR profile over admitted "
            "saved scans without reading media"
        ),
    )
    evaluate_pose_fusion.add_argument("--review", required=True)
    evaluate_pose_fusion.add_argument(
        "--media-admission",
        action="append",
        required=True,
    )
    evaluate_pose_fusion.add_argument(
        "--detector-scan",
        action="append",
        required=True,
    )
    evaluate_pose_fusion.add_argument(
        "--pose-scan",
        action="append",
        required=True,
    )
    evaluate_pose_fusion.add_argument("--profile", required=True)
    evaluate_pose_fusion.add_argument("--bin-seconds", type=float, default=5.0)
    evaluate_pose_fusion.add_argument("--output", required=True)
    evaluate_pose_fusion.set_defaults(handler=command_evaluate_household_pose_fusion_diagnostic)

    validate_pose_fusion = subparsers.add_parser(
        "validate-household-pose-fusion-diagnostic",
        help="validate one strict evaluation-only detector/pose fusion report",
    )
    validate_pose_fusion.add_argument("path")
    validate_pose_fusion.set_defaults(handler=command_validate_household_pose_fusion_diagnostic)

    combine_pose_fusion = subparsers.add_parser(
        "combine-household-pose-fusion-diagnostic-batches",
        help="combine disjoint same-profile detector/pose fusion diagnostics",
    )
    combine_pose_fusion.add_argument("paths", nargs="+")
    combine_pose_fusion.add_argument("--output", required=True)
    combine_pose_fusion.set_defaults(
        handler=command_combine_household_pose_fusion_diagnostic_batches
    )

    validate_pose_fusion_batch = subparsers.add_parser(
        "validate-household-pose-fusion-batch-diagnostic",
        help="validate one strict cross-batch detector/pose fusion report",
    )
    validate_pose_fusion_batch.add_argument("path")
    validate_pose_fusion_batch.set_defaults(
        handler=command_validate_household_pose_fusion_batch_diagnostic
    )

    attest_pose_fusion_holdout = subparsers.add_parser(
        "attest-household-pose-fusion-temporal-holdout",
        help=(
            "bind a frozen-profile temporal holdout to its disjoint profile-selection "
            "batch without granting accuracy or deployment authority"
        ),
    )
    attest_pose_fusion_holdout.add_argument("--selection-batch", required=True)
    attest_pose_fusion_holdout.add_argument("--holdout-report", required=True)
    attest_pose_fusion_holdout.add_argument(
        "--confirm-selection-batch-used-for-profile-selection",
        action="store_true",
    )
    attest_pose_fusion_holdout.add_argument(
        "--confirm-profile-frozen-before-holdout-capture",
        action="store_true",
    )
    attest_pose_fusion_holdout.add_argument("--output", required=True)
    attest_pose_fusion_holdout.set_defaults(
        handler=command_attest_household_pose_fusion_temporal_holdout
    )

    validate_pose_fusion_holdout = subparsers.add_parser(
        "validate-household-pose-fusion-temporal-holdout",
        help="validate one strict frozen-profile temporal holdout attestation",
    )
    validate_pose_fusion_holdout.add_argument("path")
    validate_pose_fusion_holdout.set_defaults(
        handler=command_validate_household_pose_fusion_temporal_holdout
    )

    import_human_presence_audit = subparsers.add_parser(
        "import-household-human-presence-audit",
        help=(
            "strictly bind a completed local blind human frame review to its "
            "uniform selection manifest"
        ),
    )
    import_human_presence_audit.add_argument("--audit-manifest", required=True)
    import_human_presence_audit.add_argument("--result", required=True)
    import_human_presence_audit.add_argument("--selection-manifest", required=True)
    import_human_presence_audit.add_argument("--output", required=True)
    import_human_presence_audit.set_defaults(handler=command_import_household_human_presence_audit)

    validate_human_presence_audit = subparsers.add_parser(
        "validate-household-human-presence-audit",
        help="validate one strict path-free single-human blind frame audit",
    )
    validate_human_presence_audit.add_argument("path")
    validate_human_presence_audit.set_defaults(
        handler=command_validate_household_human_presence_audit
    )

    evaluate_human_pose_fusion_audit = subparsers.add_parser(
        "evaluate-household-human-pose-fusion-audit",
        help=(
            "replay a frozen temporal-holdout detector/Pose profile against a "
            "bound single-human blind frame reference"
        ),
    )
    evaluate_human_pose_fusion_audit.add_argument("--audit", required=True)
    evaluate_human_pose_fusion_audit.add_argument("--model-review", required=True)
    evaluate_human_pose_fusion_audit.add_argument(
        "--temporal-holdout-attestation",
        required=True,
    )
    evaluate_human_pose_fusion_audit.add_argument(
        "--media-admission",
        action="append",
        required=True,
    )
    evaluate_human_pose_fusion_audit.add_argument(
        "--detector-scan",
        action="append",
        required=True,
    )
    evaluate_human_pose_fusion_audit.add_argument(
        "--pose-scan",
        action="append",
        required=True,
    )
    evaluate_human_pose_fusion_audit.add_argument("--output", required=True)
    evaluate_human_pose_fusion_audit.set_defaults(
        handler=command_evaluate_household_human_pose_fusion_audit
    )

    validate_human_pose_fusion_audit = subparsers.add_parser(
        "validate-household-human-pose-fusion-audit",
        help="validate one conservative single-human frozen-fusion audit report",
    )
    validate_human_pose_fusion_audit.add_argument("path")
    validate_human_pose_fusion_audit.set_defaults(
        handler=command_validate_household_human_pose_fusion_audit
    )

    process_file = subparsers.add_parser("process-file", help="process one bounded local video")
    process_file.add_argument("--config", default="config/default.toml")
    process_file.add_argument("--source", required=True)
    process_file.add_argument("--camera-id")
    process_file.add_argument("--detector", choices=["mock", "scene", "yolo", "yolo_command"])
    process_file.add_argument("--database")
    process_file.add_argument("--output-root")
    process_file.add_argument(
        "--tracker-config",
        help=(
            "enable the bounded-file DetectionFrameBatch/IoU bridge with this "
            "strict tracker config; requires --zone-config"
        ),
    )
    process_file.add_argument(
        "--zone-config",
        help=(
            "run temporal zones over tracked detector facts and bind emitted "
            "signals to candidates; requires --tracker-config"
        ),
    )
    process_file.add_argument(
        "--detection-frames-output",
        help=(
            "exclusively save the path-free DetectionFrameBatch used by the bounded tracker bridge"
        ),
    )
    process_file.set_defaults(handler=command_process_file)

    list_events = subparsers.add_parser("list-events", help="list recent SQLite events")
    list_events.add_argument("--database", default="./data/events.sqlite3")
    list_events.add_argument("--limit", type=int, default=50)
    list_events.set_defaults(handler=command_list_events)

    report = subparsers.add_parser("report", help="generate Markdown and HTML daily reports")
    report.add_argument("--config", default="config/default.toml")
    report.add_argument("--database")
    report.add_argument("--output-root")
    report_date_group = report.add_mutually_exclusive_group()
    report_date_group.add_argument("--date", help="local date in YYYY-MM-DD")
    report_date_group.add_argument(
        "--day-offset",
        type=int,
        choices=range(-30, 1),
        metavar="[-30..0]",
        help="generate a recent local day relative to today; systemd uses -1",
    )
    report.set_defaults(handler=command_report)

    systemd_render = subparsers.add_parser(
        "systemd-render",
        help="render hardened user units without installing, enabling, or starting them",
    )
    systemd_render.add_argument("--profile", required=True)
    systemd_render.add_argument("--output-dir", required=True)
    systemd_render.set_defaults(handler=command_systemd_render)

    home_control = subparsers.add_parser(
        "home-control",
        help="control the fixed local camera, HA, and bounded shadow units as one stack",
    )
    home_control.add_argument("action", choices=HOME_CONTROL_ACTIONS)
    home_control.add_argument("--config", default="config/default.toml")
    home_control.add_argument("--ha-outbox-database")
    home_control.add_argument("--analysis-database")
    home_control.add_argument("--candidate-health", action="append")
    home_control.add_argument(
        "--candidate-health-stale-after-seconds",
        type=float,
        default=120.0,
    )
    home_control.add_argument("--candidate-soak-ledger")
    home_control.add_argument("--confirm-buffer-eviction", action="store_true")
    home_control.set_defaults(handler=command_home_control)

    home_daily_report = subparsers.add_parser(
        "home-daily-report",
        help="generate a path-free aggregate local daily report without reading media",
    )
    home_daily_report.add_argument("--config", default="config/default.toml")
    home_daily_report.add_argument("--database")
    home_daily_report.set_defaults(handler=command_home_daily_report)

    daily_digest_plan = subparsers.add_parser(
        "daily-digest-plan",
        help="hash admitted local clips and persist one facts-only daily highlight plan",
    )
    daily_digest_plan.add_argument("--config", default="config/default.toml")
    daily_digest_plan.add_argument("--database")
    daily_digest_plan.add_argument("--storage-root")
    daily_digest_plan.add_argument("--date", help="local date in YYYY-MM-DD")
    daily_digest_plan.add_argument(
        "--target-duration-seconds",
        type=float,
        default=60.0,
    )
    daily_digest_plan.add_argument("--confirmation", required=True)
    daily_digest_plan.set_defaults(handler=command_daily_digest_plan)

    daily_digest_status = subparsers.add_parser(
        "daily-digest-plan-status",
        help="inspect private daily digest plans without reading media or creating files",
    )
    daily_digest_status.add_argument("--config", default="config/default.toml")
    daily_digest_status.add_argument("--database")
    daily_digest_status.set_defaults(handler=command_daily_digest_plan_status)

    daily_digest_export = subparsers.add_parser(
        "daily-digest-export",
        help="decode one private facts-only plan into a bounded local MP4",
    )
    daily_digest_export.add_argument("--config", default="config/default.toml")
    daily_digest_export.add_argument("--database")
    daily_digest_export.add_argument("--storage-root")
    daily_digest_export.add_argument("--date", help="local date in YYYY-MM-DD")
    daily_digest_export.add_argument("--confirmation", required=True)
    daily_digest_export.set_defaults(handler=command_daily_digest_export)

    daily_digest_export_status = subparsers.add_parser(
        "daily-digest-export-status",
        help="inspect private digest exports without reading video or creating files",
    )
    daily_digest_export_status.add_argument("--config", default="config/default.toml")
    daily_digest_export_status.add_argument("--database")
    daily_digest_export_status.set_defaults(handler=command_daily_digest_export_status)

    home_dashboard = subparsers.add_parser(
        "home-dashboard",
        help="serve the fixed manual home controller on IPv4 loopback only",
    )
    home_dashboard.add_argument("--config", default="config/default.toml")
    home_dashboard.add_argument("--ha-outbox-database")
    home_dashboard.add_argument("--analysis-database")
    home_dashboard.add_argument("--candidate-health", action="append")
    home_dashboard.add_argument(
        "--candidate-health-stale-after-seconds",
        type=float,
        default=120.0,
    )
    home_dashboard.add_argument("--candidate-soak-ledger")
    home_dashboard.add_argument("--room-registry")
    home_dashboard.add_argument("--event-policy")
    home_dashboard.add_argument("--room-presence-samples", action="append")
    home_dashboard.add_argument("--room-presence-health", action="append")
    home_dashboard.add_argument("--port", type=int, default=8765)
    home_dashboard.add_argument("--max-runtime-seconds", type=float, default=28_800.0)
    home_dashboard.set_defaults(handler=command_home_dashboard)

    ab_run = subparsers.add_parser("ab-run", help="run sequential VLM A/B evaluation")
    ab_run.add_argument("--config", default="config/vlm-ab.example.toml")
    ab_run.set_defaults(handler=command_ab_run)

    validate_ab = subparsers.add_parser(
        "validate-ab-dataset",
        help="validate a strict local VLM evaluation JSONL dataset and frame hashes",
    )
    validate_ab.add_argument("path")
    validate_ab.set_defaults(handler=command_validate_ab_dataset)

    validate_source = subparsers.add_parser(
        "validate-source-admission",
        help=(
            "validate a strict external model/data source admission manifest "
            "without reading the referenced asset"
        ),
    )
    validate_source.add_argument("path")
    validate_source.set_defaults(handler=command_validate_source_admission)

    verify_source = subparsers.add_parser(
        "verify-source-asset",
        help=(
            "verify one bounded local external asset against an approved "
            "admission manifest without network access"
        ),
    )
    verify_source.add_argument("--manifest", required=True)
    verify_source.add_argument("--asset", required=True)
    verify_source.set_defaults(handler=command_verify_source_asset)

    retention = subparsers.add_parser("retention-scan", help="list expired media; never delete")
    retention.add_argument("--config", default="config/default.toml")
    retention.add_argument("--database")
    retention.set_defaults(handler=command_retention_scan)

    archive_plan = subparsers.add_parser(
        "archive-plan",
        help="hash a bounded set of local spool files into a strict private manifest",
    )
    archive_plan.add_argument("--archive-id", required=True)
    archive_plan.add_argument("--source-root", required=True)
    archive_plan.add_argument("--file", required=True, action="append")
    archive_plan.add_argument("--output", required=True)
    archive_plan.set_defaults(handler=command_archive_plan)

    archive_copy = subparsers.add_parser(
        "archive-copy",
        help="atomically copy one strict manifest without overwriting unknown files",
    )
    archive_copy.add_argument("--manifest", required=True)
    archive_copy.add_argument("--source-root", required=True)
    archive_copy.add_argument("--target-root", required=True)
    archive_copy.set_defaults(handler=command_archive_copy)

    archive_enqueue = subparsers.add_parser(
        "archive-enqueue",
        help="validate and persist one archive manifest without copying it",
    )
    archive_enqueue.add_argument("--config", default="config/default.toml")
    archive_enqueue.add_argument("--manifest", required=True)
    archive_enqueue.add_argument("--source-root", required=True)
    archive_enqueue.add_argument("--database")
    archive_enqueue.set_defaults(handler=command_archive_enqueue)

    archive_event = subparsers.add_parser(
        "archive-enqueue-event",
        help="bind one stored event's clip and evidence frames to an archive job",
    )
    archive_event.add_argument("--config", default="config/default.toml")
    archive_event.add_argument("--event-id", required=True)
    archive_event.add_argument("--source-root", required=True)
    archive_event.add_argument("--database")
    archive_event.set_defaults(handler=command_archive_enqueue_event)

    archive_report = subparsers.add_parser(
        "archive-enqueue-report",
        help="bind one generated Markdown/HTML daily report to an archive job",
    )
    archive_report.add_argument("--config", default="config/default.toml")
    archive_report.add_argument("--date", required=True, help="local date in YYYY-MM-DD")
    archive_report.add_argument("--source-root", required=True)
    archive_report.add_argument("--database")
    archive_report.set_defaults(handler=command_archive_enqueue_report)

    archive_run = subparsers.add_parser(
        "archive-run-once",
        help="claim and process at most one persisted archive job",
    )
    archive_run.add_argument("--config", default="config/default.toml")
    archive_run.add_argument("--target-root", required=True)
    archive_run.add_argument("--database")
    archive_run.set_defaults(handler=command_archive_run_once)

    archive_worker = subparsers.add_parser(
        "archive-worker",
        help="process archive jobs sequentially within explicit stopping bounds",
    )
    archive_worker.add_argument("--config", default="config/default.toml")
    archive_worker.add_argument("--target-root", required=True)
    archive_worker.add_argument("--database")
    archive_worker.add_argument("--max-jobs", type=int)
    archive_worker.add_argument("--idle-exit-seconds", type=float)
    archive_worker.add_argument("--max-runtime-seconds", type=float)
    archive_worker.add_argument("--poll-seconds", type=float)
    archive_worker.set_defaults(handler=command_archive_worker)

    archive_status = subparsers.add_parser(
        "archive-queue-status",
        help="show archive queue aggregates without exposing paths or manifests",
    )
    archive_status.add_argument("--config", default="config/default.toml")
    archive_status.add_argument("--database")
    archive_status.set_defaults(handler=command_archive_queue_status)

    archive_retry = subparsers.add_parser(
        "archive-retry",
        help="explicitly reset one failed archive job for another bounded attempt cycle",
    )
    archive_retry.add_argument("--config", default="config/default.toml")
    archive_retry.add_argument("--database")
    archive_retry.add_argument("--archive-id", required=True)
    archive_retry.set_defaults(handler=command_archive_retry)

    schema = subparsers.add_parser("schema", help="print the strict VLM JSON schema")
    schema.set_defaults(handler=command_schema)

    rtsp_soak = subparsers.add_parser(
        "rtsp-soak",
        help="continuously decode a loopback RTSP stream and report health metrics",
    )
    rtsp_soak.add_argument("--uri", required=True)
    rtsp_soak.add_argument("--seconds", type=float, default=7200.0)
    rtsp_soak.add_argument("--startup-timeout", type=float, default=30.0)
    rtsp_soak.add_argument("--ffmpeg", default="ffmpeg")
    rtsp_soak.add_argument("--output", help="create a new JSON result file; never overwrites")
    rtsp_soak.set_defaults(handler=command_rtsp_soak)

    rtsp_buffer = subparsers.add_parser(
        "rtsp-buffer",
        help="record a bounded RTSP segment buffer and write a redacted manifest",
    )
    rtsp_buffer.add_argument("--config", default="config/default.toml")
    rtsp_buffer.add_argument("--uri", required=True)
    rtsp_buffer.add_argument("--camera-id")
    rtsp_buffer.add_argument("--seconds", type=float, default=30.0)
    rtsp_buffer.add_argument("--buffer-root")
    rtsp_buffer.add_argument("--allow-non-loopback", action="store_true")
    rtsp_buffer.add_argument(
        "--confirm-ephemeral-eviction",
        action="store_true",
        help="one-shot confirmation; persistent buffer.eviction_enabled must also be true",
    )
    rtsp_buffer.set_defaults(handler=command_rtsp_buffer)

    media_health_status = subparsers.add_parser(
        "media-health-status",
        help="assess one MediaSource health snapshot without restarting any service",
    )
    media_health_status.add_argument("--config", default="config/default.toml")
    media_health_status.add_argument("--health", required=True)
    media_health_status.add_argument("--stale-after-seconds", type=float)
    media_health_status.add_argument("--future-tolerance-seconds", type=float)
    media_health_status.set_defaults(handler=command_media_health_status)

    visual_predicate_health_status = subparsers.add_parser(
        "visual-predicate-health-status",
        help="assess one private resident visual-predicate health snapshot read-only",
    )
    visual_predicate_health_status.add_argument("--health", required=True)
    visual_predicate_health_status.add_argument(
        "--stale-after-seconds",
        type=float,
        default=60.0,
    )
    visual_predicate_health_status.add_argument(
        "--future-tolerance-seconds",
        type=float,
        default=5.0,
    )
    visual_predicate_health_status.set_defaults(handler=command_visual_predicate_health_status)

    visual_predicate_shadow_run = subparsers.add_parser(
        "visual-predicate-shadow-run",
        help=(
            "run the default-off local atomic-predicate shadow path from explicitly "
            "admitted candidate metadata and private frames"
        ),
    )
    visual_predicate_shadow_run.add_argument("--config", default="config/default.toml")
    visual_predicate_shadow_run.add_argument("--database")
    visual_predicate_shadow_run.add_argument("--candidate", required=True)
    visual_predicate_shadow_run.add_argument(
        "--frame",
        required=True,
        action="append",
        help="FRAME_ID=/absolute/private/frame.jpg; repeat up to 12 times",
    )
    visual_predicate_shadow_run.add_argument("--person-detector-id")
    visual_predicate_shadow_run.add_argument(
        "--periodic-sample-due",
        action="store_true",
        help="explicit scheduling fact; it does not imply a visual predicate answer",
    )
    visual_predicate_shadow_run.set_defaults(handler=command_visual_predicate_shadow_run)

    visual_predicate_shadow_status = subparsers.add_parser(
        "visual-predicate-shadow-queue-status",
        help="show the path-free shadow attempt ledger without reading media",
    )
    visual_predicate_shadow_status.add_argument("--config", default="config/default.toml")
    visual_predicate_shadow_status.add_argument("--database")
    visual_predicate_shadow_status.set_defaults(
        handler=command_visual_predicate_shadow_queue_status
    )

    visual_predicate_shadow_report = subparsers.add_parser(
        "visual-predicate-shadow-report",
        help="show recent path-free shadow states and atomic answers without reading media",
    )
    visual_predicate_shadow_report.add_argument("--config", default="config/default.toml")
    visual_predicate_shadow_report.add_argument("--database")
    visual_predicate_shadow_report.add_argument("--limit", type=int, default=100)
    visual_predicate_shadow_report.set_defaults(handler=command_visual_predicate_shadow_report)

    visual_predicate_review_report = subparsers.add_parser(
        "visual-predicate-review-report",
        help=(
            "apply the deterministic shadow-only manual-review policy to recent "
            "digest-bound atomic predicate results without reading media"
        ),
    )
    visual_predicate_review_report.add_argument("--config", default="config/default.toml")
    visual_predicate_review_report.add_argument("--database")
    visual_predicate_review_report.add_argument("--limit", type=int, default=100)
    visual_predicate_review_report.set_defaults(handler=command_visual_predicate_review_report)

    visual_predicate_calibration = subparsers.add_parser(
        "visual-predicate-human-calibration-report",
        help=(
            "compare latest local human visible-fact labels with bound shadow "
            "answers without reading media or changing decisions"
        ),
    )
    visual_predicate_calibration.add_argument(
        "--config",
        default="config/default.toml",
    )
    visual_predicate_calibration.add_argument("--database")
    visual_predicate_calibration.add_argument("--limit", type=int, default=1_000)
    visual_predicate_calibration.set_defaults(
        handler=command_visual_predicate_human_calibration_report
    )

    visual_predicate_shadow_retry = subparsers.add_parser(
        "visual-predicate-shadow-retry",
        help="explicitly reset one failed shadow attempt without changing main analysis",
    )
    visual_predicate_shadow_retry.add_argument("--config", default="config/default.toml")
    visual_predicate_shadow_retry.add_argument("--database")
    visual_predicate_shadow_retry.add_argument("--event-id", required=True)
    visual_predicate_shadow_retry.add_argument("--max-active", type=int, default=256)
    visual_predicate_shadow_retry.set_defaults(handler=command_visual_predicate_shadow_retry)

    visual_predicate_shadow_next = subparsers.add_parser(
        "visual-predicate-shadow-run-next",
        help=(
            "sync pending hardware evidence and completed detector events, then process "
            "at most one isolated shadow attempt"
        ),
    )
    visual_predicate_shadow_next.add_argument("--config", default="config/default.toml")
    visual_predicate_shadow_next.add_argument("--database")
    visual_predicate_shadow_next.add_argument("--max-active", type=int, default=256)
    visual_predicate_shadow_next.add_argument("--lease-seconds", type=int, default=300)
    visual_predicate_shadow_next.add_argument("--max-attempts", type=int, default=3)
    visual_predicate_shadow_next.add_argument("--retry-delay-seconds", type=int, default=30)
    visual_predicate_shadow_next.set_defaults(handler=command_visual_predicate_shadow_run_next)

    visual_predicate_shadow_worker = subparsers.add_parser(
        "visual-predicate-shadow-worker",
        help="reuse one local adapter while consuming isolated shadow attempts",
    )
    visual_predicate_shadow_worker.add_argument("--config", default="config/default.toml")
    visual_predicate_shadow_worker.add_argument("--database")
    visual_predicate_shadow_worker.add_argument("--max-active", type=int, default=256)
    visual_predicate_shadow_worker.add_argument("--lease-seconds", type=int, default=300)
    visual_predicate_shadow_worker.add_argument("--max-attempts", type=int, default=3)
    visual_predicate_shadow_worker.add_argument("--retry-delay-seconds", type=int, default=30)
    visual_predicate_shadow_worker.add_argument("--max-jobs", type=int)
    visual_predicate_shadow_worker.add_argument("--idle-exit-seconds", type=float)
    visual_predicate_shadow_worker.add_argument("--max-runtime-seconds", type=float)
    visual_predicate_shadow_worker.add_argument("--poll-seconds", type=float, default=2.0)
    visual_predicate_shadow_worker.set_defaults(handler=command_visual_predicate_shadow_worker)

    runtime_status = subparsers.add_parser(
        "runtime-status",
        help="aggregate read-only media, queue, storage, GPU, and safety status",
    )
    runtime_status.add_argument("--config", default="config/default.toml")
    runtime_status.add_argument(
        "--health",
        required=True,
        action="append",
        help="MediaSourceHealth file; repeat for each required camera",
    )
    runtime_status.add_argument("--database")
    runtime_status.add_argument("--storage-root")
    runtime_status.add_argument("--min-free-gib", type=float, default=5.0)
    runtime_status.add_argument(
        "--max-pending-age-seconds",
        type=float,
        default=3_600.0,
    )
    runtime_status.add_argument(
        "--max-archive-pending-age-seconds",
        type=float,
        default=86_400.0,
    )
    runtime_status.add_argument(
        "--max-notification-pending-age-seconds",
        type=float,
        help="override the configured notification backlog age threshold",
    )
    runtime_status.set_defaults(handler=command_runtime_status)

    candidate_health_status = subparsers.add_parser(
        "candidate-health-status",
        help="read path-free freshness and execution metrics for candidate workers",
    )
    candidate_health_status.add_argument(
        "--health",
        required=True,
        action="append",
        help="private CandidateWorkerHealth file; repeat for each camera",
    )
    candidate_health_status.add_argument(
        "--stale-after-seconds",
        type=float,
        default=120.0,
    )
    candidate_health_status.add_argument(
        "--future-tolerance-seconds",
        type=float,
        default=5.0,
    )
    candidate_health_status.set_defaults(handler=command_candidate_health_status)

    candidate_presence_health_init = subparsers.add_parser(
        "candidate-presence-health-init",
        help="create or validate one stopped lifetime health baseline for a presence-only worker",
    )
    candidate_presence_health_init.add_argument("--config", default="config/default.toml")
    candidate_presence_health_init.add_argument("--camera-id", required=True)
    candidate_presence_health_init.add_argument("--health", required=True)
    candidate_presence_health_init.add_argument(
        "--detector",
        choices=["scene", "yolo", "yolo_command"],
    )
    candidate_presence_health_init.add_argument("--image-size", type=int)
    candidate_presence_health_init.add_argument(
        "--rotation-degrees",
        type=int,
        choices=(0, 90, 180, 270),
    )
    candidate_presence_health_init.set_defaults(handler=command_candidate_presence_health_init)

    candidate_soak_start = subparsers.add_parser(
        "candidate-soak-start",
        help="capture a private v2 baseline before the fixed manual YOLO soak",
    )
    candidate_soak_start.add_argument("--ledger", required=True)
    candidate_soak_start.add_argument("--health", required=True, action="append")
    candidate_soak_start.add_argument(
        "--expected-duration-seconds",
        type=int,
        default=86_400,
    )
    candidate_soak_start.set_defaults(handler=command_candidate_soak_start)

    candidate_soak_finish = subparsers.add_parser(
        "candidate-soak-finish",
        help="finish the active private YOLO soak after all fixed workers stop",
    )
    candidate_soak_finish.add_argument("--ledger", required=True)
    candidate_soak_finish.add_argument("--health", required=True, action="append")
    candidate_soak_finish.set_defaults(handler=command_candidate_soak_finish)

    candidate_soak_status = subparsers.add_parser(
        "candidate-soak-status",
        help="read the path-free latest manual YOLO soak report",
    )
    candidate_soak_status.add_argument("--ledger", required=True)
    candidate_soak_status.set_defaults(handler=command_candidate_soak_status)

    runtime_trial_start = subparsers.add_parser(
        "runtime-trial-start",
        help="capture a private baseline for one bounded home-runtime trial",
    )
    runtime_trial_start.add_argument("--ledger", required=True)
    runtime_trial_start.add_argument("--health", required=True, action="append")
    runtime_trial_start.add_argument("--database", required=True)
    runtime_trial_start.add_argument(
        "--expected-duration-seconds",
        type=int,
        required=True,
    )
    runtime_trial_start.add_argument(
        "--health-wait-seconds",
        type=float,
        default=0.0,
    )
    runtime_trial_start.add_argument("--shadow-expected", action="store_true")
    runtime_trial_start.set_defaults(handler=command_runtime_trial_start)

    runtime_trial_finish = subparsers.add_parser(
        "runtime-trial-finish",
        help="settle one bounded home-runtime trial after its workers stop",
    )
    runtime_trial_finish.add_argument("--ledger", required=True)
    runtime_trial_finish.add_argument("--health", required=True, action="append")
    runtime_trial_finish.add_argument("--database", required=True)
    runtime_trial_finish.set_defaults(handler=command_runtime_trial_finish)

    runtime_trial_status = subparsers.add_parser(
        "runtime-trial-status",
        help="read the latest path-free bounded home-runtime trial result",
    )
    runtime_trial_status.add_argument("--ledger", required=True)
    runtime_trial_status.set_defaults(handler=command_runtime_trial_status)

    room_registry_status = subparsers.add_parser(
        "room-registry-status",
        help="read the private physical-camera, logical-stream, and room topology",
    )
    room_registry_status.add_argument("--registry", required=True)
    room_registry_status.set_defaults(handler=command_room_registry_status)

    room_presence_evaluate = subparsers.add_parser(
        "room-presence-evaluate",
        help="evaluate conservative identity-free room presence from bounded scan facts",
    )
    room_presence_evaluate.add_argument("--registry", required=True)
    room_presence_evaluate.add_argument("--samples", required=True, action="append")
    room_presence_evaluate.add_argument(
        "--health",
        action="append",
        default=[],
        help="optional private CandidateWorkerHealth file; repeat for each stream",
    )
    room_presence_evaluate.add_argument(
        "--checked-at",
        help="optional timezone-aware ISO 8601 evaluation time",
    )
    room_presence_evaluate.set_defaults(handler=command_room_presence_evaluate)

    release_audit = subparsers.add_parser(
        "release-audit",
        help="scan the Git index for private or heavyweight release artifacts",
    )
    release_audit.add_argument("--root", default=".")
    release_audit.add_argument(
        "--max-file-bytes",
        type=int,
        default=10 * 1024 * 1024,
    )
    release_audit.add_argument(
        "--worktree",
        action="store_true",
        help="also audit modified and untracked non-ignored worktree files",
    )
    release_audit.set_defaults(handler=command_release_audit)

    release_sbom = subparsers.add_parser(
        "release-sbom",
        help="generate a deterministic CycloneDX source SBOM from the safe Git index",
    )
    release_sbom.add_argument("--root", default=".")
    release_sbom.add_argument(
        "--output",
        help="exclusively create one .cdx.json file in an existing directory",
    )
    release_sbom.set_defaults(handler=command_release_sbom)

    release_bundle = subparsers.add_parser(
        "release-bundle",
        help="atomically create a deterministic audited source tar.gz with SBOM",
    )
    release_bundle.add_argument("--root", default=".")
    release_bundle.add_argument("--output", required=True)
    release_bundle.set_defaults(handler=command_release_bundle)

    candidate_worker = subparsers.add_parser(
        "candidate-worker",
        help="scan new buffer segments and enqueue mature low-cost candidate events",
    )
    candidate_worker.add_argument("--config", default="config/default.toml")
    candidate_worker.add_argument("--manifest", required=True)
    candidate_worker.add_argument("--state")
    candidate_worker.add_argument("--health")
    candidate_worker.add_argument(
        "--presence",
        help="absolute private path for bounded identity-free per-scan person facts",
    )
    candidate_worker.add_argument(
        "--presence-only",
        action="store_true",
        help=(
            "write bounded person facts without extracting evidence or touching the analysis queue"
        ),
    )
    candidate_worker.add_argument(
        "--pose-presence-shadow-profile",
        help="frozen evaluation profile for a separate detector/Pose shadow fact stream",
    )
    candidate_worker.add_argument(
        "--pose-presence-shadow-attestation",
        help="temporal holdout attestation bound to the frozen shadow profile",
    )
    candidate_worker.add_argument(
        "--pose-presence-shadow-audit",
        help="single-human blind audit report bound to the frozen shadow profile",
    )
    candidate_worker.add_argument(
        "--pose-presence-shadow-output",
        help="absolute private path for bounded shadow-only presence samples",
    )
    candidate_worker.add_argument("--pose-shadow-detector-manifest")
    candidate_worker.add_argument("--pose-shadow-detector-weights")
    candidate_worker.add_argument(
        "--pose-shadow-detector-runner-executable",
        type=_absolute_runner_executable,
    )
    candidate_worker.add_argument("--pose-shadow-detector-device", default="0")
    candidate_worker.add_argument("--pose-shadow-manifest")
    candidate_worker.add_argument("--pose-shadow-weights")
    candidate_worker.add_argument(
        "--pose-shadow-runner-executable",
        type=_absolute_runner_executable,
    )
    candidate_worker.add_argument("--pose-shadow-device", default="0")
    candidate_worker.add_argument("--pose-shadow-timeout-seconds", type=float, default=600.0)
    candidate_worker.add_argument(
        "--pose-shadow-min-interval-seconds",
        type=float,
        default=300.0,
        help="minimum persisted per-camera interval between experimental shadow scans",
    )
    candidate_worker.add_argument("--database")
    candidate_worker.add_argument("--output-root")
    candidate_worker.add_argument("--detector", choices=["mock", "scene", "yolo", "yolo_command"])
    candidate_worker.add_argument("--max-segments", type=int)
    candidate_worker.add_argument("--idle-exit-seconds", type=float)
    candidate_worker.add_argument("--max-runtime-seconds", type=float)
    candidate_worker.add_argument("--poll-seconds", type=float)
    candidate_worker.add_argument("--image-size", type=int)
    candidate_worker.add_argument("--rotation-degrees", type=int, choices=(0, 90, 180, 270))
    candidate_worker.add_argument(
        "--allow-fixture-detector",
        action="store_true",
        help="explicitly allow deterministic mock detections for tests only",
    )
    candidate_worker.set_defaults(handler=command_candidate_worker)

    buffer_extract = subparsers.add_parser(
        "buffer-extract",
        help="assemble an event clip and keyframes from a segment manifest",
    )
    buffer_extract.add_argument("--config", default="config/default.toml")
    buffer_extract.add_argument("--manifest", required=True)
    buffer_extract.add_argument("--trigger-at", required=True)
    buffer_extract.add_argument("--event-id")
    buffer_extract.add_argument("--pre-seconds", type=float)
    buffer_extract.add_argument("--post-seconds", type=float)
    buffer_extract.add_argument("--keyframes", type=int)
    buffer_extract.add_argument("--output-root")
    buffer_extract.set_defaults(handler=command_buffer_extract)

    process_buffer = subparsers.add_parser(
        "process-buffer-event",
        help="persist one buffered RTSP event through VLM, SQLite, and daily reports",
    )
    process_buffer.add_argument("--config", default="config/default.toml")
    process_buffer.add_argument("--manifest", required=True)
    process_buffer.add_argument("--trigger-at", required=True)
    process_buffer.add_argument("--event-id")
    process_buffer.add_argument("--pre-seconds", type=float)
    process_buffer.add_argument("--post-seconds", type=float)
    process_buffer.add_argument("--keyframes", type=int)
    process_buffer.add_argument("--database")
    process_buffer.add_argument("--output-root")
    process_buffer.set_defaults(handler=command_process_buffer_event)

    enqueue_buffer = subparsers.add_parser(
        "enqueue-buffer-event",
        help="extract buffered evidence and persist a bounded analysis job without loading VLM",
    )
    enqueue_buffer.add_argument("--config", default="config/default.toml")
    enqueue_buffer.add_argument("--manifest", required=True)
    enqueue_buffer.add_argument("--trigger-at", required=True)
    enqueue_buffer.add_argument("--event-id")
    enqueue_buffer.add_argument("--pre-seconds", type=float)
    enqueue_buffer.add_argument("--post-seconds", type=float)
    enqueue_buffer.add_argument("--keyframes", type=int)
    enqueue_buffer.add_argument("--database")
    enqueue_buffer.add_argument("--output-root")
    enqueue_buffer.set_defaults(handler=command_enqueue_buffer_event)

    enqueue_signal = subparsers.add_parser(
        "enqueue-signal-event",
        help="bind one strict hardware/HA signal to buffered evidence and the analysis queue",
    )
    enqueue_signal.add_argument("--config", default="config/default.toml")
    enqueue_signal.add_argument("--signal-json", required=True)
    enqueue_signal.add_argument("--manifest", required=True)
    enqueue_signal.add_argument("--pre-seconds", type=float)
    enqueue_signal.add_argument("--post-seconds", type=float)
    enqueue_signal.add_argument("--keyframes", type=int)
    enqueue_signal.add_argument("--database")
    enqueue_signal.add_argument("--output-root")
    enqueue_signal.set_defaults(handler=command_enqueue_signal_event)

    track_iou_run = subparsers.add_parser(
        "track-iou-run",
        help="associate normalized detections into anonymous short-lived tracks",
    )
    track_iou_run.add_argument("--tracker-config", required=True)
    track_iou_run.add_argument("--frames", required=True)
    track_iou_run.add_argument("--state", required=True)
    track_iou_run.set_defaults(handler=command_track_iou_run)

    track_motion_run = subparsers.add_parser(
        "track-motion-run",
        help=(
            "associate normalized detections with the persistent evaluated two-stage motion tracker"
        ),
    )
    track_motion_run.add_argument("--tracker-config", required=True)
    track_motion_run.add_argument("--frames", required=True)
    track_motion_run.add_argument("--state", required=True)
    track_motion_run.set_defaults(handler=command_track_motion_run)

    tracker_review = subparsers.add_parser(
        "prepare-tracker-review",
        help=(
            "create a path-free pending human association review for every "
            "detection in one fixed batch"
        ),
    )
    tracker_review.add_argument("--frames", required=True)
    tracker_review.add_argument("--output", required=True)
    tracker_review.add_argument("--source-admission-sha256")
    tracker_review.set_defaults(handler=command_prepare_tracker_review)

    caviar_import = subparsers.add_parser(
        "import-caviar-tracking-ground-truth",
        help=(
            "verify and convert admitted CAVIAR CVML person boxes into a "
            "path-free association batch and publisher-human ground truth"
        ),
    )
    caviar_import.add_argument("--xml", required=True)
    caviar_import.add_argument("--source-admission", required=True)
    caviar_import.add_argument("--first-frame", type=int)
    caviar_import.add_argument("--last-frame", type=int)
    caviar_import.add_argument("--frames-output", required=True)
    caviar_import.add_argument("--review-output", required=True)
    caviar_import.set_defaults(handler=command_import_caviar_tracking_ground_truth)

    validate_tracker_review = subparsers.add_parser(
        "validate-tracker-review",
        help="validate pending or approved anonymous labels against one detection batch",
    )
    validate_tracker_review.add_argument("--frames", required=True)
    validate_tracker_review.add_argument("--review", required=True)
    validate_tracker_review.add_argument("--source-admission")
    validate_tracker_review.set_defaults(handler=command_validate_tracker_review)

    tracker_predict = subparsers.add_parser(
        "track-iou-predict",
        help=("export deterministic IoU tracking assignments for one fixed DetectionFrameBatch"),
    )
    tracker_predict.add_argument("--tracker-config", required=True)
    tracker_predict.add_argument("--frames", required=True)
    tracker_predict.add_argument("--output", required=True)
    tracker_predict.set_defaults(handler=command_track_iou_predict)

    motion_tracker_predict = subparsers.add_parser(
        "track-motion-predict",
        help=(
            "export deterministic constant-velocity, two-stage global "
            "association assignments for one fixed DetectionFrameBatch"
        ),
    )
    motion_tracker_predict.add_argument("--tracker-config", required=True)
    motion_tracker_predict.add_argument("--frames", required=True)
    motion_tracker_predict.add_argument("--output", required=True)
    motion_tracker_predict.set_defaults(handler=command_track_motion_predict)

    validate_tracker_predictions = subparsers.add_parser(
        "validate-tracker-predictions",
        help="validate tracker assignments against the exact source detection batch",
    )
    validate_tracker_predictions.add_argument("--frames", required=True)
    validate_tracker_predictions.add_argument("--predictions", required=True)
    validate_tracker_predictions.set_defaults(handler=command_validate_tracker_predictions)

    compare_tracker = subparsers.add_parser(
        "compare-tracker-predictions",
        help=(
            "compare anonymous association links from two predictions on the "
            "same batch without claiming accuracy"
        ),
    )
    compare_tracker.add_argument("--frames", required=True)
    compare_tracker.add_argument("--left", required=True)
    compare_tracker.add_argument("--right", required=True)
    compare_tracker.add_argument("--output", required=True)
    compare_tracker.set_defaults(handler=command_compare_tracker_predictions)

    validate_tracker_comparison = subparsers.add_parser(
        "validate-tracker-comparison",
        help="validate a path-free, no-ground-truth tracker disagreement report",
    )
    validate_tracker_comparison.add_argument("path")
    validate_tracker_comparison.set_defaults(handler=command_validate_tracker_comparison)

    prepare_disagreement_review = subparsers.add_parser(
        "prepare-tracker-disagreement-review",
        help=("prepare one pending human same/different question for every exclusive tracker link"),
    )
    prepare_disagreement_review.add_argument("--frames", required=True)
    prepare_disagreement_review.add_argument("--left", required=True)
    prepare_disagreement_review.add_argument("--right", required=True)
    prepare_disagreement_review.add_argument("--comparison", required=True)
    prepare_disagreement_review.add_argument("--output", required=True)
    prepare_disagreement_review.set_defaults(handler=command_prepare_tracker_disagreement_review)

    validate_disagreement_review = subparsers.add_parser(
        "validate-tracker-disagreement-review",
        help=("validate human same/different labels against the exact two prediction link sets"),
    )
    validate_disagreement_review.add_argument("--frames", required=True)
    validate_disagreement_review.add_argument("--left", required=True)
    validate_disagreement_review.add_argument("--right", required=True)
    validate_disagreement_review.add_argument("--comparison", required=True)
    validate_disagreement_review.add_argument("--review", required=True)
    validate_disagreement_review.set_defaults(handler=command_validate_tracker_disagreement_review)

    evaluate_disagreement = subparsers.add_parser(
        "evaluate-tracker-disagreement",
        help=(
            "compare relative errors only on human-reviewed exclusive links; "
            "do not claim overall accuracy"
        ),
    )
    evaluate_disagreement.add_argument("--frames", required=True)
    evaluate_disagreement.add_argument("--left", required=True)
    evaluate_disagreement.add_argument("--right", required=True)
    evaluate_disagreement.add_argument("--comparison", required=True)
    evaluate_disagreement.add_argument("--review", required=True)
    evaluate_disagreement.add_argument("--output", required=True)
    evaluate_disagreement.set_defaults(handler=command_evaluate_tracker_disagreement)

    validate_disagreement = subparsers.add_parser(
        "validate-tracker-disagreement",
        help="validate a path-free human-reviewed relative disagreement report",
    )
    validate_disagreement.add_argument("path")
    validate_disagreement.set_defaults(handler=command_validate_tracker_disagreement)

    tracker_evaluate = subparsers.add_parser(
        "evaluate-tracker-association",
        help=("score tracker association and fragmentation only on approved human-labelled boxes"),
    )
    tracker_evaluate.add_argument("--frames", required=True)
    tracker_evaluate.add_argument("--review", required=True)
    tracker_evaluate.add_argument("--predictions", required=True)
    tracker_evaluate.add_argument("--source-admission")
    tracker_evaluate.add_argument("--output", required=True)
    tracker_evaluate.set_defaults(handler=command_evaluate_tracker_association)

    validate_tracker_report = subparsers.add_parser(
        "validate-tracker-association",
        help="validate a path-free tracker association report",
    )
    validate_tracker_report.add_argument("path")
    validate_tracker_report.set_defaults(handler=command_validate_tracker_association)

    track_iou_pending = subparsers.add_parser(
        "track-iou-pending",
        help="replay unacknowledged tracked observations from the local outbox",
    )
    track_iou_pending.add_argument("--state", required=True)
    track_iou_pending.set_defaults(handler=command_track_iou_pending)

    track_iou_ack = subparsers.add_parser(
        "track-iou-ack",
        help="acknowledge tracked observations after downstream persistence succeeds",
    )
    track_iou_ack.add_argument("--state", required=True)
    track_iou_ack.add_argument("--observation-id", required=True, action="append")
    track_iou_ack.set_defaults(handler=command_track_iou_ack)

    track_iou_zone = subparsers.add_parser(
        "track-iou-zone-run",
        help="persist pending tracked observations into temporal zones before acknowledging them",
    )
    track_iou_zone.add_argument("--tracker-state", required=True)
    track_iou_zone.add_argument("--zone-config", required=True)
    track_iou_zone.add_argument("--zone-state", required=True)
    track_iou_zone.set_defaults(handler=command_track_iou_zone_run)

    track_motion_pending = subparsers.add_parser(
        "track-motion-pending",
        help=("replay unacknowledged motion-tracker observations from the local outbox"),
    )
    track_motion_pending.add_argument("--state", required=True)
    track_motion_pending.set_defaults(handler=command_track_motion_pending)

    track_motion_ack = subparsers.add_parser(
        "track-motion-ack",
        help=("acknowledge motion-tracker observations after downstream persistence succeeds"),
    )
    track_motion_ack.add_argument("--state", required=True)
    track_motion_ack.add_argument(
        "--observation-id",
        required=True,
        action="append",
    )
    track_motion_ack.set_defaults(handler=command_track_motion_ack)

    track_motion_zone = subparsers.add_parser(
        "track-motion-zone-run",
        help=(
            "persist pending motion-tracker observations into temporal zones "
            "before acknowledging them"
        ),
    )
    track_motion_zone.add_argument("--tracker-state", required=True)
    track_motion_zone.add_argument("--zone-config", required=True)
    track_motion_zone.add_argument("--zone-state", required=True)
    track_motion_zone.set_defaults(handler=command_track_motion_zone_run)

    temporal_run = subparsers.add_parser(
        "temporal-zone-run",
        help="convert tracked observations into persistent zone signals without a model",
    )
    temporal_run.add_argument("--zone-config", required=True)
    temporal_run.add_argument("--observations", required=True)
    temporal_run.add_argument("--state", required=True)
    temporal_run.set_defaults(handler=command_temporal_zone_run)

    temporal_pending = subparsers.add_parser(
        "temporal-zone-pending",
        help="replay unacknowledged temporal zone signals from the local outbox",
    )
    temporal_pending.add_argument("--state", required=True)
    temporal_pending.set_defaults(handler=command_temporal_zone_pending)

    temporal_ack = subparsers.add_parser(
        "temporal-zone-ack",
        help="acknowledge temporal signals only after downstream persistence succeeds",
    )
    temporal_ack.add_argument("--state", required=True)
    temporal_ack.add_argument("--signal-id", required=True, action="append")
    temporal_ack.set_defaults(handler=command_temporal_zone_ack)

    analysis_run = subparsers.add_parser(
        "analysis-run-once",
        help="claim and process at most one persisted analysis job",
    )
    analysis_run.add_argument("--config", default="config/default.toml")
    analysis_run.add_argument("--database")
    analysis_run.add_argument(
        "--defer-reports", action="store_true", help="enqueue daily refreshes for a report worker"
    )
    analysis_run.set_defaults(handler=command_analysis_run_once)

    analysis_worker = subparsers.add_parser(
        "analysis-worker",
        help="consume persisted analysis jobs sequentially within explicit stopping bounds",
    )
    analysis_worker.add_argument("--config", default="config/default.toml")
    analysis_worker.add_argument("--database")
    analysis_worker.add_argument("--max-jobs", type=int)
    analysis_worker.add_argument("--idle-exit-seconds", type=float)
    analysis_worker.add_argument("--max-runtime-seconds", type=float)
    analysis_worker.add_argument("--poll-seconds", type=float)
    analysis_worker.add_argument(
        "--defer-reports", action="store_true", help="enqueue daily refreshes for a report worker"
    )
    analysis_worker.set_defaults(handler=command_analysis_worker)

    analysis_status = subparsers.add_parser(
        "analysis-queue-status",
        help="show bounded analysis queue counts without exposing event content",
    )
    analysis_status.add_argument("--config", default="config/default.toml")
    analysis_status.add_argument("--database")
    analysis_status.set_defaults(handler=command_analysis_queue_status)

    analysis_gpu_status = subparsers.add_parser(
        "analysis-gpu-status",
        help="read NVIDIA capacity and show whether configured analysis thresholds allow a job",
    )
    analysis_gpu_status.add_argument("--config", default="config/default.toml")
    analysis_gpu_status.set_defaults(handler=command_analysis_gpu_status)

    analysis_retry = subparsers.add_parser(
        "analysis-retry",
        help="explicitly reset one failed analysis job for another bounded attempt cycle",
    )
    analysis_retry.add_argument("--config", default="config/default.toml")
    analysis_retry.add_argument("--database")
    analysis_retry.add_argument("--event-id", required=True)
    analysis_retry.set_defaults(handler=command_analysis_retry)

    refresh_request = subparsers.add_parser(
        "report-refresh-request",
        help="durably request a coalesced local daily report without reading media",
    )
    refresh_request.add_argument("--config", default="config/default.toml")
    refresh_request.add_argument("--database")
    refresh_request.add_argument("--date", required=True)
    refresh_request.add_argument("--output-root", required=True)
    refresh_request.add_argument("--request-id", required=True)
    refresh_request.add_argument("--debounce-seconds", type=int, default=30)
    refresh_request.set_defaults(handler=command_report_refresh_request)

    for name, help_text, handler in (
        (
            "report-refresh-run-once",
            "render at most one queued daily report without models or notifications",
            command_report_refresh_run_once,
        ),
        (
            "report-refresh-worker",
            "refresh daily reports independently within explicit stopping bounds",
            command_report_refresh_worker,
        ),
    ):
        refresh = subparsers.add_parser(name, help=help_text)
        refresh.add_argument("--config", default="config/default.toml")
        refresh.add_argument("--database")
        refresh.add_argument("--lease-seconds", type=int, default=300)
        refresh.add_argument("--max-attempts", type=int, default=3)
        refresh.add_argument("--retry-delay-seconds", type=int, default=60)
        if name == "report-refresh-worker":
            refresh.add_argument("--max-jobs", type=int, default=32)
            refresh.add_argument("--idle-exit-seconds", type=float, default=60)
            refresh.add_argument("--max-runtime-seconds", type=float, default=3600)
            refresh.add_argument("--poll-seconds", type=float, default=5)
        refresh.set_defaults(handler=handler)

    refresh_status = subparsers.add_parser(
        "report-refresh-status",
        help="read report refresh counts without creating files or disclosing private paths",
    )
    refresh_status.add_argument("--config", default="config/default.toml")
    refresh_status.add_argument("--database")
    refresh_status.set_defaults(handler=command_report_refresh_status)

    refresh_retry = subparsers.add_parser(
        "report-refresh-retry",
        help="explicitly retry one failed daily refresh without repeating recognition",
    )
    refresh_retry.add_argument("--config", default="config/default.toml")
    refresh_retry.add_argument("--database")
    refresh_retry.add_argument("--job-id", required=True)
    refresh_retry.set_defaults(handler=command_report_refresh_retry)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
