"""Identity enrollment, listing, and deletion commands."""

from __future__ import annotations

import argparse

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

from ..config import load_settings
from ..identity_runtime import (
    delete_local_identity_profile,
    enroll_local_identity,
    list_local_identity_profiles,
)
from ..identity_service import (
    authorize_enrollment,
    authorize_profile_deletion,
)


def command_identity_consent_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _json_print(
        {
            "identity": settings.identity.to_public_dict(),
            "operation_performed": False,
            "enrollment_command_available": True,
            "media_read": False,
            "event_history_read": False,
            "storage_mutated": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_identity_enrollment_preflight(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    receipt = authorize_enrollment(
        settings.identity,
        one_shot_confirmation=args.confirm_biometric_enrollment,
    )
    payload = receipt.to_public_dict()
    payload.update(
        {
            "persistent_identity_enabled": settings.identity.enabled,
            "persistent_biometric_enrollment_consent": (
                settings.identity.biometric_enrollment_consent
            ),
            "preflight_only": True,
        }
    )
    _json_print(payload)
    return 0


def command_identity_deletion_preflight(args: argparse.Namespace) -> int:
    receipt = authorize_profile_deletion(
        profile_id=args.profile_id,
        one_shot_confirmation=args.confirm_identity_profile_deletion,
    )
    payload = receipt.to_public_dict()
    payload["preflight_only"] = True
    _json_print(payload)
    return 0


def command_identity_enroll(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    # Keep this command-level gate ahead of every model, manifest, image, staging,
    # and identity-store operation. The runtime repeats it before doing any I/O.
    authorize_enrollment(
        settings.identity,
        one_shot_confirmation=args.confirm_biometric_enrollment,
    )
    payload = enroll_local_identity(
        settings=settings.identity,
        one_shot_confirmation=args.confirm_biometric_enrollment,
        manifest_path=args.manifest,
        manifest_root=args.manifest_root,
        store_root=args.store_root,
        staging_parent=args.staging_parent,
        runner_executable=args.runner_executable,
        detector_model_path=args.detector_model,
        detector_manifest_path=args.detector_manifest,
        embedding_model_path=args.embedding_model,
        embedding_manifest_path=args.embedding_manifest,
        timeout_seconds=args.timeout_seconds,
        detector_score_threshold=args.detector_score_threshold,
        detector_nms_threshold=args.detector_nms_threshold,
        detector_top_k=args.detector_top_k,
        min_face_size_pixels=args.min_face_size_pixels,
        min_eye_distance_pixels=args.min_eye_distance_pixels,
    )
    _json_print(payload)
    return 0


def command_identity_profiles(args: argparse.Namespace) -> int:
    _json_print(list_local_identity_profiles(store_root=args.store_root))
    return 0


def command_identity_delete_profile(args: argparse.Namespace) -> int:
    # Deletion deliberately remains available after identity enrollment is
    # disabled, but it always requires this exact-profile one-shot gate.
    authorize_profile_deletion(
        profile_id=args.profile_id,
        one_shot_confirmation=args.confirm_identity_profile_deletion,
    )
    payload = delete_local_identity_profile(
        profile_id=args.profile_id,
        one_shot_confirmation=args.confirm_identity_profile_deletion,
        store_root=args.store_root,
    )
    _json_print(payload)
    return 0
