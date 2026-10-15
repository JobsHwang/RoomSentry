from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .adapters.identity import (
    CommandFaceEmbeddingAdapter,
    FaceEnrollmentCommandConfig,
)
from .identity_enrollment import (
    IdentityEnrollmentResult,
    enroll_identity_from_manifest,
)
from .identity_model import FaceModelManifest
from .identity_protocol import FaceModelReference
from .identity_service import (
    IdentitySettings,
    authorize_enrollment,
    authorize_profile_deletion,
)
from .identity_store import (
    IdentityDeletionSummary,
    IdentityProfileSummary,
    IdentityTemplateStore,
)

FIXED_FACE_RUNNER_ARGS = ("-m", "baby_monitor.local_runners.opencv_face")


_SYSTEMD_ENV_EXECUTABLE = Path("/usr/bin/env")
_SYSTEMD_RUN_EXECUTABLE = Path("/usr/bin/systemd-run")


def _require_fixed_system_executable(path: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError:
        raise IdentityRuntimeError("enrollment_failed") from None
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        resolved != path
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_nlink != 1
        or mode & 0o022
        or mode & 0o111 == 0
        or not os.access(path, os.X_OK)
    ):
        raise IdentityRuntimeError("enrollment_failed")
    return str(path)


def _require_runner_executable(value: str) -> str:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise IdentityRuntimeError("enrollment_failed")
    runner = Path(value)
    if not runner.is_absolute() or ".." in runner.parts or str(runner) != value:
        raise IdentityRuntimeError("enrollment_failed")
    uid = os.geteuid()
    try:
        current = Path(runner.anchor)
        for part in runner.parts[1:-1]:
            current /= part
            metadata = current.lstat()
            if metadata.st_uid not in {0, uid}:
                raise OSError
            if stat.S_ISLNK(metadata.st_mode):
                target_metadata = current.resolve(strict=True).stat()
                if (
                    not stat.S_ISDIR(target_metadata.st_mode)
                    or target_metadata.st_uid not in {0, uid}
                    or stat.S_IMODE(target_metadata.st_mode) & 0o022
                ):
                    raise OSError
            elif not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o022:
                raise OSError
        lexical_metadata = runner.lstat()
        if lexical_metadata.st_uid not in {0, uid}:
            raise OSError
        if not (stat.S_ISREG(lexical_metadata.st_mode) or stat.S_ISLNK(lexical_metadata.st_mode)):
            raise OSError
        if (
            stat.S_ISREG(lexical_metadata.st_mode)
            and stat.S_IMODE(lexical_metadata.st_mode) & 0o022
        ):
            raise OSError
        resolved = runner.resolve(strict=True)
        resolved_metadata = resolved.lstat()
    except OSError:
        raise IdentityRuntimeError("enrollment_failed") from None
    resolved_mode = stat.S_IMODE(resolved_metadata.st_mode)
    if (
        not stat.S_ISREG(resolved_metadata.st_mode)
        or stat.S_ISLNK(resolved_metadata.st_mode)
        or resolved_metadata.st_uid != 0
        or resolved_mode & 0o022
        or resolved_mode & 0o111 == 0
        or not os.access(resolved, os.X_OK)
    ):
        raise IdentityRuntimeError("enrollment_failed")
    # A same-UID venv symlink can still be swapped after admission. That local
    # account-compromise TOCTOU is outside this process boundary.
    return value


def _require_systemd_user_bus() -> tuple[Path, Path]:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise IdentityRuntimeError("enrollment_failed")
    uid = os.geteuid()
    runtime_root = Path(f"/run/user/{uid}")
    bus_path = runtime_root / "bus"
    try:
        resolved = runtime_root.resolve(strict=True)
        runtime_metadata = runtime_root.lstat()
        bus_metadata = bus_path.lstat()
    except OSError:
        raise IdentityRuntimeError("enrollment_failed") from None
    if (
        resolved != runtime_root
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or stat.S_ISLNK(runtime_metadata.st_mode)
        or runtime_metadata.st_uid != uid
        or stat.S_IMODE(runtime_metadata.st_mode) != 0o700
        or not stat.S_ISSOCK(bus_metadata.st_mode)
        or stat.S_ISLNK(bus_metadata.st_mode)
        or bus_metadata.st_uid != uid
        or bus_metadata.st_nlink != 1
    ):
        raise IdentityRuntimeError("enrollment_failed")
    return runtime_root, bus_path


def _systemd_isolated_runner_command(
    runner_executable: str,
    *,
    _executable_admission: Callable[[Path], str] | None = None,
    _bus_admission: Callable[[], tuple[Path, Path]] | None = None,
    _runner_admission: Callable[[str], str] | None = None,
) -> tuple[str, ...]:
    runner = Path(runner_executable)
    if not runner.is_absolute() or ".." in runner.parts or str(runner) != runner_executable:
        raise IdentityRuntimeError("enrollment_failed")
    admit_runner = _runner_admission or _require_runner_executable
    admitted_runner = admit_runner(runner_executable)
    admit_executable = _executable_admission or _require_fixed_system_executable
    admit_bus = _bus_admission or _require_systemd_user_bus
    env_executable = admit_executable(_SYSTEMD_ENV_EXECUTABLE)
    systemd_run = admit_executable(_SYSTEMD_RUN_EXECUTABLE)
    runtime_root, bus_path = admit_bus()
    return (
        env_executable,
        f"XDG_RUNTIME_DIR={runtime_root}",
        f"DBUS_SESSION_BUS_ADDRESS=unix:path={bus_path}",
        systemd_run,
        "--user",
        "--pipe",
        "--wait",
        "--collect",
        "--quiet",
        "--property=IPAddressDeny=any",
        "--property=RestrictAddressFamilies=AF_UNIX",
        "--property=MemoryMax=1610612736",
        "--property=NoNewPrivileges=yes",
        "--property=PrivateDevices=yes",
        admitted_runner,
        *FIXED_FACE_RUNNER_ARGS,
    )


class IdentityRuntimeError(RuntimeError):
    """A fixed, path-free failure from the local identity management boundary."""

    _MESSAGES = {
        "enrollment_failed": "local identity enrollment failed safely",
        "profiles_failed": "local identity profiles could not be read safely",
        "deletion_failed": "local identity profile deletion failed safely",
    }

    def __init__(self, code: str) -> None:
        if code not in self._MESSAGES:
            code = "enrollment_failed"
        self.code = code
        super().__init__(self._MESSAGES[code])


def _reviewed_model_reference(
    *,
    role: str,
    model_path: str | Path,
    manifest_path: str | Path,
) -> FaceModelReference:
    """Admit one local model only after its built-in review and file hash pass."""

    model = Path(model_path).expanduser()
    manifest_file = Path(manifest_path).expanduser()
    if not model.is_absolute() or not manifest_file.is_absolute():
        raise IdentityRuntimeError("enrollment_failed")
    manifest = FaceModelManifest.read(manifest_file)
    if manifest.role != role:
        raise IdentityRuntimeError("enrollment_failed")
    manifest.require_reviewed()
    verification = manifest.verify_model(model)
    return FaceModelReference(
        role=role,
        model_path=str(model),
        manifest_path=str(manifest_file),
        model_id=manifest.model_id,
        upstream_revision=manifest.upstream_revision,
        model_sha256=verification.sha256,
    )


def _build_command_enrollment_extractor(
    *,
    runner_executable: str,
    staging_parent: str | Path,
    detector_model_path: str | Path,
    detector_manifest_path: str | Path,
    embedding_model_path: str | Path,
    embedding_manifest_path: str | Path,
    timeout_seconds: float,
    detector_score_threshold: float,
    detector_nms_threshold: float,
    detector_top_k: int,
    min_face_size_pixels: int,
    min_eye_distance_pixels: float,
) -> CommandFaceEmbeddingAdapter:
    isolated_command = _systemd_isolated_runner_command(runner_executable)
    detector = _reviewed_model_reference(
        role="detector",
        model_path=detector_model_path,
        manifest_path=detector_manifest_path,
    )
    embedding = _reviewed_model_reference(
        role="embedding",
        model_path=embedding_model_path,
        manifest_path=embedding_manifest_path,
    )
    enrollment = FaceEnrollmentCommandConfig(
        detector_model=detector,
        embedding_model=embedding,
        detector_score_threshold=detector_score_threshold,
        detector_nms_threshold=detector_nms_threshold,
        detector_top_k=detector_top_k,
        min_face_size_pixels=min_face_size_pixels,
        min_eye_distance_pixels=min_eye_distance_pixels,
    )
    return CommandFaceEmbeddingAdapter(
        command=isolated_command,
        staging_parent=staging_parent,
        timeout_seconds=timeout_seconds,
        enrollment=enrollment,
    )


def _lazy_identity_store_factory(
    root: str | Path,
) -> Callable[[], IdentityTemplateStore]:
    """Return a closure without touching the private store path."""

    def open_store() -> IdentityTemplateStore:
        return IdentityTemplateStore(root)

    return open_store


def _existing_store_root(root: str | Path) -> Path:
    """Admit an already initialized private store without creating any files."""

    try:
        if os.name != "posix" or not hasattr(os, "geteuid"):
            raise OSError
        path = Path(root)
        if not path.is_absolute():
            raise OSError
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except (OSError, TypeError, ValueError):
        raise IdentityRuntimeError("profiles_failed") from None
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved.parts[:3] == ("/", "mnt", "c")
    ):
        raise IdentityRuntimeError("profiles_failed")
    for filename in ("identity.key", "identity.sqlite3"):
        try:
            private_path = path / filename
            private_metadata = private_path.lstat()
        except OSError:
            raise IdentityRuntimeError("profiles_failed") from None
        if (
            not stat.S_ISREG(private_metadata.st_mode)
            or stat.S_ISLNK(private_metadata.st_mode)
            or private_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(private_metadata.st_mode) != 0o600
            or private_metadata.st_nlink != 1
        ):
            raise IdentityRuntimeError("profiles_failed")
    return path


def enroll_local_identity(
    *,
    settings: IdentitySettings,
    one_shot_confirmation: bool,
    manifest_path: str | Path,
    manifest_root: str | Path,
    store_root: str | Path,
    staging_parent: str | Path,
    runner_executable: str,
    detector_model_path: str | Path,
    detector_manifest_path: str | Path,
    embedding_model_path: str | Path,
    embedding_manifest_path: str | Path,
    timeout_seconds: float = 120.0,
    detector_score_threshold: float = 0.8,
    detector_nms_threshold: float = 0.3,
    detector_top_k: int = 100,
    min_face_size_pixels: int = 40,
    min_eye_distance_pixels: float = 12.0,
    _extractor_factory: Callable[..., Any] | None = None,
    _store_factory_builder: Callable[[str | Path], Callable[[], Any]] | None = None,
    _coordinator: Callable[..., IdentityEnrollmentResult] | None = None,
) -> dict[str, object]:
    """Run one explicitly authorized local enrollment and disclose no hashes.

    The authorization call intentionally precedes model admission, manifest/image
    reads, adapter construction, and private-store creation.  The coordinator
    repeats the same immutable authorization check immediately before reading the
    private enrollment manifest.
    """

    authorize_enrollment(
        settings,
        one_shot_confirmation=one_shot_confirmation,
    )
    extractor_factory = _extractor_factory or _build_command_enrollment_extractor
    store_factory_builder = _store_factory_builder or _lazy_identity_store_factory
    coordinator = _coordinator or enroll_identity_from_manifest
    try:
        extractor = extractor_factory(
            runner_executable=runner_executable,
            staging_parent=staging_parent,
            detector_model_path=detector_model_path,
            detector_manifest_path=detector_manifest_path,
            embedding_model_path=embedding_model_path,
            embedding_manifest_path=embedding_manifest_path,
            timeout_seconds=timeout_seconds,
            detector_score_threshold=detector_score_threshold,
            detector_nms_threshold=detector_nms_threshold,
            detector_top_k=detector_top_k,
            min_face_size_pixels=min_face_size_pixels,
            min_eye_distance_pixels=min_eye_distance_pixels,
        )
        store_factory = store_factory_builder(store_root)
        result = coordinator(
            settings=settings,
            one_shot_confirmation=one_shot_confirmation,
            manifest_path=manifest_path,
            manifest_root=manifest_root,
            extractor=extractor,
            store_factory=store_factory,
        )
        if not isinstance(result, IdentityEnrollmentResult):
            raise TypeError
    except IdentityRuntimeError:
        raise
    except Exception:
        raise IdentityRuntimeError("enrollment_failed") from None
    return {
        "status": "enrolled",
        "profile_id": result.profile_id,
        "declared_kind": result.declared_kind.value,
        "template_count": len(result.template_ids),
        "source_image_count": result.source_image_count,
        "source_images_persisted": False,
        "embeddings_disclosed": False,
        "model_hashes_disclosed": False,
        "runner_network_status": "os_denied",
        "runner_os_network_isolation_enforced": True,
        "runner_network_isolation_boundary": "systemd_transient_user_unit",
        "coordinator_network_calls": False,
        "identity_inference": False,
        "paths_disclosed": False,
        "atomic_store_write": True,
    }


def list_local_identity_profiles(
    *,
    store_root: str | Path,
    _store_factory: Callable[[str | Path], Any] | None = None,
) -> dict[str, object]:
    """List only opaque profile metadata; never load or export embeddings."""

    try:
        root = _existing_store_root(store_root)
        if _store_factory is None:
            profiles = IdentityTemplateStore.list_profiles_read_only(root)
        else:
            store = _store_factory(root)
            profiles = store.list_profiles()
        if not isinstance(profiles, tuple) or any(
            not isinstance(profile, IdentityProfileSummary) for profile in profiles
        ):
            raise TypeError
    except Exception:
        raise IdentityRuntimeError("profiles_failed") from None
    return {
        "status": "ok",
        "profile_count": len(profiles),
        "template_count": sum(profile.template_count for profile in profiles),
        "profiles": [profile.to_public_dict() for profile in profiles],
        "embeddings_loaded": False,
        "embeddings_disclosed": False,
        "model_hashes_disclosed": False,
        "raw_images_stored": False,
        "paths_disclosed": False,
    }


def delete_local_identity_profile(
    *,
    profile_id: str,
    one_shot_confirmation: bool,
    store_root: str | Path,
    _store_factory: Callable[[str | Path], Any] | None = None,
) -> dict[str, object]:
    """Delete exactly one opaque profile after the independent one-shot gate."""

    receipt = authorize_profile_deletion(
        profile_id=profile_id,
        one_shot_confirmation=one_shot_confirmation,
    )
    try:
        root = _existing_store_root(store_root)
        store = (
            _store_factory(root)
            if _store_factory is not None
            else IdentityTemplateStore.open_existing(root)
        )
        result = store.delete_profile(profile_id)
        if (
            not isinstance(result, IdentityDeletionSummary)
            or result.deleted_profiles != 1
            or receipt.target_profile_id != profile_id
        ):
            raise TypeError
    except Exception:
        raise IdentityRuntimeError("deletion_failed") from None
    return {
        "status": "deleted",
        "profile_id": profile_id,
        **result.to_public_dict(),
        "embeddings_disclosed": False,
        "model_hashes_disclosed": False,
        "raw_images_deleted": 0,
        "paths_disclosed": False,
    }
