"""Frame-local role references for diagnostics, never a live identity grant.

A user may confirm that one visible head belongs to an adult or baby without
annotating a body box. The confirmation stays bound to that frame and anonymous
reference. It does not prove body localization, whole-frame occupancy, persistent
identity, sleeping, or current presence. Geometry reviews remain separate.

Pure standard-library validation and comparison; no files, media or model calls.
Declared human provenance is caller input, not authenticated by this parser.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

_MAX_REFERENCE_BYTES = 8192
_CAMERA_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SUBJECT_REFERENCE = re.compile(r"subject_[0-9]{1,6}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLES = frozenset({"baby", "adult", "unknown"})
_PROVENANCE = frozenset({"human_user_confirmation", "synthetic"})
_ASSOCIATIONS = frozenset({"unresolved", "assistant_weak", "synthetic"})


class FrameRoleReferenceError(ValueError):
    """A role reference or diagnostic binding is invalid."""


def _string(value: object, pattern: re.Pattern[str], field: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise FrameRoleReferenceError(f"invalid frame role {field}")
    return value


def _choice(value: object, choices: frozenset[str], field: str) -> str:
    if type(value) is not str or value not in choices:
        raise FrameRoleReferenceError(f"invalid frame role {field}")
    return value


def _utc_time(value: object) -> str:
    if type(value) is not str or not 1 <= len(value) <= 64:
        raise FrameRoleReferenceError("invalid frame role timestamp")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError("naive timestamp")
        return parsed.astimezone(UTC).isoformat()
    except (ValueError, OverflowError):
        raise FrameRoleReferenceError("invalid frame role timestamp") from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FrameRoleReferenceError("duplicate frame role JSON field")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise FrameRoleReferenceError("non-standard frame role JSON number")


@dataclass(frozen=True, slots=True)
class FrameRoleReference:
    camera_id: str
    frame_sha256: str
    frame_received_at: str
    reference_id: str
    role: str
    provenance: str
    version: int = 1
    scope: str = "visible_head_reference_only"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise FrameRoleReferenceError("unsupported frame role reference version")
        if type(self.scope) is not str or self.scope != "visible_head_reference_only":
            raise FrameRoleReferenceError("invalid frame role reference scope")
        _string(self.camera_id, _CAMERA_ID, "camera ID")
        _string(self.frame_sha256, _SHA256, "SHA-256")
        _string(self.reference_id, _SUBJECT_REFERENCE, "anonymous reference ID")
        _choice(self.role, _ROLES, "role")
        _choice(self.provenance, _PROVENANCE, "provenance")
        object.__setattr__(self, "frame_received_at", _utc_time(self.frame_received_at))

    @classmethod
    def from_dict(cls, value: object) -> FrameRoleReference:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise FrameRoleReferenceError("invalid frame role reference fields")
        return cls(**value)

    @classmethod
    def from_json(cls, value: object) -> FrameRoleReference:
        if type(value) is not str:
            raise FrameRoleReferenceError("frame role reference must be bounded JSON text")
        try:
            if not 0 < len(value.encode("utf-8")) <= _MAX_REFERENCE_BYTES:
                raise FrameRoleReferenceError("frame role reference JSON size limit")
            decoded = json.loads(
                value, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
            raise FrameRoleReferenceError("invalid frame role reference JSON") from None
        return cls.from_dict(decoded)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def diagnose_frame_role_claim(
    reference: FrameRoleReference,
    *,
    camera_id: str,
    frame_sha256: str,
    reference_id: str | None,
    claimed_role: str,
    association_provenance: str = "unresolved",
) -> dict[str, object]:
    """Compare an explicitly mapped claim with a caller-declared role reference.

    Wrong cameras, frames and anonymous IDs fail closed. Unresolved subject
    association is unknown, even if there is only one model box. ``person`` or
    ``unknown`` is abstention, not a wrong baby/adult classification. A declared
    adult reference never proves that another baby is absent or covered safely.

    This helper deliberately has no authenticated human-association path. Its
    comparisons remain conditional diagnostics, not formal accuracy or an input
    to fact_policy, tracking, room state, risk or notifications.
    """
    if type(reference) is not FrameRoleReference:
        raise FrameRoleReferenceError("frame role reference required")
    _string(camera_id, _CAMERA_ID, "camera ID")
    _string(frame_sha256, _SHA256, "SHA-256")
    _choice(claimed_role, _ROLES | {"person"}, "claimed role")
    _choice(association_provenance, _ASSOCIATIONS, "association provenance")
    if camera_id != reference.camera_id or frame_sha256 != reference.frame_sha256:
        raise FrameRoleReferenceError("frame role reference camera/frame binding mismatch")
    if reference_id is not None:
        _string(reference_id, _SUBJECT_REFERENCE, "anonymous reference ID")
        if reference_id != reference.reference_id:
            raise FrameRoleReferenceError("frame role reference subject binding mismatch")

    if reference_id is None or association_provenance == "unresolved":
        comparison = "unresolved_subject_association"
    elif reference.role == "unknown":
        comparison = "unknown_reference_role"
    elif claimed_role in ("person", "unknown"):
        comparison = "age_role_abstention"
    elif claimed_role == reference.role:
        comparison = "consistent_with_reference"
    else:
        comparison = "contradicts_reference"

    return {
        "schema_version": "frame-role-diagnostic-v1",
        "camera_id": camera_id,
        "frame_sha256": frame_sha256,
        "frame_received_at": reference.frame_received_at,
        "reference_id": reference_id,
        "reference_role": reference.role,
        "reference_provenance": reference.provenance,
        "claimed_role": claimed_role,
        "association_provenance": association_provenance,
        "comparison": comparison,
        "offline_diagnostic_only": True,
        "human_role_confirmation_declared": reference.provenance == "human_user_confirmation",
        "subject_association_human_verified": False,
        "formal_accuracy_authorized": False,
        "deployment_approved": False,
        "persistent_identity_authorized": False,
        "whole_frame_occupancy_established": False,
        "baby_location": "unknown",
        "baby_state": "unknown",
        "live_state_established": False,
    }
