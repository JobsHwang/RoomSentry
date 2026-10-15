from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class IdentityAuthorizationError(ValueError):
    """Raised when an identity operation has not passed every required gate."""


class IdentityOperation(StrEnum):
    ENROLLMENT = "enrollment"
    PROFILE_DELETION = "profile_deletion"


@dataclass(frozen=True, slots=True)
class IdentitySettings:
    """Fail-closed identity feature and consent settings.

    The persistent enrollment consent is intentionally separate from the
    feature switch.  Event matching and external identity disclosure are
    independent opt-ins and remain disabled unless the identity feature is
    also enabled.
    """

    enabled: bool = False
    biometric_enrollment_consent: bool = False
    event_matching_enabled: bool = False
    external_identity_disclosure_enabled: bool = False

    @property
    def enrollment_persistent_gate_satisfied(self) -> bool:
        return self.enabled and self.biometric_enrollment_consent

    @property
    def event_matching_allowed(self) -> bool:
        return self.enabled and self.event_matching_enabled

    @property
    def external_identity_disclosure_allowed(self) -> bool:
        return self.event_matching_allowed and self.external_identity_disclosure_enabled

    def to_public_dict(self) -> dict[str, bool]:
        return {
            "enabled": self.enabled,
            "biometric_enrollment_consent": self.biometric_enrollment_consent,
            "enrollment_persistent_gate_satisfied": (self.enrollment_persistent_gate_satisfied),
            "event_matching_enabled": self.event_matching_enabled,
            "event_matching_allowed": self.event_matching_allowed,
            "external_identity_disclosure_enabled": (self.external_identity_disclosure_enabled),
            "external_identity_disclosure_allowed": (self.external_identity_disclosure_allowed),
        }


@dataclass(frozen=True, slots=True)
class IdentityAuthorizationReceipt:
    """Path-free receipt proving only that an operation's gates passed."""

    operation: IdentityOperation
    target_profile_id: str | None = None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "authorized": True,
            "target_profile_id": self.target_profile_id,
            "one_shot_confirmation_present": True,
            "media_read": False,
            "event_history_read": False,
            "storage_mutated": False,
            "paths_disclosed": False,
        }


_PROFILE_ID_PATTERN = re.compile(r"profile_[0-9a-f]{32}")


def authorize_enrollment(
    settings: IdentitySettings,
    *,
    one_shot_confirmation: bool,
) -> IdentityAuthorizationReceipt:
    """Authorize, but do not perform, one local biometric enrollment.

    Every command that later performs enrollment must call this function
    before it opens an image, frame, event, camera, model, or identity store.
    """

    if not settings.enabled:
        raise IdentityAuthorizationError("identity capability is disabled")
    if not settings.biometric_enrollment_consent:
        raise IdentityAuthorizationError("persistent biometric enrollment consent is disabled")
    if not one_shot_confirmation:
        raise IdentityAuthorizationError("one-shot biometric enrollment confirmation is required")
    return IdentityAuthorizationReceipt(IdentityOperation.ENROLLMENT)


def authorize_profile_deletion(
    *,
    profile_id: str,
    one_shot_confirmation: bool,
) -> IdentityAuthorizationReceipt:
    """Authorize, but do not perform, deletion of one opaque identity profile.

    Deletion deliberately does not depend on identity being enabled or on
    enrollment consent, so a user can remove biometric data after opting out.
    """

    if _PROFILE_ID_PATTERN.fullmatch(profile_id) is None:
        raise IdentityAuthorizationError("identity profile id must be an opaque profile_ id")
    if not one_shot_confirmation:
        raise IdentityAuthorizationError(
            "one-shot identity profile deletion confirmation is required"
        )
    return IdentityAuthorizationReceipt(
        IdentityOperation.PROFILE_DELETION,
        target_profile_id=profile_id,
    )
