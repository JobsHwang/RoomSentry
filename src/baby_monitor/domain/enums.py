"""Stable enumerations shared across the domain contracts."""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    SLEEPING = "sleeping"
    AWAKE = "awake"
    CRYING = "crying"
    MOVEMENT = "movement"
    ADULT_INTERACTION = "adult_interaction"
    POSSIBLE_RISK = "possible_risk"
    UNKNOWN = "unknown"


class RiskLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {
            RiskLevel.NONE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
        }[self]


class ReviewTier(StrEnum):
    NONE = "none"
    LOCAL_PRIMARY = "local_primary"
    LOCAL_SECONDARY = "local_secondary"
    REMOTE = "remote"

    @property
    def rank(self) -> int:
        return {
            ReviewTier.NONE: 0,
            ReviewTier.LOCAL_PRIMARY: 1,
            ReviewTier.LOCAL_SECONDARY: 2,
            ReviewTier.REMOTE: 3,
        }[self]


class ReviewStatus(StrEnum):
    FINAL = "final"
    PENDING_REVIEW = "pending_review"


class FinalSource(StrEnum):
    LOCAL_PRIMARY = "local_primary"
    LOCAL_SECONDARY = "local_secondary"
    REMOTE = "remote"


class SubjectKind(StrEnum):
    BABY = "baby"
    ADULT = "adult"
    PERSON = "person"
    PET = "pet"
    UNKNOWN = "unknown"


class SignalType(StrEnum):
    MOTION = "motion"
    PERSON = "person"
    FACE = "face"
    SOUND_PEAK = "sound_peak"
    HARDWARE_CRY = "hardware_cry"
    ZONE_ENTER = "zone_enter"
    ZONE_DWELL = "zone_dwell"
    ZONE_EXIT = "zone_exit"
    FALL_HINT = "fall_hint"
    MANUAL_TRIGGER = "manual_trigger"
