from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_SOURCE_ADMISSION_BYTES = 1024 * 1024
MAX_EXTERNAL_ASSET_BYTES = 256 * 1024 * 1024 * 1024
LARGE_MODEL_WEIGHT_BYTES = 1024 * 1024 * 1024

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SAFE_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}")
_SAFE_SPDX = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,79}")
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_ASSET_KINDS = {
    "audio_asset",
    "code_archive",
    "dataset_archive",
    "media_asset",
    "model_weights",
    "vendor_event_export",
}
_FACT_STATES = {"yes", "no", "not_applicable", "unknown"}
_LICENSE_USE = {"allowed", "restricted", "prohibited", "unknown"}
_LABEL_ORIGINS = {
    "contributor_self_report",
    "human_expert",
    "human_nonexpert",
    "model_generated",
    "none",
    "publisher_description",
    "synthetic_program",
    "unknown",
    "vendor_rule",
}
_GROUND_TRUTH_ORIGINS = {
    "human_expert",
    "human_nonexpert",
    "synthetic_program",
}
_PURPOSES = {
    "adapter_development",
    "dynamic_failure_baseline",
    "model_evaluation",
    "public_regression",
    "research_only",
    "vendor_signal_correlation",
}
_ADMISSION_STATUSES = {
    "approved_for_download",
    "blocked",
    "verified_local",
}
_TIERS = {"A", "B", "C"}


class SourceAdmissionError(ValueError):
    """An external-source admission record or local asset is unsafe."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SourceAdmissionError("source admission JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                SourceAdmissionError("source admission JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise SourceAdmissionError("source admission JSON is invalid") from exc


def _exact_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SourceAdmissionError(f"source admission {context} fields do not match v1")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SourceAdmissionError(f"source admission {field} is invalid")
    return value


def _optional_revision(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SAFE_REVISION.fullmatch(value) is None:
        raise SourceAdmissionError("source admission source_revision is invalid")
    return value


def _optional_text(value: Any, field: str, *, maximum: int = 160) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise SourceAdmissionError(f"source admission {field} is invalid")
    return value


def _optional_https_url(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise SourceAdmissionError(f"source admission {field} is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SourceAdmissionError(f"source admission {field} must be a credential-free HTTPS URL")
    return value


def _optional_date(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SourceAdmissionError(f"source admission {field} is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SourceAdmissionError(f"source admission {field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise SourceAdmissionError(f"source admission {field} is invalid")
    return value


def _optional_size(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= MAX_EXTERNAL_ASSET_BYTES:
        raise SourceAdmissionError(f"source admission {field} is invalid")
    return value


def _fact_state(value: Any, field: str) -> str:
    if value not in _FACT_STATES:
        raise SourceAdmissionError(f"source admission {field} is invalid")
    return value


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise SourceAdmissionError(f"source admission {field} must be boolean")
    return value


def _codes(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 32:
        raise SourceAdmissionError(f"source admission {field} must contain at most 32 codes")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or _SAFE_CODE.fullmatch(item) is None:
            raise SourceAdmissionError(f"source admission {field} contains an invalid code")
        result.append(item)
    if len(set(result)) != len(result):
        raise SourceAdmissionError(f"source admission {field} codes must be unique")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SourceLicense:
    name: str | None
    spdx_id: str | None
    terms_url: str | None
    commercial_use: str
    modification: str
    redistribution: str

    def __post_init__(self) -> None:
        _optional_text(self.name, "license name")
        if self.spdx_id is not None and (
            not isinstance(self.spdx_id, str) or _SAFE_SPDX.fullmatch(self.spdx_id) is None
        ):
            raise SourceAdmissionError("source admission license spdx_id is invalid")
        _optional_https_url(self.terms_url, "license terms_url")
        for field in ("commercial_use", "modification", "redistribution"):
            if getattr(self, field) not in _LICENSE_USE:
                raise SourceAdmissionError(f"source admission license {field} is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> SourceLicense:
        data = _exact_object(
            value,
            {
                "name",
                "spdx_id",
                "terms_url",
                "commercial_use",
                "modification",
                "redistribution",
            },
            "license",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "spdx_id": self.spdx_id,
            "terms_url": self.terms_url,
            "commercial_use": self.commercial_use,
            "modification": self.modification,
            "redistribution": self.redistribution,
        }


@dataclass(frozen=True, slots=True)
class SourceAccess:
    gated: str
    authentication_required: str
    expected_bytes: int | None

    def __post_init__(self) -> None:
        _fact_state(self.gated, "access gated")
        _fact_state(self.authentication_required, "access authentication_required")
        _optional_size(self.expected_bytes, "access expected_bytes")

    @classmethod
    def from_dict(cls, value: Any) -> SourceAccess:
        data = _exact_object(
            value,
            {"gated", "authentication_required", "expected_bytes"},
            "access",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "gated": self.gated,
            "authentication_required": self.authentication_required,
            "expected_bytes": self.expected_bytes,
        }


@dataclass(frozen=True, slots=True)
class SourceContent:
    observed_bytes: int | None
    sha256: str | None

    def __post_init__(self) -> None:
        _optional_size(self.observed_bytes, "content observed_bytes")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise SourceAdmissionError("source admission content sha256 is invalid")
        if (self.observed_bytes is None) != (self.sha256 is None):
            raise SourceAdmissionError(
                "source admission content size and sha256 must be recorded together"
            )

    @property
    def fingerprint_present(self) -> bool:
        return self.observed_bytes is not None and self.sha256 is not None

    @classmethod
    def from_dict(cls, value: Any) -> SourceContent:
        data = _exact_object(
            value,
            {"observed_bytes", "sha256"},
            "content",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_bytes": self.observed_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SourcePrivacy:
    real_minors: str
    faces: str
    audio: str
    device_or_account_ids: str
    precise_timestamps: str
    private_household_data: str

    def __post_init__(self) -> None:
        for field in (
            "real_minors",
            "faces",
            "audio",
            "device_or_account_ids",
            "precise_timestamps",
            "private_household_data",
        ):
            _fact_state(getattr(self, field), f"privacy {field}")

    @classmethod
    def from_dict(cls, value: Any) -> SourcePrivacy:
        data = _exact_object(
            value,
            {
                "real_minors",
                "faces",
                "audio",
                "device_or_account_ids",
                "precise_timestamps",
                "private_household_data",
            },
            "privacy",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "real_minors": self.real_minors,
            "faces": self.faces,
            "audio": self.audio,
            "device_or_account_ids": self.device_or_account_ids,
            "precise_timestamps": self.precise_timestamps,
            "private_household_data": self.private_household_data,
        }


@dataclass(frozen=True, slots=True)
class SourceLabels:
    origin: str
    human_verified: bool
    ground_truth: bool

    def __post_init__(self) -> None:
        if self.origin not in _LABEL_ORIGINS:
            raise SourceAdmissionError("source admission label origin is invalid")
        _boolean(self.human_verified, "labels human_verified")
        _boolean(self.ground_truth, "labels ground_truth")
        if self.ground_truth and self.origin not in _GROUND_TRUTH_ORIGINS:
            raise SourceAdmissionError(
                "source admission weak or publisher labels cannot be ground truth"
            )
        if (
            self.ground_truth
            and self.origin in {"human_expert", "human_nonexpert"}
            and not self.human_verified
        ):
            raise SourceAdmissionError("source admission human ground truth must be human verified")

    @classmethod
    def from_dict(cls, value: Any) -> SourceLabels:
        data = _exact_object(
            value,
            {"origin", "human_verified", "ground_truth"},
            "labels",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "human_verified": self.human_verified,
            "ground_truth": self.ground_truth,
        }


@dataclass(frozen=True, slots=True)
class SourceAdmissionDecision:
    tier: str | None
    status: str
    purpose: str
    requires_user_approval: bool
    user_approval_recorded: bool
    git_allowed: bool
    ci_allowed: bool
    redistribution_allowed: bool
    training_allowed: bool
    reviewed_on: str | None
    blocker_codes: tuple[str, ...]
    restriction_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tier is not None and self.tier not in _TIERS:
            raise SourceAdmissionError("source admission tier is invalid")
        if self.status not in _ADMISSION_STATUSES:
            raise SourceAdmissionError("source admission status is invalid")
        if self.purpose not in _PURPOSES:
            raise SourceAdmissionError("source admission purpose is invalid")
        for field in (
            "requires_user_approval",
            "user_approval_recorded",
            "git_allowed",
            "ci_allowed",
            "redistribution_allowed",
            "training_allowed",
        ):
            _boolean(getattr(self, field), f"admission {field}")
        _optional_date(self.reviewed_on, "admission reviewed_on")
        if not isinstance(self.blocker_codes, tuple) or not isinstance(
            self.restriction_codes, tuple
        ):
            raise SourceAdmissionError("source admission codes must be arrays")
        for field in ("blocker_codes", "restriction_codes"):
            values = getattr(self, field)
            if len(values) > 32 or any(
                not isinstance(item, str) or _SAFE_CODE.fullmatch(item) is None for item in values
            ):
                raise SourceAdmissionError(f"source admission {field} is invalid")
            if len(set(values)) != len(values):
                raise SourceAdmissionError(f"source admission {field} codes must be unique")

    @classmethod
    def from_dict(cls, value: Any) -> SourceAdmissionDecision:
        data = _exact_object(
            value,
            {
                "tier",
                "status",
                "purpose",
                "requires_user_approval",
                "user_approval_recorded",
                "git_allowed",
                "ci_allowed",
                "redistribution_allowed",
                "training_allowed",
                "reviewed_on",
                "blocker_codes",
                "restriction_codes",
            },
            "decision",
        )
        return cls(
            **{
                **data,
                "blocker_codes": _codes(data["blocker_codes"], "blocker_codes"),
                "restriction_codes": _codes(data["restriction_codes"], "restriction_codes"),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "status": self.status,
            "purpose": self.purpose,
            "requires_user_approval": self.requires_user_approval,
            "user_approval_recorded": self.user_approval_recorded,
            "git_allowed": self.git_allowed,
            "ci_allowed": self.ci_allowed,
            "redistribution_allowed": self.redistribution_allowed,
            "training_allowed": self.training_allowed,
            "reviewed_on": self.reviewed_on,
            "blocker_codes": list(self.blocker_codes),
            "restriction_codes": list(self.restriction_codes),
        }


@dataclass(frozen=True, slots=True)
class ExternalSourceAdmission:
    source_id: str
    asset_kind: str
    canonical_url: str | None
    accessed_on: str | None
    source_revision: str | None
    upstream_name: str | None
    license: SourceLicense
    access: SourceAccess
    content: SourceContent
    privacy: SourcePrivacy
    labels: SourceLabels
    admission: SourceAdmissionDecision
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise SourceAdmissionError("unsupported source admission version")
        _identifier(self.source_id, "source_id")
        if self.asset_kind not in _ASSET_KINDS:
            raise SourceAdmissionError("source admission asset_kind is invalid")
        _optional_https_url(self.canonical_url, "canonical_url")
        _optional_date(self.accessed_on, "accessed_on")
        _optional_revision(self.source_revision)
        _optional_text(self.upstream_name, "upstream_name")
        if not isinstance(self.license, SourceLicense):
            raise SourceAdmissionError("source admission license is invalid")
        if not isinstance(self.access, SourceAccess):
            raise SourceAdmissionError("source admission access is invalid")
        if not isinstance(self.content, SourceContent):
            raise SourceAdmissionError("source admission content is invalid")
        if not isinstance(self.privacy, SourcePrivacy):
            raise SourceAdmissionError("source admission privacy is invalid")
        if not isinstance(self.labels, SourceLabels):
            raise SourceAdmissionError("source admission labels are invalid")
        if not isinstance(self.admission, SourceAdmissionDecision):
            raise SourceAdmissionError("source admission decision is invalid")
        self._validate_policy()

    def _validate_policy(self) -> None:
        decision = self.admission
        license_profile = self.license
        privacy = self.privacy
        access = self.access

        if (
            self.accessed_on
            and decision.reviewed_on
            and date.fromisoformat(self.accessed_on) > date.fromisoformat(decision.reviewed_on)
        ):
            raise SourceAdmissionError("source admission access date cannot follow review date")

        if decision.user_approval_recorded and not decision.requires_user_approval:
            raise SourceAdmissionError(
                "source admission cannot record approval when none is required"
            )

        large_model = (
            self.asset_kind == "model_weights"
            and access.expected_bytes is not None
            and access.expected_bytes >= LARGE_MODEL_WEIGHT_BYTES
        )
        risk_requires_approval = any(
            (
                access.gated == "yes",
                access.authentication_required == "yes",
                privacy.real_minors == "yes",
                privacy.private_household_data == "yes",
                large_model,
            )
        )
        if risk_requires_approval and not decision.requires_user_approval:
            raise SourceAdmissionError(
                "source admission sensitive or large sources require user approval"
            )
        if (
            decision.status != "blocked"
            and decision.requires_user_approval
            and not decision.user_approval_recorded
        ):
            raise SourceAdmissionError(
                "source admission cannot proceed without recorded user approval"
            )

        if decision.status == "blocked":
            if not decision.blocker_codes:
                raise SourceAdmissionError("blocked source admission requires blocker codes")
            if any(
                (
                    decision.git_allowed,
                    decision.ci_allowed,
                    decision.redistribution_allowed,
                    decision.training_allowed,
                )
            ):
                raise SourceAdmissionError("blocked source admission cannot grant use permissions")
        elif decision.blocker_codes:
            raise SourceAdmissionError("approved source admission cannot retain blocker codes")

        fixed_facts = (
            self.canonical_url,
            self.accessed_on,
            self.source_revision,
            self.upstream_name,
            license_profile.name,
            license_profile.terms_url,
            access.expected_bytes,
            decision.reviewed_on,
            decision.tier,
        )
        unknown_facts = (
            license_profile.commercial_use,
            license_profile.modification,
            license_profile.redistribution,
            access.gated,
            access.authentication_required,
            privacy.real_minors,
            privacy.faces,
            privacy.audio,
            privacy.device_or_account_ids,
            privacy.precise_timestamps,
            privacy.private_household_data,
            self.labels.origin,
        )
        if decision.status != "blocked" and (
            any(value is None for value in fixed_facts) or "unknown" in unknown_facts
        ):
            raise SourceAdmissionError(
                "source admission with unknown provenance must remain blocked"
            )

        if decision.status == "approved_for_download":
            if self.content.fingerprint_present:
                raise SourceAdmissionError(
                    "download-approved source must not claim a local fingerprint"
                )
        elif decision.status == "verified_local":
            if not self.content.fingerprint_present:
                raise SourceAdmissionError("verified source admission requires size and sha256")
            if access.expected_bytes != self.content.observed_bytes:
                raise SourceAdmissionError("source admission expected and observed sizes differ")

        sensitive_metadata = (
            privacy.device_or_account_ids == "yes" or privacy.precise_timestamps == "yes"
        )
        if decision.tier == "A" and (
            privacy.real_minors == "yes"
            or privacy.private_household_data == "yes"
            or sensitive_metadata
            or access.gated == "yes"
            or access.authentication_required == "yes"
            or license_profile.commercial_use != "allowed"
            or license_profile.modification != "allowed"
            or license_profile.redistribution != "allowed"
        ):
            raise SourceAdmissionError(
                "source admission Tier A conflicts with access, license, or privacy"
            )
        if decision.tier in {"B", "C"} and any(
            (
                decision.git_allowed,
                decision.ci_allowed,
                decision.redistribution_allowed,
                decision.training_allowed,
            )
        ):
            raise SourceAdmissionError(
                "source admission Tier B/C must remain local and non-training"
            )
        if decision.tier is None and decision.status != "blocked":
            raise SourceAdmissionError("approved source admission requires an A/B/C tier")
        if privacy.real_minors == "yes" and decision.tier == "A":
            raise SourceAdmissionError("source admission with real minors cannot enter Tier A")
        if privacy.private_household_data == "yes" and decision.tier not in {
            None,
            "C",
        }:
            raise SourceAdmissionError("private household source admission must use Tier C")
        if self.asset_kind == "vendor_event_export" and (
            decision.status != "blocked" and decision.tier != "C"
        ):
            raise SourceAdmissionError("vendor event exports require private Tier C admission")

        if decision.redistribution_allowed and (
            decision.tier != "A"
            or decision.status != "verified_local"
            or license_profile.redistribution != "allowed"
        ):
            raise SourceAdmissionError("source admission redistribution permission is inconsistent")
        if (decision.git_allowed or decision.ci_allowed) and (
            decision.tier != "A"
            or decision.status != "verified_local"
            or not self.content.fingerprint_present
        ):
            raise SourceAdmissionError(
                "source admission Git/CI use requires verified Tier A content"
            )
        if decision.training_allowed and (
            decision.tier != "A"
            or decision.status != "verified_local"
            or privacy.real_minors != "no"
            or privacy.private_household_data != "no"
            or license_profile.commercial_use != "allowed"
            or license_profile.modification != "allowed"
        ):
            raise SourceAdmissionError("source admission training permission is inconsistent")

    @classmethod
    def from_dict(cls, value: Any) -> ExternalSourceAdmission:
        data = _exact_object(
            value,
            {
                "version",
                "source_id",
                "asset_kind",
                "canonical_url",
                "accessed_on",
                "source_revision",
                "upstream_name",
                "license",
                "access",
                "content",
                "privacy",
                "labels",
                "admission",
            },
            "record",
        )
        return cls(
            version=data["version"],
            source_id=data["source_id"],
            asset_kind=data["asset_kind"],
            canonical_url=data["canonical_url"],
            accessed_on=data["accessed_on"],
            source_revision=data["source_revision"],
            upstream_name=data["upstream_name"],
            license=SourceLicense.from_dict(data["license"]),
            access=SourceAccess.from_dict(data["access"]),
            content=SourceContent.from_dict(data["content"]),
            privacy=SourcePrivacy.from_dict(data["privacy"]),
            labels=SourceLabels.from_dict(data["labels"]),
            admission=SourceAdmissionDecision.from_dict(data["admission"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> ExternalSourceAdmission:
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> ExternalSourceAdmission:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_SOURCE_ADMISSION_BYTES:
                raise SourceAdmissionError("source admission manifest exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except SourceAdmissionError:
            raise
        except (OSError, UnicodeError) as exc:
            raise SourceAdmissionError("source admission manifest is missing or unsafe") from exc
        return cls.from_json(raw)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_id": self.source_id,
            "asset_kind": self.asset_kind,
            "canonical_url": self.canonical_url,
            "accessed_on": self.accessed_on,
            "source_revision": self.source_revision,
            "upstream_name": self.upstream_name,
            "license": self.license.to_dict(),
            "access": self.access.to_dict(),
            "content": self.content.to_dict(),
            "privacy": self.privacy.to_dict(),
            "labels": self.labels.to_dict(),
            "admission": self.admission.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_public_dict(
        self,
        *,
        asset_read: bool = False,
        asset_verified: bool = False,
    ) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.version,
            "source_id": self.source_id,
            "asset_kind": self.asset_kind,
            "tier": self.admission.tier,
            "status": self.admission.status,
            "purpose": self.admission.purpose,
            "manifest_sha256": self.fingerprint,
            "content_fingerprint_present": self.content.fingerprint_present,
            "ready_for_download": self.admission.status
            in {"approved_for_download", "verified_local"},
            "requires_user_approval": self.admission.requires_user_approval,
            "user_approval_recorded": self.admission.user_approval_recorded,
            "contains_real_minors": self.privacy.real_minors,
            "label_origin": self.labels.origin,
            "ground_truth": self.labels.ground_truth,
            "blocker_codes": list(self.admission.blocker_codes),
            "restriction_codes": list(self.admission.restriction_codes),
            "asset_read": asset_read,
            "asset_verified": asset_verified,
            "paths_disclosed": False,
            "urls_disclosed": False,
            "network_used": False,
        }

    def verify_asset(self, path: str | Path) -> SourceAssetVerification:
        if self.admission.status != "verified_local":
            raise SourceAdmissionError(
                "source admission must be verified_local before asset verification"
            )
        if not self.content.fingerprint_present:
            raise SourceAdmissionError("source admission content fingerprint is missing")
        asset = Path(path).expanduser()
        try:
            if asset.is_symlink() or not asset.is_file():
                raise OSError
            stat = asset.stat()
        except OSError as exc:
            raise SourceAdmissionError("external source asset is missing or unsafe") from exc
        if not 1 <= stat.st_size <= MAX_EXTERNAL_ASSET_BYTES:
            raise SourceAdmissionError("external source asset size is unsafe")
        if stat.st_size != self.content.observed_bytes:
            raise SourceAdmissionError(
                "external source asset size does not match admission manifest"
            )
        digest = hashlib.sha256()
        try:
            with asset.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
        except OSError as exc:
            raise SourceAdmissionError("external source asset could not be read") from exc
        sha256 = digest.hexdigest()
        if sha256 != self.content.sha256:
            raise SourceAdmissionError(
                "external source asset sha256 does not match admission manifest"
            )
        return SourceAssetVerification(
            manifest=self,
            observed_bytes=stat.st_size,
            sha256=sha256,
        )


@dataclass(frozen=True, slots=True)
class SourceAssetVerification:
    manifest: ExternalSourceAdmission
    observed_bytes: int
    sha256: str

    def to_public_dict(self) -> dict[str, object]:
        payload = self.manifest.to_public_dict(
            asset_read=True,
            asset_verified=True,
        )
        payload.update(
            {
                "observed_bytes": self.observed_bytes,
                "sha256": self.sha256,
            }
        )
        return payload
