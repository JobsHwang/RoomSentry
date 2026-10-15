"""Explicit, idempotent local secondary review of a historically bound event.

No primary/remote factories, notification routing, or report publishing are
available to this use case. Endpoint verification is not a model-input snapshot.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from .adapters.vlm import CommandVLMAdapter, MockVLMAdapter, VLMAdapter, VLMError, VLMRequest
from .analysis_queue import _candidate_from_dict
from .evidence_binding import EvidenceBindingError
from .local_review import LocalReviewError, LocalReviewReservation, LocalReviewStore
from .pipeline import _run_recognition_attempt
from .policy import (
    ReviewPolicy,
    has_semantic_contradiction,
    record_secondary_result,
    result_contradictions,
)
from .recognition_attempt import RecognitionAttemptAuditStore, RecognitionStage


def _receipt(reservation: LocalReviewReservation, *, decision=None, error_code=None) -> dict:
    current = decision or reservation.event.decision
    return {
        "request_id": reservation.request_id,
        "event_id": reservation.event.event_id,
        "status": reservation.status,
        "reused_existing": reservation.reused_existing,
        "error_code": error_code or reservation.error_code,
        "review_tier": current.review_tier.value,
        "needs_review": current.needs_review,
        "primary_rerun": False,
        "remote_provider_called": False,
        "notifications_sent": False,
        "reports_refreshed": False,
    }


def _retry_local_review_once(
    *,
    database_path: Path,
    event_id: str,
    request_id: str,
    secondary_adapter_factory: Callable[[], VLMAdapter | None],
    policy: ReviewPolicy,
    lease_seconds: int = 600,
    report_storage_root: Path | None = None,
    report_timezone: str | None = None,
) -> dict:
    """Run at most one local secondary attempt; replayed request IDs never run.

    The injected factory is trusted application code, and must return a local
    mock/command adapter. This is not a sandbox for an arbitrary command script.
    A failed attempt requires a new explicit request ID, not an automatic loop.
    """
    if not isinstance(policy, ReviewPolicy):
        raise LocalReviewError("invalid_input")
    store = LocalReviewStore(database_path)
    reservation = store.reserve(
        event_id,
        request_id,
        lease_seconds=lease_seconds,
        report_storage_root=report_storage_root,
        report_timezone=report_timezone,
    )
    if reservation.reused_existing:
        return _receipt(reservation)
    event = reservation.event
    try:
        # Refuse changed historical evidence before constructing/probing a model.
        reservation.binding.verify(event.evidence)
        candidate = _candidate_from_dict(event.candidate)
        if candidate.to_dict() != event.candidate or candidate.event_id != event.event_id:
            raise LocalReviewError("invalid_record")
        adapter = secondary_adapter_factory()
        if not isinstance(adapter, (MockVLMAdapter, CommandVLMAdapter)):
            raise VLMError(
                "local review requires an enabled local secondary adapter",
                code="local_secondary_required",
            )
        secondary, metrics = _run_recognition_attempt(
            audit=RecognitionAttemptAuditStore(database_path),
            event_id=event.event_id,
            stage=RecognitionStage.LOCAL_SECONDARY,
            adapter=adapter,
            request=VLMRequest(candidate=candidate, media=event.evidence),
            media=event.evidence,
        )
        reservation.binding.verify(event.evidence)
        decision = policy.decide(
            secondary,
            signals=candidate.signals,
            signal_frequency_count=event.decision.signal_frequency_count,
            contradiction_detected=has_semantic_contradiction(
                result_contradictions(event.local_result, secondary)
            ),
            remote_allowed=event.decision.remote_allowed,
            evidence_complete=event.evidence.complete_coverage,
        )
        # Retrying a local opinion cannot silently erase historical escalation
        # reasons or grant upload permission. A remote need stays a pending need.
        decision = replace(
            decision,
            review_tier=max(
                (decision.review_tier, event.decision.review_tier), key=lambda tier: tier.rank
            ),
            review_reasons=tuple(
                dict.fromkeys((*event.decision.review_reasons, *decision.review_reasons))
            ),
            contradictions=tuple(
                dict.fromkeys((*event.decision.contradictions, *decision.contradictions))
            ),
            signal_frequency_window_seconds=event.decision.signal_frequency_window_seconds,
            frequent_signal_types=event.decision.frequent_signal_types,
        )
        decision = record_secondary_result(
            decision, primary=event.local_result, secondary=secondary, final=secondary
        )
        store.complete(
            request_id,
            secondary_result=secondary,
            secondary_provenance=adapter.provenance,
            secondary_metrics=metrics,
            decision=decision,
        )
        return _receipt(replace(reservation, status="succeeded"), decision=decision)
    except (EvidenceBindingError, VLMError, LocalReviewError) as exc:
        if isinstance(exc, EvidenceBindingError):
            code = "evidence_changed"
        elif isinstance(exc, VLMError):
            code = "model_failed"
        else:
            code = "review_failed"
        # If persistence fails too, propagate that failure. Do not manufacture a
        # terminal receipt for an attempt that may still be recorded as running.
        store.fail(request_id, error_code=code)
        return _receipt(replace(reservation, status="failed"), error_code=code)
    except BaseException:
        with suppress(Exception):
            store.fail(request_id, error_code="review_failed")
        raise


def retry_local_review(
    *,
    database_path: Path,
    event_id: str,
    request_id: str,
    secondary_adapter_factory: Callable[[], VLMAdapter | None],
    policy: ReviewPolicy,
    lease_seconds: int = 600,
    report_storage_root: Path | None = None,
    report_timezone: str | None = None,
) -> dict:
    """Review once, with an optional durable report intent but no publication.

    Report-status lookup happens after the review outcome has been decided. A
    status-read failure must not relabel a committed successful review as failed.
    """
    receipt = _retry_local_review_once(
        database_path=database_path,
        event_id=event_id,
        request_id=request_id,
        secondary_adapter_factory=secondary_adapter_factory,
        policy=policy,
        lease_seconds=lease_seconds,
        report_storage_root=report_storage_root,
        report_timezone=report_timezone,
    )
    try:
        receipt["report_refresh"] = LocalReviewStore(database_path).report_refresh_state(request_id)
    except LocalReviewError:
        receipt["report_refresh"] = {
            "status": "unknown",
            "error_code": "report_status_unavailable",
        }
    return receipt
