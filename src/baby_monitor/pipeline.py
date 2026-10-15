from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic_ns

from baby_monitor.adapters.camera import CameraAdapter
from baby_monitor.adapters.detector import Detector, group_detections
from baby_monitor.adapters.media import MediaProcessor
from baby_monitor.adapters.vlm import VLMAdapter, VLMError, VLMRequest
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    RecognitionRunMetrics,
    ReviewTier,
    SignalEvent,
    VLMEvent,
)
from baby_monitor.evidence_binding import EvidenceBindingError, capture_evidence_binding
from baby_monitor.fact_policy import (
    enforce_subject_identity_boundary,
    enforce_temporal_fact_boundary,
)
from baby_monitor.policy import (
    ReviewPolicy,
    has_semantic_contradiction,
    reconcile_remote,
    record_remote_result,
    record_secondary_result,
    result_contradictions,
)
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)
from baby_monitor.signal_binding import bind_signals_to_candidates
from baby_monitor.storage import StorageResolver


@dataclass(frozen=True, slots=True)
class PipelineRunSummary:
    detections: int
    candidates: int
    stored_event_ids: tuple[str, ...]
    secondary_reviews: int
    remote_reviews: int
    pending_remote_reviews: int
    reused_events: int


@dataclass(frozen=True, slots=True)
class PreparedEventSummary:
    event_id: str
    secondary_reviewed: bool
    remote_reviewed: bool
    pending_remote_review: bool
    reused_existing: bool
    review_tier: ReviewTier
    review_reasons: tuple[str, ...]


def _normalize_evidence_bindings(result: VLMEvent, media: EventMedia) -> VLMEvent:
    allowed_paths = set(media.frame_paths)
    paths_by_name: dict[str, list[str]] = {}
    for path in media.frame_paths:
        paths_by_name.setdefault(Path(path).name, []).append(path)

    def canonical_path(reference: str) -> str:
        if reference in allowed_paths:
            return reference
        if reference == Path(reference).name:
            matches = paths_by_name.get(reference, [])
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise VLMError(
                    "VLM evidence basename is ambiguous within the event",
                    code="invalid_evidence_binding",
                )
        raise VLMError(
            "VLM result referenced an evidence frame outside the event",
            code="invalid_evidence_binding",
        )

    def canonical_paths(references: tuple[str, ...]) -> tuple[str, ...]:
        paths = tuple(canonical_path(reference) for reference in references)
        if len(set(paths)) != len(paths):
            raise VLMError(
                "VLM evidence references resolve to duplicate event frames",
                code="invalid_evidence_binding",
            )
        return paths

    if not result.evidence_frames:
        raise VLMError(
            "VLM result must cite at least one event evidence frame",
            code="invalid_evidence_binding",
        )
    root_frames = canonical_paths(result.evidence_frames)
    root_references = set(root_frames)
    normalized_subjects = []
    for subject in result.subjects:
        if not subject.evidence_frames:
            raise VLMError(
                "VLM subject must cite at least one event evidence frame",
                code="invalid_evidence_binding",
            )
        subject_frames = canonical_paths(subject.evidence_frames)
        subject_references = set(subject_frames)
        if not subject_references.issubset(root_references):
            raise VLMError(
                "VLM subject evidence must also appear in root evidence_frames",
                code="invalid_evidence_binding",
            )
        normalized_subjects.append(replace(subject, evidence_frames=subject_frames))
    return replace(
        result,
        subjects=tuple(normalized_subjects),
        evidence_frames=root_frames,
    )


def _run_recognition_attempt(
    *,
    audit: RecognitionAttemptAuditStore,
    event_id: str,
    stage: RecognitionStage,
    adapter: VLMAdapter,
    request: VLMRequest,
    media: EventMedia,
) -> tuple[VLMEvent, RecognitionRunMetrics]:
    started_at = datetime.now(UTC)
    try:
        attempt_id = audit.start(
            event_id=event_id,
            stage=stage,
            provenance=adapter.provenance,
            started_at=started_at,
        )
    except Exception:
        raise VLMError(
            "recognition attempt audit is unavailable; model run was refused",
            code="recognition_audit_unavailable",
        ) from None

    started_ns = monotonic_ns()
    analysis = None
    try:
        analysis = adapter.analyze_with_metrics(request)
        temporal_bounded_result = enforce_temporal_fact_boundary(
            analysis.result,
            request.candidate,
        )
        bounded_result = enforce_subject_identity_boundary(
            temporal_bounded_result,
            request.candidate,
        )
        result = _normalize_evidence_bindings(bounded_result, media)
    except BaseException as exc:
        if analysis is not None:
            metrics = analysis.metrics
        elif isinstance(exc, VLMError) and exc.metrics is not None:
            metrics = exc.metrics
        else:
            metrics = RecognitionRunMetrics(
                duration_ms=min(
                    86_400_000,
                    max(0, (monotonic_ns() - started_ns) // 1_000_000),
                ),
                json_compliant=False,
            )
        error_code = exc.code if isinstance(exc, VLMError) else "unexpected_failure"
        try:
            audit.complete(
                attempt_id,
                status=RecognitionAttemptStatus.FAILED,
                completed_at=datetime.now(UTC),
                metrics=metrics,
                error_code=error_code,
            )
        except Exception:
            if not isinstance(exc, VLMError):
                # Audit failure must not reclassify cancellation or an
                # unexpected adapter error as recoverable model failure.
                raise exc from None
            raise VLMError(
                "recognition attempt audit could not record the failed run",
                code="recognition_audit_write_failed",
            ) from None
        raise

    try:
        audit.complete(
            attempt_id,
            status=RecognitionAttemptStatus.SUCCEEDED,
            completed_at=datetime.now(UTC),
            metrics=analysis.metrics,
        )
    except Exception:
        raise VLMError(
            "recognition attempt audit could not record the completed run",
            code="recognition_audit_write_failed",
        ) from None
    return result, analysis.metrics


def process_prepared_event(
    *,
    candidate: CandidateEvent,
    occurred_at: datetime,
    source_kind: str,
    source_ref: str,
    media: EventMedia,
    local_vlm: VLMAdapter,
    database: EventDatabase,
    policy: ReviewPolicy,
    secondary_vlm: VLMAdapter | None = None,
    remote_vlm: VLMAdapter | None = None,
    remote_uploads_allowed: bool = False,
) -> PreparedEventSummary:
    if occurred_at.tzinfo is None:
        raise ValueError("occurred_at must be timezone-aware")
    if not source_kind or not source_ref:
        raise ValueError("source identity cannot be empty")

    database.initialize()
    existing = database.get_event(candidate.event_id)
    if existing is not None:
        immutable_matches = (
            existing.camera_id == candidate.camera_id
            and existing.occurred_at.astimezone(UTC) == occurred_at.astimezone(UTC)
            and existing.source_kind == source_kind
            and existing.source_ref == source_ref
            and existing.candidate == candidate.to_dict()
            and existing.clip_path == media.clip_path
            and (not existing.evidence_metadata_present or existing.evidence == media)
        )
        if not immutable_matches:
            raise ValueError("event_id already exists with different immutable event data")
        return PreparedEventSummary(
            event_id=candidate.event_id,
            secondary_reviewed=existing.secondary_result is not None,
            remote_reviewed=existing.remote_used,
            pending_remote_review=existing.decision.needs_remote_review,
            reused_existing=True,
            review_tier=existing.decision.review_tier,
            review_reasons=existing.decision.review_reasons,
        )

    # Legacy/unavailable assets may still be processed, but cannot gain a
    # historical content binding by being read later during a review retry.
    try:
        evidence_binding = capture_evidence_binding(media)
    except EvidenceBindingError:
        evidence_binding = None

    request = VLMRequest(candidate=candidate, media=media)
    attempt_audit = RecognitionAttemptAuditStore(database.path)
    local_result, local_metrics = _run_recognition_attempt(
        audit=attempt_audit,
        event_id=candidate.event_id,
        stage=RecognitionStage.LOCAL_PRIMARY,
        adapter=local_vlm,
        request=request,
        media=media,
    )
    signal_types = tuple(sorted({signal.signal_type for signal in candidate.signals}, key=str))
    signal_frequency_count = 0
    if signal_types:
        signal_frequency_count = 1 + database.count_recent_signal_events(
            camera_id=candidate.camera_id,
            signal_types=signal_types,
            occurred_at=occurred_at,
            window_seconds=policy.signal_frequency_window_seconds,
        )
    decision = policy.decide(
        local_result,
        signals=candidate.signals,
        signal_frequency_count=signal_frequency_count,
        remote_allowed=remote_uploads_allowed,
        evidence_complete=media.complete_coverage,
    )
    remote_result = None
    remote_metrics = None
    secondary_result = None
    secondary_metrics = None
    remote_used = False
    final_result = local_result

    if decision.review_tier.rank >= ReviewTier.LOCAL_SECONDARY.rank and secondary_vlm:
        try:
            secondary_result, secondary_metrics = _run_recognition_attempt(
                audit=attempt_audit,
                event_id=candidate.event_id,
                stage=RecognitionStage.LOCAL_SECONDARY,
                adapter=secondary_vlm,
                request=request,
                media=media,
            )
        except VLMError:
            # Preserve the successful primary result and its pending decision.
            # Failure details belong to the bounded attempt audit, not facts.
            # This failure alone never promotes a local review to a remote call.
            decision = replace(
                decision,
                review_reasons=tuple(
                    dict.fromkeys((*decision.review_reasons, "local_secondary_review_failed"))
                ),
            )
        else:
            secondary_has_contradiction = has_semantic_contradiction(
                result_contradictions(local_result, secondary_result)
            )
            decision = policy.decide(
                secondary_result,
                signals=candidate.signals,
                signal_frequency_count=signal_frequency_count,
                contradiction_detected=secondary_has_contradiction,
                remote_allowed=remote_uploads_allowed,
                evidence_complete=media.complete_coverage,
            )
            final_result = secondary_result
            decision = record_secondary_result(
                decision,
                primary=local_result,
                secondary=secondary_result,
                final=final_result,
            )

    if decision.needs_remote_review:
        if remote_vlm is not None and remote_uploads_allowed:
            reviewed_local_result = final_result
            try:
                remote_result, remote_metrics = _run_recognition_attempt(
                    audit=attempt_audit,
                    event_id=candidate.event_id,
                    stage=RecognitionStage.REMOTE,
                    adapter=remote_vlm,
                    request=request,
                    media=media,
                )
            except VLMError:
                # A provider, network, budget, or response failure must never
                # prevent the already-produced local facts from being stored.
                final_result = reviewed_local_result.require_remote_review()
                if "remote_review_failed" not in decision.review_reasons:
                    decision = replace(
                        decision,
                        review_reasons=decision.review_reasons + ("remote_review_failed",),
                    )
            else:
                remote_used = True
                final_result = reconcile_remote(reviewed_local_result, remote_result)
                decision = record_remote_result(
                    decision,
                    local=reviewed_local_result,
                    remote=remote_result,
                    final=final_result,
                )
        else:
            final_result = final_result.require_remote_review()

    if evidence_binding is not None:
        try:
            evidence_binding.verify(media)
        except EvidenceBindingError:
            raise VLMError(
                "event evidence changed or became unavailable during recognition",
                code="evidence_binding_changed",
            ) from None

    database.insert_event(
        candidate=candidate,
        occurred_at=occurred_at,
        source_kind=source_kind,
        source_ref=source_ref,
        local_result=local_result,
        local_provenance=local_vlm.provenance,
        local_metrics=local_metrics,
        secondary_result=secondary_result,
        secondary_provenance=(
            secondary_vlm.provenance if secondary_result is not None and secondary_vlm else None
        ),
        secondary_metrics=secondary_metrics,
        remote_result=remote_result,
        remote_provenance=(
            remote_vlm.provenance if remote_result is not None and remote_vlm else None
        ),
        remote_metrics=remote_metrics,
        result=final_result,
        decision=decision,
        media=media,
        remote_used=remote_used,
        created_at=datetime.now(UTC),
        evidence_binding=evidence_binding,
    )
    return PreparedEventSummary(
        event_id=candidate.event_id,
        secondary_reviewed=secondary_result is not None,
        remote_reviewed=remote_used,
        pending_remote_review=(decision.needs_remote_review and not remote_used),
        reused_existing=False,
        review_tier=decision.review_tier,
        review_reasons=decision.review_reasons,
    )


class PipelineRunner:
    def __init__(
        self,
        *,
        camera: CameraAdapter,
        detector: Detector,
        media: MediaProcessor,
        local_vlm: VLMAdapter,
        database: EventDatabase,
        storage: StorageResolver,
        policy: ReviewPolicy,
        candidate_gap_seconds: float,
        pre_event_seconds: float,
        post_event_seconds: float,
        remote_vlm: VLMAdapter | None = None,
        secondary_vlm: VLMAdapter | None = None,
        remote_uploads_allowed: bool = False,
    ):
        self.camera = camera
        self.detector = detector
        self.media = media
        self.local_vlm = local_vlm
        self.database = database
        self.storage = storage
        self.policy = policy
        self.candidate_gap_seconds = candidate_gap_seconds
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.remote_vlm = remote_vlm
        self.secondary_vlm = secondary_vlm
        self.remote_uploads_allowed = remote_uploads_allowed

    def run(
        self,
        *,
        source_started_at: datetime | None = None,
        precomputed_detections: Sequence[Detection] | None = None,
        signals: Sequence[SignalEvent] = (),
    ) -> PipelineRunSummary:
        source = self.camera.source()
        detections = tuple(
            self.detector.scan(source) if precomputed_detections is None else precomputed_detections
        )
        if any(not isinstance(detection, Detection) for detection in detections):
            raise ValueError("detector output must contain Detection values only")
        base_time = source_started_at or datetime.now(UTC)
        if base_time.tzinfo is None or base_time.utcoffset() is None:
            raise ValueError("source_started_at must be timezone-aware")
        candidates = group_detections(
            detections,
            camera_id=source.camera_id,
            gap_seconds=self.candidate_gap_seconds,
            pre_event_seconds=self.pre_event_seconds,
            post_event_seconds=self.post_event_seconds,
        )
        candidates = bind_signals_to_candidates(
            candidates,
            signals,
            source_started_at=base_time,
        )
        self.database.initialize()

        stored_ids: list[str] = []
        remote_reviews = 0
        secondary_reviews = 0
        pending_reviews = 0
        reused_events = 0
        for candidate in candidates:
            occurred_at = base_time + timedelta(seconds=candidate.center_seconds)
            prepared_media = self.media.prepare(
                source=source,
                candidate=candidate,
                occurred_at=occurred_at,
                storage=self.storage,
            )
            processed = process_prepared_event(
                candidate=candidate,
                occurred_at=occurred_at,
                source_kind=source.kind,
                source_ref=source.audit_ref,
                media=prepared_media,
                local_vlm=self.local_vlm,
                secondary_vlm=self.secondary_vlm,
                remote_vlm=self.remote_vlm,
                remote_uploads_allowed=self.remote_uploads_allowed,
                database=self.database,
                policy=self.policy,
            )
            remote_reviews += int(processed.remote_reviewed)
            secondary_reviews += int(processed.secondary_reviewed)
            pending_reviews += int(processed.pending_remote_review)
            reused_events += int(processed.reused_existing)
            stored_ids.append(candidate.event_id)

        return PipelineRunSummary(
            detections=len(detections),
            candidates=len(candidates),
            stored_event_ids=tuple(stored_ids),
            secondary_reviews=secondary_reviews,
            remote_reviews=remote_reviews,
            pending_remote_reviews=pending_reviews,
            reused_events=reused_events,
        )
