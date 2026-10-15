from __future__ import annotations

import hashlib
import json

from .candidate_health import CandidateHealthState, CandidateWorkerHealth
from .room_presence import (
    StreamPresenceBatch,
    StreamPresenceSample,
    StreamPresenceSampleState,
)


def candidate_health_gap_sample(
    health: CandidateWorkerHealth,
) -> StreamPresenceSample | None:
    if not isinstance(health, CandidateWorkerHealth):
        raise ValueError("candidate health is invalid")
    if health.worker_running and health.state in {
        CandidateHealthState.STARTING,
        CandidateHealthState.SCANNING,
        CandidateHealthState.IDLE,
    }:
        return None
    identity = {
        "version": 1,
        "camera_id": health.camera_id,
        "scanner_id": health.scanner_identity,
        "detector_id": health.detector_identity,
        "checked_at": health.checked_at.isoformat(),
        "state": health.state.value,
        "worker_running": health.worker_running,
        "scans_succeeded": health.scans_succeeded,
        "scans_failed": health.scans_failed,
        "last_error_code": health.last_error_code,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    reason_code = (
        "candidate_worker_failed"
        if health.state is CandidateHealthState.FAILED
        else "candidate_worker_not_running"
    )
    return StreamPresenceSample(
        sample_id=f"presence_gap_{fingerprint[:20]}",
        camera_id=health.camera_id,
        observed_at=health.checked_at,
        state=StreamPresenceSampleState.COVERAGE_GAP,
        scanner_id=health.scanner_identity,
        detector_id=None,
        source_fingerprint=fingerprint,
        reason_code=reason_code,
    )


def merge_stream_presence_batches(
    batches: tuple[StreamPresenceBatch, ...],
    *,
    extra_samples: tuple[StreamPresenceSample, ...] = (),
) -> StreamPresenceBatch:
    if any(not isinstance(item, StreamPresenceBatch) for item in batches):
        raise ValueError("stream presence batch is invalid")
    if any(not isinstance(item, StreamPresenceSample) for item in extra_samples):
        raise ValueError("stream presence sample is invalid")
    samples = tuple(item for batch in batches for item in batch.samples) + extra_samples
    return StreamPresenceBatch(
        samples=tuple(
            sorted(
                samples,
                key=lambda item: (item.observed_at, item.camera_id, item.sample_id),
            )
        )
    )
