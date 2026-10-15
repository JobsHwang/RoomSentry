"""Bind bounded source signals to one candidate without performing I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

from .domain import CandidateEvent, SignalEvent


def bind_signals_to_candidates(
    candidates: tuple[CandidateEvent, ...],
    signals: Sequence[SignalEvent],
    *,
    source_started_at: datetime,
) -> tuple[CandidateEvent, ...]:
    """Bind each bounded-file signal to exactly one containing candidate."""

    if source_started_at.tzinfo is None or source_started_at.utcoffset() is None:
        raise ValueError("source_started_at must be timezone-aware")
    if any(not isinstance(signal, SignalEvent) for signal in signals):
        raise ValueError("candidate signals must contain SignalEvent values only")

    unique_signals: dict[str, SignalEvent] = {}
    for signal in signals:
        existing = unique_signals.get(signal.signal_id)
        if existing is not None and existing != signal:
            raise ValueError("one signal_id has conflicting immutable data")
        unique_signals[signal.signal_id] = signal

    bound: list[list[SignalEvent]] = [list(candidate.signals) for candidate in candidates]
    base_time = source_started_at.astimezone(UTC)
    for signal in sorted(
        unique_signals.values(),
        key=lambda item: (item.occurred_at.astimezone(UTC), item.signal_id),
    ):
        matching = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.camera_id == signal.camera_id
            and candidate.start_seconds - 1e-6
            <= (signal.occurred_at.astimezone(UTC) - base_time).total_seconds()
            <= candidate.end_seconds + 1e-6
        ]
        if not matching:
            raise ValueError("a local perception signal is outside every candidate window")
        offset_seconds = (signal.occurred_at.astimezone(UTC) - base_time).total_seconds()
        target = min(
            matching,
            key=lambda index: (
                abs(candidates[index].center_seconds - offset_seconds),
                candidates[index].event_id,
            ),
        )
        existing = {item.signal_id: item for item in bound[target]}
        previous = existing.get(signal.signal_id)
        if previous is not None and previous != signal:
            raise ValueError("candidate contains a conflicting signal identity")
        if previous is None:
            bound[target].append(signal)

    return tuple(
        replace(
            candidate,
            signals=tuple(
                sorted(
                    candidate_signals,
                    key=lambda item: (
                        item.occurred_at.astimezone(UTC),
                        item.signal_id,
                    ),
                )
            ),
        )
        for candidate, candidate_signals in zip(candidates, bound, strict=True)
    )
