"""Injected audit and budget contract for remote-review providers.

The provider depends on this protocol; SQLite is one application-supplied store.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol


class RemoteAttemptStatus(StrEnum):
    RESERVED = "reserved"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RemoteBudgetError(RuntimeError):
    def __init__(self, code: str):
        if code not in {"daily_budget_exhausted", "rate_limited"}:
            raise ValueError("unsupported remote budget error code")
        self.code = code
        super().__init__("remote request refused by the local call budget")


class RemoteReviewAudit(Protocol):
    def initialize(self) -> None: ...

    def reserve(
        self,
        *,
        event_id: str,
        requested_at: datetime,
        target_origin: str,
        model: str,
        frame_count: int,
        frame_bytes: int,
        max_requests_per_day: int,
        min_interval_seconds: int,
    ) -> str: ...

    def complete(
        self,
        attempt_id: str,
        *,
        status: RemoteAttemptStatus,
        completed_at: datetime,
        duration_ms: int,
        error_code: str | None = None,
    ) -> None: ...
