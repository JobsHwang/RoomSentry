"""Explicit local-only review retry and media-free attempt status."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.vlm import VLMError
from ..cli_support import _json_print, _review_policy, _secondary_adapter
from ..config import load_settings
from ..gpu_guard import probe_gpu_capacity
from ..local_review import LocalReviewError, LocalReviewStore
from ..local_review_reports import dispatch_local_review_reports
from ..local_review_service import retry_local_review


def command_local_review_retry(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    if not settings.secondary_vlm.enabled:
        raise ValueError("local review requires secondary_vlm.enabled; no model was started")
    database_path = Path(args.database) if args.database else settings.storage.database

    def adapter_factory():
        worker = settings.analysis_worker
        if worker.gpu_guard_enabled:
            capacity = probe_gpu_capacity(
                device=worker.gpu_device,
                min_free_vram_mib=worker.min_free_vram_mib,
                max_utilization_percent=worker.max_gpu_utilization_percent,
                timeout_seconds=worker.gpu_probe_timeout_seconds,
            )
            if not capacity.allowed:
                raise VLMError("GPU capacity does not permit local review", code="gpu_busy")
        return _secondary_adapter(settings)

    try:
        receipt = retry_local_review(
            database_path=database_path,
            event_id=args.event_id,
            request_id=args.request_id,
            secondary_adapter_factory=adapter_factory,
            policy=_review_policy(settings),
            lease_seconds=settings.secondary_vlm.timeout_seconds + 120,
            report_storage_root=settings.storage.local_root,
            report_timezone=settings.timezone,
        )
    except LocalReviewError:
        raise
    except Exception:
        # The use case preserves programming errors for embedded callers. The
        # CLI must not print arbitrary factory/adapter exception text or paths.
        raise RuntimeError(
            "local review failed unexpectedly; inspect the local attempt audit"
        ) from None
    if receipt["status"] == "succeeded":
        try:
            receipt["report_dispatch"] = dispatch_local_review_reports(
                database_path, request_id=args.request_id
            )
            receipt["report_refresh"] = LocalReviewStore(database_path).report_refresh_state(
                args.request_id
            )
        except LocalReviewError:
            # The review has already committed. Report delivery can recover
            # through the worker without running the recognizer a second time.
            receipt["report_refresh"] = {
                "status": "unknown",
                "error_code": "report_dispatch_unavailable",
            }
        except Exception:
            raise RuntimeError("review saved; report dispatch failed unexpectedly") from None
    _json_print(receipt)
    return 0 if receipt["status"] == "succeeded" else 1


def command_local_review_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    store = LocalReviewStore(database_path)
    rows = tuple(
        {**row, "report_refresh": store.report_refresh_state(row["request_id"])}
        for row in store.recent(limit=args.limit)
    )
    _json_print(
        {
            "attempts": list(rows),
            "returned": len(rows),
            "contains_media_content": False,
            "contains_media_paths": False,
            "contains_error_text": False,
        }
    )
    return 0
