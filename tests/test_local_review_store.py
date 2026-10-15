"""Synthetic metadata-only local-review reservation and CAS tests."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    DecisionRecord,
    EventMedia,
    EventType,
    FinalSource,
    RecognitionProvenance,
    RecognitionRunMetrics,
    ReviewTier,
    RiskLevel,
)
from baby_monitor.evidence_binding import EvidenceAssetBinding, EvidenceBinding, media_fingerprint
from baby_monitor.local_review import LocalReviewError, LocalReviewStore
from tests.test_policy import _result


class LocalReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "events.sqlite3"
        self.database = EventDatabase(self.path)
        self.database.initialize()
        self.store = LocalReviewStore(self.path)
        self.now = datetime(2026, 9, 12, tzinfo=UTC)
        self.media = EventMedia("/absent/private/clip.mp4", ("/absent/private/frame.jpg",))
        self.binding = EvidenceBinding(
            media_fingerprint(self.media),
            (
                EvidenceAssetBinding("clip", 0, 1, "a" * 64),
                EvidenceAssetBinding("frame", 0, 1, "b" * 64),
            ),
        )
        self.primary = _result(confidence=0.4)
        self.secondary = _result(
            event_type=EventType.UNKNOWN, risk_level=RiskLevel.LOW, confidence=0.8
        )
        self.provenance = RecognitionProvenance(adapter_name="synthetic_secondary")
        self.metrics = RecognitionRunMetrics(duration_ms=12, json_compliant=True)
        self.decision = DecisionRecord(
            risk_level=self.primary.risk_level,
            review_tier=ReviewTier.LOCAL_SECONDARY,
            review_reasons=("low_confidence",),
            contradictions=(),
            remote_allowed=False,
            remote_used=False,
            final_source=FinalSource.LOCAL_PRIMARY,
        )
        self.after_decision = replace(
            self.decision,
            risk_level=self.secondary.risk_level,
            final_source=FinalSource.LOCAL_SECONDARY,
        )
        self._insert()

    def _insert(self, event_id="event_1", *, binding=True, decision=None, **overrides):
        values = dict(
            candidate=CandidateEvent(event_id, "cam_01", 0, 1, ()),
            occurred_at=self.now,
            source_kind="synthetic",
            source_ref="synthetic-source",
            local_result=self.primary,
            local_provenance=RecognitionProvenance(adapter_name="synthetic_primary"),
            local_metrics=RecognitionRunMetrics(duration_ms=3, json_compliant=True),
            secondary_result=None,
            secondary_provenance=None,
            secondary_metrics=None,
            remote_result=None,
            remote_provenance=None,
            remote_metrics=None,
            result=self.primary,
            decision=decision or self.decision,
            media=self.media,
            remote_used=False,
            created_at=self.now,
            evidence_binding=self.binding if binding else None,
        )
        values.update(overrides)
        self.database.insert_event(**values)

    def _sql(self, sql, values=()):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def _event(self):
        return self._sql("SELECT * FROM events WHERE event_id='event_1'")[0]

    def _complete(self, request="request_1", **changes):
        kwargs = dict(
            secondary_result=self.secondary,
            secondary_provenance=self.provenance,
            secondary_metrics=self.metrics,
            decision=self.after_decision,
            now=self.now + timedelta(seconds=1),
        )
        kwargs.update(changes)
        self.store.complete(request, **kwargs)

    def assertCode(self, code, call):
        with self.assertRaises(LocalReviewError) as caught:
            call()
        self.assertEqual(code, caught.exception.code)
        self.assertNotIn("/", str(caught.exception))

    def test_reserve_is_metadata_only_running_and_idempotent(self):
        before = self._event()
        with patch.object(EvidenceBinding, "verify", side_effect=AssertionError("no media reads")):
            first = self.store.reserve("event_1", "request_1", now=self.now)
            replay = self.store.reserve("event_1", "request_1", now=self.now)
        self.assertFalse(first.reused_existing)
        self.assertTrue(replay.reused_existing)
        self.assertEqual("running", replay.status)
        self.assertEqual(first.binding, self.binding)
        self.assertEqual(first.event.local_result, self.primary)
        self.assertEqual(before, self._event())
        self.assertEqual(1, len(self.store.recent()))

    def test_complete_updates_derived_columns_and_preserves_original_fields(self):
        before = self._event()
        self.store.reserve("event_1", "request_1", now=self.now)
        self._complete()
        after = self._event()
        preserved = (
            "event_id",
            "camera_id",
            "occurred_at",
            "source_kind",
            "source_ref",
            "candidate_json",
            "evidence_json",
            "clip_path",
            "created_at",
            "local_result_json",
            "local_provenance_json",
            "local_metrics_json",
            "remote_used",
            "remote_result_json",
            "remote_provenance_json",
            "remote_metrics_json",
        )
        for key in preserved:
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(json.loads(after["secondary_result_json"]), self.secondary.to_dict())
        self.assertEqual(json.loads(after["result_json"]), self.secondary.to_dict())
        self.assertEqual(after["risk_level"], "low")
        self.assertEqual(after["confidence"], 0.8)
        self.assertEqual(after["summary_zh"], self.secondary.summary_zh)
        self.assertEqual(
            json.loads(after["evidence_frames_json"]), list(self.secondary.evidence_frames)
        )
        ledger = self._sql("SELECT * FROM local_review_attempts")[0]
        self.assertEqual(ledger["before_result_json"], before["result_json"])
        self.assertEqual(ledger["after_result_json"], after["result_json"])
        self.assertEqual(ledger["after_provenance_json"], after["secondary_provenance_json"])
        self.assertEqual("succeeded", ledger["status"])
        replay = self.store.reserve("event_1", "request_1", now=self.now + timedelta(seconds=2))
        self.assertTrue(replay.reused_existing)
        self.assertEqual("succeeded", replay.status)
        self.store.fail("request_1", error_code="model_failed", now=self.now + timedelta(seconds=3))
        self.assertEqual("succeeded", self.store.recent()[0]["status"])
        self.assertCode("not_running", self._complete)

    def test_pending_remote_final_flag_does_not_change_raw_secondary(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        self._complete(decision=replace(self.after_decision, review_tier=ReviewTier.REMOTE))
        after = self._event()
        self.assertFalse(json.loads(after["secondary_result_json"])["needs_remote_review"])
        self.assertTrue(json.loads(after["result_json"])["needs_remote_review"])
        self.assertEqual(1, after["needs_remote_review"])
        self.assertEqual(0, after["remote_used"])

    def test_failed_idempotency_and_retry_with_new_request(self):
        before = self._event()
        self.store.reserve("event_1", "request_1", now=self.now)
        self.store.fail("request_1", error_code="/private/exception", now=self.now)
        replay = self.store.reserve("event_1", "request_1", now=self.now)
        self.assertEqual(
            (True, "failed", "review_failed"),
            (replay.reused_existing, replay.status, replay.error_code),
        )
        self.assertEqual(before, self._event())
        self.assertFalse(self.store.reserve("event_1", "request_2", now=self.now).reused_existing)

    def test_request_conflict_and_event_busy(self):
        self._insert("event_2")
        self.store.reserve("event_1", "request_1", now=self.now)
        self.assertCode(
            "request_conflict", lambda: self.store.reserve("event_2", "request_1", now=self.now)
        )
        self.assertCode(
            "event_busy", lambda: self.store.reserve("event_1", "request_2", now=self.now)
        )

    def test_concurrent_reservations_have_exactly_one_owner(self):
        barrier = Barrier(2)

        def reserve(request):
            barrier.wait()
            try:
                return self.store.reserve("event_1", request, now=self.now).status
            except LocalReviewError as error:
                return error.code

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(reserve, ("request_1", "request_2")))
        self.assertCountEqual(["running", "event_busy"], results)
        self.assertEqual(1, len(self.store.recent()))

    def test_expiry_is_a_write_fence_and_new_owner_can_reserve(self):
        before = self._event()
        self.store.reserve("event_1", "request_1", lease_seconds=1, now=self.now)
        self.assertCode("lease_expired", self._complete)
        self.assertEqual("lease_expired", self.store.recent()[0]["error_code"])
        self.assertEqual(before, self._event())
        self.store.reserve("event_1", "request_2", now=self.now + timedelta(seconds=1))
        self.assertCode("not_running", self._complete)
        self._complete("request_2", now=self.now + timedelta(seconds=2))

    def test_reserve_expires_old_owner_and_same_request_does_not_restart(self):
        self.store.reserve("event_1", "request_1", lease_seconds=1, now=self.now)
        replay = self.store.reserve("event_1", "request_1", now=self.now + timedelta(seconds=1))
        self.assertEqual(("failed", "lease_expired"), (replay.status, replay.error_code))
        self.store.reserve(
            "event_1", "request_2", lease_seconds=1, now=self.now + timedelta(seconds=1)
        )
        self.store.reserve("event_1", "request_3", now=self.now + timedelta(seconds=2))
        statuses = {row["request_id"]: row["status"] for row in self.store.recent()}
        self.assertEqual(
            {"request_1": "failed", "request_2": "failed", "request_3": "running"}, statuses
        )

    def test_entire_row_and_binding_cas_reject_even_unrelated_changes(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        self._sql("UPDATE events SET source_ref='changed' WHERE event_id='event_1'")
        before = self._event()
        self.assertCode("event_changed", self._complete)
        self.assertEqual(before, self._event())
        self._sql("UPDATE events SET source_ref='synthetic-source' WHERE event_id='event_1'")
        changed = replace(
            self.binding,
            assets=(replace(self.binding.assets[0], sha256="c" * 64), self.binding.assets[1]),
        )
        self._sql("UPDATE event_evidence_bindings SET binding_json=?", (changed.to_json(),))
        self.assertCode("event_changed", self._complete)
        self.assertEqual("running", self.store.recent()[0]["status"])

    def test_ledger_failure_rolls_back_event_update(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        before = self._event()
        self._sql(
            "CREATE TRIGGER reject_success BEFORE UPDATE ON local_review_attempts "
            "WHEN NEW.status='succeeded' "
            "BEGIN SELECT RAISE(ABORT,'/private/exception'); END"
        )
        self.assertCode("database_unavailable", self._complete)
        self.assertEqual(before, self._event())
        self.assertEqual("running", self.store.recent()[0]["status"])

    def test_ineligible_missing_event_and_historical_binding_required(self):
        self._insert("legacy", binding=False)
        self._insert(
            "no_review", decision=replace(self.decision, review_tier=ReviewTier.LOCAL_PRIMARY)
        )
        self._insert("already_secondary", secondary_result=self.secondary)
        self._insert("already_remote", remote_result=self.secondary)
        for event, code in (
            ("missing", "event_not_found"),
            ("legacy", "binding_required"),
            ("no_review", "not_eligible"),
            ("already_secondary", "not_eligible"),
            ("already_remote", "not_eligible"),
        ):
            with self.subTest(event=event):
                self.assertCode(
                    code, lambda event=event: self.store.reserve(event, "request_1", now=self.now)
                )
        self.assertEqual((), self.store.recent())

    def test_missing_database_is_never_created_and_recent_never_migrates(self):
        missing = self.path.parent / "missing.sqlite3"
        store = LocalReviewStore(missing)
        self.assertCode(
            "database_unavailable", lambda: store.reserve("event", "request", now=self.now)
        )
        self.assertCode("database_unavailable", store.recent)
        self.assertFalse(missing.exists())
        before = self.path.read_bytes()
        self.assertEqual((), self.store.recent())
        self.assertEqual(before, self.path.read_bytes())

    def test_input_completion_and_time_validation(self):
        for request in (True, "", "a" * 65, "../private", "非ASCII"):
            self.assertCode(
                "invalid_input", lambda request=request: self.store.reserve("event_1", request)
            )
        for lease in (True, 0, 86401, float("nan")):
            self.assertCode(
                "invalid_input",
                lambda lease=lease: self.store.reserve("event_1", "request_1", lease_seconds=lease),
            )
        self.assertCode(
            "invalid_input",
            lambda: self.store.reserve("event_1", "request_1", now=datetime(2026, 1, 1)),
        )
        self.store.reserve("event_1", "request_1", now=self.now)
        self.assertCode("invalid_completion", lambda: self._complete(secondary_metrics=None))
        self.assertCode("invalid_completion", lambda: self._complete(decision=self.decision))
        self.assertCode(
            "invalid_input", lambda: self._complete(now=self.now - timedelta(seconds=1))
        )
        self.assertEqual("running", self.store.recent()[0]["status"])
        for limit in (True, 0, 101):
            self.assertCode("invalid_input", lambda limit=limit: self.store.recent(limit))

    def test_recent_is_path_free_and_fixed_error_codes(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        self.store.fail("request_1", error_code="model_failed", now=self.now)
        record = self.store.recent()[0]
        self.assertEqual(
            {
                "request_id",
                "event_id",
                "status",
                "error_code",
                "requested_at",
                "completed_at",
                "lease_until",
            },
            set(record),
        )
        self.assertNotIn("/private", json.dumps(record))
        self.assertEqual("model_failed", record["error_code"])
        self.assertEqual("review_failed", LocalReviewError("/private/exception").code)

    def test_complete_cannot_change_original_consent_reasons_or_signal_context(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        invalid = (
            replace(self.after_decision, remote_allowed=True),
            replace(self.after_decision, review_reasons=()),
            replace(
                self.after_decision, signal_frequency_count=1, signal_frequency_window_seconds=60
            ),
        )
        before = self._event()
        for decision in invalid:
            with self.subTest(decision=decision):
                self.assertCode(
                    "invalid_completion",
                    lambda decision=decision: self._complete(decision=decision),
                )
        self.assertEqual(before, self._event())
        self.assertEqual("running", self.store.recent()[0]["status"])

    def test_complete_cannot_downgrade_existing_remote_review_tier(self):
        original = replace(self.decision, review_tier=ReviewTier.REMOTE)
        self._sql(
            "UPDATE events SET decision_json=? WHERE event_id='event_1'", (original.to_json(),)
        )
        self.store.reserve("event_1", "request_1", now=self.now)
        self.assertCode("invalid_completion", self._complete)

    def test_recent_and_replay_reject_corrupt_attempt_without_leaking_content(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        for key, value in (
            ("error_code", "/private/" * 100),
            ("completed_at", "/private/" * 100),
            ("lease_until", "not-a-time"),
            ("event_sha256", "not-a-hash"),
        ):
            original = self._sql("SELECT * FROM local_review_attempts")[0][key]
            self._sql(f"UPDATE local_review_attempts SET {key}=?", (value,))
            self.assertCode(
                "invalid_record", lambda: self.store.reserve("event_1", "request_1", now=self.now)
            )
            if key != "event_sha256":
                self.assertCode("invalid_record", self.store.recent)
            self._sql(f"UPDATE local_review_attempts SET {key}=?", (original,))

    def test_oversized_or_duplicate_event_json_is_rejected_and_no_ledger_created(self):
        original = self._event()["result_json"]
        for payload in (original[:-1] + ',"confidence":0.5}', "x" * (8 * 1024 * 1024 + 1)):
            self._sql("UPDATE events SET result_json=? WHERE event_id='event_1'", (payload,))
            self.assertCode(
                "invalid_record", lambda: self.store.reserve("event_1", "request_1", now=self.now)
            )
            self.assertEqual((), self.store.recent())
        self._sql("UPDATE events SET result_json=? WHERE event_id='event_1'", (original,))

    def test_legacy_missing_binding_table_does_not_create_binding_or_review_table(self):
        self._sql("DROP TABLE event_evidence_bindings")
        self.assertCode(
            "binding_required", lambda: self.store.reserve("event_1", "request_1", now=self.now)
        )
        tables = self._sql("SELECT name FROM sqlite_master WHERE type='table'")
        self.assertNotIn("local_review_attempts", [row["name"] for row in tables])
        self.assertNotIn("event_evidence_bindings", [row["name"] for row in tables])

    def test_fail_after_expiry_records_expiry_without_changing_event(self):
        before = self._event()
        self.store.reserve("event_1", "request_1", lease_seconds=1, now=self.now)
        self.store.fail("request_1", error_code="model_failed", now=self.now + timedelta(seconds=1))
        self.assertEqual("lease_expired", self.store.recent()[0]["error_code"])
        self.assertEqual(before, self._event())

    def _reserve_report(self, **overrides):
        args = dict(now=self.now, report_storage_root=self.path.parent, report_timezone="UTC")
        args.update(overrides)
        return self.store.reserve("event_1", "request_1", **args)

    def test_report_target_becomes_pending_only_with_atomic_success(self):
        self._reserve_report()
        self.assertEqual(
            {"status": "awaiting_review"}, self.store.report_refresh_state("request_1")
        )
        self.assertEqual((), self.store.pending_report_refreshes())
        self._complete()
        self.assertEqual({"status": "pending"}, self.store.report_refresh_state("request_1"))
        (intent,) = self.store.pending_report_refreshes(request_id="request_1")
        self.assertEqual("request_1", intent.request_id)
        self.assertEqual(date(2026, 9, 12), intent.local_date)
        self.assertEqual("UTC", intent.timezone_name)
        self.assertEqual(self.path.parent, intent.storage_root)
        self.assertNotIn(str(self.path.parent), repr(intent))
        self.assertEqual((), self.store.pending_report_refreshes(request_id="missing"))

    def test_report_intent_insert_failure_rolls_back_reservation(self):
        self.store.reserve("event_1", "old_request", now=self.now)
        self.store.fail("old_request", error_code="model_failed", now=self.now)
        self._sql(
            "CREATE TRIGGER reject_intent BEFORE INSERT ON local_review_report_refreshes "
            "BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
        )
        before = self._event()
        self.assertCode("database_unavailable", self._reserve_report)
        self.assertEqual(before, self._event())
        self.assertEqual(["old_request"], [row["request_id"] for row in self.store.recent()])
        self.assertEqual({"status": "not_requested"}, self.store.report_refresh_state("request_1"))

    def test_failed_completion_cannot_expose_pending_intent(self):
        self._reserve_report()
        self._sql(
            "CREATE TRIGGER reject_report_success BEFORE UPDATE ON local_review_attempts "
            "WHEN NEW.status='succeeded' BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
        )
        before = self._event()
        self.assertCode("database_unavailable", self._complete)
        self.assertEqual(before, self._event())
        self.assertEqual((), self.store.pending_report_refreshes())
        self.assertEqual(
            {"status": "awaiting_review"}, self.store.report_refresh_state("request_1")
        )
        self._sql("DROP TRIGGER reject_report_success")
        self._complete()
        self.assertEqual(1, len(self.store.pending_report_refreshes()))

    def test_failed_review_is_never_pending_or_implied_automatic_retry(self):
        self._reserve_report()
        self.store.fail("request_1", error_code="model_failed", now=self.now)
        self.assertEqual((), self.store.pending_report_refreshes())
        self.assertEqual({"status": "not_requested"}, self.store.report_refresh_state("request_1"))
        self.assertCode("not_running", lambda: self.store.mark_report_failed("request_1"))

    def test_report_target_conflict_and_omission_preserves_frozen_target(self):
        self._reserve_report()
        other = self.path.parent / "other"
        other.mkdir()
        self.assertCode("request_conflict", lambda: self._reserve_report(report_storage_root=other))
        self.assertCode(
            "request_conflict", lambda: self._reserve_report(report_timezone="Asia/Shanghai")
        )
        self.assertTrue(self.store.reserve("event_1", "request_1", now=self.now).reused_existing)
        self.assertTrue(self._reserve_report().reused_existing)
        self._complete()
        self.assertEqual(self.path.parent, self.store.pending_report_refreshes()[0].storage_root)

    def test_existing_request_without_target_is_not_backfilled(self):
        self.store.reserve("event_1", "request_1", now=self.now)
        self._complete()
        self.assertTrue(self._reserve_report().reused_existing)
        self.assertEqual({"status": "not_requested"}, self.store.report_refresh_state("request_1"))
        self.assertEqual((), self.store.pending_report_refreshes())
        self._sql("DROP TABLE local_review_report_refreshes")
        before = self.path.read_bytes()
        self.assertEqual({"status": "not_requested"}, self.store.report_refresh_state("request_1"))
        self.assertEqual((), self.store.pending_report_refreshes())
        self.assertEqual(before, self.path.read_bytes())

    def test_report_enqueue_failure_preserves_success_then_ack_is_idempotent(self):
        self._reserve_report()
        self._complete()
        event, ledger = self._event(), self._sql("SELECT * FROM local_review_attempts")
        self.store.mark_report_failed("request_1", error_code="/private/failure")
        self.assertEqual(
            {"status": "pending", "error_code": "report_enqueue_failed"},
            self.store.report_refresh_state("request_1"),
        )
        job_id = "report_" + "a" * 64
        self.store.mark_report_enqueued(
            "request_1", job_id=job_id, now=self.now + timedelta(seconds=2)
        )
        self.store.mark_report_enqueued(
            "request_1", job_id=job_id, now=self.now + timedelta(seconds=3)
        )
        self.store.mark_report_failed("request_1")
        self.assertEqual(
            {"status": "queued", "job_id": job_id}, self.store.report_refresh_state("request_1")
        )
        self.assertEqual((), self.store.pending_report_refreshes())
        self.assertCode(
            "request_conflict",
            lambda: self.store.mark_report_enqueued("request_1", job_id="report_" + "b" * 64),
        )
        self.assertEqual(event, self._event())
        self.assertEqual(ledger, self._sql("SELECT * FROM local_review_attempts"))

    def test_report_timezone_uses_event_day_not_completion_day(self):
        occurred = datetime(2026, 11, 1, 3, 30, tzinfo=UTC)
        self._sql("UPDATE events SET occurred_at=?", (occurred.isoformat(),))
        self._reserve_report(report_timezone="America/New_York")
        self._complete()
        (intent,) = self.store.pending_report_refreshes()
        self.assertEqual(date(2026, 10, 31), intent.local_date)
        self.assertEqual("America/New_York", intent.timezone_name)

    def test_report_target_validated_before_reservation_and_model(self):
        bad = (
            {"report_storage_root": None},
            {"report_timezone": None},
            {"report_timezone": "Invalid/Private"},
            {"report_storage_root": Path("relative")},
        )
        for args in bad:
            with self.subTest(args=args):
                self.assertCode("invalid_input", lambda args=args: self._reserve_report(**args))
        link = self.path.parent / "linked"
        link.symlink_to(self.path.parent, target_is_directory=True)
        for root in (link, self.path.parent / "missing"):
            self.assertCode(
                "invalid_record", lambda root=root: self._reserve_report(report_storage_root=root)
            )
        self.assertEqual((), self.store.recent())

    def test_report_readers_missing_database_do_not_create_it(self):
        missing = self.path.parent / "missing.sqlite3"
        store = LocalReviewStore(missing)
        self.assertEqual({"status": "not_requested"}, store.report_refresh_state("request_1"))
        self.assertEqual((), store.pending_report_refreshes())
        self.assertFalse(missing.exists())

    def test_report_corrupt_paths_dates_and_ids_fail_closed_with_bounded_reads(self):
        self._reserve_report()
        self._complete()
        for key, value in (
            ("storage_root", "/" + "x" * 17000),
            ("storage_root", "../private"),
            ("timezone_name", "x" * 600),
            ("local_date", "2026-13-99"),
            ("last_error_code", "/private/error"),
        ):
            original = self._sql("SELECT * FROM local_review_report_refreshes")[0][key]
            self._sql(f"UPDATE local_review_report_refreshes SET {key}=?", (value,))
            self.assertCode("invalid_record", self.store.pending_report_refreshes)
            self.assertCode("invalid_record", lambda: self.store.report_refresh_state("request_1"))
            self._sql(f"UPDATE local_review_report_refreshes SET {key}=?", (original,))
        for bad_id in ("report_" + "a" * 65, "../private", None):
            self.assertCode(
                "invalid_input",
                lambda bad_id=bad_id: self.store.mark_report_enqueued("request_1", job_id=bad_id),
            )
        for limit in (True, 0, 101):
            self.assertCode(
                "invalid_input", lambda limit=limit: self.store.pending_report_refreshes(limit)
            )

    def test_report_pending_rotation_does_not_starve_twenty_first_request(self):
        self._reserve_report()
        self._complete()
        original = self._sql("SELECT * FROM local_review_attempts")[0]
        columns = tuple(original)
        names = ",".join(columns)
        placeholders = ",".join("?" for _ in columns)
        # Metadata-only terminal fixtures: no additional model calls or media reads.
        for index in range(20):
            row = {**original, "request_id": f"rotation_{index:02d}"}
            self._sql(
                f"INSERT INTO local_review_attempts ({names}) VALUES ({placeholders})",
                tuple(row[key] for key in columns),
            )
            self._sql(
                "INSERT INTO local_review_report_refreshes "
                "(request_id,storage_root,timezone_name,local_date) VALUES (?,?,?,?)",
                (row["request_id"], str(self.path.parent), "UTC", "2026-09-12"),
            )
        first = self.store.pending_report_refreshes(limit=20)
        self.assertEqual(20, len(first))
        seen = {intent.request_id for intent in first}
        for intent in first:
            self.store.mark_report_failed(intent.request_id, now=self.now + timedelta(seconds=2))
        second = self.store.pending_report_refreshes(limit=20)
        self.assertNotIn(second[0].request_id, seen)
        self.assertEqual(21, len(seen | {intent.request_id for intent in second}))
        self.assertTrue(all(row["status"] == "succeeded" for row in self.store.recent(limit=100)))

    def test_report_failure_time_is_validated_and_corrupt_stored_clock_rejected(self):
        self._reserve_report()
        self._complete()
        for timestamp in (datetime(2026, 1, 1), self.now):
            self.assertCode(
                "invalid_input",
                lambda timestamp=timestamp: self.store.mark_report_failed(
                    "request_1", now=timestamp
                ),
            )
        self._sql("UPDATE local_review_report_refreshes SET last_attempt_at=?", ("/private/time",))
        self.assertCode("invalid_record", self.store.pending_report_refreshes)
        self.assertCode("invalid_record", lambda: self.store.report_refresh_state("request_1"))

    def test_report_ack_requires_success_and_is_transactional(self):
        self._reserve_report()
        job_id = "report_" + "a" * 64
        self.assertCode(
            "not_running",
            lambda: self.store.mark_report_enqueued("request_1", job_id=job_id, now=self.now),
        )
        self._complete()
        self._sql(
            "CREATE TRIGGER reject_ack BEFORE UPDATE ON local_review_report_refreshes "
            "BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
        )
        before = self._event()
        self.assertCode(
            "database_unavailable",
            lambda: self.store.mark_report_enqueued(
                "request_1", job_id=job_id, now=self.now + timedelta(seconds=2)
            ),
        )
        self.assertEqual({"status": "pending"}, self.store.report_refresh_state("request_1"))
        self.assertEqual(before, self._event())


if __name__ == "__main__":
    unittest.main()
