from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.adapters.vlm import MockVLMAdapter, VLMAdapter, VLMError
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, EventMedia, RecognitionRunMetrics
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)


class FailingVLM(VLMAdapter):
    name = "failing_fixture"

    def analyze(self, request):  # type: ignore[no-untyped-def]
        raise VLMError(
            "private runner output must not be recorded",
            code="invalid_structured_output",
            metrics=RecognitionRunMetrics(
                duration_ms=123,
                json_compliant=False,
                process_rss_peak_kib=456,
                gpu_device_memory_peak_mib=789,
            ),
        )


class RecognitionAttemptTests(unittest.TestCase):
    @staticmethod
    def _event(root: Path, event_id: str) -> tuple[CandidateEvent, EventMedia]:
        return (
            CandidateEvent(event_id, "cam_01", 0.0, 1.0, ()),
            EventMedia(
                str(root / "private-event.mp4"),
                (str(root / "private-frame.jpg"),),
            ),
        )

    def test_store_lifecycle_is_strict_path_free_and_matches_schema_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            database.initialize()
            store = RecognitionAttemptAuditStore(database.path)
            started = datetime(2026, 8, 3, 1, 0, tzinfo=UTC)
            adapter = MockVLMAdapter(name="fixture")
            attempt_id = store.start(
                event_id="event_01",
                stage=RecognitionStage.LOCAL_PRIMARY,
                provenance=adapter.provenance,
                started_at=started,
            )
            store.complete(
                attempt_id,
                status=RecognitionAttemptStatus.SUCCEEDED,
                completed_at=started + timedelta(milliseconds=12),
                metrics=RecognitionRunMetrics(duration_ms=12, json_compliant=True),
            )

            attempt = store.recent()[0]
            public = attempt.to_public_dict()
            schema = json.loads(
                Path("schemas/recognition-attempt-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(schema["required"]), set(public))
            self.assertEqual("succeeded", public["status"])
            self.assertEqual(1, len(store.between(started, started + timedelta(days=1))))
            serialized = json.dumps(public, ensure_ascii=False)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private", serialized)

            with self.assertRaisesRegex(ValueError, "safe error code"):
                failed_id = store.start(
                    event_id="event_02",
                    stage=RecognitionStage.REMOTE,
                    provenance=adapter.provenance,
                    started_at=started,
                )
                store.complete(
                    failed_id,
                    status=RecognitionAttemptStatus.FAILED,
                    completed_at=started,
                    metrics=RecognitionRunMetrics(
                        duration_ms=1,
                        json_compliant=False,
                    ),
                    error_code="unsafe/path",
                )

    def test_pipeline_records_failed_and_successful_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            candidate, media = self._event(root, "failed_event")
            with self.assertRaisesRegex(VLMError, "private runner output"):
                process_prepared_event(
                    candidate=candidate,
                    occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                    source_kind="fixture",
                    source_ref="fixture",
                    media=media,
                    local_vlm=FailingVLM(),
                    database=database,
                    policy=ReviewPolicy(),
                )

            second_candidate, second_media = self._event(root, "success_event")
            process_prepared_event(
                candidate=second_candidate,
                occurred_at=datetime(2026, 8, 3, 0, 1, tzinfo=UTC),
                source_kind="fixture",
                source_ref="fixture",
                media=second_media,
                local_vlm=MockVLMAdapter(),
                database=database,
                policy=ReviewPolicy(),
            )

            attempts = RecognitionAttemptAuditStore(database.path).recent()
            self.assertEqual(2, len(attempts))
            by_event = {attempt.event_id: attempt for attempt in attempts}
            failed = by_event["failed_event"]
            self.assertEqual(RecognitionAttemptStatus.FAILED, failed.status)
            self.assertEqual("invalid_structured_output", failed.error_code)
            assert failed.metrics is not None
            self.assertFalse(failed.metrics.json_compliant)
            self.assertEqual(123, failed.metrics.duration_ms)
            self.assertEqual(456, failed.metrics.process_rss_peak_kib)
            self.assertEqual(789, failed.metrics.gpu_device_memory_peak_mib)
            succeeded = by_event["success_event"]
            self.assertEqual(RecognitionAttemptStatus.SUCCEEDED, succeeded.status)
            assert succeeded.metrics is not None
            self.assertTrue(succeeded.metrics.json_compliant)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "recognition-audit-status",
                        "--database",
                        str(database.path),
                        "--limit",
                        "20",
                    ]
                )
            self.assertEqual(0, code)
            payload = json.loads(output.getvalue())
            self.assertEqual({"running": 0, "succeeded": 1, "failed": 1}, payload["counts"])
            self.assertEqual(0.5, payload["json_compliance_rate"])
            self.assertFalse(payload["contains_media_paths"])
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("private-frame", serialized)
            self.assertNotIn("private runner output", serialized)


if __name__ == "__main__":
    unittest.main()
