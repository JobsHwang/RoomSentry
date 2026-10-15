from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    RecognitionProvenance,
    RecognitionRunMetrics,
    SchemaError,
)
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver


class RecognitionProvenanceTests(unittest.TestCase):
    def test_contract_is_strict_path_free_and_matches_public_schema_shape(self) -> None:
        provenance = RecognitionProvenance(
            adapter_name="minicpm_v46",
            model_revision="abc123def456",
        )
        payload = provenance.to_dict()
        schema = json.loads(
            Path("schemas/recognition-provenance-v1.schema.json").read_text(encoding="utf-8")
        )

        self.assertEqual(set(schema["required"]), set(payload))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(provenance, RecognitionProvenance.from_json(provenance.to_json()))
        with self.assertRaisesRegex(SchemaError, "model_revision"):
            RecognitionProvenance(
                adapter_name="minicpm_v46",
                model_revision="/private/model/path",
            )
        invalid = payload | {"private_model_path": "/not-allowed"}
        with self.assertRaisesRegex(SchemaError, "invalid keys"):
            RecognitionProvenance.from_dict(invalid)

    def test_run_metrics_are_strict_nullable_and_match_public_schema_shape(self) -> None:
        metrics = RecognitionRunMetrics(
            duration_ms=1234,
            json_compliant=True,
            process_rss_peak_kib=45678,
            gpu_device_memory_peak_mib=None,
        )
        payload = metrics.to_dict()
        schema = json.loads(
            Path("schemas/recognition-run-metrics-v1.schema.json").read_text(encoding="utf-8")
        )

        self.assertEqual(set(schema["required"]), set(payload))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(metrics, RecognitionRunMetrics.from_json(metrics.to_json()))
        with self.assertRaisesRegex(SchemaError, "duration_ms"):
            RecognitionRunMetrics(duration_ms=-1, json_compliant=True)
        with self.assertRaisesRegex(SchemaError, "json_compliant"):
            RecognitionRunMetrics(duration_ms=1, json_compliant=1)  # type: ignore[arg-type]

    def test_pipeline_persists_provenance_and_legacy_null_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            frame = root / "frame.jpg"
            frame.write_bytes(b"synthetic-fixture")
            clip = root / "clip.mp4"
            clip.write_bytes(b"synthetic-fixture")
            candidate = CandidateEvent("provenance_event", "cam_01", 0.0, 1.0, ())
            process_prepared_event(
                candidate=candidate,
                occurred_at=datetime(2026, 8, 3, 1, 0, tzinfo=UTC),
                source_kind="fixture",
                source_ref="synthetic-provenance-fixture",
                media=EventMedia(str(clip), (str(frame),)),
                local_vlm=MockVLMAdapter(),
                database=database,
                policy=ReviewPolicy(confidence_threshold=0.0),
            )

            event = database.get_event(candidate.event_id)
            assert event is not None
            self.assertTrue(event.provenance_metadata_present)
            self.assertEqual("mock", event.local_provenance.adapter_name)
            self.assertEqual("builtin_mock_v1", event.local_provenance.model_revision)
            self.assertTrue(event.metrics_metadata_present)
            self.assertIsNotNone(event.local_metrics)
            assert event.local_metrics is not None
            self.assertGreaterEqual(event.local_metrics.duration_ms, 0)
            self.assertTrue(event.local_metrics.json_compliant)
            self.assertIsNone(event.local_metrics.process_rss_peak_kib)
            self.assertIsNone(event.local_metrics.gpu_device_memory_peak_mib)
            self.assertNotIn(str(root), event.local_provenance.to_json())

            storage = StorageResolver(
                StorageSettings(local_root=root / "storage", database=database.path)
            )
            report = DailyReportGenerator(
                database=database,
                storage=storage,
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 3))
            for content in (
                report.markdown_path.read_text(encoding="utf-8"),
                report.html_path.read_text(encoding="utf-8"),
            ):
                self.assertIn("mock @ builtin_mock_v1", content)
                self.assertIn("baby_monitor_vlm_prompt_v3_coverage_unverified", content)
                self.assertIn("vlm_event_v1", content)
                self.assertIn("JSON 合规", content)
                self.assertIn("RSS 未测量", content)
                self.assertIn("GPU 峰值未测量", content)
                self.assertNotIn(str(root), content)

            with closing(sqlite3.connect(database.path)) as connection, connection:
                connection.execute(
                    "UPDATE events SET local_provenance_json = NULL WHERE event_id = ?",
                    (candidate.event_id,),
                )
            legacy = database.get_event(candidate.event_id)
            assert legacy is not None
            self.assertFalse(legacy.provenance_metadata_present)
            self.assertEqual("legacy_unknown", legacy.local_provenance.adapter_name)
            self.assertIsNone(legacy.local_provenance.model_revision)

            with closing(sqlite3.connect(database.path)) as connection, connection:
                connection.execute(
                    "UPDATE events SET local_metrics_json = NULL WHERE event_id = ?",
                    (candidate.event_id,),
                )
            metrics_legacy = database.get_event(candidate.event_id)
            assert metrics_legacy is not None
            self.assertFalse(metrics_legacy.metrics_metadata_present)
            self.assertIsNone(metrics_legacy.local_metrics)


if __name__ == "__main__":
    unittest.main()
