from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from baby_monitor.database import SCHEMA_SQL, EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    RecognitionProvenance,
    RecognitionRunMetrics,
)
from baby_monitor.evidence_binding import (
    MAX_BINDING_BYTES,
    EvidenceAssetBinding,
    EvidenceBinding,
    EvidenceBindingError,
    media_fingerprint,
)
from baby_monitor.policy import ReviewPolicy
from tests.test_policy import _result


class DatabaseEvidenceBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = EventDatabase(self.root / "events.sqlite3")
        # Deliberately absent: this suite verifies metadata storage, not bytes.
        self.media = EventMedia(str(self.root / "absent.mp4"), (str(self.root / "absent.jpg"),))
        self.binding = EvidenceBinding(
            media_sha256=media_fingerprint(self.media),
            assets=(
                EvidenceAssetBinding("clip", 0, 1, "a" * 64),
                EvidenceAssetBinding("frame", 0, 1, "b" * 64),
            ),
        )

    def _insert(self, *, event_id="binding_event", binding=None) -> None:
        result = _result()
        now = datetime(2026, 9, 12, tzinfo=UTC)
        self.database.insert_event(
            candidate=CandidateEvent(event_id, "cam_01", 0.0, 1.0, ()),
            occurred_at=now,
            source_kind="fixture",
            source_ref="synthetic-binding-test",
            local_result=result,
            local_provenance=RecognitionProvenance(adapter_name="synthetic"),
            local_metrics=RecognitionRunMetrics(duration_ms=1, json_compliant=True),
            secondary_result=None,
            secondary_provenance=None,
            secondary_metrics=None,
            remote_result=None,
            remote_provenance=None,
            remote_metrics=None,
            result=result,
            decision=ReviewPolicy().decide(result),
            media=self.media,
            remote_used=False,
            created_at=now,
            evidence_binding=binding,
        )

    def _execute(self, statement: str, parameters=()) -> None:
        with closing(sqlite3.connect(self.database.path)) as connection, connection:
            connection.execute(statement, parameters)

    def _row_count(self, table: str) -> int:
        self.assertIn(table, ("events", "event_evidence_bindings"))
        with closing(sqlite3.connect(self.database.path)) as connection:
            return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_binding_and_event_round_trip_without_reading_any_media(self) -> None:
        self.database.initialize()
        with (
            patch(
                "baby_monitor.evidence_binding.capture_evidence_binding",
                side_effect=AssertionError("media read"),
            ),
            patch.object(
                EvidenceBinding, "verify", side_effect=AssertionError("media verification")
            ),
            patch("os.open", side_effect=AssertionError("asset descriptor opened")),
        ):
            self._insert(binding=self.binding)
            self.assertEqual(self.binding, self.database.get_evidence_binding("binding_event"))
        self.assertEqual(1, self._row_count("events"))
        self.assertEqual(1, self._row_count("event_evidence_bindings"))
        self.assertFalse(Path(self.media.clip_path).exists())
        self.assertFalse(Path(self.media.frame_paths[0]).exists())

    def test_binding_insert_failure_rolls_back_event_in_same_transaction(self) -> None:
        self.database.initialize()
        self._execute(
            "CREATE TRIGGER reject_binding BEFORE INSERT ON event_evidence_bindings "
            "BEGIN SELECT RAISE(ABORT, 'synthetic binding insert failure'); END"
        )
        with self.assertRaises((sqlite3.DatabaseError, EvidenceBindingError)):
            self._insert(binding=self.binding)
        self.assertEqual(0, self._row_count("events"))
        self.assertEqual(0, self._row_count("event_evidence_bindings"))

    def test_wrong_media_fingerprint_rejected_without_event_or_binding(self) -> None:
        self.database.initialize()
        with self.assertRaises(EvidenceBindingError):
            self._insert(binding=replace(self.binding, media_sha256="0" * 64))
        self.assertEqual(0, self._row_count("events"))
        self.assertEqual(0, self._row_count("event_evidence_bindings"))

    def _media_with_serialized_bytes(self, size: int) -> EventMedia:
        empty = replace(self.media, clip_path="")
        baseline = len(json.dumps(empty.to_dict(), ensure_ascii=False).encode("utf-8"))
        return replace(empty, clip_path="/" + "x" * (size - baseline - 1))

    def test_oversized_event_metadata_rejected_at_insert_without_partial_rows(self) -> None:
        self.database.initialize()
        cases = (
            self._media_with_serialized_bytes(65537),
            replace(self.media, clip_path="/" + "中" * 22000),
        )
        for index, media in enumerate(cases):
            with self.subTest(case=index):
                self.assertGreater(
                    len(json.dumps(media.to_dict(), ensure_ascii=False).encode("utf-8")), 65536
                )
                self.media = media
                binding = replace(self.binding, media_sha256=media_fingerprint(media))
                with self.assertRaises(EvidenceBindingError):
                    self._insert(binding=binding)
                self.assertEqual(0, self._row_count("events"))
                self.assertEqual(0, self._row_count("event_evidence_bindings"))

    def test_event_metadata_at_and_below_64kib_limit_round_trips(self) -> None:
        self.database.initialize()
        for size in (65535, 65536):
            with self.subTest(size=size):
                self.media = self._media_with_serialized_bytes(size)
                self.assertEqual(
                    size, len(json.dumps(self.media.to_dict(), ensure_ascii=False).encode("utf-8"))
                )
                binding = replace(self.binding, media_sha256=media_fingerprint(self.media))
                self._insert(event_id=f"size_{size}", binding=binding)
                self.assertEqual(binding, self.database.get_evidence_binding(f"size_{size}"))

    def test_matching_metadata_hash_does_not_allow_extra_or_missing_frame_assets(self) -> None:
        self.database.initialize()
        original_media = self.media
        extra = EvidenceAssetBinding("frame", 1, 1, "c" * 64)
        cases = (
            (original_media, replace(self.binding, assets=(*self.binding.assets, extra))),
            (
                replace(original_media, frame_paths=(*original_media.frame_paths, "extra.jpg")),
                self.binding,
            ),
        )
        for index, (media, binding) in enumerate(cases):
            with self.subTest(case=index):
                self.media = media
                mismatched = replace(binding, media_sha256=media_fingerprint(media))
                with self.assertRaises(EvidenceBindingError):
                    self._insert(binding=mismatched)
                self.assertEqual(0, self._row_count("events"))
                self.assertEqual(0, self._row_count("event_evidence_bindings"))

    def test_stored_asset_count_mismatch_is_rejected_even_with_matching_media_hash(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        mismatched = replace(
            self.binding,
            assets=(*self.binding.assets, EvidenceAssetBinding("frame", 1, 1, "c" * 64)),
        )
        self._execute(
            "UPDATE event_evidence_bindings SET binding_json = ?", (mismatched.to_json(),)
        )
        with self.assertRaises(EvidenceBindingError):
            self.database.get_evidence_binding("binding_event")

    def test_optional_binding_does_not_invent_row_or_retroactively_hash_media(self) -> None:
        self.database.initialize()
        self._insert()
        self.assertIsNone(self.database.get_evidence_binding("binding_event"))
        self.assertIsNone(self.database.get_evidence_binding("missing_event"))
        self.assertEqual(1, self._row_count("events"))
        self.assertEqual(0, self._row_count("event_evidence_bindings"))

    def test_legacy_database_missing_binding_table_is_not_migrated_on_read(self) -> None:
        with closing(sqlite3.connect(self.database.path)) as connection:
            connection.executescript(SCHEMA_SQL)
            connection.execute("DROP TABLE IF EXISTS event_evidence_bindings")
            connection.commit()
        before = hashlib.sha256(self.database.path.read_bytes()).hexdigest()
        with patch.object(EventDatabase, "initialize", side_effect=AssertionError("migration")):
            self.assertIsNone(self.database.get_evidence_binding("legacy_event"))
        self.assertEqual(before, hashlib.sha256(self.database.path.read_bytes()).hexdigest())
        with closing(sqlite3.connect(self.database.path)) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'event_evidence_bindings'"
                ).fetchone()
            )

    def test_missing_database_getter_does_not_create_database(self) -> None:
        self.assertIsNone(self.database.get_evidence_binding("missing_event"))
        self.assertFalse(self.database.path.exists())

    def test_corrupt_oversized_duplicate_and_nonobject_bindings_fail_closed(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        bad_payloads = (
            "PRIVATE_CORRUPT_BINDING_PAYLOAD",
            self.binding.to_json() + " " * MAX_BINDING_BYTES,
            self.binding.to_json().replace('"version":1', '"version":1,"version":1'),
            "[]",
            "null",
            json.dumps({**self.binding.to_dict(), "unexpected": True}),
        )
        for index, payload in enumerate(bad_payloads):
            with self.subTest(case=index):
                self._execute("UPDATE event_evidence_bindings SET binding_json = ?", (payload,))
                with self.assertRaises(EvidenceBindingError) as raised:
                    self.database.get_evidence_binding("binding_event")
                self.assertNotIn("PRIVATE_CORRUPT", str(raised.exception))
                self.assertNotIn(str(self.root), str(raised.exception))

    def test_sql_null_and_blob_binding_payloads_fail_closed(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        # A corrupt legacy table can bypass the normal NOT NULL constraint.
        self._execute("DROP TABLE event_evidence_bindings")
        self._execute(
            "CREATE TABLE event_evidence_bindings(event_id TEXT PRIMARY KEY, binding_json)"
        )
        self._execute("INSERT INTO event_evidence_bindings VALUES (?, NULL)", ("binding_event",))
        for payload in (None, self.binding.to_json().encode()):
            with self.subTest(kind=type(payload).__name__):
                self._execute("UPDATE event_evidence_bindings SET binding_json = ?", (payload,))
                with self.assertRaises(EvidenceBindingError):
                    self.database.get_evidence_binding("binding_event")

    def test_invalid_utf8_sql_text_is_sanitized_before_leaving_getter(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        for table, column in (
            ("event_evidence_bindings", "binding_json"),
            ("events", "evidence_json"),
        ):
            with self.subTest(column=column):
                self._execute(
                    "UPDATE event_evidence_bindings SET binding_json = ?",
                    (self.binding.to_json(),),
                )
                self._execute(
                    "UPDATE events SET evidence_json = ?", (json.dumps(self.media.to_dict()),)
                )
                self._execute(
                    f"UPDATE {table} SET {column} = CAST(? AS TEXT)",
                    (b"PRIVATE_CORRUPT_UTF8\xff",),
                )
                with self.assertRaises(EvidenceBindingError) as raised:
                    self.database.get_evidence_binding("binding_event")
                self.assertNotIn("PRIVATE_CORRUPT", str(raised.exception))

    def test_changed_media_metadata_is_not_silently_rebound(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        modified = replace(self.media, selected_segments=2)
        self._execute("UPDATE events SET evidence_json = ?", (json.dumps(modified.to_dict()),))
        with self.assertRaises(EvidenceBindingError):
            self.database.get_evidence_binding("binding_event")
        with closing(sqlite3.connect(self.database.path)) as connection:
            payload = connection.execute(
                "SELECT binding_json FROM event_evidence_bindings"
            ).fetchone()[0]
        self.assertEqual(self.binding, EvidenceBinding.from_json(payload))

    def test_missing_or_corrupt_event_metadata_does_not_fall_back_to_legacy_paths(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        media_json = json.dumps(self.media.to_dict())
        payloads = (
            None,
            "{}",
            "not-json",
            "null",
            media_json.encode(),
            media_json + " " * 65536,
            media_json.replace('"version": 1', '"version": 1, "version": 1'),
        )
        for index, payload in enumerate(payloads):
            with self.subTest(case=index):
                self._execute("UPDATE events SET evidence_json = ?", (payload,))
                with self.assertRaises(EvidenceBindingError):
                    self.database.get_evidence_binding("binding_event")

    def test_orphan_binding_is_corruption_not_an_unbound_event(self) -> None:
        self.database.initialize()
        self._insert(binding=self.binding)
        # Direct SQLite connection has foreign keys off, simulating corruption.
        self._execute("DELETE FROM events")
        with self.assertRaises(EvidenceBindingError):
            self.database.get_evidence_binding("binding_event")

    def test_binding_table_has_event_foreign_key(self) -> None:
        self.database.initialize()
        with (
            closing(self.database._connect()) as connection,
            connection,
            self.assertRaises(sqlite3.IntegrityError),
        ):
            connection.execute(
                "INSERT INTO event_evidence_bindings(event_id, binding_json) VALUES (?, ?)",
                ("missing_event", self.binding.to_json()),
            )


if __name__ == "__main__":
    unittest.main()
