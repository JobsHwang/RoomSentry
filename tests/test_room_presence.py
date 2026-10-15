from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.room_presence import (
    RoomPresenceError,
    RoomPresencePolicy,
    RoomPresenceReport,
    RoomPresenceStatus,
    StreamPresenceBatch,
    StreamPresenceSample,
    StreamPresenceSampleState,
    evaluate_room_presence,
)
from tests.test_room_registry import fixture_registry


class RoomPresenceTests(unittest.TestCase):
    NOW = datetime(2026, 8, 20, 4, 0, tzinfo=UTC)
    DETECTOR = "yolo_command_v1_fixture"

    @classmethod
    def _sample(
        cls,
        camera: int,
        seconds: int,
        state: StreamPresenceSampleState,
        *,
        sample_id: str | None = None,
    ) -> StreamPresenceSample:
        return StreamPresenceSample(
            sample_id=sample_id or f"sample_{camera}_{seconds}",
            camera_id=f"camera_{camera:02d}",
            observed_at=cls.NOW + timedelta(seconds=seconds),
            state=state,
            scanner_id="candidate_v1_fixture",
            detector_id=(None if state is StreamPresenceSampleState.COVERAGE_GAP else cls.DETECTOR),
            source_fingerprint=f"{camera:064x}",
            reason_code=(
                "source_unavailable" if state is StreamPresenceSampleState.COVERAGE_GAP else None
            ),
        )

    @staticmethod
    def _batch(*samples: StreamPresenceSample) -> StreamPresenceBatch:
        return StreamPresenceBatch(
            samples=tuple(
                sorted(
                    samples,
                    key=lambda item: (item.observed_at, item.camera_id, item.sample_id),
                )
            )
        )

    def _report(
        self,
        *samples: StreamPresenceSample,
        seconds: int = 30,
        policy: RoomPresencePolicy | None = None,
    ) -> RoomPresenceReport:
        return evaluate_room_presence(
            registry=fixture_registry(),
            batch=self._batch(*samples),
            checked_at=self.NOW + timedelta(seconds=seconds),
            policy=policy,
        )

    def test_two_c500_views_can_report_independent_rooms_without_identity(self) -> None:
        report = self._report(
            self._sample(1, 0, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(1, 10, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(2, 20, StreamPresenceSampleState.COVERAGE_GAP),
        )

        room = report.rooms[0]
        self.assertEqual(RoomPresenceStatus.PERSON_PRESENT, room.status)
        self.assertEqual(1, room.configured_stream_count)
        self.assertEqual(1, room.confirmed_person_stream_count)
        self.assertEqual(0, room.coverage_gap_stream_count)
        self.assertEqual(self.NOW + timedelta(seconds=10), room.last_confirmed_person_at)
        self.assertEqual(("person_confirmed",), room.reason_codes)
        self.assertEqual(RoomPresenceStatus.COVERAGE_GAP, report.rooms[1].status)
        self.assertEqual(("no_samples", "stream_coverage_gap"), report.rooms[1].reason_codes)
        self.assertFalse(report.identity_inference)
        self.assertFalse(report.risk_assessment)
        self.assertFalse(report.notification_authority)

    def test_one_positive_is_unknown_and_never_called_baby(self) -> None:
        report = self._report(
            self._sample(1, 10, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(2, 10, StreamPresenceSampleState.CLEAR),
        )

        room = report.rooms[0]
        self.assertEqual(RoomPresenceStatus.UNKNOWN, room.status)
        self.assertEqual(("confirmation_pending",), room.reason_codes)
        self.assertIsNone(room.last_confirmed_person_at)
        self.assertNotIn("baby", json.dumps(report.to_dict(), ensure_ascii=False).lower())

    def test_presence_hysteresis_requires_three_clear_scans_on_every_view(self) -> None:
        base = (
            self._sample(2, 0, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(2, 10, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(2, 20, StreamPresenceSampleState.CLEAR),
            self._sample(2, 30, StreamPresenceSampleState.CLEAR),
            self._sample(3, 20, StreamPresenceSampleState.CLEAR),
            self._sample(3, 30, StreamPresenceSampleState.CLEAR),
        )
        still_present = self._report(*base, seconds=35)
        self.assertEqual(RoomPresenceStatus.PERSON_PRESENT, still_present.rooms[1].status)

        one_stream_clear = self._report(
            *base,
            self._sample(2, 40, StreamPresenceSampleState.CLEAR),
            seconds=45,
        )
        self.assertEqual(RoomPresenceStatus.UNKNOWN, one_stream_clear.rooms[1].status)

        all_clear = self._report(
            *base,
            self._sample(2, 40, StreamPresenceSampleState.CLEAR),
            self._sample(3, 40, StreamPresenceSampleState.CLEAR),
            seconds=45,
        )
        room = all_clear.rooms[1]
        self.assertEqual(RoomPresenceStatus.TEMPORARILY_CLEAR, room.status)
        self.assertEqual(("all_streams_clear",), room.reason_codes)
        self.assertIsNone(room.last_confirmed_person_at)

    def test_missing_stale_and_explicit_gap_are_coverage_gaps(self) -> None:
        report = self._report(
            self._sample(1, 0, StreamPresenceSampleState.CLEAR),
            self._sample(2, 0, StreamPresenceSampleState.CLEAR),
            seconds=100,
        )
        room = report.rooms[0]
        self.assertEqual(RoomPresenceStatus.COVERAGE_GAP, room.status)
        self.assertEqual(("stream_stale",), room.reason_codes)

        no_samples = self._report(seconds=0)
        self.assertTrue(
            all(item.status is RoomPresenceStatus.COVERAGE_GAP for item in no_samples.rooms)
        )
        self.assertTrue(all(item.reason_codes == ("no_samples",) for item in no_samples.rooms))

        explicit = self._report(
            self._sample(4, 0, StreamPresenceSampleState.COVERAGE_GAP),
            seconds=1,
        )
        self.assertEqual(RoomPresenceStatus.COVERAGE_GAP, explicit.rooms[2].status)
        self.assertEqual(("stream_coverage_gap",), explicit.rooms[2].reason_codes)

    def test_rooms_are_independent_and_multiple_rooms_can_be_present(self) -> None:
        report = self._report(
            self._sample(1, 0, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(1, 10, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(3, 0, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(3, 10, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(4, 0, StreamPresenceSampleState.COVERAGE_GAP),
            seconds=20,
        )

        self.assertEqual(RoomPresenceStatus.PERSON_PRESENT, report.rooms[0].status)
        self.assertEqual(RoomPresenceStatus.PERSON_PRESENT, report.rooms[1].status)
        self.assertEqual(RoomPresenceStatus.COVERAGE_GAP, report.rooms[2].status)

    def test_future_unknown_camera_duplicate_and_unbounded_history_fail_closed(self) -> None:
        future = self._sample(1, 10, StreamPresenceSampleState.CLEAR)
        with self.assertRaisesRegex(RoomPresenceError, "future"):
            self._report(future, seconds=0)

        unknown = StreamPresenceSample(
            sample_id="unknown_sample",
            camera_id="camera_99",
            observed_at=self.NOW,
            state=StreamPresenceSampleState.CLEAR,
            scanner_id="candidate_v1_fixture",
            detector_id=self.DETECTOR,
            source_fingerprint="f" * 64,
        )
        with self.assertRaisesRegex(RoomPresenceError, "unregistered"):
            self._report(unknown, seconds=1)

        duplicate = self._sample(1, 0, StreamPresenceSampleState.CLEAR)
        with self.assertRaisesRegex(RoomPresenceError, "IDs must be unique"):
            StreamPresenceBatch(samples=(duplicate, duplicate))

        samples = tuple(
            self._sample(
                1,
                index,
                StreamPresenceSampleState.CLEAR,
                sample_id=f"bounded_{index}",
            )
            for index in range(3)
        )
        with self.assertRaisesRegex(RoomPresenceError, "history exceeds"):
            self._report(
                *samples,
                seconds=3,
                policy=RoomPresencePolicy(max_samples_per_stream=2),
            )

    def test_contract_round_trip_and_schema_keys(self) -> None:
        batch = self._batch(
            self._sample(1, 0, StreamPresenceSampleState.PERSON_DETECTED),
            self._sample(1, 10, StreamPresenceSampleState.PERSON_DETECTED),
        )
        report = evaluate_room_presence(
            registry=fixture_registry(),
            batch=batch,
            checked_at=self.NOW + timedelta(seconds=20),
        )
        self.assertEqual(batch, StreamPresenceBatch.from_dict(batch.to_dict()))
        self.assertEqual(report, RoomPresenceReport.from_dict(report.to_dict()))
        self.assertEqual(report.policy, RoomPresencePolicy.from_dict(report.policy.to_dict()))

        batch_schema = json.loads(
            Path("schemas/stream-presence-batch-v1.schema.json").read_text(encoding="utf-8")
        )
        report_schema = json.loads(
            Path("schemas/room-presence-report-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(batch_schema["required"]), set(batch.to_dict()))
        self.assertEqual(set(report_schema["required"]), set(report.to_dict()))
        self.assertEqual(
            {item.value for item in RoomPresenceStatus},
            set(report_schema["$defs"]["assessment"]["properties"]["status"]["enum"]),
        )

    def test_authority_flags_and_state_combinations_fail_closed(self) -> None:
        batch = self._batch()
        payload = batch.to_dict()
        payload["identity_inference"] = True
        with self.assertRaisesRegex(RoomPresenceError, "must remain false"):
            StreamPresenceBatch.from_dict(payload)

        report = self._report(seconds=0)
        payload = report.to_dict()
        payload["rooms"][0]["status"] = "person_present"  # type: ignore[index]
        with self.assertRaisesRegex(RoomPresenceError, "lacks confirmed evidence"):
            RoomPresenceReport.from_dict(payload)

    def test_private_append_is_atomic_bounded_idempotent_and_conflict_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-room-presence-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            path = root / "presence.json"
            first = self._sample(1, 0, StreamPresenceSampleState.CLEAR)
            second = self._sample(1, 10, StreamPresenceSampleState.CLEAR)
            third = self._sample(1, 20, StreamPresenceSampleState.CLEAR)

            created = StreamPresenceBatch.append_private(path, first)
            reused = StreamPresenceBatch.append_private(path, first)
            self.assertEqual(created, reused)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)

            StreamPresenceBatch.append_private(path, second, max_samples_per_stream=2)
            bounded = StreamPresenceBatch.append_private(
                path,
                third,
                max_samples_per_stream=2,
            )
            self.assertEqual((second, third), bounded.samples)
            self.assertEqual(bounded, StreamPresenceBatch.read_private(path))
            self.assertEqual([], list(root.glob(".presence.json.*.tmp")))

            conflict = StreamPresenceSample(
                sample_id=third.sample_id,
                camera_id=third.camera_id,
                observed_at=third.observed_at,
                state=StreamPresenceSampleState.PERSON_DETECTED,
                scanner_id=third.scanner_id,
                detector_id=third.detector_id,
                source_fingerprint="a" * 64,
            )
            with self.assertRaisesRegex(RoomPresenceError, "conflicts"):
                StreamPresenceBatch.append_private(path, conflict)

            root.chmod(0o755)
            with self.assertRaisesRegex(RoomPresenceError, "parent permissions"):
                StreamPresenceBatch.append_private(path, third)


if __name__ == "__main__":
    unittest.main()
