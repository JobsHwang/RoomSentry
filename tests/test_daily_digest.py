from __future__ import annotations

import hashlib
import json
import unittest
from datetime import UTC, date, datetime
from unittest.mock import patch

from baby_monitor.daily_digest import (
    DailyDigestCandidate,
    DailyDigestConfig,
    DailyDigestError,
    DailyDigestPlan,
    plan_daily_digest,
)
from baby_monitor.domain import RiskLevel


def _candidate(
    index: int,
    *,
    camera: str = "camera_01",
    hour: int = 1,
    minute: int = 0,
    track: str | None = None,
    duration: float = 12.0,
    person: float = 0.0,
    motion: float = 0.0,
    hardware: bool = False,
    zone: bool = False,
    risk: RiskLevel = RiskLevel.NONE,
    keep: bool = False,
    gaps: tuple[str, ...] = (),
    offsets: tuple[float, ...] = (),
) -> DailyDigestCandidate:
    return DailyDigestCandidate(
        event_id=f"event_{index:02d}",
        camera_id=camera,
        occurred_at=datetime(2026, 8, 19, hour, minute, tzinfo=UTC),
        source_binding_sha256=f"{index:064x}",
        media_duration_seconds=duration,
        fact_offsets_seconds=offsets,
        person_confidence=person,
        motion_score=motion,
        hardware_signal=hardware,
        zone_signal=zone,
        risk_level=risk,
        human_keep=keep,
        anonymous_track_key=track,
        coverage_gap_reasons=gaps,
    )


class DailyDigestTests(unittest.TestCase):
    def test_empty_day_is_explicit_and_never_claims_video_or_risk(self) -> None:
        plan = plan_daily_digest(
            (),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
        )
        payload = plan.to_dict()
        self.assertEqual(0, payload["source_event_count"])
        self.assertEqual(0, payload["selected_event_count"])
        self.assertFalse(payload["target_met"])
        self.assertFalse(payload["video_generated"])
        self.assertFalse(payload["identity_inference"])
        self.assertFalse(payload["risk_inference"])
        self.assertFalse(payload["notification_decision"])

    def test_score_is_auditable_and_human_keep_has_visible_component(self) -> None:
        plan = plan_daily_digest(
            (
                _candidate(1, person=0.8, motion=0.6, keep=True),
                _candidate(2, hour=2, motion=0.2),
            ),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=8, max_items=1),
        )
        item = plan.items[0]
        components = dict(item.score_components)
        self.assertEqual("event_01", item.event_id)
        self.assertEqual(0.35, components["human_keep"])
        self.assertEqual(0.2, components["person_fact"])
        self.assertEqual(0.15, components["motion_fact"])
        self.assertIn("human_keep", item.selection_reasons)
        self.assertEqual(round(sum(components.values()), 6), item.selection_score)

    def test_no_recorded_gaps_never_earn_unverified_coverage_bonus(self) -> None:
        scores = []
        for gaps in ((), ("source_disconnect",)):
            candidate = _candidate(1, person=0.8, gaps=gaps)
            plan = plan_daily_digest(
                (candidate,),
                local_date=date(2026, 8, 19),
                timezone_name="UTC",
                config=DailyDigestConfig(target_duration_seconds=8, max_items=1),
            )
            item = plan.items[0]
            self.assertEqual(0.0, dict(item.score_components)["coverage_complete"])
            self.assertNotIn("coverage_complete", item.selection_reasons)
            self.assertEqual(0.2, dict(item.score_components)["person_fact"])
            self.assertEqual(candidate.source_binding_sha256, item.source_binding_sha256)
            self.assertEqual(gaps, candidate.coverage_gap_reasons)
            self.assertEqual({reason: 1 for reason in gaps}, dict(plan.coverage.gap_reason_counts))
            scores.append(item.selection_score)
        self.assertEqual([0.25, 0.25], scores)

    def test_empty_gap_list_does_not_outrank_equivalent_earlier_gapped_candidate(self) -> None:
        candidates = (
            _candidate(1, hour=1, person=0.8, gaps=("source_disconnect",)),
            _candidate(2, hour=2, person=0.8),
        )
        plan = plan_daily_digest(
            candidates,
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=8, max_items=1),
        )
        self.assertEqual("event_01", plan.items[0].event_id)
        self.assertEqual(0.25, plan.items[0].selection_score)

    def test_legacy_plan_keeps_original_score_and_bytes_without_new_certification(self) -> None:
        candidate = _candidate(1, person=0.8)
        current = plan_daily_digest(
            (candidate,),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=8, max_items=1),
        )
        # Reconstruct the synthetic pre-fix artifact observed before this change;
        # its literal ID and JSON SHA ensure all original fields stay intact.
        legacy_payload = current.to_dict()
        legacy_payload["digest_id"] = "d405f9383732ebeb8519c57bb6ac1657"
        legacy_payload["items"][0]["score_components"]["coverage_complete"] = 0.05
        legacy_payload["items"][0]["selection_score"] = 0.3
        legacy_payload["items"][0]["selection_reasons"].append("coverage_complete")
        with patch(
            "baby_monitor.daily_digest._score", side_effect=AssertionError("must not rescore")
        ):
            restored = DailyDigestPlan.from_dict(legacy_payload)
            reread = DailyDigestPlan.from_json(restored.to_json())
        self.assertEqual(legacy_payload, restored.to_dict())
        self.assertEqual(restored, reread)
        self.assertEqual(
            "fce24561f225a20831d9babafc63c5c71f6a04d031f0dddc035d1d1dbd2f8d8d",
            hashlib.sha256(restored.to_json(indent=None).encode()).hexdigest(),
        )
        self.assertEqual(0.05, dict(restored.items[0].score_components)["coverage_complete"])
        self.assertEqual(0.3, restored.items[0].selection_score)
        self.assertEqual(0.0, dict(current.items[0].score_components)["coverage_complete"])
        self.assertEqual(0.25, current.items[0].selection_score)
        self.assertNotEqual(restored.digest_id, current.digest_id)

    def test_camera_hour_gap_and_track_quotas_deduplicate_dense_events(self) -> None:
        candidates = (
            _candidate(1, minute=0, track="track_a", motion=1.0),
            _candidate(2, minute=1, track="track_a", motion=0.9),
            _candidate(3, minute=1, track="track_b", motion=0.8),
            _candidate(4, camera="camera_02", minute=3, motion=0.7),
            _candidate(5, camera="camera_02", hour=2, minute=0, motion=0.6),
        )
        plan = plan_daily_digest(
            candidates,
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(
                target_duration_seconds=40,
                max_items=5,
                per_camera_quota=2,
                per_hour_quota=2,
                min_same_camera_gap_seconds=120,
            ),
        )
        selected = {item.event_id for item in plan.items}
        self.assertIn("event_01", selected)
        self.assertNotIn("event_02", selected)
        self.assertNotIn("event_03", selected)
        self.assertLessEqual(dict(plan.coverage.selected_by_camera)["camera_02"], 2)
        self.assertLessEqual(dict(plan.coverage.selected_by_hour)["01"], 2)

    def test_multi_camera_hour_coverage_and_gap_reasons_are_preserved(self) -> None:
        plan = plan_daily_digest(
            (
                _candidate(1, camera="camera_01", hour=1, hardware=True),
                _candidate(
                    2,
                    camera="camera_02",
                    hour=2,
                    zone=True,
                    gaps=("source_disconnect",),
                ),
            ),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=16),
        )
        coverage = plan.coverage.to_dict()
        self.assertEqual({"camera_01": 1, "camera_02": 1}, coverage["source_by_camera"])
        self.assertEqual({"01": 1, "02": 1}, coverage["source_by_hour"])
        self.assertEqual({"source_disconnect": 1}, coverage["gap_reason_counts"])

    def test_fact_offsets_center_bounded_clip_without_reading_media(self) -> None:
        plan = plan_daily_digest(
            (_candidate(1, duration=20, offsets=(1.0, 3.0), motion=1.0),),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=8, max_item_seconds=8),
        )
        item = plan.items[0]
        self.assertEqual(0.0, item.clip_start_seconds)
        self.assertEqual(8.0, item.clip_end_seconds)
        self.assertEqual(8.0, item.duration_seconds)

    def test_target_shortfall_and_hard_item_limits_are_explicit(self) -> None:
        plan = plan_daily_digest(
            tuple(
                _candidate(index, camera=f"camera_{index:02d}", hour=index, motion=1.0)
                for index in range(1, 5)
            ),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(
                target_duration_seconds=30,
                max_items=2,
                max_item_seconds=5,
            ),
        )
        self.assertEqual(2, plan.selected_event_count)
        self.assertEqual(10.0, plan.selected_duration_seconds)
        self.assertFalse(plan.target_met)

    def test_plan_is_deterministic_path_free_and_contains_no_hidden_text_score(self) -> None:
        candidates = (
            _candidate(1, motion=0.8),
            _candidate(2, hour=2, person=0.7),
        )
        first = plan_daily_digest(
            candidates,
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
        )
        second = plan_daily_digest(
            candidates,
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
        )
        self.assertEqual(first.to_json(), second.to_json())
        rendered = first.to_json()
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("summary", rendered)
        self.assertNotIn("warmth", rendered)
        self.assertTrue(json.loads(rendered)["facts_only"])
        self.assertEqual(first, DailyDigestPlan.from_json(rendered))

        duplicated = rendered.replace('"version": 1', '"version": 1, "version": 1', 1)
        with self.assertRaisesRegex(DailyDigestError, "JSON is invalid"):
            DailyDigestPlan.from_json(duplicated)

    def test_invalid_dates_offsets_duplicates_and_config_fail_closed(self) -> None:
        with self.assertRaisesRegex(DailyDigestError, "outside the local date"):
            plan_daily_digest(
                (_candidate(1, hour=1),),
                local_date=date(2026, 8, 20),
                timezone_name="UTC",
            )
        duplicate = _candidate(1)
        with self.assertRaisesRegex(DailyDigestError, "event IDs must be unique"):
            plan_daily_digest(
                (duplicate, duplicate),
                local_date=date(2026, 8, 19),
                timezone_name="UTC",
            )
        with self.assertRaisesRegex(DailyDigestError, "fact offsets"):
            _candidate(1, duration=5, offsets=(6.0,))
        with self.assertRaisesRegex(DailyDigestError, "duration range"):
            DailyDigestConfig(min_item_seconds=10, max_item_seconds=5)

    def test_timezone_day_boundary_is_real_not_file_order(self) -> None:
        candidate = DailyDigestCandidate(
            event_id="event_boundary",
            camera_id="camera_01",
            occurred_at=datetime(2026, 8, 18, 16, 30, tzinfo=UTC),
            source_binding_sha256="f" * 64,
            media_duration_seconds=8,
            motion_score=1.0,
        )
        plan = plan_daily_digest(
            (candidate,),
            local_date=date(2026, 8, 19),
            timezone_name="Asia/Shanghai",
            config=DailyDigestConfig(target_duration_seconds=8),
        )
        self.assertEqual(1, plan.selected_event_count)
        self.assertEqual("00", dict(plan.coverage.source_by_hour).popitem()[0])


if __name__ == "__main__":
    unittest.main()
