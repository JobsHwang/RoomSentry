from __future__ import annotations

import unittest
from dataclasses import replace

from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.domain import Detection
from baby_monitor.presence_sampling import project_presence_sampling
from baby_monitor.room_presence import StreamPresenceSampleState


def _evidence(
    records=((0.0, ()), (1.0, ()), (2.0, ())),
    *,
    fallback_indices=(),
    fallback_frame_count=None,
    regressions=0,
    native_fps_source="reported",
    maximum_pts_gap=0.04,
    sample_fps=1.0,
):
    """Synthetic actual calls, never source completeness or media evidence."""
    detections = []
    samples = []
    step = max(1, round(25.0 / sample_fps))
    for index, (seconds, labels) in enumerate(records):
        samples.append(
            DetectorSampleFrame(
                decoded_frame_index=index * step,
                timestamp_seconds=seconds,
                timestamp_source="nominal_fps" if index in fallback_indices else "container_pts",
                detection_start=len(detections),
                detection_count=len(labels),
            )
        )
        detections.extend(Detection(seconds, label, 0.9, (0.1, 0.1, 0.4, 0.8)) for label in labels)
    decoded_count = (len(samples) - 1) * step + 1
    ledger = DetectorSamplingLedger(
        native_fps=25.0,
        native_fps_source=native_fps_source,
        requested_sample_fps=sample_fps,
        frame_step=step,
        decoded_frame_count=decoded_count,
        samples=tuple(samples),
        timestamp_fallback_frame_count=(
            len(fallback_indices) if fallback_frame_count is None else fallback_frame_count
        ),
        max_adjacent_container_pts_gap_seconds=maximum_pts_gap if decoded_count > 1 else None,
        container_pts_regression_count=regressions,
    )
    return tuple(detections), ledger


class PresenceSamplingTests(unittest.TestCase):
    def project(self, *, records=None, duration=3.0, **options):
        detections, ledger = _evidence(
            **({"records": records} if records is not None else {}), **options
        )
        return project_presence_sampling(
            detections=detections, ledger=ledger, duration_seconds=duration
        )

    def test_missing_ledger_empty_or_nonempty_detections_is_only_gap(self):
        positive = Detection(0.25, "person", 0.9, (0.1, 0.1, 0.4, 0.8))
        for detections in ((), (positive,)):
            with self.subTest(detections=len(detections)):
                result = project_presence_sampling(
                    detections=detections, ledger=None, duration_seconds=3.0
                )
                self.assertEqual(1, len(result))
                self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, result[0].state)
                self.assertEqual("sampling_unavailable", result[0].reason_code)
                self.assertEqual(3.0, result[0].offset_seconds)

    def test_valid_empty_calls_clear_at_last_actual_pts_not_segment_end(self):
        result = self.project(records=((0.03, ()), (1.02, ()), (2.01, ())))
        self.assertEqual(1, len(result))
        self.assertIs(StreamPresenceSampleState.CLEAR, result[0].state)
        self.assertEqual(2.01, result[0].offset_seconds)
        self.assertIsNone(result[0].reason_code)

    def test_early_person_keeps_acquisition_time_and_one_segment_vote(self):
        result = self.project(records=((0.1, ("person",)), (1.0, ()), (2.0, ())))
        self.assertEqual(1, len(result))
        self.assertIs(StreamPresenceSampleState.PERSON_DETECTED, result[0].state)
        self.assertEqual(0.1, result[0].offset_seconds)

    def test_multiple_people_and_calls_still_emit_one_latest_positive_vote(self):
        result = self.project(
            records=((0.0, ("person", "person")), (1.0, ("person",)), (2.0, ("cat",)))
        )
        self.assertEqual(1, len(result))
        self.assertIs(StreamPresenceSampleState.PERSON_DETECTED, result[0].state)
        self.assertEqual(1.0, result[0].offset_seconds)

    def test_truncated_tail_has_gap_and_retains_known_pts_person(self):
        result = self.project(records=((0.0, ()), (1.0, ("person",))), duration=5.0)
        self.assertEqual(
            (StreamPresenceSampleState.PERSON_DETECTED, StreamPresenceSampleState.COVERAGE_GAP),
            tuple(item.state for item in result),
        )
        self.assertEqual((1.0, 5.0), tuple(item.offset_seconds for item in result))
        self.assertEqual("sampling_tail_gap", result[-1].reason_code)
        negative = self.project(records=((0.0, ()), (1.0, ())), duration=5.0)
        self.assertEqual(1, len(negative))
        self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, negative[0].state)

    def test_large_sample_or_decoded_pts_gap_never_clears(self):
        for options in (
            {"records": ((0.0, ("person",)), (2.0, ()), (3.0, ()))},
            {"records": ((0.0, ("person",)), (1.0, ()), (2.0, ())), "maximum_pts_gap": 3.07},
            {"records": ((1.1, ("person",)), (2.0, ()), (3.0, ()))},
        ):
            with self.subTest(fields=tuple(options)):
                result = self.project(duration=4.0, **options)
                self.assertEqual(2, len(result))
                self.assertIs(StreamPresenceSampleState.PERSON_DETECTED, result[0].state)
                self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, result[-1].state)
                self.assertEqual("sampling_time_gap", result[-1].reason_code)

    def test_uncertain_clock_duplicates_fallback_and_regressions_never_clear(self):
        for options in (
            {"records": ((0.0, ()), (1.0, ()), (1.0, ()))},
            {"fallback_indices": (1,)},
            {"fallback_frame_count": 1},
            {"regressions": 1},
        ):
            with self.subTest(fields=tuple(options)):
                result = self.project(**options)
                self.assertEqual(1, len(result))
                self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, result[0].state)
                self.assertEqual("sampling_clock_uncertain", result[0].reason_code)

    def test_fallback_person_has_no_positive_vote_but_known_pts_person_survives(self):
        unknown = self.project(
            records=((0.0, ()), (1.0, ("person",)), (2.0, ())), fallback_indices=(1,)
        )
        self.assertEqual(
            (StreamPresenceSampleState.COVERAGE_GAP,), tuple(item.state for item in unknown)
        )
        mixed = self.project(
            records=((0.0, ("person",)), (1.0, ("person",)), (2.0, ())), fallback_indices=(1,)
        )
        self.assertEqual(
            (StreamPresenceSampleState.PERSON_DETECTED, StreamPresenceSampleState.COVERAGE_GAP),
            tuple(item.state for item in mixed),
        )
        self.assertEqual(0.0, mixed[0].offset_seconds)

    def test_decoder_pts_regression_discards_unreliable_positive_placement(self):
        result = self.project(records=((0.0, ("person",)), (1.0, ()), (2.0, ())), regressions=1)
        self.assertEqual(1, len(result))
        self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, result[0].state)
        self.assertEqual("sampling_clock_uncertain", result[0].reason_code)

    def test_native_fps_fallback_cannot_establish_clear_sampling_cadence(self):
        result = self.project(native_fps_source="fallback_25")
        self.assertEqual(1, len(result))
        self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, result[0].state)
        self.assertEqual("sampling_cadence_unknown", result[0].reason_code)

    def test_sampling_cadence_uses_executed_stride_and_boundary_tolerance(self):
        # 25 / 2 rounds to a 12-frame stride (0.48 s), not exactly 0.5 s.
        result = self.project(
            records=((0.0, ()), (0.48, ()), (0.96, ())), sample_fps=2.0, duration=1.48
        )
        self.assertIs(StreamPresenceSampleState.CLEAR, result[0].state)
        beyond = self.project(
            records=((0.0, ()), (0.48, ()), (0.96, ())), sample_fps=2.0, duration=1.480001
        )
        self.assertIs(StreamPresenceSampleState.COVERAGE_GAP, beyond[0].state)
        self.assertEqual("sampling_tail_gap", beyond[0].reason_code)

    def test_sample_after_segment_boundary_is_rejected_not_clamped(self):
        detections, ledger = _evidence(((0.0, ()), (1.0, ()), (2.000001, ())))
        with self.assertRaisesRegex(ValueError, "exceeds segment"):
            project_presence_sampling(detections=detections, ledger=ledger, duration_seconds=2.0)
        result = self.project(duration=2.0)
        self.assertEqual(2.0, result[0].offset_seconds)

    def test_input_validation_and_detection_slice_binding(self):
        detections, ledger = _evidence(((0.0, ("person",)), (1.0, ()), (2.0, ())))
        for duration in (True, 0, -1, float("nan"), float("inf"), 86401, "3", 10**1000):
            with self.subTest(kind=type(duration)), self.assertRaises(ValueError):
                project_presence_sampling(
                    detections=detections, ledger=ledger, duration_seconds=duration
                )
        for invalid in ([], None, (object(),)):
            with self.subTest(kind=type(invalid)), self.assertRaises(ValueError):
                project_presence_sampling(detections=invalid, ledger=ledger, duration_seconds=3.0)
        with self.assertRaises(ValueError):
            project_presence_sampling(detections=detections, ledger={}, duration_seconds=3.0)
        with self.assertRaisesRegex(ValueError, "total does not match"):
            project_presence_sampling(detections=(), ledger=ledger, duration_seconds=3.0)
        with self.assertRaisesRegex(ValueError, "timestamp does not match"):
            project_presence_sampling(
                detections=(replace(detections[0], timestamp_seconds=0.5),),
                ledger=ledger,
                duration_seconds=3.0,
            )


if __name__ == "__main__":
    unittest.main()
