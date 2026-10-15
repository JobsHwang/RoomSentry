from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from baby_monitor.detector_sampling import (
    DetectorSampleFrame,
    DetectorSamplingError,
    DetectorSamplingLedger,
    validate_nominal_sampling_grid,
)
from baby_monitor.domain import Detection


def sample(index: int = 0, timestamp: float = 0, start: int = 0, count: int = 0):
    return DetectorSampleFrame(index, timestamp, "container_pts", start, count)


def ledger() -> DetectorSamplingLedger:
    return DetectorSamplingLedger(
        native_fps=25,
        native_fps_source="reported",
        requested_sample_fps=1,
        frame_step=25,
        decoded_frame_count=51,
        samples=(sample(), sample(25, 1.2, 0, 1), sample(50, 2.4, 1, 0)),
        timestamp_fallback_frame_count=0,
        max_adjacent_container_pts_gap_seconds=0.24,
        container_pts_regression_count=0,
    )


class DetectorSamplingTests(unittest.TestCase):
    def test_nominal_grid_accepts_complete_actual_empty_samples_without_mutation(self) -> None:
        evidence = replace(ledger(), samples=(sample(), sample(25, 1), sample(50, 2)))
        before = evidence.to_dict()
        validate_nominal_sampling_grid(evidence, source_fps=25, source_frame_count=51)
        self.assertEqual(before, evidence.to_dict())
        self.assertFalse(evidence.source_exhaustion_verified)
        # No statement is made about the time of unsampled decoded frames.
        validate_nominal_sampling_grid(
            replace(evidence, timestamp_fallback_frame_count=1),
            source_fps=25,
            source_frame_count=51,
        )

    def test_nominal_grid_rejects_count_and_fps_metadata_disagreement(self) -> None:
        evidence = replace(ledger(), samples=(sample(), sample(25, 1), sample(50, 2)))
        for count in (50, 52):
            with self.subTest(count=count), self.assertRaises(DetectorSamplingError):
                validate_nominal_sampling_grid(evidence, source_fps=25, source_frame_count=count)
        for invalid in (
            replace(evidence, native_fps=25.01),
            replace(evidence, native_fps_source="fallback_25"),
        ):
            with self.subTest(evidence=invalid), self.assertRaises(DetectorSamplingError):
                validate_nominal_sampling_grid(invalid, source_fps=25, source_frame_count=51)

    def test_nominal_grid_rejects_empty_off_grid_duplicate_and_fallback_samples(self) -> None:
        evidence = replace(ledger(), samples=(sample(), sample(25, 1), sample(50, 2)))
        cases = (
            ledger(),
            replace(evidence, samples=(sample(), sample(25, 0), sample(50, 2))),
            replace(evidence, samples=(sample(0, 1), sample(25, 2), sample(50, 3))),
            replace(
                evidence,
                samples=(
                    sample(),
                    replace(sample(25, 1), timestamp_source="nominal_fps"),
                    sample(50, 2),
                ),
                timestamp_fallback_frame_count=1,
            ),
            replace(evidence, samples=(sample(), sample(25, 1), sample(50, 1e308))),
        )
        for invalid in cases:
            with self.subTest(evidence=invalid), self.assertRaises(DetectorSamplingError):
                validate_nominal_sampling_grid(invalid, source_fps=25, source_frame_count=51)

    def test_nominal_grid_tolerance_is_in_frame_units(self) -> None:
        for error, accepted in ((0.5e-6, True), (1.5e-6, False)):
            evidence = replace(
                ledger(), samples=(sample(), sample(25, 1), sample(50, 2 + error / 25))
            )
            if accepted:
                validate_nominal_sampling_grid(evidence, source_fps=25, source_frame_count=51)
            else:
                with self.assertRaises(DetectorSamplingError):
                    validate_nominal_sampling_grid(evidence, source_fps=25, source_frame_count=51)

    def test_nominal_grid_requires_typed_bounded_inputs(self) -> None:
        for kwargs in (
            {"source_fps": True, "source_frame_count": 51},
            {"source_fps": float("nan"), "source_frame_count": 51},
            {"source_fps": 10**1000, "source_frame_count": 51},
            {"source_fps": 25, "source_frame_count": 51.0},
            {"source_fps": 25, "source_frame_count": True},
        ):
            with self.subTest(fields=tuple(kwargs)), self.assertRaises(DetectorSamplingError):
                validate_nominal_sampling_grid(ledger(), **kwargs)
        with self.assertRaises(DetectorSamplingError):
            validate_nominal_sampling_grid(None, source_fps=25, source_frame_count=51)

    def test_round_trip_preserves_empty_samples_and_no_paths(self) -> None:
        evidence = ledger()
        restored = DetectorSamplingLedger.from_dict(json.loads(json.dumps(evidence.to_dict())))
        self.assertEqual(evidence, restored)
        self.assertEqual([0, 1, 0], [entry.detection_count for entry in restored.samples])
        restored.validate_detections((Detection(1.2, "person", 0.8, (0, 0, 1, 1)),))
        self.assertNotIn("/", json.dumps(evidence.to_dict()))
        self.assertFalse(restored.source_exhaustion_verified)
        with self.assertRaises(FrozenInstanceError):
            restored.frame_step = 1

    def test_all_empty_calls_are_valid_observations(self) -> None:
        evidence = replace(ledger(), samples=(sample(), sample(25, 1), sample(50, 2)))
        evidence.validate_detections(())

    def test_equal_timestamps_remain_distinct_invocations(self) -> None:
        evidence = replace(
            ledger(), samples=(sample(0, 1, 0, 1), sample(25, 1, 1, 1), sample(50, 1, 2))
        )
        box = Detection(1, "person", 0.8, (0, 0, 1, 1))
        evidence.validate_detections((box, box))

    def test_missing_extra_or_off_grid_samples_rejected(self) -> None:
        cases = [
            (sample(), sample(25, 1)),
            (sample(), sample(25, 1), sample(50, 2), sample(75, 3)),
            (sample(), sample(26, 1), sample(50, 2)),
            (sample(), sample(50, 1), sample(25, 2)),
            (sample(1), sample(25, 1), sample(50, 2)),
        ]
        for samples in cases:
            with self.subTest(samples=samples), self.assertRaises(DetectorSamplingError):
                replace(ledger(), samples=samples)

    def test_offsets_cannot_skip_overlap_or_start_after_zero(self) -> None:
        for samples in (
            (sample(0, 0, 1), sample(25, 1, 1), sample(50, 2, 1)),
            (sample(), sample(25, 1, 1), sample(50, 2, 1)),
            (sample(0, 0, 0, 1), sample(25, 1), sample(50, 2)),
        ):
            with self.subTest(samples=samples), self.assertRaises(DetectorSamplingError):
                replace(ledger(), samples=samples)

    def test_decreasing_timestamps_rejected(self) -> None:
        with self.assertRaises(DetectorSamplingError):
            replace(ledger(), samples=(sample(0, 2), sample(25, 1), sample(50, 3)))

    def test_wrong_step_or_decoded_count_rejected(self) -> None:
        for field, values in {
            "frame_step": (0, 24, 26, True, 25.0, 10_000_001),
            "decoded_frame_count": (0, 25, 76, True, 51.0, 10_000_001),
        }.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(DetectorSamplingError),
                ):
                    replace(ledger(), **{field: value})

    def test_step_uses_round_with_minimum_one(self) -> None:
        for native, requested, step in ((25, 2, 12), (1, 10, 1), (0.01, 10, 1)):
            evidence = replace(
                ledger(),
                native_fps=native,
                requested_sample_fps=requested,
                frame_step=step,
                decoded_frame_count=1,
                samples=(sample(),),
                max_adjacent_container_pts_gap_seconds=None,
            )
            self.assertEqual(step, evidence.frame_step)

    def test_numeric_types_and_bounds_rejected(self) -> None:
        cases = {
            "native_fps": (True, "25", None, 0, -1, 1001, float("nan"), float("inf"), 10**1000),
            "requested_sample_fps": (True, 0, 11, float("nan"), float("inf")),
            "max_adjacent_container_pts_gap_seconds": (True, -1, "1", float("nan"), float("inf")),
            "timestamp_fallback_frame_count": (True, -1, 52, 1.0),
            "container_pts_regression_count": (True, -1, 51, 1.0),
        }
        for field, values in cases.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(DetectorSamplingError),
                ):
                    replace(ledger(), **{field: value})

    def test_sample_numeric_types_and_bounds_rejected(self) -> None:
        cases = {
            "decoded_frame_index": (True, -1, 10_000_000, 1.0),
            "timestamp_seconds": (True, -1, "0", float("nan"), float("inf"), 10**1000),
            "detection_start": (True, -1, 100_001, 0.0),
            "detection_count": (True, -1, 100_001, 0.0),
        }
        for field, values in cases.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(DetectorSamplingError),
                ):
                    replace(sample(), **{field: value})
        with self.assertRaises(DetectorSamplingError):
            sample(start=100_000, count=1)

    def test_unsampled_pts_regressions_are_retained(self) -> None:
        evidence = replace(ledger(), container_pts_regression_count=3)
        self.assertEqual(
            3, DetectorSamplingLedger.from_dict(evidence.to_dict()).container_pts_regression_count
        )
        with self.assertRaises(DetectorSamplingError):
            replace(
                evidence,
                decoded_frame_count=1,
                samples=(sample(),),
                container_pts_regression_count=1,
            )

    def test_fallback_source_and_counts_are_bound(self) -> None:
        evidence = replace(ledger(), native_fps_source="fallback_25")
        self.assertEqual(25, evidence.native_fps)
        with self.assertRaises(DetectorSamplingError):
            replace(evidence, native_fps=24, frame_step=24)
        fallback = replace(sample(), timestamp_source="nominal_fps")
        with self.assertRaises(DetectorSamplingError):
            replace(ledger(), samples=(fallback, sample(25, 1), sample(50, 2)))
        evidence = replace(
            ledger(),
            samples=(fallback, sample(25, 1), sample(50, 2)),
            timestamp_fallback_frame_count=1,
        )
        self.assertEqual(1, evidence.timestamp_fallback_frame_count)
        self.assertIsNone(
            replace(
                evidence, max_adjacent_container_pts_gap_seconds=None
            ).max_adjacent_container_pts_gap_seconds
        )

    def test_fallback_count_cannot_include_sampled_pts_frames(self) -> None:
        evidence = replace(ledger(), timestamp_fallback_frame_count=48)
        self.assertEqual(48, evidence.timestamp_fallback_frame_count)
        for count in (49, 50, 51):
            with self.subTest(count=count), self.assertRaises(DetectorSamplingError):
                replace(ledger(), timestamp_fallback_frame_count=count)

    def test_pts_gap_requires_at_least_two_nonfallback_frames(self) -> None:
        samples = tuple(replace(row, timestamp_source="nominal_fps") for row in ledger().samples)
        for count in (50, 51):
            evidence = replace(
                ledger(),
                samples=samples,
                timestamp_fallback_frame_count=count,
                max_adjacent_container_pts_gap_seconds=None,
            )
            for gap in (0, 0.1):
                with self.subTest(count=count, gap=gap), self.assertRaises(DetectorSamplingError):
                    replace(evidence, max_adjacent_container_pts_gap_seconds=gap)
        evidence = replace(ledger(), samples=samples, timestamp_fallback_frame_count=49)
        self.assertEqual(0.24, evidence.max_adjacent_container_pts_gap_seconds)

    def test_pts_regression_count_excludes_fallback_frames(self) -> None:
        samples = tuple(replace(row, timestamp_source="nominal_fps") for row in ledger().samples)
        evidence = replace(
            ledger(),
            samples=samples,
            timestamp_fallback_frame_count=49,
            container_pts_regression_count=1,
        )
        with self.assertRaises(DetectorSamplingError):
            replace(evidence, container_pts_regression_count=2)
        for count in (50, 51):
            evidence = replace(
                ledger(),
                samples=samples,
                timestamp_fallback_frame_count=count,
                max_adjacent_container_pts_gap_seconds=None,
            )
            with self.subTest(count=count), self.assertRaises(DetectorSamplingError):
                replace(evidence, container_pts_regression_count=1)

    def test_native_fps_fallback_does_not_imply_timestamp_fallback(self) -> None:
        evidence = replace(ledger(), native_fps_source="fallback_25")
        self.assertEqual(0, evidence.timestamp_fallback_frame_count)
        self.assertTrue(all(row.timestamp_source == "container_pts" for row in evidence.samples))

    def test_unknown_sources_or_completeness_claim_rejected(self) -> None:
        for field, values in {
            "native_fps_source": ("unknown", "/private/source", None, True, []),
            "end_reason": ("eof", "complete", None, True),
            "source_exhaustion_verified": (True, 0, None, "false"),
        }.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(DetectorSamplingError),
                ):
                    replace(ledger(), **{field: value})
        for value in ("unknown", "/private/source", None, True, []):
            with self.subTest(value=value), self.assertRaises(DetectorSamplingError):
                replace(sample(), timestamp_source=value)

    def test_dicts_require_exact_fields_and_json_arrays(self) -> None:
        for contract, value in (
            (DetectorSamplingLedger, ledger().to_dict()),
            (DetectorSampleFrame, sample().to_dict()),
        ):
            for invalid in (None, [], {}, {**value, "source_path": "/private/source"}):
                with (
                    self.subTest(contract=contract, invalid=invalid),
                    self.assertRaises(DetectorSamplingError),
                ):
                    contract.from_dict(invalid)
            for key in value:
                with self.subTest(missing=key), self.assertRaises(DetectorSamplingError):
                    contract.from_dict({k: v for k, v in value.items() if k != key})
        for samples in ((), {}, None, [], [sample().to_dict()] * 100_001):
            with self.subTest(kind=type(samples)), self.assertRaises(DetectorSamplingError):
                DetectorSamplingLedger.from_dict({**ledger().to_dict(), "samples": samples})

    def test_constructor_rejects_mutable_or_untyped_samples(self) -> None:
        for samples in ([], (), (None, None, None), (sample(),) * 100_001):
            with self.subTest(kind=type(samples)), self.assertRaises(DetectorSamplingError):
                replace(ledger(), samples=samples)

    def test_detections_must_match_exact_offsets_and_timestamps(self) -> None:
        box = Detection(1.2, "person", 0.8, (0, 0, 1, 1))
        for detections in (
            [],
            (),
            (box, box),
            (None,),
            (replace(box, timestamp_seconds=1.20000001),),
        ):
            with self.subTest(detections=detections), self.assertRaises(DetectorSamplingError):
                ledger().validate_detections(detections)


if __name__ == "__main__":
    unittest.main()
