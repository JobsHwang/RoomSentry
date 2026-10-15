from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.adapters.vlm import MockVLMAdapter, VLMError
from baby_monitor.live_detection import ReviewCandidate
from baby_monitor.local_runners.bonsai_27b import response_diagnostics
from baby_monitor.local_runners.subject_probe import PROMPT_VERSION
from baby_monitor.recognition_attempt import RecognitionAttemptAuditStore
from tests.test_live_detection import observation


def diagnostics(*, finish="stop"):
    return response_diagnostics(
        {
            "choices": [{"finish_reason": finish}],
            "usage": {
                "prompt_tokens": 2133,
                "completion_tokens": 317 if finish == "stop" else 768,
                "total_tokens": 2450 if finish == "stop" else 2901,
                "prompt_tokens_details": {"cached_tokens": 12},
            },
            "timings": {"prompt_ms": 125.5, "predicted_ms": 2000.0},
        },
        SimpleNamespace(context_tokens=4096, image_tokens=1024, max_new_tokens=768),
    )


class DiagnosticVLM(MockVLMAdapter):
    def __init__(self, metadata, *, failure=False):
        super().__init__()
        self.metadata = metadata
        self.failure = failure
        self.last_diagnostics = None
        self.calls = 0

    def analyze(self, request):
        self.calls += 1
        self.last_diagnostics = deepcopy(self.metadata)
        if self.failure:
            raise VLMError("synthetic failure", code="bonsai_output_token_limit")
        return super().analyze(request)


@unittest.skipUnless(os.name == "posix", "live monitor requires POSIX")
class LiveMonitorDiagnosticsTests(unittest.TestCase):
    def run_monitor(
        self,
        adapter,
        *,
        cycles=1,
        fail_before=None,
        review_task="event",
        review_png_profile="compact",
    ):
        """Run the real review loop and pipeline against synthetic pixels only."""

        class Stop:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

        class Thread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

            def is_alive(self):
                return False

        class Source:
            def __init__(self, source, **_kwargs):
                self.source = source

            def start(self):
                pass

            def close(self):
                pass

            def snapshot(self):
                return {"phase": "fixture"}

        stop = Stop()
        selected = [
            ReviewCandidate(observation(sequence=index, now=time.monotonic()), "fixture", 1, 0)
            for index in range(cycles)
        ]
        queue = Mock(condition=MagicMock())
        queue.snapshot.return_value = {}

        def take():
            if selected:
                return selected.pop(0)
            stop.set()
            return None

        queue.take_live.side_effect = take
        before_calls = 0

        def fail_second(original):
            def call(*args, **kwargs):
                nonlocal before_calls
                before_calls += 1
                if before_calls == 2:
                    raise RuntimeError("synthetic pre-analyze failure")
                return original(*args, **kwargs)

            return call

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with (
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "ReviewMailbox", return_value=queue),
                patch.object(live_monitor.threading, "Thread", Thread),
                patch.object(live_monitor.threading, "Event", side_effect=(stop, Stop())),
                patch.object(
                    live_monitor, "_review_storage_summary", return_value={"available": True}
                ),
                ExitStack() as failures,
            ):
                if fail_before == "encode":
                    failures.enter_context(
                        patch.object(
                            live_monitor,
                            "encode_bgr_png",
                            side_effect=fail_second(live_monitor.encode_bgr_png),
                        )
                    )
                elif fail_before == "pipeline":
                    failures.enter_context(
                        patch.object(
                            live_monitor,
                            "process_prepared_event",
                            side_effect=fail_second(live_monitor.process_prepared_event),
                        )
                    )
                elif fail_before == "audit":
                    failures.enter_context(
                        patch.object(
                            RecognitionAttemptAuditStore,
                            "start",
                            new=fail_second(RecognitionAttemptAuditStore.start),
                        )
                    )
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("cam1", "rtsp://127.0.0.1/fixture").source(), 16, 16),),
                    detector=Mock(identity={"fixture": True}),
                    local_vlm=adapter,
                    root=root,
                    review_admission=lambda: True,
                    review_task=review_task,
                    review_png_profile=review_png_profile,
                )
            records = sorted(
                (json.loads(path.read_text()) for path in (root / "records").glob("*.json")),
                key=lambda item: item["frame_sequence"],
            )
            reviews = [json.loads(path.read_text()) for path in (root / "reviews").glob("*.json")]
            status = json.loads((root / "status.json").read_text())
            # Compare business output across independent temporary roots, without
            # treating per-run opaque evidence paths as model-content changes.
            for review in reviews:
                if "result" in review:
                    review["result"]["evidence_frames"] = ["synthetic-frame.png"]
            serialized = "\n".join(
                path.read_text()
                for path in [
                    root / "status.json",
                    *(root / "records").glob("*.json"),
                    *(root / "reviews").glob("*.json"),
                ]
            )
        self.assertEqual(cycles, len(records))
        self.assertNotIn("local_vlm_diagnostics", json.dumps(status))
        self.assertNotIn("prompt_tokens_reported", json.dumps(status))
        self.assertEqual("stopped", status["phase"])
        self.assertIs(type(status["heartbeat_monotonic_seconds"]), float)
        self.assertGreaterEqual(status["heartbeat_monotonic_seconds"], 0)
        self.assertLessEqual(status["heartbeat_monotonic_seconds"], time.monotonic())
        self.assertEqual("unknown", status["cameras"]["cam1"]["baby_location"])
        self.assertEqual("unknown", status["cameras"]["cam1"]["baby_state"])
        self.assertEqual(
            "unimplemented_state_estimator_unknown_placeholders", status["baby_fields_basis"]
        )
        evidence = status["cameras"]["cam1"]["review_evidence"]
        self.assertEqual("monitor_not_detecting", evidence["status"])
        self.assertFalse(evidence["live_baby_state_established"])
        self.assertNotIn("received_monotonic", serialized)
        return records, reviews, status, serialized

    def test_fast_encoding_reaches_probe_and_binds_actual_encoded_bytes(self):
        records, reviews, status, _ = self.run_monitor(
            SubjectVLM(), review_task="subjects", review_png_profile="fast"
        )
        expected = hashlib.sha256(
            live_monitor.encode_bgr_png(observation().frame, profile="fast")
        ).hexdigest()
        self.assertEqual("fast", status["review_png_profile"])
        self.assertEqual("fast", records[0]["review_png_profile"])
        self.assertGreaterEqual(records[0]["evidence_encoding_ms"], 0)
        self.assertEqual(expected, records[0]["subject_result"]["input_sha256"])
        self.assertEqual(expected, reviews[0]["subject_result"]["input_sha256"])
        self.assertEqual(1, status["subject_probe_successes"])

    def test_invalid_encoding_profile_has_no_runtime_or_output_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            detector = Mock()
            for profile in (None, True, "jpeg", []):
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    live_monitor.run_live_monitor(
                        streams=(),
                        detector=detector,
                        local_vlm=Mock(),
                        root=root,
                        review_admission=lambda: True,
                        review_png_profile=profile,
                    )
            self.assertFalse(root.exists())
            self.assertEqual([], detector.mock_calls)

    def test_subject_mode_uses_one_explicit_call_and_no_event_writeback(self):
        adapter = SubjectVLM()
        records, reviews, status, _ = self.run_monitor(adapter, review_task="subjects")
        self.assertEqual(1, adapter.probe_calls)
        self.assertEqual(0, adapter.event_calls)
        self.assertEqual(1, status["subject_probe_successes"])
        self.assertEqual(0, status["review_successes"])
        self.assertEqual("subjects", records[0]["review_task"])
        self.assertIn("subject_result", reviews[0])
        self.assertEqual(reviews[0]["subject_result"], records[0]["subject_result"])
        self.assertNotIn("result", reviews[0])
        self.assertIsNone(status["cameras"]["cam1"]["subject_claims"])
        self.assertNotIn("private-visual-text", json.dumps(status))

    def test_subject_binding_failure_is_not_event_success_or_old_claim(self):
        adapter = SubjectVLM(bad_second=True)
        records, _, status, _ = self.run_monitor(adapter, cycles=2, review_task="subjects")
        self.assertEqual(["analyzed", "analysis_failed"], [r["status"] for r in records])
        self.assertEqual(1, status["subject_probe_successes"])
        self.assertEqual(1, status["subject_probe_failures"])
        self.assertEqual(0, status["review_failures"])
        self.assertIsNone(status["cameras"]["cam1"]["subject_claims"])

    def test_subject_mode_requires_capability_before_opening_resources(self):
        with self.assertRaisesRegex(ValueError, "missing subject capability"):
            live_monitor.run_live_monitor(
                streams=(),
                detector=None,
                local_vlm=MockVLMAdapter(),
                root=Path("/does-not-exist"),
                review_admission=lambda: True,
                review_task="subjects",
            )

    def test_hybrid_worker_keeps_event_and_subject_attempts_separate(self):
        class HybridVLM(SubjectVLM):
            def analyze(self, request):
                self.event_calls += 1
                return MockVLMAdapter.analyze(self, request)

        adapter = HybridVLM(bad_second=True)
        records, _, status, _ = self.run_monitor(adapter, cycles=4, review_task="hybrid")
        self.assertEqual(
            ["subjects", "event", "subjects", "event"], [r["review_task"] for r in records]
        )
        self.assertEqual(2, adapter.probe_calls)
        self.assertEqual(2, adapter.event_calls)
        self.assertEqual(2, status["review_successes"])
        self.assertEqual(1, status["subject_probe_successes"])
        self.assertEqual(1, status["subject_probe_failures"])
        data = status["cameras"]["cam1"]
        self.assertEqual("analyzed", data["review_evidence"]["attempt_status"])
        self.assertEqual("analysis_failed", data["subject_evidence"]["attempt_status"])
        self.assertEqual(records[2]["completed_at"], data["subject_evidence"]["completed_at"])
        self.assertEqual(records[3]["completed_at"], data["review_evidence"]["completed_at"])

    def test_success_persists_only_private_diagnostics_without_business_changes(self):
        baseline = self.run_monitor(MockVLMAdapter())
        metadata = diagnostics()
        records, reviews, status, _ = self.run_monitor(DiagnosticVLM(metadata))
        self.assertEqual(metadata, records[0]["local_vlm_diagnostics"])
        self.assertEqual(metadata, reviews[0]["local_vlm_diagnostics"])
        self.assertEqual("analyzed", records[0]["status"])
        self.assertEqual("analyzed", status["cameras"]["cam1"]["review_evidence"]["attempt_status"])
        self.assertEqual(1, status["review_successes"])
        self.assertEqual(0, status["review_failures"])
        for field in ("result", "decision"):
            self.assertEqual(baseline[1][0][field], reviews[0][field])
        self.assertEqual(baseline[2].keys(), status.keys())
        self.assertEqual(baseline[2]["cameras"]["cam1"].keys(), status["cameras"]["cam1"].keys())

    def test_length_failure_keeps_numeric_diagnostics_and_existing_error(self):
        metadata = diagnostics(finish="length")
        records, reviews, status, _ = self.run_monitor(DiagnosticVLM(metadata, failure=True))
        self.assertEqual(metadata, records[0]["local_vlm_diagnostics"])
        self.assertEqual("analysis_failed", records[0]["status"])
        self.assertEqual("bonsai_output_token_limit", records[0]["error_code"])
        self.assertEqual([], reviews)
        self.assertEqual(0, status["review_successes"])
        self.assertEqual(1, status["review_failures"])
        self.assertEqual(
            "analysis_failed", status["cameras"]["cam1"]["review_evidence"]["attempt_status"]
        )
        self.assertEqual(
            "bonsai_output_token_limit", status["cameras"]["cam1"]["last_review_error"]
        )

    def test_invalid_metadata_is_discarded_without_serialization_or_business_changes(self):
        baseline = self.run_monitor(MockVLMAdapter())
        for metadata in (
            {**diagnostics(), "prompt": "private-diagnostic-text"},
            {**diagnostics(), "prompt_tokens_reported": "private-diagnostic-text"},
            {**diagnostics(), "prompt_tokens_reported": True},
            {**diagnostics(), "predicted_ms_reported": float("nan")},
            {**diagnostics(), "predicted_ms_reported": 10**400},
            ["private-diagnostic-text"],
        ):
            with self.subTest(metadata_type=type(metadata).__name__):
                records, reviews, status, serialized = self.run_monitor(DiagnosticVLM(metadata))
                self.assertIsNone(records[0]["local_vlm_diagnostics"])
                self.assertIsNone(reviews[0]["local_vlm_diagnostics"])
                self.assertNotIn("private-diagnostic-text", serialized)
                self.assertEqual("analyzed", records[0]["status"])
                self.assertEqual(1, status["review_successes"])
                for field in ("result", "decision"):
                    self.assertEqual(baseline[1][0][field], reviews[0][field])

    def test_previous_diagnostics_are_not_reused_before_analyze(self):
        for failure in ("encode", "pipeline", "audit"):
            with self.subTest(failure=failure):
                adapter = DiagnosticVLM(diagnostics())
                records, _reviews, status, _ = self.run_monitor(
                    adapter, cycles=2, fail_before=failure
                )
                self.assertEqual(1, adapter.calls)
                self.assertEqual(diagnostics(), records[0]["local_vlm_diagnostics"])
                self.assertIsNone(records[1]["local_vlm_diagnostics"])
                self.assertEqual("analysis_failed", records[1]["status"])
                self.assertEqual(1, status["review_successes"])
                self.assertEqual(1, status["review_failures"])
                evidence = status["cameras"]["cam1"]["review_evidence"]
                self.assertEqual("analysis_failed", evidence["attempt_status"])
                self.assertEqual(records[1]["frame_received_at"], evidence["frame_received_at"])
                self.assertEqual(records[1]["completed_at"], evidence["completed_at"])
                self.assertEqual(
                    "recognition_audit_unavailable"
                    if failure == "audit"
                    else "local_review_failed",
                    records[1]["error_code"],
                )

    def test_missing_or_unchanged_legacy_diagnostics_remain_unknown(self):
        for adapter in (MockVLMAdapter(), DiagnosticVLM(None)):
            with self.subTest(adapter=type(adapter).__name__):
                records, reviews, _status, _ = self.run_monitor(adapter)
                self.assertIsNone(records[0]["local_vlm_diagnostics"])
                self.assertIsNone(reviews[0]["local_vlm_diagnostics"])
        legacy = MockVLMAdapter()
        legacy.last_diagnostics = diagnostics()
        records, reviews, _status, _ = self.run_monitor(legacy)
        self.assertIsNone(records[0]["local_vlm_diagnostics"])
        self.assertIsNone(reviews[0]["local_vlm_diagnostics"])

    def test_raising_optional_metadata_property_does_not_fail_review(self):
        class LegacyVLM(MockVLMAdapter):
            @property
            def last_diagnostics(self):
                raise RuntimeError("private-diagnostic-text")

        records, reviews, status, serialized = self.run_monitor(LegacyVLM())
        self.assertIsNone(records[0]["local_vlm_diagnostics"])
        self.assertIsNone(reviews[0]["local_vlm_diagnostics"])
        self.assertEqual(1, status["review_successes"])
        self.assertNotIn("private-diagnostic-text", serialized)


class SubjectVLM(MockVLMAdapter):
    def __init__(self, *, bad_second=False):
        super().__init__()
        self.probe_calls = self.event_calls = 0
        self.bad_second = bad_second

    def analyze(self, request):
        self.event_calls += 1
        raise AssertionError("subject mode must not invoke event generation")

    def probe_subjects(self, request):
        self.probe_calls += 1
        path = Path(request.media.frame_paths[0])
        return {
            "version": 1,
            "prompt_revision": PROMPT_VERSION,
            "semantics": "unverified_model_claims",
            "frame_id": path.name,
            "input_sha256": "0" * 64
            if self.bad_second and self.probe_calls == 2
            else hashlib.sha256(path.read_bytes()).hexdigest(),
            "coordinate_space": "submitted_image_thousandths",
            "claims": {
                "people": [
                    {
                        "head_center": [500, 500],
                        "age_group": "baby",
                        "visual_evidence": "private-visual-text",
                    }
                ],
                "uncertainties": [],
            },
        }


if __name__ == "__main__":
    unittest.main()
