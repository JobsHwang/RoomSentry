from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from baby_monitor.detector_protocol import (
    AdmittedDetectorScan,
    DetectorProtocolError,
    DetectorScanResult,
)


def result_payload(*, version: int = 2) -> dict:
    result = {
        "version": version,
        "camera_id": "cam_01",
        "detector_adapter": "fixture_runner",
        "model_id": "fixture_yolo",
        "model_revision": "revision_01",
        "model_sha256": "a" * 64,
        "detections": [
            {
                "timestamp_seconds": 1.0,
                "label": "person",
                "confidence": 0.8,
                "bbox_xyxy": [0.1, 0.2, 0.5, 0.9],
            }
        ],
        "subject_identity_inference": False,
        "paths_disclosed": False,
    }
    if version == 2:
        result["sampling"] = {
            "native_fps": 25.0,
            "native_fps_source": "reported",
            "requested_sample_fps": 1.0,
            "frame_step": 25,
            "decoded_frame_count": 51,
            "samples": [
                {
                    "decoded_frame_index": index,
                    "timestamp_seconds": timestamp,
                    "timestamp_source": "container_pts",
                    "detection_start": start,
                    "detection_count": count,
                }
                for index, timestamp, start, count in (
                    (0, 0.0, 0, 0),
                    (25, 1.0, 0, 1),
                    (50, 2.0, 1, 0),
                )
            ],
            "timestamp_fallback_frame_count": 0,
            "container_pts_regression_count": 0,
            "max_adjacent_container_pts_gap_seconds": 0.04,
            "end_reason": "read_returned_false",
            "source_exhaustion_verified": False,
        }
    return result


def admitted_payload(*, version: int = 2) -> dict:
    return {
        "version": version,
        "source_admission_sha256": "b" * 64,
        "labels": ["person"],
        "confidence_threshold": 0.35,
        "requested_sample_fps": 1.0,
        "result": result_payload(version=version),
    }


class DetectorSamplingProtocolTests(unittest.TestCase):
    def test_v2_result_round_trip_retains_empty_invocations(self) -> None:
        payload = result_payload()
        result = DetectorScanResult.from_dict(payload)
        self.assertEqual(payload, result.to_dict())
        self.assertEqual(result, DetectorScanResult.from_json(result.to_json()))
        self.assertEqual([0, 1, 0], [s.detection_count for s in result.sampling.samples])
        self.assertFalse(result.sampling.source_exhaustion_verified)
        self.assertNotIn("source_path", result.to_json())
        self.assertNotIn("identity", result.to_dict()["sampling"])

    def test_v2_admitted_scan_round_trip_and_private_file_io(self) -> None:
        scan = AdmittedDetectorScan.from_dict(admitted_payload())
        self.assertEqual(scan, AdmittedDetectorScan.from_json(scan.to_json()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.json"
            scan.write_new(path)
            restored = AdmittedDetectorScan.read(path)
            self.assertEqual(scan, restored)
            self.assertEqual(scan.fingerprint, restored.fingerprint)
            result_path = Path(directory) / "result.json"
            scan.result.write_new(result_path)
            self.assertEqual(scan.result, DetectorScanResult.read(result_path))

    def test_v1_serialization_and_fingerprint_remain_unchanged(self) -> None:
        payload = admitted_payload(version=1)
        scan = AdmittedDetectorScan.from_dict(payload)
        self.assertEqual(payload, scan.to_dict())
        self.assertIsNone(scan.result.sampling)
        self.assertNotIn("sampling", scan.result.to_dict())
        self.assertEqual(scan, AdmittedDetectorScan.from_json(scan.to_json()))
        # Frozen canonical v1 fixture hash, not recomputed from the implementation.
        self.assertEqual(
            "c04e5599ee4723c341ee0ec1a05547f5bbed63ce6b1c2e2d148729c2d933e2b5",
            scan.fingerprint,
        )

    def test_v2_requires_sampling_and_v1_must_not_claim_it(self) -> None:
        without_sampling = result_payload()
        del without_sampling["sampling"]
        with self.assertRaises(DetectorProtocolError):
            DetectorScanResult.from_dict(without_sampling)
        with self.assertRaises(DetectorProtocolError):
            DetectorScanResult.from_dict({**result_payload(), "sampling": None})
        with self.assertRaises(DetectorProtocolError):
            DetectorScanResult.from_dict({**result_payload(), "version": 1})
        with self.assertRaises(DetectorProtocolError):
            replace(DetectorScanResult.from_dict(result_payload(version=1)), version=2)

    def test_wrong_frame_index_or_detection_offset_is_rejected(self) -> None:
        for field, value in (("decoded_frame_index", 24), ("detection_start", 1)):
            with self.subTest(field=field):
                payload = result_payload()
                payload["sampling"]["samples"][1][field] = value
                with self.assertRaises(DetectorProtocolError):
                    DetectorScanResult.from_dict(payload)

    def test_detection_timestamp_must_match_actual_invocation_exactly(self) -> None:
        for timestamp in (1.0000001, 1.04, 2.0):
            with self.subTest(timestamp=timestamp):
                payload = result_payload()
                payload["detections"][0]["timestamp_seconds"] = timestamp
                with self.assertRaises(DetectorProtocolError):
                    DetectorScanResult.from_dict(payload)

    def test_detection_slice_count_and_decoded_grid_must_be_complete(self) -> None:
        for mutation in ("omit_empty_sample", "unbound_detection", "overcount"):
            with self.subTest(mutation=mutation):
                payload = result_payload()
                if mutation == "omit_empty_sample":
                    payload["sampling"]["samples"].pop()
                elif mutation == "unbound_detection":
                    payload["detections"].append(copy.deepcopy(payload["detections"][0]))
                else:
                    payload["sampling"]["samples"][-1]["detection_count"] = 1
                with self.assertRaises(DetectorProtocolError):
                    DetectorScanResult.from_dict(payload)

    def test_outer_and_inner_versions_must_match(self) -> None:
        for outer_version, inner_version in ((1, 2), (2, 1)):
            with self.subTest(outer=outer_version, inner=inner_version):
                payload = admitted_payload(version=outer_version)
                payload["result"] = result_payload(version=inner_version)
                with self.assertRaises(DetectorProtocolError):
                    AdmittedDetectorScan.from_dict(payload)

    def test_outer_fps_cannot_conflict_with_execution_evidence(self) -> None:
        payload = admitted_payload()
        payload["requested_sample_fps"] = 0.5
        with self.assertRaises(DetectorProtocolError):
            AdmittedDetectorScan.from_dict(payload)

    def test_v2_threshold_filter_rebinds_slices_and_keeps_empty_frames(self) -> None:
        payload = result_payload()
        low = copy.deepcopy(payload["detections"][0])
        low["confidence"] = 0.4
        later = copy.deepcopy(low)
        later["timestamp_seconds"] = 2.0
        payload["detections"] = [low, payload["detections"][0], later]
        payload["sampling"]["samples"][1]["detection_count"] = 2
        payload["sampling"]["samples"][2].update(detection_start=2, detection_count=1)
        original = DetectorScanResult.from_dict(payload)
        filtered = original.at_confidence_threshold(0.8)
        self.assertEqual([0.8], [d.confidence for d in filtered.detections])
        self.assertEqual(
            [(0, 0), (0, 1), (1, 0)],
            [(s.detection_start, s.detection_count) for s in filtered.sampling.samples],
        )
        for before, after in zip(original.sampling.samples, filtered.sampling.samples, strict=True):
            self.assertEqual(before.decoded_frame_index, after.decoded_frame_index)
            self.assertEqual(before.timestamp_seconds, after.timestamp_seconds)
            self.assertEqual(before.timestamp_source, after.timestamp_source)
        before_evidence = original.sampling.to_dict()
        after_evidence = filtered.sampling.to_dict()
        del before_evidence["samples"], after_evidence["samples"]
        self.assertEqual(before_evidence, after_evidence)
        self.assertEqual(payload, original.to_dict())
        self.assertEqual(filtered, DetectorScanResult.from_json(filtered.to_json()))
        empty = original.at_confidence_threshold(0.9)
        self.assertEqual((), empty.detections)
        self.assertEqual(
            [(0, 0)] * 3, [(s.detection_start, s.detection_count) for s in empty.sampling.samples]
        )

    def test_v1_threshold_filter_does_not_invent_sampling(self) -> None:
        result = DetectorScanResult.from_dict(result_payload(version=1))
        filtered = result.at_confidence_threshold(0.9)
        self.assertEqual(1, filtered.version)
        self.assertEqual((), filtered.detections)
        self.assertNotIn("sampling", filtered.to_dict())
        for invalid in (True, -0.1, 1.1, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(DetectorProtocolError):
                result.at_confidence_threshold(invalid)

    def test_v2_schemas_match_serialized_contract_without_external_refs(self) -> None:
        result = DetectorScanResult.from_dict(result_payload())
        scan = AdmittedDetectorScan.from_dict(admitted_payload())
        schemas = []
        for name, payload in (
            ("detector-scan-result-v2.schema.json", result.to_dict()),
            ("admitted-detector-scan-v2.schema.json", scan.to_dict()),
        ):
            schema = json.loads((Path("schemas") / name).read_text(encoding="utf-8"))
            schemas.append(schema)
            self.assertEqual(set(payload), set(schema["required"]))
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(2, schema["properties"]["version"]["const"])
            ledger_schema = schema["$defs"]["sampling"]
            self.assertEqual(set(result.sampling.to_dict()), set(ledger_schema["required"]))
            self.assertFalse(ledger_schema["additionalProperties"])
            self.assertEqual(
                set(result.sampling.samples[0].to_dict()),
                set(schema["$defs"]["sample"]["required"]),
            )
            self.assertEqual(0.01, ledger_schema["properties"]["native_fps"]["minimum"])
            self.assertEqual(
                9999999, ledger_schema["properties"]["container_pts_regression_count"]["maximum"]
            )
            self.assertFalse(ledger_schema["properties"]["source_exhaustion_verified"]["const"])
            self._check_local_refs(schema, schema)
        for name in ("sample", "sampling", "detection", "bbox", "label", "safeId"):
            self.assertEqual(schemas[0]["$defs"][name], schemas[1]["$defs"][name])

    def _check_local_refs(self, node: object, schema: dict) -> None:
        if isinstance(node, dict):
            if "$ref" in node:
                self.assertTrue(node["$ref"].startswith("#/$defs/"))
                self.assertIn(node["$ref"].split("/")[-1], schema["$defs"])
            if node.get("type") == "object":
                self.assertFalse(node["additionalProperties"])
            for value in node.values():
                self._check_local_refs(value, schema)
        elif isinstance(node, list):
            for value in node:
                self._check_local_refs(value, schema)


if __name__ == "__main__":
    unittest.main()
