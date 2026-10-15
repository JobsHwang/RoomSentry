from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from baby_monitor.caviar_tracking import (
    CaviarTrackingError,
    import_caviar_tracking_ground_truth,
)
from baby_monitor.cli import main
from baby_monitor.object_tracker import IoUTrackerConfig
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.tracker_evaluation import (
    TrackerEvaluationError,
    build_iou_tracker_predictions,
    evaluate_tracker_association,
)

_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<dataset name="Meet_WalkSplit">
  <frame number="0">
    <objectlist>
      <object id="7"><box h="40" w="20" xc="100" yc="120"/></object>
      <object id="3"><box h="40" w="20" xc="250" yc="120"/></object>
    </objectlist>
    <grouplist><group id="1"/></grouplist>
  </frame>
  <frame number="1">
    <objectlist>
      <object id="7"><box h="40" w="20" xc="260" yc="120"/></object>
      <object id="3"><box h="40" w="20" xc="90" yc="120"/></object>
    </objectlist>
    <grouplist/>
  </frame>
</dataset>
"""


class CaviarTrackingTests(unittest.TestCase):
    def _write_source(
        self,
        root: Path,
        raw: bytes = _XML,
    ) -> tuple[Path, Path]:
        xml_path = root / "ground-truth.xml"
        xml_path.write_bytes(raw)
        admission_path = root / "source-admission.json"
        admission_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_id": "caviar_fixture",
                    "asset_kind": "dataset_archive",
                    "canonical_url": "https://example.com/caviar/ground-truth.xml",
                    "accessed_on": "2026-08-10",
                    "source_revision": "fixture_v1",
                    "upstream_name": "CAVIAR fixture",
                    "license": {
                        "name": "Creative Commons BY-SA",
                        "spdx_id": None,
                        "terms_url": "https://example.com/caviar/",
                        "commercial_use": "allowed",
                        "modification": "allowed",
                        "redistribution": "allowed",
                    },
                    "access": {
                        "gated": "no",
                        "authentication_required": "no",
                        "expected_bytes": len(raw),
                    },
                    "content": {
                        "observed_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                    "privacy": {
                        "real_minors": "no",
                        "faces": "not_applicable",
                        "audio": "not_applicable",
                        "device_or_account_ids": "no",
                        "precise_timestamps": "no",
                        "private_household_data": "no",
                    },
                    "labels": {
                        "origin": "human_nonexpert",
                        "human_verified": True,
                        "ground_truth": True,
                    },
                    "admission": {
                        "tier": "B",
                        "status": "verified_local",
                        "purpose": "model_evaluation",
                        "requires_user_approval": False,
                        "user_approval_recorded": False,
                        "git_allowed": False,
                        "ci_allowed": False,
                        "redistribution_allowed": False,
                        "training_allowed": False,
                        "reviewed_on": "2026-08-10",
                        "blocker_codes": [],
                        "restriction_codes": ["local_only"],
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return xml_path, admission_path

    def test_imports_individual_boxes_with_publisher_human_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xml_path, admission_path = self._write_source(Path(directory))
            imported = import_caviar_tracking_ground_truth(
                xml_path,
                admission_path,
            )
            admission = ExternalSourceAdmission.read(admission_path)

        self.assertEqual(2, imported.source_frame_count)
        self.assertEqual("caviar_meet_walk_split", imported.batch.camera_id)
        self.assertEqual(2, imported.imported_frame_count)
        self.assertEqual(4, imported.imported_detection_count)
        self.assertEqual(1, imported.ignored_group_count)
        self.assertEqual("synthetic_frame_index_at_25fps", imported.timestamp_semantics)
        self.assertEqual(
            0.04,
            (
                imported.batch.frames[1].occurred_at - imported.batch.frames[0].occurred_at
            ).total_seconds(),
        )
        self.assertEqual("admitted_publisher_human", imported.review.ground_truth_provenance)
        self.assertEqual((), imported.review.reviews)
        first_subjects = [item.subject_id for item in imported.review.assignments[:2]]
        second_subjects = [item.subject_id for item in imported.review.assignments[2:]]
        self.assertEqual(["subject_002", "subject_001"], first_subjects)
        self.assertEqual(["subject_001", "subject_002"], second_subjects)

        prediction = build_iou_tracker_predictions(
            imported.batch,
            IoUTrackerConfig(),
        )
        with self.assertRaisesRegex(TrackerEvaluationError, "source admission"):
            evaluate_tracker_association(
                imported.batch,
                imported.review,
                prediction,
            )
        report = evaluate_tracker_association(
            imported.batch,
            imported.review,
            prediction,
            source_admission=admission,
        )
        self.assertEqual("admitted_publisher_human", report.ground_truth_provenance)
        self.assertEqual(0, report.reviewer_count)
        self.assertFalse(report.detection_accuracy_evaluated)

        with tempfile.TemporaryDirectory() as directory:
            long_name_xml = _XML.replace(
                b'name="Meet_WalkSplit"',
                ('name="' + "A" * 80 + '"').encode(),
            )
            xml_path, admission_path = self._write_source(
                Path(directory),
                long_name_xml,
            )
            long_name_import = import_caviar_tracking_ground_truth(
                xml_path,
                admission_path,
            )
        self.assertEqual(80, len(long_name_import.batch.camera_id))

    def test_rejects_unsafe_xml_and_out_of_range_window(self) -> None:
        unsafe = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY y "z">]><dataset name="x"/>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml_path, admission_path = self._write_source(root, unsafe)
            with self.assertRaisesRegex(CaviarTrackingError, "declarations"):
                import_caviar_tracking_ground_truth(xml_path, admission_path)

            xml_path, admission_path = self._write_source(root, _XML)
            with self.assertRaisesRegex(CaviarTrackingError, "window"):
                import_caviar_tracking_ground_truth(
                    xml_path,
                    admission_path,
                    first_frame=0,
                    last_frame=2,
                )

            non_identifier_xml = _XML.replace(
                b'name="Meet_WalkSplit"',
                b'name="---"',
            )
            xml_path, admission_path = self._write_source(root, non_identifier_xml)
            with self.assertRaisesRegex(CaviarTrackingError, "camera identifier"):
                import_caviar_tracking_ground_truth(xml_path, admission_path)

    def test_cli_writes_path_free_batch_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml_path, admission_path = self._write_source(root)
            frames_path = root / "frames.json"
            review_path = root / "review.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "import-caviar-tracking-ground-truth",
                        "--xml",
                        str(xml_path),
                        "--source-admission",
                        str(admission_path),
                        "--frames-output",
                        str(frames_path),
                        "--review-output",
                        str(review_path),
                    ]
                )

            summary = json.loads(stdout.getvalue())
            missing_stderr = StringIO()
            with redirect_stderr(missing_stderr):
                missing_admission_code = main(
                    [
                        "validate-tracker-review",
                        "--frames",
                        str(frames_path),
                        "--review",
                        str(review_path),
                    ]
                )
            validation_stdout = StringIO()
            with redirect_stdout(validation_stdout):
                validation_code = main(
                    [
                        "validate-tracker-review",
                        "--frames",
                        str(frames_path),
                        "--review",
                        str(review_path),
                        "--source-admission",
                        str(admission_path),
                    ]
                )
            self.assertEqual(0, code)
            self.assertTrue(frames_path.is_file())
            self.assertTrue(review_path.is_file())
            self.assertFalse(summary["paths_disclosed"])
            self.assertEqual("admitted_publisher_human", summary["ground_truth_provenance"])
            self.assertEqual(2, missing_admission_code)
            self.assertIn("source admission", missing_stderr.getvalue())
            self.assertEqual(0, validation_code)
            self.assertEqual(
                "admitted_publisher_human",
                json.loads(validation_stdout.getvalue())["ground_truth_provenance"],
            )


if __name__ == "__main__":
    unittest.main()
