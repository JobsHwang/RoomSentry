from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.source_admission import (
    LARGE_MODEL_WEIGHT_BYTES,
    ExternalSourceAdmission,
    SourceAdmissionError,
)

_FIXTURE_PATH = Path("tests/fixtures/valid_external_source_admission.json")


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _local_manifest(content: bytes) -> dict[str, object]:
    value = _fixture()
    value["access"]["expected_bytes"] = len(content)  # type: ignore[index]
    value["content"] = {
        "observed_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return value


class ExternalSourceAdmissionTests(unittest.TestCase):
    def test_fixture_round_trips_and_matches_schema_root(self) -> None:
        record = ExternalSourceAdmission.read(_FIXTURE_PATH)
        self.assertEqual(record, ExternalSourceAdmission.from_json(record.to_json()))
        schema = json.loads(
            Path("schemas/external-source-admission-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(record.to_dict()))
        self.assertEqual("A", record.admission.tier)
        self.assertTrue(record.labels.ground_truth)

        public = record.to_public_dict()
        self.assertTrue(public["valid"])
        self.assertFalse(public["asset_read"])
        self.assertFalse(public["asset_verified"])
        self.assertFalse(public["paths_disclosed"])
        self.assertFalse(public["urls_disclosed"])
        self.assertNotIn(record.canonical_url or "missing", json.dumps(public))

    def test_unknown_provenance_is_allowed_only_while_blocked(self) -> None:
        value = _fixture()
        value.update(
            {
                "canonical_url": None,
                "accessed_on": None,
                "source_revision": None,
                "upstream_name": None,
            }
        )
        value["license"] = {
            "name": None,
            "spdx_id": None,
            "terms_url": None,
            "commercial_use": "unknown",
            "modification": "unknown",
            "redistribution": "unknown",
        }
        value["access"] = {
            "gated": "unknown",
            "authentication_required": "unknown",
            "expected_bytes": None,
        }
        value["content"] = {"observed_bytes": None, "sha256": None}
        value["privacy"] = {
            "real_minors": "unknown",
            "faces": "unknown",
            "audio": "unknown",
            "device_or_account_ids": "unknown",
            "precise_timestamps": "unknown",
            "private_household_data": "unknown",
        }
        value["labels"] = {
            "origin": "unknown",
            "human_verified": False,
            "ground_truth": False,
        }
        value["admission"] = {
            "tier": None,
            "status": "blocked",
            "purpose": "research_only",
            "requires_user_approval": False,
            "user_approval_recorded": False,
            "git_allowed": False,
            "ci_allowed": False,
            "redistribution_allowed": False,
            "training_allowed": False,
            "reviewed_on": None,
            "blocker_codes": ["license_unknown", "size_unknown"],
            "restriction_codes": [],
        }
        record = ExternalSourceAdmission.from_dict(value)
        self.assertEqual("blocked", record.admission.status)
        self.assertFalse(record.to_public_dict()["ready_for_download"])

        value["admission"]["status"] = "approved_for_download"  # type: ignore[index]
        value["admission"]["blocker_codes"] = []  # type: ignore[index]
        with self.assertRaisesRegex(SourceAdmissionError, "unknown provenance"):
            ExternalSourceAdmission.from_dict(value)

    def test_real_minor_stock_media_is_local_tier_b_and_not_ground_truth(self) -> None:
        value = _fixture()
        value["asset_kind"] = "media_asset"
        value["license"]["name"] = "Pexels License"  # type: ignore[index]
        value["license"]["spdx_id"] = None  # type: ignore[index]
        value["license"]["redistribution"] = "restricted"  # type: ignore[index]
        value["privacy"]["real_minors"] = "yes"  # type: ignore[index]
        value["privacy"]["faces"] = "yes"  # type: ignore[index]
        value["labels"] = {
            "origin": "publisher_description",
            "human_verified": False,
            "ground_truth": False,
        }
        value["admission"].update(  # type: ignore[union-attr]
            {
                "tier": "B",
                "purpose": "dynamic_failure_baseline",
                "requires_user_approval": True,
                "user_approval_recorded": True,
                "git_allowed": False,
                "ci_allowed": False,
                "redistribution_allowed": False,
                "training_allowed": False,
                "restriction_codes": [
                    "real_minor_local_only",
                    "no_ground_truth_claim",
                ],
            }
        )
        record = ExternalSourceAdmission.from_dict(value)
        self.assertEqual("B", record.admission.tier)
        self.assertFalse(record.labels.ground_truth)

        tier_a = copy.deepcopy(value)
        tier_a["admission"]["tier"] = "A"  # type: ignore[index]
        with self.assertRaisesRegex(SourceAdmissionError, "Tier A"):
            ExternalSourceAdmission.from_dict(tier_a)

    def test_vendor_and_model_labels_cannot_be_ground_truth(self) -> None:
        for origin in (
            "vendor_rule",
            "model_generated",
            "publisher_description",
            "contributor_self_report",
        ):
            value = _fixture()
            value["labels"] = {
                "origin": origin,
                "human_verified": False,
                "ground_truth": True,
            }
            with (
                self.subTest(origin=origin),
                self.assertRaisesRegex(SourceAdmissionError, "ground truth"),
            ):
                ExternalSourceAdmission.from_dict(value)

    def test_large_model_download_requires_explicit_recorded_approval(self) -> None:
        value = _fixture()
        value["asset_kind"] = "model_weights"
        value["access"]["expected_bytes"] = LARGE_MODEL_WEIGHT_BYTES  # type: ignore[index]
        value["content"] = {"observed_bytes": None, "sha256": None}
        value["privacy"]["faces"] = "not_applicable"  # type: ignore[index]
        value["admission"].update(  # type: ignore[union-attr]
            {
                "status": "approved_for_download",
                "git_allowed": False,
                "ci_allowed": False,
                "redistribution_allowed": False,
                "training_allowed": False,
            }
        )
        with self.assertRaisesRegex(SourceAdmissionError, "require user approval"):
            ExternalSourceAdmission.from_dict(value)

        value["admission"]["requires_user_approval"] = True  # type: ignore[index]
        with self.assertRaisesRegex(SourceAdmissionError, "without recorded"):
            ExternalSourceAdmission.from_dict(value)

        value["admission"]["user_approval_recorded"] = True  # type: ignore[index]
        record = ExternalSourceAdmission.from_dict(value)
        self.assertTrue(record.to_public_dict()["ready_for_download"])

    def test_local_asset_verification_is_exact_and_path_free(self) -> None:
        content = b"synthetic-source"
        record = ExternalSourceAdmission.from_dict(_local_manifest(content))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset.bin"
            asset.write_bytes(content)
            public = record.verify_asset(asset).to_public_dict()
            self.assertTrue(public["asset_read"])
            self.assertTrue(public["asset_verified"])
            self.assertEqual(len(content), public["observed_bytes"])
            self.assertNotIn(str(root), json.dumps(public))

            asset.write_bytes(b"different-source")
            with self.assertRaisesRegex(SourceAdmissionError, "does not match"):
                record.verify_asset(asset)

    def test_local_asset_symlink_is_rejected(self) -> None:
        content = b"synthetic-source"
        record = ExternalSourceAdmission.from_dict(_local_manifest(content))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "asset.bin"
            target.write_bytes(content)
            link = root / "asset-link.bin"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(SourceAdmissionError, "missing or unsafe"):
                record.verify_asset(link)

    def test_strict_json_rejects_duplicates_nonstandard_numbers_and_extra_fields(self) -> None:
        with self.assertRaisesRegex(SourceAdmissionError, "duplicate keys"):
            ExternalSourceAdmission.from_json('{"version":1,"version":1}')
        with self.assertRaisesRegex(SourceAdmissionError, "non-standard number"):
            ExternalSourceAdmission.from_json('{"version":NaN}')
        value = _fixture()
        value["private_path"] = "/private/source"
        with self.assertRaisesRegex(SourceAdmissionError, "fields do not match"):
            ExternalSourceAdmission.from_dict(value)

    def test_cli_validates_and_verifies_without_disclosing_paths_or_urls(self) -> None:
        content = b"synthetic-source"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "admission.json"
            asset = root / "asset.bin"
            manifest.write_text(
                json.dumps(_local_manifest(content)),
                encoding="utf-8",
            )
            asset.write_bytes(content)

            validated = io.StringIO()
            with redirect_stdout(validated):
                validate_code = main(["validate-source-admission", str(manifest)])
            self.assertEqual(0, validate_code)
            validate_payload = json.loads(validated.getvalue())
            self.assertFalse(validate_payload["asset_read"])
            self.assertFalse(validate_payload["asset_verified"])

            verified = io.StringIO()
            with redirect_stdout(verified):
                verify_code = main(
                    [
                        "verify-source-asset",
                        "--manifest",
                        str(manifest),
                        "--asset",
                        str(asset),
                    ]
                )
            self.assertEqual(0, verify_code)
            verify_payload = json.loads(verified.getvalue())
            self.assertTrue(verify_payload["asset_verified"])
            combined = validated.getvalue() + verified.getvalue()
            self.assertNotIn(str(root), combined)
            self.assertNotIn("https://", combined)


if __name__ == "__main__":
    unittest.main()
