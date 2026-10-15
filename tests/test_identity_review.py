from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.identity_review import (
    IdentityReviewError,
    ReviewBundle,
    ReviewCandidate,
    build_review_html,
    read_review_bundle,
    run_review_server,
    validate_selection,
    write_private_file,
    write_private_json,
)


class IdentityReviewTests(unittest.TestCase):
    def _root(self, temporary: str) -> Path:
        root = Path(temporary) / "private-review"
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
        return root

    def _bundle(self) -> ReviewBundle:
        candidates = tuple(
            ReviewCandidate(
                candidate_id=f"candidate_{index:016x}",
                ordinal=index,
                source_filename=f"candidate_{index:016x}.source.jpg",
                source_sha256=f"{index:064x}",
                preview_filename=f"candidate_{index:016x}.preview.jpg",
                preview_sha256=f"{index + 100:064x}",
                width=1280,
                height=720,
                bbox_xyxy=(0.1, 0.2, 0.4, 0.7),
                detector_confidence=0.95,
                mean_luma=100.0,
                luma_stddev=30.0,
            )
            for index in range(1, 7)
        )
        return ReviewBundle(
            review_id="review_20260811T120000Z_0123abcd",
            candidates=candidates,
            created_at_utc=datetime.now(UTC).isoformat(),
        )

    def _selection(self, bundle: ReviewBundle) -> dict[str, object]:
        return {
            "version": 1,
            "review_id": bundle.review_id,
            "labels": {
                candidate.candidate_id: "baby" if candidate.ordinal <= 3 else "adult"
                for candidate in bundle.candidates
            },
            "confirmations": {
                "baby_same_subject": True,
                "adult_same_subject": True,
                "live_camera_not_replay": True,
                "single_clear_face": True,
            },
        }

    def _write_bundle_page(self, root: Path, bundle: ReviewBundle) -> None:
        write_private_json(
            root / "candidates.json",
            bundle.to_private_dict(),
            maximum_bytes=1024**2,
        )
        previews = {
            candidate.candidate_id: b"\xff\xd8private-jpeg" for candidate in bundle.candidates
        }
        write_private_file(
            root / "review.local.html",
            build_review_html(bundle, previews),
            maximum_bytes=4 * 1024 * 1024,
        )

    def test_private_bundle_round_trip_and_exclusive_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            bundle = self._bundle()
            write_private_json(
                root / "candidates.json",
                bundle.to_private_dict(),
                maximum_bytes=1024**2,
            )
            self.assertEqual(bundle, read_review_bundle(root))
            self.assertEqual(stat.S_IMODE((root / "candidates.json").stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_private_json(
                    root / "candidates.json",
                    bundle.to_private_dict(),
                    maximum_bytes=1024**2,
                )

    def test_page_embeds_jpegs_and_never_embeds_private_paths_or_hashes(self) -> None:
        bundle = self._bundle()
        previews = {
            candidate.candidate_id: b"\xff\xd8private-jpeg" for candidate in bundle.candidates
        }
        page = build_review_html(bundle, previews).decode("utf-8")
        self.assertEqual(page.count("data:image/jpeg;base64,"), 6)
        self.assertIn("宝宝一组，大人一组", page)
        self.assertIn("不上传", page)
        for candidate in bundle.candidates:
            self.assertNotIn(candidate.source_filename, page)
            self.assertNotIn(candidate.source_sha256, page)

    def test_selection_requires_two_complete_confirmed_groups(self) -> None:
        bundle = self._bundle()
        value = self._selection(bundle)
        normalized = validate_selection(value, bundle)
        self.assertEqual(normalized["baby_count"], 3)
        self.assertEqual(normalized["adult_count"], 3)
        self.assertFalse(normalized["risk_decision_input"])
        self.assertFalse(normalized["notification_input"])

        value["labels"][bundle.candidates[0].candidate_id] = "reject"  # type: ignore[index]
        with self.assertRaisesRegex(IdentityReviewError, "baby selection"):
            validate_selection(value, bundle)
        value = self._selection(bundle)
        value["confirmations"]["adult_same_subject"] = False  # type: ignore[index]
        with self.assertRaisesRegex(IdentityReviewError, "confirmations"):
            validate_selection(value, bundle)

    def test_loopback_server_saves_once_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            bundle = self._bundle()
            self._write_bundle_page(root, bundle)
            thread = threading.Thread(target=run_review_server, args=(root,), daemon=True)
            thread.start()
            url_path = root / "review.url"
            for _ in range(100):
                if url_path.exists():
                    break
                time.sleep(0.01)
            url = url_path.read_text(encoding="ascii").strip()
            with urllib.request.urlopen(url, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("本地人脸录入审核", response.read().decode("utf-8"))
            payload = json.dumps(self._selection(bundle)).encode("utf-8")
            request = urllib.request.Request(
                url.replace("/?", "/selection?"),
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            saved = json.loads((root / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["baby_count"], 3)
            self.assertEqual(saved["adult_count"], 3)
            self.assertEqual(stat.S_IMODE((root / "selection.json").stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
