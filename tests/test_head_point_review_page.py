from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import re
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.head_point_review import HeadPointReviewError, fingerprint
from baby_monitor.live_detection import LiveFrame, encode_bgr_png
from scripts.prepare_head_point_review import admit_answers, build_page, prepare, read_bundle
from scripts.serve_head_point_review import BoundedServer, handler_type, serve


@unittest.skipUnless(os.name == "posix", "private household artifact helper requires POSIX")
class HeadPointReviewPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "synthetic.png"
        frame = LiveFrame(
            "camera_test", 1, 32, 16, datetime(2026, 10, 10, tzinfo=UTC), 1, b"\x18\x80\xe0" * 512
        )
        self.data = encode_bgr_png(frame)
        self.write(self.image, self.data)
        self.source = self.root / "source.json"
        self.source_value = {
            "version": 1,
            "items": [
                {
                    "item_id": "frame_01",
                    "camera_id": "camera_test",
                    "frame_sha256": hashlib.sha256(self.data).hexdigest(),
                    "frame_received_at": "2026-10-10T00:00:00+00:00",
                    "width": 32,
                    "height": 16,
                    "image_path": str(self.image),
                }
            ],
        }
        self.write(self.source, json.dumps(self.source_value).encode())
        self.output = self.root / "new_review"

    def write(self, path, data):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)

    def build(self):
        return prepare(self.source, self.output, synthetic=True)

    def answers(self, m):
        return {
            "version": 1,
            "scope": m["scope"],
            "manifest_sha256": fingerprint(m),
            "reviewer_id": "reviewer_synthetic",
            "provenance": "synthetic",
            "answers": [
                {
                    "item_id": "frame_01",
                    "frame_sha256": m["items"][0]["frame_sha256"],
                    "decision": "unknown",
                    "points": [],
                    "inventory_complete": False,
                    "reviewed": True,
                }
            ],
        }

    def test_self_contained_private_package_and_no_paths_or_models(self):
        result = self.build()
        m, p, page = read_bundle(self.output)
        self.assertEqual(1, result["item_count"])
        self.assertTrue(p["synthetic"])
        self.assertEqual(0o700, self.output.stat().st_mode & 0o777)
        for file in self.output.iterdir():
            self.assertEqual(0o600, file.stat().st_mode & 0o777)
        text = page.decode()
        self.assertIn("data:image/png;base64,", text)
        self.assertIn("connect-src 'none'", text)
        self.assertIn("provenance:'synthetic'", text)
        self.assertIn(fingerprint(m), text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("fetch(", text)
        self.assertNotIn("XMLHttpRequest", text)
        self.assertNotIn("__MANIFEST__", text)

    def test_existing_output_is_not_overwritten(self):
        self.build()
        before = (self.output / "review.local.html").read_bytes()
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(before, (self.output / "review.local.html").read_bytes())

    def test_tampered_source_hash_or_dimensions_rejected_before_output(self):
        for change in ({"frame_sha256": "a" * 64}, {"width": 31}):
            value = copy.deepcopy(self.source_value)
            value["items"][0].update(change)
            source = (
                self.root
                / f"source_{len(change['frame_sha256']) if 'frame_sha256' in change else 31}.json"
            )
            self.write(source, json.dumps(value).encode())
            with self.subTest(change=change), self.assertRaises(HeadPointReviewError):
                prepare(source, self.output)
            self.assertFalse(self.output.exists())

    def test_shared_readable_source_and_symlink_are_rejected(self):
        self.source.chmod(0o644)
        with self.assertRaises(ValueError):
            self.build()
        self.source.chmod(0o600)
        link = self.root / "linked.json"
        link.symlink_to(self.source)
        with self.assertRaises(ValueError):
            prepare(link, self.output)

    def test_admitted_download_is_bound_private_and_exclusive(self):
        self.build()
        m, _, _ = read_bundle(self.output)
        source = self.root / "answers.json"
        self.write(source, json.dumps(self.answers(m)).encode())
        output = self.root / "admitted.json"
        result = admit_answers(self.output, source, output)
        self.assertEqual(1, result["answer_count"])
        self.assertEqual("synthetic", result["provenance"])
        self.assertEqual(0o600, output.stat().st_mode & 0o777)
        with self.assertRaises(FileExistsError):
            admit_answers(self.output, source, output)

    def test_synthetic_page_answer_cannot_be_promoted_to_human(self):
        self.build()
        m, _, _ = read_bundle(self.output)
        answer = self.answers(m) | {"provenance": "human_user_review"}
        source = self.root / "promoted.json"
        self.write(source, json.dumps(answer).encode())
        with self.assertRaises(HeadPointReviewError):
            admit_answers(self.output, source, self.root / "admitted.json")

    def test_page_tampering_invalidates_package_and_admission(self):
        self.build()
        path = self.output / "review.local.html"
        with path.open("ab") as stream:
            stream.write(b" changed")
        with self.assertRaises(HeadPointReviewError):
            read_bundle(self.output)

    def test_build_page_checks_input_order_length_and_png_header(self):
        self.build()
        m, _, _ = read_bundle(self.output)
        with self.assertRaises(HeadPointReviewError):
            build_page(m, [])
        with self.assertRaises(HeadPointReviewError):
            build_page(m, [b"not PNG"])

    def test_template_tokens_in_legal_ids_are_not_recursively_replaced(self):
        self.build()
        manifest, _, _ = read_bundle(self.output)
        for token in (
            "__CARDS__",
            "__MANIFEST__",
            "__FINGERPRINT__",
            "__PROVENANCE__",
            "__REVIEWER__",
        ):
            for field in ("review_id", "item_id", "camera_id"):
                value = copy.deepcopy(manifest)
                if field == "review_id":
                    value[field] = token
                else:
                    value["items"][0][field] = token
                with self.subTest(token=token, field=field):
                    page = build_page(value, [self.data], synthetic=True)
                    embedded = re.search(r"const manifest=(.*);\nconst answers=", page).group(1)
                    self.assertEqual(value, json.loads(embedded))
                    self.assertIn(f"manifest_sha256:'{fingerprint(value)}'", page)
                    self.assertIn("provenance:'synthetic'", page)

    def test_image_load_guard_applies_to_marked_and_no_visible_head_not_unknown(self):
        self.build()
        _, _, page = read_bundle(self.output)
        text = page.decode()
        self.assertIn("image.complete && image.naturalWidth===item.width", text)
        self.assertIn("a.decision==='unknown'||imageReady(index)", text)
        self.assertIn("image.addEventListener('error',showImageStatus)", text)
        self.assertIn("answers.every((a,index)=>valid(a,index))", text)

    def test_invalid_source_fingerprint_in_package_is_rejected(self):
        self.build()
        path = self.output / "package.json"
        package = json.loads(path.read_bytes())
        package["source_manifest_sha256"] = None
        path.write_text(json.dumps(package))
        with self.assertRaises(HeadPointReviewError):
            read_bundle(self.output)

    def test_loopback_server_has_only_frozen_page_and_health_routes(self):
        self.build()
        _, _, page = read_bundle(self.output)
        server = BoundedServer(("127.0.0.1", 0), handler_type(page))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        self.assertEqual("127.0.0.1", server.server_address[0])
        for method, path, headers, status in (
            ("GET", "/review", {}, 200),
            ("GET", "/health", {}, 200),
            ("GET", "/manifest.json", {}, 404),
            ("GET", "/../private", {}, 404),
            ("GET", "/review?source=private", {}, 404),
            ("POST", "/review", {}, 405),
            ("GET", "/review", {"Host": "attacker.invalid"}, 403),
            ("GET", "/review", {"Origin": "https://attacker.invalid"}, 403),
            ("GET", "/review", {"Sec-Fetch-Site": "cross-site"}, 403),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            try:
                conn.request(method, path, headers=headers)
                response = conn.getresponse()
                body = response.read()
                with self.subTest(method=method, path=path, headers=headers):
                    self.assertEqual(status, response.status)
                    self.assertEqual("no-store", response.getheader("Cache-Control"))
                    self.assertEqual("DENY", response.getheader("X-Frame-Options"))
                    self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
                    if method == "GET" and path == "/review" and status == 200:
                        self.assertEqual(page, body)
                    else:
                        self.assertNotIn(b"data:image/png", body)
            finally:
                conn.close()

    def test_server_invalid_bounds_do_not_bind(self):
        for port, seconds in ((True, 1), (-1, 1), (0, True), (0, 0), (0, 86401)):
            with self.subTest(port=port, seconds=seconds), self.assertRaises(ValueError):
                serve(self.root, port=port, seconds=seconds)


if __name__ == "__main__":
    unittest.main()
