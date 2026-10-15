from __future__ import annotations

import argparse
import hashlib
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.run_yolo_mage_co_resident_public import (
    _bounded_request_count,
    _private_file,
    _python_executable,
    _safe_hash,
    _write_private,
    build_parser,
)


class YoloMageCoResidentPublicTests(unittest.TestCase):
    def test_python_runtime_validates_target_but_preserves_venv_style_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "python-runtime"
            target.write_bytes(b"test-runtime")
            target.chmod(0o755)
            link = root / "python"
            link.symlink_to(target)
            admitted = _python_executable(str(link))
            self.assertEqual(link, admitted)
            self.assertTrue(admitted.is_symlink())
            self.assertNotEqual(admitted, admitted.resolve())

            target.chmod(0o775)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                _python_executable(str(link))

    def test_repeated_public_frames_and_predicate_remain_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "--yolo-python",
                "/runtime/yolo-python",
                "--yolo-weight",
                "/model/yolo.pt",
                "--yolo-weight-sha256",
                "a" * 64,
                "--mage-python",
                "/runtime/mage-python",
                "--mage-model",
                "/model/mage",
                "--mage-manifest",
                "/manifest/mage.json",
                "--public-frame",
                "/public/frame-01.jpg",
                "--public-frame-sha256",
                "b" * 64,
                "--public-frame",
                "/public/frame-02.jpg",
                "--public-frame-sha256",
                "c" * 64,
                "--predicate",
                "posture_transition_visible",
                "--output",
                "/private/result.json",
                "--request-count",
                "32",
            ]
        )
        self.assertEqual(
            ["/public/frame-01.jpg", "/public/frame-02.jpg"],
            args.public_frame,
        )
        self.assertEqual(["b" * 64, "c" * 64], args.public_frame_sha256)
        self.assertEqual("posture_transition_visible", args.predicate)
        self.assertEqual(32, args.request_count)

    def test_private_input_and_output_are_hash_bound_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            root.chmod(0o700)
            source = root / "public.jpg"
            payload = b"public-benchmark-frame"
            source.write_bytes(payload)
            source.chmod(0o600)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(
                payload,
                _private_file(source, expected_hash=digest, max_bytes=1024),
            )
            with self.assertRaisesRegex(RuntimeError, "admission"):
                _private_file(source, expected_hash="0" * 64, max_bytes=1024)

            output = root / "result.local.json"
            _write_private(output, {"version": 1, "paths_disclosed": False})
            self.assertEqual(0o600, stat.S_IMODE(output.stat().st_mode))
            with self.assertRaisesRegex(RuntimeError, "output"):
                _write_private(output, {"version": 1})

    def test_hash_contract_rejects_non_sha256(self) -> None:
        self.assertEqual("a" * 64, _safe_hash("a" * 64, "fixture"))
        for value in ("a" * 63, "A" * 64, "g" * 64):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "fixture"),
            ):
                _safe_hash(value, "fixture")

    def test_request_count_is_strictly_bounded(self) -> None:
        self.assertEqual(1, _bounded_request_count("1"))
        self.assertEqual(32, _bounded_request_count("32"))
        for value in ("0", "33", "1.5"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _bounded_request_count(value)


if __name__ == "__main__":
    unittest.main()
