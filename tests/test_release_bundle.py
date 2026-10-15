from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.release_bundle import build_source_bundle


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise AssertionError("fixture Git command failed")


def _fixture_repository(root: Path) -> None:
    _git(root, "init", "--quiet")
    (root / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    (root / "README.md").write_text("staged content\n", encoding="utf-8")
    script = root / "scripts" / "run.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "baby-monitor"
version = "0.1.0"
description = "fixture"
license = { file = "LICENSE" }
dependencies = []

[project.optional-dependencies]
vision = ["opencv-python-headless>=4.10,<5"]
""".lstrip(),
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "update-index", "--chmod=+x", "scripts/run.sh")


class ReleaseBundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_index_only_and_has_safe_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-bundle-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            (root / "README.md").write_text("unstaged private change\n", encoding="utf-8")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"

            first_summary = build_source_bundle(root, first)
            second_summary = build_source_bundle(root, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_summary["sha256"], second_summary["sha256"])
            self.assertEqual(4, first_summary["staged_source_files"])
            self.assertEqual(5, first_summary["archive_files"])
            self.assertTrue(first_summary["includes_source_sbom"])
            self.assertFalse(first_summary["path_disclosed"])
            with tarfile.open(first, mode="r:gz") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                self.assertEqual(sorted(names[:-1]), names[:-1])
                self.assertEqual("baby-monitor-0.1.0/SBOM.cdx.json", names[-1])
                self.assertTrue(all(member.isfile() for member in members))
                self.assertTrue(all(member.uid == 0 and member.gid == 0 for member in members))
                self.assertTrue(all(member.mtime == 0 for member in members))
                executable = archive.getmember("baby-monitor-0.1.0/scripts/run.sh")
                self.assertEqual(0o755, executable.mode)
                readme = archive.extractfile("baby-monitor-0.1.0/README.md")
                assert readme is not None
                self.assertEqual(b"staged content\n", readme.read())
                sbom_file = archive.extractfile("baby-monitor-0.1.0/SBOM.cdx.json")
                assert sbom_file is not None
                sbom = json.loads(sbom_file.read())
                source_files = sbom["metadata"]["component"]["components"]
                readme_component = next(
                    item for item in source_files if item["name"] == "README.md"
                )
                self.assertEqual(
                    hashlib.sha256(b"staged content\n").hexdigest(),
                    readme_component["hashes"][0]["content"],
                )

    def test_output_is_exclusive_suffix_bounded_and_cleans_partial(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-bundle-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            output = root / "source.tar.gz"
            build_source_bundle(root, output)
            original = output.read_bytes()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_source_bundle(root, output)
            self.assertEqual(original, output.read_bytes())
            self.assertEqual([], list(root.glob("*.partial-*")))
            with self.assertRaisesRegex(RuntimeError, "tar.gz suffix"):
                build_source_bundle(root, root / "source.zip")
            with self.assertRaisesRegex(RuntimeError, "parent must already exist"):
                build_source_bundle(root, root / "missing" / "source.tar.gz")

    def test_cli_unsafe_index_fails_without_output_or_private_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-bundle-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            (root / "private.txt").write_text("password" + "=private-value\n", encoding="utf-8")
            _git(root, "add", "private.txt")
            output_path = root / "source.tar.gz"
            output = io.StringIO()
            errors = io.StringIO()

            with redirect_stdout(output), redirect_stderr(errors):
                code = main(
                    [
                        "release-bundle",
                        "--root",
                        str(root),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(2, code)
            self.assertFalse(output_path.exists())
            self.assertEqual("", output.getvalue())
            self.assertNotIn("private-value", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())


if __name__ == "__main__":
    unittest.main()
