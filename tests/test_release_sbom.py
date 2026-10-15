from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.release_audit import _tracked_entries
from baby_monitor.release_sbom import (
    _generate_source_sbom,
    generate_source_sbom,
    source_sbom_json,
    write_source_sbom,
)


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
dev = ["ruff>=0.9,<1"]
""".lstrip(),
        encoding="utf-8",
    )
    _git(root, "add", ".")


class ReleaseSBOMTests(unittest.TestCase):
    def test_captured_sbom_snapshot_uses_blob_ids_after_index_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-sbom-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            captured = _tracked_entries(root)
            (root / "README.md").write_text("later staged content\n", encoding="utf-8")
            _git(root, "add", "README.md")

            document = _generate_source_sbom(root, captured)

            files = {item["name"]: item for item in document["metadata"]["component"]["components"]}
            self.assertEqual(
                hashlib.sha256(b"staged content\n").hexdigest(),
                files["README.md"]["hashes"][0]["content"],
            )

    def test_sbom_is_deterministic_index_only_and_lists_declared_scopes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-sbom-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            (root / "README.md").write_text("unstaged private change\n", encoding="utf-8")

            first = source_sbom_json(root)
            second = source_sbom_json(root)
            document = json.loads(first)

            self.assertEqual(first, second)
            self.assertEqual("CycloneDX", document["bomFormat"])
            self.assertEqual("1.6", document["specVersion"])
            self.assertNotIn("serialNumber", document)
            self.assertNotIn("timestamp", document["metadata"])
            root_component = document["metadata"]["component"]
            self.assertEqual("pkg:pypi/baby-monitor@0.1.0", root_component["bom-ref"])
            files = {item["name"]: item for item in root_component["components"]}
            expected_hash = hashlib.sha256(b"staged content\n").hexdigest()
            self.assertEqual(expected_hash, files["README.md"]["hashes"][0]["content"])
            self.assertNotIn("unstaged private change", first)

            dependencies = {item["name"]: item for item in document["components"]}
            self.assertEqual("excluded", dependencies["setuptools"]["scope"])
            self.assertEqual("optional", dependencies["opencv-python-headless"]["scope"])
            self.assertEqual("excluded", dependencies["ruff"]["scope"])
            self.assertEqual([], document["dependencies"][0]["dependsOn"])

    def test_cli_refuses_unsafe_index_without_disclosing_content_or_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-sbom-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            secret = "password" + "=private-value\n"
            (root / "private.txt").write_text(secret, encoding="utf-8")
            _git(root, "add", "private.txt")
            output = io.StringIO()
            errors = io.StringIO()

            with redirect_stdout(output), redirect_stderr(errors):
                code = main(["release-sbom", "--root", str(root)])

            self.assertEqual(2, code)
            self.assertEqual("", output.getvalue())
            self.assertIn("must pass release-audit", errors.getvalue())
            self.assertNotIn("private-value", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())

    def test_generated_sbom_self_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-sbom-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            (root / "release.cdx.json").write_text("{}\n", encoding="utf-8")
            _git(root, "add", "release.cdx.json")

            with self.assertRaisesRegex(RuntimeError, "self-reference"):
                generate_source_sbom(root)

    def test_atomic_output_is_path_free_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-sbom-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            output = root / "artifacts" / "source.cdx.json"
            output.parent.mkdir()

            summary = write_source_sbom(root, output)

            self.assertEqual("written", summary["status"])
            self.assertFalse(summary["path_disclosed"])
            self.assertFalse(summary["content_disclosed"])
            self.assertEqual(summary["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            original = output.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_source_sbom(root, output)
            self.assertEqual(original, output.read_bytes())
            self.assertEqual([], list(output.parent.glob("*.partial-*")))

    def test_fixture_project_metadata_matches_sbom_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-sbom-") as directory:
            root = Path(directory)
            _fixture_repository(root)
            document = generate_source_sbom(root)
            root_component = document["metadata"]["component"]
            metadata_properties = {
                item["name"]: item["value"] for item in document["metadata"]["properties"]
            }

            self.assertEqual("baby-monitor", root_component["name"])
            self.assertEqual("Apache-2.0", root_component["licenses"][0]["expression"])
            self.assertEqual("git-index-only", metadata_properties["baby-monitor:generated-from"])


if __name__ == "__main__":
    unittest.main()
