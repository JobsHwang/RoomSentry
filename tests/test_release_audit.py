from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.release_audit import _audit_entries, _tracked_entries, audit_release


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


class ReleaseAuditTests(unittest.TestCase):
    def test_captured_index_entries_are_read_by_immutable_blob_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            tracked = root / "README.md"
            tracked.write_text("safe fixture\n", encoding="utf-8")
            _git(root, "add", "README.md")
            captured = _tracked_entries(root)
            tracked.write_text("password" + "=later-private-value\n", encoding="utf-8")
            _git(root, "add", "README.md")

            captured_report = _audit_entries(root, captured, max_file_bytes=10 * 1024 * 1024)
            current_report = audit_release(root)

            self.assertTrue(captured_report.passed)
            self.assertFalse(current_report.passed)
            self.assertEqual(
                {"password_assignment"},
                {item.code for item in current_report.violations},
            )

    def test_safe_index_passes_schema_shape_and_ignores_unstaged_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            tracked = root / "README.md"
            tracked.write_text("safe fixture\n", encoding="utf-8")
            (root / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
            fixture = root / "tests" / "credential_fixture.py"
            fixture.parent.mkdir()
            fixture.write_text(
                "# release-audit: allow=fixture-credential\n"
                + "URI = 'rtsp://"
                + "user:secret@127.0.0.1/live'\n",
                encoding="utf-8",
            )
            _git(root, "add", "README.md", ".env.example", "tests/credential_fixture.py")
            tracked.write_text("password" + "=unstaged-value\n", encoding="utf-8")

            report = audit_release(root)

            self.assertTrue(report.passed)
            self.assertEqual(3, report.tracked_files)
            self.assertEqual(3, report.scanned_text_files)
            self.assertFalse(report.to_dict()["content_disclosed"])
            schema = json.loads(
                Path("schemas/release-audit-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(schema["required"]), set(report.to_dict()))
            self.assertEqual(
                set(schema["$defs"]["violation"]["required"]),
                {"code", "path", "line"},
            )

    def test_machine_home_paths_are_reported_but_placeholders_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            (root / "docs").mkdir()
            (root / "docs" / "concrete.md").write_text(
                # release-audit: allow=fixture-home-path
                "run from /home/operator/project and /mnt/c/Users/alice/Documents\n",
                encoding="utf-8",
            )
            (root / "docs" / "placeholder.md").write_text(
                "run from ~/projects and /mnt/c/Users/<WINDOWS_USER>/Documents\n",
                encoding="utf-8",
            )
            _git(root, "add", ".")

            report = audit_release(root)
            codes = {item.code: item.path for item in report.violations}

            self.assertIn("machine_home_path", codes)
            self.assertEqual("docs/concrete.md", codes["machine_home_path"])
            self.assertNotIn("docs/placeholder.md", set(codes.values()))

    def test_portable_home_path_variants_are_rejected_without_echoing_values(self) -> None:
        operator = "fixture-operator"
        slash = "/"
        backslash = "\\"
        paths = (
            "/home/" + operator + "/project",
            "/Users/" + operator + "/project",
            "C:" + backslash + "Users" + backslash + operator + backslash + "project",
            "C:" + slash + "Users" + slash + operator + slash + "project",
            "/mnt/c/Users/" + operator + "/project",
            backslash * 2
            + "wsl$"
            + backslash
            + "Ubuntu"
            + backslash
            + "home"
            + backslash
            + operator
            + backslash
            + "project",
            "//wsl.localhost/Ubuntu/home/" + operator + "/project",
        )
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            for index, value in enumerate(paths):
                (root / f"path-{index}.md").write_text(value + "\n", encoding="utf-8")
            (root / "safe.md").write_text(
                "~/projects and C:/Users/<USER>/project\n", encoding="utf-8"
            )
            _git(root, "add", ".")

            report = audit_release(root)

            self.assertEqual(len(paths), len(report.violations))
            self.assertEqual({"machine_home_path"}, {v.code for v in report.violations})
            self.assertNotIn(operator, report.to_json())
            self.assertNotIn("safe.md", {v.path for v in report.violations})

    def test_sensitive_index_reports_only_codes_paths_and_lines(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            secret_text = "password" + "=family-secret\n"
            (root / "settings.txt").write_text(secret_text, encoding="utf-8")
            (root / "connection.txt").write_text(
                "rtsp://" + "user:real-private-value@127.0.0.1/live\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "TOKEN=" + "ghp_" + ("a" * 24) + "\n",
                encoding="utf-8",
            )
            media = root / "data" / "camera.mp4"
            media.parent.mkdir()
            media.write_bytes(b"fixture")
            (root / "weight.safetensors").write_bytes(b"fixture")
            (root / "large.txt").write_bytes(b"x" * 256)
            _git(root, "add", "-f", ".")

            report = audit_release(root, max_file_bytes=128)
            payload = report.to_json(indent=None)
            codes = {item.code for item in report.violations}

            self.assertFalse(report.passed)
            self.assertTrue(
                {
                    "forbidden_path",
                    "forbidden_extension",
                    "file_too_large",
                    "password_assignment",
                    "credential_url",
                }.issubset(codes)
            )
            self.assertNotIn("provider_token", codes)
            self.assertNotIn("family-secret", payload)
            self.assertNotIn(str(root), payload)
            self.assertFalse(json.loads(payload)["content_disclosed"])

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "release-audit",
                        "--root",
                        str(root),
                        "--max-file-bytes",
                        "128",
                    ]
                )
            self.assertEqual(1, code)
            self.assertNotIn("family-secret", output.getvalue())
            self.assertNotIn(str(root), output.getvalue())

    def test_worktree_mode_scans_modified_and_untracked_but_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
            tracked = root / "README.md"
            tracked.write_text("safe fixture\n", encoding="utf-8")
            _git(root, "add", ".gitignore", "README.md")
            tracked.write_text(
                "password" + "=modified-private-value\n",
                encoding="utf-8",
            )
            (root / "new.txt").write_text(
                "rtsp://" + "user:new-private-value@127.0.0.1/live\n",
                encoding="utf-8",
            )
            (root / "ignored.txt").write_text(
                "password" + "=ignored-private-value\n",
                encoding="utf-8",
            )

            report = audit_release(root, source="worktree")
            payload = report.to_json(indent=None)

            self.assertFalse(report.passed)
            self.assertEqual("worktree", report.source)
            self.assertEqual(2, report.tracked_files)
            self.assertEqual(1, report.untracked_files)
            self.assertEqual(
                {"password_assignment", "credential_url"},
                {item.code for item in report.violations},
            )
            self.assertNotIn("modified-private-value", payload)
            self.assertNotIn("new-private-value", payload)
            self.assertNotIn("ignored-private-value", payload)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["release-audit", "--root", str(root), "--worktree"])
            self.assertEqual(1, code)
            self.assertEqual(
                "worktree",
                json.loads(output.getvalue())["source"],
            )

    def test_worktree_mode_rejects_symlinks_without_reading_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            target = root / "target.txt"
            target.write_text("safe fixture\n", encoding="utf-8")
            link = root / "link.txt"
            link.symlink_to(target)

            report = audit_release(root, source="worktree")

            self.assertFalse(report.passed)
            self.assertEqual(2, report.untracked_files)
            self.assertIn(
                "unsupported_worktree_type",
                {item.code for item in report.violations},
            )

    def test_invalid_root_returns_two_without_disclosing_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-audit-") as directory:
            missing = Path(directory) / "private-missing-root"
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main(["release-audit", "--root", str(missing)])
            self.assertEqual(2, code)
            self.assertIn("repository root is missing or invalid", errors.getvalue())
            self.assertNotIn(str(missing), errors.getvalue())


if __name__ == "__main__":
    unittest.main()
