from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from baby_monitor.archive import (
    ArchiveConflictError,
    ArchiveError,
    ArchiveManifest,
    ArchiveUnavailableError,
    FilesystemArchiveAdapter,
    build_archive_manifest,
)
from baby_monitor.cli import main


class ArchiveTests(unittest.TestCase):
    @staticmethod
    def _source_tree(root: Path) -> tuple[Path, tuple[Path, ...]]:
        source = root / "source"
        event_dir = source / "events/2026/08/03/cam_01"
        report_dir = source / "reports/2026/08/03"
        event_dir.mkdir(parents=True)
        report_dir.mkdir(parents=True)
        clip = event_dir / "20260803T010203_000000Z_cam_01_event_01.mp4"
        frame = event_dir / "20260803T010203_000000Z_cam_01_event_01_f01.jpg"
        report = report_dir / "2026-08-03_daily_report.md"
        clip.write_bytes(b"synthetic-clip")
        frame.write_bytes(b"synthetic-frame")
        report.write_text("# Synthetic report\n", encoding="utf-8")
        return source, (clip, frame, report)

    @staticmethod
    def _manifest(source: Path, files: tuple[Path, ...]) -> ArchiveManifest:
        return build_archive_manifest(
            archive_id="archive_20260803_event_01",
            source_root=source,
            files=files,
            created_at=datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC),
        )

    def test_copy_is_hash_verified_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "target"
            target.mkdir()
            manifest = self._manifest(source, files)

            first = FilesystemArchiveAdapter(
                source_root=source,
                target_root=target,
            ).archive(manifest)
            self.assertEqual(3, first.copied_items)
            self.assertEqual(0, first.reused_items)
            self.assertEqual(manifest.total_bytes, first.copied_bytes)
            for item in manifest.items:
                archived = target.joinpath(*item.relative_path.split("/"))
                self.assertEqual(item.size_bytes, archived.stat().st_size)
                self.assertEqual(item.sha256, hashlib.sha256(archived.read_bytes()).hexdigest())
            completion = target / "archive-manifests/archive_20260803_event_01.json"
            self.assertEqual(manifest, ArchiveManifest.read(completion))
            self.assertEqual([], list(target.rglob("*.partial")))

            second = FilesystemArchiveAdapter(
                source_root=source,
                target_root=target,
            ).archive(manifest)
            self.assertEqual(0, second.copied_items)
            self.assertEqual(3, second.reused_items)
            self.assertEqual(0, second.copied_bytes)

    def test_existing_different_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "target"
            target.mkdir()
            manifest = self._manifest(source, files)
            first = manifest.items[0]
            conflict = target.joinpath(*first.relative_path.split("/"))
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b"unknown-existing-content")

            with self.assertRaisesRegex(ArchiveConflictError, "different content"):
                FilesystemArchiveAdapter(
                    source_root=source,
                    target_root=target,
                ).archive(manifest)
            self.assertEqual(b"unknown-existing-content", conflict.read_bytes())
            self.assertFalse((target / "archive-manifests").exists())

    def test_changed_source_fails_without_publishing_partial_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "target"
            target.mkdir()
            manifest = self._manifest(source, files)
            files[0].write_bytes(b"changed-after-plan")

            with self.assertRaisesRegex(ArchiveError, "source changed"):
                FilesystemArchiveAdapter(
                    source_root=source,
                    target_root=target,
                ).archive(manifest)
            self.assertEqual([], list(target.rglob("*.partial")))
            self.assertFalse((target / "archive-manifests").exists())

    def test_missing_target_is_not_created_and_source_remains_available(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "nas-not-mounted"
            source_before = [path.read_bytes() for path in files]

            with self.assertRaisesRegex(ArchiveUnavailableError, "missing or unsafe"):
                FilesystemArchiveAdapter(source_root=source, target_root=target)
            self.assertFalse(target.exists())
            self.assertEqual(source_before, [path.read_bytes() for path in files])

    def test_unsafe_names_outside_files_and_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source = root / "source"
            source.mkdir()
            unsafe = source / "family private name.jpg"
            unsafe.write_bytes(b"private")
            with self.assertRaisesRegex(ArchiveError, "relative_path"):
                build_archive_manifest(
                    archive_id="archive_01",
                    source_root=source,
                    files=(unsafe,),
                )

            outside = root / "outside.jpg"
            outside.write_bytes(b"outside")
            with self.assertRaisesRegex(ArchiveError, "outside"):
                build_archive_manifest(
                    archive_id="archive_01",
                    source_root=source,
                    files=(outside,),
                )

            safe = source / "safe.jpg"
            safe.write_bytes(b"safe")
            link = source / "link.jpg"
            try:
                link.symlink_to(safe)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(ArchiveError, "missing or unsafe"):
                build_archive_manifest(
                    archive_id="archive_01",
                    source_root=source,
                    files=(link,),
                )

    def test_manifest_is_strict_versioned_and_matches_public_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            manifest = self._manifest(source, files)
            output = root / "private/archive.json"
            output.parent.mkdir()
            manifest.write_new(output)
            manifest.write_new(output)
            self.assertEqual(manifest, ArchiveManifest.read(output))

            schema = json.loads(
                Path("schemas/archive-manifest-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(schema["required"]), set(manifest.to_dict()))
            self.assertEqual(
                set(schema["$defs"]["item"]["required"]),
                set(manifest.items[0].to_dict()),
            )

            output.write_text(
                '{"version":1,"version":1,"archive_id":"a",'
                '"created_at":"2026-08-03T00:00:00+00:00","items":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArchiveError, "duplicate JSON keys"):
                ArchiveManifest.read(output)

    def test_cli_plan_and_copy_do_not_disclose_paths_or_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "target"
            target.mkdir()
            manifest_path = root / "private/archive.json"
            manifest_path.parent.mkdir()
            plan_output = io.StringIO()
            plan_args = [
                "archive-plan",
                "--archive-id",
                "archive_cli_01",
                "--source-root",
                str(source),
                "--output",
                str(manifest_path),
            ]
            for path in files:
                plan_args.extend(["--file", str(path)])
            with redirect_stdout(plan_output):
                plan_code = main(plan_args)
            plan = json.loads(plan_output.getvalue())
            self.assertEqual(0, plan_code)
            self.assertEqual(3, plan["items"])
            self.assertFalse(plan["paths_disclosed"])
            self.assertNotIn(str(root), plan_output.getvalue())
            self.assertNotIn(manifest_path.read_text(encoding="utf-8"), plan_output.getvalue())

            copy_output = io.StringIO()
            with redirect_stdout(copy_output):
                copy_code = main(
                    [
                        "archive-copy",
                        "--manifest",
                        str(manifest_path),
                        "--source-root",
                        str(source),
                        "--target-root",
                        str(target),
                    ]
                )
            result = json.loads(copy_output.getvalue())
            self.assertEqual(0, copy_code)
            self.assertEqual("archived", result["status"])
            self.assertFalse(result["paths_disclosed"])
            self.assertNotIn(str(root), copy_output.getvalue())
            for item in ArchiveManifest.read(manifest_path).items:
                self.assertNotIn(item.sha256, copy_output.getvalue())
                self.assertNotIn(item.relative_path, copy_output.getvalue())

    def test_atomic_promotion_failure_cleans_only_its_partial(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "target"
            target.mkdir()
            manifest = self._manifest(source, files)
            with (
                patch("baby_monitor.archive.os.link", side_effect=OSError("unsupported")),
                self.assertRaisesRegex(
                    ArchiveUnavailableError,
                    "atomic no-replace",
                ),
            ):
                FilesystemArchiveAdapter(
                    source_root=source,
                    target_root=target,
                ).archive(manifest)
            self.assertEqual([], list(target.rglob("*.partial")))
            self.assertEqual(b"synthetic-clip", files[0].read_bytes())

    def test_cli_io_failure_is_generic_and_does_not_disclose_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-") as directory:
            root = Path(directory).resolve()
            source, files = self._source_tree(root)
            target = root / "target"
            target.mkdir()
            manifest_path = root / "manifest.json"
            self._manifest(source, files).write_new(manifest_path)
            errors = io.StringIO()

            with (
                patch(
                    "baby_monitor.archive.os.open",
                    side_effect=PermissionError(str(root / "private-target")),
                ),
                redirect_stderr(errors),
            ):
                code = main(
                    [
                        "archive-copy",
                        "--manifest",
                        str(manifest_path),
                        "--source-root",
                        str(source),
                        "--target-root",
                        str(target),
                    ]
                )

            self.assertEqual(2, code)
            self.assertIn("archive target is unavailable", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())
            self.assertEqual([], list(target.rglob("*.partial")))


if __name__ == "__main__":
    unittest.main()
