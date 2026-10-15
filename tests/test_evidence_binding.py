"""Synthetic local evidence only; no camera, upload, or model is involved."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from baby_monitor.domain import EventMedia
from baby_monitor.evidence_binding import (
    HASH_CHUNK_BYTES,
    MAX_BINDING_BYTES,
    MAX_CLIP_BYTES,
    MAX_FRAME_BYTES,
    EvidenceAssetBinding,
    EvidenceBinding,
    EvidenceBindingError,
    capture_evidence_binding,
    media_fingerprint,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@unittest.skipUnless(os.name == "posix", "private evidence reads require POSIX")
class EvidenceBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clip = self._file("clip.bin", b"synthetic clip")
        self.frames = (
            self._file("frame0.bin", b"synthetic frame 0"),
            self._file("frame1.bin", b"synthetic frame 1"),
        )
        self.media = EventMedia(clip_path=str(self.clip), frame_paths=tuple(map(str, self.frames)))

    def _file(self, name: str, content: bytes) -> Path:
        path = self.root / name
        path.write_bytes(content)
        path.chmod(0o600)
        return path

    def _reject(self, media: EventMedia) -> None:
        with self.assertRaises(EvidenceBindingError) as caught:
            capture_evidence_binding(media)
        self.assertNotIn(str(self.root), str(caught.exception))

    def test_capture_roundtrip_order_hash_and_path_free_payload(self) -> None:
        binding = capture_evidence_binding(self.media)
        self.assertEqual(binding.media_sha256, _sha(self.media.to_dict()))
        self.assertEqual(
            [(a.role, a.index) for a in binding.assets], [("clip", 0), ("frame", 0), ("frame", 1)]
        )
        for asset, path in zip(binding.assets, (self.clip, *self.frames), strict=True):
            self.assertEqual(asset.size_bytes, len(path.read_bytes()))
            self.assertEqual(asset.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(set(asset.to_dict()), {"role", "index", "size_bytes", "sha256"})
        self.assertNotIn(str(self.root), binding.to_json())
        self.assertNotIn("clip_path", binding.to_json())
        self.assertEqual(binding.fingerprint, _sha(binding.to_dict()))
        self.assertEqual(EvidenceBinding.from_json(binding.to_json()), binding)
        self.assertIsNone(binding.verify(self.media))

    def test_fingerprint_has_no_file_io_and_binds_all_media_metadata(self) -> None:
        media = replace(self.media, clip_path="/nonexistent/synthetic.bin")
        with patch("baby_monitor.evidence_binding.os.open", side_effect=AssertionError("no IO")):
            self.assertEqual(media_fingerprint(media), _sha(media.to_dict()))
        self.assertNotEqual(
            media_fingerprint(media), media_fingerprint(replace(media, selected_segments=1))
        )
        with self.assertRaises(EvidenceBindingError):
            media_fingerprint({})

    def test_verify_rejects_changed_bytes_metadata_and_frame_order(self) -> None:
        binding = capture_evidence_binding(self.media)
        for media in (
            replace(self.media, selected_segments=1),
            replace(self.media, frame_paths=tuple(reversed(self.media.frame_paths))),
        ):
            with (
                self.subTest(media=media.selected_segments),
                self.assertRaises(EvidenceBindingError),
            ):
                binding.verify(media)
        self.clip.write_bytes(b"Synthetic clip")  # Same size is not equivalent content.
        with self.assertRaises(EvidenceBindingError):
            binding.verify(self.media)

    def test_frame_count_and_duplicate_normalized_paths(self) -> None:
        for frames in (
            (),
            self.media.frame_paths * 5,
            (str(self.clip),),
            (str(self.frames[0]), str(self.frames[0])),
            (str(self.frames[0]), str(self.root) + "/./frame0.bin"),
        ):
            with self.subTest(frames=frames):
                self._reject(replace(self.media, frame_paths=frames))
        extra = tuple(self._file(f"extra{i}.bin", b"x") for i in range(8))
        self.assertEqual(
            len(
                capture_evidence_binding(
                    replace(self.media, frame_paths=tuple(map(str, extra)))
                ).assets
            ),
            9,
        )

    def test_unsafe_paths_missing_file_and_nonregular_file(self) -> None:
        fifo = self.root / "fifo"
        os.mkfifo(fifo, 0o600)
        for path in (
            "",
            "clip.bin",
            "//tmp/clip.bin",
            str(self.root / ".." / "clip.bin"),
            str(self.root / "missing"),
            str(self.root),
            str(fifo),
            str(self.clip) + "\x00suffix",
        ):
            with self.subTest(path=path):
                self._reject(replace(self.media, clip_path=path))

    def test_symlink_leaf_ancestor_and_hardlink_rejected(self) -> None:
        link = self.root / "leaflink"
        link.symlink_to(self.clip)
        self._reject(replace(self.media, clip_path=str(link)))
        directory_link = self.root / "ancestorlink"
        directory_link.symlink_to(self.root, target_is_directory=True)
        self._reject(replace(self.media, clip_path=str(directory_link / self.clip.name)))
        hardlink = self.root / "hardlink"
        os.link(self.clip, hardlink)
        self._reject(self.media)
        self._reject(replace(self.media, clip_path=str(hardlink)))

    def test_asset_private_owner_permissions(self) -> None:
        for mode in (0o640, 0o604, 0o601, 0o610, 0o666):
            with self.subTest(mode=mode):
                self.clip.chmod(mode)
                self._reject(self.media)
        self.clip.chmod(0o400)
        capture_evidence_binding(self.media)
        with patch("baby_monitor.evidence_binding.os.geteuid", return_value=os.geteuid() + 1):
            self._reject(self.media)

    def test_empty_and_oversized_assets_rejected_before_read(self) -> None:
        self.clip.write_bytes(b"")
        self._reject(self.media)
        for path, limit in ((self.clip, MAX_CLIP_BYTES), (self.frames[0], MAX_FRAME_BYTES)):
            self.clip.write_bytes(b"clip")
            with path.open("r+b") as handle:
                handle.truncate(limit + 1)
            self._reject(self.media)
            path.write_bytes(b"restored")

    def test_chunk_bound_and_no_follow_nonblocking_flags(self) -> None:
        content = b"x" * (HASH_CHUNK_BYTES + 19)
        self.clip.write_bytes(content)
        real_read, real_open = os.read, os.open
        with (
            patch("baby_monitor.evidence_binding.os.read", wraps=real_read) as reads,
            patch("baby_monitor.evidence_binding.os.open", wraps=real_open) as opens,
        ):
            result = capture_evidence_binding(self.media)
        self.assertEqual(result.assets[0].sha256, hashlib.sha256(content).hexdigest())
        self.assertTrue(all(0 < call.args[1] <= HASH_CHUNK_BYTES for call in reads.call_args_list))
        self.assertTrue(all(call.args[1] & os.O_NOFOLLOW for call in opens.call_args_list))
        self.assertTrue(all(call.args[1] & os.O_NONBLOCK for call in opens.call_args_list))

    def test_same_size_mutation_while_reading_is_rejected(self) -> None:
        real_read = os.read
        changed = False

        def mutate(descriptor: int, count: int) -> bytes:
            nonlocal changed
            chunk = real_read(descriptor, count)
            if chunk and not changed:
                changed = True
                self.clip.write_bytes(b"z" * len(b"synthetic clip"))
                # Guarantee an endpoint metadata change, independent of clock resolution.
                os.utime(self.clip, ns=(1, 1))
            return chunk

        with patch("baby_monitor.evidence_binding.os.read", side_effect=mutate):
            self._reject(self.media)

    def test_replaced_leaf_while_descriptor_open_is_rejected(self) -> None:
        replacement = self._file("replacement.bin", self.clip.read_bytes())
        real_read = os.read
        changed = False

        def replace_path(descriptor: int, count: int) -> bytes:
            nonlocal changed
            chunk = real_read(descriptor, count)
            if chunk and not changed:
                changed = True
                os.replace(replacement, self.clip)
            return chunk

        with patch("baby_monitor.evidence_binding.os.read", side_effect=replace_path):
            self._reject(self.media)

    def test_replaced_ancestor_while_reading_is_rejected(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        clip = nested / "clip"
        clip.write_bytes(b"synthetic")
        clip.chmod(0o600)
        media = replace(self.media, clip_path=str(clip))
        real_read = os.read
        changed = False

        def replace_ancestor(descriptor: int, count: int) -> bytes:
            nonlocal changed
            chunk = real_read(descriptor, count)
            if chunk and not changed:
                changed = True
                nested.rename(self.root / "moved")
                nested.symlink_to(self.root / "moved", target_is_directory=True)
            return chunk

        with patch("baby_monitor.evidence_binding.os.read", side_effect=replace_ancestor):
            self._reject(media)

    def test_io_failure_is_sanitized_and_descriptors_closed(self) -> None:
        real_open, real_close = os.open, os.close
        with (
            patch("baby_monitor.evidence_binding.os.open", wraps=real_open) as opens,
            patch("baby_monitor.evidence_binding.os.close", wraps=real_close) as closes,
            patch(
                "baby_monitor.evidence_binding.os.read",
                side_effect=OSError("/private/secret-token"),
            ),
            self.assertRaises(EvidenceBindingError) as caught,
        ):
            capture_evidence_binding(self.media)
        self.assertNotIn("secret-token", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(opens.call_count, closes.call_count)

    def test_non_posix_and_unsupported_dirfd_fail_closed(self) -> None:
        with patch("baby_monitor.evidence_binding.os.name", "nt"):
            self._reject(self.media)
        with patch(
            "baby_monitor.evidence_binding.os.open", side_effect=NotImplementedError("private path")
        ):
            self._reject(self.media)

    def test_strict_wire_types_shapes_order_and_limits(self) -> None:
        valid = capture_evidence_binding(self.media).to_dict()
        variants = []
        for key, values in (
            ("version", (True, 1.0, 2, None)),
            ("media_sha256", ("A" * 64, "0" * 63, None)),
            ("assets", (None, (), [], "private")),
        ):
            variants.extend({**valid, key: value} for value in values)
        variants.extend(
            (
                {**valid, "unknown": 1},
                {key: value for key, value in valid.items() if key != "version"},
            )
        )
        for key, values in (
            ("role", (None, [], "unknown")),
            ("index", (True, 1.0, 1, -1)),
            ("size_bytes", (True, 1.0, 0, -1, MAX_CLIP_BYTES + 1)),
            ("sha256", ("z" * 64, None)),
        ):
            for value in values:
                rows = [dict(row) for row in valid["assets"]]
                rows[0][key] = value
                variants.append({**valid, "assets": rows})
        variants.append({**valid, "assets": list(reversed(valid["assets"]))})
        rows = [dict(row) for row in valid["assets"]]
        rows[1]["index"] = 1
        variants.append({**valid, "assets": rows})
        rows = [dict(row) for row in valid["assets"]]
        rows[1]["size_bytes"] = MAX_FRAME_BYTES + 1
        variants.append({**valid, "assets": rows})
        for number, variant in enumerate(variants):
            with self.subTest(number=number), self.assertRaises(EvidenceBindingError):
                EvidenceBinding.from_json(json.dumps(variant))

    def test_json_duplicate_nonfinite_nul_size_and_recursion_rejected(self) -> None:
        payload = capture_evidence_binding(self.media).to_json()
        bad = (
            payload[:-1] + ',"version":1}',
            payload.replace('"role":"clip"', '"role":"clip","role":"clip"'),
            payload.replace('"version":1', '"version":NaN'),
            payload.replace('"version":1', '"version":Infinity'),
            payload + "\x00garbage",
            " " * (MAX_BINDING_BYTES + 1),
            "",
            None,
            "[" * 2000 + "]" * 2000,
            "\ud800",
        )
        for number, value in enumerate(bad):
            with self.subTest(number=number), self.assertRaises(EvidenceBindingError):
                EvidenceBinding.from_json(value)

    def test_direct_constructors_enforce_boundaries(self) -> None:
        clip = EvidenceAssetBinding("clip", 0, MAX_CLIP_BYTES, "a" * 64)
        frame = EvidenceAssetBinding("frame", 0, MAX_FRAME_BYTES, "b" * 64)
        EvidenceBinding("c" * 64, (clip, frame))
        for assets in ([clip, frame], (clip,), (frame, clip), (clip, object())):
            with self.subTest(assets=assets), self.assertRaises(EvidenceBindingError):
                EvidenceBinding("c" * 64, assets)


if __name__ == "__main__":
    unittest.main()
