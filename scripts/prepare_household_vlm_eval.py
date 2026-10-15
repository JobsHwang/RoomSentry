#!/usr/bin/env python3
"""Prepare a bounded, local-only household VLM review set from loopback RTSP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_MAX_FRAME_BYTES = 50 * 1024 * 1024
_QUESTION = (
    "请按帧顺序只标注直接可见的事实。普通 person 检测不能证明宝宝、成人或具体身份；"
    "无法从画面确认的状态必须保持 unknown。observations 只放可见事实，推测只能放入 "
    "inferences，并明确不确定性。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loopback_rtsp_uri(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "rtsp"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path
        or parsed.path == "/"
    ):
        raise ValueError("each source must be a credential-free loopback RTSP URI with a path")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("each source must contain a valid RTSP port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("each source must contain an explicit RTSP port")
    return value


def _ffmpeg_executable(value: str | Path) -> str:
    raw = str(value)
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError("ffmpeg must be an executable regular file")
        return str(resolved)
    located = shutil.which(raw)
    if located is None:
        raise ValueError("ffmpeg executable was not found")
    return str(Path(located).resolve(strict=True))


def _capture_source(
    *,
    ffmpeg: str,
    source: str,
    frame_pattern: Path,
    frame_count: int,
    frame_interval_seconds: float,
) -> None:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps=1/{frame_interval_seconds:g}",
        "-frames:v",
        str(frame_count),
        "-q:v",
        "3",
        "-start_number",
        "1",
        str(frame_pattern),
    ]
    timeout_seconds = max(30.0, frame_count * frame_interval_seconds + 20.0)
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("bounded local frame capture failed") from exc
    if completed.returncode != 0:
        raise RuntimeError("bounded local frame capture failed")


def _frame_record(path: Path, root: Path, *, frame_id: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("captured frame is missing or is not a regular file")
    size = path.stat().st_size
    if not 0 < size <= _MAX_FRAME_BYTES:
        raise RuntimeError("captured frame size is outside the allowed range")
    return {
        "frame_id": frame_id,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def prepare(
    output_dir: Path,
    *,
    sources: Sequence[str],
    cases_per_source: int,
    frames_per_case: int,
    frame_interval_seconds: float,
    ffmpeg: str | Path = "ffmpeg",
    confirm_private_media: bool,
) -> tuple[int, int, str]:
    if not confirm_private_media:
        raise ValueError("--confirm-private-media is required")
    if not 1 <= len(sources) <= 4:
        raise ValueError("one to four loopback RTSP sources are required")
    parsed_sources = tuple(_loopback_rtsp_uri(source) for source in sources)
    if len(set(parsed_sources)) != len(parsed_sources):
        raise ValueError("loopback RTSP sources must be unique")
    if not 1 <= cases_per_source <= 30:
        raise ValueError("cases per source must be between 1 and 30")
    if not 1 <= frames_per_case <= 8:
        raise ValueError("frames per case must be between 1 and 8")
    case_count = len(parsed_sources) * cases_per_source
    if not 10 <= case_count <= 30:
        raise ValueError("the household review set must contain 10 to 30 cases")
    if not 0.5 <= frame_interval_seconds <= 30.0:
        raise ValueError("frame interval must be between 0.5 and 30 seconds")
    executable = _ffmpeg_executable(ffmpeg)
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite")
    if output_dir.parent.is_symlink():
        raise ValueError("output parent must not be a symlink")

    old_umask = os.umask(0o077)
    try:
        output_dir.mkdir(mode=0o700)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(mode=0o700)
        try:
            frames_per_source = cases_per_source * frames_per_case
            cases: list[dict[str, object]] = []
            next_case = 1
            for source_index, source in enumerate(parsed_sources, start=1):
                pattern = frames_dir / f"s{source_index:02d}_%04d.jpg"
                _capture_source(
                    ffmpeg=executable,
                    source=source,
                    frame_pattern=pattern,
                    frame_count=frames_per_source,
                    frame_interval_seconds=frame_interval_seconds,
                )
                captured = [
                    frames_dir / f"s{source_index:02d}_{index:04d}.jpg"
                    for index in range(1, frames_per_source + 1)
                ]
                for source_case in range(cases_per_source):
                    case_id = f"household_{next_case:03d}"
                    start = source_case * frames_per_case
                    selected = captured[start : start + frames_per_case]
                    cases.append(
                        {
                            "version": 1,
                            "review_status": "pending",
                            "case_id": case_id,
                            "frames": [
                                _frame_record(
                                    path,
                                    output_dir,
                                    frame_id=f"f{frame_index:02d}",
                                )
                                for frame_index, path in enumerate(selected, start=1)
                            ],
                            "question": _QUESTION,
                            "expected": None,
                            "reviews": [],
                        }
                    )
                    next_case += 1

            review_path = output_dir / "review.pending.jsonl"
            review_path.write_text(
                "".join(
                    json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for case in cases
                ),
                encoding="utf-8",
                newline="\n",
            )
            marker = output_dir / "PRIVATE_MEDIA_DO_NOT_COMMIT"
            marker.write_text(
                "Local household review media. Keep ignored; do not upload or commit.\n",
                encoding="utf-8",
                newline="\n",
            )
            for path in (review_path, marker):
                path.chmod(0o600)
            frame_count = case_count * frames_per_case
            return case_count, frame_count, _sha256(review_path)
        except BaseException:
            # The target did not exist before this command. Remove only this command's
            # incomplete output so partial private frames cannot look review-ready.
            shutil.rmtree(output_dir)
            raise
    finally:
        os.umask(old_umask)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append")
    parser.add_argument("--cases-per-source", type=int, default=5)
    parser.add_argument("--frames-per-case", type=int, default=3)
    parser.add_argument("--frame-interval-seconds", type=float, default=2.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--confirm-private-media", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases, frames, manifest_sha256 = prepare(
        args.output_dir,
        sources=args.source,
        cases_per_source=args.cases_per_source,
        frames_per_case=args.frames_per_case,
        frame_interval_seconds=args.frame_interval_seconds,
        ffmpeg=args.ffmpeg,
        confirm_private_media=args.confirm_private_media,
    )
    print(
        json.dumps(
            {
                "version": 1,
                "cases": cases,
                "frames": frames,
                "sources": len(args.source),
                "review_status": "pending",
                "review_manifest_sha256": manifest_sha256,
                "private_media_used": True,
                "paths_disclosed": False,
                "uploads_performed": False,
                "ready_for_ab": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
