"""Finite metadata-only audit of the existing local supervision session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.live_monitor import _write_private
from scripts.live_desktop_backend import WORKER, read_private_status, services


def sample(status):
    result = {
        k: status.get(k)
        for k in (
            "worker_pid",
            "session_started_at",
            "heartbeat_at",
            "phase",
            "review_phase",
            "subject_probe_successes",
            "subject_probe_failures",
            "detection_failures",
        )
    }
    supervision = status.get("baby_supervision") or {}
    result["baby_status"] = supervision.get("status")
    result["current_baby_camera"] = (supervision.get("current") or {}).get("camera_id")
    result["cameras"] = {}
    for camera, data in status.get("cameras", {}).items():
        tracked = data.get("tracked_people") or {}
        people = tracked.get("people", [])
        result["cameras"][camera] = {
            "receiving": data.get("phase") == "receiving",
            "generation": data.get("generation"),
            "inferred_frames": data.get("inferred_frames"),
            "fps": data.get("recent_inference_fps"),
            "native_boxes": data.get("last_person_count"),
            "tracking_status": tracked.get("status"),
            "view": tracked.get("detection_view"),
            "tracked_count": len(people),
            "supported_roles": [p["age_claim"] for p in people if p["role_status"] == "supported"],
            "candidate_roles": [p["age_claim"] for p in people if p["role_status"] == "candidate"],
            "untracked_roles": [p["age_claim"] for p in tracked.get("untracked_people", [])],
            "fresh_probe": (data.get("subject_evidence") or {}).get("status")
            == "recent_completed_review",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 600:
        parser.error("audit duration must be 10..600 seconds")
    source = Path.home() / ".local/share/baby-monitor/live-debug/four-streams/status.json"
    parent = Path.home() / ".local/share/baby-monitor/household-eval"
    metadata = parent.stat()
    if parent.is_symlink() or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("unsafe private audit directory")
    root = Path(
        tempfile.mkdtemp(prefix=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_people_"), dir=parent)
    )
    root.chmod(0o700)
    code = (
        "src/baby_monitor/live_people.py",
        "src/baby_monitor/live_monitor.py",
        "src/baby_monitor/local_runners/subject_state_probe.py",
        "scripts/collect_live_people.py",
    )
    metadata = {
        "version": 1,
        "scope": "finite_metadata_audit",
        "started_at": datetime.now(UTC).isoformat(),
        "requested_seconds": args.seconds,
        "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code},
        "media_read": False,
        "accuracy_measured": False,
    }
    _write_private(root / "manifest.json", metadata)
    started, samples, gaps, session = time.monotonic(), [], 0, None
    expected_pid = services()[WORKER]["pid"]
    while time.monotonic() - started < args.seconds:
        try:
            status = read_private_status(source)
            point = sample(status)
            wall_age = (
                datetime.now(UTC) - datetime.fromisoformat(status["heartbeat_at"])
            ).total_seconds()
            mono_age = time.monotonic() - status["heartbeat_monotonic_seconds"]
            if (
                point["worker_pid"] != expected_pid
                or min(wall_age, mono_age) < 0
                or max(wall_age, mono_age) > 10
            ):
                raise ValueError("inactive session or stale heartbeat")
            key = (point["worker_pid"], point["session_started_at"])
            if session is None:
                session = key
            if key != session or point["phase"] != "detecting":
                raise ValueError("session changed or stopped")
            point["sampled_at"] = datetime.now(UTC).isoformat()
            point["sample_elapsed"] = round(time.monotonic() - started, 3)
            samples.append(point)
        except (ValueError, OSError, TypeError, KeyError):
            gaps += 1
        time.sleep(0.5)
    rows, receiving, fresh, tracked, unbound, frames = (
        {},
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        {},
    )
    for point in samples:
        for camera, data in point["cameras"].items():
            rows[camera] = rows.get(camera, 0) + 1
            receiving[camera] += data["receiving"]
            fresh[camera] += data["fresh_probe"]
            tracked[camera] += data["tracked_count"] > 0
            unbound[camera] += bool(data["untracked_roles"])
            if camera not in frames:
                frames[camera] = [data["inferred_frames"], data["inferred_frames"]]
            else:
                frames[camera][1] = data["inferred_frames"]
    first, last = (samples[0], samples[-1]) if samples else ({}, {})
    summary = {
        "completed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "samples": len(samples),
        "collection_gaps": gaps,
        "same_session": session,
        "baby_status_samples": dict(Counter(p["baby_status"] for p in samples)),
        "current_baby_camera_samples": dict(
            Counter(p["current_baby_camera"] for p in samples if p["current_baby_camera"])
        ),
        "camera_sample_denominators": rows,
        "receiving_samples": dict(receiving),
        "fresh_probe_samples": dict(fresh),
        "tracked_samples": dict(tracked),
        "unbound_claim_samples": dict(unbound),
        "inferred_frame_ranges": frames,
        "successful_probes_added": last.get("subject_probe_successes", 0)
        - first.get("subject_probe_successes", 0),
        "failed_probes_added": last.get("subject_probe_failures", 0)
        - first.get("subject_probe_failures", 0),
        "detection_failures_added": last.get("detection_failures", 0)
        - first.get("detection_failures", 0),
        "sampled_metadata_not_every_frame": True,
        "accuracy_measured": False,
    }
    _write_private(root / "samples.json", {"points": samples})
    _write_private(root / "summary.json", summary)
    print(json.dumps({"private_audit": str(root), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
