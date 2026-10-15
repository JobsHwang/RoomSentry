"""Local desktop adapter for the two existing private live-debug units.

No web server, credentials, media, notifications, arbitrary unit names or model
calls. JSON output is an allowlisted operational summary, not baby supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.live_people_view import supervision_view
from baby_monitor.live_review_control import process_start_ticks

GATEWAY = "xiaomi-stream-gateway.service"
WORKER = "baby-monitor-yolo-bonsai-live.service"
UNITS = (GATEWAY, WORKER)
_ACTIVE = {"active", "activating", "reloading", "inactive", "deactivating", "failed"}
_REVIEW = {
    "not_reviewed_this_session",
    "analyzing",
    "latest_analysis_failed",
    "recent_completed_review",
    "stale_completed_review",
    "previous_source_generation",
    "source_generation_unavailable",
    "receipt_clock_unavailable",
    "attempt_unavailable",
    "source_not_receiving",
    "monitor_not_detecting",
    "local_review_suspended",
}
_ERRORS = {
    "service_command_unavailable",
    "service_command_failed",
    "service_not_installed",
    "service_state_invalid",
    "invalid_control_action",
    "start_not_confirmed",
    "stop_not_confirmed",
    "review_control_unavailable",
    "review_control_not_supported",
}


class DesktopControlError(RuntimeError):
    pass


def systemctl(*args: str, timeout: float = 15) -> str:
    try:
        completed = subprocess.run(
            ("/usr/bin/systemctl", "--user", *args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise DesktopControlError("service_command_unavailable") from None
    if completed.returncode or len(completed.stdout.encode()) > 16384:
        raise DesktopControlError("service_command_failed")
    return completed.stdout


def services(runner=systemctl) -> dict[str, dict[str, object]]:
    result = {}
    for unit in UNITS:
        text = runner("show", unit, "--property=LoadState,ActiveState,MainPID")
        fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        state = fields.get("ActiveState")
        if fields.get("LoadState") != "loaded" or state not in _ACTIVE:
            raise DesktopControlError("service_not_installed")
        try:
            pid = int(fields["MainPID"])
            if not 0 <= pid < 2**31:
                raise ValueError
        except (KeyError, ValueError):
            raise DesktopControlError("service_state_invalid") from None
        result[unit] = {"state": state, "pid": pid}
    return result


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate status key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("invalid status number")


def read_private_status(path: Path, *, max_bytes: int = 512 * 1024) -> dict:
    # The fixed adapter reads only this operational file, never reviews or config.
    if path.resolve() != path or not path.parent.is_dir():
        raise ValueError("unsafe status path")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("unsafe status directory")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= max_bytes
        ):
            raise ValueError("unsafe status file")
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("status size limit")
    result = json.loads(payload, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if type(result) is not dict:
        raise ValueError("invalid status object")
    return result


def number(value, *, integer=False, maximum=10**12):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not 0 <= value <= maximum:
        return None
    if not integer and not math.isfinite(value):
        return None
    return value


def current_subjects(data, *, task, heartbeat_age, now):
    """Strict current-session projection; no prose, geometry, or identity facts."""
    evidence = data.get(
        "subject_evidence", data.get("review_evidence") if task == "subjects" else None
    )
    claims = data.get("subject_claims")
    if (
        task not in ("subjects", "hybrid")
        or data.get("phase") != "receiving"
        or type(evidence) is not dict
        or evidence.get("status") != "recent_completed_review"
        or evidence.get("model_output_verified") is not False
        or evidence.get("live_baby_state_established") is not False
        or type(data.get("generation")) is not int
        or data["generation"] < 0
        or type(evidence.get("frame_generation")) is not int
        or evidence["frame_generation"] != data["generation"]
        or type(claims) is not list
        or len(claims) > 8
    ):
        return None
    age = number(evidence.get("evidence_age_seconds"), maximum=30)
    updating = evidence.get("update_in_progress", False)
    if type(updating) is not bool or age is None or age + heartbeat_age > 30:
        return None
    try:
        stamp = datetime.fromisoformat(evidence["frame_received_at"])
        if stamp.utcoffset() is None or stamp > now:
            return None
        # Both ages must fit the window. A wall-clock step must not make an
        # older receipt look current; a rollback must not shorten monotonic age.
        age = max(age + heartbeat_age, (now - stamp).total_seconds())
        if not math.isfinite(age) or age > 30:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    ages = []
    for claim in claims:
        if type(claim) is not dict or set(claim) != {"age_claim", "head_center"}:
            return None
        if claim["age_claim"] not in ("baby", "adult", "unknown"):
            return None
        point = claim["head_center"]
        if point is not None and (
            type(point) is not list
            or len(point) != 2
            or any(type(x) is not int or not 0 <= x <= 1000 for x in point)
        ):
            return None
        ages.append(claim["age_claim"])
    return {
        "semantics": "unverified_model_claims",
        "frame_received_at": stamp.isoformat(),
        "evidence_age_seconds": round(age, 3),
        "age_claims": ages,
        "update_in_progress": updating,
    }


def review_control_snapshot(value):
    """Closed operational capability, never model prose or an experiment lease."""
    if type(value) is not dict:
        return None
    if value == {"enabled": False} and type(value.get("enabled")) is bool:
        return {"enabled": False}
    if (
        value.keys()
        != {
            "enabled",
            "pause_requested",
            "resident_unloaded",
            "timeout_seconds",
            "remaining_seconds",
            "suspension_count",
        }
        or value.get("enabled") is not True
    ):
        return None
    if any(type(value.get(k)) is not bool for k in ("pause_requested", "resident_unloaded")):
        return None
    timeout = number(value.get("timeout_seconds"), integer=True, maximum=600)
    remaining = number(value.get("remaining_seconds"), maximum=600)
    count = number(value.get("suspension_count"), integer=True)
    if timeout is None or timeout < 1 or remaining is None or remaining > timeout or count is None:
        return None
    return dict(value)


def baby_hint_summary(cameras):
    """Aggregate already session/freshness-validated projections, never room facts."""
    expected = tuple(f"camera_{i:02}" for i in range(1, 5))
    by_id = {row["camera_id"]: row for row in cameras}
    hints, reviewed, ambiguous = [], [], []
    for camera in expected:
        row = by_id.get(camera, {})
        probe = row.get("current_subjects")
        if not row.get("receiving") or probe is None:
            continue
        reviewed.append(camera)
        count = probe["age_claims"].count("baby")
        if count:
            hints.append(
                {
                    "camera_id": camera,
                    "baby_claim_count": count,
                    "frame_received_at": probe["frame_received_at"],
                    "evidence_age_seconds": probe["evidence_age_seconds"],
                    "update_in_progress": probe.get("update_in_progress", False),
                }
            )
        if count > 1:
            ambiguous.append(camera)
    status = (
        "no_fresh_reviews"
        if not reviewed
        else "no_baby_claim_in_fresh_reviews"
        if not hints
        else "ambiguous_baby_claims"
        if ambiguous or len(hints) > 1
        else "single_camera_baby_hint"
    )
    return {
        "semantics": "unverified_camera_hints_only",
        "status": status,
        "hints": hints,
        "fresh_reviewed_cameras": reviewed,
        "unavailable_cameras": [c for c in expected if c not in reviewed],
        "multiple_baby_claim_cameras": ambiguous,
        "coverage_complete": len(reviewed) == len(expected),
        "absence_established": False,
        "room_established": False,
        "identity_established": False,
    }


def snapshot(unit_states, status, *, boot_id, now=None, now_monotonic=None):
    now = now or datetime.now(UTC)
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    active = {u: unit_states[u]["state"] == "active" for u in UNITS}
    output = {
        "version": 1,
        "scope": "local_live_debug_desktop",
        "checked_at": now.isoformat(),
        "services": unit_states,
        "state": "stopped",
        "status_reason": "not_running",
        "heartbeat_age_seconds": None,
        "review_phase": None,
        "review_control": None,
        "review_suspended_frames": None,
        "review_pre_resume_frame_drops": None,
        "detected_frames": None,
        "review_successes": None,
        "review_failures": None,
        "review_task": None,
        "subject_probe_successes": None,
        "subject_probe_failures": None,
        "detection_failures": None,
        "storage_available": None,
        "storage_used_bytes": None,
        "storage_limit_bytes": None,
        "storage_reason": "unknown",
        "storage_scan_complete": None,
        "storage_writable_budget_bytes": None,
        "cameras": [],
        "baby_hint_summary": baby_hint_summary([]),
        "baby_location": "unknown",
        "baby_state": "unknown",
        "baby_supervision": None,
        "accuracy_verified": False,
        "media_read": False,
    }
    if any(active.values()):
        output["state"] = "partial" if not all(active.values()) else "starting"
        output["status_reason"] = "awaiting_current_session"
    if not all(active.values()):
        return output
    if (
        type(status) is not dict
        or type(status.get("worker_pid")) is not int
        or status["worker_pid"] != unit_states[WORKER]["pid"]
        or not boot_id
        or status.get("boot_id") != boot_id
    ):
        return output
    try:
        heartbeat = datetime.fromisoformat(status["heartbeat_at"])
        if heartbeat.utcoffset() is None:
            raise ValueError
        age = (now - heartbeat).total_seconds()
        if not math.isfinite(age) or age < 0:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        output["status_reason"] = "heartbeat_invalid"
        return output
    # This private timestamp is meaningful only after the boot/PID check above.
    # UTC alone lets a backwards clock step extend a frozen status file's TTL.
    published = number(status.get("heartbeat_monotonic_seconds"))
    current = number(now_monotonic)
    if published is None or current is None or current < published:
        output.update(state="degraded", status_reason="heartbeat_clock_unavailable")
        return output
    age = max(age, current - published)
    output["heartbeat_age_seconds"] = round(age, 1)
    if age > 10:
        output.update(state="degraded", status_reason="heartbeat_stale")
        return output
    if status.get("phase") != "detecting":
        output.update(state="degraded", status_reason="worker_not_detecting")
        return output
    output.update(state="running", status_reason="current_session")
    phase = status.get("review_phase")
    output["review_phase"] = (
        phase
        if type(phase) is str
        and phase
        in {
            "waiting",
            "analyzing",
            "storage_deferred",
            "memory_deferred",
            "worker_failed",
            "suspending",
            "suspended",
        }
        else "unknown"
    )
    output["review_control"] = review_control_snapshot(status.get("review_control"))
    invalid_control = "review_control" in status and output["review_control"] is None
    blocked = (
        invalid_control
        or phase in ("suspending", "suspended")
        or (
            output["review_control"] is not None
            and output["review_control"].get("enabled") is True
            and (
                output["review_control"]["pause_requested"]
                or output["review_control"]["resident_unloaded"]
            )
        )
    )
    if blocked:
        output.update(state="degraded", status_reason="local_review_suspended")
    if invalid_control:
        output.update(state="degraded", status_reason="local_review_unavailable")
    if output["review_phase"] in {
        "storage_deferred",
        "memory_deferred",
        "worker_failed",
        "unknown",
    }:
        output["state"] = "degraded"
        output["status_reason"] = "local_review_unavailable"
    task = status.get("review_task")
    output["review_task"] = task if task in ("event", "subjects", "hybrid") else "unknown"
    for key in (
        "review_successes",
        "review_failures",
        "detection_failures",
        "subject_probe_successes",
        "subject_probe_failures",
        "review_suspended_frames",
        "review_pre_resume_frame_drops",
    ):
        output[key] = number(status.get(key), integer=True)
    storage = status.get("review_storage")
    if isinstance(storage, dict):
        output["storage_available"] = (
            storage.get("available") if type(storage.get("available")) is bool else None
        )
        output["storage_used_bytes"] = number(storage.get("used_bytes"), integer=True)
        output["storage_limit_bytes"] = number(storage.get("limit_bytes"), integer=True)
        reason = storage.get("reason")
        output["storage_reason"] = (
            reason
            if type(reason) is str
            and reason
            in {
                "available",
                "capacity_limit",
                "entry_limit",
                "disk_space_low",
                "unsafe_symlink",
                "io_error",
            }
            else "unknown"
        )
        output["storage_scan_complete"] = (
            storage.get("scan_complete") if type(storage.get("scan_complete")) is bool else None
        )
        reserve = number(storage.get("reserve_bytes"), integer=True)
        if output["storage_scan_complete"] is True and all(
            v is not None
            for v in (
                output["storage_used_bytes"],
                output["storage_limit_bytes"],
                reserve,
            )
        ):
            output["storage_writable_budget_bytes"] = max(
                0, output["storage_limit_bytes"] - output["storage_used_bytes"] - reserve
            )
    cameras = status.get("cameras", {})
    if not blocked and output["review_phase"] not in ("worker_failed", "unknown"):
        output["baby_supervision"] = supervision_view(
            status.get("baby_supervision"),
            camera_sources=cameras if type(cameras) is dict else {},
            heartbeat_age=age,
            now=now,
        )
        current_baby = (output["baby_supervision"] or {}).get("current")
        if current_baby:
            output["baby_location"] = current_baby["room_id"] or current_baby["camera_id"]
            output["baby_state"] = current_baby["visible_posture"]
    total = 0
    for camera in ("camera_01", "camera_02", "camera_03", "camera_04"):
        data = cameras.get(camera, {}) if isinstance(cameras, dict) else {}
        if not isinstance(data, dict):
            data = {}
        evidence = data.get("review_evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        phase = data.get("phase")
        count = number(data.get("inferred_frames"), integer=True)
        if count is not None:
            total += count
        review = evidence.get("status")
        output["cameras"].append(
            {
                "camera_id": camera,
                "receiving": phase == "receiving",
                "sample_fps": number(data.get("recent_inference_fps"), maximum=240),
                "candidate_boxes": number(
                    data.get("last_person_count"), integer=True, maximum=1000
                ),
                "review_status": review if type(review) is str and review in _REVIEW else "unknown",
                "evidence_age_seconds": number(evidence.get("evidence_age_seconds")),
                "current_subjects": None
                if blocked
                else current_subjects(data, task=task, heartbeat_age=age, now=now),
            }
        )
    output["detected_frames"] = total
    if not all(row["receiving"] for row in output["cameras"]):
        output.update(state="degraded", status_reason="camera_not_receiving")
    output["baby_hint_summary"] = baby_hint_summary(output["cameras"])
    return output


def _signal_review_worker(pid, start_ticks, signum):
    """Pin a process before sending; never signal a reused PID or whole cgroup."""
    if (
        type(pid) is not int
        or not 1 <= pid < 2**31
        or type(start_ticks) is not int
        or start_ticks < 0
        or type(signum) not in (int, signal.Signals)
        or signum not in (signal.SIGUSR1, signal.SIGUSR2)
    ):
        raise DesktopControlError("review_control_unavailable")
    descriptor = None
    try:
        descriptor = os.pidfd_open(pid)
        if process_start_ticks(pid) != start_ticks:
            raise DesktopControlError("review_control_unavailable")
        signal.pidfd_send_signal(descriptor, signum)
    except (OSError, ValueError, AttributeError):
        raise DesktopControlError("review_control_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def control(
    action,
    runner=systemctl,
    *,
    status_reader=None,
    signal_sender=None,
    boot_id=None,
    now=None,
    now_monotonic=None,
):
    if action not in {"status", "start", "stop", "suspend-review", "resume-review"}:
        raise DesktopControlError("invalid_control_action")
    if action in {"suspend-review", "resume-review"}:
        states = services(runner)
        try:
            status = (
                status_reader
                or (
                    lambda: read_private_status(
                        Path.home()
                        / ".local/share/baby-monitor/live-debug/four-streams/status.json"
                    )
                )
            )()
            current = snapshot(
                states,
                status,
                boot_id=boot_id or Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                now=now,
                now_monotonic=now_monotonic,
            )
            capability = current["review_control"]
            ticks = number(status.get("worker_start_ticks"), integer=True, maximum=10**15)
            if capability is None or capability.get("enabled") is not True or ticks is None:
                raise DesktopControlError("review_control_not_supported")
            (signal_sender or _signal_review_worker)(
                states[WORKER]["pid"],
                ticks,
                signal.SIGUSR1 if action == "suspend-review" else signal.SIGUSR2,
            )
        except DesktopControlError:
            raise
        except Exception:
            raise DesktopControlError("review_control_unavailable") from None
        return services(runner)
    if action == "start":
        runner("start", GATEWAY, timeout=90)
        runner("start", WORKER, timeout=90)
    elif action == "stop":
        runner("stop", WORKER, GATEWAY, timeout=240)
    states = services(runner)
    if action == "start" and not all(row["state"] == "active" for row in states.values()):
        raise DesktopControlError("start_not_confirmed")
    if action == "stop" and not all(
        row["state"] in {"inactive", "failed"} and row["pid"] == 0 for row in states.values()
    ):
        raise DesktopControlError("stop_not_confirmed")
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("status", "start", "stop", "suspend-review", "resume-review"),
        default="status",
    )
    args = parser.parse_args()
    try:
        states = control(args.action)
        try:
            status = read_private_status(
                Path.home() / ".local/share/baby-monitor/live-debug/four-streams/status.json"
            )
        except (OSError, ValueError, RecursionError):
            status = None
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        result = snapshot(states, status, boot_id=boot_id)
        # Historical, opt-in export is separate from service liveness and baby state.
        from baby_monitor.subject_diagnostic import validate

        try:
            diagnostic = read_private_status(
                Path.home() / ".local/share/baby-monitor/desktop/subject-diagnostic.json"
            )
            result["subject_diagnostic"] = validate(diagnostic)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            result["subject_diagnostic"] = None
        result.update(ok=True, action=args.action)
        if args.action in {"suspend-review", "resume-review"}:
            # Delivery is not acknowledgement: callers must await suspended /
            # unloaded and separately acquire the shared VLM gate before tests.
            result["control_request_sent"] = True
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except Exception as error:
        code = (
            str(error)
            if isinstance(error, DesktopControlError) and str(error) in _ERRORS
            else "desktop_control_unavailable"
        )
        print(json.dumps({"ok": False, "error_code": code, "action": args.action}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
