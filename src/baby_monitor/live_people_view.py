"""Closed desktop projection of live people; never returns model prose or paths."""

from __future__ import annotations

import math
import re
from datetime import datetime

POSTURES = {"lying", "sitting", "standing", "held", "unknown"}


def _id(value):
    return value if type(value) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) else None


def _number(value, maximum):
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum
        else None
    )


def _stamp_age(stamp, now):
    try:
        parsed = datetime.fromisoformat(stamp)
        if parsed.utcoffset() is None:
            return None
        age = (now - parsed).total_seconds()
        return age if math.isfinite(age) and age >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def supervision_view(value, *, camera_sources, heartbeat_age, now):
    """Re-age fields at delivery. Stale/foreign session is filtered by the caller."""
    if (
        type(value) is not dict
        or value.get("enabled") is not True
        or value.get("version") != 1
        or type(value.get("version")) is not int
        or value.get("semantics") != "unverified_model_inferences"
        or value.get("accuracy_verified") is not False
        or value.get("sleep_established") is not False
        or type(value.get("cameras")) is not dict
        or _number(heartbeat_age, 10) is None
    ):
        return None
    cameras, supported, candidates, last_seen = [], [], [], []
    for camera_id, source in camera_sources.items():
        if _id(camera_id) is None or type(source) is not dict:
            continue
        raw = value["cameras"].get(camera_id)
        if type(raw) is not dict:
            continue
        room = raw.get("room")
        room = room if type(room) is dict else {}
        room_id, room_name = _id(room.get("room_id")), room.get("display_name")
        if (
            type(room_name) is not str
            or not 1 <= len(room_name) <= 80
            or any(ord(c) < 32 for c in room_name)
        ):
            room_name = None
        detection_age = _number(raw.get("detection_age_seconds"), 2)
        available = (
            raw.get("status") == "current"
            and source.get("phase") == "receiving"
            and type(raw.get("generation")) is int
            and raw["generation"] == source.get("generation")
            and detection_age is not None
            and detection_age + heartbeat_age <= 2
        )
        rows = []
        people = raw.get("people")
        if available and type(people) is list and len(people) <= 128:
            for person in people:
                if type(person) is not dict or _id(person.get("track_id")) is None:
                    continue
                age = _number(person.get("evidence_age_seconds"), 30)
                wall_age = _stamp_age(person.get("evidence_frame_received_at"), now)
                valid = (
                    age is not None
                    and wall_age is not None
                    and max(age + heartbeat_age, wall_age) <= 30
                )
                role = person.get("age_claim") if valid else "unknown"
                role = role if role in ("baby", "adult", "unknown") else "unknown"
                role_status = person.get("role_status") if valid else "unknown"
                role_status = (
                    role_status
                    if role_status in ("supported", "candidate", "unknown")
                    else "unknown"
                )
                posture = person.get("visible_posture")
                posture = (
                    posture if type(posture) is str and posture in POSTURES and valid else "unknown"
                )
                item = {
                    "track_id": person["track_id"],
                    "age_claim": role,
                    "role_status": role_status,
                    "visible_posture": posture,
                    "evidence_age_seconds": round(max(age + heartbeat_age, wall_age), 3)
                    if valid
                    else None,
                    "evidence_frame_received_at": person.get("evidence_frame_received_at")
                    if valid
                    else None,
                }
                rows.append(item)
                if role == "baby":
                    entry = {
                        **item,
                        "camera_id": camera_id,
                        "room_id": room_id,
                        "room_name": room_name,
                        "detection_age_seconds": round(detection_age + heartbeat_age, 3),
                    }
                    (supported if role_status == "supported" else candidates).append(entry)
        untracked = raw.get("untracked_people")
        if available and type(untracked) is list and len(untracked) <= 8:
            for person in untracked:
                if type(person) is not dict or person.get("age_claim") != "baby":
                    continue
                age = _number(person.get("evidence_age_seconds"), 30)
                wall_age = _stamp_age(person.get("evidence_frame_received_at"), now)
                if (
                    age is not None
                    and wall_age is not None
                    and max(age + heartbeat_age, wall_age) <= 30
                ):
                    candidates.append(
                        {
                            "camera_id": camera_id,
                            "room_id": room_id,
                            "room_name": room_name,
                            "role_status": "untracked_candidate",
                        }
                    )
        old = raw.get("last_seen_baby")
        if (
            type(old) is dict
            and old.get("camera_id") == camera_id
            and _id(old.get("track_id"))
            and _stamp_age(old.get("last_detected_at"), now) is not None
        ):
            last_seen.append(
                {
                    "camera_id": camera_id,
                    "room_id": room_id,
                    "room_name": room_name,
                    "last_detected_at": old["last_detected_at"],
                    "track_id": old["track_id"],
                }
            )
        cameras.append(
            {
                "camera_id": camera_id,
                "room_id": room_id,
                "room_name": room_name,
                "available": available,
                "detection_age_seconds": None
                if detection_age is None
                else round(detection_age + heartbeat_age, 3),
                "people": rows,
            }
        )
    current = supported[0] if len(supported) == 1 and not candidates else None
    status = (
        "ambiguous"
        if len(supported) + len(candidates) > 1
        else "current_model_estimate"
        if current
        else "candidate"
        if candidates
        else "last_seen"
        if last_seen
        else "unknown"
    )
    last_seen.sort(key=lambda row: row["last_detected_at"], reverse=True)
    return {
        "enabled": True,
        "semantics": "unverified_model_inferences",
        "status": status,
        "current": current,
        "candidates": supported + candidates,
        "cameras": cameras,
        "last_seen": last_seen[0] if last_seen else None,
        "accuracy_verified": False,
        "sleep_established": False,
    }
