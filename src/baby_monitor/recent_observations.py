"""Local-only, read-only view of the last recorded description per camera.

This is not a live baby locator. Descriptions are model output, not independently
verified observations; in particular a stored sleeping label is not sleep proof.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .domain import VLMEvent
from .home_event_digest import HomeEventDigestError, _private_existing_database
from .room_registry import RoomRegistry

_MAX_RESULT_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def read_recent_observations(
    database_path: str | Path,
    registry: RoomRegistry,
    *,
    now: datetime | None = None,
    recent_seconds: int = 120,
) -> dict[str, object]:
    """Read at most one bounded result per registered stream, never media.

    Mapping describes the current registry, not necessarily the camera's
    historical placement. Even recent events never establish current presence.
    """
    checked_at = now if now is not None else datetime.now(UTC)
    if not isinstance(checked_at, datetime) or checked_at.utcoffset() is None:
        raise ValueError("recent observations require an aware timestamp")
    if type(recent_seconds) is not int or not 1 <= recent_seconds <= 3600:
        raise ValueError("recent observations window must be 1..3600 seconds")
    if not isinstance(registry, RoomRegistry):
        raise ValueError("recent observations require a room registry")
    checked_at = checked_at.astimezone(UTC)
    try:
        database = _private_existing_database(database_path)
    except HomeEventDigestError:
        raise ValueError("recent observations database is unavailable") from None
    names = {room.room_id: room.display_name for room in registry.rooms}
    entries: list[dict[str, object]] = []
    try:
        with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("BEGIN")
            for stream in registry.logical_streams:
                entry: dict[str, object] = {
                    "camera_id": stream.camera_id,
                    "mapped_room": names[stream.room_id],
                    "status": "no_record",
                }
                # Do not read source_ref, media paths, candidate data or profiles.
                row = connection.execute(
                    "SELECT occurred_at, CASE WHEN typeof(result_json) = 'text' "
                    "AND length(CAST(result_json AS BLOB)) BETWEEN 1 AND ? "
                    "THEN result_json ELSE NULL END FROM events WHERE camera_id = ? "
                    "ORDER BY occurred_at DESC, event_id DESC LIMIT 1",
                    (_MAX_RESULT_BYTES, stream.camera_id),
                ).fetchone()
                if row is not None:
                    try:
                        recorded_at = datetime.fromisoformat(row[0])
                        if recorded_at.utcoffset() is None:
                            raise ValueError("naive event time")
                        age = (checked_at - recorded_at).total_seconds()
                        if age < 0:
                            raise ValueError("future event time")
                        json.loads(row[1], object_pairs_hook=_unique_object)
                        result = VLMEvent.from_json(row[1])
                    except (ValueError, TypeError, OverflowError, RecursionError):
                        entry["status"] = "invalid_record"
                    else:
                        entry.update(
                            status="recent_record" if age <= recent_seconds else "stale_record",
                            recorded_at=recorded_at.astimezone(UTC).isoformat(),
                            age_seconds=round(age, 3),
                            model_reported={
                                "observations": list(result.observations),
                                "inferences": list(result.inferences),
                                "event_type": result.event_type.value,
                                "needs_remote_review": result.needs_remote_review,
                            },
                        )
                entries.append(entry)
    except sqlite3.Error:
        raise ValueError("recent observations database is invalid") from None
    return {
        "version": 1,
        "checked_at": checked_at.isoformat(),
        "recent_seconds": recent_seconds,
        "baby_location": "unknown",
        "baby_state": "unknown",
        "live_state_established": False,
        "room_mapping_basis": "current_registry_not_historical_proof",
        "description_basis": "stored_model_output_not_independently_verified",
        "contains_model_text": True,
        "media_read": False,
        "entries": entries,
    }
