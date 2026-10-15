"""Build and validate a text-free historical model diagnostic for local desktop UI."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.local_runners.subject_head_match import match_head_centers
from baby_monitor.local_runners.subject_probe import parse_response

STATUSES = {"head_not_located", "no_head_match", "ambiguous_head_match", "unique_spatial_match"}


def validate(value, *, now=None):
    now = now or datetime.now(UTC)
    if type(value) is not dict or set(value) != {"version", "scope", "camera_id", "rows"}:
        raise ValueError("invalid diagnostic fields")
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or value["scope"] != "historical_model_claims"
    ):
        raise ValueError("invalid diagnostic scope")
    if value["camera_id"] not in ("camera_01", "camera_02", "camera_03", "camera_04"):
        raise ValueError("invalid diagnostic camera")
    if type(value["rows"]) is not list or not 1 <= len(value["rows"]) <= 8:
        raise ValueError("invalid diagnostic rows")
    rows = []
    for row in value["rows"]:
        if type(row) is not dict or set(row) != {"received_at", "subjects"}:
            raise ValueError("invalid diagnostic row")
        if type(row["received_at"]) is not str or len(row["received_at"]) > 40:
            raise ValueError("invalid diagnostic time")
        stamp = datetime.fromisoformat(row["received_at"])
        if stamp.utcoffset() is None or stamp > now:
            raise ValueError("invalid diagnostic time")
        if type(row["subjects"]) is not list or len(row["subjects"]) > 8:
            raise ValueError("invalid diagnostic subjects")
        subjects = []
        for subject in row["subjects"]:
            if type(subject) is not dict or set(subject) != {"age_claim", "spatial_status"}:
                raise ValueError("invalid diagnostic subject")
            if (
                subject["age_claim"] not in ("baby", "adult", "unknown")
                or type(subject["spatial_status"]) is not str
                or subject["spatial_status"] not in STATUSES
            ):
                raise ValueError("invalid diagnostic claim")
            subjects.append(dict(subject))
        rows.append(
            dict(
                received_at=stamp.astimezone(UTC).isoformat(),
                age_seconds=round((now - stamp).total_seconds(), 1),
                subjects=subjects,
            )
        )
    return dict(version=1, scope="historical_model_claims", camera_id=value["camera_id"], rows=rows)


def build(capture, protocol, probe, heads):
    camera = protocol["camera_id"]
    if len(probe["records"]) > 8:
        raise ValueError("too many diagnostic frames")
    result = []
    for record in probe["records"]:
        index = record["index"]
        if type(index) is not int or not 0 <= index < len(capture["frames"]):
            raise ValueError("invalid frame index")
        frame = capture["frames"][index]
        response = record["result"]
        if (
            response.get("version") != 1
            or response.get("semantics") != "unverified_model_claims"
            or response.get("coordinate_space") != "submitted_image_thousandths"
        ):
            raise ValueError("invalid source diagnostic contract")
        if (
            record["source_sha256"] != frame["sha256"]
            or response["input_sha256"] != frame["sha256"]
        ):
            raise ValueError("different input frames")
        candidates = [h for h in heads["records"] if h["index"] == index and h["rotation"] == 180]
        if len(candidates) != 1:
            raise ValueError("missing or duplicate head diagnostic")
        head = candidates[0]
        claims = parse_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(response["claims"], allow_nan=False)},
                    }
                ]
            }
        )
        match = match_head_centers(
            probe_input_sha256=response["input_sha256"],
            head_input_sha256=head["source_sha256"],
            centers=[p["head_center"] for p in claims["people"]],
            heads=[h["bbox_xyxy"] for h in head["heads"]],
        )
        result.append(
            dict(
                received_at=frame["received_at"],
                subjects=[
                    dict(age_claim=p["age_group"], spatial_status=m["status"])
                    for p, m in zip(claims["people"], match["subjects"], strict=True)
                ],
            )
        )
    output = dict(version=1, scope="historical_model_claims", camera_id=camera, rows=result)
    validate(output)
    return output


def main():
    from scripts.live_desktop_backend import read_private_status

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "probe", "heads", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    # Read bounded private JSON metadata only; never open the frame paths it mentions.
    capture = read_private_status(args.capture / "frames.json")
    protocol = read_private_status(args.capture / "capture_protocol.json")
    probe = read_private_status(args.probe / "summary.json")
    heads = read_private_status(args.heads / "summary.json", max_bytes=2 * 1024 * 1024)
    value = build(capture, protocol, probe, heads)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = args.output.parent.stat()
    if (
        args.output.parent.resolve() != args.output.parent
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError("unsafe export directory")
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False)
    print(json.dumps({"exported_rows": len(value["rows"]), "media_read": False}))


if __name__ == "__main__":
    main()
