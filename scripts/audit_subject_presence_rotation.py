"""Score a complete, hash-bound subject rotation matrix against presence labels.

Reads metadata only. Presence labels cannot validate age, identity or state.
"""

import argparse
import json
import re
from pathlib import Path

from baby_monitor.local_runners.subject_probe import parse_response


def score(frames, records):
    if not frames or len(frames) > 1000:
        raise ValueError("invalid reference matrix")
    for row in frames:
        if row.get("human_presence") not in ("present", "absent") or not re.fullmatch(
            "[0-9a-f]{64}", row.get("sha256", "")
        ):
            raise ValueError("invalid reference label or fingerprint")
    seen = set()
    metrics = {
        str(rotation): {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "error_indices": []}
        for rotation in (0, 180)
    }
    for record in records:
        index, rotation = record.get("index"), record.get("rotation")
        if (
            type(index) is not int
            or not 0 <= index < len(frames)
            or type(rotation) is not int
            or rotation not in (0, 180)
            or (index, rotation) in seen
        ):
            raise ValueError("invalid or repeated matrix cell")
        seen.add((index, rotation))
        row = frames[index]
        if record.get("source_sha256") != row["sha256"]:
            raise ValueError("source binding mismatch")
        claims = parse_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(record.get("claims"), allow_nan=False)},
                    }
                ]
            }
        )
        present = row["human_presence"] == "present"
        predicted = bool(claims["people"])
        key = ("tp" if predicted else "fn") if present else ("fp" if predicted else "tn")
        metrics[str(rotation)][key] += 1
        if predicted != present:
            metrics[str(rotation)]["error_indices"].append(index)
    if len(seen) != len(frames) * 2:
        raise ValueError("incomplete matrix; failures are not negative detections")
    return {
        "semantics": "reused_reference_presence_only",
        "frames": len(frames),
        "calls": len(seen),
        "metrics": metrics,
        "age_accuracy_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    summary = json.loads((args.directory / "summary.json").read_text())
    print(json.dumps(score(protocol["frames"], summary["records"]), sort_keys=True))


if __name__ == "__main__":
    main()
