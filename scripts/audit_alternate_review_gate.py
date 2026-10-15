"""Replay saved detector scores without reading media or changing live policy."""

import argparse
import json
from collections import Counter
from pathlib import Path

from baby_monitor.domain import Detection
from baby_monitor.live_detection import AlternateViewReviewPolicy


def audit(rows, threshold):
    policy = AlternateViewReviewPolicy(minimum_frame_confidence=threshold)
    groups = {}
    withheld = []
    for row in rows:
        scores = row.get("confidences")
        if scores is None:
            scores = [box["confidence"] for box in row.get("boxes", row.get("hints") or [])]
        detections = tuple(Detection(0, "person", score, (0, 0, 1, 1)) for score in scores)
        key = f"{row.get('shape', 'cuda')}:{row.get('rotation', 'review')}"
        counts = groups.setdefault(key, Counter())
        eligible = policy.frame_eligible(detections)
        counts["rows"] += 1
        counts["raw_nonempty"] += bool(scores)
        counts["eligible"] += eligible
        counts["withheld_nonempty"] += bool(scores) and not eligible
        counts["raw_boxes"] += len(scores)
        counts["eligible_frame_boxes"] += len(scores) if eligible else 0
        if scores and not eligible:
            withheld.append(
                {"group": key, "index": row.get("index"), "case": row.get("case"), "scores": scores}
            )
        if "truth_presence" in row:
            counts[("true" if row["truth_presence"] else "false") + "_eligible"] += eligible
    return {"threshold": threshold, "media_read": False, "groups": groups, "withheld": withheld}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.results.read_text()), args.threshold), sort_keys=True))


if __name__ == "__main__":
    main()
