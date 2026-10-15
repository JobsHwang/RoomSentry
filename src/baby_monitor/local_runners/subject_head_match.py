"""Spatial agreement of independent single-image outputs, never role truth.

Both callers must supply the hash of the exact same submitted image and map
their coordinates to that image. A match does not identify a person or permit
propagation of an age claim to a track, another frame, or another camera.
"""

import math
import re


def match_head_centers(*, probe_input_sha256, head_input_sha256, centers, heads):
    """Return only one-to-one point/box containment diagnostics.

    Centers are nullable integer thousandths; head boxes are normalized xyxy.
    Multiple points in one box and a point in overlapping boxes stay ambiguous.
    Never choose the nearest box to force agreement or use confidence as truth.
    """
    for digest in (probe_input_sha256, head_input_sha256):
        if type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid input fingerprint")
    if probe_input_sha256 != head_input_sha256:
        raise ValueError("different input images")
    if type(centers) is not list or len(centers) > 8:
        raise ValueError("invalid center list")
    if type(heads) is not list or len(heads) > 4096:
        raise ValueError("invalid head list")
    for box in heads:
        if (
            type(box) is not list
            or len(box) != 4
            or any(
                type(n) not in (int, float) or not 0 <= n <= 1 or not math.isfinite(n) for n in box
            )
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ValueError("invalid normalized head box")
    matches = []
    for point in centers:
        if point is not None and (
            type(point) is not list
            or len(point) != 2
            or any(type(n) is not int or not 0 <= n <= 1000 for n in point)
        ):
            raise ValueError("invalid head center")
        matches.append(
            []
            if point is None
            else [
                index
                for index, box in enumerate(heads)
                if box[0] <= point[0] / 1000 <= box[2] and box[1] <= point[1] / 1000 <= box[3]
            ]
        )
    consumers = {index: sum(index in row for row in matches) for index in range(len(heads))}
    results = []
    for point, matched in zip(centers, matches, strict=True):
        if point is None:
            status = "head_not_located"
        elif not matched:
            status = "no_head_match"
        elif len(matched) != 1 or consumers[matched[0]] != 1:
            status = "ambiguous_head_match"
        else:
            status = "unique_spatial_match"
        results.append({"status": status, "head_indices": matched})
    return {
        "input_sha256": probe_input_sha256,
        "semantics": "spatial_agreement_only",
        "subjects": results,
        "unmatched_head_indices": [i for i, count in consumers.items() if count == 0],
    }
