"""Offline global point/box assignment; geometry is not subject or role truth.

The existing live containment rule is deliberately unchanged. A unique complete
bipartite assignment only identifies a structural candidate on one exact image.
No media, model calls, role writeback, confidence sorting or temporal propagation.
"""

from __future__ import annotations

from baby_monitor.local_runners.subject_head_match import match_head_centers

REVISION = "unique_complete_subject_geometry_v1"


def _cover(options, *, forbidden=None):
    owners = {}

    def augment(subject, visited):
        for box in options[subject]:
            if (subject, box) == forbidden or box in visited:
                continue
            visited.add(box)
            if box not in owners or augment(owners[box], visited):
                owners[box] = subject
                return True
        return False

    for subject in range(len(options)):
        if not augment(subject, set()):
            return None
    assigned = [None] * len(options)
    for box, subject in owners.items():
        assigned[subject] = box
    return assigned


def diagnose_global_assignment(*, probe_input_sha256, detector_input_sha256, centers, boxes):
    """Require exact-image binding and a unique complete assignment, not a nearest box.

    Every located model head must have its own box. Nullable/unlocated heads and
    multiple equally possible full assignments cannot be repaired geometrically.
    Surplus or unconsumed boxes are not automatically false-positive people.
    Legacy unique matches are exported independently, never silently replaced.
    """
    if type(boxes) is not list or len(boxes) > 256:
        raise ValueError("bounded subject assignment boxes required")
    legacy = match_head_centers(
        probe_input_sha256=probe_input_sha256,
        head_input_sha256=detector_input_sha256,
        centers=centers,
        heads=boxes,
    )
    options = [row["head_indices"] for row in legacy["subjects"]]
    candidate = None
    alternatives = None
    if not centers:
        status = "no_model_subjects"
    elif any(point is None for point in centers):
        status = "unlocated_model_head"
    else:
        candidate = _cover(options)
        if candidate is None:
            status = "no_complete_matching"
        else:
            # Any second complete assignment must omit at least one edge in
            # the first. Testing each selected edge proves uniqueness without
            # enumerating every possible assignment or choosing a tie-breaker.
            alternatives = [
                subject
                for subject, box in enumerate(candidate)
                if _cover(options, forbidden=(subject, box)) is not None
            ]
            if alternatives:
                status = "multiple_complete_matchings"
                candidate = None
            else:
                status = "unique_complete_matching"
                for subject, row in enumerate(legacy["subjects"]):
                    if row["status"] == "unique_spatial_match":
                        assert candidate[subject] == row["head_indices"][0]
    return {
        "revision": REVISION,
        "input_sha256": legacy["input_sha256"],
        "semantics": "offline_single_image_geometry_only",
        "status": status,
        "candidate_box_indices": candidate,
        "alternative_edge_subject_indices": alternatives,
        "legacy": legacy,
        "model_claims_are_ground_truth": False,
        "accuracy_measured": False,
        "roles_propagated": False,
        "production_changed": False,
        "deployment_approved": False,
    }
