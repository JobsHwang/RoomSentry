"""Same-frame geometric comparison of age claims, never an identity decision."""

from __future__ import annotations

from baby_monitor.local_runners.subject_head_match import match_head_centers


def _match(source_sha256, view_source_sha256, people, heads):
    if type(people) is not list or len(people) > 8:
        raise ValueError("invalid source-space claims")
    for person in people:
        if (
            type(person) is not dict
            or set(person) != {"age_group", "head_center"}
            or person["age_group"] not in ("baby", "adult", "unknown")
        ):
            raise ValueError("invalid source-space claim")
    return match_head_centers(
        probe_input_sha256=source_sha256,
        head_input_sha256=view_source_sha256,
        centers=[p["head_center"] for p in people],
        heads=heads,
    )["subjects"]


def compare_views(
    *,
    source_sha256: str,
    primary_source_sha256: str,
    check_source_sha256: str,
    primary_people: list,
    check_people: list,
    heads: list,
) -> dict:
    """Compare only uniquely grounded points within the exact same source frame.

    The caller must verify each submitted view's pixels and inverse transform
    before passing source-space integer-thousandths centers. Source hashes bind
    the common original; they are not the hashes of crop or rotated encodings.
    Candidate indices last for this frame only. Missing points, absent reports,
    ambiguous correspondence and age abstention never become age agreement.
    """
    primary_matches = _match(source_sha256, primary_source_sha256, primary_people, heads)
    check_matches = _match(source_sha256, check_source_sha256, check_people, heads)
    rows = []
    for head_index in range(len(heads)):
        primary = [
            (p, m)
            for p, m in zip(primary_people, primary_matches, strict=True)
            if head_index in m["head_indices"]
        ]
        check = [
            (p, m)
            for p, m in zip(check_people, check_matches, strict=True)
            if head_index in m["head_indices"]
        ]
        if any(m["status"] != "unique_spatial_match" for _, m in primary + check):
            status = "ambiguous_correspondence"
        elif not primary or not check:
            status = "not_comparable"
        else:
            first, second = primary[0][0]["age_group"], check[0][0]["age_group"]
            if "unknown" in (first, second):
                status = "age_abstention"
            elif first != second:
                status = "age_conflict"
            else:
                status = "compatible_age_claims"
        rows.append(
            {
                "head_index": head_index,
                "status": status,
                "primary_age_claims": [p["age_group"] for p, _ in primary],
                "check_age_claims": [p["age_group"] for p, _ in check],
            }
        )
    return {
        "source_sha256": source_sha256,
        "semantics": "same_frame_geometric_diagnostic",
        "heads": rows,
        "unresolved_primary_indices": [
            i for i, m in enumerate(primary_matches) if m["status"] != "unique_spatial_match"
        ],
        "unresolved_check_indices": [
            i for i, m in enumerate(check_matches) if m["status"] != "unique_spatial_match"
        ],
        "age_verified": False,
        "identity_established": False,
    }
