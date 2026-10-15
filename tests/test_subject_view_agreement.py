import unittest

from baby_monitor.local_runners.subject_view_agreement import compare_views


def person(age, x=100, y=100):
    return {"age_group": age, "head_center": None if x is None else [x, y]}


class SubjectViewAgreementTests(unittest.TestCase):
    def compare(self, primary, check, heads=None, **changes):
        arguments = dict(
            source_sha256="a" * 64,
            primary_source_sha256="a" * 64,
            check_source_sha256="a" * 64,
            primary_people=primary,
            check_people=check,
            heads=heads if heads is not None else [[0, 0, 0.2, 0.2]],
        )
        arguments.update(changes)
        return compare_views(**arguments)

    def test_row_order_does_not_swap_two_people(self):
        result = self.compare(
            [person("baby"), person("adult", 800, 800)],
            [person("baby", 800, 800), person("baby")],
            [[0, 0, 0.2, 0.2], [0.7, 0.7, 0.9, 0.9]],
        )
        self.assertEqual(
            ["compatible_age_claims", "age_conflict"],
            [h["status"] for h in result["heads"]],
        )
        self.assertFalse(result["age_verified"])
        self.assertFalse(result["identity_established"])

    def test_unknown_is_abstention_even_when_both_views_match(self):
        for first, second in (("unknown", "unknown"), ("baby", "unknown")):
            with self.subTest(first=first, second=second):
                result = self.compare([person(first)], [person(second)])
                self.assertEqual("age_abstention", result["heads"][0]["status"])

    def test_missing_and_unlocated_people_are_not_agreement_or_conflict(self):
        for primary, check in (
            ([], []),
            ([person("baby")], []),
            ([person("baby", None)], [person("adult")]),
        ):
            with self.subTest(primary=primary, check=check):
                result = self.compare(primary, check)
                self.assertEqual("not_comparable", result["heads"][0]["status"])

    def test_two_points_in_one_head_do_not_vote_or_force_correspondence(self):
        result = self.compare([person("baby"), person("adult")], [person("baby")])
        self.assertEqual("ambiguous_correspondence", result["heads"][0]["status"])
        self.assertEqual([0, 1], result["unresolved_primary_indices"])

    def test_overlapping_head_candidates_keep_both_ambiguous(self):
        result = self.compare(
            [person("baby")], [person("adult")], [[0, 0, 0.2, 0.2], [0.05, 0.05, 0.3, 0.3]]
        )
        self.assertEqual(["ambiguous_correspondence"] * 2, [h["status"] for h in result["heads"]])

    def test_different_source_frames_rejected_even_with_identical_points(self):
        for field in ("primary_source_sha256", "check_source_sha256"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare([person("baby")], [person("baby")], **{field: "b" * 64})

    def test_detector_missing_head_does_not_discard_unmatched_claim(self):
        result = self.compare([person("baby")], [person("adult")], [])
        self.assertEqual([], result["heads"])
        self.assertEqual([0], result["unresolved_primary_indices"])
        self.assertEqual([0], result["unresolved_check_indices"])

    def test_invalid_claim_shapes_and_coordinates_rejected(self):
        for value in (
            [dict(person("baby"), confidence=1)],
            [person("father")],
            [person("baby", True)],
            [person("baby", 1001)],
            [person("baby")] * 9,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare(value, [])
