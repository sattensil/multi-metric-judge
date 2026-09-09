"""The 40-criterion ladder pool. The pool is the instrument, so its invariants
are mechanical rather than reviewed by eye."""

from __future__ import annotations

from packing.criteria import HELPSTEER2_CRITERIA
from packing.ladder_criteria import (
    LADDER_CRITERIA,
    LADDER_MANIFEST,
    LADDER_ONLY_CRITERIA,
)
from packing.synthetic_criteria import SYNTHETIC_ONLY_CRITERIA

EXPECTED_KEYS = {
    "actionability", "assumptions", "attribution", "audience_fit", "caveating",
    "cohesion", "concreteness", "edge_cases", "example_quality",
    "internal_reference", "ordering", "precision", "question_coverage",
    "redundancy", "register", "scope_discipline", "self_consistency",
    "signposting", "terminology_consistency", "unit_handling",
}


def test_there_are_exactly_twenty_new_criteria():
    assert len(LADDER_ONLY_CRITERIA) == 20
    assert {c.key for c in LADDER_ONLY_CRITERIA} == EXPECTED_KEYS


def test_the_pool_is_forty_and_composed_of_the_three_tiers():
    assert len(LADDER_CRITERIA) == 40
    assert LADDER_CRITERIA[:5] == HELPSTEER2_CRITERIA
    assert LADDER_CRITERIA[5:20] == SYNTHETIC_ONLY_CRITERIA
    assert LADDER_CRITERIA[20:] == LADDER_ONLY_CRITERIA


def test_manifest_validates_at_forty():
    LADDER_MANIFEST.validate(expect=40)


def test_every_scale_point_is_anchored_with_nonempty_prose():
    for c in LADDER_ONLY_CRITERIA:
        assert set(c.anchors) == {0, 1, 2, 3, 4}, c.key
        for point, text in c.anchors.items():
            assert text.strip(), f"{c.key} anchor {point} is empty"
            assert len(text.strip()) >= 10, f"{c.key} anchor {point} is too terse"


def test_no_duplicate_anchor_prose_within_a_criterion():
    for c in LADDER_ONLY_CRITERIA:
        texts = [t.strip().lower() for t in c.anchors.values()]
        assert len(set(texts)) == 5, f"{c.key} reuses anchor prose across points"


def test_descriptions_are_distinct_and_present():
    descriptions = [c.description.strip() for c in LADDER_ONLY_CRITERIA]
    assert all(descriptions)
    assert len(set(descriptions)) == 20


def test_length_distribution_of_the_new_twenty_is_bounded():
    # Scoped to the twenty this task authors. The full manifest includes legacy
    # criteria the task cannot alter, so bounding the whole pool would fail for
    # reasons outside its control.
    lengths = [
        len(c.description) + sum(len(v) for v in c.anchors.values())
        for c in LADDER_ONLY_CRITERIA
    ]
    assert max(lengths) <= 4 * min(lengths), sorted(lengths)


def test_serialised_token_length_is_recorded_for_the_pool():
    # Characters are a proxy; the rendered rubric is what is billed. Recording
    # both keeps the cost model honest when anchor prose changes.
    rendered = [len(c.render()) // 4 for c in LADDER_ONLY_CRITERIA]
    assert min(rendered) > 0
    assert max(rendered) <= 4 * min(rendered), sorted(rendered)


def test_manifest_hash_is_recorded_for_the_run():
    assert len(LADDER_MANIFEST.hash) == 16
    assert len(LADDER_MANIFEST.full_hash) == 64
