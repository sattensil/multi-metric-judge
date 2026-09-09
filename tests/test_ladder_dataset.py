"""Operation assignment must be stable under extension, and each prompt gets one
frozen corruption realisation reused at every rung."""

from __future__ import annotations

from collections import Counter

import pytest

from packing.dataset import PairedItem
from packing.eligibility import CORE_OPERATIONS
from packing.ladder_criteria import LADDER_CRITERIA
from packing.ladder_dataset import activate, build_ladder_item, operation_queues

TEXT_A = (
    "First sentence of the answer. Second sentence adds detail. "
    "Third sentence closes it out. Fourth sentence is spare."
)
# Deliberately different from TEXT_A. An earlier version of this fixture used the
# same string for both sides, which hid a real bug: the builder was pairing the
# corrupted A against the untouched B, two different answers, rather than clean
# against corrupted of one source response.
TEXT_B = (
    "Alternative opening line here. A second line carries more detail. "
    "A third line wraps the reply. A fourth line is filler."
)


def _base(item_id: str) -> PairedItem:
    return PairedItem(
        item_id=item_id,
        prompt="Explain the thing.",
        response_a=TEXT_A,
        response_b=TEXT_B,
        human_a={},
        human_b={},
        dataset="test",
        meta={},
    )


def test_queues_are_equal_sized_and_disjoint():
    ids = [f"p{i}" for i in range(96 * 8)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=96)
    assert set(queues) == set(CORE_OPERATIONS)
    assert all(len(q) == 96 for q in queues.values())
    flat = [i for q in queues.values() for i in q]
    assert len(set(flat)) == len(flat)


def test_activation_yields_exact_equal_quotas():
    ids = [f"p{i}" for i in range(96 * 8)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=96)
    assigned = activate(queues, per_op=58)
    counts = Counter(assigned.values())
    assert set(counts) == set(CORE_OPERATIONS)
    assert all(c == 58 for c in counts.values())
    assert len(assigned) == 58 * 8


def test_extending_does_not_reassign_any_earlier_prompt():
    ids = [f"p{i}" for i in range(96 * 8)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=96)
    first = activate(queues, per_op=58)
    second = activate(queues, per_op=65)
    for item_id, op in first.items():
        assert second[item_id] == op, item_id
    assert len(second) == 65 * 8


def test_queues_are_deterministic_for_a_seed():
    ids = [f"p{i}" for i in range(80)]
    assert operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10) == (
        operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10)
    )


def test_queues_are_independent_of_input_order():
    ids = [f"p{i}" for i in range(80)]
    assert operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10) == (
        operation_queues(list(reversed(ids)), CORE_OPERATIONS, seed=5, per_op_max=10)
    )


def test_a_frame_too_small_for_the_maximum_is_rejected():
    with pytest.raises(ValueError, match="needs 800 units, frame has 80"):
        operation_queues(
            [f"p{i}" for i in range(80)], CORE_OPERATIONS, seed=5, per_op_max=100
        )


def test_activating_beyond_the_queue_is_rejected():
    ids = [f"p{i}" for i in range(80)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10)
    with pytest.raises(ValueError, match="per_op 11 exceeds queue depth 10"):
        activate(queues, per_op=11)


def test_realisation_is_identical_across_repeated_builds():
    base = _base("p1")
    first = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    second = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert first is not None
    assert first.response_a == second.response_a
    assert first.response_b == second.response_b
    assert first.meta["corrupted_side"] == second.meta["corrupted_side"]


def test_the_targeted_criterion_is_the_operations_target():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.targeted_criterion == "coherence"


def test_base_item_key_is_set_for_clustering():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.meta["base_item"] == "p1"


def test_inapplicable_operation_yields_none_rather_than_an_unchanged_item():
    base = PairedItem(
        item_id="p2", prompt="p", response_a="Short.", response_b="Short.",
        human_a={}, human_b={}, dataset="test", meta={},
    )
    assert build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9) is None


def test_the_pair_is_clean_against_corrupted_of_one_source_response():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    halves = {item.response_a, item.response_b}
    # Exactly one half is the untouched source response, the other is its
    # corruption. TEXT_B must not appear at all: side "a" was selected.
    assert TEXT_A in halves, "the clean source response must be one half"
    assert TEXT_B not in halves, "the unselected response must not enter the pair"
    assert len(halves) == 2, "the two halves must differ"


def test_selecting_side_b_uses_only_side_b():
    item = build_ladder_item(_base("p1"), "shuffle", "b", LADDER_CRITERIA, seed=9)
    halves = {item.response_a, item.response_b}
    assert TEXT_B in halves
    assert TEXT_A not in halves


def test_ground_truth_is_encoded_as_pseudo_human_labels():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    target = item.targeted_criterion
    # shuffle lowers coherence, so the corrupted half must score lower on target.
    corrupted_side = item.meta["corrupted_side"]
    labels = (item.human_a, item.human_b)
    corrupted_labels = labels[0] if corrupted_side == "a" else labels[1]
    clean_labels = labels[1] if corrupted_side == "a" else labels[0]
    assert corrupted_labels[target] < clean_labels[target]


def test_companion_criteria_are_labelled_neutral_so_they_tie():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    target = item.targeted_criterion
    for key in (c.key for c in LADDER_CRITERIA):
        if key == target:
            continue
        assert item.human_a[key] == item.human_b[key], key


def test_every_pool_criterion_has_a_label():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    keys = {c.key for c in LADDER_CRITERIA}
    assert set(item.human_a) == keys
    assert set(item.human_b) == keys


def test_the_corrupted_half_is_not_always_the_same_side():
    sides = set()
    for i in range(30):
        item = build_ladder_item(
            _base(f"q{i}"), "shuffle", "a", LADDER_CRITERIA, seed=9
        )
        sides.add(item.meta["corrupted_side"])
    assert sides == {"a", "b"}, "side placement must be randomised"


def test_an_unknown_side_is_rejected():
    with pytest.raises(ValueError, match="side must be"):
        build_ladder_item(_base("p1"), "shuffle", "c", LADDER_CRITERIA, seed=9)
