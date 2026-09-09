"""The ladder planner. Target at position 0, companions as nested prefixes of one
seeded permutation, so K varies the amount of trailing load and not which
criteria are present."""

from __future__ import annotations

import pytest

from packing.dataset import PairedItem
from packing.ladder_criteria import LADDER_CRITERIA
from packing.runner import plan_fixed_position

LEVELS = (1, 5, 10, 20, 40)


def _item(item_id: str, target: str | None) -> PairedItem:
    return PairedItem(
        item_id=item_id,
        prompt="p",
        response_a="a",
        response_b="b",
        human_a={},
        human_b={},
        dataset="test",
        meta={},
        source="synthetic",
        targeted_criterion=target,
    )


def test_target_is_always_at_position_zero():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    for spec in specs:
        assert spec.group[0].key == "precision", spec.nominal_n
        assert spec.partition.placement_for("precision").position == 0


def test_group_size_equals_the_requested_level():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    for spec in specs:
        assert len(spec.group) == spec.nominal_n


def test_companions_are_nested_prefixes_across_rungs():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    by_level = {
        s.nominal_n: [c.key for c in s.group] for s in specs if s.side == "a"
    }
    for smaller, larger in zip(LEVELS, LEVELS[1:]):
        assert by_level[larger][: len(by_level[smaller])] == by_level[smaller], (
            smaller,
            larger,
        )


def test_both_sides_share_the_companion_sequence():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=(40,)
    )
    a = [c.key for s in specs if s.side == "a" for c in s.group]
    b = [c.key for s in specs if s.side == "b" for c in s.group]
    assert a == b


def test_the_target_appears_once_per_call():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    for spec in specs:
        keys = [c.key for c in spec.group]
        assert keys.count("precision") == 1
        assert len(set(keys)) == len(keys)


def test_companion_order_varies_between_prompts():
    orders = set()
    for i in range(20):
        specs = plan_fixed_position(
            [_item(f"i{i}", "precision")], LADDER_CRITERIA, model="mock", levels=(40,)
        )
        orders.add(tuple(c.key for c in specs[0].group[1:]))
    assert len(orders) > 1, "each prompt draws its own companion permutation"


def test_call_count_is_two_sides_times_levels():
    specs = plan_fixed_position(
        [_item("i1", "precision"), _item("i2", "ordering")],
        LADDER_CRITERIA,
        model="mock",
        levels=LEVELS,
    )
    assert len(specs) == 2 * len(LEVELS) * 2


def test_identities_are_unique():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    identities = [s.identity for s in specs]
    assert len(set(identities)) == len(identities)


def test_level_above_the_pool_is_rejected():
    with pytest.raises(ValueError, match="level 41 exceeds pool size 40"):
        plan_fixed_position(
            [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=(41,)
        )


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="has no targeted_criterion"):
        plan_fixed_position(
            [_item("i1", None)], LADDER_CRITERIA, model="mock", levels=(5,)
        )


def test_target_outside_the_pool_is_rejected():
    with pytest.raises(ValueError, match="not in the criterion pool"):
        plan_fixed_position(
            [_item("i1", "not_a_criterion")],
            LADDER_CRITERIA,
            model="mock",
            levels=(5,),
        )
