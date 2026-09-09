"""Eligibility is decided from the source response alone, before any outcome
exists and before an operation is assigned."""

from __future__ import annotations

from packing.dataset import PairedItem
from packing.eligibility import (
    CORE_OPERATIONS,
    NUMBERS_OPERATION,
    applies,
    select_response,
)

LONG = "One sentence here. Two sentences here. Three sentences here."


def _item(a: str, b: str) -> PairedItem:
    return PairedItem(
        item_id="t-1",
        prompt="Explain the thing.",
        response_a=a,
        response_b=b,
        human_a={},
        human_b={},
        dataset="test",
        meta={},
        source="synthetic",
    )


def test_core_is_the_eight_high_applicability_operations():
    assert CORE_OPERATIONS == (
        "flatten", "truncate", "shuffle", "drop_answer",
        "agreement", "depunct", "duplicate", "filler",
    )
    assert NUMBERS_OPERATION == "numbers"


def test_numbers_needs_a_number_outside_code():
    with_number = "The limit is 30 seconds. It applies broadly. And then more."
    without = "The limit is short. It applies broadly. And then more text here."
    assert applies("numbers", with_number, "prompt", seed=1) is True
    assert applies("numbers", without, "prompt", seed=1) is False


def test_select_response_returns_none_when_neither_side_qualifies():
    assert select_response(_item("no.", "no."), ("shuffle",), seed=7) is None


def test_select_response_picks_the_only_eligible_side():
    assert select_response(_item("no.", LONG), ("shuffle",), seed=7) == "b"
    assert select_response(_item(LONG, "no."), ("shuffle",), seed=7) == "a"


def test_select_response_is_deterministic_when_both_qualify():
    item = _item(LONG, LONG)
    first = select_response(item, ("shuffle",), seed=7)
    assert first in {"a", "b"}
    assert select_response(item, ("shuffle",), seed=7) == first


def test_both_eligible_split_is_not_always_the_same_side():
    picks = set()
    for seed in range(40):
        picks.add(select_response(_item(LONG, LONG), ("shuffle",), seed=seed))
    assert picks == {"a", "b"}, "side choice must vary with the seed"


def test_a_side_must_satisfy_every_operation_in_the_set():
    # `numbers` needs a numeral; `shuffle` needs three prose sentences. Side a
    # has the numeral but too few sentences, side b has sentences but no numeral,
    # so neither satisfies both and the prompt is ineligible for the pair.
    a = "The limit is 30 seconds."
    b = LONG
    assert select_response(_item(a, b), ("numbers",), seed=3) == "a"
    assert select_response(_item(a, b), ("shuffle",), seed=3) == "b"
    assert select_response(_item(a, b), ("numbers", "shuffle"), seed=3) is None


def test_unknown_operation_raises_rather_than_returning_none():
    import pytest

    with pytest.raises(KeyError, match="nope"):
        select_response(_item(LONG, LONG), ("nope",), seed=1)
