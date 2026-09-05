"""Label-side gating: tie structure and the inclusion filter."""

from __future__ import annotations

import pytest

from packing.criteria import get_criteria
from packing.dataset import (
    PairedItem,
    filter_min_scoreable,
    scoreable_count,
    scoreable_report,
    validate,
)

CRITERIA = get_criteria("helpsteer2")
KEYS = [c.key for c in CRITERIA]


def _item(item_id: str, deltas: list[int]) -> PairedItem:
    """deltas[i] == 0 makes criterion i a tie; non-zero makes it scoreable."""
    human_a, human_b = {}, {}
    for key, delta in zip(KEYS, deltas):
        human_a[key] = 2
        human_b[key] = 2 + delta
    return PairedItem(
        item_id=item_id,
        prompt="p",
        response_a="a",
        response_b="b",
        human_a=human_a,
        human_b=human_b,
        dataset="test",
        meta={},
    )


def test_scoreable_count() -> None:
    assert scoreable_count(_item("x", [0, 0, 0, 0, 0]), CRITERIA) == 0
    assert scoreable_count(_item("x", [1, 0, 0, 0, 0]), CRITERIA) == 1
    assert scoreable_count(_item("x", [1, 1, -1, 1, -2]), CRITERIA) == 5


def test_filter_keeps_only_eligible_items() -> None:
    items = [
        _item("none", [0, 0, 0, 0, 0]),
        _item("two", [1, 1, 0, 0, 0]),
        _item("four", [1, 1, 1, 1, 0]),
        _item("five", [1, 1, 1, 1, 1]),
    ]
    assert [i.item_id for i in filter_min_scoreable(items, CRITERIA, 4)] == ["four", "five"]
    assert [i.item_id for i in filter_min_scoreable(items, CRITERIA, 2)] == [
        "two",
        "four",
        "five",
    ]


def test_filter_zero_is_a_passthrough() -> None:
    items = [_item("a", [0, 0, 0, 0, 0]), _item("b", [1, 0, 0, 0, 0])]
    assert len(filter_min_scoreable(items, CRITERIA, 0)) == 2


def test_filter_is_orthogonal_to_group_size() -> None:
    """The design's key defence: the filter depends only on human labels.

    It touches nothing about judging or packing, so every surviving item remains
    fully crossed with group size. Pinned as a test because the argument that the
    filter cannot bias the estimand rests entirely on this property.
    """
    items = [_item(f"i{n}", [1, 1, 1, 1, 0]) for n in range(5)]
    kept = filter_min_scoreable(items, CRITERIA, 4)
    assert len(kept) == len(items)
    # Filtering is idempotent and order-preserving: no hidden reshuffling that
    # could correlate with anything downstream.
    assert filter_min_scoreable(kept, CRITERIA, 4) == kept


def test_scoreable_report_shape() -> None:
    items = [
        _item("a", [1, 1, 1, 1, 1]),
        _item("b", [1, 1, 0, 0, 0]),
        _item("c", [0, 0, 0, 0, 0]),
    ]
    report = scoreable_report(items, CRITERIA)
    assert report["n_items"] == 3
    assert report["items_with_zero_scoreable"] == 1
    assert report["mean_scoreable_per_item"] == pytest.approx((5 + 2 + 0) / 3)
    # helpfulness is scoreable in a and b, tied in c
    assert report["per_criterion"]["helpfulness"]["usable_n"] == 2
    assert report["per_criterion"]["helpfulness"]["human_tie_rate"] == pytest.approx(1 / 3)
    assert sum(report["scoreable_per_item_distribution"].values()) == 3


def test_scoreable_report_handles_empty() -> None:
    assert scoreable_report([], CRITERIA) == {"n_items": 0}


def test_validate_rejects_out_of_scale() -> None:
    bad = _item("bad", [0, 0, 0, 0, 0])
    bad.human_b["helpfulness"] = 99
    with pytest.raises(ValueError, match="outside"):
        validate([bad], CRITERIA)
