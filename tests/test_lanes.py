"""Lanes are exact index splits of a seed-permuted frame, and each lane is
consumed in seeded order rather than file order."""

from __future__ import annotations

import pytest

from packing.lanes import fill_lane, split_lanes


def test_lane_sizes_are_exact():
    ids = [f"p{i}" for i in range(1222)]
    core, numbers = split_lanes(ids, seed=11, core_candidates=508)
    assert len(core) == 508
    assert len(numbers) == 714


def test_lanes_are_disjoint_and_cover_the_frame():
    ids = [f"p{i}" for i in range(100)]
    core, numbers = split_lanes(ids, seed=11, core_candidates=40)
    assert set(core) & set(numbers) == set()
    assert set(core) | set(numbers) == set(ids)


def test_split_is_deterministic_for_a_seed():
    ids = [f"p{i}" for i in range(100)]
    assert split_lanes(ids, seed=11, core_candidates=40) == split_lanes(
        ids, seed=11, core_candidates=40
    )


def test_split_changes_with_the_seed():
    ids = [f"p{i}" for i in range(100)]
    assert split_lanes(ids, seed=11, core_candidates=40) != split_lanes(
        ids, seed=12, core_candidates=40
    )


def test_split_is_independent_of_input_order():
    ids = [f"p{i}" for i in range(200)]
    a = split_lanes(ids, seed=11, core_candidates=80)
    b = split_lanes(list(reversed(ids)), seed=11, core_candidates=80)
    assert a == b, "lane content must not depend on file order"


def test_within_lane_order_is_not_file_order():
    ids = [f"p{i:04d}" for i in range(400)]
    core, _ = split_lanes(ids, seed=11, core_candidates=200)
    assert core != sorted(core), "consumption order must be seeded, not sorted"


def test_frame_too_small_for_the_requested_core_lane_is_rejected():
    with pytest.raises(ValueError, match="frame of 10 cannot supply 20"):
        split_lanes([f"p{i}" for i in range(10)], seed=1, core_candidates=20)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate base-prompt ids"):
        split_lanes(["p1", "p1", "p2"], seed=1, core_candidates=1)


def test_fill_lane_stops_at_the_quota_and_reports_the_remainder():
    ids = [f"p{i}" for i in range(20)]
    filled, remainder = fill_lane(ids, lambda i: True, quota=5)
    assert filled == ids[:5]
    assert remainder == ids[5:]


def test_fill_lane_skips_ineligible_prompts_in_frozen_order():
    ids = [f"p{i}" for i in range(10)]
    filled, remainder = fill_lane(ids, lambda i: i.endswith(("1", "3", "5")), quota=2)
    assert filled == ["p1", "p3"]
    assert remainder == ["p4", "p5", "p6", "p7", "p8", "p9"]


def test_fill_lane_returns_short_when_the_stream_is_exhausted():
    ids = ["p0", "p1"]
    filled, remainder = fill_lane(ids, lambda i: True, quota=5)
    assert filled == ids
    assert remainder == []
