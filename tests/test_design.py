"""Design-balance tests.

Balance is not cosmetic: an unbalanced sweep puts the least data in the largest
group sizes, which are the cells the study is about.
"""

from __future__ import annotations

import pytest

from packing.design import (
    balance_ratio,
    observation_matrix,
    observations_per_replicate,
    parse_replicate_spec,
    projected_observations,
    suggest_replicates,
)
from packing.partition import block_sizes


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_observation_matrix_conserves_criteria(k: int) -> None:
    """One replicate at nominal level n must account for exactly k observations."""
    matrix, sizes, levels = observation_matrix(k)
    for j, n in enumerate(levels):
        assert matrix[:, j].sum() == k, f"level {n} lost or duplicated observations"
        assert sum(block_sizes(k, n)) == k


def test_naive_sweep_is_imbalanced_at_k5() -> None:
    """Pins the motivating problem, so a regression is visible."""
    baseline = observations_per_replicate(5)
    assert baseline == {1: 7, 2: 6, 3: 3, 4: 4, 5: 5}
    assert balance_ratio(baseline) == pytest.approx(7 / 3)


@pytest.mark.parametrize("k", [4, 5, 6])
def test_suggested_allocation_improves_balance(k: int) -> None:
    baseline = observations_per_replicate(k)
    replicates = suggest_replicates(k, 7)
    projected = projected_observations(k, replicates)
    assert balance_ratio(projected) <= balance_ratio(baseline)


def test_default_allocation_at_k5() -> None:
    """The documented default, pinned so the README and code cannot drift apart."""
    replicates = suggest_replicates(5, 7)
    assert replicates == {1: 1, 2: 1, 3: 3, 4: 2, 5: 2}
    projected = projected_observations(5, replicates)
    assert projected == {1: 8, 2: 10, 3: 9, 4: 8, 5: 10}
    assert balance_ratio(projected) == pytest.approx(1.25)


def test_every_group_size_gets_observations() -> None:
    for k in (3, 4, 5, 6):
        projected = projected_observations(k, suggest_replicates(k, 7))
        assert all(v > 0 for v in projected.values()), projected
        assert set(projected) == set(range(1, k + 1))


def test_replicates_are_at_least_one() -> None:
    """A zeroed level would silently drop a group size from the design."""
    for k in (3, 4, 5, 6):
        for target in (1, 3, 7, 20):
            assert all(v >= 1 for v in suggest_replicates(k, target).values())


def test_restricting_levels_is_respected() -> None:
    replicates = suggest_replicates(5, 7, levels=(1, 5))
    assert set(replicates) == {1, 5}
    projected = projected_observations(5, replicates, levels=(1, 5))
    assert set(projected) == {1, 5}


def test_suggest_rejects_bad_target() -> None:
    with pytest.raises(ValueError):
        suggest_replicates(5, 0)


def test_parse_replicate_spec_defaults_missing_levels() -> None:
    parsed = parse_replicate_spec("5:4", (1, 2, 3, 4, 5))
    assert parsed == {1: 1, 2: 1, 3: 1, 4: 1, 5: 4}
