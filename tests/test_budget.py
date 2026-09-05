"""Pins the invariant that protects the primary estimand from a token-ceiling artifact."""

from __future__ import annotations

import pytest

from packing.budget import (
    ENVELOPE_OVERHEAD_TOKENS,
    PER_CRITERION_OUTPUT_TOKENS,
    budget_for,
    near_cap,
)

SIZES = range(1, 13)


@pytest.mark.parametrize("group_size", SIZES)
def test_per_criterion_budget_is_invariant(group_size: int) -> None:
    budget = budget_for(group_size)
    assert budget.per_criterion_tokens == PER_CRITERION_OUTPUT_TOKENS


def test_content_budget_is_exactly_linear_in_group_size() -> None:
    """The whole control: content budget must be envelope + k * per-criterion.

    Any superlinearity would hand larger groups extra slack; any sublinearity
    would squeeze them. Either direction fabricates a group-size effect.
    """
    for group_size in SIZES:
        budget = budget_for(group_size)
        expected = ENVELOPE_OVERHEAD_TOKENS + PER_CRITERION_OUTPUT_TOKENS * group_size
        assert budget.content_budget == expected


def test_marginal_budget_per_added_criterion_is_constant() -> None:
    deltas = {
        budget_for(k + 1).content_budget - budget_for(k).content_budget
        for k in range(1, 12)
    }
    assert deltas == {PER_CRITERION_OUTPUT_TOKENS}


def test_reasoning_headroom_does_not_scale_with_group_size() -> None:
    """Reasoning space is per-call scratch, not per-field output."""
    headrooms = {budget_for(k).reasoning_headroom for k in SIZES}
    assert len(headrooms) == 1


def test_reasoning_can_be_disabled() -> None:
    assert budget_for(3, reasoning=False).reasoning_headroom == 0
    assert (
        budget_for(3, reasoning=False).max_output_tokens
        == budget_for(3, reasoning=False).content_budget
    )


def test_budget_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        budget_for(0)


def test_near_cap_detection() -> None:
    budget = budget_for(2)
    assert near_cap(2, budget.content_budget)
    assert near_cap(2, int(0.95 * budget.content_budget))
    assert not near_cap(2, int(0.10 * budget.content_budget))


def test_near_cap_excludes_reasoning_tokens() -> None:
    """Regression: providers report output_tokens including reasoning, while
    content_budget excludes the reasoning headroom. Comparing them directly
    over-flagged 25% of real group_size=1 calls whose true content use was ~44%.
    """
    budget = budget_for(1)
    # Total 178 of which 64 is reasoning -> 114 content, well under 260.
    assert not near_cap(1, 178, 64)
    # Same total, no reasoning -> 178 content, still under threshold 234.
    assert not near_cap(1, 178, 0)
    # Content genuinely at the cap.
    assert near_cap(1, budget.content_budget + 64, 64)


def test_near_cap_is_monotone_in_content() -> None:
    reasoning = 50
    budget = budget_for(3)
    low = near_cap(3, reasoning + 10, reasoning)
    high = near_cap(3, reasoning + budget.content_budget, reasoning)
    assert not low
    assert high


def test_near_cap_handles_reasoning_exceeding_output() -> None:
    """Defensive: a provider reporting inconsistent counts must not go negative."""
    assert not near_cap(1, 10, 999)
