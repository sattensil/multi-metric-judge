"""Correctness tests for the partition sampler.

These are the most important tests in the repo. A biased partition sampler
silently confounds the primary estimand, and the resulting curve would look
perfectly plausible.
"""

from __future__ import annotations

import collections

import pytest
from scipy import stats

from packing.criteria import HELPSTEER2_CRITERIA, get_criteria
from packing.partition import (
    block_sizes,
    make_partition,
    nominal_levels,
    partition_seed,
)

K = len(HELPSTEER2_CRITERIA)


# --------------------------------------------------------------------------
# Structural invariants: exact coverage, correct sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nominal_n", range(1, K + 1))
def test_every_criterion_exactly_once(nominal_n: int) -> None:
    for seed in range(50):
        part = make_partition(HELPSTEER2_CRITERIA, nominal_n, seed=seed)
        keys = [c.key for g in part.groups for c in g]
        assert sorted(keys) == sorted(c.key for c in HELPSTEER2_CRITERIA)
        assert len(keys) == len(set(keys)), "a criterion appeared twice"


@pytest.mark.parametrize("nominal_n", range(1, K + 1))
def test_group_sizes_match_spec(nominal_n: int) -> None:
    expected = block_sizes(K, nominal_n)
    for seed in range(20):
        part = make_partition(HELPSTEER2_CRITERIA, nominal_n, seed=seed)
        assert part.group_sizes == expected
        assert part.n_calls == len(expected)


def test_placements_agree_with_groups() -> None:
    for nominal_n in nominal_levels(K):
        part = make_partition(HELPSTEER2_CRITERIA, nominal_n, seed=7)
        assert len(part.placements) == K
        for placement in part.placements:
            group = part.groups[placement.group_index]
            assert group[placement.position] is placement.criterion
            assert placement.group_size == len(group)


def test_block_sizes_examples() -> None:
    assert block_sizes(5, 1) == (1, 1, 1, 1, 1)
    assert block_sizes(5, 2) == (2, 2, 1)
    assert block_sizes(5, 3) == (3, 2)
    assert block_sizes(5, 4) == (4, 1)
    assert block_sizes(5, 5) == (5,)
    # nominal_n larger than K collapses to a single call
    assert block_sizes(5, 9) == (5,)
    assert block_sizes(4, 2) == (2, 2)


def test_block_sizes_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        block_sizes(0, 2)
    with pytest.raises(ValueError):
        block_sizes(5, 0)


def test_size_one_group_always_position_zero() -> None:
    """Structural, and worth pinning: it constrains the position covariate."""
    for nominal_n in nominal_levels(K):
        for seed in range(30):
            part = make_partition(HELPSTEER2_CRITERIA, nominal_n, seed=seed)
            for placement in part.placements:
                if placement.group_size == 1:
                    assert placement.position == 0


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_seed_reproducible() -> None:
    a = make_partition(HELPSTEER2_CRITERIA, 2, seed=1234)
    b = make_partition(HELPSTEER2_CRITERIA, 2, seed=1234)
    assert [[c.key for c in g] for g in a.groups] == [[c.key for c in g] for g in b.groups]
    assert a.hash == b.hash


def test_hash_tracks_content_not_seed() -> None:
    """Two seeds producing the same partition must share a hash, so a resumable
    run dedupes on the work actually done."""
    seen: dict[str, list[list[str]]] = {}
    for seed in range(400):
        part = make_partition(HELPSTEER2_CRITERIA, 2, seed=seed)
        layout = [[c.key for c in g] for g in part.groups]
        if part.hash in seen:
            assert seen[part.hash] == layout, "hash collision across distinct layouts"
        seen[part.hash] = layout
    assert len(seen) > 1, "hash is not discriminating between partitions"


def test_partition_seed_deterministic_and_distinct() -> None:
    assert partition_seed("item-1", 2, 0, 42) == partition_seed("item-1", 2, 0, 42)
    distinct = {
        partition_seed(item, n, rep, 42)
        for item in ("item-1", "item-2")
        for n in (1, 2, 3)
        for rep in (0, 1)
    }
    assert len(distinct) == 12, "per-cell seeds collided"


# --------------------------------------------------------------------------
# Uniformity: the tests that actually protect the estimand
# --------------------------------------------------------------------------

DRAWS = 30_000
ALPHA = 1e-3  # loose enough not to flake, tight enough to catch real bias


def test_pairing_is_uniform_k4() -> None:
    """With K=4 at N=2 there are exactly 3 unordered partitions; each must be 1/3.

    Block order is meaningless (independent API calls), so uniformity is over
    unordered partitions.
    """
    criteria = get_criteria("summeval")  # K=4
    assert len(criteria) == 4
    counts: collections.Counter = collections.Counter()
    for seed in range(DRAWS):
        part = make_partition(criteria, 2, seed=seed)
        key = frozenset(frozenset(c.key for c in g) for g in part.groups)
        counts[key] += 1
    assert len(counts) == 3, f"expected 3 distinct pairings, saw {len(counts)}"
    observed = list(counts.values())
    chi2, p = stats.chisquare(observed)
    assert p > ALPHA, f"pairings not uniform: counts={observed}, p={p:.2e}"


def test_ragged_block_membership_is_uniform() -> None:
    """K=5 at N=2 gives sizes (2,2,1). Each criterion must land in the
    singleton block with probability 1/5."""
    counts: collections.Counter = collections.Counter()
    for seed in range(DRAWS):
        part = make_partition(HELPSTEER2_CRITERIA, 2, seed=seed)
        for placement in part.placements:
            if placement.group_size == 1:
                counts[placement.criterion.key] += 1
    assert set(counts) == {c.key for c in HELPSTEER2_CRITERIA}
    chi2, p = stats.chisquare([counts[c.key] for c in HELPSTEER2_CRITERIA])
    assert p > ALPHA, f"singleton membership not uniform: {dict(counts)}, p={p:.2e}"


def test_position_is_uniform_within_group_size() -> None:
    """Each criterion must be equally likely at each position of a full block.

    This is what lets `position` be read as a covariate rather than as a proxy
    for criterion identity.
    """
    # N=5 -> one block of 5, positions 0..4
    table = {c.key: collections.Counter() for c in HELPSTEER2_CRITERIA}
    for seed in range(DRAWS):
        part = make_partition(HELPSTEER2_CRITERIA, 5, seed=seed)
        for placement in part.placements:
            table[placement.criterion.key][placement.position] += 1
    for key, counter in table.items():
        assert set(counter) == set(range(K)), f"{key} missing positions"
        chi2, p = stats.chisquare([counter[i] for i in range(K)])
        assert p > ALPHA, f"position not uniform for {key}: {dict(counter)}, p={p:.2e}"


def test_criterion_position_independence() -> None:
    """Joint test: criterion identity and position must be independent."""
    matrix = []
    table = {c.key: collections.Counter() for c in HELPSTEER2_CRITERIA}
    for seed in range(DRAWS):
        part = make_partition(HELPSTEER2_CRITERIA, 5, seed=seed)
        for placement in part.placements:
            table[placement.criterion.key][placement.position] += 1
    for c in HELPSTEER2_CRITERIA:
        matrix.append([table[c.key][i] for i in range(K)])
    chi2, p, dof, expected = stats.chi2_contingency(matrix)
    assert p > ALPHA, f"criterion and position are dependent: p={p:.2e}"
