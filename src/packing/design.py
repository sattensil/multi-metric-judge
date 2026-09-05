"""Design balance: how many observations each group size actually gets.

A naive sweep of nominal levels 1..K is badly unbalanced in the quantity that
matters. Ragged partitions mean small group sizes are produced as a by-product of
several nominal levels, while the largest group size is produced by exactly one.
For K=5 with one replicate per level, criterion-observations per group size come
out as::

    size 1: 7    size 2: 6    size 3: 3    size 4: 4    size 5: 5

so the cells with the least data are the large-group cells -- the ones the study
is actually about. Unbalanced designs are handled correctly by the mixed model,
but power is not: the standard error on the group_size=3 contrast would be twice
that of group_size=1 for no good reason.

This module makes the imbalance visible and solves for replicate counts per
nominal level that equalise it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .partition import block_sizes


def observation_matrix(k: int, levels: tuple[int, ...] | None = None) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Criterion-observations per group size, per replicate at each nominal level.

    Returns ``(A, sizes, levels)`` where ``A[i, j]`` is the number of
    criterion-observations at group size ``sizes[i]`` produced by one replicate at
    nominal level ``levels[j]`` (counting one side; both sides scale it equally).
    """
    levels = levels or tuple(range(1, k + 1))
    sizes = tuple(sorted({s for n in levels for s in block_sizes(k, n)}))
    size_index = {s: i for i, s in enumerate(sizes)}

    matrix = np.zeros((len(sizes), len(levels)))
    for j, n in enumerate(levels):
        for size, count in Counter(block_sizes(k, n)).items():
            # `count` blocks of this size, each yielding `size` observations.
            matrix[size_index[size], j] += count * size
    return matrix, sizes, levels


def observations_per_replicate(k: int, levels: tuple[int, ...] | None = None) -> dict[int, int]:
    """Baseline imbalance: observations per group size at one replicate per level."""
    matrix, sizes, _ = observation_matrix(k, levels)
    totals = matrix.sum(axis=1)
    return {size: int(total) for size, total in zip(sizes, totals)}


def suggest_replicates(
    k: int, target: int, levels: tuple[int, ...] | None = None
) -> dict[int, int]:
    """Replicates per nominal level that get each group size near ``target`` observations.

    Solved as a non-negative least-squares problem rather than by hand, so it
    generalises to any K and any subset of levels. Results are rounded up: it is
    better to overshoot a cell than to leave it underpowered, and the analysis
    handles mild imbalance.
    """
    if target < 1:
        raise ValueError(f"target must be >= 1, got {target}")
    from scipy.optimize import nnls

    matrix, _, levels = observation_matrix(k, levels)
    goal = np.full(matrix.shape[0], float(target))
    solution, _ = nnls(matrix, goal)
    return {int(n): max(1, int(np.ceil(r))) for n, r in zip(levels, solution)}


def projected_observations(
    k: int, replicates: dict[int, int], levels: tuple[int, ...] | None = None
) -> dict[int, int]:
    """Observations per group size under a given replicate allocation."""
    matrix, sizes, levels = observation_matrix(k, levels)
    vector = np.array([replicates.get(n, 0) for n in levels], dtype=float)
    totals = matrix @ vector
    return {size: int(total) for size, total in zip(sizes, totals)}


def balance_ratio(observations: dict[int, int]) -> float:
    """max/min observations across group sizes. 1.0 is perfectly balanced."""
    values = [v for v in observations.values() if v > 0]
    if not values:
        return float("inf")
    return max(values) / min(values)


def parse_replicate_spec(spec: str, levels: tuple[int, ...]) -> dict[int, int]:
    """Parse ``"1:8,2:5,3:20"`` into a replicate map, defaulting missing levels to 1."""
    out = {n: 1 for n in levels}
    if not spec.strip():
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        level_str, _, count_str = chunk.partition(":")
        if not count_str:
            raise ValueError(f"malformed replicate spec segment {chunk!r}; want 'level:count'")
        level, count = int(level_str), int(count_str)
        if level not in out:
            raise ValueError(f"level {level} not in swept levels {levels}")
        if count < 1:
            raise ValueError(f"replicate count for level {level} must be >= 1")
        out[level] = count
    return out
