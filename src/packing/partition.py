"""Uniform random partitions of criteria into judge-call groups.

This is the load-bearing correctness module. If partitions are not uniform, or
if a criterion can appear twice or be dropped, then every downstream number is
confounded and the study is worthless.

Uniformity argument
-------------------
For a fixed sequence of block sizes, chunking a uniformly random permutation of
the K criteria into consecutive blocks yields a uniformly random partition into
blocks of those sizes. Every distinct assignment of criteria to labelled blocks
arises from exactly ``prod(size_i!)`` permutations -- the same count for all
assignments -- so all assignments are equally likely. Position within a block is
likewise uniform, because it is inherited from the permutation.

Why blocks are independent
--------------------------
Each block becomes one separate API call with no shared state, so the *order* of
blocks carries no information and needs no randomisation. That is why the ragged
block (when N does not divide K) can always be placed last without introducing a
confound: which criteria land in it is still uniformly random.

What "position" means
---------------------
``position`` is the 0-indexed ordinal of a criterion inside its own call. It is
the rival explanation for any group-size effect: accuracy may degrade with where
a criterion sits in the prompt rather than with how many criteria share the call.
A group of size 1 therefore always has position 0, by construction.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from math import ceil

from .criteria import Criterion


@dataclass(frozen=True)
class Placement:
    """Where one criterion landed in one partition."""

    criterion: Criterion
    group_index: int
    position: int
    group_size: int


@dataclass(frozen=True)
class Partition:
    """A full assignment of every criterion to exactly one judge call."""

    nominal_n: int
    groups: tuple[tuple[Criterion, ...], ...]
    seed: int
    placements: tuple[Placement, ...] = field(repr=False)

    @property
    def n_calls(self) -> int:
        return len(self.groups)

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(g) for g in self.groups)

    @property
    def hash(self) -> str:
        """Stable identity of this partition, for resumable runs.

        Derived from the criterion ordering, which is what actually varies --
        not from the seed, so two seeds that happen to produce the same
        partition share a hash and dedupe correctly.
        """
        payload = json.dumps(
            {
                "nominal_n": self.nominal_n,
                "groups": [[c.key for c in g] for g in self.groups],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def placement_for(self, criterion_key: str) -> Placement:
        for placement in self.placements:
            if placement.criterion.key == criterion_key:
                return placement
        raise KeyError(criterion_key)


def block_sizes(k: int, nominal_n: int) -> tuple[int, ...]:
    """Block sizes for K criteria packed at most ``nominal_n`` per call.

    Full blocks first, ragged remainder last. ``block_sizes(5, 2) == (2, 2, 1)``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if nominal_n < 1:
        raise ValueError(f"nominal_n must be >= 1, got {nominal_n}")
    n = min(nominal_n, k)
    full, remainder = divmod(k, n)
    sizes = [n] * full
    if remainder:
        sizes.append(remainder)
    assert sum(sizes) == k
    assert len(sizes) == ceil(k / n)
    return tuple(sizes)


def make_partition(
    criteria: tuple[Criterion, ...],
    nominal_n: int,
    seed: int,
) -> Partition:
    """Draw one uniformly random partition of ``criteria`` into blocks of <= n.

    Every criterion appears in exactly one group, exactly once.
    """
    if not criteria:
        raise ValueError("criteria must be non-empty")
    if len({c.key for c in criteria}) != len(criteria):
        raise ValueError("criteria keys must be unique")

    rng = random.Random(seed)
    shuffled = list(criteria)
    rng.shuffle(shuffled)

    sizes = block_sizes(len(criteria), nominal_n)
    groups: list[tuple[Criterion, ...]] = []
    placements: list[Placement] = []
    cursor = 0
    for group_index, size in enumerate(sizes):
        block = tuple(shuffled[cursor : cursor + size])
        cursor += size
        groups.append(block)
        for position, criterion in enumerate(block):
            placements.append(
                Placement(
                    criterion=criterion,
                    group_index=group_index,
                    position=position,
                    group_size=size,
                )
            )
    assert cursor == len(criteria), "partition did not consume every criterion"

    return Partition(
        nominal_n=nominal_n,
        groups=tuple(groups),
        seed=seed,
        placements=tuple(placements),
    )


def partition_seed(item_id: str, nominal_n: int, replicate: int, base_seed: int) -> int:
    """Deterministic per-cell seed.

    Derived by hashing rather than arithmetic so that adjacent cells do not get
    correlated RNG streams, which would quietly couple partitions across
    conditions for the same item.
    """
    key = f"{base_seed}|{item_id}|{nominal_n}|{replicate}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def nominal_levels(k: int) -> tuple[int, ...]:
    """Nominal N levels to sweep for K criteria: every value from 1 to K.

    Ragged partitions are kept deliberately -- analysis keys on *actual* group
    size, so N=2 with K=5 contributes observations at sizes 2 and 1 alike.
    """
    return tuple(range(1, k + 1))
