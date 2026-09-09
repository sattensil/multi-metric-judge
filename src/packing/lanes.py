"""Split the fetched frame into arm-specific streams before eligibility is
inspected.

The order matters and is the whole reason this module exists. Screening one shared
frame and then carving the numbers quota out of it leaves the core pool 4.8%
numbers-eligible against a population value of 31.0%, because numbers eligibility
correlates with response content. The primary estimate would then describe
"core-eligible prompts not allocated to numbers" rather than the population the
estimand names.

Two properties the design needs that a per-prompt hash threshold does not give:

1. Exact lane sizes. A hash threshold makes lane membership an independent draw,
   so a 1,222-prompt frame yields roughly 508 core candidates rather than exactly
   508, and the spec's fill probabilities are computed for fixed lane sizes.
2. Seeded order within a lane. Filling in file order lets any structure in the
   input (collection, topic, score) select the sample. Lane membership being
   order-independent is not enough; the order the lane is consumed in has to be
   seeded too.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .partition import partition_seed


def split_lanes(
    item_ids: list[str], *, seed: int, core_candidates: int
) -> tuple[list[str], list[str]]:
    """Exact, seed-permuted, disjoint core and numbers streams.

    Both returned lists are already in the order they should be consumed, so a
    caller never needs to sort and cannot accidentally fall back to file order.
    """
    if len(set(item_ids)) != len(item_ids):
        seen: set[str] = set()
        dupes: list[str] = []
        for item_id in item_ids:
            if item_id in seen:
                dupes.append(item_id)
            seen.add(item_id)
        raise ValueError(f"duplicate base-prompt ids: {sorted(set(dupes))[:5]}")
    if core_candidates < 0:
        raise ValueError(f"core_candidates must be >= 0, got {core_candidates}")
    if core_candidates > len(item_ids):
        raise ValueError(
            f"frame of {len(item_ids)} cannot supply {core_candidates} "
            "core candidates"
        )

    # Sorted first so the permutation is a function of ids and seed alone, never
    # of the order the file happened to list them in.
    ordered = sorted(item_ids)
    random.Random(partition_seed("lane-split", 0, 0, seed)).shuffle(ordered)
    core = ordered[:core_candidates]
    numbers = ordered[core_candidates:]

    # Consumption order inside each lane is seeded separately, so extending one
    # lane cannot reshuffle the other.
    random.Random(partition_seed("core-order", 0, 0, seed)).shuffle(core)
    random.Random(partition_seed("numbers-order", 0, 0, seed)).shuffle(numbers)
    return core, numbers


def fill_lane(
    item_ids: list[str],
    eligible: Callable[[str], bool],
    quota: int,
) -> tuple[list[str], list[str]]:
    """Take the first `quota` eligible prompts in the stream's frozen order.

    Returns the filled quota and the untouched remainder, so a later variance
    review extends the sample by continuing where this stopped rather than by
    drawing afresh.
    """
    if quota < 0:
        raise ValueError(f"quota must be >= 0, got {quota}")
    filled: list[str] = []
    for index, item_id in enumerate(item_ids):
        if len(filled) >= quota:
            return filled, item_ids[index:]
        if eligible(item_id):
            filled.append(item_id)
    return filled, []
