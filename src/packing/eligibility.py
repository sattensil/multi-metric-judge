"""Which arm a base prompt can serve, decided from the source text alone.

Eligibility is whether the mechanical edit changes the text at all. An operation
that cannot apply returns `changed=False` and would be dropped downstream, so
screening for it up front is what lets an arm have a defined population.

Nothing here touches an outcome, a score, or a provider. That ordering is the
point: arm assignment happens before eligibility is inspected (spec section 5),
and eligibility is inspected before an operation is assigned.

The returned side must be persisted by the caller. It is the side that satisfied
the *whole* operation set. Calling `select_response` again later with a single
assigned operation can return the other side, which is eligible for that
operation but not for all eight, and that silently changes the population from
`A_core = 1`.
"""

from __future__ import annotations

import random

from .dataset import PairedItem
from .partition import partition_seed
from .registry import corruption_for

CORE_OPERATIONS: tuple[str, ...] = (
    "flatten",
    "truncate",
    "shuffle",
    "drop_answer",
    "agreement",
    "depunct",
    "duplicate",
    "filler",
)

NUMBERS_OPERATION = "numbers"


def applies(op_key: str, text: str, prompt: str, *, seed: int) -> bool:
    """Does this operation change this text?"""
    corruption = corruption_for(op_key)
    result = corruption.apply(text, seed=seed, severity=None, prompt=prompt)
    return bool(result.changed)


def select_response(
    item: PairedItem, op_keys: tuple[str, ...], *, seed: int
) -> str | None:
    """Pick the side this prompt will contribute, or None if neither qualifies.

    A side qualifies when *every* operation in `op_keys` applies to it. Where both
    sides qualify one is chosen uniformly at random from `seed`, so the choice is
    recorded rather than left to whichever side happened to be tested first.
    """
    if not op_keys:
        raise ValueError("op_keys must be non-empty")
    # Look each key up before testing, so an unknown operation fails loudly
    # rather than quietly making every prompt ineligible.
    for op in op_keys:
        corruption_for(op)

    qualifying = [
        side
        for side in ("a", "b")
        if all(
            applies(
                op,
                item.response_a if side == "a" else item.response_b,
                item.prompt,
                seed=partition_seed(f"{item.item_id}:{side}:{op}", 0, 0, seed),
            )
            for op in op_keys
        )
    ]
    if not qualifying:
        return None
    if len(qualifying) == 1:
        return qualifying[0]
    rng = random.Random(partition_seed(f"{item.item_id}:side", 0, 0, seed))
    return rng.choice(qualifying)
