"""Build ladder units: one operation, one realisation, both frozen per prompt.

Assignment is queue-based rather than shuffle-and-modulo so that it survives the
increase-only variance review. A single shuffle assigns by position in the
shuffled list, so raising the target reshuffles everything and prompts whose
outcomes are already collected can receive a different operation. Dealing into
per-operation queues once and activating a prefix means raising the target reveals
new units without changing any assignment already made.

The realisation is frozen because it is reused at every rung. Redrawn per rung,
the paired difference between K = 1 and K = 40 would carry realisation noise on
top of the load effect and the estimator could not separate them.
"""

from __future__ import annotations

import random

from .criteria import Criterion
from .dataset import _HIGH, _LOW, _NEUTRAL, PairedItem
from .partition import partition_seed
from .registry import corruption_for


def operation_queues(
    item_ids: list[str],
    op_keys: tuple[str, ...],
    *,
    seed: int,
    per_op_max: int,
) -> dict[str, list[str]]:
    """Deal the frame into one fixed queue per operation, sized to the maximum."""
    if not op_keys:
        raise ValueError("op_keys must be non-empty")
    if per_op_max < 1:
        raise ValueError(f"per_op_max must be >= 1, got {per_op_max}")
    needed = per_op_max * len(op_keys)
    if len(item_ids) < needed:
        raise ValueError(
            f"queue plan needs {needed} units, frame has {len(item_ids)}"
        )
    # Sorted first so the deal is a function of ids and seed alone, never of the
    # order the caller happened to supply.
    ordered = sorted(item_ids)
    random.Random(partition_seed("operation-queues", 0, 0, seed)).shuffle(ordered)
    return {
        op: ordered[index * per_op_max : (index + 1) * per_op_max]
        for index, op in enumerate(op_keys)
    }


def activate(queues: dict[str, list[str]], per_op: int) -> dict[str, str]:
    """Reveal the first `per_op` units of every queue.

    Monotone in `per_op` by construction: a larger value returns a superset with
    identical operations for every unit already returned. That is what the
    increase-only variance review requires.
    """
    if per_op < 0:
        raise ValueError(f"per_op must be >= 0, got {per_op}")
    for op, queue in queues.items():
        if per_op > len(queue):
            raise ValueError(
                f"per_op {per_op} exceeds queue depth {len(queue)} for {op}"
            )
    return {
        item_id: op for op, queue in queues.items() for item_id in queue[:per_op]
    }


def build_ladder_item(
    base: PairedItem,
    op_key: str,
    side: str,
    criteria: tuple[Criterion, ...],
    *,
    seed: int,
) -> PairedItem | None:
    """Apply one frozen realisation of `op_key` to one side of `base`.

    Returns None when the operation does not apply, rather than an unchanged item
    carrying a ground-truth claim, which would be a false label. The CLI treats a
    None here as an invariant violation, because eligibility was screened first.

    `meta["base_item"]` is the key that matters downstream: `analyze.py:164`
    derives the clustering column `base_unit` from it, and `independence_unit`
    falls back to `item_id` when it is absent, which would treat every corrupted
    item as its own cluster and silently inflate n.
    """
    corruption = corruption_for(op_key)
    if corruption.targeted_criterion not in {c.key for c in criteria}:
        raise ValueError(
            f"{corruption.targeted_criterion!r} not in the criterion pool"
        )
    if side not in {"a", "b"}:
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    clean = base.response_a if side == "a" else base.response_b
    realisation_seed = partition_seed(
        f"{base.item_id}:{side}:{op_key}", 0, 0, seed
    )
    result = corruption.apply(
        clean, seed=realisation_seed, severity=None, prompt=base.prompt
    )
    if not result.changed:
        return None

    # The pair is clean against corrupted of the SAME source response, so the
    # only difference between the two halves is the corruption. Pairing one
    # response against the other would make the contrast a comparison of two
    # different answers and the seeded direction would mean nothing.
    #
    # Which half holds the corrupted text is randomised from an independent seed,
    # so a judge that always prefers the first response gains nothing. The seed is
    # independent of the realisation seed for the reason recorded in
    # `dataset.build_corrupted_items`: items are dropped when the corruption does
    # not apply, so conditioning on survival skews the realisation seed
    # distribution and would skew the side choice with it.
    side_seed = partition_seed(f"lad-side:{base.item_id}:{op_key}", 0, 0, seed)
    corrupt_is_b = random.Random(side_seed).random() < 0.5
    first, second = (
        (clean, result.text) if corrupt_is_b else (result.text, clean)
    )

    # Ground truth by construction, expressed as pseudo-human labels the way the
    # synthetic stratum already does (`dataset.py:308-318`): the targeted
    # criterion is ordered in the seeded direction, every companion is neutral so
    # it contributes a tie rather than a spurious outcome.
    keys = [c.key for c in criteria]
    target = corruption.targeted_criterion
    corrupted_score = _HIGH if corruption.direction > 0 else _LOW
    clean_score = _LOW if corruption.direction > 0 else _HIGH
    labels_first = {k: _NEUTRAL for k in keys}
    labels_second = {k: _NEUTRAL for k in keys}
    if corrupt_is_b:
        labels_first[target] = clean_score
        labels_second[target] = corrupted_score
    else:
        labels_first[target] = corrupted_score
        labels_second[target] = clean_score

    return PairedItem(
        item_id=f"lad-{op_key}-{base.item_id}-{side}",
        prompt=base.prompt,
        response_a=first,
        response_b=second,
        human_a=labels_first,
        human_b=labels_second,
        dataset="synthetic",
        meta={
            **base.meta,
            "base_item": base.item_id,
            "operation": op_key,
            "direction": corruption.direction,
            "source_side": side,
            "corrupted_side": "b" if corrupt_is_b else "a",
            "n_edits": result.n_edits,
            "realisation_seed": realisation_seed,
            "provenance": result.provenance,
        },
        source="synthetic",
        targeted_criterion=target,
    )
