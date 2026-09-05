"""Normalised items with per-criterion human labels.

Every substrate is reduced to the same shape so the runner and the analysis are
substrate-blind:

    Item(item_id, prompt, response_a, response_b, human={criterion_key: score})

Pairwise ground truth requires two responses to the same prompt, which is
precisely why HelpSteer2 is the primary substrate.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .criteria import Criterion, get_criteria
from .partition import partition_seed


@dataclass(frozen=True)
class PairedItem:
    """Two responses to one prompt, each with its own per-criterion human labels.

    Human labels are stored per response rather than as a delta so that the
    absolute-score outcomes (QWK, mean absolute error) remain available for any
    criterion whose tie rate makes the pairwise outcome unusable.
    """

    item_id: str
    prompt: str
    response_a: str
    response_b: str
    human_a: dict[str, int]
    human_b: dict[str, int]
    dataset: str
    meta: dict[str, Any]
    source: str = "human"
    """`human` (annotator labels) or `synthetic` (corruption by construction)."""
    targeted_criterion: str | None = None
    """For synthetic items: the one criterion whose ordering is known. The others
    were not modified, so their expected ordering is 'unchanged' -- which is a
    specificity observation, not an accuracy observation."""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(items: list[PairedItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item.as_row(), separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[PairedItem]:
    items: list[PairedItem] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(PairedItem(**json.loads(line)))
    return items


def validate(items: list[PairedItem], criteria: tuple[Criterion, ...]) -> None:
    """Fail loudly on malformed labels rather than analysing garbage."""
    keys = {c.key for c in criteria}
    by_key = {c.key: c for c in criteria}
    for item in items:
        for side, labels in (("a", item.human_a), ("b", item.human_b)):
            missing = keys - set(labels)
            if missing:
                raise ValueError(
                    f"{item.item_id} response_{side}: missing labels {sorted(missing)}"
                )
            for key in keys:
                criterion = by_key[key]
                value = labels[key]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        f"{item.item_id} response_{side} {key}: non-integer {value!r}"
                    )
                if not criterion.scale_min <= value <= criterion.scale_max:
                    raise ValueError(
                        f"{item.item_id} response_{side} {key}: {value} outside "
                        f"[{criterion.scale_min}, {criterion.scale_max}]"
                    )


# ------------------------------------------------------------- HelpSteer2 ----

HELPSTEER2_REPO = "nvidia/HelpSteer2"


def fetch_helpsteer2(limit: int | None = None, split: str = "train") -> list[PairedItem]:
    """Load HelpSteer2 and pair the two responses per prompt.

    HelpSteer2 stores one row per (prompt, response) with the five attribute
    scores attached. Two rows share each prompt, so pairing is by prompt text.
    Any prompt that does not have exactly two responses is dropped -- silently
    admitting a 1- or 3-response group would break the pairwise outcome.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("`datasets` not installed") from exc

    criteria = get_criteria("helpsteer2")
    keys = [c.key for c in criteria]

    rows = load_dataset(HELPSTEER2_REPO, split=split)

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        prompt = row["prompt"]
        if prompt not in grouped:
            grouped[prompt] = []
            order.append(prompt)
        grouped[prompt].append(row)
        if limit is not None and len(order) > limit and len(grouped[order[-1]]) >= 2:
            break

    items: list[PairedItem] = []
    dropped = 0
    for index, prompt in enumerate(order):
        bucket = grouped[prompt]
        if len(bucket) != 2:
            dropped += 1
            continue
        first, second = bucket
        if any(k not in first or k not in second for k in keys):
            dropped += 1
            continue
        items.append(
            PairedItem(
                item_id=f"hs2-{index:06d}",
                prompt=prompt,
                response_a=first["response"],
                response_b=second["response"],
                human_a={k: int(first[k]) for k in keys},
                human_b={k: int(second[k]) for k in keys},
                dataset="helpsteer2",
                meta={"dropped_siblings": 0},
            )
        )
        if limit is not None and len(items) >= limit:
            break

    if dropped:
        # Reported, never silent: a systematic drop pattern would bias the sample.
        items = [
            PairedItem(**{**i.as_row(), "meta": {**i.meta, "dropped_prompts": dropped}})
            for i in items
        ]
    validate(items, criteria)
    return items


def fetch(dataset: str, limit: int | None = None) -> list[PairedItem]:
    if dataset == "helpsteer2":
        return fetch_helpsteer2(limit=limit)
    if dataset == "summeval":
        raise NotImplementedError(
            "SummEval replication (gate 5) needs a conversion step: the released "
            "annotations are per (system, article) with 8 annotators each, and a "
            "pairwise substrate requires choosing two systems per article. "
            "Convert to the PairedItem JSONL shape and load with --from-jsonl. "
            "Source: https://github.com/Yale-LILY/SummEval"
        )
    raise ValueError(f"unknown dataset {dataset!r}")


def scoreable_count(item: PairedItem, criteria: tuple[Criterion, ...]) -> int:
    """How many criteria have a non-tied human ordering for this item."""
    return sum(
        1 for c in criteria if item.human_a[c.key] != item.human_b[c.key]
    )


def filter_min_scoreable(
    items: list[PairedItem], criteria: tuple[Criterion, ...], minimum: int
) -> list[PairedItem]:
    """Keep items with at least ``minimum`` non-tied criteria.

    Why this is legitimate, and what it costs:

    The filter is a property of the *human labels* only. It is applied before any
    judging, and it is orthogonal to group size -- every surviving item is still
    judged at every group size. So it cannot bias the group_size contrast, which
    is the estimand.

    What it does change is the population. Items where the two responses differ
    clearly are probably easier to judge, so absolute accuracy will read higher
    than it would on the full distribution. That is a generalisation limit and
    must be reported as one: the curve's *shape* transfers, its *level* does not.

    Measured on HelpSteer2 train (10,160 valid pairs): 54.7% of items have >= 3
    scoreable criteria, 27.0% have >= 4, and only 7.3% have all 5. Without a
    filter, 10.3% of items contribute nothing at all to the pairwise outcome and
    the mean item carries just 2.1 usable criteria, which makes per-item accuracy
    -- the unit the paired estimator operates on -- very noisy.
    """
    if minimum <= 0:
        return list(items)
    return [i for i in items if scoreable_count(i, criteria) >= minimum]


def scoreable_report(
    items: list[PairedItem], criteria: tuple[Criterion, ...]
) -> dict[str, Any]:
    """Gate 2 evidence: tie rates per criterion and the per-item distribution."""
    n = len(items)
    if n == 0:
        return {"n_items": 0}
    per_criterion = {}
    for criterion in criteria:
        ties = sum(
            1 for i in items if i.human_a[criterion.key] == i.human_b[criterion.key]
        )
        per_criterion[criterion.key] = {
            "human_tie_rate": ties / n,
            "usable_n": n - ties,
        }
    counts = [scoreable_count(i, criteria) for i in items]
    distribution = {
        c: sum(1 for x in counts if x == c) for c in range(len(criteria) + 1)
    }
    return {
        "n_items": n,
        "per_criterion": per_criterion,
        "scoreable_per_item_distribution": distribution,
        "mean_scoreable_per_item": sum(counts) / n,
        "items_with_zero_scoreable": distribution.get(0, 0),
    }


# ------------------------------------------------------------- synthetic -----

# Pseudo-labels encoding the known ordering. Values are arbitrary; only their
# ORDER carries meaning, which is all the outcome machinery consumes. Equal values
# on non-targeted criteria mark them as having no expected ordering.
_HIGH, _LOW, _NEUTRAL = 3, 1, 2


def build_corrupted_items(
    bases: list[PairedItem],
    corruption,
    criteria: tuple[Criterion, ...],
    *,
    limit: int | None = None,
    base_seed: int = 20260827,
    severity: int | None = None,
) -> list[PairedItem]:
    """Build (clean, corrupted) items with operation-level ground truth by construction.

    One corruption per item. Gold is directional recovery of that operation on
    its targeted criterion, not criterion-isolated human-quality accuracy.
    Untouched companions are marked neutral and feed the specificity / leakage
    diagnostics rather than the recovery outcome.

    Each base response is used independently, so a HelpSteer2 pair contributes two
    candidate bases. Items whose corruption does not apply are dropped, never
    emitted unchanged -- an unmodified variant carrying a ground-truth claim would
    be a false label.

    Which side receives the corruption is randomised per item, so a judge that
    always prefers the first response gains nothing.
    """
    keys = [c.key for c in criteria]
    target = corruption.targeted_criterion
    if target not in keys:
        raise ValueError(f"{target!r} not in criteria {keys}")

    sev = corruption.default_severity if severity is None else severity
    out: list[PairedItem] = []
    dropped = 0
    for base in bases:
        for side in ("response_a", "response_b"):
            if limit is not None and len(out) >= limit:
                break
            clean = getattr(base, side)
            seed = partition_seed(f"{base.item_id}:{side}", 0, 0, base_seed)
            result = corruption.apply(
                clean, seed=seed, severity=severity, prompt=base.prompt
            )
            if not result.changed:
                dropped += 1
                continue

            # Derived from an INDEPENDENT seed, not from `seed`. Reusing the
            # corruption seed here induced a measurable side bias (51.4/48.6,
            # binomial p = 0.019): items are dropped when the corruption does not
            # apply, whether it applies depends on the seed, so conditioning on
            # survival skewed the seed distribution and with it the side choice.
            side_seed = partition_seed(f"side:{base.item_id}:{side}", 0, 0, base_seed)
            corrupt_is_b = random.Random(side_seed).random() < 0.5
            first, second = (
                (clean, result.text) if corrupt_is_b else (result.text, clean)
            )

            # Direction is expressed relative to the CORRUPTED text.
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

            out.append(
                PairedItem(
                    item_id=(
                        f"syn-{corruption.key}-s{sev}-{base.item_id}-{side[-1]}"
                    ),
                    prompt=base.prompt,
                    response_a=first,
                    response_b=second,
                    human_a=labels_first,
                    human_b=labels_second,
                    dataset="synthetic",
                    meta={
                        "corruption": corruption.key,
                        "severity": sev,
                        "direction": corruption.direction,
                        "corrupted_side": "b" if corrupt_is_b else "a",
                        "n_edits": result.n_edits,
                        "base_item": base.item_id,
                        "base_response_side": side[-1],
                        "provenance": result.provenance,
                        "dropped_inapplicable": dropped,
                    },
                    source="synthetic",
                    targeted_criterion=target,
                )
            )
    return out
