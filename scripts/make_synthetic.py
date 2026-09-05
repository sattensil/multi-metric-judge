"""Generate synthetic paired items for offline pipeline checks.

Human labels are constructed with a strict ordering on every criterion, so the
pairwise outcome is always defined and a constant-guessing judge scores 0.5.
Which side wins is randomised, so a judge that always prefers response A gains
nothing.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from packing.criteria import get_criteria
from packing.dataset import PairedItem, write_jsonl


def make(n: int, seed: int, dataset: str) -> list[PairedItem]:
    criteria = get_criteria(dataset)
    rng = random.Random(seed)
    items = []
    for i in range(n):
        human_a: dict[str, int] = {}
        human_b: dict[str, int] = {}
        for criterion in criteria:
            high, low = rng.sample(criterion.scale_points, 2)
            if high < low:
                high, low = low, high
            if rng.random() < 0.5:
                human_a[criterion.key], human_b[criterion.key] = high, low
            else:
                human_a[criterion.key], human_b[criterion.key] = low, high
        items.append(
            PairedItem(
                item_id=f"syn-{i:04d}",
                prompt=f"Synthetic prompt number {i}. Explain something.",
                response_a=f"Response A for item {i}. " * 5,
                response_b=f"Response B for item {i}. " * 5,
                human_a=human_a,
                human_b=human_b,
                dataset="synthetic",
                meta={"synthetic": True, "seed": seed},
            )
        )
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--dataset", default="helpsteer2")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic.jsonl"))
    args = parser.parse_args()

    items = make(args.n, args.seed, args.dataset)
    write_jsonl(items, args.out)
    print(f"wrote {len(items)} synthetic paired items to {args.out}")
