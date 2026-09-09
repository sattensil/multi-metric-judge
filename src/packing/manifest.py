"""The criterion pool as a validated, hashed artefact.

The pool is part of the experimental treatment, not test data: criterion count,
anchor completeness and prose length all determine input-token growth and what a
judge is actually asked to do. So it is validated on construction and hashed, and
the full digest is recorded with every run.

The digest covers anchor prose, not just keys. Two pools with the same criterion
names but different descriptors are different instruments, and a run must not
resume across that boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .criteria import Criterion


@dataclass(frozen=True)
class CriterionManifest:
    criteria: tuple[Criterion, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.criteria)

    @property
    def hash(self) -> str:
        """Display prefix of `full_hash`. Persist `full_hash`, show this."""
        return self.full_hash[:16]

    @property
    def full_hash(self) -> str:
        """Stable identity of the pool, ordering included.

        Ordering matters because the ladder draws companions as a prefix of a
        seeded permutation of this tuple, so two pools with the same members in a
        different order are different instruments.
        """
        payload = json.dumps(
            [
                {
                    "key": c.key,
                    "label": c.label,
                    "description": c.description,
                    "anchors": {str(k): v for k, v in sorted(c.anchors.items())},
                    "scale": [c.scale_min, c.scale_max],
                    "evaluative": c.evaluative,
                }
                for c in self.criteria
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def length_stats(self) -> dict[str, int]:
        lengths = [
            len(c.description) + sum(len(v) for v in c.anchors.values())
            for c in self.criteria
        ]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths)),
            "total": sum(lengths),
        }

    def validate(self, expect: int) -> None:
        if len(self.criteria) != expect:
            raise ValueError(
                f"expected {expect} criteria, found {len(self.criteria)}"
            )
        keys = [c.key for c in self.criteria]
        if len(set(keys)) != len(keys):
            duplicates = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate criterion keys: {duplicates}")
        # Anchor completeness is enforced by Criterion.__post_init__, so reaching
        # here means every scale point of every criterion is described.


def manifest_for(criteria: tuple[Criterion, ...]) -> CriterionManifest:
    return CriterionManifest(criteria=criteria)
