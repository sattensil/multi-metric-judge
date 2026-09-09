"""The frozen sampling plan.

The dry run writes this and the live run consumes it. That ordering is what makes
the design's guarantees real rather than incidental: the response side is visibly
frozen before an operation is assigned, an extension cannot reassign a collected
unit, a reordered input file cannot select a different sample, and edited anchor
prose cannot merge into a resumed run because the manifest digest is recorded
here.

Hashes are full SHA-256. A 16-character prefix is fine for a console line and not
fine for an artefact that decides whether two runs are the same experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_STATUSES = ("active", "reserve")


@dataclass(frozen=True)
class Unit:
    base_item_id: str
    lane: str
    side: str
    eligible: bool
    operation: str
    realisation_seed: int
    companion_seed: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                f"status must be one of {_STATUSES}, got {self.status!r}"
            )
        if self.side not in ("a", "b"):
            raise ValueError(f"side must be 'a' or 'b', got {self.side!r}")


@dataclass(frozen=True)
class SamplingPlan:
    dataset_id: str
    dataset_sha256: str
    manifest_sha256: str
    seed: int
    model: str
    model_settings: dict[str, object]
    levels: tuple[int, ...]
    units: tuple[Unit, ...]
    call_identities: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        ids = [u.base_item_id for u in self.units]
        if len(set(ids)) != len(ids):
            seen: set[str] = set()
            dupes: list[str] = []
            for item_id in ids:
                if item_id in seen:
                    dupes.append(item_id)
                seen.add(item_id)
            raise ValueError(
                f"duplicate base item in the plan: {sorted(set(dupes))[:5]}. "
                "A base prompt contributes to at most one arm."
            )
        for name, digest in (
            ("dataset_sha256", self.dataset_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            if len(digest) != 64:
                raise ValueError(
                    f"{name} must be a full SHA-256, got {len(digest)} characters"
                )

    @property
    def active_count(self) -> int:
        return sum(1 for u in self.units if u.status == "active")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_plan(path: Path, plan: SamplingPlan) -> None:
    payload = asdict(plan)
    payload["levels"] = list(plan.levels)
    payload["call_identities"] = list(plan.call_identities)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_plan(path: Path) -> SamplingPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SamplingPlan(
        dataset_id=payload["dataset_id"],
        dataset_sha256=payload["dataset_sha256"],
        manifest_sha256=payload["manifest_sha256"],
        seed=payload["seed"],
        model=payload["model"],
        model_settings=payload["model_settings"],
        levels=tuple(payload["levels"]),
        units=tuple(Unit(**u) for u in payload["units"]),
        call_identities=tuple(payload["call_identities"]),
    )
