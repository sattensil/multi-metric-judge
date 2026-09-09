"""Prerequisites for a paid run: integrity gates, replacement, archival.

The dry run writes a sampling plan; this module is how a live run consumes it.
Everything here exists to make the design's guarantees hold at execution time
rather than only at planning time.

The gates refuse rather than warn. A changed items file or an edited criterion
manifest means the plan describes a different experiment, and continuing would
mix two instruments in one dataset with no way to separate them afterwards.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .dataset import PairedItem, read_jsonl
from .ladder_criteria import LADDER_CRITERIA, LADDER_MANIFEST
from .ladder_dataset import build_ladder_item
from .runner import CallSpec, plan_fixed_position
from .sampling_plan import SamplingPlan, Unit, file_sha256


class IntegrityError(RuntimeError):
    """The plan and the world disagree. Stop rather than mix instruments."""


def verify_plan(plan: SamplingPlan, items_path: Path) -> None:
    """Refuse to run a plan whose inputs have moved under it.

    Two digests, two distinct failure modes. A changed dataset means different
    prompts, so the sample is no longer the one that was screened. A changed
    manifest means different anchor prose, which is part of the treatment: the
    criterion keys can be identical while the instrument is not.
    """
    actual_dataset = file_sha256(items_path)
    if actual_dataset != plan.dataset_sha256:
        raise IntegrityError(
            f"dataset digest mismatch for {items_path}: plan expects "
            f"{plan.dataset_sha256[:16]}, found {actual_dataset[:16]}. The frame "
            "changed, so this plan describes a different sample."
        )
    if LADDER_MANIFEST.full_hash != plan.manifest_sha256:
        raise IntegrityError(
            f"manifest digest mismatch: plan expects "
            f"{plan.manifest_sha256[:16]}, code has "
            f"{LADDER_MANIFEST.hash}. The criterion pool changed, and anchor "
            "prose is part of the treatment."
        )


def rebuild_items(
    plan: SamplingPlan, items_path: Path
) -> tuple[list[PairedItem], list[CallSpec]]:
    """Reconstruct the exact items and calls the plan froze.

    Nothing is re-screened and nothing is re-assigned. Every choice comes from
    the plan: which prompt, which side, which operation, which seed. That is what
    makes the artifact authoritative rather than decorative.

    The rebuilt call identities are checked against the frozen list, so a
    reconstruction that drifts for any reason fails here rather than halfway
    through a paid run.
    """
    verify_plan(plan, items_path)
    by_id = {b.item_id: b for b in read_jsonl(items_path)}

    items: list[PairedItem] = []
    for unit in plan.units:
        if unit.status != "active":
            continue
        base = by_id.get(unit.base_item_id)
        if base is None:
            raise IntegrityError(
                f"{unit.base_item_id} is in the plan but not in {items_path}"
            )
        built = build_ladder_item(
            base, unit.operation, unit.side, LADDER_CRITERIA, seed=plan.seed
        )
        if built is None:
            raise IntegrityError(
                f"{unit.base_item_id}/{unit.operation}/{unit.side} was planned "
                "but no longer builds; the corruption catalogue has changed"
            )
        if built.meta["realisation_seed"] != unit.realisation_seed:
            raise IntegrityError(
                f"{unit.base_item_id}: realisation seed drifted from "
                f"{unit.realisation_seed} to {built.meta['realisation_seed']}"
            )
        items.append(built)

    specs = plan_fixed_position(
        items,
        LADDER_CRITERIA,
        model=plan.model,
        levels=plan.levels,
        base_seed=plan.seed,
    )
    rebuilt = tuple(s.identity for s in specs)
    if plan.call_identities and rebuilt != plan.call_identities:
        raise IntegrityError(
            f"rebuilt {len(rebuilt)} call identities do not match the "
            f"{len(plan.call_identities)} frozen in the plan"
        )
    return items, specs


def next_replacement(
    plan: SamplingPlan, *, lane: str, used: set[str]
) -> Unit:
    """The next unused reserve in `lane`, in the plan's frozen order.

    Replacement is only legitimate for a transient infrastructure failure. A
    treatment failure is data and is scored, so replacing it would condition the
    analysis on surviving high load.
    """
    for unit in plan.units:
        if unit.status != "reserve" or unit.lane != lane:
            continue
        if unit.base_item_id in used:
            continue
        return unit
    raise IntegrityError(
        f"no reserve left in lane {lane!r}; fetch a larger frame and re-plan "
        "rather than shrinking the sample"
    )


def archive_run(run_dir: Path, archive_root: Path) -> Path:
    """Copy a run directory somewhere `.gitignore` does not cover.

    `runs/*` is ignored, and the K = 20 sweep was nearly lost to exactly that.
    Archival is a precondition for analysis, not a courtesy afterwards.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise IntegrityError(f"{run_dir} is not a directory")
    destination = Path(archive_root) / run_dir.name
    if destination.exists():
        raise IntegrityError(
            f"{destination} already exists; archives are never overwritten "
            "because that is how a run gets silently replaced"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, destination)
    return destination
