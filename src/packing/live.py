"""Prerequisites for a paid run: integrity gates, replacement, archival.

The dry run writes a sampling plan; this module is how a live run consumes it.
Everything here exists to make the design's guarantees hold at execution time
rather than only at planning time.

The gates refuse rather than warn. A changed items file or an edited criterion
manifest means the plan describes a different experiment, and continuing would
mix two instruments in one dataset with no way to separate them afterwards.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import PairedItem, read_jsonl
from .ladder_criteria import LADDER_CRITERIA, LADDER_MANIFEST
from .ladder_dataset import build_ladder_item
from .runner import CallSpec, completed_identities, plan_fixed_position
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


def verify_run_dir(run_dir: Path, plan: SamplingPlan) -> None:
    """Refuse to append a different experiment into an existing run directory.

    `items.jsonl` and `meta.json` are rewritten on every pass while
    `calls.jsonl` is appended and deduplicated by call identity. That
    combination is deliberate, because it is what lets a later rung extend an
    earlier pass without repaying for it, and it is also how two different
    plans could end up sharing one dataset: the surviving records would carry
    identities the current items file no longer describes, and no downstream
    step could separate them.

    `levels` is allowed to grow, since extending the ladder is the intended
    use. Everything that defines the instrument is immutable.
    """
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    if not (run_dir / "calls.jsonl").exists():
        return

    import json

    previous = json.loads(meta_path.read_text(encoding="utf-8"))
    frozen = {
        "model": plan.model,
        "seed": plan.seed,
        "manifest_sha256": plan.manifest_sha256,
        "dataset_sha256": plan.dataset_sha256,
    }
    for field, expected in frozen.items():
        found = previous.get(field)
        if found is not None and found != expected:
            raise IntegrityError(
                f"{run_dir} already holds calls from a different plan: "
                f"{field} was {str(found)[:16]}, this plan has "
                f"{str(expected)[:16]}. Appending would mix two instruments in "
                "one dataset. Use a new --name."
            )

    old_levels = previous.get("levels")
    if old_levels is not None:
        dropped = sorted(set(old_levels) - set(plan.levels))
        if dropped:
            raise IntegrityError(
                f"{run_dir} holds calls at K = {dropped}, which this plan does "
                "not include. The retained records would describe rungs the "
                "items file no longer covers. Use a new --name."
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


# ------------------------------------------------------------ replacement ---


@dataclass(frozen=True)
class Substitution:
    """One active unit swapped for a reserve from its own lane.

    Recorded per replacement rather than inferred, because the draw depends on
    which reserves were already consumed. Reading the log back is what makes a
    resumed run pick up where the previous one stopped instead of handing the
    same reserve out twice.
    """

    replaced_base_item_id: str
    replacement_base_item_id: str
    lane: str
    operation: str
    side: str
    unresolved_identities: tuple[str, ...]


def write_substitution(path: Path, substitution: Substitution) -> None:
    """Append one substitution. Append-only, like the call log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(substitution), separators=(",", ":")) + "\n")
        handle.flush()


def read_substitutions(path: Path) -> tuple[Substitution, ...]:
    if not path.exists():
        return ()
    out: list[Substitution] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row["unresolved_identities"] = tuple(row.get("unresolved_identities", ()))
        out.append(Substitution(**row))
    return tuple(out)


def unresolved_by_unit(
    specs: list[CallSpec], calls_path: Path
) -> dict[str, tuple[str, ...]]:
    """Base prompts whose planned calls did not all close, and which are open.

    "Closed" is `completed_identities`, so a success or a scored treatment
    failure counts and a transient one does not. That is the distinction
    replacement turns on: a truncated or unparseable response is data and is
    kept, while a call that never reached the judge is a hole.
    """
    closed = completed_identities(calls_path)
    open_by_base: dict[str, list[str]] = {}
    for spec in specs:
        if spec.identity in closed:
            continue
        base = spec.item.meta.get("base_item", spec.item.item_id)
        open_by_base.setdefault(base, []).append(spec.identity)
    return {base: tuple(ids) for base, ids in open_by_base.items()}


def plan_replacements(
    plan: SamplingPlan,
    items_path: Path,
    calls_path: Path,
    *,
    specs: list[CallSpec],
    existing: tuple[Substitution, ...] = (),
) -> list[tuple[Substitution, PairedItem, list[CallSpec]]]:
    """Draw a same-lane reserve for every unit whose ladder is incomplete.

    The replacement inherits the replaced unit's operation and response side,
    because the quotas are per operation: substituting a different one would
    quietly re-weight the estimand that the equal-weight estimator assumes.

    It is planned across every rung, not only the rung that failed.
    `analyze.stratified_endpoint` pairs within a cluster and drops any cluster
    missing an endpoint, so a replacement covering one rung would cost a call
    and contribute nothing to the primary contrast.
    """
    verify_plan(plan, items_path)
    by_id = {b.item_id: b for b in read_jsonl(items_path)}
    units_by_id = {u.base_item_id: u for u in plan.units}

    already = {s.replaced_base_item_id for s in existing}
    used = {s.replacement_base_item_id for s in existing}

    unresolved = unresolved_by_unit(specs, calls_path)
    out: list[tuple[Substitution, PairedItem, list[CallSpec]]] = []

    # Plan order, not dictionary order, so two runs replacing the same units
    # draw the same reserves.
    for unit in plan.units:
        if unit.status != "active" or unit.base_item_id in already:
            continue
        identities = unresolved.get(unit.base_item_id)
        if not identities:
            continue

        # A reserve can itself fail to build, since a corruption need not apply
        # to a given response. Keep drawing rather than dropping the unit.
        while True:
            reserve = next_replacement(plan, lane=unit.lane, used=used)
            used.add(reserve.base_item_id)
            base = by_id.get(reserve.base_item_id)
            if base is None:
                raise IntegrityError(
                    f"{reserve.base_item_id} is a reserve in the plan but not "
                    f"in {items_path}"
                )
            built = build_ladder_item(
                base, unit.operation, unit.side, LADDER_CRITERIA, seed=plan.seed
            )
            if built is not None:
                break

        new_specs = plan_fixed_position(
            [built],
            LADDER_CRITERIA,
            model=plan.model,
            levels=plan.levels,
            base_seed=plan.seed,
        )
        out.append(
            (
                Substitution(
                    replaced_base_item_id=unit.base_item_id,
                    replacement_base_item_id=reserve.base_item_id,
                    lane=unit.lane,
                    operation=unit.operation,
                    side=unit.side,
                    unresolved_identities=identities,
                ),
                built,
                new_specs,
            )
        )
    return out


def rebuild_with_replacements(
    plan: SamplingPlan,
    items_path: Path,
    substitutions: tuple[Substitution, ...],
) -> tuple[list[PairedItem], list[CallSpec]]:
    """The frozen plan plus every recorded replacement.

    `rebuild_items` deliberately reproduces only what the plan froze, and its
    identity check is what proves the reconstruction did not drift. Replacements
    are additive and were drawn after the freeze, so they are layered on here
    rather than folded into that check.
    """
    items, specs = rebuild_items(plan, items_path)
    by_id = {b.item_id: b for b in read_jsonl(items_path)}

    for substitution in substitutions:
        base = by_id.get(substitution.replacement_base_item_id)
        if base is None:
            raise IntegrityError(
                f"{substitution.replacement_base_item_id} was recorded as a "
                f"replacement but is not in {items_path}"
            )
        built = build_ladder_item(
            base,
            substitution.operation,
            substitution.side,
            LADDER_CRITERIA,
            seed=plan.seed,
        )
        if built is None:
            raise IntegrityError(
                f"{substitution.replacement_base_item_id} no longer builds "
                f"under {substitution.operation!r}; the catalogue has changed"
            )
        items.append(built)
        specs.extend(
            plan_fixed_position(
                [built],
                LADDER_CRITERIA,
                model=plan.model,
                levels=plan.levels,
                base_seed=plan.seed,
            )
        )
    return items, specs


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
