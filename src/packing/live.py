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
import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import PairedItem, read_jsonl
from .ladder_criteria import LADDER_CRITERIA, LADDER_MANIFEST
from .ladder_dataset import build_ladder_item
from .runner import CallSpec, completed_identities, estimate, plan_fixed_position
from .sampling_plan import SamplingPlan, Unit, file_sha256


class IntegrityError(RuntimeError):
    """The plan and the world disagree. Stop rather than mix instruments."""


def sample_fingerprint(plan: SamplingPlan) -> str:
    """Hash the immutable sample definition, excluding extensible rung levels."""
    payload = {
        "dataset_id": plan.dataset_id,
        "dataset_sha256": plan.dataset_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "seed": plan.seed,
        "model": plan.model,
        "model_settings": plan.model_settings,
        "units": [asdict(unit) for unit in plan.units],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def active_unit_manifest(plan: SamplingPlan) -> list[dict[str, str]]:
    """Small metadata record used to prove that an extension is monotone."""
    return [
        {
            "base_item_id": unit.base_item_id,
            "lane": unit.lane,
            "side": unit.side,
            "operation": unit.operation,
        }
        for unit in plan.units
        if unit.status == "active"
    ]


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
        "model_settings": plan.model_settings,
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

    expected_sample = sample_fingerprint(plan)
    previous_sample = previous.get("sample_sha256")
    if previous_sample is None:
        raise IntegrityError(
            f"{run_dir} already holds calls but its metadata has no sample "
            "fingerprint, so compatibility cannot be proved. Use a new --name."
        )
    if previous_sample != expected_sample:
        previous_active = previous.get("active_units")
        if not isinstance(previous_active, list):
            raise IntegrityError(
                f"{run_dir} already holds calls from a different sample "
                f"({previous_sample[:16]} vs {expected_sample[:16]}) and has "
                "no active-unit manifest proving a monotone extension. "
                "Use a new --name."
            )
        current = {
            entry["base_item_id"]: entry for entry in active_unit_manifest(plan)
        }
        for entry in previous_active:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("base_item_id"), str)
                or current.get(entry["base_item_id"]) != entry
            ):
                raise IntegrityError(
                    f"{run_dir} already holds calls from a different sample; "
                    "the new plan removes or reassigns a previously active "
                    "unit. Use a new --name."
                )
        replacement_path = run_dir / "replacements.jsonl"
        if replacement_path.exists():
            used_replacements = {
                row.replacement_base_item_id
                for row in read_substitutions(replacement_path)
            }
            newly_active = {
                row["base_item_id"] for row in active_unit_manifest(plan)
            }
            conflict = sorted(used_replacements & newly_active)
            if conflict:
                raise IntegrityError(
                    f"{run_dir} cannot extend this sample because replacement "
                    f"reserve(s) became ordinary active units: {conflict[:5]}. "
                    "Generate an extension that skips consumed reserves."
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


@dataclass(frozen=True)
class LiveUnit:
    """A unit currently in the sample, whether original or a substitute."""

    base_item_id: str
    lane: str
    operation: str
    side: str


def current_units(
    plan: SamplingPlan, substitutions: tuple[Substitution, ...]
) -> tuple[LiveUnit, ...]:
    """The sample as it stands: actives, minus the replaced, plus substitutes.

    Substitution replaces rather than adds. A replaced unit that still holds
    both endpoints would otherwise keep contributing a per-unit difference
    while its substitute contributed another, putting the operation over the
    quota the equal-weight estimator assumes (§6.1). Retiring it also keeps
    resume from re-issuing calls for a unit that is no longer in the sample.

    A substitute is itself a unit and can itself be replaced, so the chain is
    followed rather than assumed to be one deep.
    """
    superseded = {s.replaced_base_item_id for s in substitutions}
    sides = {u.base_item_id: u.side for u in plan.units}
    lanes = {u.base_item_id: u.lane for u in plan.units}

    live: list[LiveUnit] = [
        LiveUnit(u.base_item_id, u.lane, u.operation, u.side)
        for u in plan.units
        if u.status == "active" and u.base_item_id not in superseded
    ]
    for substitution in substitutions:
        if substitution.replacement_base_item_id in superseded:
            continue
        live.append(
            LiveUnit(
                base_item_id=substitution.replacement_base_item_id,
                lane=lanes.get(
                    substitution.replacement_base_item_id, substitution.lane
                ),
                operation=substitution.operation,
                # The reserve's own screened side, not the replaced unit's.
                # Eligibility is a property of a prompt and a side together.
                side=sides.get(
                    substitution.replacement_base_item_id, substitution.side
                ),
            )
        )
    return tuple(live)


def _build_unit_item(
    unit: LiveUnit, by_id: dict[str, PairedItem], plan: SamplingPlan
) -> PairedItem | None:
    base = by_id.get(unit.base_item_id)
    if base is None:
        raise IntegrityError(
            f"{unit.base_item_id} is in the sample but not in the frame"
        )
    return build_ladder_item(
        base, unit.operation, unit.side, LADDER_CRITERIA, seed=plan.seed
    )


def estimate_replacement_round(
    plan: SamplingPlan,
    items_path: Path,
    substitutions: tuple[Substitution, ...],
) -> dict[str, int]:
    """Bound one replacement round using the largest eligible reserve prompts.

    A replacement inherits the failed unit's operation but uses the reserve's
    prompt and response. Those texts can be longer than the active sample, so
    repricing the active calls is not a valid authorization bound.
    """
    verify_plan(plan, items_path)
    by_id = {item.item_id: item for item in read_jsonl(items_path)}
    used = {sub.replacement_base_item_id for sub in substitutions}
    reserves = [
        unit
        for unit in plan.units
        if unit.status == "reserve" and unit.base_item_id not in used
    ]
    live = current_units(plan, substitutions)

    fields = (
        "calls",
        "est_input_tokens",
        "est_input_tokens_ceiling",
        "est_output_tokens_ceiling",
    )
    largest: dict[tuple[str, str], dict[str, int]] = {}
    for lane, operation in {(unit.lane, unit.operation) for unit in live}:
        bound = {field: 0 for field in fields}
        for reserve in reserves:
            if reserve.lane != lane:
                continue
            candidate = _build_unit_item(
                LiveUnit(
                    reserve.base_item_id,
                    lane,
                    operation,
                    reserve.side,
                ),
                by_id,
                plan,
            )
            if candidate is None:
                continue
            candidate_estimate = estimate(
                plan_fixed_position(
                    [candidate],
                    LADDER_CRITERIA,
                    model=plan.model,
                    levels=plan.levels,
                    base_seed=plan.seed,
                )
            )
            for field in fields:
                bound[field] = max(bound[field], candidate_estimate[field])
        largest[(lane, operation)] = bound

    total = {field: 0 for field in fields}
    for unit in live:
        bound = largest[(unit.lane, unit.operation)]
        for field in fields:
            total[field] += bound[field]
    return total


def plan_replacements(
    plan: SamplingPlan,
    items_path: Path,
    calls_path: Path,
    *,
    specs: list[CallSpec],
    existing: tuple[Substitution, ...] = (),
) -> list[tuple[Substitution, PairedItem, list[CallSpec]]]:
    """Draw a same-lane reserve for every unit whose ladder is incomplete.

    The substitute inherits the replaced unit's operation, because the quotas
    are per operation, but carries its *own* screened response side, because
    eligibility was decided for that prompt and that side together.

    It is planned across every rung, not only the rung that failed.
    `analyze.stratified_endpoint` pairs within a unit across the two endpoints
    and drops any unit missing one, so a substitute covering one rung would
    cost a call and contribute nothing to the primary contrast.
    """
    verify_plan(plan, items_path)
    by_id = {b.item_id: b for b in read_jsonl(items_path)}

    used = {s.replacement_base_item_id for s in existing}
    reserves_by_id = {u.base_item_id: u for u in plan.units}
    unresolved = unresolved_by_unit(specs, calls_path)
    out: list[tuple[Substitution, PairedItem, list[CallSpec]]] = []

    # Deterministic order, so two runs replacing the same units draw the same
    # reserves. Substitutes sort after the actives they stand in for.
    for unit in current_units(plan, existing):
        identities = unresolved.get(unit.base_item_id)
        if not identities:
            continue

        # A reserve can itself fail to build, since a corruption need not apply
        # to a given response. Keep drawing rather than dropping the unit.
        while True:
            reserve = next_replacement(plan, lane=unit.lane, used=used)
            used.add(reserve.base_item_id)
            candidate = LiveUnit(
                base_item_id=reserve.base_item_id,
                lane=reserve.lane,
                operation=unit.operation,
                side=reserve.side,
            )
            built = _build_unit_item(candidate, by_id, plan)
            if built is not None:
                break

        out.append(
            (
                Substitution(
                    replaced_base_item_id=unit.base_item_id,
                    replacement_base_item_id=reserve.base_item_id,
                    lane=unit.lane,
                    operation=unit.operation,
                    side=reserve.side,
                    unresolved_identities=identities,
                ),
                built,
                plan_fixed_position(
                    [built],
                    LADDER_CRITERIA,
                    model=plan.model,
                    levels=plan.levels,
                    base_seed=plan.seed,
                ),
            )
        )
    return out


def rebuild_with_replacements(
    plan: SamplingPlan,
    items_path: Path,
    substitutions: tuple[Substitution, ...],
) -> tuple[list[PairedItem], list[CallSpec]]:
    """The sample as it currently stands, substitutions applied.

    `rebuild_items` is called first and unmodified, because its identity check
    against the frozen list is what proves the reconstruction did not drift.
    Substitutions are then applied to its output: the replaced units are
    retired and their substitutes take the slots, so the unit count is
    preserved and a retired unit is neither analysed nor re-issued on resume.
    """
    items, specs = rebuild_items(plan, items_path)
    if not substitutions:
        return items, specs

    by_id = {b.item_id: b for b in read_jsonl(items_path)}
    superseded = {s.replaced_base_item_id for s in substitutions}

    items = [i for i in items if i.meta.get("base_item") not in superseded]
    specs = [
        s for s in specs if s.item.meta.get("base_item") not in superseded
    ]

    live_ids = {u.base_item_id for u in current_units(plan, substitutions)}
    present = {i.meta.get("base_item") for i in items}
    for unit in current_units(plan, substitutions):
        if unit.base_item_id in present:
            continue
        built = _build_unit_item(unit, by_id, plan)
        if built is None:
            raise IntegrityError(
                f"{unit.base_item_id} no longer builds under "
                f"{unit.operation!r}; the corruption catalogue has changed"
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
    assert {i.meta["base_item"] for i in items} == live_ids
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
        def manifest(root: Path) -> dict[str, str]:
            return {
                str(path.relative_to(root)): file_sha256(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        if destination.is_dir() and manifest(destination) == manifest(run_dir):
            return destination

        def is_append_only_revision() -> bool:
            old_files = manifest(destination)
            new_files = manifest(run_dir)
            for filename in ("calls.jsonl", "replacements.jsonl"):
                if filename not in old_files:
                    continue
                old = (destination / filename).read_bytes()
                candidate = run_dir / filename
                if not candidate.exists() or not candidate.read_bytes().startswith(old):
                    return False
            mutable = {
                "calls.jsonl",
                "replacements.jsonl",
                "items.jsonl",
                "meta.json",
            }
            for filename, digest in old_files.items():
                if filename not in mutable and new_files.get(filename) != digest:
                    return False
            old_meta_path = destination / "meta.json"
            new_meta_path = run_dir / "meta.json"
            if old_meta_path.exists() != new_meta_path.exists():
                return False
            if old_meta_path.exists():
                old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
                new_meta = json.loads(new_meta_path.read_text(encoding="utf-8"))
                for field in (
                    "model",
                    "seed",
                    "manifest_sha256",
                    "dataset_sha256",
                    "model_settings",
                ):
                    if old_meta.get(field) != new_meta.get(field):
                        return False
                if not set(old_meta.get("levels", ())) <= set(
                    new_meta.get("levels", ())
                ):
                    return False
                old_active = {
                    json.dumps(row, sort_keys=True)
                    for row in old_meta.get("active_units", ())
                }
                new_active = {
                    json.dumps(row, sort_keys=True)
                    for row in new_meta.get("active_units", ())
                }
                if not old_active <= new_active:
                    return False
            return True

        if is_append_only_revision():
            old_manifest = manifest(destination)
            digest = hashlib.sha256(
                json.dumps(old_manifest, sort_keys=True).encode("utf-8")
            ).hexdigest()
            history = (
                Path(archive_root) / ".history" / run_dir.name / digest
            )
            pending = Path(archive_root) / f".{run_dir.name}-{digest}.pending"
            if pending.exists():
                raise IntegrityError(
                    f"{pending} exists from an interrupted archival update; "
                    "inspect it before retrying"
                )
            shutil.copytree(run_dir, pending)
            history.parent.mkdir(parents=True, exist_ok=True)
            if history.exists():
                if manifest(history) != old_manifest:
                    raise IntegrityError(
                        f"archive history collision at {history}; refusing update"
                    )
                shutil.rmtree(destination)
            else:
                destination.rename(history)
            pending.rename(destination)
            return destination
        raise IntegrityError(
            f"{destination} already exists with different content; archives "
            "are never overwritten because that is how a run gets silently "
            "replaced"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, destination)
    return destination
