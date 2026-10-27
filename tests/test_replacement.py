"""Same-lane replacement, from detection through to the calls it issues.

The selection rule and the reserves existed from the start; nothing called them,
so a transient transport failure that outlasted its retries left a permanent
hole. These tests pin the part that was missing: finding the units whose ladder
is incomplete, drawing a replacement for each from its own lane, and planning
the replacement's *whole* ladder rather than the one call that failed.

That last point is the one worth stating. `analyze.stratified_endpoint` pairs
within a cluster across rungs and drops any cluster missing an endpoint, so a
replacement supplying a single rung would contribute nothing to the primary
contrast while still costing money.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from packing.cli import cli
from packing.ladder_criteria import LADDER_CRITERIA
from packing.live import (
    IntegrityError,
    Substitution,
    plan_replacements,
    read_substitutions,
    rebuild_with_replacements,
    sample_fingerprint,
    unresolved_by_unit,
    write_substitution,
)
from packing.runner import plan_fixed_position
from packing.sampling_plan import read_plan

import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_live_prereqs import _items, _plan_args  # noqa: E402


def _write_calls(path: Path, specs, *, skip=(), failure=None) -> None:
    """A calls file covering `specs`, minus `skip`, optionally as failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for spec in specs:
            if spec.identity in skip:
                continue
            row = {
                "identity": spec.identity,
                "item_id": spec.item.item_id,
                "side": spec.side,
                "nominal_n": spec.nominal_n,
                "group_size": len(spec.group),
                "replicate": spec.replicate,
            }
            if failure and spec.identity in failure:
                row["failure"] = failure[spec.identity]
                row["failure_class"] = "transient_infrastructure"
            handle.write(json.dumps(row) + "\n")


@pytest.fixture()
def ladder(tmp_path, monkeypatch):
    """A planned ladder with its items and full call list, in a temp tree."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    _items("items.jsonl", 200)
    result = runner.invoke(cli, _plan_args())
    assert result.exit_code == 0, result.output
    plan = read_plan(Path("plan.json"))
    from packing.live import rebuild_items

    items, specs = rebuild_items(plan, Path("items.jsonl"))
    yield plan, items, specs, tmp_path


def test_a_complete_run_needs_no_replacement(ladder):
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    _write_calls(calls, specs)

    assert unresolved_by_unit(specs, calls) == {}
    assert plan_replacements(plan, Path("items.jsonl"), calls, specs=specs) == []


def test_an_unresolved_call_marks_its_whole_unit(ladder):
    """One missing call is one incomplete unit, not one missing observation."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    unresolved = unresolved_by_unit(specs, calls)
    base = victim.item.meta["base_item"]
    assert set(unresolved) == {base}
    assert unresolved[base] == (victim.identity,)


def test_a_transient_failure_row_counts_as_unresolved(ladder):
    """A recorded transient failure is not a completed identity."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(
        calls, specs,
        failure={victim.identity: "gpt-5-mini: request failed: Connection error."},
    )

    unresolved = unresolved_by_unit(specs, calls)
    assert victim.item.meta["base_item"] in unresolved


def test_a_scored_unevaluable_row_is_not_replaced(ladder):
    """Treatment failures are data. Replacing them would condition on survival."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(
        calls, specs,
        failure={
            victim.identity: (
                "gpt-5-mini: truncated at max_completion_tokens=9352 for group_size=40"
            )
        },
    )

    assert unresolved_by_unit(specs, calls) == {}


def test_a_replacement_supplies_the_whole_ladder(ladder):
    """The estimator pairs across rungs, so a partial replacement buys nothing."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    substitutions = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )
    assert len(substitutions) == 1
    sub, item, new_specs = substitutions[0]

    # Every rung, both sides: exactly what an active unit contributes.
    assert len(new_specs) == len(plan.levels) * 2
    assert {s.nominal_n for s in new_specs} == set(plan.levels)
    assert {s.side for s in new_specs} == {"a", "b"}
    assert len(new_specs) > len(unresolved_by_unit(specs, calls)[sub.replaced_base_item_id])


def test_a_replacement_inherits_the_operation_and_comes_from_the_same_lane(ladder):
    """Quotas are per operation, so the substitute has to carry the same one."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    replaced_id = victim.item.meta["base_item"]
    replaced = next(u for u in plan.units if u.base_item_id == replaced_id)

    (sub, item, _), = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )
    assert sub.operation == replaced.operation
    assert sub.lane == replaced.lane

    reserve = next(u for u in plan.units if u.base_item_id == sub.replacement_base_item_id)
    assert reserve.status == "reserve"
    assert reserve.lane == replaced.lane
    # The operation is inherited so the quota holds; the side is the reserve's
    # own, because eligibility was screened for that prompt and that side.
    assert sub.side == reserve.side

    # And the built item is clustered on the replacement prompt, not the victim.
    assert item.meta["base_item"] == sub.replacement_base_item_id
    assert item.meta["operation"] == replaced.operation


def test_two_broken_units_draw_two_different_reserves(ladder):
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    bases = []
    skip = set()
    for spec in specs:
        base = spec.item.meta["base_item"]
        if base not in bases:
            bases.append(base)
            skip.add(spec.identity)
        if len(bases) == 2:
            break
    _write_calls(calls, specs, skip=skip)

    substitutions = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )
    assert len(substitutions) == 2
    drawn = {s.replacement_base_item_id for s, _, _ in substitutions}
    assert len(drawn) == 2, "a reserve was handed out twice"


def test_an_already_recorded_substitution_is_not_repeated(ladder):
    """Resume must not draw a second reserve for a unit already replaced."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    (sub, _, _), = plan_replacements(plan, Path("items.jsonl"), calls, specs=specs)

    log = root / "replacements.jsonl"
    write_substitution(log, sub)
    assert read_substitutions(log) == (sub,)

    again = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs,
        existing=read_substitutions(log),
    )
    assert again == [], "the same unit was replaced twice"


def test_a_recorded_replacement_is_rebuilt_on_resume(ladder):
    """Its items must come back, or its calls cannot be scored or resumed."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})
    (sub, item, new_specs), = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )

    rebuilt_items, rebuilt_specs = rebuild_with_replacements(
        plan, Path("items.jsonl"), (sub,)
    )
    # Substitution, so the sample size is unchanged.
    assert len(rebuilt_items) == len(items)
    assert any(i.item_id == item.item_id for i in rebuilt_items)
    # Identities match, so resume skips what the replacement already collected.
    assert {s.identity for s in new_specs} <= {s.identity for s in rebuilt_specs}


def test_every_broken_unit_is_replaced_when_reserves_allow(ladder):
    """The whole sample failing is recoverable, because reserves outnumber it."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    _write_calls(calls, specs, skip={s.identity for s in specs})

    substitutions = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )
    assert len(substitutions) == plan.active_count
    drawn = {s.replacement_base_item_id for s, _, _ in substitutions}
    assert len(drawn) == plan.active_count, "a reserve was handed out twice"
    # Each lane draws only from itself.
    lanes = {u.base_item_id: u.lane for u in plan.units}
    for sub, _, _ in substitutions:
        assert lanes[sub.replacement_base_item_id] == sub.lane


def test_exhausting_a_lane_is_an_error_not_a_silent_shrink(ladder):
    """Shrinking the sample silently would change the estimand mid-run."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    replaced_id = victim.item.meta["base_item"]
    lane = next(u.lane for u in plan.units if u.base_item_id == replaced_id)
    # Every reserve in that lane already spent on earlier replacements.
    spent = tuple(
        Substitution(
            replaced_base_item_id=f"spent-{i}",
            replacement_base_item_id=unit.base_item_id,
            lane=unit.lane,
            operation="flatten",
            side="a",
            unresolved_identities=(),
        )
        for i, unit in enumerate(
            u for u in plan.units if u.status == "reserve" and u.lane == lane
        )
    )

    with pytest.raises(IntegrityError, match="no reserve left"):
        plan_replacements(
            plan, Path("items.jsonl"), calls, specs=specs, existing=spent
        )


class _FlakyForOneUnit:
    """A mock judge that always fails one base prompt, and only that one.

    Replacement only fires for a call still unresolved *after* the pool has
    drained, so a provider that always succeeds cannot reach the code path. This
    fails one unit's calls with a transient message on every attempt, which is
    the persistent-transport case replacement exists for.
    """

    name = "mock"
    price_in_per_m = 0.0
    price_out_per_m = 0.0

    def __init__(self, doomed_base: str, **kwargs) -> None:
        self.model = "mock"
        self.doomed_base = doomed_base
        self.seen: list[str] = []
        from packing.providers.mock_provider import MockJudge

        self._inner = MockJudge("mock", 0.0, 0.0)

    def judge(self, prompt, group, budget):
        from packing.providers.base import ProviderError

        self.seen.append(prompt.user)
        if self.doomed_base in prompt.user:
            raise ProviderError("gpt-5-mini: request failed: Connection error.")
        return self._inner.judge(prompt, group, budget)


def test_ladder_run_replaces_a_persistently_failing_unit(monkeypatch, tmp_path):
    """The end-to-end path: detect, draw, issue, record, and rebuild items."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    _items("items.jsonl", 200)
    plan_args = [
        "ladder", "--items", "items.jsonl", "--model", "mock",
        "--core-n", "16", "--numbers-n", "8", "--per-op-max", "2",
        "--core-candidates", "60", "--plan-out", "plan.json", "--dry-run",
    ]
    assert runner.invoke(cli, plan_args).exit_code == 0

    plan = read_plan(Path("plan.json"))
    active = [u for u in plan.units if u.status == "active"]
    doomed = active[0].base_item_id

    # The prompt text is what the provider sees, and every synthetic item
    # here shares one prompt, so target the doomed unit by its own prompt.
    # Distinguish it by writing a unique prompt for that base item.
    rows = [json.loads(line) for line in Path("items.jsonl").read_text().splitlines()]
    for row in rows:
        if row["item_id"] == doomed:
            row["prompt"] = "DOOMED PROMPT explain the thing."
    Path("items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # The plan pins the frame digest, so re-plan against the edited file.
    assert runner.invoke(cli, plan_args).exit_code == 0
    plan = read_plan(Path("plan.json"))
    doomed = next(
        u.base_item_id for u in plan.units
        if u.status == "active" and u.base_item_id == doomed
    )

    provider = _FlakyForOneUnit("DOOMED PROMPT")
    monkeypatch.setattr(
        "packing.cli.get_provider", lambda model, **kwargs: provider
    )

    result = runner.invoke(
        cli,
        [
            "ladder-run", "--plan", "plan.json", "--items", "items.jsonl",
            "--name", "rep", "--workers", "4", "--archive-to", "artifacts",
            "--max-failures", "10000", "--replacement-rounds", "1",
            "--confirm-spend",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "replacement round 1" in result.output, result.output
    assert doomed in result.output

    log = Path("runs/rep/replacements.jsonl")
    assert log.exists(), "the substitution was not recorded"
    substitutions = read_substitutions(log)
    assert len(substitutions) == 1
    assert substitutions[0].replaced_base_item_id == doomed
    assert substitutions[0].operation == next(
        u.operation for u in plan.units if u.base_item_id == doomed
    )

    # The replacement's item is on disk, or its calls cannot be scored.
    item_ids = {
        json.loads(line)["item_id"]
        for line in Path("runs/rep/items.jsonl").read_text().splitlines()
    }
    assert any(
        substitutions[0].replacement_base_item_id in item_id
        for item_id in item_ids
    ), "the replacement item was not written"

    # And the replacement's own calls landed and succeeded.
    rows = [
        json.loads(line)
        for line in Path("runs/rep/calls.jsonl").read_text().splitlines()
    ]
    replacement_rows = [
        r for r in rows
        if substitutions[0].replacement_base_item_id in r["item_id"]
    ]
    assert len(replacement_rows) == len(plan.levels) * 2
    assert not any(r.get("failure") for r in replacement_rows)


def test_replacement_can_be_switched_off(monkeypatch, tmp_path):
    """Zero rounds must issue nothing, so the extra spend is opt-out."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    _items("items.jsonl", 200)
    rows = [json.loads(line) for line in Path("items.jsonl").read_text().splitlines()]
    rows[0]["prompt"] = "DOOMED PROMPT explain the thing."
    Path("items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert runner.invoke(cli, [
        "ladder", "--items", "items.jsonl", "--model", "mock",
        "--core-n", "16", "--numbers-n", "8", "--per-op-max", "2",
        "--core-candidates", "60", "--plan-out", "plan.json", "--dry-run",
    ]).exit_code == 0

    provider = _FlakyForOneUnit("DOOMED PROMPT")
    monkeypatch.setattr(
        "packing.cli.get_provider", lambda model, **kwargs: provider
    )
    result = runner.invoke(cli, [
        "ladder-run", "--plan", "plan.json", "--items", "items.jsonl",
        "--name", "off", "--workers", "4", "--archive-to", "artifacts",
        "--max-failures", "10000", "--replacement-rounds", "0",
        "--confirm-spend",
    ])
    assert result.exit_code == 0, result.output
    assert "replacement round" not in result.output
    assert not Path("runs/off/replacements.jsonl").exists()


def test_a_replacement_substitutes_rather_than_adds(ladder):
    """Both units surviving would put an operation over its quota.

    The estimator weights operations equally over per-unit differences, so a
    replaced unit that still carries both endpoints keeps contributing a
    difference while its substitute contributes another. The slot was designed
    to hold one unit, and the quota is what the equal-weight estimator assumes.
    """
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    # Break a middle rung, so the victim's endpoints are both intact and it
    # would otherwise still enter the primary contrast.
    victim = next(s for s in specs if s.nominal_n not in (1, max(plan.levels)))
    _write_calls(calls, specs, skip={victim.identity})
    replaced_id = victim.item.meta["base_item"]

    (sub, _, _), = plan_replacements(plan, Path("items.jsonl"), calls, specs=specs)
    rebuilt_items, rebuilt_specs = rebuild_with_replacements(
        plan, Path("items.jsonl"), (sub,)
    )

    bases = {i.meta["base_item"] for i in rebuilt_items}
    assert sub.replacement_base_item_id in bases, "the substitute is missing"
    assert replaced_id not in bases, (
        "the replaced unit is still in the analysed set, so its operation now "
        "holds two units where the design allots one"
    )
    # Unit count is preserved exactly: substitution, not addition.
    assert len(rebuilt_items) == len(items)
    spec_bases = {s.item.meta["base_item"] for s in rebuilt_specs}
    assert replaced_id not in spec_bases, "resume would re-issue a retired unit"


def test_a_replacement_uses_the_reserve_s_own_screened_side(ladder):
    """Eligibility is screened per prompt and per side, so the side travels.

    `eligibility.select_response` keeps the side that satisfies every core
    operation for that prompt, and records it on the unit. Applying the
    replaced unit's side to the reserve's prompt uses a side that was never
    screened for it, which can admit a response the core lane's all-operation
    requirement rejects.
    """
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    (sub, item, _), = plan_replacements(plan, Path("items.jsonl"), calls, specs=specs)
    reserve = next(
        u for u in plan.units if u.base_item_id == sub.replacement_base_item_id
    )
    assert sub.side == reserve.side, (
        f"substitution recorded side {sub.side!r} but the reserve was screened "
        f"on side {reserve.side!r}"
    )
    assert item.meta["source_side"] == reserve.side


def test_a_failed_replacement_is_itself_replaced(ladder):
    """Otherwise a second round cannot recover the first round's failure."""
    plan, items, specs, root = ladder
    calls = root / "calls.jsonl"
    victim = specs[0]
    _write_calls(calls, specs, skip={victim.identity})

    (first, _, first_specs), = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )
    log = root / "replacements.jsonl"
    write_substitution(log, first)

    # Round two: the substitute's own ladder is incomplete.
    _, live_specs = rebuild_with_replacements(
        plan, Path("items.jsonl"), read_substitutions(log)
    )
    _write_calls(calls, live_specs, skip={first_specs[0].identity})

    second = plan_replacements(
        plan, Path("items.jsonl"), calls,
        specs=live_specs, existing=read_substitutions(log),
    )
    assert len(second) == 1, (
        "a replacement that failed was neither retried nor replaced"
    )
    assert second[0][0].replaced_base_item_id == first.replacement_base_item_id
    assert second[0][0].replacement_base_item_id not in {
        first.replacement_base_item_id, first.replaced_base_item_id
    }


def test_ladder_run_resume_does_not_reissue_a_retired_unit(ladder, monkeypatch):
    """A recorded substitution changes the live sample before execution resumes."""
    plan, items, specs, root = ladder
    victim = specs[0]
    calls = root / "calls.jsonl"
    _write_calls(calls, specs, skip={victim.identity})

    (substitution, _, _), = plan_replacements(
        plan, Path("items.jsonl"), calls, specs=specs
    )
    run_dir = Path("runs/resume")
    run_dir.mkdir(parents=True)
    calls.replace(run_dir / "calls.jsonl")
    write_substitution(run_dir / "replacements.jsonl", substitution)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": plan.model,
                "seed": plan.seed,
                "manifest_sha256": plan.manifest_sha256,
                "dataset_sha256": plan.dataset_sha256,
                "sample_sha256": sample_fingerprint(plan),
                "levels": list(plan.levels),
            }
        ),
        encoding="utf-8",
    )

    from packing.providers.mock_provider import MockJudge

    monkeypatch.setattr(
        "packing.cli.get_provider",
        lambda model, **kwargs: MockJudge("mock", 0.0, 0.0),
    )
    result = CliRunner().invoke(
        cli,
        [
            "ladder-run",
            "--plan",
            "plan.json",
            "--items",
            "items.jsonl",
            "--name",
            "resume",
            "--archive-to",
            "artifacts",
            "--replacement-rounds",
            "0",
            "--confirm-spend",
        ],
    )
    assert result.exit_code == 0, result.output

    identities = {
        row["identity"]
        for row in (
            json.loads(line)
            for line in (run_dir / "calls.jsonl").read_text().splitlines()
        )
    }
    assert victim.identity not in identities, (
        "resume re-issued a call for the retired original unit instead of "
        "continuing with its recorded substitute"
    )
