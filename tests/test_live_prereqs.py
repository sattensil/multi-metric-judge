"""Prerequisites for a paid run: reserves to replace from, integrity gates that
refuse a changed frame or manifest, and archival off the ignored path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from packing.cli import cli
from packing.live import (
    IntegrityError,
    archive_run,
    estimate_replacement_round,
    next_replacement,
    rebuild_items,
    sample_fingerprint,
    verify_plan,
    verify_run_dir,
)
from packing.runner import estimate
from packing.sampling_plan import SamplingPlan, Unit, read_plan


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

TEXT_A = """## Overview

The limit is 30 seconds, which matters here. Second sentence adds detail; it also clarifies.

- First bullet point of the list
- Second bullet point of the list
- Third bullet point of the list

Third paragraph closes it out (with an aside). Fourth sentence is spare text!"""

TEXT_B = """## Summary

It takes about 45 minutes, roughly speaking. Another line carries more detail; it expands too.

- Alpha item in this list
- Beta item in this list
- Gamma item in this list

A third paragraph wraps the reply (briefly). A fourth line is filler text!"""


def _items(path: str, count: int) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(count):
            handle.write(
                json.dumps(
                    {
                        "item_id": f"hs2-{i:06d}",
                        "prompt": "Explain the thing.",
                        "response_a": TEXT_A,
                        "response_b": TEXT_B,
                        "human_a": {},
                        "human_b": {},
                        "dataset": "test",
                        "meta": {},
                    }
                )
                + "\n"
            )


def _plan_args(plan_out="plan.json"):
    return [
        "ladder", "--items", "items.jsonl", "--core-n", "16", "--numbers-n", "8",
        "--per-op-max", "2", "--core-candidates", "60", "--plan-out", plan_out,
        "--dry-run",
    ]


def test_the_plan_carries_reserves_as_well_as_active_units():
    runner = CliRunner()
    _items("items.jsonl", 200)
    result = runner.invoke(cli, _plan_args())
    assert result.exit_code == 0, result.output
    plan = read_plan(Path("plan.json"))
    statuses = {u.status for u in plan.units}
    assert statuses == {"active", "reserve"}
    assert plan.active_count == 24
    reserves = [u for u in plan.units if u.status == "reserve"]
    assert reserves, "replacement needs somewhere to draw from"
    # Reserves are screened and eligible, so a replacement is usable at once.
    assert all(u.eligible for u in reserves)


def test_replacement_comes_from_the_same_lane_in_frozen_order():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    core_reserves = [
        u for u in plan.units if u.status == "reserve" and u.lane == "core"
    ]
    first = next_replacement(plan, lane="core", used=set())
    assert first.base_item_id == core_reserves[0].base_item_id
    assert first.lane == "core"
    # Taking one already-used reserve moves to the next, deterministically.
    second = next_replacement(plan, lane="core", used={first.base_item_id})
    assert second.base_item_id == core_reserves[1].base_item_id


def test_replacement_raises_when_the_lane_is_exhausted():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    everything = {u.base_item_id for u in plan.units}
    with pytest.raises(IntegrityError, match="no reserve left in lane"):
        next_replacement(plan, lane="core", used=everything)


def test_verify_refuses_a_changed_items_file():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    _items("items.jsonl", 201)  # one more prompt: different digest
    with pytest.raises(IntegrityError, match="dataset digest"):
        verify_plan(plan, Path("items.jsonl"))


def test_verify_refuses_a_changed_manifest():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    tampered = SamplingPlan(
        dataset_id=plan.dataset_id,
        dataset_sha256=plan.dataset_sha256,
        manifest_sha256="f" * 64,
        seed=plan.seed,
        model=plan.model,
        model_settings=plan.model_settings,
        levels=plan.levels,
        units=plan.units,
        call_identities=plan.call_identities,
    )
    with pytest.raises(IntegrityError, match="manifest digest"):
        verify_plan(tampered, Path("items.jsonl"))


def test_verify_accepts_an_untouched_plan():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    verify_plan(plan, Path("items.jsonl"))


def test_rebuilt_call_identities_match_the_frozen_plan():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    items, specs = rebuild_items(plan, Path("items.jsonl"))
    assert len(items) == plan.active_count
    assert tuple(s.identity for s in specs) == plan.call_identities


def test_rebuild_is_byte_identical_across_calls():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    a, _ = rebuild_items(plan, Path("items.jsonl"))
    b, _ = rebuild_items(plan, Path("items.jsonl"))
    assert [i.response_a for i in a] == [i.response_a for i in b]
    assert [i.response_b for i in a] == [i.response_b for i in b]


def test_archive_copies_the_run_off_the_ignored_path(tmp_path):
    run_dir = tmp_path / "runs" / "ladder"
    run_dir.mkdir(parents=True)
    (run_dir / "calls.jsonl").write_text('{"identity":"x"}\n', encoding="utf-8")
    (run_dir / "meta.json").write_text("{}", encoding="utf-8")
    dest = archive_run(run_dir, tmp_path / "archive")
    assert dest.exists()
    assert (dest / "calls.jsonl").read_text(encoding="utf-8").strip()
    assert (dest / "meta.json").exists()
    # The K = 20 sweep was nearly lost to `runs/*` in .gitignore, so the archive
    # must land somewhere that rule does not cover.
    assert "runs" not in dest.relative_to(tmp_path).parts[0]


def test_archive_is_idempotent_when_the_existing_archive_is_identical(tmp_path):
    run_dir = tmp_path / "runs" / "ladder"
    run_dir.mkdir(parents=True)
    (run_dir / "calls.jsonl").write_text("{}\n", encoding="utf-8")
    first = archive_run(run_dir, tmp_path / "archive")
    assert archive_run(run_dir, tmp_path / "archive") == first


def test_archive_refuses_to_replace_different_existing_content(tmp_path):
    run_dir = tmp_path / "runs" / "ladder"
    run_dir.mkdir(parents=True)
    (run_dir / "calls.jsonl").write_text("{}\n", encoding="utf-8")
    archive_run(run_dir, tmp_path / "archive")
    (run_dir / "calls.jsonl").write_text('{"identity":"new"}\n', encoding="utf-8")
    with pytest.raises(IntegrityError, match="different content"):
        archive_run(run_dir, tmp_path / "archive")


def test_archive_preserves_a_previous_append_only_revision(tmp_path):
    run_dir = tmp_path / "runs" / "ladder"
    run_dir.mkdir(parents=True)
    calls = run_dir / "calls.jsonl"
    calls.write_text('{"identity":"old"}\n', encoding="utf-8")
    archive_root = tmp_path / "archive"
    destination = archive_run(run_dir, archive_root)

    calls.write_text(
        '{"identity":"old"}\n{"identity":"new"}\n', encoding="utf-8"
    )
    assert archive_run(run_dir, archive_root) == destination
    assert '"new"' in (destination / "calls.jsonl").read_text(encoding="utf-8")
    histories = list((archive_root / ".history" / "ladder").iterdir())
    assert len(histories) == 1
    assert '"old"' in (histories[0] / "calls.jsonl").read_text(encoding="utf-8")


def test_ladder_run_verifies_without_sending_by_default():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    result = runner.invoke(
        cli,
        ["ladder-run", "--plan", "plan.json", "--items", "items.jsonl"],
    )
    assert result.exit_code == 0, result.output
    assert "integrity verified" in result.output
    assert "nothing sent" in result.output
    assert not Path("runs").exists(), "no run directory without confirmation"


def test_ladder_run_rejects_a_watchdog_shorter_than_request_attempts():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    result = runner.invoke(
        cli,
        [
            "ladder-run",
            "--plan",
            "plan.json",
            "--items",
            "items.jsonl",
            "--timeout",
            "90",
            "--sdk-retries",
            "1",
            "--stall-seconds",
            "30",
        ],
    )
    assert result.exit_code != 0
    assert "stall-seconds must exceed" in result.output


def test_ladder_run_authorization_includes_sdk_retries():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    result = runner.invoke(
        cli,
        [
            "ladder-run",
            "--plan",
            "plan.json",
            "--items",
            "items.jsonl",
            "--sdk-retries",
            "2",
            "--stall-seconds",
            "400",
            "--replacement-rounds",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3 maximum attempts" in result.output
    assert "authorized ceiling" in result.output


def test_replacement_ceiling_uses_reserve_prompt_lengths():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args("initial.json"))
    initial = read_plan(Path("initial.json"))
    reserve_id = next(
        unit.base_item_id
        for unit in initial.units
        if unit.status == "reserve" and unit.lane == "core"
    )

    path = Path("items.jsonl")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["item_id"] == reserve_id:
            row["prompt"] = "Very long reserve prompt. " * 2_000
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = runner.invoke(cli, _plan_args("plan.json"))
    assert result.exit_code == 0, result.output
    plan = read_plan(Path("plan.json"))

    _, active_specs = rebuild_items(plan, path)
    active = estimate(active_specs)
    replacement = estimate_replacement_round(plan, path, ())
    assert (
        replacement["est_input_tokens_ceiling"]
        > active["est_input_tokens_ceiling"]
    )


def test_ladder_run_refuses_a_changed_frame():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    _items("items.jsonl", 201)
    result = runner.invoke(
        cli,
        ["ladder-run", "--plan", "plan.json", "--items", "items.jsonl"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, (IntegrityError, SystemExit))


def test_ladder_run_executes_against_the_mock_provider(tmp_path):
    runner = CliRunner()
    _items("items.jsonl", 200)
    # `mock` is priced at zero in CATALOG, so this exercises the full live
    # path including archival without sending anything to a provider.
    runner.invoke(
        cli,
        [
            "ladder", "--items", "items.jsonl", "--model", "mock",
            "--core-n", "16", "--numbers-n", "8", "--per-op-max", "2",
            "--core-candidates", "60", "--plan-out", "plan.json",
            "--dry-run",
        ],
    )
    result = runner.invoke(
        cli,
        [
            "ladder-run", "--plan", "plan.json", "--items", "items.jsonl",
            "--name", "t1", "--workers", "4", "--archive-to", "artifacts",
            "--confirm-spend",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "archived to" in result.output
    calls = Path("runs/t1/calls.jsonl")
    assert calls.exists()
    assert sum(1 for _ in calls.open()) == 24 * 5 * 2
    archived = Path("artifacts/t1/calls.jsonl")
    assert archived.exists(), "the archive is what survives .gitignore"
    assert (Path("artifacts/t1/meta.json")).exists()


def test_ladder_run_refuses_to_archive_under_the_ignored_path():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(
        cli,
        [
            "ladder", "--items", "items.jsonl", "--model", "mock",
            "--core-n", "16", "--numbers-n", "8", "--per-op-max", "2",
            "--core-candidates", "60", "--plan-out", "plan.json",
            "--dry-run",
        ],
    )
    result = runner.invoke(
        cli,
        [
            "ladder-run", "--plan", "plan.json", "--items", "items.jsonl",
            "--archive-to", "runs/archive", "--confirm-spend",
        ],
    )
    assert result.exit_code != 0
    assert "gitignore" in result.output.lower()


def _meta(run_dir: Path, plan: SamplingPlan, **overrides) -> None:
    """Write the metadata a previous pass would have left behind."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "calls.jsonl").write_text('{"identity":"prior"}\n', encoding="utf-8")
    body = {
        "model": plan.model,
        "seed": plan.seed,
        "manifest_sha256": plan.manifest_sha256,
        "dataset_sha256": plan.dataset_sha256,
        "sample_sha256": sample_fingerprint(plan),
        "levels": list(plan.levels),
    }
    body.update(overrides)
    (run_dir / "meta.json").write_text(json.dumps(body), encoding="utf-8")


def test_a_run_directory_will_not_accept_a_different_instrument(tmp_path: Path):
    """Reusing --name across plans would mix two experiments in one dataset."""
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))

    run_dir = Path("runs/reused")
    _meta(run_dir, plan, manifest_sha256="a" * 64)
    with pytest.raises(IntegrityError, match="different plan"):
        verify_run_dir(run_dir, plan)

    _meta(run_dir, plan, seed=plan.seed + 1)
    with pytest.raises(IntegrityError, match="different plan"):
        verify_run_dir(run_dir, plan)


def test_a_run_directory_will_not_silently_orphan_a_paid_rung(tmp_path: Path):
    """Records at a K the new plan omits would describe absent items."""
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))

    run_dir = Path("runs/shrunk")
    _meta(run_dir, plan, levels=list(plan.levels) + [999])
    with pytest.raises(IntegrityError, match="K = \\[999\\]"):
        verify_run_dir(run_dir, plan)


def test_extending_the_ladder_into_an_existing_run_is_allowed(tmp_path: Path):
    """The append path exists so a later rung need not repay for earlier ones."""
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))

    run_dir = Path("runs/extended")
    # A prior pass ran a strict subset of this plan's rungs.
    _meta(run_dir, plan, levels=list(plan.levels)[:1])
    verify_run_dir(run_dir, plan)

    # And a fresh directory is always fine.
    verify_run_dir(Path("runs/brand-new"), plan)


def test_monotone_sample_extension_can_resume_the_same_run_name():
    runner = CliRunner()
    _items("items.jsonl", 200)
    common = [
        "--items", "items.jsonl", "--numbers-n", "8",
        "--core-candidates", "60", "--per-op-max", "3", "--dry-run",
    ]
    runner.invoke(
        cli,
        ["ladder", *common, "--core-n", "16", "--plan-out", "first.json"],
    )
    runner.invoke(
        cli,
        ["ladder", *common, "--core-n", "24", "--plan-out", "next.json"],
    )
    first = read_plan(Path("first.json"))
    extended = read_plan(Path("next.json"))

    previous_active = [
        {
            "base_item_id": unit.base_item_id,
            "lane": unit.lane,
            "side": unit.side,
            "operation": unit.operation,
        }
        for unit in first.units
        if unit.status == "active"
    ]
    run_dir = Path("runs/adaptive")
    _meta(run_dir, first, active_units=previous_active)
    verify_run_dir(run_dir, extended)


def test_a_run_directory_rejects_a_different_sample_with_same_instrument():
    """Seed/frame/model equality is insufficient when the selected units differ."""
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))

    run_dir = Path("runs/reselected")
    _meta(run_dir, plan)
    different_sample = SamplingPlan(
        dataset_id=plan.dataset_id,
        dataset_sha256=plan.dataset_sha256,
        manifest_sha256=plan.manifest_sha256,
        seed=plan.seed,
        model=plan.model,
        model_settings=plan.model_settings,
        levels=plan.levels,
        units=plan.units[1:],
        call_identities=plan.call_identities,
    )
    with pytest.raises(IntegrityError, match="different sample"):
        verify_run_dir(run_dir, different_sample)


def test_sample_extension_cannot_activate_a_used_replacement_reserve():
    runner = CliRunner()
    _items("items.jsonl", 200)
    runner.invoke(cli, _plan_args())
    plan = read_plan(Path("plan.json"))
    active = next(unit for unit in plan.units if unit.status == "active")
    reserve = next(
        unit
        for unit in plan.units
        if unit.status == "reserve" and unit.lane == active.lane
    )
    extended = replace(
        plan,
        units=tuple(
            replace(
                unit,
                status="active",
                operation=active.operation,
                realisation_seed=active.realisation_seed,
            )
            if unit.base_item_id == reserve.base_item_id
            else unit
            for unit in plan.units
        ),
    )
    run_dir = Path("runs/with-replacement")
    _meta(
        run_dir,
        plan,
        active_units=[
            {
                "base_item_id": unit.base_item_id,
                "lane": unit.lane,
                "side": unit.side,
                "operation": unit.operation,
            }
            for unit in plan.units
            if unit.status == "active"
        ],
    )
    (run_dir / "replacements.jsonl").write_text(
        json.dumps(
            {
                "replaced_base_item_id": active.base_item_id,
                "replacement_base_item_id": reserve.base_item_id,
                "lane": active.lane,
                "operation": active.operation,
                "side": reserve.side,
                "unresolved_identities": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(IntegrityError, match="replacement reserve"):
        verify_run_dir(run_dir, extended)
