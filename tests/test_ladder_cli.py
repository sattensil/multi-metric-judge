"""The ladder command. Dry run is the default, it writes the frozen sampling
plan, and it refuses to cost a partial sample."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from packing.cli import cli
from packing.sampling_plan import read_plan

# Rich enough that all eight core operations apply: headings and a list for
# `flatten`, varied punctuation for `depunct`, several prose sentences for
# `shuffle` and `drop_answer`, and a numeral for the `numbers` arm. A plainer
# fixture leaves the core lane empty, which is a fixture problem rather than a
# command problem and is confusing to debug.
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


def _run(runner, *args):
    return runner.invoke(cli, ["ladder", *args])


def test_dry_run_reports_the_plan_without_sending():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--model", "gpt-5-mini",
            "--core-n", "16", "--numbers-n", "8", "--per-op-max", "2",
            "--core-candidates", "60", "--dry-run",
        )
        assert result.exit_code == 0, result.output
        assert "dry run: nothing sent" in result.output
        assert "K=40 pool" in result.output
        assert "calls:" in result.output


def test_lanes_are_reported_separately():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "16", "--numbers-n", "8",
            "--per-op-max", "2", "--core-candidates", "60", "--dry-run",
        )
        assert "core lane" in result.output
        assert "numbers lane" in result.output


def test_a_short_lane_is_an_error_not_a_warning():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        # The lane exists and screens fine, it just cannot reach the quota.
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "464",
            "--numbers-n", "8", "--core-candidates", "100", "--dry-run",
        )
        assert result.exit_code != 0
        assert "short of quota" in result.output


def test_a_frame_too_small_for_the_lane_split_reports_cleanly():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 12)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "464",
            "--numbers-n", "184", "--dry-run",
        )
        assert result.exit_code != 0
        assert "cannot supply 508 core candidates" in result.output
        assert "packing fetch" in result.output


def test_allow_short_is_available_for_development():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 40)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "8", "--numbers-n", "40",
            "--per-op-max", "1", "--core-candidates", "20", "--allow-short",
            "--dry-run",
        )
        assert result.exit_code == 0, result.output
        assert "SHORT SAMPLE" in result.output


def test_unknown_model_pricing_is_an_error_not_zero():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--model", "not-a-model",
            "--core-n", "16", "--numbers-n", "8", "--per-op-max", "2",
            "--core-candidates", "60", "--dry-run",
        )
        assert result.exit_code != 0
        assert "no pricing" in result.output


def test_core_n_not_divisible_by_eight_is_rejected():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "15", "--numbers-n", "8",
            "--dry-run",
        )
        assert result.exit_code != 0
        assert "not divisible by 8" in result.output


def test_levels_must_span_one_to_forty():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "16", "--numbers-n", "8",
            "--levels", "1,5,20", "--dry-run",
        )
        assert result.exit_code != 0
        assert "must run from 1 to 40" in result.output


def test_duplicate_base_prompt_ids_are_rejected():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 20)
        with open("items.jsonl", "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "item_id": "hs2-000000",
                        "prompt": "dupe",
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
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "8", "--numbers-n", "8",
            "--dry-run",
        )
        assert result.exit_code != 0
        assert "duplicate base-prompt ids" in result.output


def test_the_sampling_plan_is_written_and_readable():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "16", "--numbers-n", "8",
            "--per-op-max", "2", "--core-candidates", "60",
            "--plan-out", "plan.json", "--dry-run",
        )
        assert result.exit_code == 0, result.output
        plan = read_plan(Path("plan.json"))
        assert len(plan.manifest_sha256) == 64
        assert len(plan.dataset_sha256) == 64
        assert plan.levels == (1, 5, 10, 20, 40)
        assert plan.active_count == 24
        # A base prompt contributes to at most one arm, enforced on construction.
        lanes = {u.lane for u in plan.units}
        assert lanes == {"core", "numbers"}
        assert plan.call_identities


def test_the_plan_is_reproducible_for_a_seed():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        args = [
            "--items", "items.jsonl", "--core-n", "16", "--numbers-n", "8",
            "--per-op-max", "2", "--core-candidates", "60", "--dry-run",
        ]
        _run(runner, *args, "--plan-out", "a.json")
        _run(runner, *args, "--plan-out", "b.json")
        assert read_plan(Path("a.json")) == read_plan(Path("b.json"))


def test_live_execution_is_refused():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 200)
        result = _run(
            runner, "--items", "items.jsonl", "--core-n", "16", "--numbers-n", "8",
            "--per-op-max", "2", "--core-candidates", "60", "--no-dry-run",
        )
        assert result.exit_code != 0
        assert "frozen sampling plan" in result.output
