"""Synthetic-gold outcomes: detection, specificity, leakage, and strata."""

from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner

from packing import analyze, power
from packing import cli as cli_module
from packing.dataset import PairedItem
from packing.dataset import write_jsonl
from packing.score import make_outcome


def _item(
    item_id: str,
    *,
    operation: str = "shuffle",
    target: str = "coherence",
    direction: int = -1,
    corrupted_side: str = "b",
    base_item: str | None = None,
    base_response_side: str | None = None,
) -> PairedItem:
    return PairedItem(
        item_id=item_id,
        prompt="Explain the topic.",
        response_a="Clean response.",
        response_b="Corrupted response.",
        human_a={target: 3},
        human_b={target: 1},
        dataset="synthetic",
        meta={
            "corruption": operation,
            "direction": direction,
            "corrupted_side": corrupted_side,
            **({"base_item": base_item} if base_item else {}),
            **(
                {"base_response_side": base_response_side}
                if base_response_side
                else {}
            ),
        },
        source="synthetic",
        targeted_criterion=target,
    )


def _row(
    item_id: str,
    criterion: str,
    *,
    group_size: int,
    judge_a: int,
    judge_b: int,
    correct: bool | None,
    judge_tie: bool = False,
) -> dict:
    return {
        "item_id": item_id,
        "criterion": criterion,
        "group_size": group_size,
        "position": 0,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "judge_tie": judge_tie,
        "correct": correct,
        "correct_num": float(correct) if correct is not None else float("nan"),
        "credit": 0.5 if judge_tie else float(bool(correct)),
        "resolved": correct is not None,
        "human_tie": criterion != "coherence",
    }


def test_annotate_synthetic_identifies_target_and_corruption_aligned_delta() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "i1",
                "coherence",
                group_size=2,
                judge_a=4,
                judge_b=2,
                correct=True,
            ),
            _row(
                "i1",
                "verbosity",
                group_size=2,
                judge_a=2,
                judge_b=3,
                correct=None,
            ),
        ]
    )

    result = analyze.annotate_synthetic(
        frame,
        [_item("i1")],
    )

    target = result[result["criterion"] == "coherence"].iloc[0]
    companion = result[result["criterion"] == "verbosity"].iloc[0]
    assert bool(target["is_target"])
    assert not bool(companion["is_target"])
    assert target["operation"] == "shuffle"
    assert target["calibration_role"] == "test"
    assert target["detection_class"] == "semantic"
    assert target["corrupted_delta"] == -2
    assert target["aligned_delta"] == 2


def test_annotate_synthetic_preserves_shared_base_response_cluster() -> None:
    frame = pd.DataFrame(
        [
            _row("shuffle-item", "coherence", group_size=2, judge_a=4, judge_b=2, correct=True),
            _row("rename-item", "terminology", group_size=2, judge_a=4, judge_b=2, correct=True),
        ]
    )
    items = [
        _item(
            "shuffle-item",
            base_item="hs2-1",
            base_response_side="a",
        ),
        _item(
            "rename-item",
            operation="rename",
            target="terminology",
            base_item="hs2-1",
            base_response_side="b",
        ),
    ]

    result = analyze.annotate_synthetic(frame, items)

    assert result["base_unit"].nunique() == 1
    assert result["base_unit"].iloc[0] == "hs2-1"


def test_synthetic_recovery_keeps_hits_ties_and_wrong_directions_separate() -> None:
    frame = pd.DataFrame(
        [
            {**_row("hit", "coherence", group_size=5, judge_a=4, judge_b=2, correct=True), "is_target": True, "calibration_role": "test"},
            {**_row("wrong", "coherence", group_size=5, judge_a=1, judge_b=3, correct=False), "is_target": True, "calibration_role": "test"},
            {**_row("tie", "coherence", group_size=5, judge_a=2, judge_b=2, correct=None, judge_tie=True), "is_target": True, "calibration_role": "test"},
        ]
    )

    result = analyze.synthetic_recovery(frame).iloc[0]

    assert result["n"] == 3
    assert result["recovery_rate"] == pytest.approx(1 / 3)
    assert result["tie_rate"] == pytest.approx(1 / 3)
    assert result["discrimination"] == pytest.approx(1 / 2)


def test_synthetic_decision_diagnostics_exclude_unevaluable_calls() -> None:
    frame = pd.DataFrame(
        [
            {
                **_row(
                    "hit",
                    "coherence",
                    group_size=5,
                    judge_a=4,
                    judge_b=2,
                    correct=True,
                ),
                "is_target": True,
                "calibration_role": "test",
                "unevaluable": False,
            },
            {
                **_row(
                    "unreadable",
                    "coherence",
                    group_size=5,
                    judge_a=0,
                    judge_b=0,
                    correct=False,
                ),
                "is_target": True,
                "calibration_role": "test",
                "unevaluable": True,
            },
        ]
    )

    result = analyze.synthetic_recovery(frame).iloc[0]
    assert result["n"] == 2
    assert result["n_evaluable"] == 1
    assert result["wrong"] == 0
    assert result["recovery_rate"] == pytest.approx(1 / 2)
    assert result["discrimination"] == 1.0
    assert result["unevaluable_rate"] == pytest.approx(1 / 2)


def test_synthetic_recovery_frame_treats_ties_and_wrong_directions_as_misses() -> None:
    frame = pd.DataFrame(
        [
            {**_row("hit", "coherence", group_size=5, judge_a=4, judge_b=2, correct=True), "is_target": True},
            {**_row("wrong", "coherence", group_size=5, judge_a=1, judge_b=3, correct=False), "is_target": True},
            {**_row("tie", "coherence", group_size=5, judge_a=2, judge_b=2, correct=None, judge_tie=True), "is_target": True},
            {**_row("companion", "verbosity", group_size=5, judge_a=2, judge_b=2, correct=None, judge_tie=True), "is_target": False},
        ]
    )

    result = analyze.synthetic_recovery_frame(frame)

    assert list(result["item_id"]) == ["hit", "wrong", "tie"]
    assert list(result["correct_num"]) == [1.0, 0.0, 0.0]
    assert result["resolved"].all()
    assert result["recovered"].tolist() == [True, False, False]


def test_synthetic_specificity_counts_unchanged_companion_scores_as_correct() -> None:
    frame = pd.DataFrame(
        [
            {**_row("stable", "verbosity", group_size=3, judge_a=2, judge_b=2, correct=None, judge_tie=True), "is_target": False, "aligned_delta": 0},
            {**_row("leaked", "verbosity", group_size=3, judge_a=2, judge_b=3, correct=None), "is_target": False, "aligned_delta": -1},
        ]
    )

    result = analyze.synthetic_specificity(frame).iloc[0]

    assert result["n"] == 2
    assert result["specificity"] == pytest.approx(0.5)
    assert result["movement_rate"] == pytest.approx(0.5)
    assert result["mean_aligned_delta"] == pytest.approx(-0.5)


def test_synthetic_specificity_handles_target_only_run() -> None:
    frame = pd.DataFrame(
        [
            {
                **_row("target", "coherence", group_size=1, judge_a=4, judge_b=2, correct=True),
                "is_target": True,
                "aligned_delta": 2,
            }
        ]
    )

    result = analyze.synthetic_specificity(frame)

    assert result.empty
    assert "specificity" in result.columns


def test_paired_contrast_clusters_reused_corruptions_by_base_response() -> None:
    rows = []
    for base_unit in ("base-a", "base-b", "base-c"):
        for operation in ("shuffle", "rename"):
            for group_size, detected in (
                (1, 1.0),
                (20, 1.0 if base_unit == "base-c" else 0.0),
            ):
                rows.append(
                    {
                        "item_id": f"{base_unit}-{operation}",
                        "base_unit": base_unit,
                        "group_size": group_size,
                        "correct_num": detected,
                        "resolved": True,
                    }
                )

    result = analyze.paired_contrast(pd.DataFrame(rows), 1, 20)

    assert result is not None
    assert result.n_items == 3
    assert result.mean_difference == pytest.approx(-2 / 3)
    assert len(power.per_item_differences(pd.DataFrame(rows), 1, 20)) == 3


def test_leakage_matrix_reports_systematic_aligned_movement_not_random_noise() -> None:
    frame = pd.DataFrame(
        [
            {
                **_row("i1", "coherence", group_size=2, judge_a=4, judge_b=2, correct=True),
                "operation": "shuffle",
                "is_target": True,
                "aligned_delta": 2,
            },
            {
                **_row("i1", "verbosity", group_size=2, judge_a=2, judge_b=3, correct=None),
                "operation": "shuffle",
                "is_target": False,
                "aligned_delta": -1,
            },
            {
                **_row("i2", "verbosity", group_size=2, judge_a=3, judge_b=2, correct=None),
                "operation": "shuffle",
                "is_target": False,
                "aligned_delta": 1,
            },
        ]
    )

    matrix = analyze.synthetic_leakage_matrix(frame)

    assert matrix.loc["shuffle", "coherence"] == pytest.approx(2.0)
    assert matrix.loc["shuffle", "verbosity"] == pytest.approx(0.0)


def test_leakage_summary_compares_worst_companion_to_target_movement() -> None:
    frame = pd.DataFrame(
        [
            {
                **_row("i1", "coherence", group_size=3, judge_a=4, judge_b=2, correct=True),
                "operation": "shuffle",
                "is_target": True,
                "aligned_delta": 2,
            },
            {
                **_row("i1", "verbosity", group_size=3, judge_a=2, judge_b=3, correct=None),
                "operation": "shuffle",
                "is_target": False,
                "aligned_delta": -1,
            },
            {
                **_row("i1", "fluency", group_size=3, judge_a=2, judge_b=2, correct=None, judge_tie=True),
                "operation": "shuffle",
                "is_target": False,
                "aligned_delta": 0,
            },
        ]
    )

    result = analyze.synthetic_leakage_summary(frame).loc["shuffle"]

    assert result["target_mean_aligned_delta"] == pytest.approx(2.0)
    assert result["worst_companion"] == "verbosity"
    assert result["worst_companion_mean_aligned_delta"] == pytest.approx(-1.0)
    assert result["worst_to_target_ratio"] == pytest.approx(0.5)


def test_leakage_ratio_matches_target_to_same_companion_calls() -> None:
    frame = pd.DataFrame(
        [
            {
                **_row("i1", "coherence", group_size=1, judge_a=4, judge_b=0, correct=True),
                "operation": "shuffle",
                "is_target": True,
                "aligned_delta": 4,
                "partition_hash": "solo",
            },
            {
                **_row("i1", "coherence", group_size=2, judge_a=4, judge_b=2, correct=True),
                "operation": "shuffle",
                "is_target": True,
                "aligned_delta": 2,
                "partition_hash": "shared",
            },
            {
                **_row("i1", "verbosity", group_size=2, judge_a=2, judge_b=3, correct=None),
                "operation": "shuffle",
                "is_target": False,
                "aligned_delta": -1,
                "partition_hash": "shared",
            },
        ]
    )

    result = analyze.synthetic_leakage_summary(frame).loc["shuffle"]

    assert result["target_mean_aligned_delta"] == pytest.approx(2.0)
    assert result["worst_to_target_ratio"] == pytest.approx(0.5)


def test_synthetic_frontier_marks_only_non_dominated_cost_quality_points() -> None:
    rows = []
    for group_size, hits in ((1, 9), (5, 8), (10, 7)):
        for index in range(10):
            rows.append(
                {
                    **_row(
                        f"{group_size}-{index}",
                        "coherence",
                        group_size=group_size,
                        judge_a=4 if index < hits else 2,
                        judge_b=2,
                        correct=True if index < hits else None,
                        judge_tie=index >= hits,
                    ),
                    "is_target": True,
                    "calibration_role": "test",
                    "detection_class": "semantic",
                }
            )
    cost = pd.DataFrame(
        {"usd_per_1k_criteria": [0.6, 0.3, 0.4]},
        index=pd.Index([1, 5, 10], name="group_size"),
    )

    result = analyze.synthetic_frontier(pd.DataFrame(rows), cost)

    assert bool(result.loc[("semantic", 1), "pareto_efficient"])
    assert bool(result.loc[("semantic", 5), "pareto_efficient"])
    assert not bool(result.loc[("semantic", 10), "pareto_efficient"])
    assert "recovery_rate" in result.columns


def test_leakage_is_reported_without_excluding_operations() -> None:
    """Leakage is a diagnostic: high companion shift does not drop the operation."""
    frame = pd.DataFrame(
        [
            {
                **_row("i1", "coherence", group_size=2, judge_a=4, judge_b=2, correct=True),
                "operation": "shuffle",
                "is_target": True,
                "aligned_delta": 2,
                "calibration_role": "test",
                "detection_class": "semantic",
            },
            {
                **_row("i1", "verbosity", group_size=2, judge_a=2, judge_b=4, correct=None),
                "operation": "shuffle",
                "is_target": False,
                "aligned_delta": -2,
                "calibration_role": "test",
                "detection_class": "semantic",
            },
        ]
    )

    summary = analyze.synthetic_leakage_summary(frame)
    recovery = analyze.synthetic_recovery(frame)

    assert "shuffle" in summary.index
    assert summary.loc["shuffle", "worst_to_target_ratio"] == pytest.approx(1.0)
    assert recovery.iloc[0]["recovery_rate"] == pytest.approx(1.0)
    assert recovery.iloc[0]["n"] == 1


def test_analyze_command_uses_synthetic_gold_report_for_synthetic_run(
    monkeypatch,
) -> None:
    item = _item("i1")
    outcomes = [
        make_outcome(
            item_id="i1",
            criterion="coherence",
            group_size=2,
            position=0,
            nominal_n=2,
            replicate=0,
            model="test",
            partition_hash="target",
            human_a=3,
            human_b=1,
            judge_a=4,
            judge_b=2,
        ),
        make_outcome(
            item_id="i1",
            criterion="verbosity",
            group_size=2,
            position=1,
            nominal_n=2,
            replicate=0,
            model="test",
            partition_hash="target",
            human_a=2,
            human_b=2,
            judge_a=2,
            judge_b=3,
        ),
    ]
    usage_records = [
        {
            "group_size": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
        }
    ]
    monkeypatch.setattr(
        cli_module,
        "_load_run",
        lambda _name: (
            [item],
            (),
            usage_records,
            {
                "dataset": "synthetic",
                "model": "mock",
                    "operation_roles": {"shuffle": "test"},
                    "detection_classes": {"shuffle": "custom_class"},
            },
        ),
    )
    monkeypatch.setattr(cli_module.analyze, "build_outcomes", lambda *_args: outcomes)
    monkeypatch.setattr(
        cli_module.analyze,
        "fit_mixed",
        lambda _frame: type("Fit", (), {"summary": lambda self: "mixed-summary"})(),
    )

    result = CliRunner().invoke(
        cli_module.cli, ["analyze", "--run", "synthetic-test"]
    )

    assert result.exit_code == 0, result.output
    assert "Synthetic targeted recovery" in result.output
    assert "custom_class" in result.output
    assert "Untouched-companion specificity" in result.output
    assert "Leakage diagnostic" in result.output
    assert "Crossed mixed model" in result.output
    assert "paired recovery contrast" in result.output
    assert "Synthetic recovery/cost frontier" in result.output


def test_load_run_uses_full_criterion_pool_for_synthetic_dataset(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "synthetic"
    run_dir.mkdir()
    write_jsonl([_item("i1")], run_dir / "items.jsonl")
    (run_dir / "calls.jsonl").write_text("")
    (run_dir / "meta.json").write_text(
        '{"dataset":"synthetic","model":"mock"}'
    )
    monkeypatch.setattr(cli_module, "_run_dir", lambda _name: run_dir)

    _items, criteria, _records, _meta = cli_module._load_run("synthetic")

    assert len(criteria) == 20
    assert {criterion.key for criterion in criteria} >= {
        "coherence",
        "fluency",
        "support",
    }


def test_power_command_uses_synthetic_detection_and_base_clusters(
    monkeypatch,
) -> None:
    item = _item("i1", base_item="hs2-1", base_response_side="a")
    tie = make_outcome(
        item_id="i1",
        criterion="coherence",
        group_size=20,
        position=0,
        nominal_n=20,
        replicate=0,
        model="test",
        partition_hash="target",
        human_a=3,
        human_b=1,
        judge_a=2,
        judge_b=2,
    )
    monkeypatch.setattr(
        cli_module,
        "_load_run",
        lambda _name: (
            [item],
            (),
            [],
            {
                "dataset": "synthetic",
                "model": "gpt-5-mini",
                "operation_roles": {"shuffle": "test"},
                "detection_classes": {"shuffle": "semantic"},
            },
        ),
    )
    monkeypatch.setattr(cli_module.analyze, "build_outcomes", lambda *_args: [tie])
    captured = {}

    class Estimate:
        def as_dict(self):
            return {"ok": True}

    def fake_plan(frame, reference, comparison):
        captured["frame"] = frame
        captured["reference"] = reference
        captured["comparison"] = comparison
        return Estimate()

    monkeypatch.setattr(cli_module.power, "plan_from_pilot", fake_plan)

    result = CliRunner().invoke(
        cli_module.cli,
        ["power", "--run", "synthetic-test", "--reference", "1", "--comparison", "20"],
    )

    assert result.exit_code == 0, result.output
    assert captured["frame"]["base_unit"].iloc[0] == "hs2-1"
    assert captured["frame"]["correct_num"].iloc[0] == 0.0
    assert bool(captured["frame"]["resolved"].iloc[0])


def test_sweep_rejects_model_without_matching_calibration(tmp_path) -> None:
    items_path = tmp_path / "items.jsonl"
    write_jsonl([_item("i1")], items_path)

    result = CliRunner().invoke(
        cli_module.cli,
        ["sweep", "--items", str(items_path), "--model", "mock", "--dry-run"],
    )

    assert result.exit_code != 0
    assert "calibrated only for gpt-5-mini" in result.output
