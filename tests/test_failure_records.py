"""Failure rows, paired-unit scoring, and resume semantics."""

from __future__ import annotations

import json

import pytest

from packing.analyze import build_outcomes, to_frame
from packing.criteria import HELPSTEER2_CRITERIA
from packing.dataset import PairedItem
from packing.runner import _build_failure_record, completed_identities
from packing.unevaluable import TRANSIENT, UNEVALUABLE, FatalProtocolError

ONE = HELPSTEER2_CRITERIA[:1]
KEY = ONE[0].key


def _item():
    return PairedItem(
        item_id="lad-shuffle-hs2-000001-a",
        prompt="p",
        response_a="x",
        response_b="y",
        human_a={KEY: 3},
        human_b={KEY: 1},
        dataset="synthetic",
        meta={"base_item": "hs2-000001", "operation": "shuffle"},
        source="synthetic",
        targeted_criterion=KEY,
    )


class _Spec:
    """Minimal stand-in carrying only what the record builder reads."""

    def __init__(self, item, side, group, nominal_n=1):
        self.item = item
        self.side = side
        self.group = group
        self.group_index = 0
        self.nominal_n = nominal_n
        self.replicate = 0
        self.model = "mock"
        self.identity = f"{item.item_id}|{side}|mock|0|deadbeef|0"

        class _P:
            hash = "deadbeef"

        self.partition = _P()


def _good(side, score, nominal_n=1):
    return {
        "identity": f"lad-shuffle-hs2-000001-a|{side}|mock|0|deadbeef|0",
        "item_id": "lad-shuffle-hs2-000001-a",
        "side": side,
        "model": "mock",
        "nominal_n": nominal_n,
        "replicate": 0,
        "partition_hash": "deadbeef",
        "group_index": 0,
        "group_size": nominal_n,
        "group": [KEY],
        "positions": {KEY: 0},
        "scores": {KEY: score},
        "reasoning": {KEY: ""},
        "input_tokens": 10,
        "output_tokens": 10,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "near_cap": False,
    }


def test_truncation_is_class_two():
    record = _build_failure_record(
        _Spec(_item(), "a", ONE), "truncated at max_completion_tokens=9352"
    )
    assert record["failure_class"] == UNEVALUABLE
    assert record["scores"] == {}
    assert record["positions"] == {KEY: 0}


def test_connection_error_is_class_one():
    record = _build_failure_record(
        _Spec(_item(), "a", ONE), "Connection reset by peer"
    )
    assert record["failure_class"] == TRANSIENT


def test_a_fatal_failure_raises_rather_than_recording():
    with pytest.raises(FatalProtocolError, match="unsupported parameter"):
        _build_failure_record(
            _Spec(_item(), "a", ONE), "invalid_request_error: unsupported parameter"
        )


def test_one_unevaluable_side_yields_exactly_one_outcome():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    outcomes = build_outcomes([failure, _good("b", 3)], [_item()], ONE)
    assert len(outcomes) == 1
    assert to_frame(outcomes)["correct_num"].iloc[0] == 0.0


def test_both_sides_unevaluable_still_yields_exactly_one_outcome():
    a = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    b = _build_failure_record(_Spec(_item(), "b", ONE), "truncated")
    outcomes = build_outcomes([a, b], [_item()], ONE)
    assert len(outcomes) == 1, "a call-level rule would emit two here"


def test_both_sides_valid_scores_normally():
    outcomes = build_outcomes([_good("a", 4), _good("b", 1)], [_item()], ONE)
    assert len(outcomes) == 1
    frame = to_frame(outcomes)
    assert frame["resolved"].iloc[0]
    assert frame["correct_num"].iloc[0] == 1.0


def test_complete_case_drops_the_unevaluable_pair():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    outcomes = build_outcomes(
        [failure, _good("b", 3)], [_item()], ONE, unevaluable_rule="complete_case"
    )
    assert outcomes == []


def test_bounds_impute_oppositely_by_rung():
    ref = _build_failure_record(_Spec(_item(), "a", ONE, 1), "did not parse")
    lower = build_outcomes(
        [ref, _good("b", 3, 1)], [_item()], ONE,
        unevaluable_rule="lower_bound", reference_size=1,
    )
    upper = build_outcomes(
        [ref, _good("b", 3, 1)], [_item()], ONE,
        unevaluable_rule="upper_bound", reference_size=1,
    )
    # This pair sits at the reference rung, so the lower bound on d puts it high.
    assert to_frame(lower)["correct_num"].iloc[0] == 1.0
    assert to_frame(upper)["correct_num"].iloc[0] == 0.0


def test_class_one_failures_are_never_scored():
    failure = _build_failure_record(
        _Spec(_item(), "a", ONE), "503 Service Unavailable"
    )
    outcomes = build_outcomes([failure, _good("b", 3)], [_item()], ONE)
    assert outcomes == [], "transient failures are replaced, not scored"


def test_unevaluable_outcomes_are_flagged_and_resolved():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    frame = to_frame(build_outcomes([failure, _good("b", 3)], [_item()], ONE))
    assert bool(frame["unevaluable"].iloc[0]) is True
    # Resolved, or the estimator's `resolved()` filter would drop exactly the
    # observations the composite outcome exists to retain.
    assert bool(frame["resolved"].iloc[0]) is True


def test_resume_does_not_count_a_transient_failure_as_complete(tmp_path):
    path = tmp_path / "calls.jsonl"
    transient = _build_failure_record(
        _Spec(_item(), "a", ONE), "connection aborted"
    )
    unevaluable = _build_failure_record(_Spec(_item(), "b", ONE), "truncated")
    third = _good("a", 2) | {"identity": "third"}
    with path.open("w", encoding="utf-8") as handle:
        for record in (transient, unevaluable, third):
            handle.write(json.dumps(record) + "\n")
    done = completed_identities(path)
    assert transient["identity"] not in done, "a transient error must be retried"
    assert unevaluable["identity"] in done, "a class-2 result is terminal"
    assert "third" in done
