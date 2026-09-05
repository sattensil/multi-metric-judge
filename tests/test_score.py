"""Outcome derivation, tie handling, and QWK against hand-worked values."""

from __future__ import annotations

import math

import pytest

from packing.score import (
    accuracy,
    make_outcome,
    quadratic_weighted_kappa as qwk,
    sign,
    tie_rates,
)


def _outcome(human_a, human_b, judge_a, judge_b, **kw):
    defaults = dict(
        item_id="i1",
        criterion="helpfulness",
        group_size=1,
        position=0,
        nominal_n=1,
        replicate=0,
        model="test",
        partition_hash="deadbeef",
    )
    defaults.update(kw)
    return make_outcome(
        human_a=human_a, human_b=human_b, judge_a=judge_a, judge_b=judge_b, **defaults
    )


def test_sign() -> None:
    assert sign(3) == 1
    assert sign(-3) == -1
    assert sign(0) == 0


# ------------------------------------------------------------- correctness ---


def test_correct_when_orderings_agree() -> None:
    o = _outcome(human_a=4, human_b=1, judge_a=3, judge_b=2)
    assert o.correct is True
    assert o.credit == 1.0
    assert not o.human_tie and not o.judge_tie


def test_incorrect_when_orderings_disagree() -> None:
    o = _outcome(human_a=4, human_b=1, judge_a=1, judge_b=3)
    assert o.correct is False
    assert o.credit == 0.0


def test_magnitude_does_not_matter_only_ordering() -> None:
    """Pairwise accuracy is about ordering; a judge can be badly calibrated in
    absolute terms and still order correctly."""
    o = _outcome(human_a=4, human_b=3, judge_a=1, judge_b=0)
    assert o.correct is True
    assert o.abs_error == 3.0


# -------------------------------------------------------------------- ties ---


def test_human_tie_is_unscoreable_not_wrong() -> None:
    o = _outcome(human_a=2, human_b=2, judge_a=4, judge_b=0)
    assert o.human_tie is True
    assert o.correct is None, "a human tie has no correct answer"
    assert math.isnan(o.credit)


def test_judge_tie_is_indecision_not_error() -> None:
    o = _outcome(human_a=4, human_b=1, judge_a=2, judge_b=2)
    assert o.judge_tie is True
    assert o.correct is None, "judge indecision must not be scored as an error"
    assert o.credit == 0.5, "descriptive plots give chance credit"


def test_human_tie_takes_precedence_over_judge_tie() -> None:
    o = _outcome(human_a=2, human_b=2, judge_a=3, judge_b=3)
    assert o.human_tie and o.judge_tie
    assert math.isnan(o.credit), "unscoreable, not chance"


def test_abs_error_defined_even_when_pairwise_is_not() -> None:
    """So a criterion crippled by ties still has a usable secondary outcome."""
    o = _outcome(human_a=2, human_b=2, judge_a=4, judge_b=0)
    assert o.correct is None
    assert o.abs_error == 2.0


def test_accuracy_ignores_unresolved() -> None:
    outcomes = [
        _outcome(4, 1, 3, 2),  # correct
        _outcome(4, 1, 1, 3),  # incorrect
        _outcome(2, 2, 4, 0),  # human tie
        _outcome(4, 1, 2, 2),  # judge tie
    ]
    assert accuracy(outcomes) == 0.5


def test_accuracy_none_when_nothing_resolved() -> None:
    assert accuracy([_outcome(2, 2, 1, 1)]) is None


def test_tie_rates_partition_the_observations() -> None:
    outcomes = [
        _outcome(4, 1, 3, 2),
        _outcome(4, 1, 1, 3),
        _outcome(2, 2, 4, 0),
        _outcome(4, 1, 2, 2),
    ]
    rates = tie_rates(outcomes)["helpfulness"]
    assert rates["n"] == 4
    assert rates["human_tie_rate"] == 0.25
    # 3 scoreable after removing the human tie; 1 of them was a judge tie
    assert rates["judge_tie_rate_of_scoreable"] == pytest.approx(1 / 3)
    assert rates["resolved_n"] == 2
    assert rates["resolved_fraction"] == 0.5


# --------------------------------------------------------------------- QWK ---


def test_qwk_perfect_agreement() -> None:
    assert qwk([0, 1, 2], [0, 1, 2], 0, 2) == pytest.approx(1.0)


def test_qwk_perfect_inversion_is_minus_one() -> None:
    # Hand-worked: num=2, den=1 -> 1 - 2/1 = -1
    assert qwk([0, 2], [2, 0], 0, 2) == pytest.approx(-1.0)


def test_qwk_binary_agreement() -> None:
    assert qwk([0, 1], [0, 1], 0, 1) == pytest.approx(1.0)


def test_qwk_no_expected_disagreement_returns_one() -> None:
    """Degenerate: all raters gave the same value. Division by zero avoided."""
    assert qwk([2, 2, 2], [2, 2, 2], 0, 4) == 1.0


def test_qwk_penalises_distant_errors_more() -> None:
    """The point of the quadratic weighting."""
    near = qwk([0, 1, 2, 3], [0, 1, 2, 2], 0, 4)
    far = qwk([0, 1, 2, 4], [0, 1, 2, 0], 0, 4)
    assert near > far


def test_qwk_validates_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        qwk([1, 2], [1], 0, 4)
    with pytest.raises(ValueError, match="empty"):
        qwk([], [], 0, 4)
    with pytest.raises(ValueError, match="outside declared scale"):
        qwk([9], [1], 0, 4)
