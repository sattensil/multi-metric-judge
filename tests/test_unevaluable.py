"""Three failure classes. Transient failures are replaced, unevaluable ones are
scored, and anything unrecognised aborts rather than becoming a quiet miss."""

from __future__ import annotations

import pytest

from packing.unevaluable import (
    FATAL,
    TRANSIENT,
    UNEVALUABLE,
    classify_failure,
    unevaluable_value,
)


@pytest.mark.parametrize(
    "message",
    [
        "Connection reset by peer",
        "connection aborted",
        "rate limit exceeded, retry after 2s",
        "provider returned 503 Service Unavailable",
        "502 Bad Gateway",
        "connect timeout after 5s",
    ],
)
def test_transient_infrastructure(message):
    assert classify_failure(message) == TRANSIENT


@pytest.mark.parametrize(
    "message",
    [
        "gpt-5-mini: truncated at max_completion_tokens=9352 for group_size=40",
        "content did not parse as JSON",
        "empty content returned",
        "context_length_exceeded: 41000 tokens > 32768",
        "content_filter triggered on the response",
        "model refused to answer",
        "read timeout while streaming the completion",
        "generation timeout after 120s",
    ],
)
def test_treatment_or_system_unevaluable(message):
    assert classify_failure(message) == UNEVALUABLE


@pytest.mark.parametrize(
    "message",
    [
        "invalid_request_error: unsupported parameter",
        "criterion key set mismatch: got {'a'} expected {'b'}",
        "unexpected KeyError('scores')",
        "TypeError: got an unexpected keyword argument",
        "something nobody anticipated",
    ],
)
def test_fatal_protocol_and_configuration(message):
    assert classify_failure(message) == FATAL


def test_unrecognised_is_fatal_not_unevaluable():
    # The conservative direction is to stop, not to score. Scoring an unknown
    # failure as a miss turns a harness bug into a load effect.
    assert classify_failure("???") == FATAL


def test_primary_rule_scores_unevaluable_as_zero_at_both_endpoints():
    assert unevaluable_value("primary", 1, 1) == 0.0
    assert unevaluable_value("primary", 40, 1) == 0.0


def test_lower_bound_imputes_oppositely_at_the_endpoints():
    # d = Y40 - Y1, so the lower bound on d needs Y1 high and Y40 low.
    assert unevaluable_value("lower_bound", 1, 1) == 1.0
    assert unevaluable_value("lower_bound", 40, 1) == 0.0


def test_upper_bound_imputes_the_other_way():
    assert unevaluable_value("upper_bound", 1, 1) == 0.0
    assert unevaluable_value("upper_bound", 40, 1) == 1.0


def test_complete_case_excludes_the_observation():
    assert unevaluable_value("complete_case", 40, 1) is None
    assert unevaluable_value("complete_case", 1, 1) is None


def test_a_single_constant_would_not_bound_the_difference():
    # Regression guard for the bug this replaced: assigning every unevaluable Y
    # the same value leaves d unchanged and bounds nothing.
    lower = unevaluable_value("lower_bound", 40, 1) - unevaluable_value(
        "lower_bound", 1, 1
    )
    upper = unevaluable_value("upper_bound", 40, 1) - unevaluable_value(
        "upper_bound", 1, 1
    )
    assert lower == -1.0
    assert upper == 1.0
    assert lower < upper


def test_intermediate_rungs_are_treated_as_non_reference():
    # Only the reference rung gets the reference imputation; every other rung is
    # the comparison side of its own contrast.
    assert unevaluable_value("lower_bound", 20, 1) == 0.0
    assert unevaluable_value("upper_bound", 20, 1) == 1.0


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown rule"):
        unevaluable_value("whatever", 40, 1)
