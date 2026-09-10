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
        "generation timeout after 120s",
        "gpt-5-mini: non-JSON content: Expecting value",
        "gemini-2.5-flash: empty response",
        "claude-haiku: no tool_use block (end_turn)",
        "gpt-5-mini: malformed model output: criterion mismatch",
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


@pytest.mark.parametrize(
    "message",
    [
        "Request timed out.",
        "request timed out",
        "APITimeoutError()",
        "unexpected APITimeoutError()",
        # The transport-level name for the same event, which urllib3 and
        # requests both use. Classifying by phrasing rather than by event put
        # these two in different tiers.
        "Read timed out.",
        "HTTPSConnectionPool: Read timed out. (read timeout=90)",
    ],
)
def test_a_bare_timeout_is_transient_never_fatal(message):
    """The SDK raises a bare timeout after exhausting its own retries.

    Classified FATAL it would abort the run and discard every completed call's
    progress over a network hiccup. Class 1 rather than class 2 on measured
    grounds: the observed latency tail is rung-independent, appearing at K = 1
    where the median call is 3.2s, so it is transport rather than load-induced
    generation. Scoring it as a recovery miss would inject false failures into
    roughly 5% of observations at the observed timeout rate.
    """
    assert classify_failure(message) == TRANSIENT
    assert classify_failure(message) != FATAL


def test_a_disclosed_connect_timeout_is_still_transient():
    # Phase-specific messages keep their phase-based class: the transient
    # markers are checked before the generic timeout language.
    assert classify_failure("connect timeout after 5s") == TRANSIENT
    assert classify_failure("connection timeout") == TRANSIENT


def test_a_phase_disclosed_timeout_outranks_the_bare_form():
    # The transient list carries the bare timeout language and contains these as
    # substrings, so precedence, not membership, is what keeps them class 2.
    assert classify_failure("generation timeout after 120s") == UNEVALUABLE
    assert classify_failure("stream timeout") == UNEVALUABLE


def test_a_transport_read_timeout_is_not_a_disclosed_generation_phase():
    """Only names that unambiguously mean generation may be scored.

    "read timeout" is the transport-level name in urllib3 and requests, so
    treating it as phase-disclosed made the class depend on which library
    formatted the message: the same failure classified transient as "Request
    timed out." and scoreable as "Read timed out. (read timeout=90)".
    """
    for message in (
        "Read timed out.",
        "requests.exceptions.ReadTimeout: Read timed out. (read timeout=90)",
        "read timeout while streaming",
    ):
        assert classify_failure(message) == TRANSIENT, message


@pytest.mark.parametrize(
    "message",
    [
        "Connection refused",
        "[Errno 61] Connection refused",
        "ECONNREFUSED 127.0.0.1:443",
        "The server refused the connection",
    ],
)
def test_a_refused_connection_is_transport_not_a_refusal(message):
    """A connection that never reached the model cannot judge its content.

    "refused" was a bare refusal marker, so substring matching scored a TCP
    rejection as the judge declining to answer, which is a recovery miss
    fabricated out of a network condition.
    """
    assert classify_failure(message) == TRANSIENT
    assert classify_failure(message) != UNEVALUABLE


def test_a_genuine_model_refusal_is_still_scoreable():
    # Narrowing the refusal markers must not lose the case they exist for.
    assert classify_failure("model refused to answer") == UNEVALUABLE
    assert classify_failure("the model refused to comply") == UNEVALUABLE
    assert classify_failure("response contained a refusal") == UNEVALUABLE


REAL_CREDIT_ERROR = (
    "gpt-5-mini: request failed: Error code: 429 - {'error': {'message': "
    "'You have no credits remaining. Add credits to continue using the API at "
    "https://platform.openai.com/settings/organization/billing/.', 'type': "
    "'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}"
)


def test_an_exhausted_balance_is_fatal_not_transient():
    """A 429 is two conditions wearing one status code.

    A rate limit clears on its own; an exhausted balance never does. Classified
    transient it is retried indefinitely while making no progress, which is what
    happened on the K = 40 run: 414 of 444 failures were this, every one marked
    retryable, and the run kept issuing requests against a zero balance.
    """
    assert classify_failure(REAL_CREDIT_ERROR) == FATAL
    assert classify_failure(REAL_CREDIT_ERROR) != TRANSIENT


@pytest.mark.parametrize(
    "message",
    [
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "You have no credits remaining",
        "You exceeded your current quota",
    ],
)
def test_terminal_billing_conditions_are_fatal(message):
    assert classify_failure(message) == FATAL


def test_a_genuine_rate_limit_is_still_transient():
    # The distinguishing signal is the error code, not the 429 status, so both
    # of these have to keep working after the billing tier is added.
    assert classify_failure("Error code: 429 - rate limit exceeded") == TRANSIENT
    assert classify_failure("429 Too Many Requests, retry after 2s") == TRANSIENT
