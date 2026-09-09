"""Failure classification and the composite-outcome value for unevaluable output.

Three classes, because two conflate a harness bug with a treatment effect:

1. TRANSIENT: exogenous and content-independent. Retried, then the prompt is
   replaced from its own lane. Never scored, and never a completed identity on
   resume.
2. UNEVALUABLE: plausibly caused by the request itself, so part of the treatment.
   Retained and scored, because replacing it would condition the analysis on
   surviving high load and bias degradation downward.
3. FATAL: the harness or its configuration is wrong. The run aborts.

An unrecognised failure is FATAL. Defaulting the unknown case to UNEVALUABLE would
let an SDK change or an unsupported option arrive as evidence that load degraded
judging, which is the one error mode no amount of sample size fixes.

Timeouts split by phase rather than by keyword: a connection timeout is transient,
while a read or generation timeout is unevaluable, because generation time
plausibly rises with K.
"""

from __future__ import annotations

TRANSIENT = "transient_infrastructure"
UNEVALUABLE = "treatment_unevaluable"
FATAL = "fatal_protocol"


class FatalProtocolError(RuntimeError):
    """Raised to abort a run rather than score a harness failure."""


_TRANSIENT_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection error",
    "connect timeout",
    "connection timeout",
    "rate limit",
    "429",
    "500 ",
    "502",
    "503",
    "504",
    "service unavailable",
    "bad gateway",
    "temporarily unavailable",
    # A bare timeout of undisclosed phase, which is what the SDK raises after
    # exhausting its own retries ("Request timed out.").
    #
    # Class 1 on measured grounds rather than assumed ones. Latency on the K = 40
    # ladder has a rung-independent tail: at K = 1, where the median call takes
    # 3.2s, six of 386 calls still exceeded 90s and one reached 124.8s. A call
    # that spends two minutes on a single criterion is not generating for two
    # minutes, so the tail is transport, not load-induced generation.
    #
    # The consequence of getting this wrong is asymmetric. Scored as class 2 a
    # timeout becomes a recovery miss, and at the observed rate that would inject
    # false failures into roughly 5% of observations. Retrying and, if
    # persistent, replacing costs sample but does not corrupt the outcome.
    #
    # Phase-disclosed read and generation timeouts stay class 2, since those name
    # a phase where load plausibly matters.
    "timed out",
    "timeout",
    "apitimeouterror",
)

_UNEVALUABLE_MARKERS = (
    "truncated",
    "max_completion_tokens",
    "did not parse",
    "empty content",
    "context_length_exceeded",
    "context length",
    "content_filter",
    "content filter",
    "refused",
    "refusal",
)

# Checked before everything else. A phase-disclosed timeout names a phase where
# load plausibly matters, so it must not be swallowed by the bare timeout
# language in the transient list, which contains it as a substring.
_LOAD_PHASE_TIMEOUTS = (
    "read timeout",
    "generation timeout",
    "stream timeout",
)


def classify_failure(message: str) -> str:
    """Transient, unevaluable, or fatal. Unknown means fatal.

    Order matters, in three tiers. A phase-disclosed timeout is checked first,
    because the transient list carries the bare timeout language and contains it
    as a substring. Then transient, then the remaining unevaluable markers, then
    fatal for anything unrecognised.
    """
    lowered = message.lower()
    if any(marker in lowered for marker in _LOAD_PHASE_TIMEOUTS):
        return UNEVALUABLE
    if any(marker in lowered for marker in _TRANSIENT_MARKERS):
        return TRANSIENT
    if any(marker in lowered for marker in _UNEVALUABLE_MARKERS):
        return UNEVALUABLE
    return FATAL


def unevaluable_value(
    rule: str, group_size: int, reference_size: int
) -> float | None:
    """Outcome value for an unevaluable paired unit at one rung.

    The estimand is a difference, d = Y_cmp - Y_ref, so bounding it requires
    imputing oppositely at the two endpoints. A single constant for every
    unevaluable Y leaves d untouched and bounds nothing, which is why this
    signature takes the rung: the rule has to know which side of the contrast it
    is scoring.

    `primary` is not an imputation. Y is defined as demonstrated,
    machine-readable directional recovery, so an unreadable judgment is a 0 by
    definition rather than a missing value filled in conservatively.
    """
    at_reference = group_size == reference_size
    if rule == "primary":
        return 0.0
    if rule == "lower_bound":
        return 1.0 if at_reference else 0.0
    if rule == "upper_bound":
        return 0.0 if at_reference else 1.0
    if rule == "complete_case":
        return None
    raise ValueError(
        f"unknown rule {rule!r}; expected primary, lower_bound, upper_bound "
        "or complete_case"
    )
