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
    "read timeout",
    "generation timeout",
    "stream timeout",
)


def classify_failure(message: str) -> str:
    """Transient, unevaluable, or fatal. Unknown means fatal.

    Order matters: the transient markers are checked first, so a connect timeout
    is not caught by the generic timeout language in the unevaluable set.
    """
    lowered = message.lower()
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
