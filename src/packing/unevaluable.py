"""Failure classification and the composite-outcome value for unevaluable output.

Four ordered tiers, because fewer conflate a harness bug with a treatment effect:

0. FATAL, terminal billing: an exhausted balance arrives as 429 and would
   otherwise be caught by the rate-limit marker and retried forever. Checked
   first, matched on the error code rather than the status.
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
    # A refused connection never reached the model, so it cannot be a judgement
    # about the content. Listed here explicitly because "refused" is also a
    # refusal marker below, and substring matching would otherwise score a TCP
    # rejection as the judge declining to answer.
    "connection refused",
    "econnrefused",
    "refused the connection",
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
    "non-json content",
    "empty content",
    "empty response",
    "no tool_use block",
    "malformed model output",
    "context_length_exceeded",
    "context length",
    "content_filter",
    "content filter",
    # Narrowed from a bare "refused": that matched transport language such as
    # "Connection refused" and turned a failure to reach the provider into a
    # scored recovery miss.
    "model refused",
    "refused to answer",
    "refusal",
)

# Checked first, because these arrive as HTTP 429 and the transient list matches
# on "429" for rate limits. A rate limit and an exhausted balance are two
# conditions wearing the same status code: one clears on its own, the other never
# does. Retrying an exhausted balance cannot succeed, so it is fatal and the run
# stops instead of draining the failure budget against a wall.
#
# Matched on the provider's error code and type fields rather than the status,
# since the status is the part that does not distinguish them.
_TERMINAL_BILLING = (
    "credit_balance_exhausted",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "no credits remaining",
    "exceeded your current quota",
)

# Checked before the transient markers. A phase-disclosed timeout names a phase where
# load plausibly matters, so it must not be swallowed by the bare timeout
# language in the transient list, which contains it as a substring.
#
# "read timeout" is deliberately absent. It is the standard transport-level name
# in urllib3 and requests, not a statement about generation, and including it
# made classification depend on incidental phrasing rather than on the event:
# "Request timed out." classified transient while the same underlying failure
# rendered as "Read timed out. (read timeout=90)" classified as scoreable. Only
# names that unambiguously identify generation belong here, and the asymmetry
# argument above resolves the doubtful cases toward transient: a wrongly
# transient call costs sample, a wrongly scored one injects a false miss.
_LOAD_PHASE_TIMEOUTS = (
    "generation timeout",
    "stream timeout",
)


def classify_failure(message: str) -> str:
    """Transient, unevaluable, or fatal. Unknown means fatal.

    Order matters, in four tiers. Terminal billing conditions come first, because
    they arrive as 429 and would otherwise be caught by the rate-limit marker and
    retried forever. Then a phase-disclosed timeout is checked,
    because the transient list carries the bare timeout language and contains it
    as a substring. Then transient, then the remaining unevaluable markers, then
    fatal for anything unrecognised.
    """
    lowered = message.lower()
    if any(marker in lowered for marker in _TERMINAL_BILLING):
        return FATAL
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
