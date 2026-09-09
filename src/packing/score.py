"""Deriving outcomes from judge scores and human labels.

The primary outcome is pairwise: did the judge order the two responses on this
criterion the way the humans did? That requires care in two places.

Human ties
----------
When both responses share a human score on a criterion, there is no correct
pairwise answer. Those observations are **excluded** and the human tie rate is
reported per criterion. Tie rates differ by criterion (a descriptive dimension
like verbosity plausibly ties more often than helpfulness), which makes the
usable design unbalanced -- worth knowing before the confirmatory run rather
than after.

Judge ties
----------
When the judge gives both responses the same score, it has expressed no
preference. This is *not* symmetric with a human tie, and how it is handled
matters a great deal:

- Counting it as incorrect is unduly harsh and conflates indecision with error.
- Excluding it is dangerous, because the judge tie rate may itself vary with
  group size. A rushed judge in a packed call plausibly produces flatter, more
  clustered scores. Excluding those observations would then bias the primary
  contrast in exactly the direction the study is looking for -- a self-fulfilling
  result.

So judge ties are handled two ways at once:

1. ``correct`` is left as None for judge ties, and the binary model is fitted on
   resolved observations only.
2. ``judge_tie`` is a **co-primary outcome**, modelled with the same design. If
   the tie rate rises with group size, that is itself degradation and must be
   reported alongside the accuracy curve, not hidden by the exclusion.

For descriptive plots only, ``credit`` scores a tie as 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


@dataclass(frozen=True)
class Outcome:
    """One (item, criterion) observation."""

    item_id: str
    criterion: str
    group_size: int
    position: int
    nominal_n: int
    replicate: int
    model: str
    partition_hash: str

    human_a: int
    human_b: int
    judge_a: int
    judge_b: int

    human_tie: bool
    judge_tie: bool
    correct: bool | None
    credit: float
    abs_error: float

    output_tokens: int = 0
    schema_tokens: int = 0
    near_cap: bool = False
    unevaluable: bool = False
    """True when the judge returned nothing readable and Y is 0 by definition.

    Y is demonstrated, machine-readable directional recovery, so an unreadable
    judgment scores 0 as part of the outcome definition rather than as an
    imputation. The flag exists so the frequency can be reported by group size,
    which is its own diagnostic: a judge that stops returning usable output under
    load is degrading in a way the recovery rate alone will not show.
    """

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def make_outcome(
    *,
    item_id: str,
    criterion: str,
    group_size: int,
    position: int,
    nominal_n: int,
    replicate: int,
    model: str,
    partition_hash: str,
    human_a: int,
    human_b: int,
    judge_a: int,
    judge_b: int,
    output_tokens: int = 0,
    schema_tokens: int = 0,
    near_cap: bool = False,
) -> Outcome:
    human_delta = sign(human_a - human_b)
    judge_delta = sign(judge_a - judge_b)
    human_tie = human_delta == 0
    judge_tie = judge_delta == 0

    if human_tie:
        correct: bool | None = None
        credit = float("nan")
    elif judge_tie:
        correct = None
        credit = 0.5
    else:
        correct = judge_delta == human_delta
        credit = 1.0 if correct else 0.0

    # Absolute error is defined even when the pairwise outcome is not, so a
    # criterion with a crippling tie rate still has a usable secondary outcome.
    abs_error = (abs(judge_a - human_a) + abs(judge_b - human_b)) / 2

    return Outcome(
        item_id=item_id,
        criterion=criterion,
        group_size=group_size,
        position=position,
        nominal_n=nominal_n,
        replicate=replicate,
        model=model,
        partition_hash=partition_hash,
        human_a=human_a,
        human_b=human_b,
        judge_a=judge_a,
        judge_b=judge_b,
        human_tie=human_tie,
        judge_tie=judge_tie,
        correct=correct,
        credit=credit,
        abs_error=abs_error,
        output_tokens=output_tokens,
        schema_tokens=schema_tokens,
        near_cap=near_cap,
    )


# ------------------------------------------------------------------ rates ----


def tie_rates(outcomes: Iterable[Outcome]) -> dict[str, dict[str, float]]:
    """Human and judge tie rates per criterion, plus resolved n.

    Gate 2 of the study: an extreme human tie rate on a criterion means the
    pairwise outcome is unusable there and it should fall back to QWK.
    """
    buckets: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        c = counts.setdefault(
            outcome.criterion, {"n": 0, "human_tie": 0, "judge_tie": 0, "resolved": 0}
        )
        c["n"] += 1
        if outcome.human_tie:
            c["human_tie"] += 1
            continue
        if outcome.judge_tie:
            c["judge_tie"] += 1
            continue
        c["resolved"] += 1

    for criterion, c in counts.items():
        n = c["n"] or 1
        scoreable = c["n"] - c["human_tie"] or 1
        buckets[criterion] = {
            "n": c["n"],
            "human_tie_rate": c["human_tie"] / n,
            "judge_tie_rate_of_scoreable": c["judge_tie"] / scoreable,
            "resolved_n": c["resolved"],
            "resolved_fraction": c["resolved"] / n,
        }
    return buckets


def accuracy(outcomes: Iterable[Outcome]) -> float | None:
    """Pairwise accuracy over resolved observations only."""
    resolved = [o.correct for o in outcomes if o.correct is not None]
    if not resolved:
        return None
    return sum(resolved) / len(resolved)


def quadratic_weighted_kappa(
    judge: list[int], human: list[int], scale_min: int, scale_max: int
) -> float:
    """QWK, the standard ordinal-agreement statistic.

    Secondary outcome, and the fallback primary for any criterion whose human tie
    rate makes the pairwise outcome unusable.
    """
    if len(judge) != len(human):
        raise ValueError("judge and human must be the same length")
    if not judge:
        raise ValueError("empty input")

    points = list(range(scale_min, scale_max + 1))
    index = {v: i for i, v in enumerate(points)}
    k = len(points)
    if any(v not in index for v in judge + human):
        raise ValueError("value outside declared scale")

    observed = [[0] * k for _ in range(k)]
    for j, h in zip(judge, human):
        observed[index[j]][index[h]] += 1

    n = len(judge)
    judge_marginal = [sum(row) for row in observed]
    human_marginal = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    denom_weight = (k - 1) ** 2
    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / denom_weight
            expected = judge_marginal[i] * human_marginal[j] / n
            num += weight * observed[i][j]
            den += weight * expected

    if den == 0:
        # No expected disagreement: perfect agreement by construction.
        return 1.0
    return 1.0 - num / den
