"""Output-token budgeting.

THE main threat to validity. If a 5-criterion call must emit 5 scores and 5
rationales under the same ``max_tokens`` as a 1-criterion call, accuracy will
drop for a completely uninteresting reason and the study will have measured its
own token ceiling rather than anything about attention or packing.

So the invariant is: **output budget per criterion is constant across group
sizes.** ``tests/test_budget.py`` pins it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Enough for a short verbatim quote, one or two sentences of reasoning, and an
# integer, with slack. Applied per criterion, never per call.
PER_CRITERION_OUTPUT_TOKENS = 220

# Fixed overhead for the JSON envelope itself, independent of group size.
ENVELOPE_OVERHEAD_TOKENS = 40

# Headroom for providers that bill reasoning tokens against the same ceiling.
REASONING_HEADROOM_TOKENS = 512


@dataclass(frozen=True)
class Budget:
    group_size: int
    max_output_tokens: int
    per_criterion_tokens: int
    reasoning_headroom: int

    @property
    def content_budget(self) -> int:
        """Budget available for the JSON payload, excluding reasoning headroom."""
        return self.max_output_tokens - self.reasoning_headroom


def budget_for(group_size: int, *, reasoning: bool = True) -> Budget:
    """Output budget for a call covering ``group_size`` criteria.

    Linear in group size plus a constant envelope, so per-criterion budget is
    invariant. The reasoning headroom is also constant -- it is per call, not per
    criterion, because it is the model's scratch space and does not scale with
    the number of fields emitted.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    headroom = REASONING_HEADROOM_TOKENS if reasoning else 0
    content = ENVELOPE_OVERHEAD_TOKENS + PER_CRITERION_OUTPUT_TOKENS * group_size
    return Budget(
        group_size=group_size,
        max_output_tokens=content + headroom,
        per_criterion_tokens=PER_CRITERION_OUTPUT_TOKENS,
        reasoning_headroom=headroom,
    )


def near_cap(
    group_size: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    *,
    threshold: float = 0.9,
    reasoning: bool = True,
) -> bool:
    """Did this call come close to exhausting its *content* budget?

    Feeds the sensitivity analysis: if the group-size effect exists only among
    calls that were near the ceiling, it is a token artifact rather than a
    property of packing. That makes the flag's calibration load-bearing.

    Subtlety that produced a real bug: providers report ``output_tokens`` as
    *total* completion tokens, reasoning included, while ``content_budget``
    deliberately excludes the reasoning headroom. Comparing the two directly
    mixes a reasoning-inclusive numerator with a reasoning-exclusive denominator
    and over-flags badly -- it marked 25% of group_size=1 calls as near-cap when
    their true content use was ~44% of budget. Reasoning tokens are therefore
    subtracted here.
    """
    budget = budget_for(group_size, reasoning=reasoning)
    content_tokens = max(0, output_tokens - max(0, reasoning_tokens))
    return content_tokens >= threshold * budget.content_budget
