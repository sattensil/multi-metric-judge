"""Judge prompt construction.

Design choices that matter for variance, given that temperature is not a usable
knob here (Miller shows lowering it can triple minimum variance and inject bias,
and it is rejected outright by current frontier Claude models):

- **Anchored scale descriptors.** Explicit meanings for the endpoints and midpoint
  do far more to stabilise scores than any sampling parameter.
- **Evidence before score.** The schema orders evidence and reasoning ahead of the
  number, so the model commits to a justification first. Also a cited
  self-preference-bias mitigation.
- **No cross-criterion commentary.** Each criterion is scored on its own terms.
  Without this instruction a packed call tends to produce a single blended
  impression spread across the fields, which would confound the manipulation with
  a prompt artifact.
- **Order is randomised upstream** by the partition sampler; this module simply
  renders the order it is given, so position is a clean covariate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .criteria import Criterion

SYSTEM = """\
You are an expert evaluator. You score a single AI response against a rubric.

Rules:
- Score each criterion independently, strictly on its own definition and scale. \
Do not let your impression of one criterion influence another.
- Every integer on the scale is available to you, and the scale is fully
described. Use whichever value fits, including values whose description is a
near miss -- do not restrict yourself to a subset of the scale.
- Quote a short verbatim span from the response as evidence before scoring.
- Judge only the response given. Do not speculate about responses you cannot see.
- Some criteria are descriptive rather than evaluative: for those, a higher score \
is not better, it only means more of the named property.
"""


@dataclass(frozen=True)
class JudgePrompt:
    system: str
    user: str
    group: tuple[Criterion, ...]


def _rubric_block(group: tuple[Criterion, ...]) -> str:
    return "\n\n".join(criterion.render() for criterion in group)


def build(
    *,
    group: tuple[Criterion, ...],
    prompt_text: str,
    response_text: str,
    source_label: str = "Conversation prompt",
) -> JudgePrompt:
    """Render one pointwise judge call for ``group``.

    ``group`` order is preserved exactly: it is the randomised order drawn by the
    partition sampler, and ``position`` in the analysis refers to this order.
    """
    if not group:
        raise ValueError("group must be non-empty")

    plural = "criterion" if len(group) == 1 else "criteria"
    user = f"""\
## {source_label}

{prompt_text.strip()}

## Response to evaluate

{response_text.strip()}

## Rubric ({len(group)} {plural})

{_rubric_block(group)}

## Task

Score the response on each of the {len(group)} {plural} above, independently. \
For each one, give a short verbatim quote as evidence, one or two sentences of \
reasoning, and then the integer score.
"""
    return JudgePrompt(system=SYSTEM, user=user, group=group)


def token_estimate(prompt: JudgePrompt) -> int:
    """~4 chars/token. Used for dry-run cost planning, not for billing."""
    return max(1, (len(prompt.system) + len(prompt.user)) // 4)


def token_upper_bound(prompt: JudgePrompt) -> int:
    """Conservative input authorization bound based on encoded request text.

    Byte length dominates token count for provider text tokenizers. The fixed
    allowance covers message-role and request framing that is not represented
    in the two prompt strings.
    """
    encoded = len(prompt.system.encode("utf-8")) + len(prompt.user.encode("utf-8"))
    return max(token_estimate(prompt), encoded + 256)
