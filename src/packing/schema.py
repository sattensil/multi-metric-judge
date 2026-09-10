"""Strict JSON schemas generated per partition group.

Scoring is POINTWISE: one call scores one response on one group of criteria.
Pairwise ground truth is then derived by comparing the two responses' scores.

Why pointwise rather than showing the judge both responses at once:

1. It matches how the labels were made. HelpSteer2 annotators scored each
   response independently; deriving a pairwise ordering from pointwise scores is
   exactly how the ground-truth ordering is constructed. A pairwise-presented
   judge is solving an easier task than the annotators were.
2. It matches production use. Judges in real pipelines score one output, not a
   pair.
3. It keeps the manipulation clean. A pairwise call would double the content the
   judge must hold per criterion, entangling group size with candidate count.

The cost is 2 calls per group instead of 1. In exchange the absolute scores are
usable for QWK as a secondary outcome, which a preference-only judge could not
give.

Strict-mode note: `minimum`/`maximum` are not reliably honoured by
schema-constrained decoding, so the allowed scores are expressed as an explicit
`enum`, which is.
"""

from __future__ import annotations

import json
from typing import Any

from .criteria import Criterion

# Field order is deliberate: evidence and reasoning precede the score so the
# model commits to a justification before committing to a number. Mandatory
# textual evidence is also a cited self-preference-bias mitigation.
_FIELD_ORDER = ("evidence", "reasoning", "score")


def criterion_schema(criterion: Criterion) -> dict[str, Any]:
    allowed = list(range(criterion.scale_min, criterion.scale_max + 1))
    return {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "string",
                "description": (
                    "A short verbatim quote from the response that justifies the "
                    "score. Use the empty string only if no span is relevant."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the score.",
            },
            "score": {
                "type": "integer",
                "enum": allowed,
                "description": (
                    f"Score for {criterion.label} on the {criterion.scale_min}-"
                    f"{criterion.scale_max} scale defined in the rubric."
                ),
            },
        },
        "required": list(_FIELD_ORDER),
        "additionalProperties": False,
    }


def group_schema(group: tuple[Criterion, ...]) -> dict[str, Any]:
    """Schema for one judge call covering ``group``.

    Every criterion in the group is a required key, so a call cannot silently
    skip one -- a dropped criterion would look like missing data rather than the
    schema violation it is.
    """
    if not group:
        raise ValueError("group must be non-empty")
    keys = [c.key for c in group]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate criteria in group: {keys}")

    return {
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "object",
                "properties": {c.key: criterion_schema(c) for c in group},
                "required": keys,
                "additionalProperties": False,
            }
        },
        "required": ["evaluations"],
        "additionalProperties": False,
    }


def response_format(group: tuple[Criterion, ...], name: str = "evaluation") -> dict[str, Any]:
    """OpenAI-style ``response_format`` payload for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": group_schema(group),
        },
    }


def schema_token_estimate(group: tuple[Criterion, ...]) -> int:
    """Rough token cost of the schema itself, recorded as a covariate.

    Schema size grows with group size by construction, so it is a candidate
    alternative explanation for any group-size effect and has to be measured
    rather than assumed away. ~4 chars/token is adequate for a covariate.
    """
    blob = json.dumps(group_schema(group), separators=(",", ":"))
    return max(1, len(blob) // 4)


def schema_token_upper_bound(group: tuple[Criterion, ...]) -> int:
    """Conservative authorization bound for the serialized response schema."""
    blob = json.dumps(group_schema(group), separators=(",", ":"))
    return max(schema_token_estimate(group), len(blob.encode("utf-8")) + 128)


def parse_evaluations(
    payload: dict[str, Any], group: tuple[Criterion, ...]
) -> dict[str, dict[str, Any]]:
    """Validate a judge payload against the group, raising on any deviation.

    Strict decoding should make this unreachable, but a silent shape change from
    a provider would otherwise corrupt the dataset invisibly.
    """
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict):
        raise ValueError(f"missing or malformed 'evaluations': {payload!r}")

    expected = {c.key for c in group}
    got = set(evaluations)
    if got != expected:
        raise ValueError(f"criterion mismatch: expected {sorted(expected)}, got {sorted(got)}")

    by_key = {c.key: c for c in group}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in evaluations.items():
        criterion = by_key[key]
        score = entry.get("score")
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"{key}: score must be an integer, got {score!r}")
        if not criterion.scale_min <= score <= criterion.scale_max:
            raise ValueError(
                f"{key}: score {score} outside "
                f"[{criterion.scale_min}, {criterion.scale_max}]"
            )
        out[key] = {
            "score": score,
            "reasoning": entry.get("reasoning", ""),
            "evidence": entry.get("evidence", ""),
        }
    return out
