"""Schema strictness and round-trip tests."""

from __future__ import annotations

import pytest

from packing.criteria import HELPSTEER2_CRITERIA, SUMMEVAL_CRITERIA
from packing.partition import make_partition
from packing.schema import (
    group_schema,
    parse_evaluations,
    response_format,
    schema_token_estimate,
)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_every_object_is_closed_and_fully_required() -> None:
    """Strict structured output requires additionalProperties:false and that
    every declared property is required, at every level."""
    schema = group_schema(HELPSTEER2_CRITERIA)
    for node in _walk(schema):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node.get("properties", {}))


def test_all_criteria_required_so_none_can_be_skipped() -> None:
    for nominal_n in range(1, len(HELPSTEER2_CRITERIA) + 1):
        part = make_partition(HELPSTEER2_CRITERIA, nominal_n, seed=3)
        for group in part.groups:
            schema = group_schema(group)
            evaluations = schema["properties"]["evaluations"]
            assert set(evaluations["required"]) == {c.key for c in group}


def test_score_enum_matches_criterion_scale() -> None:
    schema = group_schema(SUMMEVAL_CRITERIA)
    for criterion in SUMMEVAL_CRITERIA:
        enum = schema["properties"]["evaluations"]["properties"][criterion.key][
            "properties"
        ]["score"]["enum"]
        assert enum == list(range(criterion.scale_min, criterion.scale_max + 1))
        assert enum == [1, 2, 3, 4, 5]

    schema = group_schema(HELPSTEER2_CRITERIA)
    enum = schema["properties"]["evaluations"]["properties"]["helpfulness"][
        "properties"
    ]["score"]["enum"]
    assert enum == [0, 1, 2, 3, 4]


def test_schema_token_estimate_grows_with_group_size() -> None:
    sizes = [
        schema_token_estimate(HELPSTEER2_CRITERIA[:k])
        for k in range(1, len(HELPSTEER2_CRITERIA) + 1)
    ]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_response_format_is_strict() -> None:
    payload = response_format(HELPSTEER2_CRITERIA[:2])
    assert payload["type"] == "json_schema"
    assert payload["json_schema"]["strict"] is True


def test_group_schema_rejects_bad_groups() -> None:
    with pytest.raises(ValueError):
        group_schema(())
    with pytest.raises(ValueError):
        group_schema((HELPSTEER2_CRITERIA[0], HELPSTEER2_CRITERIA[0]))


# ---------------------------------------------------------------- parse ------


def _payload(group, score=2):
    return {
        "evaluations": {
            c.key: {"evidence": "quote", "reasoning": "because", "score": score}
            for c in group
        }
    }


def test_parse_round_trip() -> None:
    group = HELPSTEER2_CRITERIA[:3]
    parsed = parse_evaluations(_payload(group), group)
    assert set(parsed) == {c.key for c in group}
    assert all(entry["score"] == 2 for entry in parsed.values())


def test_parse_rejects_missing_criterion() -> None:
    group = HELPSTEER2_CRITERIA[:3]
    payload = _payload(group)
    del payload["evaluations"][group[0].key]
    with pytest.raises(ValueError, match="criterion mismatch"):
        parse_evaluations(payload, group)


def test_parse_rejects_extra_criterion() -> None:
    group = HELPSTEER2_CRITERIA[:2]
    payload = _payload(group)
    payload["evaluations"]["unexpected"] = {
        "evidence": "",
        "reasoning": "",
        "score": 1,
    }
    with pytest.raises(ValueError, match="criterion mismatch"):
        parse_evaluations(payload, group)


def test_parse_rejects_out_of_range_and_non_integer_scores() -> None:
    group = HELPSTEER2_CRITERIA[:1]
    with pytest.raises(ValueError, match="outside"):
        parse_evaluations(_payload(group, score=9), group)
    payload = _payload(group)
    payload["evaluations"][group[0].key]["score"] = "3"
    with pytest.raises(ValueError, match="must be an integer"):
        parse_evaluations(payload, group)
    # bool is an int subclass in Python; must not slip through
    payload["evaluations"][group[0].key]["score"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        parse_evaluations(payload, group)


def test_parse_rejects_malformed_envelope() -> None:
    with pytest.raises(ValueError, match="evaluations"):
        parse_evaluations({}, HELPSTEER2_CRITERIA[:1])
