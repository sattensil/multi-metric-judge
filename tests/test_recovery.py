"""End-to-end validation: can the estimators recover a KNOWN planted effect?

This is the test that makes the analysis code trustworthy. A mock judge is given
an explicit degradation rate as a function of group size, the full pipeline is
run offline, and the estimators are checked for:

  - power    -- they detect a planted effect
  - calibration -- they do NOT detect an effect that was not planted

Without both halves, a plausible-looking curve out of the real run would be
unfalsifiable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packing.analyze import (
    build_outcomes,
    descriptives,
    functional_form,
    paired_ladder,
    to_frame,
)
from packing.criteria import get_criteria
from packing.dataset import PairedItem
from packing.design import parse_replicate_spec, suggest_replicates
from packing.partition import nominal_levels
from packing.providers.mock_provider import MockJudge
from packing.runner import execute, plan, read_records

CRITERIA = get_criteria("helpsteer2")
K = len(CRITERIA)


def _items(n: int, seed: int = 0) -> list[PairedItem]:
    """Synthetic items with a clear, non-tied human ordering on every criterion."""
    import random

    rng = random.Random(seed)
    items = []
    for i in range(n):
        human_a, human_b = {}, {}
        for criterion in CRITERIA:
            # Guarantee a strict ordering so the pairwise outcome is defined,
            # and vary which side wins so a constant-guessing judge scores 0.5.
            high, low = rng.sample([4, 3, 2, 1, 0], 2)
            if rng.random() < 0.5:
                human_a[criterion.key], human_b[criterion.key] = high, low
            else:
                human_a[criterion.key], human_b[criterion.key] = low, high
        items.append(
            PairedItem(
                item_id=f"syn-{i:04d}",
                prompt=f"Synthetic prompt number {i}. Explain something.",
                response_a=f"Response A for item {i}. " * 5,
                response_b=f"Response B for item {i}. " * 5,
                human_a=human_a,
                human_b=human_b,
                dataset="synthetic",
                meta={},
            )
        )
    return items


def _truth_fn(items: list[PairedItem]):
    """A perfect judge: returns the human label for whichever response it sees.

    Responses are identified by their text, which is unique per (item, side).
    """
    lookup: dict[str, dict[str, int]] = {}
    for item in items:
        lookup[item.response_a.strip()] = item.human_a
        lookup[item.response_b.strip()] = item.human_b

    def truth(prompt, criterion):
        for text, labels in lookup.items():
            if text in prompt.user:
                return labels[criterion.key]
        raise AssertionError("response text not found in prompt")

    return truth


def _run(tmp_path: Path, items, noise_per_extra: float, tag: str):
    replicates = suggest_replicates(K, 7)
    provider = MockJudge(
        seed=11,
        base_noise=0.10,
        noise_per_extra_criterion=noise_per_extra,
        truth_fn=_truth_fn(items),
    )
    out = tmp_path / f"{tag}.jsonl"
    specs = []
    for level, count in replicates.items():
        specs.extend(
            plan(items, CRITERIA, model="mock", replicates=count, levels=(level,))
        )
    # Deduplicate across levels, as the runner's identity rule does.
    seen, unique = set(), []
    for spec in specs:
        if spec.identity in seen:
            continue
        seen.add(spec.identity)
        unique.append(spec)

    stats = execute(unique, provider, out)
    assert stats.failed == 0
    frame = to_frame(build_outcomes(read_records(out), items, CRITERIA))
    assert not frame.empty
    return frame


# --------------------------------------------------------------- power -------


def test_recovers_planted_degradation(tmp_path: Path) -> None:
    items = _items(120, seed=1)
    frame = _run(tmp_path, items, noise_per_extra=0.14, tag="planted")

    desc = descriptives(frame)["by_group_size"]
    acc = desc["accuracy"]
    assert acc.loc[1] > acc.loc[K], (
        f"planted effect not visible in marginals: {acc.to_dict()}"
    )

    ladder = paired_ladder(frame, reference_size=1)
    assert not ladder.empty
    largest = ladder[ladder["comparison_size"] == K].iloc[0]
    assert largest["mean_difference"] < 0, "paired contrast has the wrong sign"
    assert largest["p_holm"] < 0.05, f"paired contrast underpowered: {largest.to_dict()}"

    form = functional_form(frame)
    assert form["linear_slope"] < 0
    assert form["linear_p"] < 0.05
    assert form["prefer"] == "not selected"
    assert form["lrt_factor_vs_linear"]["interpretation"] == (
        "descriptive only; ignores item clustering"
    )


# ---------------------------------------------------------- calibration ------


def test_no_false_positive_when_nothing_planted(tmp_path: Path) -> None:
    """The more important half: a flat judge must not produce a significant curve."""
    items = _items(120, seed=2)
    frame = _run(tmp_path, items, noise_per_extra=0.0, tag="flat")

    ladder = paired_ladder(frame, reference_size=1)
    assert not ladder.empty
    assert (ladder["p_holm"] > 0.05).all(), (
        f"false positive on a flat judge: {ladder.to_dict(orient='records')}"
    )

    form = functional_form(frame)
    assert form["linear_p"] > 0.05, f"spurious slope: {form}"


def test_perfect_judge_scores_one(tmp_path: Path) -> None:
    """Sanity: zero noise and a truthful judge must give accuracy 1.0 everywhere."""
    items = _items(30, seed=3)
    provider = MockJudge(seed=5, base_noise=0.0, truth_fn=_truth_fn(items))
    out = tmp_path / "perfect.jsonl"
    specs = plan(items, CRITERIA, model="mock", replicates=1)
    execute(specs, provider, out)
    frame = to_frame(build_outcomes(read_records(out), items, CRITERIA))
    desc = descriptives(frame)["by_group_size"]
    assert (desc["accuracy"] == 1.0).all(), desc.to_dict()


def test_resume_is_idempotent(tmp_path: Path) -> None:
    """A second pass must do no work and must not duplicate observations."""
    items = _items(20, seed=4)
    provider = MockJudge(seed=6, base_noise=0.1, truth_fn=_truth_fn(items))
    out = tmp_path / "resume.jsonl"
    specs = plan(items, CRITERIA, model="mock", replicates=1)

    first = execute(specs, provider, out)
    assert first.written > 0 and first.skipped == 0
    n_lines = sum(1 for _ in out.open())

    calls_after_first = provider.calls
    second = execute(specs, provider, out)
    assert second.written == 0
    assert second.skipped == len(specs)
    assert provider.calls == calls_after_first, "resume re-called the provider"
    assert sum(1 for _ in out.open()) == n_lines


def test_replicate_spec_parsing() -> None:
    levels = nominal_levels(K)
    assert parse_replicate_spec("3:4,5:2", levels)[3] == 4
    assert parse_replicate_spec("", levels) == {n: 1 for n in levels}
    with pytest.raises(ValueError):
        parse_replicate_spec("9:2", levels)
    with pytest.raises(ValueError):
        parse_replicate_spec("3", levels)
