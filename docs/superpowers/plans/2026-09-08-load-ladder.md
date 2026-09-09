# Fixed-Position Load Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the experiment machinery for a K = 40 load ladder in which the targeted criterion is pinned at position 0, so load varies without the position confound that made the K = 20 sweep unadjudicable.

**Architecture:** Five new focused modules plus three surgical extensions to existing ones. A criterion manifest (40 criteria, validated and hashed) feeds an eligibility pre-pass, which feeds a frozen-seed lane assignment, which feeds a fixed-position nested planner. Analysis adds a stratified equal-weight estimator with Welch-Satterthwaite degrees of freedom and an explicit unevaluable-output scoring rule. Nothing calls a provider until Task 9, which is dry-run only.

**Tech Stack:** Python 3.12, `uv`, `pytest`, `pandas`, `numpy`, `scipy`, `click`. OpenAI via strict Structured Outputs.

**Spec:** `docs/superpowers/specs/2026-09-08-load-ladder-design.md`

## Global Constraints

- No outcome-generating provider call in any task. Task 9 is `--dry-run` only. Spending is a separate decision made after the manifest-based cost recompute.
- Every criterion anchors **every** scale point. `Criterion.__post_init__` (`src/packing/criteria.py:38-52`) already raises otherwise; do not weaken it.
- No `temperature`. `reasoning_effort` fixed. Both already enforced in `src/packing/providers/openai_provider.py:61-72`.
- Determinism: every random choice derives from `partition_seed(...)` (`src/packing/partition.py:150-158`) or an explicitly passed seed. No bare `random` / `np.random` calls without a seed.
- Arm assignment happens **before** eligibility is inspected. Reversing this order silently changes the core population (spec §5).
- One corruption realisation per prompt, frozen and reused at every rung.
- No em dashes in any prose, docstring, or comment.
- Run `uv run pytest -q` before every commit. All 175 existing tests must stay green.

---

### Task 1: Criterion manifest with validation and a frozen hash

**Files:**
- Create: `src/packing/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `Criterion` from `src/packing/criteria.py:27`
- Produces: `CriterionManifest` dataclass with fields `criteria: tuple[Criterion, ...]`; properties `.keys -> tuple[str, ...]`, `.hash -> str`, `.length_stats -> dict[str, int]`; method `.validate(expect: int) -> None`. Module function `manifest_for(criteria: tuple[Criterion, ...]) -> CriterionManifest`.

- [ ] **Step 1: Write the failing test**

```python
"""Manifest invariants: the criterion pool is part of the treatment, so it is
validated and hashed rather than assumed."""

from __future__ import annotations

import pytest

from packing.criteria import Criterion
from packing.manifest import CriterionManifest, manifest_for


def _criterion(key: str) -> Criterion:
    return Criterion(
        key=key,
        label=key.title(),
        description=f"Test criterion {key}.",
        anchors={0: "none", 1: "poor", 2: "fair", 3: "good", 4: "full"},
    )


def test_hash_is_stable_and_order_sensitive():
    a = manifest_for((_criterion("x"), _criterion("y")))
    b = manifest_for((_criterion("x"), _criterion("y")))
    c = manifest_for((_criterion("y"), _criterion("x")))
    assert a.hash == b.hash
    assert a.hash != c.hash, "ordering is part of the manifest identity"
    assert len(a.hash) == 16


def test_validate_rejects_wrong_count():
    m = manifest_for((_criterion("x"),))
    with pytest.raises(ValueError, match="expected 40 criteria, found 1"):
        m.validate(expect=40)


def test_validate_rejects_duplicate_keys():
    m = CriterionManifest(criteria=(_criterion("x"), _criterion("x")))
    with pytest.raises(ValueError, match="duplicate criterion keys"):
        m.validate(expect=2)


def test_validate_accepts_a_well_formed_pool():
    m = manifest_for(tuple(_criterion(f"c{i}") for i in range(40)))
    m.validate(expect=40)


def test_length_stats_reports_the_distribution():
    m = manifest_for((_criterion("x"), _criterion("yy")))
    stats = m.length_stats
    assert set(stats) == {"min", "max", "mean", "total"}
    assert stats["max"] >= stats["min"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manifest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
"""The criterion pool as a validated, hashed artefact.

The pool is part of the experimental treatment, not test data: criterion count,
anchor completeness and prose length all determine input-token growth and what a
judge is actually asked to do. So it is validated on construction and hashed, and
the hash is recorded with every run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .criteria import Criterion


@dataclass(frozen=True)
class CriterionManifest:
    criteria: tuple[Criterion, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.criteria)

    @property
    def hash(self) -> str:
        """Stable identity of the pool, ordering included.

        Ordering matters because the ladder draws companions as a prefix of a
        seeded permutation of this tuple, so two pools with the same members in a
        different order are different instruments.
        """
        payload = json.dumps(
            [
                {
                    "key": c.key,
                    "label": c.label,
                    "description": c.description,
                    "anchors": {str(k): v for k, v in sorted(c.anchors.items())},
                    "scale": [c.scale_min, c.scale_max],
                    "evaluative": c.evaluative,
                }
                for c in self.criteria
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def length_stats(self) -> dict[str, int]:
        lengths = [
            len(c.description) + sum(len(v) for v in c.anchors.values())
            for c in self.criteria
        ]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths)),
            "total": sum(lengths),
        }

    def validate(self, expect: int) -> None:
        if len(self.criteria) != expect:
            raise ValueError(
                f"expected {expect} criteria, found {len(self.criteria)}"
            )
        keys = [c.key for c in self.criteria]
        if len(set(keys)) != len(keys):
            duplicates = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate criterion keys: {duplicates}")
        # Anchor completeness is enforced by Criterion.__post_init__, so reaching
        # here means every scale point of every criterion is described.


def manifest_for(criteria: tuple[Criterion, ...]) -> CriterionManifest:
    return CriterionManifest(criteria=criteria)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_manifest.py -q`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, 180 tests

- [ ] **Step 6: Commit**

```bash
git add src/packing/manifest.py tests/test_manifest.py
git commit -m "feat: validate and hash the criterion manifest"
```

---

### Task 2: Author the 20 ladder criteria to reach a pool of 40

**Files:**
- Create: `src/packing/ladder_criteria.py`
- Modify: `src/packing/cli.py:562` (add `LADDER_CRITERIA`)
- Test: `tests/test_ladder_criteria.py`

**Interfaces:**
- Consumes: `Criterion`, `CriterionManifest`, `manifest_for` from Task 1
- Produces: `LADDER_ONLY_CRITERIA: tuple[Criterion, ...]` (exactly 20), `LADDER_CRITERIA: tuple[Criterion, ...]` (exactly 40), `LADDER_MANIFEST: CriterionManifest`

The 20 keys, labels and descriptions are fixed here as design decisions. The anchor prose is the authoring deliverable; the test in Step 1 is its exact contract, and `Criterion.__post_init__` rejects any criterion missing a scale point.

| Key | Label | Description |
|---|---|---|
| `actionability` | Actionability | Whether the reader can act on the response without further clarification. |
| `assumptions` | Assumption handling | Whether unstated assumptions are surfaced rather than silently relied on. |
| `attribution` | Attribution | Whether claims taken from sources are attributed rather than presented bare. |
| `audience_fit` | Audience fit | Whether the register and depth match the apparent expertise of the asker. |
| `caveating` | Caveating | Whether genuine limitations are stated without hedging everything. |
| `cohesion` | Cohesion | Whether sentences connect so the argument carries across paragraphs. |
| `concreteness` | Concreteness | Whether general claims are grounded in specific detail. |
| `edge_cases` | Edge-case coverage | Whether boundary conditions implied by the question are addressed. |
| `example_quality` | Example quality | Whether examples illustrate the point rather than restating it. |
| `internal_reference` | Internal reference | Whether references to the response's own parts resolve correctly. |
| `ordering` | Ordering | Whether information arrives in an order the reader can follow. |
| `precision` | Precision | Whether quantities, units and qualifiers are stated exactly where they matter. |
| `question_coverage` | Question coverage | Whether every part of a multi-part question is answered. |
| `redundancy` | Redundancy | Whether the response avoids repeating itself. |
| `register` | Register | Whether tone is consistent throughout rather than drifting. |
| `scope_discipline` | Scope discipline | Whether the response stays inside what was asked. |
| `self_consistency` | Self consistency | Whether later statements agree with earlier ones. |
| `signposting` | Signposting | Whether structure is announced so the reader knows what is coming. |
| `terminology_consistency` | Terminology consistency | Whether one term is used for one concept throughout. |
| `unit_handling` | Unit handling | Whether units are present, correct and consistently converted. |

- [ ] **Step 1: Write the failing test**

```python
"""The 40-criterion ladder pool. The pool is the instrument, so its invariants
are mechanical rather than reviewed by eye."""

from __future__ import annotations

from packing.criteria import HELPSTEER2_CRITERIA
from packing.ladder_criteria import (
    LADDER_CRITERIA,
    LADDER_MANIFEST,
    LADDER_ONLY_CRITERIA,
)
from packing.synthetic_criteria import SYNTHETIC_ONLY_CRITERIA

EXPECTED_KEYS = {
    "actionability", "assumptions", "attribution", "audience_fit", "caveating",
    "cohesion", "concreteness", "edge_cases", "example_quality",
    "internal_reference", "ordering", "precision", "question_coverage",
    "redundancy", "register", "scope_discipline", "self_consistency",
    "signposting", "terminology_consistency", "unit_handling",
}


def test_there_are_exactly_twenty_new_criteria():
    assert len(LADDER_ONLY_CRITERIA) == 20
    assert {c.key for c in LADDER_ONLY_CRITERIA} == EXPECTED_KEYS


def test_the_pool_is_forty_and_composed_of_the_three_tiers():
    assert len(LADDER_CRITERIA) == 40
    assert LADDER_CRITERIA[:5] == HELPSTEER2_CRITERIA
    assert LADDER_CRITERIA[5:20] == SYNTHETIC_ONLY_CRITERIA
    assert LADDER_CRITERIA[20:] == LADDER_ONLY_CRITERIA


def test_manifest_validates_at_forty():
    LADDER_MANIFEST.validate(expect=40)


def test_every_scale_point_is_anchored_with_nonempty_prose():
    for c in LADDER_ONLY_CRITERIA:
        assert set(c.anchors) == {0, 1, 2, 3, 4}, c.key
        for point, text in c.anchors.items():
            assert text.strip(), f"{c.key} anchor {point} is empty"
            assert len(text.strip()) >= 10, f"{c.key} anchor {point} is too terse"


def test_no_duplicate_anchor_prose_within_a_criterion():
    for c in LADDER_ONLY_CRITERIA:
        texts = [t.strip().lower() for t in c.anchors.values()]
        assert len(set(texts)) == 5, f"{c.key} reuses anchor prose across points"


def test_descriptions_are_distinct_and_present():
    descriptions = [c.description.strip() for c in LADDER_ONLY_CRITERIA]
    assert all(descriptions)
    assert len(set(descriptions)) == 20


def test_length_distribution_is_recorded_and_bounded():
    stats = LADDER_MANIFEST.length_stats
    # A criterion whose prose dwarfs the others would change input-token growth
    # unevenly across rungs, so the spread is pinned.
    assert stats["max"] <= 4 * stats["min"], stats


def test_manifest_hash_is_recorded_for_the_run():
    assert len(LADDER_MANIFEST.hash) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_criteria.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.ladder_criteria'`

- [ ] **Step 3: Write the module**

Author all 20 criteria in the table's order. Pattern to follow exactly, shown complete for the first two:

```python
"""Criteria that exist only in the K = 40 ladder.

The synthetic stratum already extends K to 20 (`synthetic_criteria.py`). The
ladder needs 40 so that a load effect has room to appear well past the point the
K = 20 sweep could resolve. These twenty have no human-label counterpart and
carry no ground truth: in the ladder they are companions, present to create load
behind a first-position target. That is the one role unlabelled criteria can
legitimately play, since only the target needs a label.

Every criterion anchors all five scale points, for the reason established in
`criteria.py`: a partially anchored scale is adopted by the model as the whole
scale.
"""

from __future__ import annotations

from .criteria import HELPSTEER2_CRITERIA, Criterion
from .manifest import CriterionManifest, manifest_for
from .synthetic_criteria import SYNTHETIC_ONLY_CRITERIA

LADDER_ONLY_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="actionability",
        label="Actionability",
        description=(
            "Whether the reader can act on the response without further "
            "clarification."
        ),
        anchors={
            0: "nothing here can be acted on",
            1: "a direction is implied but no usable step is given",
            2: "some steps are usable, others need clarification first",
            3: "actionable with one small gap the reader can close",
            4: "every step needed to act is present and unambiguous",
        },
    ),
    Criterion(
        key="assumptions",
        label="Assumption handling",
        description=(
            "Whether unstated assumptions are surfaced rather than silently "
            "relied on."
        ),
        anchors={
            0: "rests entirely on unstated assumptions",
            1: "leans on major assumptions without naming any",
            2: "names some assumptions and hides others",
            3: "names the load-bearing assumptions; a minor one is implicit",
            4: "every assumption the answer depends on is stated",
        },
    ),
    # The remaining eighteen follow exactly this shape. Their `key`, `label` and
    # `description` are fixed verbatim in the task table above and are not open
    # decisions. What this step authors is the five anchor strings per criterion,
    # and the eight tests in Step 1 are its complete acceptance contract: five
    # anchors each, none empty, none under ten characters, no anchor prose reused
    # within a criterion, twenty distinct descriptions, and a length spread inside
    # 4x. Anchor prose is written to describe a scale point, so it cannot be
    # specified more tightly than that without writing it here.
)

LADDER_CRITERIA: tuple[Criterion, ...] = (
    HELPSTEER2_CRITERIA + SYNTHETIC_ONLY_CRITERIA + LADDER_ONLY_CRITERIA
)

LADDER_MANIFEST: CriterionManifest = manifest_for(LADDER_CRITERIA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_criteria.py -q`
Expected: PASS, 8 tests. If `test_length_distribution_is_recorded_and_bounded` fails, even out the anchor prose rather than loosening the bound.

- [ ] **Step 5: Record the manifest hash**

Run: `uv run python -c "from packing.ladder_criteria import LADDER_MANIFEST as m; print(m.hash, m.length_stats)"`
Paste the output into the module docstring as a `Manifest hash:` line so the committed pool has its identity on the record.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, 188 tests

- [ ] **Step 7: Commit**

```bash
git add src/packing/ladder_criteria.py tests/test_ladder_criteria.py
git commit -m "data: add anchored K=40 criterion set"
```

---

### Task 3: Eligibility pre-pass

**Files:**
- Create: `src/packing/eligibility.py`
- Test: `tests/test_eligibility.py`

**Interfaces:**
- Consumes: `PairedItem` from `src/packing/dataset.py:25`, `ALL_CORRUPTIONS` from `src/packing/cli.py:561`, `partition_seed` from `src/packing/partition.py:150`
- Produces: `CORE_OPERATIONS: tuple[str, ...]` (8 keys), `NUMBERS_OPERATION: str`, `applies(op_key: str, text: str, prompt: str, *, seed: int) -> bool`, `select_response(item: PairedItem, op_keys: tuple[str, ...], *, seed: int) -> str | None` returning `"a"`, `"b"` or `None`

- [ ] **Step 1: Write the failing test**

```python
"""Eligibility is decided from the source response alone, before any outcome
exists and before an operation is assigned."""

from __future__ import annotations

from packing.dataset import PairedItem
from packing.eligibility import (
    CORE_OPERATIONS,
    NUMBERS_OPERATION,
    applies,
    select_response,
)


def _item(a: str, b: str) -> PairedItem:
    return PairedItem(
        item_id="t-1",
        prompt="Explain the thing.",
        response_a=a,
        response_b=b,
        human_a={},
        human_b={},
        dataset="test",
        meta={},
        source="synthetic",
    )


def test_core_is_the_eight_high_applicability_operations():
    assert CORE_OPERATIONS == (
        "flatten", "truncate", "shuffle", "drop_answer",
        "agreement", "depunct", "duplicate", "filler",
    )
    assert NUMBERS_OPERATION == "numbers"


def test_numbers_needs_a_number_outside_code():
    with_number = "The limit is 30 seconds. It applies broadly. And then more."
    without = "The limit is short. It applies broadly. And then more text here."
    assert applies("numbers", with_number, "prompt", seed=1) is True
    assert applies("numbers", without, "prompt", seed=1) is False


def test_select_response_returns_none_when_neither_side_qualifies():
    item = _item("no.", "no.")
    assert select_response(item, ("shuffle",), seed=7) is None


def test_select_response_picks_the_only_eligible_side():
    long_text = "One sentence here. Two sentences here. Three sentences here."
    item = _item("no.", long_text)
    assert select_response(item, ("shuffle",), seed=7) == "b"


def test_select_response_is_deterministic_when_both_qualify():
    long_text = "One sentence here. Two sentences here. Three sentences here."
    item = _item(long_text, long_text)
    first = select_response(item, ("shuffle",), seed=7)
    assert first in {"a", "b"}
    assert select_response(item, ("shuffle",), seed=7) == first


def test_both_eligible_split_is_not_always_the_same_side():
    long_text = "One sentence here. Two sentences here. Three sentences here."
    picks = set()
    for seed in range(40):
        item = _item(long_text, long_text)
        picks.add(select_response(item, ("shuffle",), seed=seed))
    assert picks == {"a", "b"}, "side choice must vary with the seed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eligibility.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.eligibility'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Which arm a base prompt can serve, decided from the source text alone.

Eligibility is whether the mechanical edit changes the text at all. An operation
that cannot apply returns `changed=False` and would be dropped downstream, so
screening for it up front is what lets an arm have a defined population.

Nothing here touches an outcome, a score, or a provider. That ordering is the
point: arm assignment happens before eligibility is inspected (spec section 5),
and eligibility is inspected before an operation is assigned.
"""

from __future__ import annotations

import random

from .dataset import PairedItem
from .partition import partition_seed

CORE_OPERATIONS: tuple[str, ...] = (
    "flatten",
    "truncate",
    "shuffle",
    "drop_answer",
    "agreement",
    "depunct",
    "duplicate",
    "filler",
)

NUMBERS_OPERATION = "numbers"


def _registry() -> dict[str, object]:
    # Imported lazily: cli imports dataset, so a module-level import here would
    # close an import cycle.
    from .cli import ALL_CORRUPTIONS

    return {c.key: c for c in ALL_CORRUPTIONS}


def applies(op_key: str, text: str, prompt: str, *, seed: int) -> bool:
    """Does this operation change this text?"""
    corruption = _registry()[op_key]
    result = corruption.apply(
        text, seed=seed, severity=None, prompt=prompt
    )
    return bool(result.changed)


def select_response(
    item: PairedItem, op_keys: tuple[str, ...], *, seed: int
) -> str | None:
    """Pick the side this prompt will contribute, or None if neither qualifies.

    A side qualifies when every operation in `op_keys` applies to it. Where both
    sides qualify one is chosen uniformly at random from `seed`, so the choice is
    recorded rather than left to whichever side happened to be tested first.
    """
    qualifying = [
        side
        for side in ("a", "b")
        if all(
            applies(
                op,
                item.response_a if side == "a" else item.response_b,
                item.prompt,
                seed=partition_seed(f"{item.item_id}:{side}:{op}", 0, 0, seed),
            )
            for op in op_keys
        )
    ]
    if not qualifying:
        return None
    if len(qualifying) == 1:
        return qualifying[0]
    rng = random.Random(partition_seed(f"{item.item_id}:side", 0, 0, seed))
    return rng.choice(qualifying)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eligibility.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/eligibility.py tests/test_eligibility.py
git commit -m "feat: add the arm eligibility pre-pass"
```

---

### Task 4: Frozen-seed lane assignment

**Files:**
- Create: `src/packing/lanes.py`
- Test: `tests/test_lanes.py`

**Interfaces:**
- Consumes: `partition_seed` from `src/packing/partition.py:150`
- Produces: `LANE_CORE: str`, `LANE_NUMBERS: str`, `assign_lanes(item_ids: list[str], *, seed: int, core_share: float = 0.4157) -> dict[str, str]`, `fill_lane(item_ids: list[str], eligible: Callable[[str], bool], quota: int) -> tuple[list[str], list[str]]` returning `(filled, exhausted_remainder)`

`core_share` default is 508 / 1222, the initial lane split from spec section 5.

- [ ] **Step 1: Write the failing test**

```python
"""Lane assignment runs before eligibility is inspected. Reversing that order
changes the core population, because numbers eligibility correlates with response
content (spec section 5)."""

from __future__ import annotations

from packing.lanes import LANE_CORE, LANE_NUMBERS, assign_lanes, fill_lane


def test_lanes_are_deterministic_for_a_seed():
    ids = [f"p{i}" for i in range(100)]
    first = assign_lanes(ids, seed=11)
    assert assign_lanes(ids, seed=11) == first


def test_lanes_change_with_the_seed():
    ids = [f"p{i}" for i in range(100)]
    assert assign_lanes(ids, seed=11) != assign_lanes(ids, seed=12)


def test_every_prompt_gets_exactly_one_lane():
    ids = [f"p{i}" for i in range(100)]
    lanes = assign_lanes(ids, seed=11)
    assert set(lanes) == set(ids)
    assert set(lanes.values()) <= {LANE_CORE, LANE_NUMBERS}


def test_split_matches_the_requested_share():
    ids = [f"p{i}" for i in range(1000)]
    lanes = assign_lanes(ids, seed=11, core_share=0.4157)
    core = sum(1 for v in lanes.values() if v == LANE_CORE)
    assert 405 <= core <= 425, core


def test_assignment_is_independent_of_input_order():
    ids = [f"p{i}" for i in range(200)]
    a = assign_lanes(ids, seed=11)
    b = assign_lanes(list(reversed(ids)), seed=11)
    assert a == b, "lane must be a function of prompt id and seed, not position"


def test_fill_lane_stops_at_the_quota_and_reports_the_remainder():
    ids = [f"p{i}" for i in range(20)]
    filled, remainder = fill_lane(ids, lambda i: True, quota=5)
    assert filled == ids[:5]
    assert remainder == ids[5:]


def test_fill_lane_skips_ineligible_prompts_in_frozen_order():
    ids = [f"p{i}" for i in range(10)]
    filled, remainder = fill_lane(ids, lambda i: i.endswith(("1", "3", "5")), quota=2)
    assert filled == ["p1", "p3"]
    assert remainder == ["p4", "p5", "p6", "p7", "p8", "p9"]


def test_fill_lane_returns_short_when_the_stream_is_exhausted():
    ids = ["p0", "p1"]
    filled, remainder = fill_lane(ids, lambda i: True, quota=5)
    assert filled == ids
    assert remainder == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lanes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.lanes'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Assign base prompts to arm-specific sampling streams before eligibility is
inspected.

The order matters and is the whole reason this module exists. Screening one shared
frame and then carving the numbers quota out of it leaves the core pool 4.8%
numbers-eligible against a population value of 31.0%, because numbers eligibility
correlates with response content. The primary estimate would then describe
"core-eligible prompts not allocated to numbers" rather than the population the
estimand names.

Lane is a pure function of prompt id and seed, so it does not depend on fetch
order and a stream can be extended later without reshuffling what came before.
"""

from __future__ import annotations

from collections.abc import Callable

from .partition import partition_seed

LANE_CORE = "core"
LANE_NUMBERS = "numbers"

# 508 core candidates and 714 numbers candidates, the initial lane split.
DEFAULT_CORE_SHARE = 508 / 1222


def assign_lanes(
    item_ids: list[str], *, seed: int, core_share: float = DEFAULT_CORE_SHARE
) -> dict[str, str]:
    """Map each prompt id to exactly one lane, deterministically."""
    if not 0.0 < core_share < 1.0:
        raise ValueError(f"core_share must be in (0, 1), got {core_share}")
    cut = int(core_share * (2**32))
    lanes: dict[str, str] = {}
    for item_id in item_ids:
        # A hash of (id, seed) rather than a shuffled index, so lane membership
        # is stable when the stream is extended.
        draw = partition_seed(f"{item_id}:lane", 0, 0, seed) % (2**32)
        lanes[item_id] = LANE_CORE if draw < cut else LANE_NUMBERS
    return lanes


def fill_lane(
    item_ids: list[str],
    eligible: Callable[[str], bool],
    quota: int,
) -> tuple[list[str], list[str]]:
    """Take the first `quota` eligible prompts in frozen order.

    Returns the filled quota and the untouched remainder of the stream, so a later
    variance review can extend the sample by continuing where this stopped rather
    than by drawing afresh.
    """
    if quota < 0:
        raise ValueError(f"quota must be >= 0, got {quota}")
    filled: list[str] = []
    for index, item_id in enumerate(item_ids):
        if len(filled) >= quota:
            return filled, item_ids[index:]
        if eligible(item_id):
            filled.append(item_id)
    return filled, []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lanes.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/lanes.py tests/test_lanes.py
git commit -m "feat: assign arm lanes before eligibility screening"
```

---

### Task 5: Fixed-position nested planner

**Files:**
- Modify: `src/packing/runner.py` (add after `plan_targeted_blocks`, which ends at line 475)
- Test: `tests/test_ladder_plan.py`

**Interfaces:**
- Consumes: `CallSpec` (`src/packing/runner.py:30`), `Partition`, `Placement`, `partition_seed` (`src/packing/partition.py:41,23,150`), `PairedItem`, `Criterion`
- Produces: `plan_fixed_position(items: list[PairedItem], criteria: tuple[Criterion, ...], *, model: str, levels: tuple[int, ...], base_seed: int = 20260908) -> list[CallSpec]`

- [ ] **Step 1: Write the failing test**

```python
"""The ladder planner. Target at position 0, companions as nested prefixes of one
seeded permutation, so K varies the amount of trailing load and not which
criteria are present."""

from __future__ import annotations

import pytest

from packing.dataset import PairedItem
from packing.ladder_criteria import LADDER_CRITERIA
from packing.runner import plan_fixed_position

LEVELS = (1, 5, 10, 20, 40)


def _item(item_id: str, target: str) -> PairedItem:
    return PairedItem(
        item_id=item_id,
        prompt="p",
        response_a="a",
        response_b="b",
        human_a={},
        human_b={},
        dataset="test",
        meta={},
        source="synthetic",
        targeted_criterion=target,
    )


def test_target_is_always_at_position_zero():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    for spec in specs:
        assert spec.group[0].key == "precision", spec.nominal_n
        placement = spec.partition.placement_for("precision")
        assert placement.position == 0


def test_group_size_equals_the_requested_level():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    for spec in specs:
        assert len(spec.group) == spec.nominal_n


def test_companions_are_nested_prefixes_across_rungs():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    by_level = {}
    for spec in specs:
        if spec.side == "a":
            by_level[spec.nominal_n] = [c.key for c in spec.group]
    for smaller, larger in zip(LEVELS, LEVELS[1:]):
        assert by_level[larger][: len(by_level[smaller])] == by_level[smaller], (
            smaller,
            larger,
        )


def test_both_sides_share_the_companion_sequence():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=(40,)
    )
    a = [c.key for s in specs if s.side == "a" for c in s.group]
    b = [c.key for s in specs if s.side == "b" for c in s.group]
    assert a == b


def test_the_target_appears_once_per_call():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    for spec in specs:
        keys = [c.key for c in spec.group]
        assert keys.count("precision") == 1
        assert len(set(keys)) == len(keys)


def test_companion_order_varies_between_prompts():
    orders = set()
    for i in range(20):
        specs = plan_fixed_position(
            [_item(f"i{i}", "precision")],
            LADDER_CRITERIA,
            model="mock",
            levels=(40,),
        )
        orders.add(tuple(c.key for c in specs[0].group[1:]))
    assert len(orders) > 1, "each prompt draws its own companion permutation"


def test_call_count_is_two_sides_times_levels():
    specs = plan_fixed_position(
        [_item("i1", "precision"), _item("i2", "ordering")],
        LADDER_CRITERIA,
        model="mock",
        levels=LEVELS,
    )
    assert len(specs) == 2 * len(LEVELS) * 2


def test_identities_are_unique():
    specs = plan_fixed_position(
        [_item("i1", "precision")], LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    identities = [s.identity for s in specs]
    assert len(set(identities)) == len(identities)


def test_level_above_the_pool_is_rejected():
    with pytest.raises(ValueError, match="level 41 exceeds pool size 40"):
        plan_fixed_position(
            [_item("i1", "precision")],
            LADDER_CRITERIA,
            model="mock",
            levels=(41,),
        )


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="has no targeted_criterion"):
        plan_fixed_position(
            [_item("i1", None)], LADDER_CRITERIA, model="mock", levels=(5,)
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_plan.py -q`
Expected: FAIL with `ImportError: cannot import name 'plan_fixed_position'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/packing/runner.py`:

```python
def plan_fixed_position(
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    model: str,
    levels: tuple[int, ...],
    base_seed: int = 20260908,
) -> list[CallSpec]:
    """Plan a ladder with the targeted criterion pinned at position 0.

    The K = 20 sweep drew a random partition, which coupled load to position:
    position is nested inside the call, so E[position | K] = (K-1)/2 exactly and
    support is triangular. Here the target sits first at every rung and companions
    pad the call, so load is the only thing that varies.

    Companions are a nested prefix of one seeded permutation of the non-target
    criteria, shared by both sides. Without nesting, K would change both the amount
    of trailing load and which particular criteria are present, putting composition
    noise into the endpoint contrast.
    """
    pool_size = len(criteria)
    for level in levels:
        if level < 1:
            raise ValueError(f"level {level} must be >= 1")
        if level > pool_size:
            raise ValueError(f"level {level} exceeds pool size {pool_size}")

    specs: list[CallSpec] = []
    for item in items:
        if not item.targeted_criterion:
            raise ValueError(f"{item.item_id} has no targeted_criterion")
        target = next(
            (c for c in criteria if c.key == item.targeted_criterion), None
        )
        if target is None:
            raise ValueError(
                f"{item.targeted_criterion!r} not in the criterion pool"
            )

        companions = [c for c in criteria if c.key != target.key]
        rng = random.Random(
            partition_seed(f"{item.item_id}:companions", 0, 0, base_seed)
        )
        rng.shuffle(companions)

        for nominal_n in levels:
            group = (target, *companions[: nominal_n - 1])
            placements = tuple(
                Placement(
                    criterion=c,
                    group_index=0,
                    position=index,
                    group_size=len(group),
                )
                for index, c in enumerate(group)
            )
            partition = Partition(
                nominal_n=nominal_n,
                groups=(group,),
                seed=base_seed,
                placements=placements,
            )
            for side in ("a", "b"):
                specs.append(
                    CallSpec(
                        item=item,
                        side=side,
                        group=group,
                        group_index=0,
                        partition=partition,
                        nominal_n=nominal_n,
                        replicate=0,
                        model=model,
                    )
                )
    return specs
```

Add `Placement` to the existing partition import at `src/packing/runner.py:26`:

```python
from .partition import (
    Partition,
    Placement,
    make_partition,
    nominal_levels,
    partition_seed,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_plan.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/runner.py tests/test_ladder_plan.py
git commit -m "feat: plan the fixed-position nested ladder"
```

---

### Task 6: One frozen corruption realisation per prompt

**Files:**
- Create: `src/packing/ladder_dataset.py`
- Test: `tests/test_ladder_dataset.py`

**Interfaces:**
- Consumes: `PairedItem`, `build_corrupted_items` (`src/packing/dataset.py:251`), `partition_seed`, `CORE_OPERATIONS` and `select_response` from Task 3
- Produces: `assign_operations(item_ids: list[str], op_keys: tuple[str, ...], *, seed: int) -> dict[str, str]`, `build_ladder_item(base: PairedItem, op_key: str, side: str, criteria: tuple[Criterion, ...], *, seed: int) -> PairedItem | None`

The built item sets `meta["base_item"] = base.item_id`. That exact key matters: `analyze.py:164` derives the clustering column `base_unit` from `item.meta["base_item"]`, and `independence_unit` (`analyze.py:413`) falls back to `item_id` when it is absent, which would treat every corrupted item as its own cluster and silently inflate n.

- [ ] **Step 1: Write the failing test**

```python
"""One operation and one realisation per prompt, frozen so that the paired
difference carries load and not realisation noise."""

from __future__ import annotations

from collections import Counter

from packing.dataset import PairedItem
from packing.eligibility import CORE_OPERATIONS
from packing.ladder_criteria import LADDER_CRITERIA
from packing.ladder_dataset import assign_operations, build_ladder_item

TEXT = (
    "First sentence of the answer. Second sentence adds detail. "
    "Third sentence closes it out. Fourth sentence is spare."
)


def _base(item_id: str) -> PairedItem:
    return PairedItem(
        item_id=item_id,
        prompt="Explain the thing.",
        response_a=TEXT,
        response_b=TEXT,
        human_a={},
        human_b={},
        dataset="test",
        meta={},
    )


def test_operations_are_assigned_in_equal_quotas():
    ids = [f"p{i}" for i in range(80)]
    assigned = assign_operations(ids, CORE_OPERATIONS, seed=5)
    counts = Counter(assigned.values())
    assert set(counts) == set(CORE_OPERATIONS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_operation_assignment_is_deterministic():
    ids = [f"p{i}" for i in range(80)]
    assert assign_operations(ids, CORE_OPERATIONS, seed=5) == assign_operations(
        ids, CORE_OPERATIONS, seed=5
    )


def test_realisation_is_identical_across_repeated_builds():
    base = _base("p1")
    first = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    second = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert first is not None
    assert first.response_a == second.response_a
    assert first.response_b == second.response_b


def test_the_targeted_criterion_is_the_operations_target():
    base = _base("p1")
    item = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.targeted_criterion == "coherence"


def test_inapplicable_operation_yields_none_rather_than_an_unchanged_item():
    base = PairedItem(
        item_id="p2",
        prompt="p",
        response_a="Short.",
        response_b="Short.",
        human_a={},
        human_b={},
        dataset="test",
        meta={},
    )
    assert build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9) is None


def test_only_the_selected_side_is_corrupted():
    base = _base("p1")
    item = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.response_a != TEXT
    assert item.response_b == TEXT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.ladder_dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Build ladder items: one operation, one realisation, frozen per prompt.

The realisation is frozen because it is reused at every rung. If it were redrawn
per rung, the paired difference between K = 1 and K = 40 would carry realisation
noise on top of the load effect, and the estimator has no way to separate them.

Equal quotas are assigned by position in a seeded permutation rather than by
independent draws, so a run of 464 prompts lands 58 per operation exactly rather
than approximately.
"""

from __future__ import annotations

import random

from .criteria import Criterion
from .dataset import PairedItem
from .partition import partition_seed


def assign_operations(
    item_ids: list[str], op_keys: tuple[str, ...], *, seed: int
) -> dict[str, str]:
    """Randomised assignment subject to exact equal quotas."""
    if not op_keys:
        raise ValueError("op_keys must be non-empty")
    order = list(item_ids)
    random.Random(partition_seed("operation-assignment", 0, 0, seed)).shuffle(order)
    return {
        item_id: op_keys[index % len(op_keys)]
        for index, item_id in enumerate(order)
    }


def build_ladder_item(
    base: PairedItem,
    op_key: str,
    side: str,
    criteria: tuple[Criterion, ...],
    *,
    seed: int,
) -> PairedItem | None:
    """Apply one frozen realisation of `op_key` to one side of `base`.

    Returns None when the operation does not apply, rather than an unchanged item
    carrying a ground-truth claim, which would be a false label.
    """
    from .cli import ALL_CORRUPTIONS

    corruption = next((c for c in ALL_CORRUPTIONS if c.key == op_key), None)
    if corruption is None:
        raise ValueError(f"unknown corruption {op_key!r}")
    if corruption.targeted_criterion not in {c.key for c in criteria}:
        raise ValueError(
            f"{corruption.targeted_criterion!r} not in the criterion pool"
        )
    if side not in {"a", "b"}:
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    clean = base.response_a if side == "a" else base.response_b
    realisation_seed = partition_seed(
        f"{base.item_id}:{side}:{op_key}", 0, 0, seed
    )
    result = corruption.apply(
        clean, seed=realisation_seed, severity=None, prompt=base.prompt
    )
    if not result.changed:
        return None

    return PairedItem(
        item_id=f"lad-{op_key}-{base.item_id}-{side}",
        prompt=base.prompt,
        response_a=result.text if side == "a" else base.response_a,
        response_b=result.text if side == "b" else base.response_b,
        human_a={},
        human_b={},
        dataset=base.dataset,
        meta={
            **base.meta,
            "base_item": base.item_id,
            "operation": op_key,
            "corrupted_side": side,
            "realisation_seed": realisation_seed,
            "provenance": result.provenance,
        },
        source="synthetic",
        targeted_criterion=corruption.targeted_criterion,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_dataset.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/ladder_dataset.py tests/test_ladder_dataset.py
git commit -m "feat: freeze one corruption realisation per ladder prompt"
```

---

### Task 7: Unevaluable-output scoring rule

**Files:**
- Create: `src/packing/unevaluable.py`
- Test: `tests/test_unevaluable.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `FailureClass` enum-like constants `INFRASTRUCTURE: str`, `MODEL_OUTPUT: str`; `classify_failure(message: str) -> str`; `score_unevaluable(rule: str) -> float | None` where `rule` is `"primary"`, `"complete_case"` or `"best_case"`

- [ ] **Step 1: Write the failing test**

```python
"""Unevaluable output needs a defined value, because heavy load is exactly what
produces it. Infrastructure failures are replaced; model-output failures are not."""

from __future__ import annotations

import pytest

from packing.unevaluable import (
    INFRASTRUCTURE,
    MODEL_OUTPUT,
    classify_failure,
    score_unevaluable,
)


@pytest.mark.parametrize(
    "message",
    [
        "Connection reset by peer",
        "Request timed out after 120s",
        "rate limit exceeded, retry after 2s",
        "provider returned 503 Service Unavailable",
        "502 Bad Gateway",
    ],
)
def test_transient_transport_failures_are_infrastructure(message):
    assert classify_failure(message) == INFRASTRUCTURE


@pytest.mark.parametrize(
    "message",
    [
        "gpt-5-mini: truncated at max_completion_tokens=9352 for group_size=40",
        "content did not parse as JSON",
        "empty content returned",
        "context_length_exceeded: 41000 tokens > 32768",
        "content_filter triggered on the response",
        "model refused to answer",
        "criterion key set mismatch: got {...} expected {...}",
        "invalid_request_error: unsupported parameter",
    ],
)
def test_load_sensitive_failures_are_model_output(message):
    assert classify_failure(message) == MODEL_OUTPUT


def test_unknown_failures_default_to_model_output():
    # Defaulting to model output is conservative: it keeps the observation in the
    # analysis rather than replacing it, so an unrecognised load-induced failure
    # cannot be silently dropped.
    assert classify_failure("something nobody anticipated") == MODEL_OUTPUT


def test_primary_rule_scores_unevaluable_as_a_miss():
    assert score_unevaluable("primary") == 0.0


def test_complete_case_excludes_the_observation():
    assert score_unevaluable("complete_case") is None


def test_best_case_scores_unevaluable_as_a_hit():
    assert score_unevaluable("best_case") == 1.0


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown rule"):
        score_unevaluable("whatever")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_unevaluable.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.unevaluable'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Failure classification and the scoring rule for unevaluable output.

Two categories, because conflating them biases the result. Model-output failures
may become more common at K = 40, which makes them part of the load effect;
replacing those prompts would condition the analysis on surviving high load and
bias degradation downward.

A failure is infrastructural only if it is transient and independent of the
request's content and length. Anything that could plausibly be caused by a
40-criterion prompt stays in the analysis.
"""

from __future__ import annotations

INFRASTRUCTURE = "infrastructure"
MODEL_OUTPUT = "model_output"

_TRANSIENT_MARKERS = (
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
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


def classify_failure(message: str) -> str:
    """Infrastructure only for transient, content-independent failures.

    Unrecognised failures default to MODEL_OUTPUT. That is the conservative
    direction: an unrecognised load-induced failure stays in the analysis rather
    than being replaced out of it.
    """
    lowered = message.lower()
    if any(marker in lowered for marker in _TRANSIENT_MARKERS):
        return INFRASTRUCTURE
    return MODEL_OUTPUT


def score_unevaluable(rule: str) -> float | None:
    """Value for an observation whose targeted score cannot be determined.

    `primary` scores it a miss: recovery requires the judge to move the targeted
    score in the seeded direction, and a judgment that cannot be read is not a
    demonstrated recovery. The other two rules are the pre-specified brackets.
    """
    if rule == "primary":
        return 0.0
    if rule == "complete_case":
        return None
    if rule == "best_case":
        return 1.0
    raise ValueError(
        f"unknown rule {rule!r}; expected primary, complete_case or best_case"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_unevaluable.py -q`
Expected: PASS, 20 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/unevaluable.py tests/test_unevaluable.py
git commit -m "feat: classify failures and score unevaluable output"
```

---

### Task 8: Stratified equal-weight endpoint estimator

**Files:**
- Modify: `src/packing/analyze.py` (add after `paired_ladder`, which ends at line 570)
- Test: `tests/test_stratified.py`

**Interfaces:**
- Consumes: `resolved`, `independence_unit` (`src/packing/analyze.py:406,411`)
- Produces: `StratifiedResult` dataclass with fields `theta: float`, `se: float`, `df: float`, `t_stat: float`, `p_value: float`, `ci_low: float`, `ci_high: float`, `per_operation: dict[str, tuple[float, int]]`; function `stratified_endpoint(frame: pd.DataFrame, reference_size: int, comparison_size: int, *, operations: tuple[str, ...]) -> StratifiedResult | None`

- [ ] **Step 1: Write the failing test**

```python
"""The primary estimator: equal weight per operation, heteroskedastic SE, and
Welch-Satterthwaite degrees of freedom. Equal quotas define the weights, so the
estimator has to apply them rather than average all rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packing.analyze import stratified_endpoint

OPS = ("op1", "op2")


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "base_unit": unit,
                "operation": operation,
                "group_size": size,
                "correct_num": value,
                "judge_tie": False,
            }
            for unit, operation, size, value in rows
        ]
    )


def test_returns_none_without_both_endpoints():
    frame = _frame([("u1", "op1", 1, 1.0)])
    assert stratified_endpoint(frame, 1, 40, operations=OPS) is None


def test_equal_weighting_ignores_unequal_operation_counts():
    # op1 has 6 units all shifting +1.0, op2 has 2 units all shifting 0.0.
    # A flat mean over rows would give +0.75; equal weighting gives +0.5.
    rows = []
    for i in range(6):
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, 1.0)]
    for i in range(2):
        rows += [(f"b{i}", "op2", 1, 1.0), (f"b{i}", "op2", 40, 1.0)]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result is not None
    assert result.theta == 0.5


def test_per_operation_means_and_counts_are_reported():
    rows = []
    for i in range(4):
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, 1.0)]
    for i in range(4):
        rows += [(f"b{i}", "op2", 1, 1.0), (f"b{i}", "op2", 40, 0.0)]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result.per_operation["op1"] == (1.0, 4)
    assert result.per_operation["op2"] == (-1.0, 4)
    assert result.theta == 0.0


def test_satterthwaite_df_lies_between_the_smallest_stratum_and_the_total():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        d = rng.normal(0.1, 0.3)
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, d)]
    for i in range(10):
        d = rng.normal(0.1, 0.9)
        rows += [(f"b{i}", "op2", 1, 0.0), (f"b{i}", "op2", 40, d)]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert 9 <= result.df <= 40, result.df


def test_a_real_shift_is_detected():
    rng = np.random.default_rng(1)
    rows = []
    for op in OPS:
        for i in range(60):
            rows += [
                (f"{op}{i}", op, 1, 0.0),
                (f"{op}{i}", op, 40, rng.normal(-0.30, 0.20)),
            ]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result.theta < -0.2
    assert result.p_value < 0.01
    assert result.ci_high < 0


def test_a_null_is_not_detected():
    rng = np.random.default_rng(2)
    rows = []
    for op in OPS:
        for i in range(60):
            rows += [
                (f"{op}{i}", op, 1, 0.0),
                (f"{op}{i}", op, 40, rng.normal(0.0, 0.20)),
            ]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result.p_value > 0.05
    assert result.ci_low < 0 < result.ci_high


def test_a_stratum_with_too_few_units_is_rejected():
    rows = [("a0", "op1", 1, 0.0), ("a0", "op1", 40, 1.0)]
    for i in range(4):
        rows += [(f"b{i}", "op2", 1, 0.0), (f"b{i}", "op2", 40, 1.0)]
    assert stratified_endpoint(_frame(rows), 1, 40, operations=OPS) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stratified.py -q`
Expected: FAIL with `ImportError: cannot import name 'stratified_endpoint'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/packing/analyze.py`:

```python
@dataclass(frozen=True)
class StratifiedResult:
    """Equal-weight average across operation-specific endpoint differences."""

    theta: float
    se: float
    df: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    per_operation: dict[str, tuple[float, int]]

    def as_row(self) -> dict[str, Any]:
        return {
            "theta": round(self.theta, 4),
            "se": round(self.se, 4),
            "df": round(self.df, 1),
            "t": round(self.t_stat, 3),
            "p": self.p_value,
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
        }


def stratified_endpoint(
    frame: pd.DataFrame,
    reference_size: int,
    comparison_size: int,
    *,
    operations: tuple[str, ...],
    min_units: int = 3,
) -> StratifiedResult | None:
    """Equal weight per operation, not per observation.

    Equal quotas define the weights, so the estimator must apply them. An
    unqualified mean over all per-unit differences would re-weight operations by
    however many units each happened to contribute, which is exactly what quotas
    exist to prevent.

    theta = (1/O) * sum_o mean(d_o)
    Var   = (1/O^2) * sum_o s_o^2 / n_o
    df    = Welch-Satterthwaite over the same strata
    """
    res = resolved(frame)
    cluster = independence_unit(res)
    if "operation" not in res.columns:
        raise ValueError("frame has no `operation` column")

    per_unit = (
        res[res["group_size"].isin([reference_size, comparison_size])]
        .groupby([cluster, "operation", "group_size"])["correct_num"]
        .mean()
        .unstack("group_size")
    )
    if reference_size not in per_unit or comparison_size not in per_unit:
        return None
    paired = per_unit[[reference_size, comparison_size]].dropna()
    if paired.empty:
        return None
    diffs = (paired[comparison_size] - paired[reference_size]).rename("d")

    means: list[float] = []
    variance_terms: list[float] = []
    df_terms: list[float] = []
    per_operation: dict[str, tuple[float, int]] = {}
    for operation in operations:
        try:
            values = diffs.xs(operation, level="operation").to_numpy()
        except KeyError:
            return None
        n = len(values)
        if n < min_units:
            return None
        mean = float(values.mean())
        per_operation[operation] = (round(mean, 4), n)
        means.append(mean)
        if n > 1:
            term = float(values.var(ddof=1)) / n
        else:
            term = 0.0
        variance_terms.append(term)
        df_terms.append(term**2 / (n - 1) if n > 1 else 0.0)

    count = len(operations)
    theta = float(np.mean(means))
    variance = sum(variance_terms) / count**2
    se = float(np.sqrt(variance))
    if not se or np.isnan(se):
        return None
    numerator = sum(variance_terms) ** 2
    denominator = sum(df_terms)
    df = float(numerator / denominator) if denominator > 0 else float(count - 1)
    t_stat = theta / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
    crit = float(stats.t.ppf(0.975, df=df))
    return StratifiedResult(
        theta=theta,
        se=se,
        df=df,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=theta - crit * se,
        ci_high=theta + crit * se,
        per_operation=per_operation,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stratified.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/analyze.py tests/test_stratified.py
git commit -m "analysis: add the stratified endpoint estimator"
```

---

### Task 9: CLI command and manifest-based cost recompute

**Files:**
- Modify: `src/packing/cli.py` (add `ladder` command after the `sweep` command)
- Test: `tests/test_ladder_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1 to 8
- Produces: `packing ladder` click command with options `--items PATH`, `--model TEXT`, `--core-n INTEGER` (default 464), `--numbers-n INTEGER` (default 184), `--levels TEXT` (default `1,5,10,20,40`), `--seed INTEGER` (default 20260908), `--dry-run/--no-dry-run` (default `--dry-run`)

- [ ] **Step 1: Write the failing test**

```python
"""The ladder command. Dry run is the default, and the command refuses to send
anything until the manifest-based cost recompute has been seen."""

from __future__ import annotations

import json

from click.testing import CliRunner

from packing.cli import cli

TEXT = (
    "First sentence of the answer here. Second sentence adds some detail. "
    "Third sentence closes it out nicely. Fourth sentence is spare text. "
    "The limit is 30 seconds for this operation."
)


def _items(path, count):
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(count):
            handle.write(
                json.dumps(
                    {
                        "item_id": f"hs2-{i:06d}",
                        "prompt": "Explain the thing.",
                        "response_a": TEXT,
                        "response_b": TEXT,
                        "human_a": {},
                        "human_b": {},
                        "dataset": "test",
                        "meta": {},
                    }
                )
                + "\n"
            )


def test_dry_run_reports_the_plan_without_sending():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 60)
        result = runner.invoke(
            cli,
            [
                "ladder", "--items", "items.jsonl", "--model", "mock",
                "--core-n", "16", "--numbers-n", "8", "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "dry run: nothing sent" in result.output
        assert "manifest" in result.output
        assert "calls:" in result.output


def test_dry_run_prints_the_manifest_hash_and_pool_size():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 60)
        result = runner.invoke(
            cli,
            ["ladder", "--items", "items.jsonl", "--core-n", "16",
             "--numbers-n", "8", "--dry-run"],
        )
        assert "K=40 pool" in result.output


def test_lanes_are_reported_separately():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 60)
        result = runner.invoke(
            cli,
            ["ladder", "--items", "items.jsonl", "--core-n", "16",
             "--numbers-n", "8", "--dry-run"],
        )
        assert "core lane" in result.output
        assert "numbers lane" in result.output


def test_short_lane_is_reported_rather_than_silently_accepted():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _items("items.jsonl", 12)
        result = runner.invoke(
            cli,
            ["ladder", "--items", "items.jsonl", "--core-n", "464",
             "--numbers-n", "184", "--dry-run"],
        )
        assert "short" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_cli.py -q`
Expected: FAIL, exit code 2 with `No such command 'ladder'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/packing/cli.py`:

```python
@cli.command("ladder")
@click.option("--items", "items_path", type=click.Path(), default="data/helpsteer2.jsonl", show_default=True)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--core-n", type=int, default=464, show_default=True)
@click.option("--numbers-n", type=int, default=184, show_default=True)
@click.option("--levels", default="1,5,10,20,40", show_default=True)
@click.option("--seed", type=int, default=20260908, show_default=True)
@click.option("--dry-run/--no-dry-run", default=True, show_default=True)
def ladder_cmd(
    items_path: str,
    model: str,
    core_n: int,
    numbers_n: int,
    levels: str,
    seed: int,
    dry_run: bool,
) -> None:
    """The fixed-position load ladder: does trailing load change recovery?"""
    from .eligibility import CORE_OPERATIONS, NUMBERS_OPERATION, select_response
    from .ladder_criteria import LADDER_MANIFEST, LADDER_CRITERIA
    from .ladder_dataset import assign_operations, build_ladder_item
    from .lanes import LANE_CORE, LANE_NUMBERS, assign_lanes, fill_lane
    from .runner import plan_fixed_position

    LADDER_MANIFEST.validate(expect=40)
    swept = tuple(int(x) for x in levels.split(","))
    path = Path(items_path)
    if not path.exists():
        raise click.ClickException(f"{path} not found; run `packing fetch` first")
    bases = read_jsonl(path)
    by_id = {b.item_id: b for b in bases}

    lanes = assign_lanes([b.item_id for b in bases], seed=seed)
    core_stream = [b.item_id for b in bases if lanes[b.item_id] == LANE_CORE]
    numbers_stream = [b.item_id for b in bases if lanes[b.item_id] == LANE_NUMBERS]

    def core_eligible(item_id: str) -> bool:
        return select_response(by_id[item_id], CORE_OPERATIONS, seed=seed) is not None

    def numbers_eligible(item_id: str) -> bool:
        return select_response(by_id[item_id], (NUMBERS_OPERATION,), seed=seed) is not None

    core_ids, _ = fill_lane(core_stream, core_eligible, core_n)
    numbers_ids, _ = fill_lane(numbers_stream, numbers_eligible, numbers_n)

    console.print(
        f"K=40 pool, manifest {LADDER_MANIFEST.hash}, "
        f"lengths {LADDER_MANIFEST.length_stats}"
    )
    console.print(
        f"core lane: {len(core_stream)} candidates, {len(core_ids)}/{core_n} filled"
    )
    console.print(
        f"numbers lane: {len(numbers_stream)} candidates, "
        f"{len(numbers_ids)}/{numbers_n} filled"
    )
    if len(core_ids) < core_n or len(numbers_ids) < numbers_n:
        console.print(
            "[yellow]lane short of quota: extend the stream before running[/yellow]"
        )

    assigned = assign_operations(core_ids, CORE_OPERATIONS, seed=seed)
    items: list = []
    for item_id in core_ids:
        base = by_id[item_id]
        op = assigned[item_id]
        side = select_response(base, (op,), seed=seed) or select_response(
            base, CORE_OPERATIONS, seed=seed
        )
        built = build_ladder_item(base, op, side, LADDER_CRITERIA, seed=seed)
        if built is not None:
            items.append(built)
    for item_id in numbers_ids:
        base = by_id[item_id]
        side = select_response(base, (NUMBERS_OPERATION,), seed=seed)
        built = build_ladder_item(
            base, NUMBERS_OPERATION, side, LADDER_CRITERIA, seed=seed
        )
        if built is not None:
            items.append(built)

    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model=model, levels=swept
    )
    est = estimate(specs)
    entry = CATALOG.get(model, {"in": 0.0, "out": 0.0})
    projection = project_cost(est, entry["in"], entry["out"])
    console.print(f"items: {len(items)}  levels={swept}")
    console.print(f"calls: {len(specs)}")
    console.print(
        f"est input {est.input_tokens:,}, output ceiling {est.output_tokens:,}"
    )
    console.print(f"cost ceiling ${projection:.2f} (actual output will be lower)")
    if dry_run:
        console.print("dry run: nothing sent")
        return
    raise click.ClickException(
        "live execution is a separate decision; re-run with --dry-run reviewed "
        "and the manifest-based cost recompute recorded in the spec"
    )
```

`estimate` and `project_cost` are already imported in `cli.py` from `.runner` (see `src/packing/cli.py:34-41`), and `read_jsonl` from `.dataset`. `estimate` takes only `specs`. The `sweep` command at `src/packing/cli.py:807-812` is the pattern to copy.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_cli.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Recompute the budget against the real manifest**

Run: `uv run packing ladder --dry-run --core-n 464 --numbers-n 184`
Record the reported call count and cost ceiling. Update spec section 14, replacing the provisional `$23.40` and `$34.38` extrapolations with the manifest-based figures, and note the token growth per criterion the real 40-criterion prompts produce.

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/cli.py tests/test_ladder_cli.py docs/superpowers/specs/2026-09-08-load-ladder-design.md
git commit -m "feat: add the ladder command and recompute cost from the manifest"
```

---

### Task 10: Persist model-output failures as scorable records

**Files:**
- Modify: `src/packing/runner.py:176-200` (`_build_record`) and `src/packing/runner.py:260-278` (the concurrent failure branch)
- Modify: `src/packing/analyze.py:69-131` (`build_outcomes`)
- Test: `tests/test_failure_records.py`

**Interfaces:**
- Consumes: `classify_failure`, `score_unevaluable`, `MODEL_OUTPUT`, `INFRASTRUCTURE` from Task 7
- Produces: `_build_failure_record(spec: CallSpec, message: str) -> dict[str, Any]` in `runner.py`, emitting `{"identity", "item_id", "side", "model", "nominal_n", "replicate", "partition_hash", "group_index", "group_size", "group", "positions", "failure": str, "failure_class": str, "scores": {}}`. `build_outcomes(records, items, criteria, *, unevaluable_rule: str = "primary")` gains the keyword.

Task 7 defined the scoring rule but nothing calls it, because the runner records a
failure only in `stats.errors`, which is printed and discarded. An unevaluable
observation that never becomes a row cannot be scored, so the rule would silently
degrade to complete-case, which the spec forbids as the primary handling.

- [ ] **Step 1: Write the failing test**

```python
"""Model-output failures become rows so the scoring rule can reach them.
A failure that lives only in a log is a complete-case exclusion by default."""

from __future__ import annotations

from packing.analyze import build_outcomes, to_frame
from packing.criteria import HELPSTEER2_CRITERIA
from packing.dataset import PairedItem
from packing.runner import _build_failure_record
from packing.unevaluable import INFRASTRUCTURE, MODEL_OUTPUT


def _item():
    return PairedItem(
        item_id="lad-shuffle-hs2-000001-a",
        prompt="p",
        response_a="x",
        response_b="y",
        human_a={},
        human_b={},
        dataset="test",
        meta={"base_item": "hs2-000001", "operation": "shuffle"},
        source="synthetic",
        targeted_criterion="coherence",
    )


class _Spec:
    """Minimal stand-in carrying only what the record builder reads."""

    def __init__(self, item, side, group):
        self.item = item
        self.side = side
        self.group = group
        self.group_index = 0
        self.nominal_n = len(group)
        self.replicate = 0
        self.model = "mock"
        self.identity = f"{item.item_id}|{side}|mock|0|deadbeef|0"

        class _P:
            hash = "deadbeef"

        self.partition = _P()


def test_truncation_is_recorded_as_a_model_output_failure():
    spec = _Spec(_item(), "a", HELPSTEER2_CRITERIA[:1])
    record = _build_failure_record(spec, "truncated at max_completion_tokens=9352")
    assert record["failure_class"] == MODEL_OUTPUT
    assert record["scores"] == {}
    assert record["group_size"] == 1
    assert record["positions"] == {"helpfulness": 0}


def test_timeout_is_recorded_as_infrastructure():
    spec = _Spec(_item(), "a", HELPSTEER2_CRITERIA[:1])
    record = _build_failure_record(spec, "Request timed out after 120s")
    assert record["failure_class"] == INFRASTRUCTURE


def test_primary_rule_turns_a_failure_row_into_a_miss():
    spec = _Spec(_item(), "a", HELPSTEER2_CRITERIA[:1])
    failure = _build_failure_record(spec, "content did not parse as JSON")
    good = {
        **failure,
        "failure": None,
        "failure_class": None,
        "scores": {"helpfulness": 3},
        "input_tokens": 10,
        "output_tokens": 10,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
    }
    good["side"] = "b"
    good["identity"] = good["identity"].replace("|a|", "|b|")
    outcomes = build_outcomes(
        [failure, good], [_item()], HELPSTEER2_CRITERIA[:1], unevaluable_rule="primary"
    )
    frame = to_frame(outcomes)
    assert len(frame) == 1
    assert frame["correct_num"].iloc[0] == 0.0


def test_complete_case_rule_drops_the_failure_row():
    spec = _Spec(_item(), "a", HELPSTEER2_CRITERIA[:1])
    failure = _build_failure_record(spec, "content did not parse as JSON")
    outcomes = build_outcomes(
        [failure], [_item()], HELPSTEER2_CRITERIA[:1], unevaluable_rule="complete_case"
    )
    assert outcomes == []


def test_infrastructure_failures_are_never_scored():
    spec = _Spec(_item(), "a", HELPSTEER2_CRITERIA[:1])
    failure = _build_failure_record(spec, "503 Service Unavailable")
    outcomes = build_outcomes(
        [failure], [_item()], HELPSTEER2_CRITERIA[:1], unevaluable_rule="primary"
    )
    assert outcomes == [], "infrastructure failures are replaced, not scored"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_failure_records.py -q`
Expected: FAIL with `ImportError: cannot import name '_build_failure_record'`

- [ ] **Step 3: Add the failure record builder to `runner.py`**

```python
def _build_failure_record(spec: CallSpec, message: str) -> dict[str, Any]:
    """A failed call as a row, so the scoring rule can reach it.

    Model-output failures may become more common at high K, which makes them part
    of the load effect. Recorded only in a log they would be a complete-case
    exclusion by default, which is exactly the handling the protocol forbids as
    primary.
    """
    from .unevaluable import classify_failure

    return {
        "identity": spec.identity,
        "item_id": spec.item.item_id,
        "side": spec.side,
        "model": spec.model,
        "nominal_n": spec.nominal_n,
        "replicate": spec.replicate,
        "partition_hash": spec.partition.hash,
        "group_index": spec.group_index,
        "group_size": len(spec.group),
        "group": [c.key for c in spec.group],
        "positions": {c.key: i for i, c in enumerate(spec.group)},
        "failure": message,
        "failure_class": classify_failure(message),
        "scores": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
    }
```

- [ ] **Step 4: Write the failure record in the concurrent path**

In `src/packing/runner.py`, inside the `except ProviderError as exc:` branch at line 260, after `stats.errors.append(...)`, add:

```python
                        # Model-output failures are part of the treatment, so they
                        # are persisted rather than left in a log. Infrastructure
                        # failures are persisted too, and are filtered out at
                        # analysis time so they can be counted and replaced.
                        with write_lock:
                            fh.write(
                                json.dumps(_build_failure_record(spec, str(exc)))
                                + "\n"
                            )
                            fh.flush()
```

Apply the identical block in the bare `except Exception as exc:` branch at line 274, using `f"unexpected {exc!r}"` as the message.

- [ ] **Step 5: Teach `build_outcomes` the rule**

In `src/packing/analyze.py`, change the signature at line 69 to accept the rule, and skip or score failure rows before the paired lookup:

```python
def build_outcomes(
    records: list[dict[str, Any]],
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    unevaluable_rule: str = "primary",
) -> list[Outcome]:
```

Immediately after the records are indexed by identity, partition them:

```python
    from .unevaluable import INFRASTRUCTURE, score_unevaluable

    # Infrastructure failures are replaced upstream, so they carry no information
    # and are dropped here. Model-output failures are scored under the rule.
    usable = [r for r in records if not r.get("failure")]
    model_failures = [
        r
        for r in records
        if r.get("failure") and r.get("failure_class") != INFRASTRUCTURE
    ]
    unevaluable_value = score_unevaluable(unevaluable_rule)
```

Then, after the loop that builds outcomes from `usable`, append one outcome per
model-output failure per criterion in its group when `unevaluable_value` is not
`None`, using `judge_a=judge_b=None` and `correct_num=unevaluable_value`. Where
`Outcome` requires judge scores, add an `unevaluable: bool = False` field to
`Outcome` in `src/packing/score.py` and set `correct_num` directly rather than
deriving it from scores.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_failure_records.py -q`
Expected: PASS, 5 tests

- [ ] **Step 7: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/runner.py src/packing/analyze.py src/packing/score.py tests/test_failure_records.py
git commit -m "analysis: score model-output failures instead of dropping them"
```

---

### Task 11: Equal-quota planning variance

**Files:**
- Modify: `src/packing/power.py` (add after `paired_mde`, which ends at line 90)
- Test: `tests/test_power_stratified.py`

**Interfaces:**
- Consumes: `_z` (`src/packing/power.py:29`), `stats` from `scipy` (already imported at `src/packing/power.py:24`), `resolved`, `independence_unit` (`src/packing/analyze.py:406,411`)
- Produces: `sigma_eff(frame: pd.DataFrame, reference_size: int, comparison_size: int, *, operations: tuple[str, ...]) -> float | None`, `stratified_sample_size(sigma: float, mde: float, *, strata: int, alpha: float = 0.05, power: float = 0.80) -> int`, `stratified_mde(sigma: float, n_total: int, *, strata: int, alpha: float = 0.05, power: float = 0.80) -> float`

Spec section 8 defines the planning variance as the average within-operation
variance, and section 8's variance review recomputes it from that formula. Neither
exists in code, so the review could not be run and the reported `sigma_eff` of
0.3345 was computed in an ad hoc script.

- [ ] **Step 1: Write the failing test**

```python
"""Planning variance for the equal-quota estimator. Under fixed quotas the
relevant variance is the average WITHIN-operation variance, not the variance
pooled across all differences, which also carries between-operation mean spread."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packing.power import sigma_eff, stratified_mde, stratified_sample_size

OPS = ("op1", "op2")


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "base_unit": u,
                "operation": o,
                "group_size": g,
                "correct_num": v,
                "judge_tie": False,
            }
            for u, o, g, v in rows
        ]
    )


def test_sigma_eff_ignores_between_operation_mean_spread():
    # Both strata have zero within variance but very different means. Pooled SD
    # would be large; the equal-quota planning variance is zero.
    rows = []
    for i in range(5):
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, 1.0)]
        rows += [(f"b{i}", "op2", 1, 0.0), (f"b{i}", "op2", 40, -1.0)]
    assert sigma_eff(_frame(rows), 1, 40, operations=OPS) == 0.0


def test_sigma_eff_averages_variances_not_standard_deviations():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, rng.normal(0, 0.2))]
        rows += [(f"b{i}", "op2", 1, 0.0), (f"b{i}", "op2", 40, rng.normal(0, 0.6))]
    expected = float(np.sqrt((0.2**2 + 0.6**2) / 2))
    got = sigma_eff(_frame(rows), 1, 40, operations=OPS)
    assert abs(got - expected) < 0.05, (got, expected)


def test_sigma_eff_returns_none_when_a_stratum_is_missing():
    rows = [("a0", "op1", 1, 0.0), ("a0", "op1", 40, 1.0)]
    assert sigma_eff(_frame(rows), 1, 40, operations=OPS) is None


def test_sample_size_matches_the_spec_figures():
    # Spec section 8: sigma_eff 0.3345 needs 354 for a 0.05 effect, and the
    # CI-high 0.3763 needs 447.
    assert stratified_sample_size(0.3345, 0.05, strata=8) == 354
    assert stratified_sample_size(0.3763, 0.05, strata=8) == 447


def test_mde_and_sample_size_round_trip_within_the_approximation_gap():
    # stratified_sample_size is exact and stratified_mde is the normal
    # approximation, so they agree closely but not identically. The gap is the
    # 352-against-354 difference at this sigma.
    n = stratified_sample_size(0.3345, 0.05, strata=8)
    assert abs(stratified_mde(0.3345, n, strata=8) - 0.05) < 0.001


def test_sample_size_rejects_a_nonpositive_effect():
    import pytest

    with pytest.raises(ValueError, match="mde must be > 0"):
        stratified_sample_size(0.3, 0.0, strata=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_power_stratified.py -q`
Expected: FAIL with `ImportError: cannot import name 'sigma_eff'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/packing/power.py`:

```python
def sigma_eff(
    frame: pd.DataFrame,
    reference_size: int,
    comparison_size: int,
    *,
    operations: tuple[str, ...],
) -> float | None:
    """Planning variance for the equal-quota estimator.

    Fixed quotas remove random variation in operation composition, so the pooled
    SD across all per-unit differences is the wrong input: it carries
    between-operation mean spread that the design has already eliminated. The
    right quantity is the average of the within-operation variances.

    Variances are averaged, not standard deviations, because the estimator's
    variance is a sum of s_o^2 / n_o terms.
    """
    from .analyze import independence_unit, resolved

    res = resolved(frame)
    cluster = independence_unit(res)
    per_unit = (
        res[res["group_size"].isin([reference_size, comparison_size])]
        .groupby([cluster, "operation", "group_size"])["correct_num"]
        .mean()
        .unstack("group_size")
    )
    if reference_size not in per_unit or comparison_size not in per_unit:
        return None
    paired = per_unit[[reference_size, comparison_size]].dropna()
    if paired.empty:
        return None
    diffs = (paired[comparison_size] - paired[reference_size]).rename("d")

    variances: list[float] = []
    for operation in operations:
        try:
            values = diffs.xs(operation, level="operation").to_numpy()
        except KeyError:
            return None
        if len(values) < 2:
            return None
        variances.append(float(np.var(values, ddof=1)))
    return float(np.sqrt(np.mean(variances)))


def stratified_sample_size(
    sigma: float,
    mde: float,
    *,
    strata: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Total units for the equal-quota estimator to detect ``mde``.

    With equal quotas n_o = n / O, the estimator variance reduces to
    sigma_eff^2 / n, so the total requirement does not depend on the number of
    strata. ``strata`` is validated rather than used, because a caller passing it
    is asserting the design it belongs to.

    Solved against the noncentral-t rather than the normal approximation, because
    the spec pre-registers the exact figures: at sigma_eff = 0.3345 the normal
    form gives 352 and the exact form 354, and the pre-registered number is 354.
    A planning function that disagrees with the protocol it plans is worse than
    useless.
    """
    if mde <= 0:
        raise ValueError("mde must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if strata < 1:
        raise ValueError("strata must be >= 1")
    crit = stats.norm.ppf(1 - alpha / 2)
    n = max(2, int((crit + stats.norm.ppf(power)) ** 2 * (sigma / mde) ** 2) - 5)
    while n < 1_000_000:
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        ncp = mde / (sigma / np.sqrt(n))
        achieved = 1 - stats.nct.cdf(t_crit, n - 1, ncp) + stats.nct.cdf(
            -t_crit, n - 1, ncp
        )
        if achieved >= power:
            return n
        n += 1
    raise ValueError("no sample size reached the requested power")


def stratified_mde(
    sigma: float,
    n_total: int,
    *,
    strata: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Detectable effect at ``n_total``, normal approximation.

    The approximation is used here rather than the exact form because this is the
    reporting direction: it is quoted alongside an interval, not used to fix a
    pre-registered sample size. `stratified_sample_size` is the exact one.
    """
    if n_total < 1:
        raise ValueError("n_total must be >= 1")
    if strata < 1:
        raise ValueError("strata must be >= 1")
    return float(_z(alpha, power) * sigma / np.sqrt(n_total))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_power_stratified.py -q`
Expected: PASS, 6 tests. `test_sample_size_matches_the_spec_figures` is the load-bearing one: it pins the plan against the numbers the spec pre-registers.

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/power.py tests/test_power_stratified.py
git commit -m "analysis: add the equal-quota planning variance"
```

---

### Task 12: Interleave call order across rungs

**Files:**
- Modify: `src/packing/runner.py` (`plan_fixed_position` from Task 5)
- Test: `tests/test_ladder_plan.py` (extend)

**Interfaces:**
- Consumes: `plan_fixed_position` from Task 5
- Produces: same signature, with specs returned in a seeded interleaved order

Spec section 12 requires call order to be randomised or interleaved across K so
that provider drift cannot align with load. As written in Task 5 the planner emits
all rungs for prompt 1, then all rungs for prompt 2, so a mid-run change in
provider behaviour would land disproportionately on whichever rungs were in
flight, and K = 1 and K = 40 for the same prompt are always adjacent.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ladder_plan.py`:

```python
def test_call_order_is_interleaved_across_rungs():
    items = [_item(f"i{i}", "precision") for i in range(12)]
    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model="mock", levels=LEVELS
    )
    levels_in_order = [s.nominal_n for s in specs]
    # A prompt-major order would emit each prompt's five rungs consecutively, so
    # runs of identical nominal_n would be short and the sequence would cycle
    # through levels in blocks. Interleaving mixes them.
    blocks = 1
    for a, b in zip(levels_in_order, levels_in_order[1:]):
        if a != b:
            blocks += 1
    assert blocks > len(items) * 2, blocks


def test_interleaving_is_deterministic_for_a_seed():
    items = [_item(f"i{i}", "precision") for i in range(8)]
    first = [
        s.identity
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=42
        )
    ]
    second = [
        s.identity
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=42
        )
    ]
    assert first == second


def test_interleaving_changes_with_the_seed():
    items = [_item(f"i{i}", "precision") for i in range(8)]
    a = [
        s.identity
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=1
        )
    ]
    b = [
        s.identity
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=2
        )
    ]
    assert a != b


def test_interleaving_preserves_the_spec_set():
    items = [_item(f"i{i}", "precision") for i in range(6)]
    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=3
    )
    assert len(specs) == 6 * len(LEVELS) * 2
    assert len({s.identity for s in specs}) == len(specs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_plan.py -q`
Expected: FAIL on `test_call_order_is_interleaved_across_rungs`, and both seed tests pass trivially because the order is currently deterministic and seed-independent. Confirm the interleave test is the one failing.

- [ ] **Step 3: Add the shuffle**

At the end of `plan_fixed_position`, before `return specs`, replace the return with:

```python
    # Interleave across rungs so provider drift cannot align with load. Resume
    # keys on identity rather than order (`runner.py:146-164`), so reordering is
    # safe, and a seeded shuffle keeps the run reproducible.
    random.Random(partition_seed("ladder-order", 0, 0, base_seed)).shuffle(specs)
    return specs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_plan.py -q`
Expected: PASS, 14 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/runner.py tests/test_ladder_plan.py
git commit -m "feat: interleave ladder call order across rungs"
```

---

## Not in this plan

- **Live execution.** Task 9 refuses it by design. Spending is a separate decision after the manifest-based recompute.
- **The blinded variance review itself** (spec section 8). Task 11 supplies the variance function it calls, but the review runs after 160 complete core prompts, so it needs real data. Its null simulation is a separate plan.
- **`rename` and `scramble_list`.** Spec section 11 assigns them retained pilot data only, no prompts, no quota, no calls.
- **Deck corrections.** `presentation/index.html` slide 14 still frames load against position as unadjudicated, and its two MDE cards are wrong. Tracked separately from the experiment machinery.
