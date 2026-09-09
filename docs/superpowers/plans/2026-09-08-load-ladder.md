# Fixed-Position Load Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the experiment machinery for a K = 40 load ladder in which the targeted criterion is pinned at position 0, so load varies without the position confound that made the K = 20 sweep unadjudicable.

**Architecture:** Five new focused modules plus three surgical extensions to existing ones. A criterion manifest (40 criteria, validated and hashed) feeds an eligibility pre-pass, which feeds a frozen-seed lane assignment, which feeds a fixed-position nested planner. Analysis adds a stratified equal-weight estimator with Welch-Satterthwaite degrees of freedom and an explicit unevaluable-output scoring rule. Nothing calls a provider until Task 9, which is dry-run only.

**Tech Stack:** Python 3.12, `uv`, `pytest`, `pandas`, `numpy`, `scipy`, `click`. OpenAI via strict Structured Outputs.

**Spec:** `docs/superpowers/specs/2026-09-08-load-ladder-design.md`

## Execution order

Run the tasks in this order:

    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 9

Task 0 first, because Tasks 3 and 6 import the corruption registry and must not
reach it through the command-line layer. **Task 9, the CLI, runs last**: it wires
together every other component, including the sampling-plan artifact from Task 13,
so building it earlier would mean building it twice. Task numbering is left as
written so that cross-references stay stable; the sequence above is authoritative.

## Global Constraints

- No outcome-generating provider call in any task. Task 9 is `--dry-run` only. Spending is a separate decision made after the manifest-based cost recompute.
- Every criterion anchors **every** scale point. `Criterion.__post_init__` (`src/packing/criteria.py:38-52`) already raises otherwise; do not weaken it.
- No `temperature`. `reasoning_effort` fixed. Both already enforced in `src/packing/providers/openai_provider.py:61-72`.
- Determinism: every random choice derives from `partition_seed(...)` (`src/packing/partition.py:150-158`) or an explicitly passed seed. No bare `random` / `np.random` calls without a seed.
- Arm assignment happens **before** eligibility is inspected. Reversing this order silently changes the core population (spec §5).
- One corruption realisation per prompt, frozen and reused at every rung.
- No em dashes in any prose, docstring, or comment.
- Run `uv run pytest -q` before every commit. All 175 existing tests must stay green.
- An unrecognised failure is **fatal**, never scored. A two-class scheme scores its
  own bugs as science: an SDK change or an unsupported option would otherwise
  arrive as evidence that load degraded judging.
- Recovery is a **paired** outcome. Exactly one outcome per (prompt, rung), never
  one per call, or a prompt whose two sides both fail contributes twice.
- The response side that satisfied the full arm operation set is frozen at
  screening and reused. Re-deriving it from the assigned operation picks a side
  eligible for that operation but not for all eight, silently changing the
  population from `A_core = 1`.
- Persist the full SHA-256 of the manifest, not a display prefix.

---

### Task 0: Neutral corruption registry

**Files:**
- Create: `src/packing/registry.py`
- Modify: `src/packing/cli.py:561` (import from the registry instead of defining)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `CORRUPTIONS` (`src/packing/corrupt.py:406`), `EXTRA_CORRUPTIONS` (`src/packing/corrupt_extra.py:498`)
- Produces: `ALL_CORRUPTIONS: tuple[Corruption, ...]`, `corruption_for(key: str) -> Corruption`, `CORRUPTION_KEYS: frozenset[str]`

`ALL_CORRUPTIONS` currently lives in `cli.py:561`. Eligibility screening and
dataset construction both need it, and importing the command-line layer from a
data module creates an import cycle that Task 3 would otherwise paper over with a
function-local import.

- [ ] **Step 1: Write the failing test**

```python
"""The corruption registry, importable without pulling in the CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from packing.registry import ALL_CORRUPTIONS, CORRUPTION_KEYS, corruption_for


def test_registry_holds_the_eighteen_surviving_operations():
    # METHODS 3.4 catalogues twenty. Two were removed after calibration
    # (METHODS 10, `unsupport` on discrimination and `vague` for inverting), so
    # the live catalogue is eighteen. `jargon` stays in the pool as packing load
    # and is excluded from measurement by MUTE_OPERATIONS rather than removed.
    assert len(ALL_CORRUPTIONS) == 18
    assert len(CORRUPTION_KEYS) == 18
    assert "unsupport" not in CORRUPTION_KEYS
    assert "vague" not in CORRUPTION_KEYS
    assert "jargon" in CORRUPTION_KEYS


def test_lookup_by_key():
    assert corruption_for("shuffle").targeted_criterion == "coherence"
    assert corruption_for("numbers").targeted_criterion == "correctness"


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="nope"):
        corruption_for("nope")


def test_importing_the_registry_does_not_import_the_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import packing.registry, sys; "
            "sys.exit(1 if 'packing.cli' in sys.modules else 0)",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, "registry must not pull in the CLI layer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
"""The corruption catalogue, in one neutral place.

Eligibility screening and dataset construction both need the catalogue. Reaching
it through the command-line module would make data code depend on the CLI, so it
lives here and `cli.py` imports it like everything else.
"""

from __future__ import annotations

from .corrupt import CORRUPTIONS, Corruption
from .corrupt_extra import EXTRA_CORRUPTIONS

ALL_CORRUPTIONS: tuple[Corruption, ...] = CORRUPTIONS + EXTRA_CORRUPTIONS
CORRUPTION_KEYS: frozenset[str] = frozenset(c.key for c in ALL_CORRUPTIONS)
_BY_KEY: dict[str, Corruption] = {c.key: c for c in ALL_CORRUPTIONS}


def corruption_for(key: str) -> Corruption:
    if key not in _BY_KEY:
        raise KeyError(f"unknown corruption {key!r}; known: {sorted(_BY_KEY)}")
    return _BY_KEY[key]
```

Then in `src/packing/cli.py`, delete the `ALL_CORRUPTIONS = CORRUPTIONS + EXTRA_CORRUPTIONS` line at 561 and import it instead: `from .registry import ALL_CORRUPTIONS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/registry.py src/packing/cli.py tests/test_registry.py
git commit -m "refactor: move the corruption registry out of the CLI layer"
```

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
    assert len(a.hash) == 16, "display prefix"
    assert len(a.full_hash) == 64, "persist the full digest"
    assert a.full_hash.startswith(a.hash)


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
        """Display prefix of `full_hash`. Persist `full_hash`, show this."""
        return self.full_hash[:16]

    @property
    def full_hash(self) -> str:
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
        return hashlib.sha256(payload.encode()).hexdigest()

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


def test_length_distribution_of_the_new_twenty_is_bounded():
    # Scoped to the twenty this task authors. The full manifest includes legacy
    # criteria the task cannot alter, so bounding the whole pool would fail for
    # reasons outside its control.
    lengths = [
        len(c.description) + sum(len(v) for v in c.anchors.values())
        for c in LADDER_ONLY_CRITERIA
    ]
    assert max(lengths) <= 4 * min(lengths), sorted(lengths)


def test_serialised_token_length_is_recorded_for_the_pool():
    # Characters are a proxy; the rendered rubric is what is billed. Recording
    # both keeps the cost model honest when anchor prose changes.
    rendered = [len(c.render()) // 4 for c in LADDER_ONLY_CRITERIA]
    assert min(rendered) > 0
    assert max(rendered) <= 4 * min(rendered), sorted(rendered)


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
- Consumes: `PairedItem` from `src/packing/dataset.py:25`, `corruption_for` from Task 0, `partition_seed` from `src/packing/partition.py:150`
- Produces: `CORE_OPERATIONS: tuple[str, ...]` (8 keys), `NUMBERS_OPERATION: str`, `applies(op_key: str, text: str, prompt: str, *, seed: int) -> bool`, `select_response(item: PairedItem, op_keys: tuple[str, ...], *, seed: int) -> str | None` returning `"a"`, `"b"` or `None`

**The returned side must be persisted by the caller.** It is the side that
satisfied the *whole* operation set. Calling `select_response` again later with a
single assigned operation can return the other side, which is eligible for that
operation but not for all eight, and that silently changes the population from
`A_core = 1`. Task 9 stores it; nothing re-derives it.

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
from .registry import corruption_for

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


def applies(op_key: str, text: str, prompt: str, *, seed: int) -> bool:
    """Does this operation change this text?"""
    corruption = corruption_for(op_key)
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

### Task 4: Exact lane split with seeded within-lane order

**Files:**
- Create: `src/packing/lanes.py`
- Test: `tests/test_lanes.py`

**Interfaces:**
- Consumes: `partition_seed` from `src/packing/partition.py:150`
- Produces: `split_lanes(item_ids: list[str], *, seed: int, core_candidates: int) -> tuple[list[str], list[str]]` returning `(core_stream, numbers_stream)` both in seeded order; `fill_lane(item_ids: list[str], eligible: Callable[[str], bool], quota: int) -> tuple[list[str], list[str]]` returning `(filled, remainder)`

Lane assignment runs before eligibility is inspected. Reversing that order changes
the core population, because `numbers` eligibility correlates with response
content (spec section 5).

Two properties the design requires and a hash-threshold assignment would not give:

1. **Exact lane sizes.** A per-prompt hash threshold makes lane membership an
   independent draw, so a 1,222-prompt frame yields roughly 508 core candidates
   rather than exactly 508. The 97.5% fill figures in the spec are computed for
   fixed lane sizes, and adding lane-count variance invalidates them. An index
   split of the permuted frame is exact.
2. **Seeded order within a lane.** Filling in source-file order lets any structure
   in the input file (collection, topic, score) select the sample. Lane membership
   being order-independent is not enough; the order the lane is consumed in has to
   be seeded too.

- [ ] **Step 1: Write the failing test**

```python
"""Lanes are exact index splits of a seed-permuted frame, and each lane is
consumed in seeded order rather than file order."""

from __future__ import annotations

from packing.lanes import fill_lane, split_lanes


def test_lane_sizes_are_exact():
    ids = [f"p{i}" for i in range(1222)]
    core, numbers = split_lanes(ids, seed=11, core_candidates=508)
    assert len(core) == 508
    assert len(numbers) == 714


def test_lanes_are_disjoint_and_cover_the_frame():
    ids = [f"p{i}" for i in range(100)]
    core, numbers = split_lanes(ids, seed=11, core_candidates=40)
    assert set(core) & set(numbers) == set()
    assert set(core) | set(numbers) == set(ids)


def test_split_is_deterministic_for_a_seed():
    ids = [f"p{i}" for i in range(100)]
    assert split_lanes(ids, seed=11, core_candidates=40) == split_lanes(
        ids, seed=11, core_candidates=40
    )


def test_split_changes_with_the_seed():
    ids = [f"p{i}" for i in range(100)]
    a = split_lanes(ids, seed=11, core_candidates=40)
    b = split_lanes(ids, seed=12, core_candidates=40)
    assert a != b


def test_split_is_independent_of_input_order():
    ids = [f"p{i}" for i in range(200)]
    a = split_lanes(ids, seed=11, core_candidates=80)
    b = split_lanes(list(reversed(ids)), seed=11, core_candidates=80)
    assert a == b, "lane content must not depend on file order"


def test_within_lane_order_is_not_file_order():
    ids = [f"p{i:04d}" for i in range(400)]
    core, _ = split_lanes(ids, seed=11, core_candidates=200)
    assert core != sorted(core), "consumption order must be seeded, not sorted"


def test_frame_too_small_for_the_requested_core_lane_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="frame of 10 cannot supply 20"):
        split_lanes([f"p{i}" for i in range(10)], seed=1, core_candidates=20)


def test_duplicate_ids_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="duplicate base-prompt ids"):
        split_lanes(["p1", "p1", "p2"], seed=1, core_candidates=1)


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
"""Split the fetched frame into arm-specific streams before eligibility is
inspected.

The order matters and is the whole reason this module exists. Screening one shared
frame and then carving the numbers quota out of it leaves the core pool 4.8%
numbers-eligible against a population value of 31.0%, because numbers eligibility
correlates with response content. The primary estimate would then describe
"core-eligible prompts not allocated to numbers" rather than the population the
estimand names.

The split is an exact index cut of a seed-permuted frame rather than a per-prompt
hash threshold, so lane sizes are the planned 508 and 714 rather than draws around
them. The spec's fill probabilities assume fixed lane sizes.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .partition import partition_seed


def split_lanes(
    item_ids: list[str], *, seed: int, core_candidates: int
) -> tuple[list[str], list[str]]:
    """Exact, seed-permuted, disjoint core and numbers streams.

    Both returned lists are already in the order they should be consumed, so a
    caller never needs to sort and cannot accidentally fall back to file order.
    """
    if len(set(item_ids)) != len(item_ids):
        seen: set[str] = set()
        dupes = sorted({i for i in item_ids if i in seen or seen.add(i)})
        raise ValueError(f"duplicate base-prompt ids: {dupes[:5]}")
    if core_candidates < 0:
        raise ValueError(f"core_candidates must be >= 0, got {core_candidates}")
    if core_candidates > len(item_ids):
        raise ValueError(
            f"frame of {len(item_ids)} cannot supply {core_candidates} core candidates"
        )

    # Sort first so the permutation is a function of ids and seed alone, never of
    # the order the file happened to list them in.
    ordered = sorted(item_ids)
    random.Random(partition_seed("lane-split", 0, 0, seed)).shuffle(ordered)
    core = ordered[:core_candidates]
    numbers = ordered[core_candidates:]

    # Consumption order inside each lane is seeded separately, so extending one
    # lane cannot reshuffle the other.
    random.Random(partition_seed("core-order", 0, 0, seed)).shuffle(core)
    random.Random(partition_seed("numbers-order", 0, 0, seed)).shuffle(numbers)
    return core, numbers


def fill_lane(
    item_ids: list[str],
    eligible: Callable[[str], bool],
    quota: int,
) -> tuple[list[str], list[str]]:
    """Take the first `quota` eligible prompts in the stream's frozen order.

    Returns the filled quota and the untouched remainder, so a later variance
    review extends the sample by continuing where this stopped rather than by
    drawing afresh.
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
Expected: PASS, 11 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/lanes.py tests/test_lanes.py
git commit -m "feat: split arm lanes exactly, before eligibility screening"
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
    pad the call, which removes the ordinal-position confound.

    Companions are a nested prefix of one seeded permutation of the non-target
    criteria, shared by both sides. Without nesting, K would change both the amount
    of trailing load and which particular criteria are present, putting composition
    noise into the endpoint contrast.

    This does not make load the only thing that varies. The target stays first, so
    K also varies the amount of trailing material, the target's recency, and its
    distance from the end of the context. Those stay coupled by design and the
    estimand is written to match.
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

### Task 6: Extendable operation queues and one frozen realisation per prompt

**Files:**
- Create: `src/packing/ladder_dataset.py`
- Test: `tests/test_ladder_dataset.py`

**Interfaces:**
- Consumes: `PairedItem` (`src/packing/dataset.py:25`), `corruption_for` from Task 0, `partition_seed`, `Criterion`
- Produces: `operation_queues(item_ids: list[str], op_keys: tuple[str, ...], *, seed: int, per_op_max: int) -> dict[str, list[str]]`; `activate(queues: dict[str, list[str]], per_op: int) -> dict[str, str]` mapping item id to operation; `build_ladder_item(base: PairedItem, op_key: str, side: str, criteria: tuple[Criterion, ...], *, seed: int) -> PairedItem | None`

A single shuffle-and-modulo assignment cannot survive the variance review. If the
first stage assigns 464 prompts by position in a shuffled list, then the review
raises the target to 520, re-running the assignment reshuffles everything and
prompts whose outcomes are already collected can receive a different operation.
The review is increase-only and keeps first-stage prompts in the analysis, so
assignment has to be stable under extension.

Queues solve it: deal the frame into one queue per operation once, sized to the
768-unit maximum, and activate a prefix of each. Raising `per_op` reveals more
units without touching any prior assignment.

- [ ] **Step 1: Write the failing test**

```python
"""Operation assignment must be stable under extension, and each prompt gets one
frozen corruption realisation reused at every rung."""

from __future__ import annotations

from collections import Counter

from packing.dataset import PairedItem
from packing.eligibility import CORE_OPERATIONS
from packing.ladder_criteria import LADDER_CRITERIA
from packing.ladder_dataset import activate, build_ladder_item, operation_queues

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


def test_queues_are_equal_sized_and_disjoint():
    ids = [f"p{i}" for i in range(96 * 8)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=96)
    assert set(queues) == set(CORE_OPERATIONS)
    assert all(len(q) == 96 for q in queues.values())
    flat = [i for q in queues.values() for i in q]
    assert len(set(flat)) == len(flat)


def test_activation_yields_exact_equal_quotas():
    ids = [f"p{i}" for i in range(96 * 8)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=96)
    assigned = activate(queues, per_op=58)
    counts = Counter(assigned.values())
    assert set(counts) == set(CORE_OPERATIONS)
    assert all(c == 58 for c in counts.values())
    assert len(assigned) == 58 * 8


def test_extending_does_not_reassign_any_earlier_prompt():
    ids = [f"p{i}" for i in range(96 * 8)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=96)
    first = activate(queues, per_op=58)
    second = activate(queues, per_op=65)
    for item_id, op in first.items():
        assert second[item_id] == op, item_id
    assert len(second) == 65 * 8


def test_queues_are_deterministic_for_a_seed():
    ids = [f"p{i}" for i in range(80)]
    assert operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10) == (
        operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10)
    )


def test_a_frame_too_small_for_the_maximum_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="needs 800 units, frame has 80"):
        operation_queues([f"p{i}" for i in range(80)], CORE_OPERATIONS,
                         seed=5, per_op_max=100)


def test_activating_beyond_the_queue_is_rejected():
    import pytest

    ids = [f"p{i}" for i in range(80)]
    queues = operation_queues(ids, CORE_OPERATIONS, seed=5, per_op_max=10)
    with pytest.raises(ValueError, match="per_op 11 exceeds queue depth 10"):
        activate(queues, per_op=11)


def test_realisation_is_identical_across_repeated_builds():
    base = _base("p1")
    first = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    second = build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert first is not None
    assert first.response_a == second.response_a
    assert first.response_b == second.response_b


def test_the_targeted_criterion_is_the_operations_target():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.targeted_criterion == "coherence"


def test_base_item_key_is_set_for_clustering():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.meta["base_item"] == "p1"


def test_inapplicable_operation_yields_none_rather_than_an_unchanged_item():
    base = PairedItem(
        item_id="p2", prompt="p", response_a="Short.", response_b="Short.",
        human_a={}, human_b={}, dataset="test", meta={},
    )
    assert build_ladder_item(base, "shuffle", "a", LADDER_CRITERIA, seed=9) is None


def test_only_the_selected_side_is_corrupted():
    item = build_ladder_item(_base("p1"), "shuffle", "a", LADDER_CRITERIA, seed=9)
    assert item.response_a != TEXT
    assert item.response_b == TEXT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.ladder_dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Build ladder units: one operation, one realisation, both frozen per prompt.

Assignment is queue-based rather than shuffle-and-modulo so that it survives the
increase-only variance review. Dealing into per-operation queues once and
activating a prefix means raising the target reveals new units without changing
any assignment already collected.

The realisation is frozen because it is reused at every rung. Redrawn per rung,
the paired difference between K = 1 and K = 40 would carry realisation noise on
top of the load effect and the estimator could not separate them.
"""

from __future__ import annotations

import random

from .criteria import Criterion
from .dataset import PairedItem
from .partition import partition_seed
from .registry import corruption_for


def operation_queues(
    item_ids: list[str],
    op_keys: tuple[str, ...],
    *,
    seed: int,
    per_op_max: int,
) -> dict[str, list[str]]:
    """Deal the frame into one fixed queue per operation, sized to the maximum."""
    if not op_keys:
        raise ValueError("op_keys must be non-empty")
    if per_op_max < 1:
        raise ValueError(f"per_op_max must be >= 1, got {per_op_max}")
    needed = per_op_max * len(op_keys)
    if len(item_ids) < needed:
        raise ValueError(
            f"queue plan needs {needed} units, frame has {len(item_ids)}"
        )
    ordered = sorted(item_ids)
    random.Random(partition_seed("operation-queues", 0, 0, seed)).shuffle(ordered)
    return {
        op: ordered[index * per_op_max : (index + 1) * per_op_max]
        for index, op in enumerate(op_keys)
    }


def activate(queues: dict[str, list[str]], per_op: int) -> dict[str, str]:
    """Reveal the first `per_op` units of every queue.

    Monotone in `per_op` by construction: a larger value returns a superset with
    identical operations for every unit already returned.
    """
    if per_op < 0:
        raise ValueError(f"per_op must be >= 0, got {per_op}")
    for op, queue in queues.items():
        if per_op > len(queue):
            raise ValueError(
                f"per_op {per_op} exceeds queue depth {len(queue)} for {op}"
            )
    return {
        item_id: op
        for op, queue in queues.items()
        for item_id in queue[:per_op]
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
    carrying a ground-truth claim, which would be a false label. Task 14 treats a
    None here as an invariant violation, because eligibility was screened first.
    """
    corruption = corruption_for(op_key)
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

`meta["base_item"]` is the exact key that matters: `analyze.py:164` derives the
clustering column `base_unit` from it, and `independence_unit` (`analyze.py:413`)
falls back to `item_id` when it is absent, which would treat every corrupted item
as its own cluster and silently inflate n.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_dataset.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/ladder_dataset.py tests/test_ladder_dataset.py
git commit -m "feat: extendable operation queues and frozen realisations"
```

---

### Task 7: Three failure classes and the composite outcome value

**Files:**
- Create: `src/packing/unevaluable.py`
- Test: `tests/test_unevaluable.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `TRANSIENT: str`, `UNEVALUABLE: str`, `FATAL: str`; `classify_failure(message: str) -> str`; `FatalProtocolError(RuntimeError)`; `unevaluable_value(rule: str, group_size: int, reference_size: int) -> float | None`

Two classes are not enough. A substring classifier that defaults unknown failures
to "model output" scores its own bugs as science: an SDK change, an unsupported
option, or a stray `KeyError` would arrive as evidence that load degraded judging.
So an unrecognised failure is **fatal** and aborts the run. The burden sits on the
harness to classify, not on the estimate to absorb.

Timeouts split by phase rather than by keyword. A connection timeout is transient.
A read or generation timeout is unevaluable, because generation time plausibly
rises with K and replacing those calls would erase part of the treatment effect.

- [ ] **Step 1: Write the failing test**

```python
"""Three failure classes. Transient failures are replaced, unevaluable ones are
scored, and anything unrecognised aborts rather than becoming a quiet miss."""

from __future__ import annotations

import pytest

from packing.unevaluable import (
    FATAL,
    TRANSIENT,
    UNEVALUABLE,
    classify_failure,
    unevaluable_value,
)


@pytest.mark.parametrize(
    "message",
    [
        "Connection reset by peer",
        "connection aborted",
        "rate limit exceeded, retry after 2s",
        "provider returned 503 Service Unavailable",
        "502 Bad Gateway",
        "connect timeout after 5s",
    ],
)
def test_transient_infrastructure(message):
    assert classify_failure(message) == TRANSIENT


@pytest.mark.parametrize(
    "message",
    [
        "gpt-5-mini: truncated at max_completion_tokens=9352 for group_size=40",
        "content did not parse as JSON",
        "empty content returned",
        "context_length_exceeded: 41000 tokens > 32768",
        "content_filter triggered on the response",
        "model refused to answer",
        "read timeout while streaming the completion",
        "generation timeout after 120s",
    ],
)
def test_treatment_or_system_unevaluable(message):
    assert classify_failure(message) == UNEVALUABLE


@pytest.mark.parametrize(
    "message",
    [
        "invalid_request_error: unsupported parameter",
        "criterion key set mismatch: got {'a'} expected {'b'}",
        "unexpected KeyError('scores')",
        "TypeError: got an unexpected keyword argument",
        "something nobody anticipated",
    ],
)
def test_fatal_protocol_and_configuration(message):
    assert classify_failure(message) == FATAL


def test_unrecognised_is_fatal_not_unevaluable():
    # The conservative direction is to stop, not to score. Scoring an unknown
    # failure as a miss turns a harness bug into a load effect.
    assert classify_failure("???") == FATAL


def test_primary_rule_scores_unevaluable_as_zero_at_both_endpoints():
    assert unevaluable_value("primary", 1, 1) == 0.0
    assert unevaluable_value("primary", 40, 1) == 0.0


def test_lower_bound_imputes_oppositely_at_the_endpoints():
    # d = Y40 - Y1, so the lower bound on d needs Y1 high and Y40 low.
    assert unevaluable_value("lower_bound", 1, 1) == 1.0
    assert unevaluable_value("lower_bound", 40, 1) == 0.0


def test_upper_bound_imputes_the_other_way():
    assert unevaluable_value("upper_bound", 1, 1) == 0.0
    assert unevaluable_value("upper_bound", 40, 1) == 1.0


def test_complete_case_excludes_the_observation():
    assert unevaluable_value("complete_case", 40, 1) is None


def test_a_single_constant_would_not_bound_the_difference():
    # Regression guard for the bug this replaced: assigning every unevaluable Y
    # the same value leaves d unchanged and bounds nothing.
    lower = (
        unevaluable_value("lower_bound", 40, 1)
        - unevaluable_value("lower_bound", 1, 1)
    )
    upper = (
        unevaluable_value("upper_bound", 40, 1)
        - unevaluable_value("upper_bound", 1, 1)
    )
    assert lower == -1.0
    assert upper == 1.0
    assert lower < upper


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown rule"):
        unevaluable_value("whatever", 40, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_unevaluable.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.unevaluable'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Failure classification and the composite-outcome value for unevaluable output.

Three classes, because two conflate a harness bug with a treatment effect:

1. TRANSIENT: exogenous and content-independent. Retried, then the prompt is
   replaced from its own lane. Never scored.
2. UNEVALUABLE: plausibly caused by the request itself, so part of the treatment.
   Retained and scored, because replacing it would condition the analysis on
   surviving high load.
3. FATAL: the harness or its configuration is wrong. The run aborts.

An unrecognised failure is FATAL. Defaulting the unknown case to UNEVALUABLE would
let an SDK change or an unsupported option arrive as evidence that load degraded
judging, which is the one error mode no amount of sample size fixes.
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

    Order matters: the transient markers are checked first so that a connect
    timeout is not caught by the generic timeout language in the unevaluable set.
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
    unevaluable Y leaves d untouched and bounds nothing, which is the bug this
    signature exists to prevent: the rule needs to know which rung it is scoring.

    `primary` is not an imputation. Y is defined as demonstrated,
    machine-readable directional recovery, so an unreadable judgment is a 0 by
    definition.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_unevaluable.py -q`
Expected: PASS, 27 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/unevaluable.py tests/test_unevaluable.py
git commit -m "feat: three failure classes and endpoint-aware sensitivity bounds"
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
- Produces: `packing ladder` click command with options `--items`, `--model`, `--core-n` (464), `--numbers-n` (184), `--core-candidates` (508), `--per-op-max` (96), `--levels` (`1,5,10,20,40`), `--seed` (20260908), `--plan-out` (`runs/ladder-plan.json`), `--allow-short` (flag, development only), `--dry-run/--no-dry-run` (default dry). Writes a `SamplingPlan` from Task 13.

This task runs **last**: it wires together every other component. See the execution order at the top.

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

Add to `src/packing/cli.py`. Note the guards: every one of them exists because
its absence would silently change the experiment rather than fail.

```python
@cli.command("ladder")
@click.option("--items", "items_path", type=click.Path(), default="data/helpsteer2.jsonl", show_default=True)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--core-n", type=int, default=464, show_default=True)
@click.option("--numbers-n", type=int, default=184, show_default=True)
@click.option("--core-candidates", type=int, default=508, show_default=True)
@click.option("--per-op-max", type=int, default=96, show_default=True)
@click.option("--levels", default="1,5,10,20,40", show_default=True)
@click.option("--seed", type=int, default=20260908, show_default=True)
@click.option("--plan-out", type=click.Path(), default="runs/ladder-plan.json", show_default=True)
@click.option("--allow-short", is_flag=True, default=False,
              help="Development only. Never use for the pre-registered budget.")
@click.option("--dry-run/--no-dry-run", default=True, show_default=True)
def ladder_cmd(
    items_path: str,
    model: str,
    core_n: int,
    numbers_n: int,
    core_candidates: int,
    per_op_max: int,
    levels: str,
    seed: int,
    plan_out: str,
    allow_short: bool,
    dry_run: bool,
) -> None:
    """The fixed-position load ladder: does trailing load change recovery?"""
    from .eligibility import CORE_OPERATIONS, NUMBERS_OPERATION, select_response
    from .ladder_criteria import LADDER_CRITERIA, LADDER_MANIFEST
    from .ladder_dataset import activate, build_ladder_item, operation_queues
    from .lanes import fill_lane, split_lanes
    from .runner import plan_fixed_position
    from .sampling_plan import SamplingPlan, Unit, file_sha256, write_plan

    LADDER_MANIFEST.validate(expect=40)

    # Pricing must not silently become zero. A missing catalogue entry is a
    # configuration error, not a free model.
    if model not in CATALOG:
        raise click.ClickException(
            f"no pricing for {model!r}; add it to CATALOG before costing a run"
        )
    entry = CATALOG[model]

    swept = tuple(int(x) for x in levels.split(",") if x.strip())
    if len(set(swept)) != len(swept):
        raise click.ClickException(f"levels must be unique, got {swept}")
    if swept[0] != 1 or swept[-1] != len(LADDER_CRITERIA):
        raise click.ClickException(
            f"levels must run from 1 to {len(LADDER_CRITERIA)}; got {swept}"
        )
    if core_n % len(CORE_OPERATIONS) != 0:
        raise click.ClickException(
            f"core-n {core_n} is not divisible by {len(CORE_OPERATIONS)}; equal "
            "quotas are part of the primary estimand"
        )

    path = Path(items_path)
    if not path.exists():
        raise click.ClickException(f"{path} not found; run `packing fetch` first")
    bases = read_jsonl(path)
    ids = [b.item_id for b in bases]
    if len(set(ids)) != len(ids):
        raise click.ClickException("duplicate base-prompt ids in the input file")
    by_id = {b.item_id: b for b in bases}

    core_stream, numbers_stream = split_lanes(
        ids, seed=seed, core_candidates=core_candidates
    )

    # Screen once, and KEEP the side. The side that satisfied all eight core
    # operations is the one that put this prompt in A_core = 1. Re-deriving it
    # later from the assigned operation alone can pick the other response, which
    # is eligible for that operation but not for all eight.
    core_sides = {
        item_id: select_response(by_id[item_id], CORE_OPERATIONS, seed=seed)
        for item_id in core_stream
    }
    numbers_sides = {
        item_id: select_response(by_id[item_id], (NUMBERS_OPERATION,), seed=seed)
        for item_id in numbers_stream
    }

    core_ids, _ = fill_lane(
        core_stream, lambda i: core_sides[i] is not None, core_n
    )
    numbers_ids, _ = fill_lane(
        numbers_stream, lambda i: numbers_sides[i] is not None, numbers_n
    )

    console.print(
        f"K={len(LADDER_CRITERIA)} pool, manifest {LADDER_MANIFEST.hash}, "
        f"lengths {LADDER_MANIFEST.length_stats}"
    )
    console.print(
        f"core lane: {len(core_stream)} candidates, {len(core_ids)}/{core_n} filled"
    )
    console.print(
        f"numbers lane: {len(numbers_stream)} candidates, "
        f"{len(numbers_ids)}/{numbers_n} filled"
    )
    short = len(core_ids) < core_n or len(numbers_ids) < numbers_n
    if short and not allow_short:
        raise click.ClickException(
            "lane short of quota. A partial sample costs and plans a different "
            "experiment, so this is an error rather than a warning. Fetch a larger "
            "frame, or pass --allow-short for development only."
        )
    if short:
        console.print("[yellow]SHORT SAMPLE: development only, not the budget[/yellow]")

    queues = operation_queues(
        core_ids, CORE_OPERATIONS, seed=seed, per_op_max=per_op_max
    ) if len(core_ids) >= per_op_max * len(CORE_OPERATIONS) else operation_queues(
        core_ids, CORE_OPERATIONS, seed=seed,
        per_op_max=len(core_ids) // len(CORE_OPERATIONS),
    )
    assigned = activate(queues, per_op=min(
        core_n // len(CORE_OPERATIONS), len(next(iter(queues.values())))
    ))

    items: list = []
    units: list[Unit] = []
    for item_id, op in assigned.items():
        side = core_sides[item_id]
        if side is None:
            raise click.ClickException(
                f"{item_id} passed screening with no side; invariant violated"
            )
        built = build_ladder_item(by_id[item_id], op, side, LADDER_CRITERIA, seed=seed)
        if built is None:
            # Screening already proved the operation applies to this side, so a
            # None here means the two paths disagree. That is a bug, not a
            # droppable item: skipping it would break equal quotas.
            raise click.ClickException(
                f"{item_id}/{op}/{side} screened eligible but did not build"
            )
        items.append(built)
        units.append(Unit(item_id, "core", side, True, op,
                          built.meta["realisation_seed"], seed, "active"))
    for item_id in numbers_ids:
        side = numbers_sides[item_id]
        built = build_ladder_item(
            by_id[item_id], NUMBERS_OPERATION, side, LADDER_CRITERIA, seed=seed
        )
        if built is None:
            raise click.ClickException(
                f"{item_id}/numbers/{side} screened eligible but did not build"
            )
        items.append(built)
        units.append(Unit(item_id, "numbers", side, True, NUMBERS_OPERATION,
                          built.meta["realisation_seed"], seed, "active"))

    # Post-build quota check is mandatory: a silent shortfall destroys the equal
    # weighting the primary estimand assumes.
    from collections import Counter
    built_counts = Counter(u.operation for u in units if u.lane == "core")
    if not allow_short and len(set(built_counts.values())) > 1:
        raise click.ClickException(f"unequal core quotas after build: {dict(built_counts)}")

    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model=model, levels=swept, base_seed=seed
    )
    est = estimate(specs)
    projection = project_cost(est, entry["in"], entry["out"])
    console.print(f"items: {len(items)}  levels={swept}")
    console.print(f"calls: {len(specs)}")
    console.print(
        f"est input {est.input_tokens:,}, output ceiling {est.output_tokens:,}"
    )
    console.print(f"cost ceiling ${projection:.2f} (actual output will be lower)")

    plan = SamplingPlan(
        dataset_id=path.stem,
        dataset_sha256=file_sha256(path),
        manifest_sha256=LADDER_MANIFEST.full_hash,
        seed=seed,
        model=model,
        model_settings={"reasoning_effort": "low", "temperature": None},
        levels=swept,
        units=tuple(units),
        call_identities=tuple(s.identity for s in specs),
    )
    Path(plan_out).parent.mkdir(parents=True, exist_ok=True)
    write_plan(Path(plan_out), plan)
    console.print(f"sampling plan written to {plan_out} ({plan.active_count} active units)")

    if dry_run:
        console.print("dry run: nothing sent")
        return
    raise click.ClickException(
        "live execution consumes the frozen sampling plan and is not implemented "
        "here; see the live-execution prerequisites in the plan"
    )
```

`estimate` and `project_cost` are already imported in `cli.py` from `.runner`
(`src/packing/cli.py:34-41`), and `read_jsonl` from `.dataset`. `estimate` takes
only `specs`. The `sweep` command at `src/packing/cli.py:807-812` is the pattern.


- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ladder_cli.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Recompute the budget against the real manifest**

First fetch a frame large enough to fill both lanes. The 200-prompt file on disk
is the pilot frame and cannot supply 464 core plus 184 `numbers` units, so running
the budget command against it would cost a partial plan rather than the planned
experiment. That is now a hard error rather than a warning.

```bash
uv run packing fetch --limit 1600
uv run packing ladder --dry-run --core-n 464 --numbers-n 184 --core-candidates 508
```

Record the reported call count and cost ceiling. Update spec section 14, replacing
the provisional `$23.40` and `$34.38` extrapolations with the manifest-based
figures, and note the per-criterion token growth the real 40-criterion prompts
produce. Do not pass `--allow-short` for this step.

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/cli.py tests/test_ladder_cli.py docs/superpowers/specs/2026-09-08-load-ladder-design.md
git commit -m "feat: add the ladder command and recompute cost from the manifest"
```

---

### Task 10: One paired outcome per prompt and rung

**Files:**
- Modify: `src/packing/runner.py:176-200` (`_build_record`), `src/packing/runner.py:146-164` (resume), `src/packing/runner.py:260-278` (failure branches)
- Modify: `src/packing/analyze.py:69-131` (`build_outcomes`)
- Test: `tests/test_failure_records.py`

**Interfaces:**
- Consumes: `classify_failure`, `unevaluable_value`, `TRANSIENT`, `UNEVALUABLE`, `FATAL`, `FatalProtocolError` from Task 7
- Produces: `_build_failure_record(spec: CallSpec, message: str) -> dict[str, Any]` in `runner.py`; `completed_identities(path: Path) -> set[str]` counting an identity complete only on success or a class-2 record; `build_outcomes(records, items, criteria, *, unevaluable_rule: str = "primary", reference_size: int = 1)` in `analyze.py`

Three things have to be true together, and each one alone is insufficient:

1. A class-2 failure must become a **row**, or the scoring rule from Task 7 has
   nothing to reach and the primary handling silently degrades to complete-case.
2. Scoring must happen at the **paired unit**. Calls are side-specific but
   recovery is a comparison, so a call-level rule emits two observations for a
   prompt whose two sides both fail.
3. A class-1 failure must **not** count as a completed identity, or resume treats
   a transient network error as a finished call and never retries it.

- [ ] **Step 1: Write the failing test**

```python
"""Failure rows, paired-unit scoring, and resume semantics."""

from __future__ import annotations

import json

import pytest

from packing.analyze import build_outcomes, to_frame
from packing.criteria import HELPSTEER2_CRITERIA
from packing.dataset import PairedItem
from packing.runner import _build_failure_record, completed_identities
from packing.unevaluable import FATAL, TRANSIENT, UNEVALUABLE, FatalProtocolError

ONE = HELPSTEER2_CRITERIA[:1]


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
        targeted_criterion="helpfulness",
    )


class _Spec:
    def __init__(self, item, side, group, nominal_n=1):
        self.item = item
        self.side = side
        self.group = group
        self.group_index = 0
        self.nominal_n = nominal_n
        self.replicate = 0
        self.model = "mock"
        self.identity = f"{item.item_id}|{side}|mock|0|deadbeef|0"

        class _P:
            hash = "deadbeef"

        self.partition = _P()


def _good(side, score):
    spec = _Spec(_item(), side, ONE)
    return {
        "identity": spec.identity,
        "item_id": spec.item.item_id,
        "side": side,
        "model": "mock",
        "nominal_n": 1,
        "replicate": 0,
        "partition_hash": "deadbeef",
        "group_index": 0,
        "group_size": 1,
        "group": ["helpfulness"],
        "positions": {"helpfulness": 0},
        "scores": {"helpfulness": score},
        "reasoning": {"helpfulness": ""},
        "input_tokens": 10,
        "output_tokens": 10,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "near_cap": False,
    }


def test_truncation_is_class_two():
    record = _build_failure_record(
        _Spec(_item(), "a", ONE), "truncated at max_completion_tokens=9352"
    )
    assert record["failure_class"] == UNEVALUABLE
    assert record["scores"] == {}


def test_connection_error_is_class_one():
    record = _build_failure_record(
        _Spec(_item(), "a", ONE), "Connection reset by peer"
    )
    assert record["failure_class"] == TRANSIENT


def test_a_fatal_failure_raises_rather_than_recording():
    with pytest.raises(FatalProtocolError, match="unsupported parameter"):
        _build_failure_record(
            _Spec(_item(), "a", ONE), "invalid_request_error: unsupported parameter"
        )


def test_one_unevaluable_side_yields_exactly_one_outcome():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    outcomes = build_outcomes([failure, _good("b", 3)], [_item()], ONE)
    assert len(outcomes) == 1
    assert to_frame(outcomes)["correct_num"].iloc[0] == 0.0


def test_both_sides_unevaluable_still_yields_exactly_one_outcome():
    a = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    b = _build_failure_record(_Spec(_item(), "b", ONE), "truncated")
    outcomes = build_outcomes([a, b], [_item()], ONE)
    assert len(outcomes) == 1, "a call-level rule would emit two here"


def test_both_sides_valid_scores_normally():
    outcomes = build_outcomes([_good("a", 1), _good("b", 4)], [_item()], ONE)
    assert len(outcomes) == 1
    assert to_frame(outcomes)["correct_num"].iloc[0] in {0.0, 1.0}


def test_complete_case_drops_the_unevaluable_pair():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    outcomes = build_outcomes(
        [failure, _good("b", 3)], [_item()], ONE, unevaluable_rule="complete_case"
    )
    assert outcomes == []


def test_bounds_impute_oppositely_by_rung():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "did not parse")
    lower = build_outcomes(
        [failure, _good("b", 3)], [_item()], ONE,
        unevaluable_rule="lower_bound", reference_size=1,
    )
    upper = build_outcomes(
        [failure, _good("b", 3)], [_item()], ONE,
        unevaluable_rule="upper_bound", reference_size=1,
    )
    # This pair sits at the reference rung, so the lower bound on d puts it high.
    assert to_frame(lower)["correct_num"].iloc[0] == 1.0
    assert to_frame(upper)["correct_num"].iloc[0] == 0.0


def test_class_one_failures_are_never_scored():
    failure = _build_failure_record(_Spec(_item(), "a", ONE), "503 Service Unavailable")
    outcomes = build_outcomes([failure, _good("b", 3)], [_item()], ONE)
    assert outcomes == [], "transient failures are replaced, not scored"


def test_resume_does_not_count_a_transient_failure_as_complete(tmp_path):
    path = tmp_path / "calls.jsonl"
    transient = _build_failure_record(_Spec(_item(), "a", ONE), "connection aborted")
    unevaluable = _build_failure_record(_Spec(_item(), "b", ONE), "truncated")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(transient) + "\n")
        handle.write(json.dumps(unevaluable) + "\n")
        handle.write(json.dumps(_good("a", 2) | {"identity": "third"}) + "\n")
    done = completed_identities(path)
    assert transient["identity"] not in done, "a transient error must be retried"
    assert unevaluable["identity"] in done, "a class-2 result is terminal"
    assert "third" in done
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_failure_records.py -q`
Expected: FAIL with `ImportError: cannot import name '_build_failure_record'`

- [ ] **Step 3: Add the failure record builder and resume filter to `runner.py`**

```python
def _build_failure_record(spec: CallSpec, message: str) -> dict[str, Any]:
    """A failed call as a row, or an abort if the harness is at fault.

    Class-2 failures may become more common at high K, which makes them part of
    the load effect. Recorded only in a log they would be a complete-case
    exclusion by default, which the protocol forbids as primary handling.
    """
    from .unevaluable import FATAL, FatalProtocolError, classify_failure

    failure_class = classify_failure(message)
    if failure_class == FATAL:
        raise FatalProtocolError(
            f"{spec.identity}: {message}. This is a protocol or configuration "
            "error, not a judge behaviour, so the run stops rather than scoring it."
        )
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
        "failure_class": failure_class,
        "scores": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
    }


def completed_identities(path: Path) -> set[str]:
    """Identities that need no further attempt.

    A transient failure is not complete: resume must retry it. Only a success or
    a terminal class-2 record closes an identity.
    """
    from .unevaluable import TRANSIENT

    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a hard kill. Redo that call.
                continue
            if record.get("failure_class") == TRANSIENT:
                continue
            identity = record.get("identity")
            if identity:
                done.add(identity)
    return done
```

Replace the body of the existing resume reader at `src/packing/runner.py:146-164` with a call to `completed_identities`, and in both failure branches write the record under `write_lock` exactly as successes are written. Let `FatalProtocolError` propagate: do not catch it in the bare `except Exception` branch.

- [ ] **Step 4: Score at the paired unit in `analyze.py`**

`build_outcomes` already looks up `rec_a` and `rec_b` for each (item, rung). Extend
that pairing rather than appending separately:

```python
    from .unevaluable import TRANSIENT, unevaluable_value

    def _unevaluable(record: dict[str, Any] | None) -> bool:
        return bool(record) and bool(record.get("failure"))

    # Transient failures are replaced upstream and carry no information.
    if (rec_a and rec_a.get("failure_class") == TRANSIENT) or (
        rec_b and rec_b.get("failure_class") == TRANSIENT
    ):
        continue

    if _unevaluable(rec_a) or _unevaluable(rec_b):
        value = unevaluable_value(
            unevaluable_rule, rec_a["group_size"], reference_size
        )
        if value is None:
            continue
        outcomes.append(
            Outcome(
                item_id=item.item_id,
                criterion=item.targeted_criterion,
                group_size=rec_a["group_size"],
                position=0,
                judge_a=None,
                judge_b=None,
                human_a=None,
                human_b=None,
                correct_num=value,
                judge_tie=False,
                unevaluable=True,
            )
        )
        continue
```

Add `unevaluable: bool = False` to `Outcome` in `src/packing/score.py` and let
`correct_num` be set directly rather than derived from judge scores when it is set.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_failure_records.py -q`
Expected: PASS, 10 tests

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/runner.py src/packing/analyze.py src/packing/score.py tests/test_failure_records.py
git commit -m "analysis: score unevaluable pairs once, and keep transients retryable"
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

    This is still a planning approximation, not the analysis distribution. The
    analysis uses Welch-Satterthwaite degrees of freedom, which depend on the
    individual stratum variances; this function has only their root-mean-square
    summary and uses df = n - 1. Label it as planning, never as the test.
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
def _order_key(spec):
    """Stable identity for ordering assertions.

    Deliberately not `spec.identity`: identity embeds the partition hash, which
    changes when the companion permutation changes, so an identity comparison
    would report a difference the ordering algorithm did not cause.
    """
    return (spec.item.item_id, spec.side, spec.nominal_n)


def _prompt_major(items, levels):
    """The order Task 5 produces before interleaving, for comparison."""
    return [
        (item.item_id, side, level)
        for item in items
        for level in levels
        for side in ("a", "b")
    ]


def test_order_is_not_prompt_major():
    items = [_item(f"i{i}", "precision") for i in range(12)]
    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=7
    )
    assert [_order_key(s) for s in specs] != _prompt_major(items, LEVELS)


def test_one_prompts_calls_are_not_contiguous():
    items = [_item(f"i{i}", "precision") for i in range(12)]
    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=7
    )
    positions = [
        index for index, s in enumerate(specs) if s.item.item_id == "i0"
    ]
    span = max(positions) - min(positions) + 1
    assert span > len(positions), (
        "a prompt's ten calls should be spread through the run, not adjacent"
    )


def test_ordering_is_deterministic_for_a_seed():
    items = [_item(f"i{i}", "precision") for i in range(8)]
    a = [
        _order_key(s)
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=42
        )
    ]
    b = [
        _order_key(s)
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=42
        )
    ]
    assert a == b


def test_ordering_changes_with_the_seed():
    items = [_item(f"i{i}", "precision") for i in range(8)]
    a = [
        _order_key(s)
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=1
        )
    ]
    b = [
        _order_key(s)
        for s in plan_fixed_position(
            items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=2
        )
    ]
    assert a != b


def test_ordering_preserves_the_call_set_exactly():
    items = [_item(f"i{i}", "precision") for i in range(6)]
    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model="mock", levels=LEVELS, base_seed=3
    )
    assert sorted(_order_key(s) for s in specs) == sorted(
        _prompt_major(items, LEVELS)
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ladder_plan.py -q`
Expected: FAIL on `test_order_is_not_prompt_major`, `test_one_prompts_calls_are_not_contiguous` and `test_ordering_changes_with_the_seed`. The first two fail because Task 5 emits prompt-major; the third fails because the order does not yet depend on the seed at all. `test_ordering_is_deterministic_for_a_seed` and `test_ordering_preserves_the_call_set_exactly` pass before the change, which is correct: they are regression guards, not the red test.

An earlier version of this task counted runs of identical `nominal_n` and asserted the count exceeded `len(items) * 2`. That assertion is satisfied by the unmodified prompt-major planner, which already produces about five blocks per prompt, so it would never have gone red. Comparing against the explicit prompt-major construction is the check that actually discriminates.

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

### Task 13: Frozen sampling-plan artifact

**Files:**
- Create: `src/packing/sampling_plan.py`
- Test: `tests/test_sampling_plan.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `Unit` dataclass (`base_item_id`, `lane`, `side`, `eligible`, `operation`, `realisation_seed`, `companion_seed`, `status`); `SamplingPlan` dataclass (`dataset_id`, `dataset_sha256`, `manifest_sha256`, `seed`, `model`, `model_settings`, `levels`, `units`, `call_identities`); `write_plan(path: Path, plan: SamplingPlan) -> None`; `read_plan(path: Path) -> SamplingPlan`; `file_sha256(path: Path) -> str`

Printing the manifest hash to a console is not freezing anything. Without a
durable artifact, the live run would repeat screening and assignment from
command-line arguments, and every guarantee the design rests on becomes a
coincidence: a reordered input file selects a different sample, an extension
reassigns collected units, and edited anchor prose merges into a resumed run.

- [ ] **Step 1: Write the failing test**

```python
"""The sampling plan is the frozen record the live run consumes."""

from __future__ import annotations

import pytest

from packing.sampling_plan import (
    SamplingPlan,
    Unit,
    file_sha256,
    read_plan,
    write_plan,
)


def _plan(units=None):
    return SamplingPlan(
        dataset_id="helpsteer2",
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
        seed=20260908,
        model="gpt-5-mini",
        model_settings={"reasoning_effort": "low", "temperature": None},
        levels=(1, 5, 10, 20, 40),
        units=units
        or (
            Unit(
                base_item_id="hs2-000001",
                lane="core",
                side="b",
                eligible=True,
                operation="shuffle",
                realisation_seed=123,
                companion_seed=456,
                status="active",
            ),
        ),
        call_identities=("hs2-000001|b|gpt-5-mini|0|abc|0",),
    )


def test_round_trips_through_disk(tmp_path):
    path = tmp_path / "plan.json"
    write_plan(path, _plan())
    assert read_plan(path) == _plan()


def test_hashes_are_full_sha256(tmp_path):
    path = tmp_path / "plan.json"
    write_plan(path, _plan())
    loaded = read_plan(path)
    assert len(loaded.dataset_sha256) == 64
    assert len(loaded.manifest_sha256) == 64


def test_file_sha256_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("x\n", encoding="utf-8")
    b.write_text("y\n", encoding="utf-8")
    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)
    assert len(file_sha256(a)) == 64


def test_active_and_reserve_units_are_distinguished():
    plan = _plan(
        units=(
            Unit("p1", "core", "a", True, "shuffle", 1, 2, "active"),
            Unit("p2", "core", "b", True, "filler", 3, 4, "reserve"),
        )
    )
    assert [u.status for u in plan.units] == ["active", "reserve"]
    assert plan.active_count == 1


def test_duplicate_base_items_are_rejected():
    with pytest.raises(ValueError, match="duplicate base item"):
        _plan(
            units=(
                Unit("p1", "core", "a", True, "shuffle", 1, 2, "active"),
                Unit("p1", "numbers", "b", True, "numbers", 3, 4, "active"),
            )
        )


def test_a_unit_may_belong_to_only_one_arm():
    # Enforced by the duplicate check above, stated separately because it is the
    # design constraint rather than an implementation detail.
    with pytest.raises(ValueError, match="duplicate base item"):
        _plan(
            units=(
                Unit("p9", "core", "a", True, "flatten", 1, 2, "active"),
                Unit("p9", "numbers", "a", True, "numbers", 1, 2, "active"),
            )
        )


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="status must be"):
        _plan(units=(Unit("p1", "core", "a", True, "shuffle", 1, 2, "maybe"),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sampling_plan.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'packing.sampling_plan'`

- [ ] **Step 3: Write minimal implementation**

```python
"""The frozen sampling plan.

The dry run writes this and the live run consumes it. That ordering is what makes
the design's guarantees real rather than incidental: the response side is visibly
frozen before an operation is assigned, an extension cannot reassign a collected
unit, a reordered input file cannot select a different sample, and edited anchor
prose cannot merge into a resumed run because the manifest digest is recorded here.

Hashes are full SHA-256. A 16-character prefix is fine for a console line and not
fine for an artifact that decides whether two runs are the same experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_STATUSES = ("active", "reserve")


@dataclass(frozen=True)
class Unit:
    base_item_id: str
    lane: str
    side: str
    eligible: bool
    operation: str
    realisation_seed: int
    companion_seed: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}, got {self.status!r}")
        if self.side not in ("a", "b"):
            raise ValueError(f"side must be 'a' or 'b', got {self.side!r}")


@dataclass(frozen=True)
class SamplingPlan:
    dataset_id: str
    dataset_sha256: str
    manifest_sha256: str
    seed: int
    model: str
    model_settings: dict[str, object]
    levels: tuple[int, ...]
    units: tuple[Unit, ...]
    call_identities: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        ids = [u.base_item_id for u in self.units]
        if len(set(ids)) != len(ids):
            seen: set[str] = set()
            dupes = sorted({i for i in ids if i in seen or seen.add(i)})
            raise ValueError(
                f"duplicate base item in the plan: {dupes[:5]}. A base prompt "
                "contributes to at most one arm."
            )
        for name, digest in (
            ("dataset_sha256", self.dataset_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            if len(digest) != 64:
                raise ValueError(f"{name} must be a full SHA-256, got {len(digest)}")

    @property
    def active_count(self) -> int:
        return sum(1 for u in self.units if u.status == "active")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_plan(path: Path, plan: SamplingPlan) -> None:
    payload = asdict(plan)
    payload["levels"] = list(plan.levels)
    payload["call_identities"] = list(plan.call_identities)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_plan(path: Path) -> SamplingPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SamplingPlan(
        dataset_id=payload["dataset_id"],
        dataset_sha256=payload["dataset_sha256"],
        manifest_sha256=payload["manifest_sha256"],
        seed=payload["seed"],
        model=payload["model"],
        model_settings=payload["model_settings"],
        levels=tuple(payload["levels"]),
        units=tuple(Unit(**u) for u in payload["units"]),
        call_identities=tuple(payload["call_identities"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sampling_plan.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run pytest -q
git add src/packing/sampling_plan.py tests/test_sampling_plan.py
git commit -m "feat: freeze the sampling plan as a durable artifact"
```

---

## Prerequisites for live execution

None of these is built here, and all are required before a paid run. Listing them
so the boundary is explicit rather than discovered later.

1. **Prompt replacement after persistent transient failure.** Task 10 keeps a
   class-1 identity retryable, but nothing yet draws the replacement from the next
   candidate in the same lane in frozen order.
2. **A live command that consumes the sampling plan.** Task 9 writes the artifact;
   the live path must read it rather than re-screen, or every guarantee the
   artifact exists to provide is bypassed.
3. **Archival outside the ignore rule.** `runs/` is gitignored, and the K = 20
   sweep was nearly lost to that (METHODS section 10.8). The run directory is
   copied out before analysis begins.
4. **The blinded variance review, and its null simulation.** Spec section 8 fixes
   the rule; Task 11 supplies the variance function it calls. The review runs after
   160 complete core prompts, so it needs real data, and its type-I error has to be
   simulated under the null from the pilot distributions before it is used.
5. **Fatal-abort drill.** `FatalProtocolError` from Task 7 must be shown to stop a
   real run rather than being swallowed by the concurrent executor's bare
   `except Exception` branch.

## Not in this plan

- **Live execution.** Task 9 refuses it by design. Spending is a separate decision after the manifest-based recompute.
- **The blinded variance review itself.** Listed above as a live-execution prerequisite.
- **`rename` and `scramble_list`.** Spec section 11 assigns them retained pilot data only, no prompts, no quota, no calls.
- **Deck corrections.** `presentation/index.html` slide 14 still frames load against position as unadjudicated, and its two MDE cards are wrong. Tracked separately from the experiment machinery.
