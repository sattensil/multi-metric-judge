"""Corruption invariants.

METHODS.md 3.4 asserts three invariants; these are them. The third
(honest failure) is the one that matters most: emitting an unmodified variant
while asserting a ground-truth difference injects false labels, which is worse
than any amount of missing data.
"""

from __future__ import annotations

import re

import pytest

from packing.corrupt import (
    BY_CRITERION,
    CORRUPTIONS,
    get_corruption,
    split_sentences,
)
from packing.criteria import HELPSTEER2_CRITERIA

PROSE = (
    "Bacillus subtilis is a Gram-positive bacterium found in soil. "
    "It is a facultative anaerobe that grows with or without oxygen. "
    "The species is a model organism for bacterial genetics. "
    "It has 4215 genes and a genome of 4.2 megabases. "
    "Industrial uses include enzyme production for detergents."
)

WITH_CODE = (
    "You can use the loop below to build the list.\n"
    "```python\n"
    "for i in range(10):\n"
    "    total = total + i * 2\n"
    "```\n"
    "This runs in linear time. Call `total_sum(xs)` to check the result. "
    "The helper needs 3 arguments."
)


# ----------------------------------------------------------- registry --------


def test_every_criterion_has_exactly_one_corruption() -> None:
    keys = {c.targeted_criterion for c in CORRUPTIONS}
    assert keys == {c.key for c in HELPSTEER2_CRITERIA}
    assert len(BY_CRITERION) == len(CORRUPTIONS), "two corruptions share a criterion"


def test_directions_are_signed_units() -> None:
    for corruption in CORRUPTIONS:
        assert corruption.direction in (-1, 1), corruption.key


def test_lookup_by_key_or_criterion() -> None:
    assert get_corruption("shuffle").targeted_criterion == "coherence"
    assert get_corruption("coherence").key == "shuffle"
    with pytest.raises(ValueError, match="unknown corruption"):
        get_corruption("nope")


# ------------------------------------------- invariant 1: determinism --------


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.key)
def test_deterministic_under_seed(corruption) -> None:
    """Resume keys on call identity; a re-run must not produce different items."""
    first = corruption.apply(PROSE, seed=42, prompt="bacillus subtilis genome")
    second = corruption.apply(PROSE, seed=42, prompt="bacillus subtilis genome")
    assert first.text == second.text
    assert first.changed == second.changed
    assert first.n_edits == second.n_edits


# A long text with many substitutable words and many sentences, so stochastic
# operations have more candidates than their severity and can actually vary.
LONG_PROSE = " ".join(
    f"You can use approach {i} to make the result, and it also helps because "
    f"the small parts change fast in many ways."
    for i in range(12)
)

STOCHASTIC = [c for c in CORRUPTIONS if c.stochastic]
DETERMINISTIC = [c for c in CORRUPTIONS if not c.stochastic]


@pytest.mark.parametrize("corruption", STOCHASTIC, ids=lambda c: c.key)
def test_stochastic_ops_vary_with_seed(corruption) -> None:
    """Otherwise replicates would be duplicates and buy no variance reduction.

    Requires more candidates than the severity: an operation asked for 8 edits in
    text offering only one is degenerate, not broken.
    """
    outputs = {
        corruption.apply(LONG_PROSE, seed=s, prompt="approach result").text
        for s in range(12)
    }
    assert len(outputs) > 1, f"{corruption.key} ignores its seed"


@pytest.mark.parametrize("corruption", DETERMINISTIC, ids=lambda c: c.key)
def test_deterministic_ops_ignore_seed_by_contract(corruption) -> None:
    """Declared, not accidental. `drop_answer` picks the argmax of prompt overlap,
    so there is nothing to randomise -- and replication therefore samples only the
    judge's nondeterminism for this operation."""
    outputs = {
        corruption.apply(LONG_PROSE, seed=s, prompt="approach result").text
        for s in range(6)
    }
    assert len(outputs) == 1


def test_stochastic_flag_matches_reality() -> None:
    """Guards against the flag drifting away from the implementation."""
    for corruption in CORRUPTIONS:
        outputs = {
            corruption.apply(LONG_PROSE, seed=s, prompt="approach result").text
            for s in range(12)
        }
        varies = len(outputs) > 1
        assert varies == corruption.stochastic, (
            f"{corruption.key}: stochastic={corruption.stochastic} but "
            f"seed-variation={varies}"
        )


# --------------------------------------- invariant 2: honest failure ---------


def test_shuffle_refuses_short_text() -> None:
    result = get_corruption("shuffle").apply("One sentence only.", seed=1)
    assert result.changed is False
    assert result.text == "One sentence only."
    assert "skipped" in result.provenance


def test_numbers_refuses_text_without_numbers() -> None:
    text = "There are no numerals here at all. None whatsoever. Truly none."
    result = get_corruption("numbers").apply(text, seed=1)
    assert result.changed is False
    assert result.text == text


def test_unchanged_results_never_claim_edits() -> None:
    """A changed=False result asserting edits would be a false label."""
    for corruption in CORRUPTIONS:
        result = corruption.apply("Tiny.", seed=1, prompt="x")
        if not result.changed:
            assert result.n_edits == 0
            assert result.text == "Tiny."


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.key)
def test_changed_implies_text_differs(corruption) -> None:
    """The core labelling guarantee: changed=True means the text really moved."""
    result = corruption.apply(PROSE, seed=5, prompt="bacillus subtilis genome")
    if result.changed:
        assert result.text != PROSE
        assert result.n_edits >= 1
    else:
        assert result.text == PROSE


# ------------------------------------ invariant 3: code preservation ---------


def _code_spans(text: str) -> list[str]:
    fenced = re.findall(r"```.*?```", text, flags=re.S)
    inline = re.findall(r"`[^`\n]+`", text)
    return fenced + inline


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.key)
def test_code_spans_survive_intact(corruption) -> None:
    """Editing code would perturb several criteria at once and destroy targeting."""
    before = _code_spans(WITH_CODE)
    assert before, "fixture must contain code"
    result = corruption.apply(WITH_CODE, seed=9, prompt="build a list in python")
    for span in before:
        assert span in result.text, f"{corruption.key} damaged a code span"


def test_numbers_does_not_touch_numbers_inside_code() -> None:
    """`range(10)` and `i * 2` must be untouched even though the op targets numerals."""
    result = get_corruption("numbers").apply(WITH_CODE, seed=4)
    assert "for i in range(10):" in result.text
    assert "total = total + i * 2" in result.text


# ------------------------------------------------ targeting behaviour --------


def test_shuffle_preserves_content_exactly() -> None:
    """Only order changes: nothing added, removed, or reworded. That is what keeps
    the operation targeted on coherence."""
    result = get_corruption("shuffle").apply(PROSE, seed=3)
    assert result.changed
    assert sorted(re.findall(r"\w+", PROSE)) == sorted(re.findall(r"\w+", result.text))


def test_duplicate_only_lengthens() -> None:
    result = get_corruption("duplicate").apply(PROSE, seed=3)
    assert result.changed
    assert len(result.text) > len(PROSE)


def test_numbers_changes_a_numeral_and_records_it() -> None:
    result = get_corruption("numbers").apply(PROSE, seed=3)
    assert result.changed
    edits = result.provenance["edits"]
    assert edits and all(e["from"] != e["to"] for e in edits)
    for edit in edits:
        assert edit["to"] in result.text


def test_jargon_substitutes_and_records() -> None:
    result = get_corruption("jargon").apply(PROSE, seed=3)
    assert result.changed
    assert result.provenance["edits"]


def test_drop_answer_removes_a_sentence() -> None:
    result = get_corruption("drop_answer").apply(
        PROSE, seed=3, prompt="how many genes does bacillus subtilis have"
    )
    assert result.changed
    assert len(split_sentences(result.text)) < len(split_sentences(PROSE))


def test_drop_answer_prefers_prompt_relevant_sentence() -> None:
    """With a clear prompt signal it should remove the on-topic sentence, not just
    the first one."""
    result = get_corruption("drop_answer").apply(
        PROSE, seed=3, prompt="genes genome megabases"
    )
    assert result.provenance["used_prompt"] is True
    assert "4215 genes" not in result.text


def test_drop_answer_falls_back_without_prompt() -> None:
    result = get_corruption("drop_answer").apply(PROSE, seed=3, prompt="")
    assert result.changed
    assert result.provenance["used_prompt"] is False


def test_leakage_prone_flag_is_set_where_documented() -> None:
    """METHODS.md 3.4 flags drop_answer as leakage-prone by construction."""
    assert get_corruption("drop_answer").leakage_prone is True
    assert get_corruption("shuffle").leakage_prone is False


# ----------------------------------------------------------- splitter --------


def test_sentence_splitter_basic() -> None:
    parts = [p for p in split_sentences("One. Two! Three?") if p.strip()]
    assert len(parts) == 3


def test_sentence_splitter_keeps_decimals_together() -> None:
    """A false boundary would split mid-sentence and corrupt untargeted text."""
    parts = [p for p in split_sentences("It spans 4.2 megabases in total.") if p.strip()]
    assert len(parts) == 1
