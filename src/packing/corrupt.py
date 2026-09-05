"""Deterministic corruptions that create ground truth by construction.

Each corruption targets ONE criterion and declares the direction the targeted
score must move. That relation -- not an absolute score -- is the ground truth.
A padded response is "more verbose than its base"; it is not "verbosity = 4".

Why this exists: HelpSteer2's published inter-annotator agreement (quadratic
weighted Cohen's kappa) is 0.428 for coherence and 0.427 for complexity. Humans
barely agree with each other there, so "the judge is bad at coherence" and
"humans cannot agree on coherence" are indistinguishable against that reference.
Corruption separates them.

Three invariants every corruption must satisfy, all enforced by tests:

1. **Deterministic under seed.** Same (text, seed, severity) gives the same output,
   because the resume file keys on identity and a re-run must not silently produce
   different items.
2. **Honest failure.** If the operation cannot apply (shuffling a one-sentence
   response, swapping a number in text with no numbers), it returns
   ``changed=False`` and the item is DROPPED. Emitting an unchanged "corrupted"
   variant while asserting a ground-truth difference would inject false labels --
   the single worst failure available here.
3. **Code is not touched.** Fenced code blocks are masked before editing and
   restored after. Mangling code would corrupt several criteria at once and
   destroy the targeting the whole design rests on.

Leakage is measured, not assumed -- see ``leakage.py``. Shuffling sentences for
coherence plausibly also harms helpfulness, and a corruption whose off-target
effect rivals its on-target effect is disqualified from the primary analysis.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------- results ----


@dataclass(frozen=True)
class CorruptionResult:
    text: str
    changed: bool
    n_edits: int = 0
    provenance: dict = field(default_factory=dict)

    @classmethod
    def unchanged(cls, text: str, reason: str) -> CorruptionResult:
        return cls(text=text, changed=False, n_edits=0, provenance={"skipped": reason})


# ------------------------------------------------------- text scaffolding ----

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_MASK = "\x00CODE{}\x00"

# Sentence boundary: terminal punctuation, then whitespace, then something that
# starts a new sentence. Deliberately conservative -- a missed boundary is safe,
# a false one splits mid-sentence and corrupts text we did not intend to touch.
_SENT_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z0-9])")


def _mask_code(text: str) -> tuple[str, list[str]]:
    """Replace code spans with opaque placeholders so edits cannot reach them."""
    stash: list[str] = []

    def take(match: re.Match) -> str:
        stash.append(match.group(0))
        return _MASK.format(len(stash) - 1)

    masked = _FENCE.sub(take, text)
    masked = _INLINE_CODE.sub(take, masked)
    return masked, stash


def _unmask_code(text: str, stash: list[str]) -> str:
    for index, original in enumerate(stash):
        text = text.replace(_MASK.format(index), original)
    return text


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, treating each line of a list as its own unit."""
    parts: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            parts.append(line)
            continue
        parts.extend(_SENT_SPLIT.split(line))
    return parts


def _prose_indices(parts: list[str]) -> list[int]:
    """Indices of parts that are real prose: non-blank, not a masked code span."""
    return [
        i
        for i, part in enumerate(parts)
        if part.strip() and "\x00CODE" not in part
    ]


# ------------------------------------------------------------ operations -----


def shuffle_sentences(text: str, seed: int, severity: int) -> CorruptionResult:
    """Reorder sentences. Targets COHERENCE (down).

    Content is preserved exactly -- only order changes -- which is what keeps this
    targeted: nothing is added, removed, or reworded.
    """
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if len(idx) < 3:
        return CorruptionResult.unchanged(text, "fewer than 3 prose sentences")

    rng = random.Random(seed)
    order = idx[:]
    for _ in range(20):
        rng.shuffle(order)
        if order != idx:
            break
    else:
        return CorruptionResult.unchanged(text, "shuffle produced original order")

    rearranged = parts[:]
    for target, source in zip(idx, order):
        rearranged[target] = parts[source]

    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    if out.strip() == text.strip():
        return CorruptionResult.unchanged(text, "no net change")
    return CorruptionResult(
        text=out,
        changed=True,
        n_edits=sum(1 for a, b in zip(idx, order) if a != b),
        provenance={"op": "shuffle_sentences", "n_sentences": len(idx)},
    )


def _rejoin(parts: list[str], original_masked: str) -> list[str]:
    """Reassemble lines, preserving the original line structure."""
    lines_out: list[str] = []
    cursor = 0
    for line in original_masked.split("\n"):
        if not line.strip():
            lines_out.append(parts[cursor])
            cursor += 1
            continue
        count = len(_SENT_SPLIT.split(line))
        lines_out.append(" ".join(parts[cursor : cursor + count]))
        cursor += count
    return lines_out


def duplicate_sentences(text: str, seed: int, severity: int) -> CorruptionResult:
    """Repeat sentences verbatim. Targets VERBOSITY (up).

    Chosen over generated filler because repetition adds no new claims, so
    correctness and coherence are left as intact as this operation allows.
    """
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if not idx:
        return CorruptionResult.unchanged(text, "no prose sentences")

    rng = random.Random(seed)
    picks = [rng.choice(idx) for _ in range(max(1, severity))]
    rearranged = parts[:]
    for pick in picks:
        rearranged[pick] = rearranged[pick].rstrip() + " " + parts[pick].strip()

    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    return CorruptionResult(
        text=out,
        changed=True,
        n_edits=len(picks),
        provenance={"op": "duplicate_sentences", "duplicated": len(picks)},
    )


_NUMBER = re.compile(r"(?<![\w.])(\d{1,4}(?:\.\d+)?)(?![\w.])")


def perturb_numbers(text: str, seed: int, severity: int) -> CorruptionResult:
    """Replace numbers with clearly wrong ones. Targets CORRECTNESS (down).

    Perturbation is multiplicative and large enough to be unambiguously wrong
    rather than a rounding difference, but keeps the same magnitude class so the
    text stays plausible prose.
    """
    masked, stash = _mask_code(text)
    matches = list(_NUMBER.finditer(masked))
    if not matches:
        return CorruptionResult.unchanged(text, "no numbers outside code")

    rng = random.Random(seed)
    chosen = sorted(
        rng.sample(matches, min(max(1, severity), len(matches))),
        key=lambda m: m.start(),
        reverse=True,
    )
    out = masked
    edits = []
    for match in chosen:
        original = match.group(1)
        try:
            value = float(original)
        except ValueError:  # pragma: no cover - regex guarantees numeric
            continue
        factor = rng.choice([2, 3, 5])
        replacement = (
            str(int(value * factor)) if "." not in original else f"{value * factor:.2f}"
        )
        if replacement == original:
            continue
        out = out[: match.start(1)] + replacement + out[match.end(1) :]
        edits.append({"from": original, "to": replacement})

    if not edits:
        return CorruptionResult.unchanged(text, "no number could be perturbed")
    return CorruptionResult(
        text=_unmask_code(out, stash),
        changed=True,
        n_edits=len(edits),
        provenance={"op": "perturb_numbers", "edits": edits},
    )


# Plain -> jargon. Raises the expertise a reader needs, which is exactly what
# HelpSteer2's `complexity` measures (descriptive, not evaluative: higher is not
# better). Meaning is preserved so correctness should not move.
_JARGON: dict[str, str] = {
    "use": "utilise",
    "uses": "utilises",
    "used": "utilised",
    "using": "utilising",
    "make": "instantiate",
    "makes": "instantiates",
    "start": "initialise",
    "starts": "initialises",
    "show": "elucidate",
    "shows": "elucidates",
    "help": "facilitate",
    "helps": "facilitates",
    "helped": "facilitated",
    "change": "transmute",
    "changes": "transmutes",
    "find": "ascertain",
    "finds": "ascertains",
    "check": "verify",
    "checks": "verifies",
    "build": "construct",
    "builds": "constructs",
    "need": "necessitate",
    "needs": "necessitates",
    "get": "obtain",
    "gets": "obtains",
    "keep": "retain",
    "keeps": "retains",
    "let": "permit",
    "lets": "permits",
    "big": "substantial",
    "small": "negligible",
    "fast": "expeditious",
    "easy": "trivial",
    "hard": "non-trivial",
    "about": "concerning",
    "because": "inasmuch as",
    "before": "antecedent to",
    "after": "subsequent to",
    "also": "additionally",
    "but": "however",
    "so": "consequently",
    "many": "myriad",
    "most": "the preponderance of",
    "way": "methodology",
    "ways": "methodologies",
    "part": "constituent",
    "parts": "constituents",
    "job": "function",
    "jobs": "functions",
}


def jargonise(text: str, seed: int, severity: int) -> CorruptionResult:
    """Substitute plain words for jargon. Targets COMPLEXITY (up)."""
    masked, stash = _mask_code(text)
    rng = random.Random(seed)

    candidates: list[tuple[int, int, str, str]] = []
    for plain, fancy in _JARGON.items():
        for match in re.finditer(rf"\b{re.escape(plain)}\b", masked, flags=re.I):
            candidates.append((match.start(), match.end(), match.group(0), fancy))
    if not candidates:
        return CorruptionResult.unchanged(text, "no substitutable words")

    rng.shuffle(candidates)
    picked = sorted(
        candidates[: max(1, severity)], key=lambda c: c[0], reverse=True
    )
    out = masked
    edits = []
    for start, end, original, fancy in picked:
        replacement = fancy.capitalize() if original[0].isupper() else fancy
        out = out[:start] + replacement + out[end:]
        edits.append({"from": original, "to": replacement})

    return CorruptionResult(
        text=_unmask_code(out, stash),
        changed=True,
        n_edits=len(edits),
        provenance={"op": "jargonise", "edits": edits},
    )


_STOP = frozenset(
    "a an the is are was were be been being of to in on for with and or but if "
    "that this these those it its as at by from what how why when which who do "
    "does did can could should would will you your i me my we our they them".split()
)


def drop_answer_sentence(
    text: str, seed: int, severity: int, prompt: str = ""
) -> CorruptionResult:
    """Delete the sentence most on-topic with the prompt. Targets HELPFULNESS (down).

    LEAKAGE-PRONE by construction: removing a sentence also disturbs coherence and
    arguably correctness. Kept because helpfulness is the criterion practitioners
    care about most, but it must clear the leakage matrix before entering the
    primary analysis.
    """
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if len(idx) < 3:
        return CorruptionResult.unchanged(text, "fewer than 3 prose sentences")

    keywords = {
        w for w in re.findall(r"[a-z0-9]+", prompt.lower()) if w not in _STOP and len(w) > 2
    }
    def overlap(i: int) -> tuple[int, int]:
        words = {w for w in re.findall(r"[a-z0-9]+", parts[i].lower())}
        return (len(words & keywords), -i)

    # Severity is the number of sentences removed, taken in descending order of
    # prompt relevance. Calibration measured an 80% tie rate at severity 1: one
    # deleted sentence out of five was too subtle for the judge to register, so the
    # operation needed a dial rather than exclusion.
    ranked = idx if not keywords else sorted(idx, key=overlap, reverse=True)
    n_remove = max(1, min(severity, len(idx) - 2))
    targets = set(ranked[:n_remove])

    rearranged = [p for i, p in enumerate(parts) if i not in targets]
    out = _unmask_code("\n".join(rearranged), stash)
    return CorruptionResult(
        text=out,
        changed=True,
        n_edits=len(targets),
        provenance={
            "op": "drop_answer_sentence",
            "removed_n": len(targets),
            "removed": parts[ranked[0]].strip()[:120],
            "used_prompt": bool(keywords),
        },
    )


# ------------------------------------------------------------- registry ------


@dataclass(frozen=True)
class Corruption:
    key: str
    targeted_criterion: str
    direction: int
    """+1 if the corrupted text should score HIGHER on the target, -1 if LOWER."""
    default_severity: int
    description: str
    leakage_prone: bool = False
    needs_prompt: bool = False
    stochastic: bool = True
    """False if the operation has no RNG and ignores its seed by design.

    Matters for replication: a deterministic corruption yields identical variant
    text on every replicate, so replicates sample only the judge's own
    nondeterminism, not the corruption's. Variance reduction from replication is
    correspondingly smaller for these.
    """
    _apply: Callable[..., CorruptionResult] = field(repr=False, default=None)

    def apply(
        self, text: str, *, seed: int, severity: int | None = None, prompt: str = ""
    ) -> CorruptionResult:
        sev = self.default_severity if severity is None else severity
        if self.needs_prompt:
            return self._apply(text, seed, sev, prompt)
        return self._apply(text, seed, sev)


CORRUPTIONS: tuple[Corruption, ...] = (
    Corruption(
        key="shuffle",
        targeted_criterion="coherence",
        direction=-1,
        default_severity=1,
        description="Reorder sentences; content preserved exactly.",
        _apply=shuffle_sentences,
    ),
    Corruption(
        key="jargon",
        targeted_criterion="complexity",
        direction=+1,
        default_severity=8,
        description="Substitute plain words for jargon; meaning preserved.",
        _apply=jargonise,
    ),
    Corruption(
        key="numbers",
        targeted_criterion="correctness",
        direction=-1,
        # Ladder best: s4 gave 0.57 detection / 0.40 ties / 0.94 discrimination.
        # Sensitivity is CAPPED by the text -- a response with three numerals cannot
        # absorb eight numeral edits -- so higher severity does not help.
        default_severity=4,
        description="Replace numbers with clearly wrong values.",
        _apply=perturb_numbers,
    ),
    Corruption(
        key="duplicate",
        targeted_criterion="verbosity",
        direction=+1,
        default_severity=3,
        description="Repeat sentences verbatim; adds no new claims.",
        _apply=duplicate_sentences,
    ),
    Corruption(
        key="drop_answer",
        targeted_criterion="helpfulness",
        direction=-1,
        # Ladder: monotone in severity. s16 gave 0.67 / 0.27 / 0.91; s3 gave
        # 0.07 / 0.80 -- the original severity was far too low.
        default_severity=16,
        description="Delete the N sentences most on-topic with the prompt.",
        leakage_prone=True,
        needs_prompt=True,
        # Deterministic by construction: the target is the argmax of lexical
        # overlap with the prompt, so there is nothing to randomise.
        stochastic=False,
        _apply=drop_answer_sentence,
    ),
)

BY_KEY: dict[str, Corruption] = {c.key: c for c in CORRUPTIONS}
BY_CRITERION: dict[str, Corruption] = {c.targeted_criterion: c for c in CORRUPTIONS}


def get_corruption(key_or_criterion: str) -> Corruption:
    if key_or_criterion in BY_KEY:
        return BY_KEY[key_or_criterion]
    if key_or_criterion in BY_CRITERION:
        return BY_CRITERION[key_or_criterion]
    raise ValueError(
        f"unknown corruption {key_or_criterion!r}; "
        f"keys={sorted(BY_KEY)} criteria={sorted(BY_CRITERION)}"
    )
