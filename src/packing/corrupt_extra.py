"""Corruption operations for the fifteen synthetic-only criteria.

Same three invariants as `corrupt.py`: deterministic under seed, honest failure
(``changed=False`` and the item is dropped rather than mislabelled), and code
spans preserved.

Leakage is expected between several of these and is measured rather than assumed.
`punctuation` and `fluency` both touch the surface; `conciseness` and `verbosity`
both concern padding. The leakage matrix decides which survive into the primary
analysis.
"""

from __future__ import annotations

import random
import re

from .corrupt import (
    Corruption,
    CorruptionResult,
    _mask_code,
    _prose_indices,
    _rejoin,
    _unmask_code,
    split_sentences,
)

# ------------------------------------------------------------ surface --------


def inject_typos(text: str, seed: int, severity: int) -> CorruptionResult:
    """Transpose adjacent characters inside words. Targets FLUENCY (down)."""
    masked, stash = _mask_code(text)
    words = [
        m for m in re.finditer(r"\b[a-zA-Z]{5,}\b", masked) if "\x00" not in m.group(0)
    ]
    if not words:
        return CorruptionResult.unchanged(text, "no words long enough to transpose")

    rng = random.Random(seed)
    picked = sorted(
        rng.sample(words, min(max(1, severity), len(words))),
        key=lambda m: m.start(),
        reverse=True,
    )
    out = masked
    edits = []
    for match in picked:
        word = match.group(0)
        i = rng.randrange(1, len(word) - 2)
        swapped = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if swapped == word:
            continue
        out = out[: match.start()] + swapped + out[match.end() :]
        edits.append({"from": word, "to": swapped})
    if not edits:
        return CorruptionResult.unchanged(text, "no transposition changed a word")
    return CorruptionResult(_unmask_code(out, stash), True, len(edits),
                            {"op": "inject_typos", "edits": edits})


_AGREEMENT = (
    (r"\bis\b", "are"), (r"\bare\b", "is"), (r"\bwas\b", "were"),
    (r"\bwere\b", "was"), (r"\bhas\b", "have"), (r"\bhave\b", "has"),
    (r"\bdoes\b", "do"), (r"\bdo\b", "does"),
)


def break_agreement(text: str, seed: int, severity: int) -> CorruptionResult:
    """Break subject-verb agreement and drop articles. Targets GRAMMATICALITY (down)."""
    masked, stash = _mask_code(text)
    rng = random.Random(seed)

    candidates: list[tuple[int, int, str, str]] = []
    for pattern, replacement in _AGREEMENT:
        for m in re.finditer(pattern, masked):
            candidates.append((m.start(), m.end(), m.group(0), replacement))
    for m in re.finditer(r"\b(the|a|an)\s+", masked):
        candidates.append((m.start(), m.end(), m.group(0), ""))
    if not candidates:
        return CorruptionResult.unchanged(text, "no agreement or article targets")

    rng.shuffle(candidates)
    picked = sorted(candidates[: max(1, severity)], key=lambda c: c[0], reverse=True)
    out = masked
    for start, end, _original, replacement in picked:
        out = out[:start] + replacement + out[end:]
    return CorruptionResult(_unmask_code(out, stash), True, len(picked),
                            {"op": "break_agreement", "n": len(picked)})


def break_capitalisation(text: str, seed: int, severity: int) -> CorruptionResult:
    """Lowercase sentence-initial letters. Targets CAPITALISATION (down)."""
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = [i for i in _prose_indices(parts) if parts[i].strip()[:1].isupper()]
    if not idx:
        return CorruptionResult.unchanged(text, "no capitalised sentence starts")

    rng = random.Random(seed)
    picked = rng.sample(idx, min(max(1, severity), len(idx)))
    rearranged = parts[:]
    for i in picked:
        stripped = rearranged[i].lstrip()
        lead = rearranged[i][: len(rearranged[i]) - len(stripped)]
        rearranged[i] = lead + stripped[0].lower() + stripped[1:]
    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    return CorruptionResult(out, True, len(picked),
                            {"op": "break_capitalisation", "n": len(picked)})


def strip_punctuation(text: str, seed: int, severity: int) -> CorruptionResult:
    """Remove internal commas and semicolons. Targets PUNCTUATION (down).

    Sentence-terminal punctuation is left alone so the sentence splitter -- and
    therefore every other operation's notion of a sentence -- stays valid.
    """
    masked, stash = _mask_code(text)
    marks = [m for m in re.finditer(r"[,;:]", masked)]
    if not marks:
        return CorruptionResult.unchanged(text, "no internal punctuation")

    rng = random.Random(seed)
    picked = sorted(
        rng.sample(marks, min(max(1, severity), len(marks))),
        key=lambda m: m.start(), reverse=True,
    )
    out = masked
    for m in picked:
        out = out[: m.start()] + out[m.end() :]
    return CorruptionResult(_unmask_code(out, stash), True, len(picked),
                            {"op": "strip_punctuation", "n": len(picked)})


# ---------------------------------------------------------- structure --------


def flatten_structure(text: str, seed: int, severity: int) -> CorruptionResult:
    """Collapse headings, bullets and paragraph breaks. Targets FORMATTING (down)."""
    masked, stash = _mask_code(text)
    if not re.search(r"(^|\n)\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s)|\n\s*\n", masked):
        return CorruptionResult.unchanged(text, "no structure to flatten")

    out = re.sub(r"(^|\n)\s*#{1,6}\s*", r"\1", masked)
    out = re.sub(r"(^|\n)\s*[-*+]\s+", r"\1", out)
    out = re.sub(r"(^|\n)\s*\d+[.)]\s+", r"\1", out)
    out = re.sub(r"\n\s*\n+", " ", out)
    out = re.sub(r"\n+", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    restored = _unmask_code(out, stash)
    if restored.strip() == text.strip():
        return CorruptionResult.unchanged(text, "flattening produced no change")
    return CorruptionResult(restored, True, 1, {"op": "flatten_structure"})


def scramble_list(text: str, seed: int, severity: int) -> CorruptionResult:
    """Reorder list items only. Targets LIST_INTEGRITY (down).

    Distinct from `shuffle`: it touches only enumerated lines, leaving prose order
    intact, which is what separates list integrity from coherence.
    """
    masked, stash = _mask_code(text)
    lines = masked.split("\n")
    idx = [
        i for i, line in enumerate(lines)
        if re.match(r"\s*([-*+]\s+|\d+[.)]\s+)\S", line)
    ]
    if len(idx) < 3:
        return CorruptionResult.unchanged(text, "fewer than 3 list items")

    rng = random.Random(seed)
    order = idx[:]
    for _ in range(20):
        rng.shuffle(order)
        if order != idx:
            break
    else:
        return CorruptionResult.unchanged(text, "shuffle produced original order")

    # Move item bodies but keep each line's original marker, so a numbered list
    # stays numbered 1..n and only the content order is wrong.
    bodies = {}
    for i in idx:
        m = re.match(r"(\s*(?:[-*+]\s+|\d+[.)]\s+))(.*)", lines[i])
        bodies[i] = (m.group(1), m.group(2))
    out_lines = lines[:]
    for target, source in zip(idx, order):
        out_lines[target] = bodies[target][0] + bodies[source][1]
    return CorruptionResult(
        _unmask_code("\n".join(out_lines), stash), True, len(idx),
        {"op": "scramble_list", "n_items": len(idx)},
    )


def truncate_tail(text: str, seed: int, severity: int) -> CorruptionResult:
    """Cut the final third of sentences. Targets COMPLETENESS (down)."""
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if len(idx) < 4:
        return CorruptionResult.unchanged(text, "too few sentences to truncate")

    # Severity selects how much of the tail is removed: 1 -> final third,
    # 2 -> final half, 3 -> final two thirds. Calibration measured a 40% tie rate
    # at severity 1.
    fraction = {1: 2 / 3, 2: 1 / 2}.get(severity, 1 / 3)
    keep = idx[: max(2, int(len(idx) * fraction))]
    cut = set(idx) - set(keep)
    rearranged = [p for i, p in enumerate(parts) if i not in cut]
    return CorruptionResult(
        _unmask_code("\n".join(rearranged), stash), True, len(cut),
        {"op": "truncate_tail", "removed": len(cut)},
    )


# ------------------------------------------------------------ content --------

_OFFTOPIC = (
    "Separately, the average domestic cat sleeps between twelve and sixteen hours a day.",
    "As an unrelated aside, the Hungarian rail network was largely electrified during the 1970s.",
    "Incidentally, sourdough fermentation depends on wild yeast strains present in the flour.",
    "On a different note, the tuba is the largest commonly used brass instrument.",
)


def insert_offtopic(text: str, seed: int, severity: int) -> CorruptionResult:
    """Insert unrelated sentences. Targets RELEVANCE (down)."""
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if not idx:
        return CorruptionResult.unchanged(text, "no prose to interleave")

    rng = random.Random(seed)
    rearranged = parts[:]
    inserted = []
    for _ in range(max(1, severity)):
        at = rng.choice(idx)
        sentence = rng.choice(_OFFTOPIC)
        rearranged[at] = rearranged[at].rstrip() + " " + sentence
        inserted.append(sentence)
    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    return CorruptionResult(out, True, len(inserted),
                            {"op": "insert_offtopic", "n": len(inserted)})


_NEGATE = (
    (r"\bis\b", "is not"), (r"\bare\b", "are not"), (r"\bcan\b", "cannot"),
    (r"\bwill\b", "will not"), (r"\bhas\b", "does not have"),
)


def insert_contradiction(text: str, seed: int, severity: int) -> CorruptionResult:
    """Append a sentence negating an earlier claim. Targets CONSISTENCY (down)."""
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if len(idx) < 2:
        return CorruptionResult.unchanged(text, "too few sentences to contradict")

    rng = random.Random(seed)
    for source in rng.sample(idx, len(idx)):
        sentence = parts[source].strip()
        for pattern, replacement in _NEGATE:
            if re.search(pattern, sentence):
                negated = re.sub(pattern, replacement, sentence, count=1)
                if negated == sentence:
                    continue
                rearranged = parts[:]
                last = idx[-1]
                rearranged[last] = rearranged[last].rstrip() + " " + negated
                out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
                return CorruptionResult(out, True, 1, {
                    "op": "insert_contradiction",
                    "negated": negated[:120],
                })
    return CorruptionResult.unchanged(text, "no negatable claim found")


_FILLER = (
    "It is important to note that", "It should be borne in mind that",
    "What this ultimately means is that", "As a general matter",
    "For all intents and purposes", "At the end of the day",
)


def insert_filler(text: str, seed: int, severity: int) -> CorruptionResult:
    """Prepend contentless phrases to sentences. Targets CONCISENESS (down).

    Adds no claims, so it should not move correctness -- which is what separates
    conciseness from verbosity in the leakage matrix.
    """
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if not idx:
        return CorruptionResult.unchanged(text, "no prose sentences")

    rng = random.Random(seed)
    rearranged = parts[:]
    for _ in range(max(1, severity)):
        at = rng.choice(idx)
        phrase = rng.choice(_FILLER)
        stripped = rearranged[at].lstrip()
        lead = rearranged[at][: len(rearranged[at]) - len(stripped)]
        rearranged[at] = (
            f"{lead}{phrase} {stripped[0].lower()}{stripped[1:]}" if stripped else rearranged[at]
        )
    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    return CorruptionResult(out, True, max(1, severity), {"op": "insert_filler"})


# Numerals are vaguified ONLY where they are genuine quantities. Matching any
# numeral would turn "Xbox 360" into "Xbox several", which is not a loss of
# specificity but a factual error -- it corrupts `correctness` while claiming to
# target `specificity`, exactly the mislabelling the leakage matrix exists to
# catch.
_VAGUE_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:percent|%)", re.I)
# A quantity is a numeral followed by a lowercase word (a unit or plural noun).
# The unit is retained and only the number is vaguified.
_VAGUE_QUANTITY = re.compile(r"\b(\d+(?:\.\d+)?)(\s+[a-z]{3,}\b)")
# Guard: a numeral immediately preceded by a capitalised token is part of a name
# (Xbox 360, Windows 11, Python 3), not a quantity.
_PROPER_NUMERAL = re.compile(r"\b[A-Z][\w.#+-]*\s+$")

_VAGUE_WORDS: dict[str, str] = {
    "specifically": "generally", "precisely": "roughly", "exactly": "approximately",
    "always": "often", "never": "rarely", "must": "might", "all": "some",
    "every": "various", "each": "certain",
}


def vaguify(text: str, seed: int, severity: int) -> CorruptionResult:
    """Replace concrete quantities and precise words with vague ones.

    Targets SPECIFICITY (down). Proper nouns containing numerals are protected --
    see the note on `_PROPER_NUMERAL`.
    """
    masked, stash = _mask_code(text)
    rng = random.Random(seed)
    out = masked
    edits: list[dict] = []

    for _ in range(max(1, severity)):
        match = _VAGUE_PERCENT.search(out)
        if match:
            out = out[: match.start()] + "a certain proportion" + out[match.end() :]
            edits.append({"kind": "percent", "from": match.group(0)})
            continue

        replaced = False
        for match in _VAGUE_QUANTITY.finditer(out):
            if _PROPER_NUMERAL.search(out[: match.start(1)]):
                continue  # part of a name, not a quantity
            out = (
                out[: match.start(1)] + "several" + match.group(2) + out[match.end(2) :]
            )
            edits.append({"kind": "quantity", "from": match.group(0).strip()})
            replaced = True
            break
        if replaced:
            continue

        available = [
            (w, r) for w, r in _VAGUE_WORDS.items()
            if re.search(rf"\b{w}\b", out, flags=re.I)
        ]
        if not available:
            break
        word, replacement = rng.choice(available)
        out = re.sub(rf"\b{word}\b", replacement, out, count=1, flags=re.I)
        edits.append({"kind": "word", "from": word, "to": replacement})

    if not edits:
        return CorruptionResult.unchanged(text, "nothing concrete to vaguify")
    return CorruptionResult(_unmask_code(out, stash), True, len(edits),
                            {"op": "vaguify", "edits": edits})


_HEDGES = (
    "arguably", "perhaps", "it may be that", "one might suppose that",
    "to some extent", "it could be argued that", "in some sense",
)


def inject_hedges(text: str, seed: int, severity: int) -> CorruptionResult:
    """Insert hedging language. Targets DIRECTNESS (down)."""
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if not idx:
        return CorruptionResult.unchanged(text, "no prose sentences")

    rng = random.Random(seed)
    rearranged = parts[:]
    for _ in range(max(1, severity)):
        at = rng.choice(idx)
        hedge = rng.choice(_HEDGES)
        stripped = rearranged[at].lstrip()
        lead = rearranged[at][: len(rearranged[at]) - len(stripped)]
        rearranged[at] = (
            f"{lead}{hedge.capitalize()}, {stripped[0].lower()}{stripped[1:]}"
            if stripped else rearranged[at]
        )
    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    return CorruptionResult(out, True, max(1, severity), {"op": "inject_hedges"})


def strip_support(text: str, seed: int, severity: int) -> CorruptionResult:
    """Remove references, figures and parenthetical justification. Targets SUPPORT (down)."""
    masked, stash = _mask_code(text)
    patterns = (r"\s*\([^)]{3,}\)", r"\s*\[\d+\]", r"\s*\bsee [A-Z][\w.]*",
                r"\s*\baccording to [^.,;]+")
    out = masked
    edits = 0
    for pattern in patterns:
        out, n = re.subn(pattern, "", out)
        edits += n
    if not edits:
        return CorruptionResult.unchanged(text, "no supporting material to strip")
    return CorruptionResult(_unmask_code(out, stash), True, edits,
                            {"op": "strip_support", "removed": edits})


def rename_entity(text: str, seed: int, severity: int) -> CorruptionResult:
    """Rename a recurring capitalised entity partway through. Targets TERMINOLOGY (down)."""
    masked, stash = _mask_code(text)
    counts: dict[str, list[re.Match]] = {}
    for m in re.finditer(r"\b[A-Z][a-zA-Z]{3,}\b", masked):
        counts.setdefault(m.group(0), []).append(m)
    recurring = {w: ms for w, ms in counts.items() if len(ms) >= 3}
    if not recurring:
        return CorruptionResult.unchanged(text, "no entity recurring 3+ times")

    rng = random.Random(seed)
    word = sorted(recurring, key=lambda w: (-len(recurring[w]), w))[
        rng.randrange(min(3, len(recurring)))
    ]
    matches = recurring[word]
    alias = word[:-1] + "ex" if len(word) > 4 else word + "ex"
    out = masked
    for m in reversed(matches[len(matches) // 2 :]):
        out = out[: m.start()] + alias + out[m.end() :]
    return CorruptionResult(_unmask_code(out, stash), True, len(matches) // 2 + 1,
                            {"op": "rename_entity", "from": word, "to": alias})


_SLANG = (
    "which is honestly kind of wild", "no cap", "which is a whole vibe",
    "basically a total mess", "which slaps, frankly",
)


def inject_slang(text: str, seed: int, severity: int) -> CorruptionResult:
    """Inject casual register into otherwise formal prose. Targets TONE (down)."""
    masked, stash = _mask_code(text)
    parts = split_sentences(masked)
    idx = _prose_indices(parts)
    if not idx:
        return CorruptionResult.unchanged(text, "no prose sentences")

    rng = random.Random(seed)
    rearranged = parts[:]
    for _ in range(max(1, severity)):
        at = rng.choice(idx)
        phrase = rng.choice(_SLANG)
        body = rearranged[at].rstrip()
        if body.endswith((".", "!", "?")):
            body = body[:-1] + f", {phrase}."
        else:
            body = body + f", {phrase}"
        rearranged[at] = body
    out = _unmask_code("\n".join(_rejoin(rearranged, masked)), stash)
    return CorruptionResult(out, True, max(1, severity), {"op": "inject_slang"})


# ------------------------------------------------------------ registry -------

# `vague` (target: specificity) was REMOVED after the severity ladder. Diagnosis:
# INVERTING -- adequately many decided cases, but discrimination fell to 0.40-0.69
# across severities 1-8, i.e. the judge systematically scored the vaguified text
# as MORE specific. Heavier vaguification does not make text read as less specific;
# it makes it read as damaged, and the judge's specificity score does not follow the
# asserted direction. Severity cannot repair a wrong direction.
#
# `jargon` (target: complexity) is retained in the CRITERION pool as packing load
# but is EXCLUDED from measurement. Diagnosis: MUTE -- 87-97% ties at every
# severity from 1 to 30, i.e. fewer than three decided cases per rung. Lexical
# jargon substitution does not move this judge's complexity score at all, which is
# a finding about the criterion rather than a defect in the operation.
#
# `unsupport` (target: support) was REMOVED after calibration. Its discrimination
# was 0.50 -- with ties excluded it was 4 hits against 4 wrong, i.e. a coin flip.
# That is a genuine accuracy failure rather than insensitivity, and severity cannot
# fix it: stripping parentheticals removes asides and hedges about as often as it
# removes actual evidential support, so the operation does not target its criterion.
# The `support` criterion consequently has no operation and is load-only.
EXTRA_CORRUPTIONS: tuple[Corruption, ...] = (
    Corruption("typos", "fluency", -1, 6, "Transpose characters inside words.", _apply=inject_typos),
    Corruption("agreement", "grammaticality", -1, 5, "Break agreement; drop articles.", _apply=break_agreement),
    Corruption("flatten", "formatting", -1, 1, "Collapse headings, lists and paragraphs.", _apply=flatten_structure, stochastic=False),
    Corruption("offtopic", "relevance", -1, 2, "Insert unrelated sentences.", _apply=insert_offtopic),
    Corruption("contradict", "consistency", -1, 1, "Append a negation of an earlier claim.", _apply=insert_contradiction),
    Corruption("filler", "conciseness", -1, 4, "Prepend contentless phrases.", _apply=insert_filler),
    Corruption("hedge", "directness", -1, 4, "Insert hedging language.", _apply=inject_hedges),
    Corruption("truncate", "completeness", -1, 8, "Cut the final third of sentences.", _apply=truncate_tail, stochastic=False),
    Corruption("rename", "terminology", -1, 1, "Rename a recurring entity partway through.", _apply=rename_entity),
    Corruption("slang", "tone", -1, 3, "Inject casual register into formal prose.", _apply=inject_slang),
    Corruption("lowercase", "capitalisation", -1, 5, "Lowercase sentence-initial letters.", _apply=break_capitalisation),
    Corruption("depunct", "punctuation", -1, 6, "Remove internal commas and semicolons.", _apply=strip_punctuation),
    Corruption("scramble_list", "list_integrity", -1, 1, "Reorder list item bodies.", _apply=scramble_list),
)
