"""Criteria that exist only in the synthetic stratum.

The five HelpSteer2 attributes are the labelled ceiling: no public corpus supplies
more than roughly five fixed, fully-annotated criteria per item. But the question
"where does packing start to hurt" cannot be answered at K = 5, and a null at 5
with no known breaking point is a weak result.

Corruption ground truth is created by construction, so it is not bound by that
ceiling. These fifteen criteria extend K to 20. They have no human-label
counterpart and therefore appear only in the synthetic stratum; the five anchor
criteria remain in both, which is what makes the strata comparable.

Every criterion here anchors all five scale points, for the reason established in
`criteria.py`: a partially anchored scale is adopted by the model as the whole
scale.
"""

from __future__ import annotations

from .criteria import Criterion

SYNTHETIC_ONLY_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="fluency",
        label="Fluency",
        description="Surface readability: spelling, word-level typos, and typographic errors.",
        anchors={
            0: "riddled with typos; hard to read at all",
            1: "frequent typos that interrupt reading",
            2: "several noticeable typos",
            3: "one or two minor typos",
            4: "clean; no spelling or typographic errors",
        },
    ),
    Criterion(
        key="grammaticality",
        label="Grammaticality",
        description="Syntactic correctness: agreement, articles, tense, and well-formed clauses.",
        anchors={
            0: "pervasive grammatical errors; syntax breaks down",
            1: "frequent agreement or article errors",
            2: "several grammatical errors",
            3: "largely well-formed; an isolated slip",
            4: "grammatically correct throughout",
        },
    ),
    Criterion(
        key="formatting",
        label="Formatting",
        description=(
            "Use of document structure: headings, lists, and paragraphing that match "
            "the content's shape."
        ),
        anchors={
            0: "no structure where structure was clearly needed",
            1: "structure largely collapsed into undifferentiated text",
            2: "some structure, inconsistently applied",
            3: "mostly well structured; minor lapses",
            4: "structure fits the content throughout",
        },
    ),
    Criterion(
        key="relevance",
        label="Relevance",
        description="Whether every part of the response bears on what was asked.",
        anchors={
            0: "largely off topic",
            1: "substantial off-topic content",
            2: "noticeable digressions",
            3: "on topic with a slight digression",
            4: "entirely on topic; nothing extraneous",
        },
    ),
    Criterion(
        key="consistency",
        label="Internal consistency",
        description="Whether the response contradicts itself.",
        anchors={
            0: "flatly self-contradictory",
            1: "multiple internal contradictions",
            2: "one clear contradiction",
            3: "consistent, with a slight tension",
            4: "fully self-consistent",
        },
    ),
    Criterion(
        key="conciseness",
        label="Conciseness",
        description=(
            "Absence of filler. Distinct from verbosity: this is about padding "
            "phrases that carry no content, not overall length."
        ),
        anchors={
            0: "dominated by filler phrasing",
            1: "heavy padding throughout",
            2: "noticeable filler in places",
            3: "mostly tight; occasional padding",
            4: "no filler; every phrase carries content",
        },
    ),
    Criterion(
        key="specificity",
        label="Specificity",
        description="Whether claims are concrete and named rather than vague.",
        anchors={
            0: "entirely vague; no concrete detail",
            1: "mostly vague generalities",
            2: "a mix of concrete and vague",
            3: "mostly concrete; a few vague passages",
            4: "consistently concrete and named",
        },
    ),
    Criterion(
        key="directness",
        label="Directness",
        description=(
            "Whether claims are asserted plainly rather than buried in hedging and "
            "qualification."
        ),
        anchors={
            0: "so hedged that no claim is discernible",
            1: "heavily hedged throughout",
            2: "noticeably hedged in places",
            3: "mostly direct; some qualification",
            4: "direct; qualifies only where genuinely warranted",
        },
    ),
    Criterion(
        key="completeness",
        label="Completeness",
        description="Whether the response finishes the job rather than stopping short.",
        anchors={
            0: "breaks off almost immediately",
            1: "stops well short of a complete answer",
            2: "covers part of what was needed",
            3: "nearly complete; a small gap",
            4: "complete; nothing important left unaddressed",
        },
    ),
    Criterion(
        key="support",
        label="Evidential support",
        description=(
            "Whether claims are backed by references, figures, or explicit "
            "justification."
        ),
        anchors={
            0: "bare assertions with no support",
            1: "mostly unsupported claims",
            2: "some claims supported, many not",
            3: "mostly supported; a gap or two",
            4: "claims consistently supported",
        },
    ),
    Criterion(
        key="terminology",
        label="Terminological consistency",
        description="Whether the same thing is called the same name throughout.",
        anchors={
            0: "names shift constantly; referents unclear",
            1: "frequent inconsistent naming",
            2: "one recurring term used inconsistently",
            3: "consistent, with a single slip",
            4: "terminology stable throughout",
        },
    ),
    Criterion(
        key="tone",
        label="Tonal consistency",
        description="Whether register stays uniform rather than lurching between styles.",
        anchors={
            0: "register lurches between extremes",
            1: "frequent tonal mismatches",
            2: "a noticeable tonal break",
            3: "consistent tone with a slight wobble",
            4: "uniform register throughout",
        },
    ),
    Criterion(
        key="capitalisation",
        label="Capitalisation",
        description="Conventional use of capital letters.",
        anchors={
            0: "capitalisation essentially random",
            1: "frequent capitalisation errors",
            2: "several capitalisation errors",
            3: "one or two lapses",
            4: "conventional throughout",
        },
    ),
    Criterion(
        key="punctuation",
        label="Punctuation",
        description="Conventional and helpful use of punctuation.",
        anchors={
            0: "punctuation largely absent; sentences run together",
            1: "frequent missing or misused punctuation",
            2: "several punctuation errors",
            3: "minor punctuation lapses",
            4: "punctuation correct and helpful",
        },
    ),
    Criterion(
        key="list_integrity",
        label="List integrity",
        description=(
            "Whether enumerated or sequential items appear in a sensible order. "
            "Score 4 if the response contains no list."
        ),
        anchors={
            0: "list order scrambled and misleading",
            1: "list order substantially wrong",
            2: "some list items out of order",
            3: "order sensible; one item misplaced",
            4: "list order correct, or no list present",
        },
    ),
)

SYNTHETIC_KEYS = tuple(c.key for c in SYNTHETIC_ONLY_CRITERIA)
