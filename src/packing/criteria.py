"""Criterion definitions with fully anchored scales.

Every scale point carries an explicit descriptor. This is not a style preference,
it is a measured requirement:

In the first live run against `gpt-5-mini`, the criteria were anchored only at
0, 2 and 4. The judge then used *only* 0, 2 and 4 -- never 1 or 3 -- across all
five criteria (judge score deltas were distributed {0: 65, 1: 1, 2: 34}). It was
effectively working with a 3-point scale, and tied on **66% of human-scoreable
pairs**, collapsing 100 observations to 29 usable ones.

A partially anchored scale becomes the scale. `__post_init__` now rejects any
criterion that leaves a scale point undescribed.

Note on HelpSteer2: `complexity` and `verbosity` are *descriptive*, not
evaluative -- a higher score is not better. Pairwise "correctness" on those two
means the judge ordered the pair the way humans did, which is a valid accuracy
measure but not a preference judgement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    description: str
    anchors: dict[int, str]
    scale_min: int = 0
    scale_max: int = 4
    evaluative: bool = True
    """False for descriptive dimensions where higher is not better."""

    def __post_init__(self) -> None:
        if self.scale_min >= self.scale_max:
            raise ValueError(
                f"{self.key}: scale_min {self.scale_min} must be < scale_max {self.scale_max}"
            )
        points = set(range(self.scale_min, self.scale_max + 1))
        stray = set(self.anchors) - points
        if stray:
            raise ValueError(f"{self.key}: anchors outside scale: {sorted(stray)}")
        missing = points - set(self.anchors)
        if missing:
            raise ValueError(
                f"{self.key}: every scale point needs an anchor; missing {sorted(missing)}. "
                f"A partially anchored scale becomes the scale -- see module docstring."
            )

    @property
    def scale_points(self) -> list[int]:
        return list(range(self.scale_min, self.scale_max + 1))

    def render(self) -> str:
        """Rubric block for the judge prompt."""
        lines = [f"### {self.label} ({self.key})", self.description, "", "Scale:"]
        for value in sorted(self.anchors):
            lines.append(f"  {value} = {self.anchors[value]}")
        return "\n".join(lines)


HELPSTEER2_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="helpfulness",
        label="Helpfulness",
        description=(
            "How useful and helpful the response is overall in addressing what the "
            "prompt actually asked for."
        ),
        anchors={
            0: "not helpful at all; misses the point or ignores the request",
            1: "mostly unhelpful; touches the topic but fails the actual request",
            2: "partially helpful; addresses the prompt but with real gaps",
            3: "mostly helpful; addresses the prompt with only minor gaps",
            4: "perfectly helpful; fully addresses the prompt",
        },
    ),
    Criterion(
        key="correctness",
        label="Correctness",
        description=(
            "Whether the response is factually accurate and includes all pertinent "
            "information the prompt requires, with no fabrications."
        ),
        anchors={
            0: "completely incorrect, or entirely irrelevant to the prompt",
            1: "mostly incorrect; major errors or fabrications dominate",
            2: "partially correct; contains some errors or notable omissions",
            3: "mostly correct; minor errors or small omissions only",
            4: "completely correct and fully relevant; no inaccuracies",
        },
    ),
    Criterion(
        key="coherence",
        label="Coherence",
        description=(
            "Whether the response is clear, self-consistent, and logically "
            "organised, with no contradictions or non-sequiturs."
        ),
        anchors={
            0: "completely incoherent; incomprehensible or self-contradictory",
            1: "largely incoherent; structure and logic frequently break down",
            2: "somewhat coherent; noticeable confusing or inconsistent passages",
            3: "mostly coherent; minor lapses in flow or consistency",
            4: "perfectly coherent, clear and consistent throughout",
        },
    ),
    Criterion(
        key="complexity",
        label="Complexity",
        description=(
            "The intellectual depth of the language used. DESCRIPTIVE, not "
            "evaluative: a higher score is not better, it only means the response "
            "requires more domain expertise to have written."
        ),
        anchors={
            0: "basic; simple vocabulary any literate person would use",
            1: "simple; everyday language, no specialist terms",
            2: "intermediate; some domain vocabulary or multi-step reasoning",
            3: "advanced; clear domain expertise evident throughout",
            4: "expert; requires deep specialist knowledge to have written",
        },
        evaluative=False,
    ),
    Criterion(
        key="verbosity",
        label="Verbosity",
        description=(
            "Amount of detail relative to what the prompt asked for. DESCRIPTIVE, "
            "not evaluative: a higher score is not better, it only means longer "
            "and more elaborated."
        ),
        anchors={
            0: "very terse; minimal detail for the question asked",
            1: "concise; slightly less detail than the question invited",
            2: "moderate; length proportionate to the question",
            3: "somewhat verbose; more elaboration than needed",
            4: "very verbose; substantially more detail or repetition than needed",
        },
        evaluative=False,
    ),
)


SUMMEVAL_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="coherence",
        label="Coherence",
        description=(
            "The collective quality of all sentences: does the summary build a "
            "coherent body of information about the topic, rather than a heap of "
            "related facts?"
        ),
        anchors={
            1: "incoherent; a disconnected heap of facts",
            2: "largely disconnected; little organising structure",
            3: "partly coherent; organisation is uneven",
            4: "mostly coherent; minor structural lapses",
            5: "fully coherent, well-structured and well-organised",
        },
        scale_min=1,
        scale_max=5,
    ),
    Criterion(
        key="consistency",
        label="Consistency",
        description=(
            "Factual alignment between the summary and the source. A consistent "
            "summary contains only statements entailed by the source document."
        ),
        anchors={
            1: "largely unsupported by the source; hallucinated",
            2: "several statements not entailed by the source",
            3: "mostly supported, with some unsupported statements",
            4: "nearly all supported; one minor unsupported detail",
            5: "every statement is entailed by the source document",
        },
        scale_min=1,
        scale_max=5,
    ),
    Criterion(
        key="fluency",
        label="Fluency",
        description=(
            "The quality of individual sentences: grammar, spelling, and readability."
        ),
        anchors={
            1: "many errors; hard to read",
            2: "frequent errors that impede reading",
            3: "some errors but readable",
            4: "largely well-formed; occasional minor error",
            5: "no errors; fluent and well-formed throughout",
        },
        scale_min=1,
        scale_max=5,
    ),
    Criterion(
        key="relevance",
        label="Relevance",
        description=(
            "Selection of important content from the source. Does the summary "
            "include the salient information and exclude redundancy?"
        ),
        anchors={
            1: "mostly unimportant or redundant content",
            2: "substantial filler or redundancy",
            3: "some important content, some filler",
            4: "mostly important content; slight redundancy",
            5: "only the most important content from the source",
        },
        scale_min=1,
        scale_max=5,
    ),
)


CRITERIA_SETS: dict[str, tuple[Criterion, ...]] = {
    "helpsteer2": HELPSTEER2_CRITERIA,
    "summeval": SUMMEVAL_CRITERIA,
}


def get_criteria(dataset: str) -> tuple[Criterion, ...]:
    try:
        return CRITERIA_SETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(CRITERIA_SETS)}"
        ) from None
