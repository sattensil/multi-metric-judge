"""Criteria that exist only in the K = 40 ladder.

The synthetic stratum already extends K to 20 (`synthetic_criteria.py`). The
ladder needs 40 so that a load effect has room to appear well past the point the
K = 20 sweep could resolve.

These twenty have no human-label counterpart and carry no ground truth. In the
ladder they are companions: they exist to create load behind a first-position
target. That is the one role unlabelled criteria can legitimately play, because
only the target needs a label. METHODS section 10 records why unlabelled padding
was rejected as a source of ground truth and is admissible as load.

Every criterion anchors all five scale points, for the reason established in
`criteria.py`: a partially anchored scale is adopted by the model as the whole
scale.

Manifest hash: 721b8ac2a2a7e15fffd7ec2b65865afa426a8762c3097efd3f4a603c9431d402
Pool character lengths: min 168, max 447, mean 289, total 11,563.
These twenty: 251 to 325 characters, a 1.29x spread, 80 to 101 rendered tokens.
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
            0: "nothing here can be acted on at all",
            1: "a direction is implied but no usable step is given",
            2: "some steps are usable, others need clarifying first",
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
            0: "rests entirely on assumptions it never states",
            1: "leans on major assumptions without naming any",
            2: "names some assumptions and leaves others hidden",
            3: "names the load-bearing ones; a minor one stays implicit",
            4: "every assumption the answer depends on is stated",
        },
    ),
    Criterion(
        key="attribution",
        label="Attribution",
        description=(
            "Whether claims taken from sources are attributed rather than "
            "presented bare."
        ),
        anchors={
            0: "borrowed claims are presented as the writer's own",
            1: "sources gestured at vaguely, never identified",
            2: "some claims attributed, comparable ones left bare",
            3: "attribution present throughout, one reference imprecise",
            4: "every sourced claim carries a usable attribution",
        },
    ),
    Criterion(
        key="audience_fit",
        label="Audience fit",
        description=(
            "Whether the register and depth match the apparent expertise of the "
            "asker."
        ),
        anchors={
            0: "pitched so far off that it is unusable to this asker",
            1: "consistently too basic or too advanced for the question",
            2: "roughly aimed, with passages that miss the level",
            3: "well aimed apart from an occasional lapse in depth",
            4: "depth and register match the asker throughout",
        },
    ),
    Criterion(
        key="caveating",
        label="Caveating",
        description=(
            "Whether genuine limitations are stated without hedging everything."
        ),
        anchors={
            0: "either no caveats at all or hedged past usefulness",
            1: "limitations mostly absent, or blanket hedging throughout",
            2: "some real caveats, mixed with needless hedging",
            3: "limitations stated cleanly, with slight over-hedging",
            4: "the real limits are named and nothing else is hedged",
        },
    ),
    Criterion(
        key="cohesion",
        label="Cohesion",
        description=(
            "Whether sentences connect so the argument carries across "
            "paragraphs."
        ),
        anchors={
            0: "sentences sit next to each other without connecting",
            1: "frequent jumps the reader has to bridge unaided",
            2: "connects locally but loses the thread between sections",
            3: "carries through, with one seam the reader notices",
            4: "the argument connects cleanly from start to finish",
        },
    ),
    Criterion(
        key="concreteness",
        label="Concreteness",
        description="Whether general claims are grounded in specific detail.",
        anchors={
            0: "entirely abstract; no specific detail anywhere",
            1: "generalities with one token specific mentioned in passing",
            2: "some claims grounded, the central ones left abstract",
            3: "grounded throughout, with one claim left general",
            4: "every general claim is anchored to specific detail",
        },
    ),
    Criterion(
        key="edge_cases",
        label="Edge-case coverage",
        description=(
            "Whether boundary conditions implied by the question are addressed."
        ),
        anchors={
            0: "handles only the happy path and ignores the rest",
            1: "acknowledges that edge cases exist without addressing them",
            2: "covers the obvious boundaries, misses the harder ones",
            3: "covers the boundaries that matter, one left unstated",
            4: "every boundary the question implies is addressed",
        },
    ),
    Criterion(
        key="example_quality",
        label="Example quality",
        description=(
            "Whether examples illustrate the point rather than restating it."
        ),
        anchors={
            0: "no examples where examples were clearly needed",
            1: "examples restate the claim without illuminating it",
            2: "one example illustrates, the others merely repeat",
            3: "examples illustrate well, one is thinner than the rest",
            4: "each example earns its place by adding understanding",
        },
    ),
    Criterion(
        key="internal_reference",
        label="Internal reference",
        description=(
            "Whether references to the response's own parts resolve correctly."
        ),
        anchors={
            0: "internal references point at material that is not there",
            1: "several references resolve to the wrong part",
            2: "most references resolve, a couple are wrong or vague",
            3: "references resolve, with one ambiguous pointer",
            4: "every internal reference resolves to the right place",
        },
    ),
    Criterion(
        key="ordering",
        label="Ordering",
        description=(
            "Whether information arrives in an order the reader can follow."
        ),
        anchors={
            0: "order actively obstructs understanding",
            1: "key material arrives well after it was needed",
            2: "broadly sensible, with sections that arrive too late",
            3: "well ordered apart from one displaced passage",
            4: "each part arrives exactly when the reader needs it",
        },
    ),
    Criterion(
        key="precision",
        label="Precision",
        description=(
            "Whether quantities, units and qualifiers are stated exactly where "
            "they matter."
        ),
        anchors={
            0: "vague everywhere that precision was required",
            1: "quantities given loosely, qualifiers missing throughout",
            2: "precise in places, imprecise in the load-bearing claims",
            3: "precise where it matters, one figure left approximate",
            4: "every claim that needs precision has it",
        },
    ),
    Criterion(
        key="question_coverage",
        label="Question coverage",
        description=(
            "Whether every part of a multi-part question is answered."
        ),
        anchors={
            0: "answers none of what was actually asked",
            1: "answers one part and ignores the remainder",
            2: "answers most parts, leaves a substantive one untouched",
            3: "answers every part, one more briefly than it warranted",
            4: "every part of the question is answered in full",
        },
    ),
    Criterion(
        key="redundancy",
        label="Redundancy",
        description="Whether the response avoids repeating itself.",
        anchors={
            0: "the same content restated over and over",
            1: "substantial passages duplicate earlier material",
            2: "noticeable repetition of several points",
            3: "largely free of repetition, one point made twice",
            4: "says each thing once and moves on",
        },
    ),
    Criterion(
        key="register",
        label="Register",
        description="Whether tone is consistent throughout rather than drifting.",
        anchors={
            0: "tone lurches between registers paragraph by paragraph",
            1: "frequent shifts in formality that distract the reader",
            2: "broadly consistent with several noticeable slips",
            3: "consistent apart from one passage that shifts",
            4: "one register held steadily from start to finish",
        },
    ),
    Criterion(
        key="scope_discipline",
        label="Scope discipline",
        description="Whether the response stays inside what was asked.",
        anchors={
            0: "answers a different question than the one posed",
            1: "large digressions crowd out the actual question",
            2: "answers the question alongside unrequested material",
            3: "stays on scope, with one brief excursion",
            4: "addresses exactly what was asked and nothing else",
        },
    ),
    Criterion(
        key="self_consistency",
        label="Self consistency",
        description="Whether later statements agree with earlier ones.",
        anchors={
            0: "contradicts itself on the central claim",
            1: "several statements conflict with earlier ones",
            2: "mostly consistent, with one clear contradiction",
            3: "consistent, with a tension the reader must reconcile",
            4: "every statement agrees with the others",
        },
    ),
    Criterion(
        key="signposting",
        label="Signposting",
        description=(
            "Whether structure is announced so the reader knows what is coming."
        ),
        anchors={
            0: "no signposting where the length clearly demanded it",
            1: "the reader has to infer the structure unaided",
            2: "some sections announced, transitions left implicit",
            3: "well signposted, with one unannounced turn",
            4: "the reader always knows where they are and what follows",
        },
    ),
    Criterion(
        key="terminology_consistency",
        label="Terminology consistency",
        description="Whether one term is used for one concept throughout.",
        anchors={
            0: "terms swapped so freely the concepts blur together",
            1: "several concepts named two or three different ways",
            2: "mostly stable, with one concept renamed midway",
            3: "consistent, with a single synonym used once",
            4: "each concept keeps exactly one name throughout",
        },
    ),
    Criterion(
        key="unit_handling",
        label="Unit handling",
        description=(
            "Whether units are present, correct and consistently converted."
        ),
        anchors={
            0: "units missing or wrong wherever they matter",
            1: "units frequently absent, conversions unreliable",
            2: "units mostly present, one conversion is wrong",
            3: "units correct throughout, one left off a figure",
            4: "every quantity carries a correct, consistent unit",
        },
    ),
)

LADDER_CRITERIA: tuple[Criterion, ...] = (
    HELPSTEER2_CRITERIA + SYNTHETIC_ONLY_CRITERIA + LADDER_ONLY_CRITERIA
)

LADDER_MANIFEST: CriterionManifest = manifest_for(LADDER_CRITERIA)
