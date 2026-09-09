# Methods

## 1. Research question

An LLM judge must read the full input before it can score anything. When several
quality dimensions are each scored by a separate judge call, that read is paid for
once per dimension. Combining the dimensions into a single call pays for it once
in total. The question is whether that consolidation degrades the resulting
scores, and where the accuracy/cost frontier lies.

**Independent variable.** `group_size`: the number of criteria scored in one judge
call.

**Dependent variables.** Per-criterion agreement with ground truth (quality) and
tokens consumed per criterion scored (cost).

**Held constant.** The set of criteria. Every condition scores all of them; only
the packing changes.

**Primary hypothesis (H1).** Per-criterion accuracy declines as `group_size`
increases.

**Rival hypothesis (H2), pre-registered.** Any apparent effect of `group_size` is
attributable to a criterion's ordinal *position* within the prompt rather than to
the number of criteria sharing the call. H2 is not a robustness check; it implies a
different practical recommendation (order your criteria) and is therefore modelled
alongside H1.

## 2. Design

Within-item repeated measures. Every item is judged at every level of
`group_size`, so item difficulty is differenced out rather than modelled. Two
ground-truth strata are analysed separately and never pooled:

| Stratum | Ground truth | Label noise |
|---|---|---|
| `human` | HelpSteer2 aggregated annotator scores | quadratic weighted Cohen's κ 0.427–0.793 |
| `synthetic` | Corruption applied by construction | none |

**Synthetic gold is operation recovery, not criterion-isolated human quality.**
The synthetic estimand asks whether the judge moves the *targeted* score in the
corruption's known direction. Conclusions attach to operations (e.g. "detect
injected typos", "recover a deleted answer"), not to abstract criteria as humans
annotate them. Off-target score movement is reported as a construct-overlap
diagnostic; it does not automatically invalidate the recovery outcome. Agreement
between the human and synthetic strata is informative only where an operation
faithfully instantiates the annotated faculty (§9.1a); no separate human
re-labelling protocol is used.

## 3. Materials

### 3.1 Judge model

`gpt-5-mini` (OpenAI), $0.25 / $2.00 per 10⁶ input / output tokens. Selected by a
pre-run screening step (§4.4) rather than on price: the design requires a judge
with headroom to degrade, since a judge already at chance cannot exhibit decline.

Scores are elicited through OpenAI Structured Outputs (schema-constrained
decoding) with `strict: true`, `additionalProperties: false`, and every criterion
declared `required`, so a call cannot silently omit a criterion. Score fields are
declared as explicit `enum` sets rather than `minimum`/`maximum`, which are not
reliably honoured under constrained decoding.

`temperature` is not set. Reducing sampling temperature to stabilise judge output
can increase the minimum variance of the score data and introduce bias by altering
the variance of conditional means rather than only the noise (Miller, 2024);
variance reduction is instead obtained through replication (§6.3). `reasoning_effort`
is held fixed across all conditions so that reasoning depth cannot act as an
uncontrolled mediator of a `group_size` effect.

### 3.2 Response corpus

HelpSteer2 (NVIDIA, CC-BY-4.0). The train split contains 20,324 rows over 10,161
distinct prompts; 10,160 prompts have exactly two responses and are retained (one
is discarded for having an irregular response count). Each response carries a
single integer score on five attributes (helpfulness, correctness, coherence,
complexity, verbosity) on a 0-4 Likert scale, from at least three annotators,
with two additional annotators recruited where initial disagreement exceeded two
points.

The aggregation function mapping annotator scores to the released integer is not
documented in the dataset publication, and individual annotator scores are not
released. Released values are integers, which excludes an unrounded mean; no
further determination is possible. Per-item annotator agreement therefore cannot
be computed, and only the published dataset-level agreement is available:

| Criterion | Human–human κ (quadratic weighted) |
|---|---|
| correctness | 0.793 |
| helpfulness | 0.791 |
| verbosity | 0.548 |
| coherence | 0.428 |
| complexity | 0.427 |

Judge agreement is therefore reported **relative to these ceilings**, not to 1.0
(§7.3).

### 3.3 Criteria and rubric construction

The same five HelpSteer2 attributes constitute the criterion set (K = 5). Two
(`complexity`, `verbosity`) are descriptive rather than evaluative: a higher score
denotes more of the named property, not better quality. Pairwise agreement on those
two measures whether the judge orders items as annotators did, and is not a
preference judgement.

**Every point on every scale carries an explicit anchor descriptor.** This
requirement is empirical, not stylistic. In an initial live run, criteria were
anchored at scale points 0, 2 and 4 only. The judge then used only 0, 2 and 4,
never 1 or 3, across all five criteria; judge score differences were distributed
{0: 65, 1: 1, 2: 34}, i.e. it operated a three-point scale. It produced tied scores
on 66.3% of scoreable pairs, reducing 100 observations to 29. Supplying anchors for
all five points reduced the tie rate to 43.0%, raised usable observations to 49,
and brought all five scale values into use. A partially anchored scale is adopted
by the model as the whole scale. The constraint is enforced in code: a criterion
with an undescribed scale point raises at construction.

The system prompt instructs that each criterion be scored independently on its own
definition. It does not tell the judge to "use the full scale", since a judge
handed that instruction alongside partial anchors reads it as licence to use
exactly the anchored values.

### 3.4 Synthetic corruption operations

Each operation is applied to a clean base response, targets one criterion, and
declares the direction the targeted score must move. **Ground truth is an ordering
relation, not an absolute score**: a padded response is more verbose than its base;
it is not "verbosity = 4."

Twenty operations were designed. Two do not survive calibration and are absent
from the shipped catalogue, which holds **eighteen**: `unsupport` and `vague` were
removed (§10), while `jargon` is retained as packing load and excluded from
measurement. The table below is the design, with the outcome of each recorded in
§8.2.

| Operation | Target criterion | Dir | Sev | Stoch | Applicability | Detection class (hypothesised) |
|---|---|---|---|---|---|---|
| `contradict` | consistency | ↓ | 1 | Y | 95.0% | **semantic** |
| `drop_answer` | helpfulness | ↓ | 1 | N | 95.5% | **semantic** |
| `numbers` | correctness | ↓ | 2 | Y | 24.2% | **semantic** |
| `rename` | terminology | ↓ | 1 | Y | 62.5% | **semantic** |
| `shuffle` | coherence | ↓ | 1 | Y | 95.5% | **semantic** |
| `truncate` | completeness | ↓ | 1 | N | 91.0% | **semantic** |
| `unsupport` | support | ↓ | 1 | N | 25.5% | **semantic** |
| `vague` | specificity | ↓ | 4 | Y | 47.2% | **semantic** |
| `agreement` | grammaticality | ↓ | 5 | Y | 99.0% | **surface** |
| `depunct` | punctuation | ↓ | 6 | Y | 99.5% | **surface** |
| `duplicate` | verbosity | ↑ | 3 | Y | 100.0% | **surface** |
| `filler` | conciseness | ↓ | 4 | Y | 100.0% | **surface** |
| `flatten` | formatting | ↓ | 1 | N | 85.8% | **surface** |
| `hedge` | directness | ↓ | 4 | Y | 100.0% | **surface** |
| `jargon` | complexity | ↑ | 8 | Y | 94.2% | **surface** |
| `lowercase` | capitalisation | ↓ | 5 | Y | 99.5% | **surface** |
| `offtopic` | relevance | ↓ | 2 | Y | 100.0% | **surface** |
| `scramble_list` | list_integrity | ↓ | 1 | Y | 43.0% | **surface** |
| `slang` | tone | ↓ | 3 | Y | 100.0% | **surface** |
| `typos` | fluency | ↓ | 6 | Y | 100.0% | **surface** |

Direction (↓ / ↑) is the direction the targeted score must move. Severity is the
configured edit count. Stoch indicates whether the operation consumes its seed
(§ invariant 1). Applicability is the proportion of 400 real HelpSteer2 responses
to which the operation could be applied.

#### Detection class, and why the analysis is stratified by it

The catalogue divides into two classes that measure different faculties, and
pooling them would be a serious error.

**Surface defects** (12 operations) are lexically detectable. Recognising
transposed characters, absent commas, casual register in formal prose, or an
inserted sentence about brass instruments requires no comprehension of the
response's content. Detection is expected at or near ceiling, and to remain there
at every group size.

**Semantic defects** (8 operations) require reading and reasoning: identifying an
incorrect figure needs world knowledge, a contradiction needs two claims held
simultaneously, a renamed entity needs coreference tracking, a deleted answer
sentence needs a model of what the prompt asked.

Only the semantic class can expose a capability limit. Pooling the two would
dilute any semantic effect toward the null, because eleven-to-twelve criteria
pinned at ceiling would average away decline in the remainder, yielding "no
degradation observed" where the correct reading is "the task was too easy to
degrade." The primary `group_size` estimand is therefore reported **within
detection class**, never pooled across it.

The substantive hypothesis this permits is sharper than a single frontier:
operations detectable without comprehension remain flat under packing, while
operations requiring comprehension may not. If supported, the recommendation is
conditional rather than global: consolidation is safe for mechanical checks and
hazardous for judgement.

Note the careful phrasing. The hypothesis concerns **operations**, not criteria.
Detection class is a property of the operation chosen to instantiate a criterion,
not of the criterion itself, and criterion-level conclusions transfer only as far
as that instantiation is faithful. See §9.1a, which is the governing limitation on
everything the synthetic stratum can establish.

**Operations at ceiling are retained as positive controls, not discarded.** If
injected typos go undetected at high group size, the judge has ceased reading
rather than degraded gradually. That is a distinct and consequential failure mode,
and a ceiling-bound operation is the instrument that detects it.

**The classification above is a hypothesis, not a measurement.** It reflects
expected difficulty and has not been observed. The calibration run (§4.4) scores
all twenty operations at `group_size` = 1, and each operation is then assigned to
test-item or positive-control status **by measured detection rate** against the
ceiling threshold. Where measurement contradicts the hypothesised class, the
measurement governs and the discrepancy is reported.

**Severity is calibrated per operation, not globally.** Six transposed characters
in four hundred words is trivially detectable; one is not. Severity is the
parameter that moves a surface operation off the ceiling, and a single global
value would leave most of the catalogue uninformative.

#### Direction asymmetry

Eighteen of twenty operations move the targeted score down; only `duplicate` and
`jargon` move it up. A judge that merely registers "this text has been damaged"
would therefore score well on most of the catalogue without engaging any specific
criterion. The specificity measure (§5.3) is the intended control, but the
asymmetry weakens it, since direction is largely predictable from the mere fact of
change. This is a known limitation of the present catalogue; adding upward
operations (introducing structure to target formatting, adding references to
target support) would strengthen the control and is recorded as future work rather
than claimed.

`numbers` is limited to the 24.2% of responses
containing a numeral outside a code span, which reduces attainable *n* for
`correctness` in the synthetic stratum; this is a sampling restriction on that
criterion, not a bias in the `group_size` contrast, since eligibility is
independent of packing.

Three invariants are enforced by test:

1. **Determinism.** Identical `(text, seed, severity)` yields identical output.
   Required because the execution log is keyed on call identity and resumption must
   not silently produce different items.

   Operations declare whether they are stochastic. `drop_answer` is deterministic
   by construction (its target is the argmax of lexical overlap with the prompt, so
   there is nothing to randomise) and therefore produces identical variant text on
   every replicate. Replication for that operation samples only the judge's own
   nondeterminism, so the variance reduction available from replication (§6.3) is
   correspondingly smaller. `jargon` is stochastic but degenerate where a text
   offers fewer substitutable words than the configured severity. A test asserts
   that each operation's declared stochasticity matches its measured
   seed-sensitivity, so the declaration cannot drift from the implementation.
2. **Honest failure.** An operation that cannot apply returns `changed=False` and
   the item is discarded. Emitting an unmodified variant while asserting a
   ground-truth difference would inject false labels.
3. **Code preservation.** Fenced and inline code spans are masked before editing
   and restored afterwards. Editing code would perturb several criteria at once and
   destroy the targeting the design depends on.

Severity is fixed per operation, not swept. It is a nuisance parameter calibrated
once (§4.4) so that detection at `group_size` = 1 falls within a measurable band;
sweeping it would answer a different question.

Operations are selected in inverse order of human-label quality. `coherence` and
`complexity` (κ ≈ 0.43) are the highest-value targets, because against a reference
that unreliable, "the judge scores coherence poorly" and "annotators cannot agree
on coherence" are not distinguishable. `drop_answer` is flagged leakage-prone by
construction: removing a sentence necessarily disturbs coherence as well as
helpfulness.

## 4. Procedure

### 4.1 Randomisation

Three independent randomisations, all seeded from
`SHA-256(base_seed, item_id, nominal_n, replicate)` so that adjacent design cells do
not share correlated RNG streams:

1. **Partition.** For each (item, nominal N, replicate), the K criteria are
   partitioned uniformly at random into blocks of size ≤ N. Without this, a
   `group_size` effect is confounded with criterion-pair interaction.
2. **Order within call.** Criterion order inside each block is randomised and each
   criterion's ordinal `position` recorded. This is what makes H2 testable.
3. **Response order** in the comparison.

Uniformity is established analytically (chunking a uniformly random permutation
into fixed-size blocks assigns equal probability to every assignment, since each
arises from exactly ∏ᵢ (sizeᵢ!) permutations) and guarded empirically by four
χ² tests over 30,000 draws each (α = 10⁻³): pairing uniformity, ragged-block
membership, position uniformity per criterion, and criterion × position
independence via a contingency test. These tests do not establish uniformity, which
is proven; they detect implementation faults that would violate the proof.

Ragged partitions are retained rather than avoided. K = 5 at N = 2 yields blocks of
(2, 2, 1), so the nominal N = 2 condition contributes an observation at group size
1. Analysis keys on *actual* group size, which preserves all data.

### 4.2 Design balance

A flat replicate allocation is unbalanced in the quantity of interest. Because
ragged partitions generate small blocks as a by-product of several nominal levels
while the largest block arises from exactly one, criterion-observations per group
size at K = 5 are {1: 7, 2: 6, 3: 3, 4: 4, 5: 5}, an imbalance ratio of 2.33, with
the least data in the conditions of primary interest. Replicates are therefore
allocated per nominal level by non-negative least squares to equalise
observations; the resulting allocation {1: 1, 2: 1, 3: 3, 4: 2, 5: 2} yields
{1: 8, 2: 10, 3: 9, 4: 8, 5: 10}, an imbalance ratio of 1.25, at a 54% increase in
calls.

### 4.3 Controls

**Output budget.** The principal threat to validity is that a call scoring five
criteria must emit five scores and five rationales under the same token ceiling as
a call scoring one, producing degradation for a purely mechanical reason.
Per-criterion output budget is therefore held constant: content budget =
40 + 220 × `group_size`, with a fixed per-call reasoning allowance that does not
scale with criterion count. The invariance is pinned by test. Truncation raises an
error rather than returning a partial payload, since a truncated response at large
group sizes is exactly the artefact the control exists to prevent.

Proximity to the ceiling is computed on *content* tokens, obtained by subtracting
reasoning tokens from reported completion tokens. Providers report completion
tokens inclusive of reasoning while the content budget excludes the reasoning
allowance; comparing the two directly overstated near-ceiling calls as 25.9% of
`group_size` = 1 calls against a corrected 0.4%.

**Acknowledged and uncontrolled.** A five-criterion prompt is genuinely longer than
a one-criterion prompt. Schema token count is recorded as a covariate. The study is
therefore not a pure test of divided attention and is not described as one.

### 4.4 Screening and calibration

Two pre-run gates, each on a small sample:

- **Judge selection.** Candidate models are run at `group_size` = 1 only; the
  cheapest whose accuracy clears a pre-registered floor (0.62) is selected. A judge
  near the 0.5 baseline cannot exhibit degradation, so selecting on price alone
  risks a null for the wrong reason.
- **Severity calibration.** Corruption severity is set so that detection at
  `group_size` = 1 falls in 0.70–0.90: above the floor so an effect is detectable,
  below the ceiling so decline has room to appear.

### 4.5 Execution

Scoring is pointwise: one call scores one response on one block of criteria. The
comparison is formed in analysis. Pointwise elicitation is used because it matches
the protocol under which the human labels were produced (annotators scored
responses independently), matches production usage, and avoids entangling
`group_size` with the number of candidates held in context.

Calls execute concurrently (default 8 workers) behind a write lock on an
append-only JSONL log keyed by
`(item_id, side, model, replicate, partition_hash, group_index)`. Resumption skips
completed identities. Concurrency does not affect the statistics (calls are
independent and analysis keys on identity rather than log order) and is verified
equivalent to serial execution by test.

## 5. Measures

### 5.1 Primary outcome

**Human stratum.** Per-criterion ordinal agreement: whether the judge orders the
two texts on that criterion as annotators do. Binary, with a 0.5 chance baseline.
Observations where the two responses share an annotator score are excluded, as no
correct ordering exists; the exclusion rate is reported per criterion.

**Synthetic stratum.** Targeted corruption recovery: whether the judge orders the
corrupted and clean texts so that the *targeted* score moves in the operation's
known direction. Ties and wrong-direction decisions are misses, not dropped.
This is an operation-level estimand; it is not a claim about criterion-isolated
human-quality accuracy.

### 5.2 Co-primary outcome: judge indecision

Where the judge assigns equal scores it has expressed no preference. Such
observations are excluded from the accuracy outcome, but the **tie rate is modelled
as a co-primary outcome under the same design**. This is necessary rather than
supplementary: were the tie rate to rise with `group_size`, conditioning accuracy
on resolved observations would condition on a post-treatment variable, and the
surviving comparisons at high `group_size` would be a more strongly selected, easier
subset. Reporting accuracy without its resolution curve would then be a collider
artefact.

A tie-tolerant analysis assigning 0.5 to judge ties, which conditions on nothing
post-treatment, is fitted as a robustness check. Disagreement in sign between the
two withdraws the resolved-only result.

In the `synthetic` stratum this outcome inverts for non-targeted criteria: a tie on
a criterion that was not corrupted is the **correct** response.

### 5.3 Specificity and leakage (diagnostic)

On a base corrupted for criterion C, criteria D ≠ C were not edited by the
operation. Their scores *should* not move if C were a surgically isolated
construct. Aggregating judge score deltas over (operation × criterion) yields a
leakage matrix and a specificity rate (unchanged companions).

These quantities are **diagnostics of construct overlap**, not acceptance gates.
Many honest corruptions necessarily disturb neighbouring dimensions (deleting an
answer sentence also hurts coherence). Under the operation-recovery estimand,
such leakage contextualises what the operation measures; it does not remove the
operation from the primary recovery analysis. Post-hoc exclusion on observed
leakage would select on the outcome and is not done.

### 5.4 Secondary and cost outcomes

Quadratic weighted κ against the absolute label; mean absolute error. Input,
output, and reasoning tokens per criterion scored, with cost computed at published
list prices and reported both cached and uncached; cache engagement is verified from
the provider's reported cached-token count rather than assumed.

**The cost saving is reported by component, because prompt caching erodes one
component and not the other.** Solving the pilot's per-call input tokens for
structure gives `input = 585 + 206 * K`: a 585-token prefix shared across the
unpacked calls for one item, and 206 tokens of rubric per criterion. The prefix is
a genuine prefix, since the item precedes the rubric in the rendered prompt
(`prompt.py:66-78`), so it is cacheable in principle.

Per 5 criteria, from `runs/pilot`:

| | K = 1, five calls | K = 5, one call | Saved |
|---|---|---|---|
| Input tokens | 3,954 | 1,615 | 2,339 |
| Output tokens | 989 | 592 | 396 |
| of which reasoning | 409 | 146 | 263 |
| of which content | 580 | 447 | 133 |

Cost per 10³ criteria and the composition of the saving:

| Scenario | K = 1 | K = 5 | Saving | Input share | Output share |
|---|---|---|---|---|---|
| Uncached | $0.593 | $0.318 | 46.4% | 42% | 58% |
| Ideal input caching | $0.488 | $0.318 | 34.9% | 7% | **93%** |

Output tokens are never cached and cost 8x input on `gpt-5-mini`, so under caching
almost the whole saving is the output side, and two thirds of that is reasoning:
82 reasoning tokens per criterion at K = 1 against 29 at K = 5. The durable claim
is therefore that the judge reasons once for five criteria rather than five times,
not that the input is read once rather than five times.

Longer inputs make the uncached saving larger and the cached saving slightly
smaller, because caching neutralises exactly the component that scales with input
length: at a 16,000-token prefix the uncached saving is 75.5% and the cached
saving 30.5%.

**On this corpus the cache cannot engage.** The 585-token prefix is below the
roughly 1,024 tokens the major providers require (§`cost.py:8-11`), and the pilot
measured 94,848 cached of 3,340,677 input tokens, 2.84%. The uncached column is
therefore the correct one to report for this run, and the cached column is a
counterfactual for deployments with longer inputs.

UNVERIFIED as external claims: the 10% cached-input multiplier (`cost.py:64-68`)
and the ~1,024-token prefix threshold are the repo's assumptions and have not been
checked against current published provider pricing. The 34.9% figure moves if the
real discount differs.

## 6. Statistical analysis

### 6.1 Estimators

Three estimators are reported together, as they fail differently.

**Primary: within-item paired contrast.** Per-item accuracy is averaged across
that item's criteria and replicates, then differenced across items against a
`group_size` = 1 reference. Averaging to the item before differencing is required:
multiple observations on one item are not independent draws, so pooling them would
be inconsistent (Miller, 2024). Differencing removes item difficulty exactly rather
than modelling it, and supplies the variance reduction described in §6.3. Reported
with Holm correction across the group-size ladder.

**Secondary: cluster-robust logistic regression**, clustered on item:
`correct ~ C(group_size) + position`.

**Secondary: crossed mixed-effects logistic regression**:
`correct ~ group_size + position + (1|item) + (1|criterion)`, with crossed random
intercepts because items and criteria differ in difficulty. Fitted variationally;
posterior standard deviations are approximate and it is never the sole estimator
reported.

**Naive standard errors are not reported.** Observations cluster by item (K criteria
× several group sizes each), and clustered standard errors have been measured at
over three times naive ones on comparable evaluation data (Miller, 2024).

### 6.2 Functional form

`group_size` is fitted both as a factor (no shape assumption; locates a
discontinuity) and as a linear term (single interpretable slope; greater power). A
likelihood-ratio test selects which is reported as primary; both are reported. The
LRT is a shape diagnostic and does not itself account for clustering.

### 6.3 Variance reduction

| Technique | Used | Rationale |
|---|---|---|
| Within-item paired differences | Yes, primary | ≈⅓ variance reduction at ρ = 0.5 (Miller, 2024) |
| Replication, K per cell | Yes | Var(sᵢ) = σᵢ²/K; item-level means computed before standard errors |
| Next-token probabilities | No | Requires single-token answers; incompatible with reasoning and structured output |
| Reduced temperature | No | Can increase minimum variance and inject bias (Miller, 2024) |

### 6.4 Interaction

`group_size × criterion` is tested, with Holm correction across the K
criterion-specific tests. A finding that some criteria degrade under packing while
others do not implies a different recommendation (consolidate robust criteria,
isolate fragile ones) than a uniform effect.

### 6.5 Power

Sample size follows Miller (2024):

n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²

with the paired form n = (z_{α/2} + z_β)² σ²_d / δ² used for the primary
estimator, σ_d being the observed standard deviation of per-item differences. The
implementation reproduces the published worked example of ~969 questions for
δ = 0.03 at 80% power, α = 0.05.

## 7. Reproducibility, pre-registration, and reporting

### 7.1 Pre-registration

Primary outcome, primary estimator, screening floor, tie rule, inclusion filter,
and the functional-form test are fixed in `PREREGISTRATION.md` before the
confirmatory run. Pilot analyses are exploratory and reported as such.

### 7.2 Reproducibility

All randomisation is seeded and every partition is recorded by content hash. The
analysis script is committed before the confirmatory run. Estimators are validated
against planted effects before contact with real data: a synthetic judge with a
specified degradation rate must be recovered (correct sign, Holm-significant,
correct functional form) and a judge with no planted effect must not produce a
significant curve. Both directions are tested, since an estimator suite never
checked against a known answer provides no evidence.

### 7.3 Reporting

Judge agreement is reported against the human κ ceiling per criterion, not against
1.0. Treating a noisy reference as a perfect gold standard is a recognised
reporting error in LLM-judge evaluation, and judge accuracy metrics are reported
alongside results so that downstream correction is possible.

Any sampling restriction or coverage limit is stated explicitly. Silent truncation
of coverage reads as complete coverage.

## 8. Results obtained under this protocol to date

### 8.1 Human stratum pilot

A pilot has been completed in the `human` stratum: 150 items, `group_size` ∈ {1, 5},
`gpt-5-mini`, 3,600 planned calls, 3,599 completed, one failure (0.03%, a truncation
at `group_size` = 1, not correlated with the treatment).

| | group_size = 1 | group_size = 5 |
|---|---|---|
| Accuracy | 0.8051 | 0.8039 |
| Judge tie rate | 0.4800 | 0.4796 |
| Cost per 10³ criteria | $0.593 | $0.318 |
| Input tokens / criterion | 791 | 323 |
| Reasoning tokens / criterion | 82 | 29 |

Paired contrast +0.0047, clustered SE 0.0194, 95% CI [−0.034, +0.043], p = 0.81,
n = 144 items. The crossed mixed model concurs (`group_size` −0.012, SD 0.027).
Quadratic weighted κ replicates the null independently on all observations rather
than the resolved subset: helpfulness 0.402 → 0.400, correctness 0.422 → 0.415,
verbosity 0.642 → 0.649, coherence 0.318 → 0.283.

The flat tie rate empirically discharges the selection concern of §5.2 for this
comparison: resolution does not vary with treatment, so conditioning on resolved
observations is not a collider here.

The synthetic stratum was subsequently calibrated (§8.2) and swept (§8.3).


### 8.2 Calibration gate: results

The gate (§4.4) was run on 40 synthetic items per operation at `group_size` = 1,
1,600 calls, $1.23. It changed the operation set substantially and exposed an error
in the classification rule.

**Detection rate conflates two failure modes.** Wrong-direction rates were
approximately zero across all twenty operations while tie rates reached 0.95. Every
low-detection operation was therefore *insensitive*, not *inaccurate*: the judge saw
no difference rather than judging backwards. An initial rule treating detection
below 0.60 as "near chance" would have excluded five operations that were
directionally perfect and merely needed more severity: chance would present as
~50% wrong-direction, and these presented 0–10%. Three quantities are now reported
separately:

- **detection rate** = hits / n
- **tie rate** = ties / n
- **discrimination** = hits / (hits + wrong), i.e. accuracy conditional on the judge
  committing to a direction

**Discrimination on few decided cases is noise.** `jargon` produced discrimination
of 0.25, 1.00, 0.00, 0.67, 0.67 across a severity ladder, an apparent direction
inversion computed on one to two decided cases per rung atop a 90% tie rate.
Discrimination is now suppressed below `MIN_DECIDED` = 8 decided cases, and an
operation that never reaches that threshold is classified `mute` rather than
`unreliable`.

**Severity is not monotone in validity, so a ladder is required.** Raising severity
on the five insensitive operations improved one and damaged three. Measured at
severities 1–16 on 30 items each:

| Operation | Ladder (detection / tie / discrimination) | Diagnosis |
|---|---|---|
| `truncate` | s1 .67/.33/1.00 → s8 .77/.23/1.00 → s16 .67/.27/.91 | **window at all rungs, best s8** |
| `drop_answer` | s1 .00/.90 → s4 .27/.67/.80 → s16 .67/.27/.91 | **window at s16**; monotone |
| `numbers` | s1 .30/.70/1.00 → s4 .57/.40/.94 → s16 .47/.53/1.00 | **capped**: direction sound, sensitivity bounded by the text |
| `vague` | s1 .13/.67/.40 → s8 .30/.57/.69 | **inverting**: direction reverses; removed |
| `jargon` | s1 .03/.87 → s30 .07/.88 | **mute**: 87–97% ties at every severity; excluded from measurement |

Three distinct no-window diagnoses, with different remedies:

- **mute**: the operation does not move the targeted score at any severity. No
  remedy; it does not measure its criterion.
- **capped**: directionally valid but severity is bounded by the text (a response
  containing three numerals cannot absorb eight numeral edits), so the tie rate
  cannot be driven down. Usable with reduced power.
- **inverting**: sufficiently sensitive, but discrimination collapses on adequate
  decided cases: heavier corruption genuinely reverses the asserted direction.
  Invalid; severity cannot repair a wrong direction.

**Operations removed.** `unsupport` (support): discrimination 0.50, four hits
against four wrong: a genuine accuracy failure, and stripping parentheticals removes
asides about as often as evidential support. `vague` (specificity): inverting.
`jargon` (complexity): mute; retained in the criterion pool as packing load but
excluded from measurement.

**Resulting design.** K = 20 criteria form the packing load; **17 operations are
measured**. Three criteria are load-only: complexity, specificity, support. Four of
the five anchor criteria retain valid operations (coherence, correctness,
helpfulness, verbosity); complexity is lost, so cross-stratum comparison narrows to
four criteria, the same narrowing §9.1a anticipated on independent grounds.

**The hypothesised detection classification largely failed.** Of twenty operations,
eight contradicted their hypothesised class; only 5 of 12 `surface` operations
reached ceiling, and `contradict`, hypothesised semantic and difficult, reached
0.93. As pre-committed in §3.4, measurement governs and the discrepancies are
reported rather than reconciled. Two consequences: the dilution risk that motivated
stratification is smaller than anticipated, and the surface/semantic framing carries
less weight than assigned to it.

Substantively, `jargon`'s muteness is a finding about the criterion rather than the
instrument: lexical jargon substitution does not move this judge's complexity score
at any severity tested, which suggests its notion of complexity tracks conceptual
depth rather than vocabulary.

### 8.3 Packing sweep: results

The synthetic-stratum packing sweep was run to completion on `gpt-5-mini` and is
analysed under the operation-recovery estimand (§10.0c). Every figure in this
section is recomputed from the retained per-call records; see §8.3.1 for the
artefacts and the reproduction check.

**Design as executed.** Reproduced exactly by `packing sweep --dry-run` against
`data/helpsteer2.jsonl` at 500 base items (verified 2026-09-05):

| | |
|---|---|
| Criterion pool | K = 20 |
| Measured operations | 17 |
| Items per operation | 40 |
| Corrupted base items | 680 |
| Nominal levels swept | 1, 2, 3, 5, 8, 12, 16, 20 |
| Replicates per level | 1 |
| Calls | 10,880 |
| Estimated input tokens | 26,440,075 |
| Cost ceiling at list price | $44.44 |

*Calls* per actual group size:

    {1: 1360, 2: 1496, 3: 1224, 4: 572, 5: 1360, 8: 1642, 12: 824, 16: 1042, 20: 1360}

These are call counts, not criterion-observations. The generating line is
`Counter(len(sp.group) for sp in specs)` (`cli.py:815`), one increment per
`CallSpec`, and the figures sum to exactly 10,880. They are not
criterion-observations, and the distinction matters because it understates the
true spread: criterion-observations are `calls x group_size` (27,200 at group size 20 against
2,288 at group size 4, a ratio near 12), while the *targeted* observations that
carry the primary recovery estimand are `calls / 2` (680 at group size 20).

**The sweep is not balanced, and unlike §4.2 it makes no attempt to be.** The
imbalance ratio is 2.87 (1,642 observations at group size 8 against 572 at group
size 4). The non-negative-least-squares replicate allocation described in §4.2 is
implemented in `design.py` and used for the human-stratum plan; the synthetic
sweep planner, `plan_targeted_blocks`, enumerates levels × replicates directly and
applies no such allocation. Group size 4 is thin because it arises only as a
ragged by-product of higher nominal levels, never as a nominal level itself. This
is a contributor to the width of the interval reported below and is a deviation
from §4.2 that was neither intended nor, until now, recorded.

**Outcomes.** Targeted directional recovery, K = 1 → K = 20:

| Detection class | K = 1 | K = 20 |
|---|---|---|
| semantic | 0.705 | 0.720 |
| surface | 0.788 | 0.683 |

Cost per 10³ criteria scored fell from approximately $0.61 to $0.22 uncached, at
list price.

**Neither K = 20 contrast survives Holm correction** across the group-size ladder.
The minimum detectable effect at this sample size is approximately 0.13–0.14, and
both observed movements, including the 10.5-point surface decline, fall inside
it. The sweep therefore bounds degradation rather than estimating it, and no
packing-degradation claim is made from it.

The surface decline is *hypothesised* to be positional rather than a load effect,
per the pre-registered rival H2 (§1). That hypothesis is **adjudicated in §8.3.2**,
which was possible only once the retained run artefacts were located (§8.3.1). The
outcome measures in the table above are position-marginal: they come from
`synthetic_recovery` and `paired_ladder`, neither of which enters `position`
(`analyze.py:206-235`, `analyze.py:504-570`). The position-adjusted picture is
materially different and is reported separately rather than folded in here.

#### 8.3.1 Provenance: retained artefacts and recomputed figures

`runs/` is gitignored (`.gitignore`: `runs/*`), so the sweep's artefacts are
absent from the *repository* while being present in the working checkout. Absent
from the repository and not retained are different claims, and any figure in §8.3
should be cited against the artefacts rather than against the repository listing.

| Artefact | Size | Contents |
|---|---|---|
| `runs/sweep/calls.jsonl` | 22.5 MB | 10,880 records, one per call, matching the design table exactly |
| `runs/sweep/items.jsonl` | 3.1 MB | corrupted item payloads |
| `runs/sweep/meta.json` | 1.7 KB | levels, `per_corruption`, `operation_roles`, `detection_classes`, calibration provenance |

Also retained: `runs/pilot`, `runs/calib`, `runs/calib2`, `runs/ladder`,
`runs/select-gpt-5-mini`, `runs/e2e`.

Every outcome figure in §8.3 is **recomputed** by `packing analyze --run sweep`,
and every one reproduces the figure frozen in the talk specification:

| Figure | Frozen figure | Recomputed |
|---|---|---|
| semantic recovery, K = 1 → 20 | 0.705 → 0.720 | 0.7050 → 0.7200 |
| surface recovery, K = 1 → 20 | 0.788 → 0.683 | 0.7875 → 0.6833 |
| MDE | ~0.13-0.14 | 0.1375 semantic, 0.1335 surface |
| Holm outcome | nothing survives | nothing survives (min `p_holm` = 0.357) |

**The MDE and its inputs.** `paired_contrast` collapses to one value per
independence unit and keeps only units present at *both* group sizes
(`analyze.py:530-537`), so the K = 1 against K = 20 contrast runs on far fewer
units than there are items.

| Class | n units | σ_d | MDE at n |
|---|---|---|---|
| semantic | 48 | 0.3400 | 0.1375 |
| surface | 32 | 0.2697 | 0.1335 |
| pooled | 53 | 0.3030 | 0.1166 |

The interval's width comes from those 32 to 48 effective units, not from the
2.87x call-count imbalance, which falls on the intermediate rungs and not on the
endpoints (group sizes 1 and 20 are both nominal levels and both carry their full
complement of items).

Items required to reach a target MDE, from the recomputed σ_d:

| Target MDE | semantic | surface |
|---|---|---|
| 0.100 | 91 | 58 |
| 0.075 | 162 | 102 |
| 0.050 | 364 | 229 |
| 0.030 | 1,009 | 635 |

**A re-run would not settle significance and is not the cheapest next step.** The
judge is nondeterministic, so a repetition yields fresh draws at the same n. Three
levers cost nothing: dropping the ragged by-product group sizes from the ladder,
pre-registering a single K = 1 against K = 20 rung instead of paying Holm across
eight (`analyze.py:566`), and one-sided testing of a directional hypothesis. Each
tightens the headline contrast without another call.

#### 8.3.2 Load and position, separated

The pre-registered rival H2 (§1) holds that an apparent packing effect is really
about where a criterion sits in the prompt. It is testable here because criterion
order within each call is randomised and each criterion's ordinal position is
recorded per observation (`partition.py:139-146`, `runner.py:199`).

**What the design does and does not identify.** Position is nested inside the
call: `0 ≤ position ≤ group_size − 1`, so `E[position | group_size = g] = (g−1)/2`
exactly, and the observed means run 0.00 at g = 1 to 9.19 at g = 20. Three
consequences, which need to be kept apart:

1. **No nonparametric separation exists.** Support is triangular. Position 19
   occurs only at g = 20, so "19th in the list" and "the g = 20 condition" are the
   same cell and no contrast in these data can distinguish them.
2. **The reference cell is degenerate.** g = 1 forces position 0
   (`partition.py:28`, pinned at `tests/test_partition.py:77-84`), so the baseline
   contributes nothing to the position curve.
3. **A linear-additive decomposition *is* identified**, because position varies
   within fixed g by randomisation. At g = 20 the 20 positions are populated with
   roughly 34 targeted observations each. This is assumption-based rather than
   design-based separation, and the assumptions (linearity in position, no
   g × position interaction) are not testable off triangular support.

Randomisation decorrelates position from criterion *identity*, which is what
`tests/test_partition.py:164-193` verifies. It cannot decorrelate position from
group size, and no claim that it does should be made anywhere in this repo.

**Within-K position slopes.** Fitting `correct_num ~ position` inside each fixed
group size holds load constant by construction. Inverse-variance pooled across
strata:

| Class | pooled within-K position slope | SE | p |
|---|---|---|---|
| semantic | −0.0496 | 0.0180 | 0.006 |
| surface | −0.0269 | 0.0172 | 0.118 |

**Joint models.** Entering both factors on the primary recovery outcome, by class:

| Class | model | group_size | position | AIC |
|---|---|---|---|---|
| semantic | load only | +0.0008 (p = 0.89) | | 1932.7 |
| semantic | position only | | −0.0220 (p = 0.093) | 1929.7 |
| semantic | load + position | +0.0260 (p = 0.008) | −0.0508 (p = 0.014) | **1927.7** |
| surface | load only | −0.0271 (p = 0.003) | | 2138.8 |
| surface | position only | | −0.0468 (p = 0.0002) | **2133.9** |
| surface | load + position | −0.0069 (p = 0.66) | −0.0395 (p = 0.067) | 2135.5 |

The `group_size × position` interaction is not significant in either class
(p = 0.755 semantic, p = 0.117 surface), so the additive form is the one reported.

**Two findings, opposite in character.**

*Surface operations: the decline does not survive position adjustment.* The load
slope is −0.0271 (p = 0.003) unadjusted and −0.0069 (p = 0.66) once position
enters, an attenuation of 74.7%, and position-only is the best-fitting model by
AIC. Decomposing the −0.1042 movement using the within-K position slope
attributes −0.0443 to the mean position shift of +9.19 and leaves −0.0599 as load
residual, so estimates of position's share range from ~43% (within-K slope) to
essentially all of it (adjusted regression). The spread reflects collinearity
between the two factors and should be read as a range, not a point estimate.

*Semantic operations: a flat total conceals two opposing effects.* Position hurts
(−0.0496, p = 0.006, within-K) while load *helps* once position is held fixed
(+0.0260, p = 0.008). The decomposition gives −0.1017 position-attributable
against +0.1167 load residual, nearly cancelling to the observed +0.0150. This is
a suppression pattern: the position penalty masks a genuine positive load effect.
Reporting the position-marginal 0.705 → 0.720 alone hides both.

**The surface decline is entirely an indecision effect.** Wrong-direction counts
at K = 20 are zero; the tie rate rises 0.2042 → 0.3167 and discrimination rises
0.9895 → 1.0000. The judge does not get the answer wrong more often under load, it
declines to commit more often, which is the §5.2 failure mode and the reason tie
rate is co-primary.

**Status and limits.** These fits are **exploratory**. They are not Holm-corrected
across the model comparisons above, and while H2 was pre-registered, this
particular decomposition was not pre-specified. Three further limits:

- The stratification variable is the *hypothesised* detection class, which
  `meta.json` carries unchanged from §3.4, and §8.2 measured 8 of 20 operations
  as contradicting it. Reclassifying by measured behaviour could strengthen or
  destroy the contrast.
- Detection class is a property of the operation, not the criterion (§9.1a), so
  none of this attaches to criteria.
- `position_competition` has no test coverage, and the mock judge used for
  estimator validation cannot express a position effect at all: its noise is a
  function of `group_size` only (`mock_provider.py:55-58`). The estimator suite is
  validated for recovering load and for not fabricating load, **never for
  distinguishing load from position**. That gap is the main reason these results
  are labelled exploratory rather than confirmatory.

**Does the detection-class split earn its keep?** Yes, and it was worth checking,
since splitting costs power (pooled MDE 0.1166 against 0.1375 and 0.1335 split).
Fitting `correct_num ~ group_size * is_surface + position` on the pooled primary
frame gives an interaction coefficient of −0.0270 (p = 0.023) and the best AIC of
the three specifications (4061.4, against 4064.7 for a class intercept alone and
4071.8 for no class term). The two classes move in opposite directions, so pooling
averages a positive and a negative effect toward a null, the dilution §10.0b
anticipated, now measured rather than assumed.

### 8.4 Planned extension: fixed-position load ladder

**Planned, not run.** No outcome-generating call has been made under this design.
Full protocol and pre-registration in
`docs/superpowers/specs/2026-09-08-load-ladder-design.md`.

The K = 20 sweep bounds degradation rather than estimating it, and §8.3.2 can
separate load from position only additively, off triangular support. The planned
extension removes the confound by construction instead: the target criterion is
pinned at position 0 and companion criteria pad the call to reach K, with lower
rungs using nested prefixes of one seeded companion sequence.

| Element | Value |
|---|---|
| Ladder | K in {1, 5, 10, 20, 40} |
| Confirmatory primary | equal-weighted average over eight operations, K = 1 against K = 40 |
| Core operation set | `flatten`, `truncate`, `shuffle`, `drop_answer`, `agreement`, `depunct`, `duplicate`, `filler` |
| Target n | 464 prompts, 58 per operation |
| Planning variance | `sigma_eff` 0.3288, cluster bootstrap CI [0.2651, 0.3693] |
| Key secondary | `numbers` correctness sentinel, n = 184, smallest effect of interest 0.10 |
| Budget | 6,480 calls, about $23.40 at list price, provisional |

The operation set is narrowed to eight because joint eligibility depends on which
operations are kept: all eleven test operations share only 8.2% of responses,
while the core eight reach 84.8% at response level and 93.5% at base-prompt
level. That makes a single common eligible population available, so equal quotas
are meaningful and operation contrasts have a causal reading within it. `numbers`
is retained as a separate arm for construct coverage, since it is the only
correctness-targeting operation; its eligible set is a strict subset of the core's.

The estimand is deliberately narrower than "the effect of load". With the target
fixed at position 0, K = 40 means 39 trailing items, so load remains coupled to
recency and distance from the end of context.

## 9. Threats to validity

1. **Detectability of synthetic corruption.** Injected defects may be more
   conspicuous than naturally occurring quality differences, inflating absolute
   accuracy and potentially understating a load effect through a ceiling. Severity
   calibration (§4.4) mitigates but does not eliminate this. The synthetic stratum
   bounds degradation rather than estimating it.

1a. **Construct validity: detection class is a property of the operation, not of
   the criterion.** This is the principal limitation of the synthetic stratum and
   it constrains every conclusion drawn from it.

   Each corruption targets a *detectable proxy* for its criterion rather than the
   criterion itself, and the same criterion admits operations of different classes.
   `verbosity` under human annotation is a graded perception of over-elaboration;
   under `duplicate` it is a length comparison resolvable by counting words. A
   judgement-engaging verbosity corruption (content-free elaboration at matched
   length) would place the same criterion in the semantic class. Equivalently,
   `coherence` scrambled at word rather than sentence level would become a surface
   operation.

   Two consequences:

   - **Conclusions attach to operations, not criteria.** A flat surface curve
     supports "packing does not degrade detection of mechanically identifiable
     defects." It does not support "packing does not degrade verbosity judgement."
     Criterion-level transfer holds only to the extent that an operation faithfully
     instantiates its criterion, which is asserted here rather than demonstrated.
   - **The cross-stratum calibration check narrows from five criteria to three.**
     Agreement between strata is informative only where the corruption engages the
     faculty the annotators exercised. That is arguable for coherence/`shuffle`,
     correctness/`numbers`, and helpfulness/`drop_answer`; it does not hold for
     complexity/`jargon` or verbosity/`duplicate`, where the two strata measure
     different things and agreement or disagreement is therefore uninformative
     about instrument validity.

   The stratum and detection-class axes are not themselves confounded (they cross,
   with 3 semantic and 2 surface operations among the human-labelled criteria
   against 5 and 10 among the synthetic-only ones) so the design does permit the
   two effects to be estimated separately. What it does not currently permit is
   separating the *operation* effect from the *construct* effect, because each
   criterion has exactly one operation.

   **Resolution requires a second operation of the opposite class for at least one
   anchor criterion**, so that criterion is measured both ways and the class effect
   can be attributed. This is not implemented and no claim separating the two is
   made without it.
2. **Leakage.** Corruption targeting is asserted by construction and verified only
   through the leakage matrix (§5.3), which measures the judge's response rather
   than the text's true multi-dimensional change.
3. **Population restriction.** The `human` stratum, where a minimum-scoreable
   filter is applied, covers a subpopulation with clearer label separation and
   plausibly easier judgements. Absolute accuracy is inflated relative to the full
   distribution; the shape of the curve transfers, its level does not.
   A looser-filter sensitivity analysis is required.
4. **Prompt length.** Larger group sizes have longer prompts (§4.3), so `group_size`
   is not a pure manipulation of divided attention.
5. **Single judge model and corpus.** Per-dataset variance in judge behaviour is
   large enough across published benchmark suites that per-use-case validation is
   recommended; results are not generalised beyond the model and corpus tested.
6. **Contamination.** The corpus is public and plausibly represented in training
   data. This inflates absolute accuracy but should not bias the `group_size`
   contrast, since every condition sees the same items.

## 10. Pre-specification status

What this protocol pre-registers and what it does not. Choices that are not
pre-specified are labelled, so nothing post-hoc reads as a priori.

0c. **Synthetic estimand is operation recovery.** The synthetic stratum's estimand
   is directional recovery of known corruption operations, which is the narrow
   claim it can carry: off-target movement is reported rather than treated as
   invalidating the synthetic gold, since demanding criterion-isolated human
   constructs from mechanical edits is a bar no mechanical edit meets. Leakage is
   reported and is not a gate. The sweep is **exploratory** with respect to any
   stronger claim that a corruption isolates a human quality dimension, and that
   claim is not asserted.

0. **K = 20 in the synthetic stratum.** The labelled corpus supports at most five
   criteria, and a null at K = 5 with no located breaking point is weak evidence.
   Corruption ground truth is not bound by the labelling ceiling, so fifteen
   synthetic-only criteria join the five anchors. The anchor criteria appear in
   both strata, which is what keeps them comparable. Judging is restricted to the
   partition block containing the targeted criterion, reducing cost from
   2·⌈K/g⌉ calls per base to 2 and making K = 20 tractable.

0a. **`vaguify` numeral rule.** Numerals are vaguified only as quantities (numeral
   followed by a unit or plural noun) or percentages, with a guard excluding
   numerals preceded by a capitalised token. Without the guard the rule matches any
   digit string and rewrites "Xbox 360" as "Xbox several", a factual error
   introduced while claiming to target specificity, which is precisely the
   mislabelling the leakage matrix exists to detect. The guard costs applicability,
   47.2% against 72.0% ungated, which is the intended trade.

0b. **Detection-class stratification.** See §3.4. Roughly twelve of the twenty
   operations are expected to be detectable without comprehension, and pooling
   those with the semantic operations would dilute a real effect toward the null.
   The stratification is **not** a primary axis: the K = 20 class contrast holds in
   only 8 of 11 leave-one-out fits (§8.3.2), and the planned extension (§8.4) does
   not stratify on it.

1. **Every scale point is anchored.** A partially anchored scale becomes the scale:
   the measured collapse to three points (§3.3) is what the full anchoring exists
   to prevent. Any criterion with an undescribed scale point is rejected at
   construction.

2. **Near-ceiling computation excludes reasoning tokens** (§4.3). Both quantities
   are logged, so every run is analysable under the same definition.

3. **Severity is not swept.** It is a nuisance parameter, and sweeping it answers a
   question other than the one posed. Severity is fixed per operation at its
   calibrated value.

4. **No placement arms.** Head, random and tail placement arms are not used;
   `position` is a modelled covariate instead. The residual argument for them
   concerns common support, which the planned extension addresses by construction
   rather than by arm (§8.4).

5. **Unlabelled padding carries no ground truth.** Synthetic corruption supplies a
   verifiable label for the targeted criterion, which unlabelled filler cannot. The
   planned extension (§8.4) uses companion criteria as load only, with no
   ground-truth role, which is the one form in which padding is admissible.

6. **K = 5 in the human stratum.** No public corpus provides more than about five
   fixed, fully-labelled criteria per item. FLASK defines twelve skills but
   annotates only the three most relevant per instance; BiGGen Bench uses
   instance-specific rubrics, so criteria differ across items and cannot be packed.
   K = 5 is the labelled ceiling and covers the practical range for stacked quality
   dimensions.

7. **H2 is adjudicated additively, and the adjudication is exploratory.** Position
   varies within fixed group size by randomisation, so a linear-additive
   decomposition is identified and is reported in §8.3.2. What the design cannot do
   is separate load from position nonparametrically, because support is triangular.
   The decomposition is labelled exploratory because the estimator suite has not
   been validated against a planted *position* effect: the mock judge's noise is a
   function of `group_size` alone (`mock_provider.py:55-58`), and
   `position_competition` has no test coverage.

8. **Artefact retention.** `runs/` is excluded by `.gitignore`, so a run directory
   is archived outside the ignore rule before analysis begins. A figure that cannot
   be recomputed from retained per-call records is reported as unavailable rather
   than as null.

## References

- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations.* arXiv:2411.00640.
- *How to Correctly Report LLM-as-a-Judge Evaluations.* arXiv:2511.21140.
- Wang, Z. et al. (2024). *HelpSteer2: Open-source dataset for training
  top-performing reward models.* arXiv:2406.08673.
- Bavaresco, A. et al. (2025). *LLMs instead of Human Judges? A Large Scale
  Empirical Study across 20 NLP Evaluation Tasks.* ACL. arXiv:2406.18403.
- Ye, S. et al. (2023). *FLASK: Fine-grained Language Model Evaluation based on
  Alignment Skill Sets.* arXiv:2307.10928.
- Kim, S. et al. (2025). *The BiGGen Bench: A Principled Benchmark for
  Fine-grained Evaluation of Language Models with Language Models.* NAACL.
- Fabbri, A. R. et al. (2021). *SummEval: Re-evaluating Summarization Evaluation.*
  TACL.
