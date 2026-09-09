# Fixed-position load ladder: design

Status: **executed**. Results in METHODS §8.5. This document was frozen before
the first call and is the pre-registration for that result.

## 1. Question

Does appending 39 criteria after a first-position target change directional
recovery at K = 40 relative to K = 1, and what descriptive pattern appears at
K = 5, 10 and 20?

The completed K = 20 sweep (METHODS §8.3) confounded load with target position.
This design fixes absolute target position at zero while varying trailing load. It
does not separately identify load, recency, and distance from the end of the
context, and it makes no inferential claim about a threshold or a shape.

## 2. Inferential hierarchy

Exactly one confirmatory hypothesis, so no multiplicity adjustment applies to it.
Everything below it is planned but not confirmatory.

| Tier | Content | Testing |
|---|---|---|
| **Confirmatory primary** | Eight-operation core contrast, K = 1 against K = 40 | Two-sided, alpha = 0.05 |
| **Key secondary** | `numbers` correctness sentinel | Estimate and 95% CI, sized around a 0.10 effect |
| **Diagnostic secondary** | Tie rate by K | Reported, no formal testing rule |
| **Exploratory** | Intermediate rungs, operation-specific effects, `rename`, `scramble_list` | Unadjusted, no confirmatory claims |

If `numbers` were ever promoted to a second confirmatory hypothesis, the family
would need a multiplicity procedure: at two-sided alpha = 0.025 the requirements
rise to 522 core prompts at `sigma_eff` = 0.3693 and 176 `numbers` prompts at
`sigma_d` = 0.4261. This design does not promote it.

## 3. Primary estimand

For operation `o` and prompt `i`, let `d_i = Y_i,40 - Y_i,1`, where `Y` is
directional recovery of the seeded corruption. The primary estimand is the
equally weighted average of the eight operation-specific mean differences:

    theta_core = (1/8) * SUM_o E[ d_i | O_i = o, A_core = 1 ]

`A_core = 1` denotes prompts eligible for all eight core operations.

Stated in one sentence: **among prompts eligible for all eight high-applicability
operations, what is the equal-weighted average change in recovery from K = 1 to
K = 40?**

### What this estimand does not identify

The target criterion is fixed at absolute position 0. At K = 1 it is both first
and last; at K = 40 it is first with 39 items trailing it. The estimand is
therefore the effect of appending 39 additional items after a first-position
target. It does **not** separately identify load, recency, and distance from the
end of the context. That is a narrower claim than "the effect of load" and is
reported as such.

## 4. Operation set

**Core set, eight operations:** `flatten`, `truncate`, `shuffle`, `drop_answer`,
`agreement`, `depunct`, `duplicate`, `filler`.

This is the *high-applicability operation set*, not the complete catalogue. The
set is chosen so that a single common eligible population exists, which is what
makes equal quotas meaningful and gives operation contrasts a causal reading
within that population.

Applicability measured over all 400 local responses, reproducing METHODS §3.4:

| Operation | Response-level applicability | Role |
|---|---|---|
| `duplicate` | 100.0% | core |
| `filler` | 100.0% | core |
| `depunct` | 99.5% | core |
| `agreement` | 99.0% | core |
| `shuffle` | 95.5% | core |
| `drop_answer` | 95.5% | core |
| `truncate` | 91.0% | core |
| `flatten` | 85.8% | core |
| `rename` | 62.5% | excluded |
| `scramble_list` | 43.0% | excluded |
| `numbers` | 24.2% | key secondary arm |

Joint eligibility is a function of the set, not a fixed constraint:

| Operations kept | Joint eligibility, response level |
|---|---|
| all 11 | 8.2% |
| drop `numbers` | 33.0% |
| drop `numbers`, `scramble_list` | 61.0% |
| **core 8** | **84.8%** |

**Rejected alternative.** An eleven-operation primary average is a valid estimand
but a compound one: it averages eleven effects measured on eleven different
prompt populations, and cross-operation differences stay inseparable from
population differences. At 8.2% joint eligibility the common-population version
is not viable, since it would select for long, list-bearing, number-bearing
responses.

## 5. Sampling and the experimental unit

**The experimental unit is one base prompt with one selected response.** The
selection procedure is fixed in advance so that the choice between two qualifying
responses is never unrecorded:

> Sample one base prompt. Evaluate arm eligibility separately for both responses.
> If exactly one response is eligible, select it. If both are eligible, select one
> uniformly at random using the study seed. Freeze that response before operation
> assignment. A base prompt contributes to at most one arm.

Any higher level shared source, template, or document that induces dependence is
instead treated as the independent unit. Clustering is on the base prompt, which
the retained sweep's cluster collapse (440 corrupted items onto 53 clusters)
makes unavoidable.

**Base-prompt eligibility**, counting a prompt eligible when either response
qualifies, measured on the 200 pilot prompts:

| Arm | Base-prompt eligibility |
|---|---|
| core 8 | **93.5%** (187 / 200) |
| `numbers` | **29.0%** (58 / 200) |

Four-cell overlap in the pilot:

| Cell | Share |
|---|---|
| core only | 64.5% |
| both | 29.0% |
| `numbers` only | 0.0% |
| neither | 6.5% |

In the 200-prompt pilot, every `numbers`-eligible base prompt was also
core-eligible. The operation predicates do not logically imply that, so it is an
empirical observation rather than a structural guarantee. The implementation
allows and records a `numbers`-only cell.

### Arm assignment must precede eligibility screening

`P(numbers-eligible | core-eligible)` is 31.0% in the pilot, so a representative
core arm of 464 contains about 144 `numbers`-eligible prompts. Screening one
shared frame and then carving the `numbers` quota out of it destroys that: a frame
of 715 yields about 207 `numbers`-eligible and 669 core-eligible prompts, and
removing 184 for the `numbers` arm leaves a core pool that is only **4.8%**
`numbers`-eligible against a target of 31.0%. The primary estimate would then
apply to "core-eligible prompts not allocated to `numbers`", not to the population
in §3.

The design therefore assigns arms **before** eligibility is inspected:

> Base prompts are assigned to independent core and `numbers` sampling streams
> using a frozen seed before eligibility is inspected. Each stream is screened only
> for its own arm's eligibility. The streams are extended in that same
> deterministic order if the blinded variance review raises the core target.

**Lane sizes are exact, not probabilistic.** The fetched frame is permuted by the
frozen seed and split by index, so the core lane holds exactly its candidate count
and the `numbers` lane exactly its own. A hash-threshold assignment would make
lane membership an independent draw per prompt, adding lane-count variance the
fill calculations below do not model.

**Within-lane order is seeded, not file order.** Each lane is ordered by
`partition_seed(f"{item_id}:lane-order", ...)` before screening, so any structure
in the source file's ordering (collection, topic, score) cannot select the sample.

**The selected response side is frozen at screening.** Eligibility is evaluated
against the full arm operation set, and the side that qualified is persisted with
the unit. Re-deriving the side later from the assigned operation alone would pick
a response eligible for that operation but not for all eight, silently changing
the population from `A_core = 1`.

**Operation quotas are preassigned to the maximum.** The 768-unit core maximum is
dealt into eight operation queues of 96 units each, and the initial run activates
the first 58 of each. Raising the target reveals further units from each queue
without altering any assignment already collected, which is what the increase-only
variance review requires.

Lane sizes, binomial, at 97.5% probability of filling each quota:

| Lane | Quota | Rate | Candidates |
|---|---|---|---|
| core, initial | 464 | 93.5% | **508** |
| core, adaptive maximum | 768 | 93.5% | **837** |
| `numbers` | 184 | 29.0% | **714** |

Initial total **1,222** base prompts; **1,551** if the core target rises to its
maximum. The streams are one physical fetch, permuted and split by the frozen seed
before any eligibility call.

Screening counts, using base-prompt rates throughout:

| Arm | Needed | Rate | 95% fill | 97.5% fill |
|---|---|---|---|---|
| core | 464 | 93.5% | 506 | 508 |
| `numbers` | 184 | 29.0% | 701 | 714 |

HelpSteer2 supplies 10,681 prompts, so the frame is not binding. The 200 prompts
on disk are a pilot frame; they would cap the core arm at 187 eligible, giving an
MDE of 0.068 at the point estimate of `sigma_eff` and 0.077 at the upper bound of
its CI.

## 6. Design

**Assignment.** Each eligible prompt is assigned exactly one operation and one
corruption realisation. Assignment is randomised subject to equal operation
quotas, before any outcome is generated. Eligibility is determined from the source
response alone, with no reference to evaluation outcomes.

One realisation per prompt, not merely one operation category: where an operation
can generate several candidate corruptions, exactly one is drawn and frozen, so
implicit weighting cannot return at the realisation level.

**Position.** The target criterion is pinned at position 0 in every call.
Companion criteria pad the call to reach K.

**Ladder, nested by construction.** K in {1, 5, 10, 20, 40}.

> For each sampled prompt, draw one seeded permutation of the 39 non-target
> criteria. The K = 5, 10, 20 and 40 conditions use prefixes of length 4, 9, 19
> and 39 from that same sequence. Companion identities and order are therefore
> nested across rungs and frozen for both sides of the paired evaluation.

Without nesting, K would change both the amount of load and which particular
criteria are present, putting composition noise into the endpoint contrast.

**Held constant across rungs.** The corruption realisation, the model and its
settings, the output-budget rule (`40 + 220 * K` content plus a fixed 512-token
reasoning headroom, METHODS §4.3), the target's position, and the companion
sequence. Only the number of trailing companions varies. At K = 40 the content
budget is 8,840 tokens and the request ceiling 9,352, against a predicted 3,327
tokens of actual output, so the ceiling is not expected to bind. Truncation is
nonetheless handled as a model-output failure under §10.

**Criterion pool.** K = 40 requires 20 new synthetic-only criteria beyond the
current 20. Every scale point of every new criterion must be anchored; partial
anchoring collapses the scale to its anchored points (METHODS §3.3).

## 7. Primary analysis

Compute `d_i = Y_i,40 - Y_i,1` per prompt. Estimate each operation's mean, then
average the eight means equally:

    theta_hat_core = (1/8) * SUM_o dbar_o

    Var_hat(theta_hat_core) = (1/64) * SUM_o (s_o^2 / n_o)

Equal weighting is preserved when quotas land unevenly or a few outcomes go
missing, which an unqualified mean over all `d_i` would not do. Equal quotas
define the weights rather than removing the need to declare them.

The estimator is a stratified heteroskedastic average, not a simple paired mean,
so the test statistic and its reference distribution are:

    T = theta_hat_core / sqrt( (1/64) * SUM_o s_o^2 / n_o )

    nu = ( SUM_o s_o^2/n_o )^2 / SUM_o [ (s_o^2/n_o)^2 / (n_o - 1) ]

with `nu` the Welch-Satterthwaite degrees of freedom. Two-sided, alpha = 0.05.

## 8. Power

Under fixed quotas the relevant planning variance is the average within-operation
variance, not the variance pooled across all differences:

    sigma_eff = sqrt( mean_o( Var(d | O = o) ) )

Estimated on retained-sweep observations belonging to prompts independently
verified as eligible for all eight core operations, which is the population the
estimand names (23 clusters):

| Quantity | Value |
|---|---|
| `sigma_eff` | **0.3345** |
| cluster bootstrap 95% CI | [0.2709, 0.3763] |
| pooled SD across all 11 operations, all prompts | 0.4238 |

Restricting to `A_core = 1` rather than merely to the eight operations moves the
estimate from 0.3288 to 0.3345 and widens the interval slightly. The restricted
figure is the one used, because `Var(d | O = o, A_core = 1)` is the quantity the
primary analysis operates on.

Within-operation SDs of `d`: `agreement` 0.154, `depunct` 0.222, `drop_answer`
0.276, `filler` 0.255, `shuffle` 0.255, `truncate` 0.375, `duplicate` 0.417,
`flatten` 0.520.

Sample sizes below come from a **noncentral-t planning approximation for the
stratified contrast**, not an exact paired-t analysis. At these sample sizes the
Satterthwaite correction moves the result negligibly, but the label should match
the estimator in §7.

Required analyzable prompts at a 0.05 effect:

| `sigma_eff` | n | Equal-quota rounding |
|---|---|---|
| 0.2709 (CI low) | 233 | 240 |
| 0.3345 (point) | 354 | 360 |
| 0.3763 (CI high) | 447 | **448** |

**Chosen target: 464 analyzable prompts, 58 per operation.** The upper bootstrap
bound implies a minimum equal-quota target of 448 prompts under the planning
approximation. 464 is a pre-specified 3.6% buffer above that minimum.

Power at a true effect of 0.05:

| n | At `sigma_eff` = 0.3345 (point) | At 0.3763 (CI high) |
|---|---|---|
| 360 | 80.7% | 71.0% |
| 448 | 88.4% | 80.1% |
| **464** | **89.5%** | **81.5%** |
| 576 | 94.8% | 88.9% |
| 768 (adaptive max) | 98.5% | 95.7% |

**Two documented assumptions.** `sigma_eff` is measured on the K = 1 against
K = 20 contrast, while the planned endpoint is K = 40; `SD(Y_40 - Y_1)` is not
directly estimated and could be larger or smaller. And it rests on 23 eligible
clusters.

### Effect-blinded variance review

One interim review, fully specified in advance:

> After 160 complete core prompts, 20 per operation, the study performs one
> effect-blinded internal variance review. Only within-operation centred residuals
> are used; operation means and the primary test statistic are not inspected. The
> required total N is recomputed from the §7 variance formula, rounded upward to a
> multiple of eight, and **may increase but not decrease** from 464. A maximum N of
> 768 is fixed before collection. First-stage prompts remain in the final analysis.

Before this rule is used, the entire re-estimation procedure is simulated under
the null from the pilot distributions to confirm the final test retains
approximately 5% type-I error. The outcome is bounded, so the usual mean and
variance independence assumptions do not hold exactly and the check is not
optional. A fixed N = 464 with no interim adjustment is the fallback if the
simulation shows material inflation.

No power claim is made for the intermediate rungs, for operation-specific
effects, or for any shape or threshold hypothesis. The closed form above applies
to the two-level contrast only; a confirmatory shape claim would require
simulating the exact planned model and is not attempted.

## 9. Key secondary: the `numbers` correctness sentinel

`numbers` is the only correctness-targeting operation, and at 29.0% base-prompt
eligibility it cannot join a common-population core. It is retained as a
pre-registered **key secondary** arm, not a second confirmatory primary:

    theta_numbers = E[ Y_40 - Y_1 | prompt eligible for numbers ]

Disjoint sample of number-eligible prompts, one fixed `numbers` realisation each,
same nested ladder. Reported as an estimate with a 95% confidence interval.

> A change of 0.10 on the directional-recovery scale is the pre-specified smallest
> effect of interest for the `numbers` arm. On the pilot baseline of 0.375 that is
> a 26.7% relative change in recovery.

Sizing, with the same upper-bound discipline applied to the core:

| Quantity | Value |
|---|---|
| `sigma_d` point estimate | 0.4261 |
| clusters behind it | **26** |
| bootstrap 95% CI | [0.3144, 0.4808] |
| n at the point estimate | 145 |
| n at the CI upper bound | **184** (80.1% power at that bound) |
| power of n = 145 at the CI upper bound | 70.1% |
| candidates to fill 184 at 29.0% | 701 for 95%, **714** for 97.5% |

**Target: 184 prompts.** Sizing this arm from a point estimate while the core is
sized from an upper bound would be inconsistent; 145 would deliver only 70.1%
power if the variance sits at the top of its interval.

Its estimate is **not** interpreted as directly comparable with the core
operations, because the populations differ.

The reason to retain it is construct coverage, not headroom. At baseline recovery
0.375 it has 0.375 of room to decline and 0.625 to rise; describing that as a
headroom advantage for detecting degradation would be backwards.

**Framework limit.** A nonsignificant two-sided test here does **not** establish
that correctness degradation of practical importance is absent. If that is the
conclusion wanted, it requires a pre-specified non-inferiority test with margin
-0.10, which this design does not currently specify.

## 10. Failure handling and the unevaluable-output rule

Two distinct categories, because conflating them biases the result. Model-output
failures may become **more** common at K = 40, which makes them part of the load
effect; dropping and replacing them conditions the analysis on surviving high load
and biases degradation downward.

> **1. Transient infrastructure.** Connection errors, rate limits, provider 5xx,
> temporary outage, connection-establishment timeouts. Three transport retries
> (`max_retries=3`, `openai_provider.py:56`). On persistent failure the prompt is
> replaced from the next candidate in its own lane, in frozen order. These are
> **not** scored and **not** treated as completed identities on resume.
>
> **2. Treatment or system unevaluable.** Truncation, malformed returned content,
> content filtering, refusal, context-limit rejection, and read or generation
> timeouts. **Retained and scored** under the composite-outcome rule below,
> because any of these may be caused by K = 40 itself, and replacing them would
> erase part of the treatment effect.
>
> **3. Fatal protocol or configuration error.** Unsupported parameter, invalid
> schema emitted by the client, criterion-key mismatch, corrupt input, unexpected
> exception, or any invariant violation. **The run aborts.** These are never
> scored.

The third class exists because a two-class scheme scores its own bugs as science:
an SDK change or an unsupported model option would otherwise arrive as evidence
that load degraded judging. An unrecognised failure is fatal, not unevaluable, so
the burden is on the harness to classify rather than on the estimate to absorb.

Timeouts are split by phase where the provider discloses one: a read or
generation timeout is class 2, since those name a phase where generation time
plausibly rises with K.

A timeout of **undisclosed** phase, which is what the SDK raises after exhausting
its own retries, is class 1. That is settled by measurement rather than argument.
Observed latency on this ladder has a rung-independent tail: at K = 1, where the
median call takes 3.2 seconds, 6 of 386 calls still exceeded 90 seconds and one
reached 124.8. A call spending two minutes on a single criterion is not
generating for two minutes, so the tail is transport.

The consequence of misclassifying is asymmetric. As class 2 a timeout becomes a
recovery miss, and at the observed rate of roughly 5% of calls that injects false
failures into the outcome. As class 1 it costs sample through replacement but
leaves the estimate uncorrupted.

**Resume semantics.** An identity counts as complete only when it has a success or
a class-2 record. Class-1 failures are written to a separate attempts log so a
transient error cannot be mistaken for a finished call.

### The outcome is a composite, not an imputation

`Y` is defined as **demonstrated, machine-readable directional recovery**: the
judge moved the targeted score in the seeded direction *and* returned output the
harness could read. Scoring an unreadable judgment as 0 is therefore part of the
outcome definition, not a missing-data convention, and it is pre-registered as
such. A judge that stops returning usable output under load has failed to
demonstrate recovery, and because unevaluable output may itself rise with K, this
keeps that effect inside the estimate rather than outside it.

**Scoring happens at the paired unit, not the call.** Recovery is a comparison
between the clean and corrupted sides, so a call-level rule would create two
observations for one prompt when both sides fail. Exactly one outcome is emitted
per (prompt, rung):

| Case | Outcome |
|---|---|
| Both sides evaluable | Directional recovery as normal |
| Either side unevaluable | One unevaluable outcome |
| Both sides unevaluable | One unevaluable outcome |

Within a side, a malformed payload whose targeted score is unambiguously
extractable is used, with a parse-repair event logged; a score readable more than
one way is unevaluable.

### Sensitivity bounds

The estimand is a difference, `d = Y_40 - Y_1`, so bounding it requires opposite
imputations at the two endpoints. Assigning every unevaluable `Y` the same value
does not bound `d`. Three pre-specified analyses:

| Analysis | Unevaluable at K = 1 | Unevaluable at K = 40 |
|---|---|---|
| **Primary** | 0 | 0 |
| **Lower bound on d** | 1 | 0 |
| **Upper bound on d** | 0 | 1 |
| **Complete case** | pair excluded | pair excluded |

If the primary estimate and the two bounds do not agree in sign, the finding is
reported as sensitive to the outcome definition rather than as a load effect.

**Unevaluable frequency by K is itself a reported diagnostic outcome**, since a
judge that stops returning usable output under load is degrading in a way
recovery rate alone will not show.

## 11. Other secondary and exploratory analyses

All unadjusted, none supporting confirmatory claims:

- **Tie rate by K.** Diagnostic secondary. Reported alongside recovery because
  the K = 20 surface decline proved to be entirely a tie increase with
  wrong-direction at zero (§8.3.2). No formal testing rule and no power
  calculation, so it is not described as co-primary.
- **Intermediate rungs** K = 5, 10, 20, reported descriptively. They may not be
  used to claim monotonicity, a threshold, or curvature.
- **Operation-specific effects.** At 58 prompts per operation the per-operation
  MDE varies with each operation's own variance, from 0.057 (`agreement`) to
  0.191 (`flatten`), averaging 0.121 at `sigma_eff`. Every one of those exceeds
  the 0.05 the core contrast is sized for, which is why operation-specific claims
  are not made.
- **`rename` and `scramble_list`.** Analysed from **retained pilot data only**
  (`runs/sweep`). This experiment allocates them no prompts, no quota, and no
  calls. Promoting either to a measured arm would require its own sample and
  budget.
- **Detection class is not a stratification axis.** The K = 20 class contrast was
  significant in only 8 of 11 leave-one-out fits. Baseline recovery is also not
  used for stratification or selection: it appears on both sides of
  `d = Y_40 - Y_1`, and a split-half check attributed roughly 43% of its apparent
  association with sensitivity to that coupling.

## 12. Execution safeguards

- Randomise or interleave call order across K so drift cannot align with load.
- Freeze the model, its version, and all sampling settings for the run. No
  `temperature`; `reasoning_effort` fixed (METHODS §3.1).
- Hold the corruption realisation and companion sequence identical across all
  rungs, so the paired difference carries load and not composition noise.
- Record the criterion manifest hash with the run, as the full SHA-256 in
  metadata and call records rather than a display prefix. Anchor prose is part of
  the treatment, so a pool whose keys are unchanged but whose descriptors differ
  must not resume into an earlier run.
- **Freeze a sampling-plan artifact before execution.** The dry run writes it and
  the live run consumes it rather than repeating screening and assignment. It
  carries the source dataset id and file hash, the manifest hash, the study seed,
  the model and its fixed settings, the levels, and per unit: base-prompt id,
  lane, selected side, eligibility result, assigned operation, corruption
  realisation seed, companion order seed, active or reserve status, and the
  planned call identities.
- Retain `runs/` artefacts. The K = 20 sweep was nearly lost to `.gitignore`
  (METHODS §10.7); the run directory is archived outside the ignore rule before
  analysis begins.

## 13. Implementation order

The 20 new criteria are **authored, reviewed and committed before final planner
integration and before any outcome-generating call.** The criteria are part of
the experimental treatment, not test data: their lengths and anchors determine
input-token growth, context and output-limit behaviour, companion composition,
and whether the cost model below still holds. The planner may be developed in
parallel against fixtures, but the K = 40 path is validated against the real
manifest.

The criteria commit carries static checks for:

- exactly 40 unique criterion IDs;
- every scale point anchored, with no duplicate or missing anchors;
- deterministic manifest ordering;
- maximum and distribution of criterion lengths;
- a frozen manifest hash.

| Change | Where |
|---|---|
| 20 new synthetic-only criteria, manifest, validation, frozen ordering | `synthetic_criteria.py` |
| Base-prompt and response selection, eligibility pre-pass, arm assignment, equal quotas | `dataset.py` |
| Fixed-position nested planner: target at position 0, seeded companion prefixes | `runner.py`, `plan_targeted_blocks` |
| One frozen realisation per prompt, reused across rungs | `dataset.py` |
| Equal-weight estimator, Satterthwaite SE, variance-review rule, failure handling | `analyze.py` |
| Wider base frame, lane split by frozen seed | `packing fetch --limit 1600` |

Commit sequence:

1. `docs: preregister fixed-position load ladder`
2. `data: add anchored K=40 criterion set`
3. `feat: implement fixed-position nested ladder`
4. `analysis: add stratified endpoint estimator`

## 14. Budget

Ladder {1, 5, 10, 20, 40}, two sides per prompt, `gpt-5-mini` at list price,
costed from the retained sweep's observed per-call token means
(`input ~ 619 + 183K`, `output ~ 187 + 78.5K`):

Computed by `packing ladder --dry-run` against the real 40-criterion manifest
(hash `721b8ac2a2a7e15f`) on a 1,600-prompt frame. The dry-run ceiling assumes
every call fills its output budget, which it will not; the expected column applies
the ceiling-to-actual ratio measured on the retained K = 20 sweep, where a $44.44
ceiling produced $22.75 of record-derived spend, a factor of 1.954. Both sides
of that ratio are computed from retained token counts, so it converts one floor
into another and is not a billing figure.

| Case | Core n | `numbers` n | Analyzable | Calls | Input tokens | Ceiling | Expected |
|---|---|---|---|---|---|---|---|
| **Initial** | 464 | 184 | 648 | 6,480 | 26,908,312 | $50.58 | **~$26** |
| **Adaptive maximum** | 768 | 184 | 952 | 9,520 | 39,393,570 | $74.28 | **~$38** |
| Endpoint-only, initial | 464 | 184 | 648 | 2,592 | 13,777,364 | $27.03 | ~$14 |

The endpoint-only variant covers the core endpoint and the pre-registered
`numbers` endpoint; the remaining spend buys the descriptive curve. Cost is not
proportional to call count, because K = 40 calls dominate: it is 40% of the calls
but 53% of the ceiling.

Fetched frame: **1,222** base prompts initially, **1,551** if the blinded variance
review raises the core target to its 768 maximum, split into arm lanes by the
frozen seed before eligibility is inspected (§5). Verified on a 1,600-prompt
frame: a 508-candidate core lane filled 464 of 464, and an 837-candidate lane
filled 768 of 768, both matching the binomial fill predictions.

## 15. Pre-registration text

> **Primary operation set and population.** The confirmatory analysis uses eight
> operations: `flatten`, `truncate`, `shuffle`, `drop_answer`, `agreement`,
> `depunct`, `duplicate`, `filler`. The target population is base prompts eligible
> for all eight. Eligibility is determined from the source response before
> operation assignment and without reference to evaluation outcomes.
>
> **Experimental unit and selection.** One base prompt with one selected response.
> Where both responses are eligible, one is selected uniformly at random from the
> study seed and frozen before operation assignment. A base prompt contributes to
> at most one arm.
>
> **Assignment.** Each eligible unit is assigned exactly one operation and one
> corruption realisation, randomised subject to equal operation quotas. The
> realisation and the seeded companion sequence are held fixed across all K
> levels, with lower rungs using nested prefixes of that sequence.
>
> **Primary estimand.** For operation `o`, with `d_i = Y_i,40 - Y_i,1`, the
> primary estimand is the equally weighted average of the eight operation-specific
> mean differences, `theta_core = (1/8) SUM_o E[d_i | O_i = o, A_core = 1]`. It
> describes the effect of adding 39 trailing items after a first-position target
> among prompts eligible for the eight high-applicability operations. It does not
> identify an effect independent of recency or distance from the end of the
> context.
>
> **Primary analysis.** Estimate each operation's mean `d` and average the eight
> means equally, with variance `(1/64) SUM_o s_o^2 / n_o` and Welch-Satterthwaite
> degrees of freedom. Two-sided, alpha = 0.05. One confirmatory hypothesis, no
> multiplicity adjustment.
>
> **Sample size.** Target 464 analyzable prompts, 58 per operation, a 3.6% buffer
> above the 448 implied by the upper bound of the planning-variance CI
> (`sigma_eff` 0.3345, CI [0.2709, 0.3763], estimated on prompts verified eligible
> for all eight core operations), giving 81.5% power for a 0.05 effect at that
> bound. One effect-blinded variance
> review after 160 complete prompts may raise N to a maximum of 768 but may not
> lower it.
>
> **Key secondary.** `numbers` is evaluated separately among prompts eligible for
> that operation, on a disjoint sample of 184, sized for a smallest effect of
> interest of 0.10 at the upper bound of its variance CI. Reported as an estimate
> with a 95% confidence interval. Its estimand is not included in the primary
> average and is not directly comparable with common-population operation
> estimates. A nonsignificant result does not establish the absence of
> practically important correctness degradation; that would require a
> non-inferiority test with margin -0.10, which is not specified here.
>
> **Sampling.** Base prompts are assigned to independent core and `numbers`
> streams by a frozen seed **before** eligibility is inspected, and each stream is
> screened only for its own arm. Assigning arms after screening a shared frame
> would leave the core pool 4.8% `numbers`-eligible against a population value of
> 31.0%, so the primary estimate would no longer describe the population in the
> estimand. Lanes are exact index splits of the seed-permuted frame, 508 core and
> 714 `numbers` initially, extended in frozen order to 837 core if the variance
> review raises the target. Within each lane, screening order is seeded rather
> than file order. The response side that satisfied the full arm operation set is
> frozen at screening and reused, never re-derived from the assigned operation.
> Core operation quotas are dealt from eight preassigned queues sized to the
> 768-unit maximum, so raising the target cannot reassign a collected unit.
>
> **Frozen sampling plan.** The dry run writes a sampling-plan artifact carrying
> the dataset and manifest hashes, the seed, the model settings, the levels, and
> every unit's lane, side, eligibility, operation, realisation seed, companion
> seed and active or reserve status. The live run consumes it rather than
> repeating screening and assignment.
>
> **Outcome definition.** `Y` is demonstrated, machine-readable directional
> recovery: the judge moved the targeted score in the seeded direction and
> returned readable output. An unreadable judgment scores 0 by definition, not by
> imputation. Exactly one outcome is emitted per (prompt, rung), so a pair with
> either or both sides unevaluable yields one unevaluable outcome.
>
> **Sensitivity bounds.** Because the estimand is a difference, the bounds impute
> oppositely at the endpoints: lower bound on d sets unevaluable K = 1 to 1 and
> K = 40 to 0; upper bound reverses that; complete case excludes the pair. A sign
> disagreement across primary and both bounds is reported as sensitivity to the
> outcome definition, not as a load effect.
>
> **Failure handling, three classes.** Transient infrastructure (connection
> errors, rate limits, 5xx, connection timeouts) receives three transport retries
> then prompt replacement from the same lane, is never scored, and never counts as
> a completed identity on resume. Treatment or system unevaluable (truncation,
> malformed content, content filtering, refusal, context-limit rejection, read and
> generation timeouts) is retained and scored. Fatal protocol or configuration
> errors (unsupported parameter, invalid schema, criterion mismatch, corrupt
> input, unexpected exception, invariant violation) abort the run and are never
> scored. An unrecognised failure is fatal rather than unevaluable. Unevaluable
> frequency by K is reported as a diagnostic outcome.
>
> **Secondary analyses.** Tie rate by K is diagnostic. Intermediate K levels,
> operation-specific effects, `rename` and `scramble_list` are exploratory;
> `rename` and `scramble_list` draw on retained pilot data only and receive no
> allocation in this experiment. Reported intervals and p values for all of these
> are unadjusted.
