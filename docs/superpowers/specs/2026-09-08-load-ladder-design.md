# Fixed-position load ladder: design

Status: **planned**. No outcome-generating call has been made under this design.

## 1. Question

Does packing more criteria into one judge call degrade per-criterion judging, and
if so where does it start?

The completed K = 20 sweep (METHODS §8.3) bounds degradation rather than
estimating it, and its load contrast cannot be separated from criterion position
nonparametrically (§8.3.2). This design answers the load question with position
removed by construction rather than by covariate adjustment.

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
qualifies, measured on the 200 local prompts:

| Arm | Base-prompt eligibility |
|---|---|
| core 8 | **93.5%** (187 / 200) |
| `numbers` | **29.0%** (58 / 200) |

Four-cell overlap:

| Cell | Share |
|---|---|
| core only | 64.5% |
| both | 29.0% |
| `numbers` only | **0.0%** |
| neither | 6.5% |

**`numbers`-eligible is a strict subset of core-eligible.** So one shared fetched
frame serves both arms, and its size is set by the core requirement after the
`numbers` quota is carved out, not by summing the two screening counts. Expected
counts give a minimum shared frame of **N = 700**; requiring at least 97.5%
probability on each constraint under a binomial model gives **N = 715**.

Screening counts for reference, using base-prompt rates throughout:

| Arm | Needed | Rate | Expected screen | Screen for 95% fill |
|---|---|---|---|---|
| core | 464 | 93.5% | 496 | 506 |
| `numbers` | 184 | 29.0% | 634 | 559 for 145 |

A `--limit` above 715 is a conservative fetch cap, not a mathematical
requirement. HelpSteer2 supplies 10,681 prompts, so the frame is not binding. The
200 prompts on disk are a pilot frame; they would cap the core arm at 187
eligible, giving an MDE of 0.067 at the point estimate of `sigma_eff` and 0.076 at
the upper bound of its CI.

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
settings, the output-budget rule (`40 + 220 * K`, METHODS §4.3), the target's
position, and the companion sequence. Only the number of trailing companions
varies. At K = 40 the rule yields a 9,352-token ceiling, against a predicted
3,327 tokens of actual output, so the ceiling is not expected to bind. Truncation
is nonetheless handled as a model-output failure under §10.

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

Estimated on the retained sweep restricted to the eight core operations
(24 clusters, 165 prompt-operation pairs):

| Quantity | Value |
|---|---|
| `sigma_eff` | **0.3288** |
| cluster bootstrap 95% CI | [0.2651, 0.3693] |
| pooled SD across all 11 operations | 0.4238 |

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
| 0.2651 (CI low) | 223 | 224 |
| 0.3288 (point) | 342 | 344 |
| 0.3693 (CI high) | 431 | **432** |

**Chosen target: 464 analyzable prompts, 58 per operation.** The upper bootstrap
bound implies a minimum equal-quota target of 432 prompts under the planning
approximation. 464 is a pre-specified 7.4% buffer above that minimum, yielding
about 82.9% power at that variance bound.

Power at a true effect of 0.05, evaluated at the upper bound of the variance CI:

| n | Power at `sigma_eff` = 0.3693 |
|---|---|
| 344 | 70.7% |
| 400 | 77.1% |
| 432 | 80.2% |
| **464** | **82.9%** |
| 576 | 90.0% |

**Two documented assumptions.** `sigma_eff` is measured on the K = 1 against
K = 20 contrast, while the planned endpoint is K = 40; `SD(Y_40 - Y_1)` is not
directly estimated and could be larger or smaller. And it rests on 20 to 23
clusters per operation.

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

## 10. Failure handling

Two distinct categories, because conflating them biases the result. Model-output
failures may become **more** common at K = 40, which makes them part of the load
effect; dropping and replacing them conditions the analysis on surviving high
load and biases degradation downward.

> **Infrastructure failure.** Transport error, provider outage, or any failure
> occurring before a valid model response. Retry a fixed number of times; after
> persistent failure, replace the prompt from the same sampling frame.
>
> **Model-output failure.** Truncated, malformed, incomplete, refusing, or
> otherwise unevaluable output returned by the model. **Do not replace.** Retain
> under a pre-specified scoring rule, report frequency by K, and provide a
> sensitivity analysis where directional recovery cannot be uniquely assigned.

Complete-case analysis is reported as a sensitivity check, never as the primary
handling rule. A prompt losing a rung to infrastructure failure is dropped whole
rather than analysed on a partial ladder.

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
- Record the criterion manifest hash with the run.
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
| Wider base frame | `packing fetch --limit 800` |

Commit sequence:

1. `docs: preregister fixed-position load ladder`
2. `data: add anchored K=40 criterion set`
3. `feat: implement fixed-position nested ladder`
4. `analysis: add stratified endpoint estimator`

## 14. Budget

Ladder {1, 5, 10, 20, 40}, two sides per prompt, `gpt-5-mini` at list price,
costed from the retained sweep's observed per-call token means
(`input ~ 619 + 183K`, `output ~ 187 + 78.5K`):

| Arm | n | Calls | Cost |
|---|---|---|---|
| Core, 8 operations at 58 each | 464 | 4,640 | $16.76 |
| `numbers` sentinel | 184 | 1,840 | $6.65 |
| **Total** | | **6,480** | **$23.40** |

An endpoint-only implementation of the core endpoint and the pre-registered
`numbers` endpoint would be 2,592 calls and about $12.15; the remaining spend buys
the descriptive curve. Cost is not proportional to call count, because K = 40
calls dominate: the endpoint-only variant is 40% of the calls but 52% of the cost.

Shared fetched frame: **715 base prompts**, which fills both arms at 97.5%
probability on each constraint.

**This budget is provisional.** It extrapolates token usage from a run whose
maximum K was 20, using a 20-criterion manifest. It is recomputed against the
actual 40-criterion manifest once the new criteria are authored, and the recomputed
figure is the one used for the spend decision.

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
> **Sample size.** Target 464 analyzable prompts, 58 per operation, a 7.4% buffer
> above the 432 implied by the upper bound of the planning-variance CI, giving
> about 82.9% power for a 0.05 effect at that bound. One effect-blinded variance
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
> **Failure handling.** Infrastructure failures are retried and replaced.
> Model-output failures are retained, scored under a pre-specified rule, and
> reported by K, because their frequency may itself vary with load.
>
> **Secondary analyses.** Tie rate by K is diagnostic. Intermediate K levels,
> operation-specific effects, `rename` and `scramble_list` are exploratory;
> `rename` and `scramble_list` draw on retained pilot data only and receive no
> allocation in this experiment. Reported intervals and p values for all of these
> are unadjusted.
