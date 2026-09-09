# Fixed-position load ladder: design

## 1. Question

Does packing more criteria into one judge call degrade per-criterion judging, and
if so where does it start?

The completed K = 20 sweep (METHODS §8.3) bounds degradation rather than
estimating it, and its load contrast cannot be separated from criterion position
nonparametrically (§8.3.2). This design answers the load question with position
removed by construction rather than by covariate adjustment.

## 2. Primary estimand

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
end of the context. That is a narrower claim than "the effect of load" and is to
be reported as such.

## 3. Operation set

**Core set, eight operations:** `flatten`, `truncate`, `shuffle`, `drop_answer`,
`agreement`, `depunct`, `duplicate`, `filler`.

This is the *high-applicability operation set*, not the complete catalogue. The
set is chosen so that a single common eligible population exists, which is what
makes equal quotas meaningful and gives operation contrasts a causal reading
within that population.

Measured applicability over all 400 local responses, reproducing METHODS §3.4:

| Operation | Applicability | In core set |
|---|---|---|
| `duplicate` | 100.0% | yes |
| `filler` | 100.0% | yes |
| `depunct` | 99.5% | yes |
| `agreement` | 99.0% | yes |
| `shuffle` | 95.5% | yes |
| `drop_answer` | 95.5% | yes |
| `truncate` | 91.0% | yes |
| `flatten` | 85.8% | yes |
| `rename` | 62.5% | no |
| `scramble_list` | 43.0% | no |
| `numbers` | 24.2% | separate arm |

Joint eligibility is a function of the set, not a fixed constraint:

| Operations kept | Joint eligibility (response level) |
|---|---|
| all 11 | 8.2% |
| drop `numbers` | 33.0% |
| drop `numbers`, `scramble_list` | 61.0% |
| **core 8** | **84.8%** |

At base-prompt level, counting a prompt eligible when either response qualifies,
the core set reaches **93.5%** (187 of 200 local prompts).

**Rejected alternative.** An eleven-operation primary average is a valid estimand
but a compound one: it averages eleven effects measured on eleven different
prompt populations, and cross-operation differences stay inseparable from
population differences. At 8.2% joint eligibility the common-population version
of it is not viable, since it would select for long, list-bearing, number-bearing
responses.

## 4. Design

**Experimental unit.** One unique source prompt and response pair. Any higher
level shared source, template, or document that induces dependence is instead
treated as the independent unit. Clustering is on the source prompt, which is
what the retained sweep's cluster collapse (440 corrupted items to 53 clusters)
makes unavoidable.

**Assignment.** Each eligible prompt is assigned exactly one operation and one
corruption realisation. Assignment is randomised subject to equal operation
quotas, before any outcome is generated. Eligibility is determined from the
source response alone, with no reference to evaluation outcomes.

One realisation per prompt, not merely one operation category: where an operation
can generate several candidate corruptions, exactly one is drawn and frozen, so
implicit weighting cannot return at the realisation level.

**Position.** The target criterion is pinned at position 0 in every call.
Companion criteria pad the call to reach K.

**Ladder.** K in {1, 5, 10, 20, 40}. The K = 1 against K = 40 contrast is the
single confirmatory comparison. K = 5, 10 and 20 are descriptive.

**Held constant across rungs.** The corruption realisation, the model and its
settings, the per-criterion output budget (`40 + 220 * K`, METHODS §4.3), and the
target's position. Only the quantity of trailing companion criteria varies.

**Criterion pool.** K = 40 requires 20 new synthetic-only criteria beyond the
current 20. Every scale point of every new criterion must be anchored; partial
anchoring collapses the scale to its anchored points (METHODS §3.3).

## 5. Primary analysis

Compute `d_i = Y_i,40 - Y_i,1` per prompt. Estimate each operation's mean, then
average the eight means equally:

    theta_hat_core = (1/8) * SUM_o dbar_o

    Var_hat(theta_hat_core) = (1/64) * SUM_o (s_o^2 / n_o)

Equal weighting is preserved when quotas land unevenly or a few outcomes go
missing, which an unqualified mean over all `d_i` would not do. Equal quotas
define the weights rather than removing the need to declare them.

Test: two-sided, alpha = 0.05, one pre-registered hypothesis, so no multiplicity
adjustment applies to it.

## 6. Power

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

Required analyzable prompts at MDE 0.05, exact paired-t:

| `sigma_eff` | n | Equal-quota target | Per operation |
|---|---|---|---|
| 0.2651 (CI low) | 223 | 224 | 28 |
| 0.3288 (point) | 342 | 344 | 43 |
| 0.3693 (CI high) | 431 | 432 | 54 |

**Chosen target: 464 analyzable prompts, 58 per operation.** Power at a true
effect of 0.05, evaluated at the *upper* bound of the variance CI:

| n | Power at `sigma_eff` = 0.3693 |
|---|---|
| 344 | 70.7% |
| 400 | 77.1% |
| **464** | **82.9%** |
| 576 | 90.0% |

464 is the smallest round equal-quota size holding at least 80% power across the
whole bootstrap CI rather than only at the point estimate.

**Two documented assumptions.** `sigma_eff` is measured on the K = 1 against
K = 20 contrast, while the planned endpoint is K = 40; `SD(Y_40 - Y_1)` is not
directly estimated and could be larger or smaller. And it rests on 20 to 23
clusters per operation. The pre-registered sample size is therefore conditional:
it is recomputed from the same equal-quota estimator once the first 100 prompts
land, and revised upward if the realised `sigma_eff` exceeds 0.3693.

No power claim is made for the intermediate rungs, for operation-specific
effects, or for any shape or threshold hypothesis. The closed form above applies
to the two-level contrast only; a confirmatory shape claim would require
simulating the exact planned model and is not attempted.

## 7. Correctness arm: `numbers`

`numbers` is the only correctness-targeting operation, and at 24.2%
applicability it cannot join a common-population core. It is retained as a
separate pre-registered arm rather than folded into the primary average:

    theta_numbers = E[ Y_40 - Y_1 | prompt eligible for numbers ]

Disjoint sample of number-eligible prompts, one fixed `numbers` realisation each,
same ladder. Measured `sigma_d` = 0.4261, giving n = 145 for MDE 0.10, which
requires screening about 599 base prompts.

Its estimate is **not** interpreted as directly comparable with the core
operations, because the populations differ.

The reason to retain it is construct coverage, not headroom. At baseline recovery
0.375 it has 0.375 of room to decline and 0.625 to rise; describing that as a
headroom advantage for detecting degradation would be backwards.

## 8. Secondary and exploratory

All unadjusted, all labelled exploratory, none supporting confirmatory claims:

- K = 5, 10, 20 rungs, reported descriptively. They may not be used to claim
  monotonicity, a threshold, or curvature.
- Operation-specific effects. At 58 prompts per operation the per-operation MDE
  varies with each operation's own variance, from 0.057 (`agreement`) to 0.191
  (`flatten`), averaging 0.121 at `sigma_eff`, all before any multiplicity
  correction. Every one of those is far above the 0.05 the core contrast is
  sized for, which is why operation-specific claims are not made.
- `rename` and `scramble_list` as specialised single-operation analyses. They do
  not enter the primary sampling frame.
- Tie rate by K, co-primary in the human stratum (METHODS §5.2) and reported
  here alongside recovery, since the K = 20 surface decline proved to be entirely
  a tie increase with wrong-direction at zero (§8.3.2).
- Detection class is **not** a stratification axis. The K = 20 class contrast was
  significant in only 8 of 11 leave-one-out fits. Baseline recovery is also not
  used for stratification or selection: it appears on both sides of
  `d = Y_40 - Y_1`, and a split-half check put roughly 43% of its apparent
  association with sensitivity down to that coupling.

## 9. Execution safeguards

- Randomise or interleave call order across K so drift cannot align with load.
- Freeze the model, its version, and all sampling settings for the run. No
  `temperature`; `reasoning_effort` fixed (METHODS §3.1).
- Hold the corruption realisation identical across all rungs, so the paired
  difference carries load and not realisation noise.
- Pre-define missing-data, API-failure and replacement rules. A prompt losing any
  rung to a failure is dropped whole rather than analysed on a partial ladder,
  and its replacement is drawn from the same eligible frame.
- Retain `runs/` artefacts. The K = 20 sweep was nearly lost to `.gitignore`
  (METHODS §10.7); the run directory is archived outside the ignore rule before
  analysis begins.

## 10. Implementation

| Change | Where |
|---|---|
| 20 new synthetic-only criteria, every scale point anchored | `synthetic_criteria.py` |
| Fixed-position planner: target at position 0, companions pad to K | `runner.py`, `plan_targeted_blocks` |
| Equal-quota randomised assignment with an eligibility pre-pass | `dataset.py` |
| One frozen realisation per prompt, reused across rungs | `dataset.py` |
| Equal-weight estimator and its heteroskedastic SE | `analyze.py` |
| Fetch a wider base frame | `packing fetch --limit ~1100` |

## 11. Budget

Ladder {1, 5, 10, 20, 40}, two sides per prompt, `gpt-5-mini` at list price,
costed from the retained sweep's observed per-call token means
(`input ≈ 619 + 183K`, `output ≈ 187 + 78.5K`):

| Arm | n | Calls | Cost |
|---|---|---|---|
| Core, 8 operations at 58 each | 464 | 4,640 | $16.76 |
| `numbers` | 145 | 1,450 | $5.24 |
| **Total** | | **6,090** | **$21.99** |

An endpoint-only implementation of the two confirmatory contrasts would be 2,436
calls and about $11.42; the remaining spend buys the descriptive curve. Cost is
not proportional to call count, because K = 40 calls dominate: the endpoint-only
variant is 40% of the calls but 52% of the cost.

Base prompts to fetch: about 496 for the core arm at 93.5% eligibility plus about
599 for the `numbers` arm at 24.2%, so roughly 1,100 in total. HelpSteer2 supplies
10,681, so the frame is not binding. The 200 prompts on disk are a pilot frame and
would cap the core arm at 187 eligible, giving an MDE of 0.067 at the point
estimate of `sigma_eff` and 0.076 at the upper bound of its CI, so the local frame
alone cannot reach the 0.05 target.

## 12. Pre-registration text

> **Primary operation set and population.** The confirmatory analysis uses eight
> operations: `flatten`, `truncate`, `shuffle`, `drop_answer`, `agreement`,
> `depunct`, `duplicate`, `filler`. The target population is source prompt and
> response pairs eligible for all eight. Eligibility is determined from the source
> response before operation assignment and without reference to evaluation
> outcomes.
>
> **Assignment.** Each eligible pair is assigned exactly one operation and one
> corruption realisation. Assignment is randomised subject to equal operation
> quotas. The realisation is held fixed across all K levels.
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
> means equally, with variance `(1/64) SUM_o s_o^2 / n_o`. Two-sided,
> alpha = 0.05. One pre-registered hypothesis, no multiplicity adjustment.
>
> **Sample size.** Target 464 analyzable prompts, 58 per operation, which holds at
> least 80% power for a true effect of 0.05 across the bootstrap CI of the
> planning variance. The calculation is conditional and is recomputed from the
> realised `sigma_eff` after the first 100 prompts.
>
> **Specialised correctness analysis.** `numbers` is evaluated separately among
> prompts eligible for that operation, on a disjoint sample. Its estimand is not
> included in the primary average and its estimate is not interpreted as directly
> comparable with common-population operation estimates.
>
> **Secondary analyses.** Intermediate K levels, operation-specific effects,
> `rename` and `scramble_list` are exploratory unless separately powered and
> assigned an explicit confirmatory testing procedure. Reported intervals and p
> values for them are unadjusted.
