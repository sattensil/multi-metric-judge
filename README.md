# Does packing criteria into one LLM judge call quietly degrade them?

> **Start here if you came from a talk:** [REPLICATION.md](REPLICATION.md)
>
> | Want to | Go to |
> |---|---|
> | Inspect what was found and which claims hold | the status table and claim map |
> | Reproduce every published number, no API key | `uv run packing reproduce` |
> | Rerun the experiment with fresh judge calls | the nondeterminism and model-alias caveats first |
> | Find the K = 40 extension | pre-registered, currently executing, no results yet |
>
> This repository holds **two experiments at different stages**. The completed
> K = 20 work and the in-flight K = 40 ladder are not one study.

You pay for your judge to read the full input once. Stack judges for multiple
quality dimensions and you re-pay that read each time. Does combining scores into
one call quietly degrade them?

This repo measures the real frontier with a **synthetic-corruption gold set**, not
judge-vs-judge. Open human-labelled data tops out at about five fixed criteria
(HelpSteer2); the synthetic stratum extends the packing load by seeding known
edits with known directions.

- **Independent variable:** number of criteria scored per judge call (`group_size`).
- **Dependent variables:** human-stratum pairwise accuracy; synthetic-stratum
  **targeted corruption recovery** (does the judge move the intended score the
  right way?).
- **Held constant:** the criteria themselves. Every condition scores all of them.
  Only the packing changes.

Synthetic conclusions attach to **operations**, not abstract criteria. Off-target
score movement is reported as construct-overlap context, not an automatic veto.

## Why this is not obvious in either direction

Packing could hurt (attention divided across criteria, one shared reasoning
budget) or help (the judge sees the criteria in context and can trade them off
coherently). Published work leans pessimistic: LongJudgeBench found that adding
a rubric on top of a reference gave *no* improvement on long-form outputs
(0.578 vs 0.584 for reference alone), with judges barely above a 0.5 random
baseline overall. But that is a different question from ours: they varied whether
a rubric was present, not how many criteria shared a call.

## Design in one paragraph

Take items with per-criterion human labels. For each item, draw a **uniformly
random partition** of the K criteria into groups of size ≤ N, and issue one judge
call per group. Sweep N from 1 to K. Because partitions are redrawn per item and
per replicate, criterion identity and criterion pairing are decorrelated from
group size. Within-call **position is not**, and cannot be: position is nested
inside the call, so `E[position | group_size = g] = (g−1)/2` exactly. Position is
randomised *within* a group size, which is what makes it a usable covariate and
what makes the pre-registered rival testable (METHODS.md §8.3.2). Score each
criterion against the human label, then model accuracy as a function of the group
size that criterion was actually judged in.

Three levels of randomisation are required, and all three are implemented:
partition membership, criterion order within a call, and candidate response
order. Dropping any one of them reintroduces a confound.

## Ground truth

**HelpSteer2** (NVIDIA, CC-BY-4.0): 10,681 prompts × 2 responses, every
response human-scored on helpfulness, correctness, coherence, complexity, and
verbosity, Likert 0–4, minimum 3 annotators each. Two responses per prompt gives
**pairwise ground truth**: did the judge order the pair the way humans did? A
binary outcome with a clean 0.5 baseline.

**SummEval** (replication): 1,600 summaries scored on coherence, consistency,
fluency, relevance, 5-point Likert, 5 crowdworkers + 3 experts each. Different
domain, different K, different scale. One substrate proves little: JUDGE-BENCH
found per-dataset variance across 20 datasets large enough that judges have to be
validated per use case.

Two caveats stated up front rather than buried:

1. **`complexity` and `verbosity` are descriptive, not evaluative.** Higher is
   not better. Pairwise accuracy on them measures whether the judge perceives the
   ordering as humans do, which is a valid accuracy measure but not a preference
   judgement.
2. **Ties have no correct answer.** When both responses share a human score on a
   criterion, that observation is excluded and the tie rate is reported. Tie rates
   are expected to differ by criterion, which makes the usable design unbalanced.

## The main threat to validity

If a 5-criterion call has to emit 5 scores and 5 rationales under the same output
budget as a 1-criterion call, accuracy will drop for a completely uninteresting
reason and the study will have measured its own token ceiling. `budget.py` holds
the **per-criterion** output budget constant, and `tests/test_budget.py` pins that
invariant.

If the group-size effect disappears once that control is in place, **that is the
finding.** Report it and stop.

## Statistical approach

Following Miller, [*Adding Error Bars to
Evals*](https://arxiv.org/html/2411.00640v1):

- **Clustered standard errors, always.** Observations cluster by item (K criteria
  × several group sizes each). Miller measured clustered SEs running over 3×
  larger than naive ones on DROP. Naive SEs are reported nowhere in this repo.
- **Within-item paired contrasts as the primary estimator.** Every item is judged
  at every group size, so per-item differences remove item difficulty entirely,
  roughly a third of the variance at ρ=0.5.
- **Replicates, not lower temperature, for variance reduction.** Miller shows
  lowering temperature can triple minimum variance and inject bias. With K
  replicates, item-level means are computed *first*; pooling all K×n observations
  would violate independence.

Primary model is a crossed mixed-effects logistic regression,
`correct ~ group_size + position + (1|item) + (1|criterion)`, with a
cluster-robust GEE fallback that always runs. `group_size` is fitted both as a
factor (where is the cliff?) and as a slope (how steep?), with an LRT deciding
which to headline.

`position` is treated as a **rival hypothesis**, not a nuisance parameter. If
accuracy degrades with where a criterion sits in the prompt rather than with how
many share the call, that is the better finding and it is more actionable.

## Imperfect ground truth

Human labels are noisy, and per [*How to Correctly Report LLM-as-a-Judge
Evaluations*](https://arxiv.org/pdf/2511.21140), treating a noisy reference as
perfect is a named reporting error. So: judge accuracy is reported relative to
inter-annotator agreement rather than to 1.0, with a sensitivity analysis on the
high-agreement subset.

The defence for the primary estimand is that label-noise attenuation is
approximately **constant across group sizes**, so it depresses absolute accuracy
without biasing the group-size *contrast*. That has to be argued, not assumed.

## Judge model

`gpt-5-mini` ($0.25 / $2.00 per 1M), deliberately not the cheapest available.

The study needs a judge with **headroom to degrade**. If accuracy at group_size 1
already sits near 0.5, there is no room to detect a decline and the study returns
a null for the wrong reason. `packing select-model` runs candidates at group_size
1 only and picks the cheapest that clears a pre-registered headroom floor.

## Scoring is pointwise, not pairwise-presented

One call scores **one response** on one group of criteria; the pairwise ordering is
then derived from the two responses' scores. Showing the judge both responses at
once would be cheaper, and it was rejected for three reasons:

1. It matches how the labels were made. HelpSteer2 annotators scored each response
   independently. A judge shown both at once is solving an easier task than the
   annotators were.
2. It matches production use. Real judges score one output, not a pair.
3. It keeps the manipulation clean. A pairwise call doubles the content the judge
   must hold per criterion, entangling group size with candidate count.

The cost is two calls per group. In exchange the absolute scores are usable for
QWK, which a preference-only judge could not provide.

## Anchor every scale point, or the judge invents a shorter scale

Measured, not theorised. With criteria anchored only at 0, 2 and 4, `gpt-5-mini`
used only 0, 2 and 4, never 1 or 3, and tied on 66% of scoreable pairs, leaving
29 usable observations out of 100. Anchoring all five points took the tie rate to
43% and usable observations to 49, with all five values in play.

`Criterion.__post_init__` now rejects a criterion with any undescribed scale
point. A partially anchored scale becomes the scale.

## Judge indecision is a co-primary outcome, not a nuisance

When the judge scores both responses equally it has expressed no preference. Those
observations are excluded from the accuracy outcome, but excluding them is only
defensible if the tie rate is itself reported, because **the tie rate may rise with
group size.** A judge that becomes flatter and more indecisive under packing is
degrading, and silently dropping those rows would bias the contrast in exactly the
direction the study is looking for. So the judge tie rate is modelled with the same
design and reported alongside.

## Design balance

A flat sweep of nominal levels 1..K is badly unbalanced in the quantity that
matters. Ragged partitions produce small groups as a by-product of several levels,
while the largest group size comes from exactly one. At K=5 with one replicate per
level, criterion-observations per group size are `{1:7, 2:6, 3:3, 4:4, 5:5}`, the
least data in the cells the study is about, a 2.33x imbalance.

Replicates are therefore allocated per nominal level by non-negative least squares
to equalise observations. The default, `{1:1, 2:1, 3:3, 4:2, 5:2}`, cuts the
imbalance to 1.25x for 54% more calls. `packing design` shows the table.

## The estimators are validated against planted effects

`tests/test_recovery.py` runs the whole pipeline offline against a mock judge with a
known degradation rate, and requires both halves:

- **Power**: it recovers a planted effect (correct sign, Holm-significant, and the
  LRT correctly prefers the factor form when the planted decline is non-linear).
- **Calibration**: it produces *no* significant curve on a judge with no planted
  effect.

An estimator suite that has never been tested against a known answer is not
evidence of anything.

## Usage

```bash
uv sync
uv run pytest                       # gates 1 and 4: uniformity, budget invariance, recovery
uv run packing design               # observations per group size, replicate allocation

uv run packing fetch --dataset helpsteer2 --limit 500
uv run packing select-model         # gate 0: cheapest model with headroom
uv run packing run --dry-run        # call plan + cost ceiling, spends nothing

uv run packing pilot                # gate 3: group_size 1 vs K only
uv run packing score --run pilot    # gate 2: tie rates, descriptives, QWK
uv run packing power --run pilot    # size the confirmatory run
uv run packing analyze --run pilot  # estimators + cost frontier
```

Everything except `run`, `pilot`, and `select-model` works with no API keys. To
exercise the full chain offline:

```bash
uv run python scripts/make_synthetic.py -n 60
uv run packing run --items data/synthetic.jsonl --model mock --name e2e
uv run packing analyze --run e2e
```

## Planned extension: the fixed-position load ladder

**Planned, not run.** The completed K = 20 sweep bounds degradation rather than
estimating it, and load cannot be separated from criterion position
nonparametrically, because position is nested inside the call. The next run
removes that confound by construction: the target criterion sits at position 0 in
every condition and companion criteria pad the call to reach K in
{1, 5, 10, 20, 40}, with lower rungs using nested prefixes of one seeded
companion sequence.

Confirmatory primary is the equal-weighted average across eight
high-applicability operations, K = 1 against K = 40, at 464 prompts and 58 per
operation. `numbers` runs as a separate pre-registered correctness sentinel.
Protocol, power and pre-registration:
`docs/superpowers/specs/2026-09-08-load-ladder-design.md`.

## Gates

0. Model selection clears the headroom floor.
1. `tests/test_partition.py` passes. A biased sampler confounds everything
   downstream and the resulting curve would look perfectly plausible.
2. Tie rates measured per criterion.
3. Pilot effect size feeds Miller's power formula before the confirmatory spend.
4. Budget control verified: if the effect vanishes, that is the result.
5. SummEval replication before any curve is presented.

## Sources

- Miller, [*Adding Error Bars to Evals*](https://arxiv.org/html/2411.00640v1)
- [*How to Correctly Report LLM-as-a-Judge Evaluations*](https://arxiv.org/pdf/2511.21140)
- [HelpSteer2](https://arxiv.org/pdf/2406.08673) · [dataset](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [SummEval](https://aclanthology.org/2021.tacl-1.24.pdf) · [repo](https://github.com/Yale-LILY/SummEval)
- [JUDGE-BENCH](https://arxiv.org/abs/2406.18403)
- [LongJudgeBench](https://arxiv.org/html/2606.01629v1)
- [Self-Preference Bias in Rubric-Based Evaluation](https://arxiv.org/pdf/2604.06996)
