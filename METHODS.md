# Methods

This study asks a practical question: **how many evaluation criteria can an LLM
judge score in one call before score quality declines?**

This file describes the method and marks which analyses were fixed in advance
versus exploratory. Results are presented in
[`presentation/index.html`](presentation/index.html).

## 1. Study logic

Scoring several criteria in one call is cheaper than making one call per
criterion because the prompt and response are read once. The possible cost is
lower-quality judgments.

- **Manipulation:** `group_size`, the number of criteria scored in one call.
- **Quality outcome:** agreement with a known ordering between two responses.
- **Cost outcome:** recorded input, output, and reasoning tokens per criterion.
- **Held constant:** the criterion set and response content. Only the packing of
  criteria changes.
- **Primary hypothesis:** agreement declines as `group_size` increases.
- **Rival explanation:** later criterion position, rather than group size
  itself, may cause the decline.

The unit of inference is the source prompt or base response, not an individual
criterion score. Repeated scores from the same source are therefore paired or
clustered in every inferential analysis.

## 2. Experiments

The project contains three related experiments:

| Experiment | Ground truth | Packing range | Purpose |
|---|---|---|---|
| Human-label pilot | HelpSteer2 annotator scores | K = 1 and 5 | Check the basic quality/cost trade-off on natural labels |
| Synthetic packing sweep | Known text corruptions | K = 1–20 | Screen a wider range of packing loads and operations |
| Fixed-position ladder | Known text corruptions | K = 1, 5, 10, 20, and 40 | Estimate the endpoint effect while removing criterion-position confounding |

The human and synthetic strata answer different questions and are never pooled.
The human stratum measures agreement with noisy human labels. The synthetic
stratum measures whether the judge detects a known edit in the expected
direction.

The K = 30 ladder rung was added after the registered ladder was run. It is used
only in exploratory trend descriptions, never in the registered K = 1 versus
K = 40 endpoint test.

## 3. Materials

### 3.1 Response corpus

Responses come from the HelpSteer2 train split (NVIDIA, CC-BY-4.0). Almost every
prompt has two independently scored responses. The released labels cover:

- helpfulness
- correctness
- coherence
- complexity
- verbosity

Each label is an integer from 0 to 4. Human agreement varies substantially by
criterion, so judge agreement is interpreted relative to the published
human-human reliability rather than against a hypothetical perfect reference.

The human pilot uses response pairs with enough label separation to define a
meaningful ordering. This raises absolute accuracy and limits population
generality, but it does not change the within-pair packing manipulation.

### 3.2 Judge

The retained study uses `gpt-5-mini`. Model choice was screened at
`group_size = 1`; a judge already near chance would have no measurable room to
degrade.

Calls use schema-constrained JSON output. Every criterion is required, every
score is restricted to its allowed integer values, and every scale point has a
written anchor. Partial anchoring was rejected because it caused the judge to
treat only the described points as valid.

Reasoning effort is fixed across conditions. Temperature is not lowered as a
variance-control technique; repeated measurements and within-item comparisons
provide the variance reduction instead.

### 3.3 Criterion sets

The human pilot uses the five HelpSteer2 criteria.

The synthetic sweep expands the prompt to 20 criteria. Seventeen corruption
operations passed calibration and contribute measured outcomes; three criteria
remain as packing load only.

The fixed-position ladder uses a 40-criterion manifest. Its primary core contains
eight operations with high joint eligibility:

`flatten`, `truncate`, `shuffle`, `drop_answer`, `agreement`, `depunct`,
`duplicate`, and `filler`.

The `numbers` operation is retained as a separate correctness sentinel because
it applies only to responses containing suitable numerals.

Criterion wording and scale anchors are part of the experimental instrument.
Their full text and hashes live in the criterion manifests in `src/packing/`.

### 3.4 Synthetic ground truth

Each synthetic unit begins with one clean response. A deterministic or seeded
operation creates a corrupted version of that same response. The operation
declares:

1. the criterion it targets;
2. whether that criterion should increase or decrease; and
3. the severity and random seed used to construct the edit.

The clean and corrupted texts are placed on response A or B using an independent
seed. A judge that always favors the first response therefore gains no advantage.

Synthetic gold is **directional operation recovery**, not an absolute human
quality score. For example, duplicated text should receive a higher verbosity
score than its clean source; the method does not assert that either response has
an absolute verbosity score of 4.

Operations must satisfy three implementation checks:

- the same text, seed, and severity reproduce the same output;
- an operation that cannot make its claimed edit fails honestly instead of
  emitting an unchanged “corruption”; and
- code spans are preserved while prose is edited.

Severity is calibrated once per operation at `group_size = 1`. It is not varied
with packing load.

## 4. Sampling and randomization

### 4.1 Human pilot and synthetic sweep

For each source item, criterion partitions are generated from a deterministic
hash of the study seed, item identity, nominal group size, and replicate.

The sweep randomizes:

- which criteria share a call;
- criterion order within a call; and
- response order.

The recorded analysis uses the **actual** group size. Ragged partitions are
retained: for example, splitting five criteria into blocks of at most two
produces groups of 2, 2, and 1.

The human design allocates replicates to reduce imbalance across actual group
sizes. The synthetic K = 20 sweep enumerates its registered nominal levels
directly and is not equally balanced across every resulting actual group size.

### 4.2 Fixed-position ladder

The ladder removes the main position confound by fixing the targeted criterion
at position 0. Additional criteria are trailing companions that create load but
carry no ground-truth role.

Lower rungs are nested prefixes of one seeded companion sequence. The response
and target criterion are therefore unchanged as load increases.

The registered core sample contains 464 prompts: 58 for each of eight
operations. A separate `numbers` lane contains 184 prompts. Operation queues and
response sides are frozen before execution so an adaptive increase can only add
units; it cannot reassign a collected unit.

### 4.3 Output budget

Larger calls must emit more scores and explanations. To avoid manufacturing a
packing effect through truncation, output capacity scales with the number of
criteria:

`content budget = 40 + 220 × group_size`

A separate fixed reasoning allowance is added per call. Near-cap diagnostics use
content tokens after subtracting reported reasoning tokens.

Prompt and response-schema length still grow with `group_size`. Schema size is
recorded, and the study is not described as a pure test of divided attention.

## 5. Execution and data integrity

One call scores one response on one criterion block. The A-versus-B comparison is
formed later in analysis, matching the pointwise protocol used for the human
labels.

Calls run concurrently, but every record carries a deterministic identity:

`(item_id, side, model, replicate, partition_hash, group_index)`

Results are appended to JSONL under a write lock. Resuming a run skips identities
that are already complete. Successfully completed calls are retained even if a
different concurrent call causes the run to abort.

Sampling plans freeze the dataset digest, criterion-manifest digest, seed, model,
units, response sides, operations, and planned call identities. A live run
refuses a changed frame, changed instrument, reassigned unit, or dropped rung.

### 5.1 Failure handling

Failures are separated by what they mean for the experiment:

- **Transient infrastructure failure:** the request did not reliably reach or
  return from the provider. It is retried and never scored as judge behavior.
- **Treatment-related unevaluable output:** truncation, filtering, or another
  recognized failure may depend on request size. It remains part of the
  treatment outcome rather than being silently dropped.
- **Fatal protocol failure:** billing exhaustion, unsupported configuration, or
  an unknown failure aborts the run instead of becoming scientific evidence.

Failure class is derived from the stored message during analysis so one
definition is applied to every record.

### 5.2 Replacement

If a unit remains incomplete after transient retries, the current runner replaces
it with the next unused reserve from the same sampling lane. The replacement:

- inherits the failed unit's operation;
- uses the reserve prompt's **own screened response side**;
- is evaluated at every ladder rung; and
- is written to an append-only substitution log before its calls are issued.

Replacement is substitutive: the failed unit leaves the live sample. A failed
replacement can itself be replaced. Unevaluable treatment-related output is not
replaced because doing so would condition the result on surviving high load.

The retained K = 40 archive contains no substitutions. Its remaining transient
holes are handled by the paired complete-unit analysis. New executions use the
replacement path above.

### 5.3 Cost records

Token and cost summaries come from retained call records. They measure recorded
per-call economics, not the provider invoice: SDK-level retries may be billed
without producing a retained record. Dry-run authorization estimates therefore
separate estimated usage from conservative input and output ceilings.

## 6. Outcomes

### 6.1 Human-label outcome

For each criterion, the outcome is whether the judge orders the two responses in
the same direction as the HelpSteer2 labels.

Human-label ties are excluded because no correct ordering exists. Judge ties are
not counted as correct or incorrect in resolved-only accuracy; their rate is
reported separately.

### 6.2 Synthetic outcome

The primary synthetic outcome is whether the targeted score moves in the
operation's known direction. A wrong direction, judge tie, or recognized
unevaluable judgment is a failure to demonstrate recovery.

Non-targeted criteria have no synthetic gold label. Their score movement is used
only to measure specificity and construct leakage.

### 6.3 Secondary outcomes

Secondary summaries include:

- judge tie rate;
- tie-tolerant credit, assigning 0.5 to a judge tie;
- quadratic weighted kappa and mean absolute error where absolute labels exist;
- unchanged-score specificity for non-targeted criteria;
- input, output, reasoning, and cached tokens per criterion; and
- proximity to the configured output cap.

Unevaluable rows are excluded from diagnostics that describe actual judge
decisions, such as judge tie rate.

## 7. Statistical analysis

### 7.1 Primary paired comparisons

For each source item, criterion and replicate outcomes are averaged within each
group size first. Group-size differences are then computed within item.

The human analysis compares packed conditions with `group_size = 1` and applies
Holm correction across the tested ladder.

The registered fixed-position endpoint compares K = 40 with K = 1. Differences
are estimated separately for each of the eight core operations and then averaged
with equal operation weight. Standard errors use the variation of unit-level
paired differences; operations with more usable units do not receive more
substantive weight.

### 7.2 Regression diagnostics

Secondary logistic regressions model correctness as a function of group size and
criterion position. Standard errors are clustered on the source item.

A crossed item-and-criterion mixed-effects logistic model is also reported as a
secondary check. It is fitted variationally and never replaces the paired
estimator.

Criterion- or operation-specific follow-ups use multiplicity correction where
they are treated as parallel tests.

### 7.3 Functional form

Group size is represented both as a categorical factor and as a linear trend.
The linear trend and the added K = 30 rung are exploratory.

The factor-versus-linear likelihood-ratio statistic ignores item clustering. It
is retained only as a descriptive diagnostic and **does not select the
inferential model or establish a threshold, monotonicity, or curvature**.

## 8. Reproducibility

No model calls are needed to verify the published numbers. The replication
workflow uses retained items, per-call records, metadata, sampling plans, and
checksum manifests.

The main safeguards are:

- deterministic seeded randomization;
- content hashes for datasets, criterion manifests, partitions, and plans;
- append-only call and substitution logs;
- exact call-identity checks when rebuilding a plan;
- archived run artifacts outside the ignored `runs/` directory; and
- checksum-pinned replication bundles.

Use `uv run packing reproduce --strict` for the pinned numerical claims. This
checks retained records; it does not rerun the model or establish that a fresh
hosted model would return identical judgments.

## 9. Interpretation limits

1. **Synthetic operations are proxies.** Detecting a mechanical edit does not
   establish broad competence on the named human construct. Conclusions attach
   first to operations, then transfer to criteria only where the operation is a
   faithful instance.
2. **Corruptions can affect neighboring criteria.** The leakage matrix describes
   this overlap; it is not used for post-hoc exclusion.
3. **The human sample is restricted.** Clearer label separation raises absolute
   accuracy, so accuracy levels do not automatically transfer to the full
   HelpSteer2 population.
4. **Packing changes prompt length.** The treatment combines criterion count,
   schema size, and shared-context load.
5. **The ladder fixes the target at the front.** It removes ordinal-position
   variation but still couples load with distance from the end of the prompt.
6. **One model and one corpus were tested.** The estimated frontier is not
   assumed to transfer to other judges, domains, or prompt formats.
7. **The corpus is public.** Training-data contamination may inflate absolute
   accuracy, though every packing condition sees the same source items.

## 10. Pre-specification boundary

The registered confirmatory ladder contains K = 1, 5, 10, 20, and 40, with the
equal-weighted K = 1 versus K = 40 core-operation contrast as its primary test.

The following are explicitly exploratory:

- the K = 30 rung;
- the linear trend across all rungs;
- additive decomposition of group size and criterion position;
- operation-specific heterogeneity beyond the registered endpoint; and
- broad criterion-level claims from synthetic operation recovery.

## References

- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations.* arXiv:2411.00640.
- Wang, Z. et al. (2024). *HelpSteer2: Open-source dataset for training
  top-performing reward models.* arXiv:2406.08673.
- *How to Correctly Report LLM-as-a-Judge Evaluations.* arXiv:2511.21140.
