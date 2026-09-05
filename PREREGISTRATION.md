# Pre-registration

Commit this file, with the placeholders resolved, **before** the confirmatory run.
Its purpose is to stop the analysis from being chosen after the data is seen. The
pilot is explicitly exploratory; everything below governs the confirmatory run only.

Status: **DRAFT** — thresholds are proposals until the pilot fills in the
variance estimates, at which point this file is frozen and committed.

## 1. Question

Does per-criterion LLM judge accuracy degrade as the number of criteria scored in
a single call increases?

## 1a. Synthetic stratum (operation recovery)

Where human labels cover only five criteria, the packing load is extended with
synthetic corruptions. The synthetic primary estimand is **targeted corruption
recovery**: does the judge order clean vs corrupted text so the targeted score
moves in the seeded direction?

- Conclusions attach to operations, not to abstract criterion constructs.
- Judge ties and wrong-direction decisions count as misses (not dropped).
- Off-target companion movement is a **reported diagnostic**, not a gate that
  removes operations from the primary recovery analysis after seeing the data.
- The completed packing sweep is **exploratory** under any stronger claim that
  each corruption isolates a single human-quality dimension.

Protocol deviation: an earlier draft treated rival off-target movement as
automatic invalidation of synthetic gold. That reading is withdrawn; see
`METHODS.md` §10.0c.

## 2. Primary outcome

**Human stratum.** Binary pairwise correctness per (item, criterion) observation:
does the judge order the two responses on that criterion the same way the human
labels do?

Excluded from the human primary outcome:

- **Human ties** — both responses share a human score, so no correct answer exists.
- **Judge ties** — the judge expressed no preference. Not scored as an error
  (but see co-primary).

**Synthetic stratum.** Targeted corruption recovery on the operated criterion, as
in §1a.

## 3. Co-primary outcome

**Judge tie rate**, modelled with the same design. This is co-primary rather than
secondary because excluding judge ties from the accuracy outcome is only defensible
if the tie rate itself is reported: if packing makes the judge flatter and more
indecisive, that is degradation, and it would otherwise be hidden by the exclusion.

## 4. Secondary outcomes

- QWK against the absolute 0–4 (or 1–5) label. **Promoted to primary for any
  criterion whose resolvable fraction falls below 0.50** (see §7).
- Mean absolute error against the human label.
- Cost: input, output, and reasoning tokens per criterion scored.

## 5. Primary estimand

**Human stratum.** The effect of **actual group size** (not nominal N) on pairwise
accuracy, holding item and criterion constant.

**Synthetic stratum.** The effect of actual group size on **targeted corruption
recovery**, holding base prompt and operation constant (clustered on source
prompt).

## 6. Primary estimator

**Within-item paired contrast.** Per-item accuracy is averaged across that item's
criteria and replicates first; differences are then taken across items, against a
reference of group_size = 1. Reported with Holm correction across the group-size
ladder.

Secondary estimators, always reported alongside:

1. Logistic regression, cluster-robust SEs clustered on item:
   `correct ~ C(group_size) + position`.
2. Crossed mixed-effects logistic: `correct ~ group_size + position + (1|item) + (1|criterion)`.

**Naive standard errors are not reported anywhere.**

## 7. Thresholds fixed in advance

| Threshold | Value | Set by |
|---|---|---|
| Gate 0 headroom floor (group_size = 1 accuracy) | **0.62** | proposal |
| Resolvable-fraction floor before a criterion switches to QWK primary | **0.50** | fixed, see §7a |
| Minimum scoreable criteria per item (inclusion filter) | **4 of 5** | fixed, see §7a |
| Sensitivity re-run at a looser filter | **≥ 3 scoreable** | fixed, see §7a |
| α | 0.05 | fixed |
| Power | 0.80 | fixed |
| Confirmatory MDE | _from pilot_ | TBD |
| Confirmatory n items | _from pilot via Miller's formula_ | TBD |
| Max tolerated failure rate before aborting a run | 25 calls | fixed |
| Every scale point must carry an anchor | enforced in code | fixed, see §7b |

## 7a. Gate 2 result — MEASURED

Run on the full HelpSteer2 train split: 20,324 rows → 10,161 distinct prompts →
**10,160 valid two-response pairs** (1 prompt dropped for not having exactly two
responses).

### Unfiltered population — pairwise outcome is not viable

| Criterion | Human tie rate | Usable n |
|---|---|---|
| helpfulness | 28.9% | 7,219 |
| correctness | 33.1% | 6,793 |
| verbosity | 50.0% | 5,079 |
| coherence | 58.6% | 4,203 |
| complexity | 73.8% | 2,664 |

Only **7.3%** of items have all five criteria scoreable, 27.0% have at least four,
54.7% at least three, and **10.3% have none at all**. The mean item carries just
**2.13 usable criteria of 5**. Three of five criteria breach the 0.50 floor, and
per-item accuracy — the unit the paired estimator operates on — would be averaged
over roughly two criteria. Not viable as specified.

### With the ≥ 4-scoreable filter — viable, on a narrower population

Filtering to items with at least 4 non-tied criteria leaves **2,741 items (27.0%)**,
and every criterion becomes pairwise-usable:

| Criterion | Human tie rate (filtered) | Usable n |
|---|---|---|
| helpfulness | 0.9% | 2,715 |
| correctness | 1.7% | 2,694 |
| verbosity | 11.2% | 2,434 |
| coherence | 19.4% | 2,208 |
| complexity | 39.7% | 1,654 |

Mean scoreable per item: **4.27 of 5**. Items contributing nothing: **0**.

**This table is partly tautological and must not be presented as a property of the
criteria.** Selecting items on "at least 4 criteria are non-tied" mechanically
drives down per-criterion tie rates. The criteria did not improve; the population
changed. What the filter legitimately buys is a usable primary estimator; what it
costs is generalisability.

### Decisions fixed

1. **Inclusion filter: ≥ 4 scoreable criteria** (2,741 items). Legitimate because
   it depends only on the human labels, is applied before any judging, and is
   orthogonal to group size — every surviving item is still judged at every group
   size, so it **cannot bias the group_size contrast**, which is the estimand.

2. **Pairwise remains the primary outcome for all five criteria** on the filtered
   population. The QWK fallback specified in §4 is retained as a secondary and
   reported, but is not needed as primary. Note that complexity at 39.7% is the
   weakest and closest to the floor.

3. **Stated generalisation limit.** Items whose two responses differ clearly on
   most criteria are plausibly easier to judge, so **absolute accuracy will read
   high. The curve's shape transfers; its level does not.** Any headline number
   must carry the filter with it.

4. **Sensitivity analysis at ≥ 3 scoreable** (5,553 items, mean ~3.5 usable).
   Required, not optional: the ≥ 4 filter is the single most consequential
   discretionary choice in the design, and the finding should not depend on it. If
   the group_size effect differs in sign or loses significance at ≥ 3, that is
   reported as the headline caveat.

5. **The packing load and the measured set are decoupled.** All five criteria stay
   in the prompt, because testing group_size = 5 requires five criteria to pack.
   This is a design feature, not a workaround: the criteria constituting the
   packing burden need not be the criteria you can measure. If complexity proves
   uninformative in the discriminability check, it remains as treatment load.

**Power.** The eligible pool of 2,741 clears the MDE-0.03 requirement (~950 items
by Miller's formula) with room to spare. Re-derive from the pilot regardless.

**Saturation, to be reported not hidden.** On the unfiltered set, coherence sits at
ceiling for 69.2% of responses with *zero* scores at 0 (mean 3.62 on 0–4);
complexity is the mirror image, compressed into 1–2 (mean 1.85). Both are
low-information dimensions and will surface in the discriminability check.

## 7b. Rubric anchoring — a measured fix, and a residual threat

**Finding.** In the first live run against `gpt-5-mini` (20 items, group_size 1,
200 calls), criteria were anchored only at scale points 0, 2 and 4. The judge then
used *only* 0, 2 and 4 — never 1 or 3 — on all five criteria. Judge score deltas
were distributed `{0: 65, 1: 1, 2: 34}`: it was operating a 3-point scale. It tied
on **66.3%** of human-scoreable pairs, collapsing 100 observations to **29** usable.

A partially anchored scale becomes the scale. This was a prompt-design defect, not
a property of the model.

**Fix.** Every scale point now carries a descriptor, enforced in
`Criterion.__post_init__` (a criterion with gaps raises). The system-prompt line
"use the full scale, the anchors are definitions not suggestions" was also read as
"use exactly the anchors" and was rewritten.

**Result**, same 20 items, same model:

| | Before | After |
|---|---|---|
| Judge tie rate (human-scoreable pairs) | 66.3% | **43.0%** |
| Resolved observations (of 100) | 29 | **49** |
| Scale points used | 3 of 5 | **5 of 5** |
| Judge \|a−b\| distribution | `{0:65, 1:1, 2:34}` | `{0:49, 1:42, 2:7, 3:2}` |
| Pairwise accuracy | 0.931 | 0.857 |

The accuracy *drop* is the healthy direction: 0.931 was computed on 29 heavily
selected observations — only the comparisons obvious enough for a 3-point scale to
separate. 0.857 on 49 observations is the more honest figure, and still clears the
0.62 headroom floor with room to degrade.

**Residual threat, declared: selection into the resolved set is post-treatment.**

At 43%, judge ties remain the largest single source of data loss, and the study's
own hypothesis predicts the tie rate will **rise with group size**. If it does,
then analysing accuracy only on resolved observations conditions on a
post-treatment variable: the surviving comparisons at group_size 5 would be a
more-selected, easier subset than those at group_size 1, and measured accuracy
could rise with packing for purely selectional reasons. This is a collider, not a
nuisance.

Three mitigations, all pre-committed:

1. **Judge tie rate is co-primary** (§3), modelled with the same design. Any
   accuracy result is reported next to its resolution-rate curve, never alone.
2. **Tie-tolerant robustness outcome.** Re-fit on all human-scoreable observations
   using `credit` (judge tie = 0.5), which conditions on nothing post-treatment.
   If the tie-tolerant and resolved-only analyses disagree in sign, the
   resolved-only result is withdrawn.
3. **Resolution rate is checked for dependence on group size before the accuracy
   curve is interpreted.** If resolution varies materially with group size, the
   headline becomes the joint (resolution, accuracy) result rather than accuracy.

**Open improvement, not yet adopted:** decoupling the judge's response scale from
the human label scale (e.g. judging on 0–10 while ground truth stays 0–4) would
cut ties further, since only the *ordering* is needed. Not adopted because it was
not pre-registered before seeing data and it changes the measurement instrument;
recorded here as the first thing to try if 43% proves too lossy in the pilot.

## 8. Functional form

`group_size` is fitted **both** as a factor and as a linear term. An LRT decides
which is headlined: factor if p < 0.05 (there is a cliff to locate), linear
otherwise (report one slope). Both are reported regardless.

## 9. Rival hypothesis, declared in advance

**Position, not count.** Accuracy may degrade with where a criterion sits in the
prompt rather than with how many share the call. The `group_size` coefficient is
reported with and without `position`, plus their interaction. If position absorbs
the effect, the conclusion is reported as an ordering effect. This is not a
robustness check bolted on afterwards; it is a pre-declared alternative
explanation with a different practical recommendation.

## 10. Controls

- **Per-criterion output budget held constant** across group sizes, enforced in
  `budget.py` and pinned by `tests/test_budget.py`. This is the main threat to
  validity: a shared ceiling would fabricate a group-size effect.
- **Reasoning effort fixed** across conditions; reasoning tokens per criterion
  reported to confirm depth is not a hidden mediator.
- **Temperature not lowered.** Miller shows lowering temperature for variance
  reduction can triple minimum variance and inject bias. Variance reduction comes
  from replicates.
- **Truncation is a hard failure**, never silent. A truncated payload at large
  group sizes is precisely the artifact the budget control exists to prevent.
- **Acknowledged, not controlled:** larger groups have genuinely longer prompts.
  Schema token count is recorded as a covariate. This study is therefore not a
  pure attention-division test, and will not be described as one.

## 11. Randomisation

Three levels, all seeded and reproducible:

1. Uniformly random partition of criteria into groups (uniformity verified by
   chi-square tests in `tests/test_partition.py`).
2. Criterion order within each call.
3. Candidate response order.

## 12. Design balance

Replicates are allocated per nominal level to equalise criterion-observations
across group sizes. A flat allocation leaves the largest group sizes — the cells
the study is about — with the least data (imbalance 2.33× at K=5). The default
allocation reduces that to 1.25×.

## 13. Imperfect ground truth

Human labels are noisy. Accuracy is reported **relative to inter-annotator
agreement**, not to 1.0, with a sensitivity analysis on the high-agreement subset.

Declared defence for the primary estimand: label-noise attenuation is
approximately constant across group sizes, so it depresses absolute accuracy
without biasing the group-size contrast. If the confirmatory data shows
agreement varying systematically with anything correlated to group size, this
defence fails and must be withdrawn.

## 14. Stopping rules

- **Gate 0 fails** (no candidate clears the headroom floor): stop. A judge at
  chance cannot demonstrate degradation.
- **Budget-control check fails** (the effect vanishes once per-criterion output
  budget is held constant): stop and report the null. That is the finding.
- **Failures correlate with group size**: stop. Missingness that tracks the
  treatment biases the estimand.

## 15. Replication

The finding is not reported as general until it replicates on SummEval (K=4,
different domain, 5-point scale). JUDGE-BENCH found per-dataset variance across
20 datasets large enough that a single substrate proves little.

## 16. Analysis validation

The estimator suite is validated against planted effects before touching real
data (`tests/test_recovery.py`): it must recover a known degradation and must not
produce a significant curve on a judge with no planted effect.
