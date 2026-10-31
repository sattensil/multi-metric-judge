# Lessons from building the harness

Things that went wrong or nearly went wrong while assembling this experiment,
kept here so the next judge built on this harness does not repeat them. The
talk covers how the experiment works; this file covers what it took to get there.

## A partially anchored scale becomes the scale

The first rubric described scores 0, 2 and 4 and left 1 and 3 undefined. The
judge used only 0, 2 and 4. Ties on human-scoreable pairs reached 66%, and only
29 of every 100 pairs produced a directional comparison. Describing all five
points brought ties to 43% and usable comparisons to 49 per 100.

That run was discarded rather than pooled. `Criterion.__post_init__` in
`src/packing/criteria.py` now refuses to build a criterion with a missing anchor.

## Randomized order confounded count with position

An earlier sweep shuffled metric order inside each call. Recovery appeared to
fall with group size, but much of that decline was the measured metric landing
further down the list: adding position to the model attenuated the surface-edit
group-size slope by about 75% (`position_analysis.surface_attenuation_fraction`
in `replication/expected.json`). The fixed-position ladder keeps the target
first at every rung so the count effect can be read on its own.

## Output budget has to scale with the payload

A fixed output ceiling makes large calls truncate, which looks exactly like a
packing effect. `budget_for` in `src/packing/budget.py` gives every metric the
same allowance plus a constant envelope, and `near_cap` flags calls that came
close to the ceiling so the sensitivity analysis can check for a token artifact.

## Check the estimators before the data

The paired and stratified estimators were run on simulated data with known
effects before any retained calls were analyzed. `stratified_endpoint` in
`src/packing/analyze.py` weights each operation equally on purpose: a plain mean
over all prompts would re-weight operations by how many prompts each happened to
contribute.

## Corruptions must fail honestly

An operation that cannot apply (no internal punctuation, no prose sentences)
returns `CorruptionResult.unchanged` and the pair is dropped. Silently returning
the original text would create pairs with no difference and a recorded
direction, which the judge can only get wrong.
