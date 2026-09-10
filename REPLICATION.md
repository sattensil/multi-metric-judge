# Replication

This repository contains **two experiments at different stages**. They should not
be read as one completed study.

Three activities, named separately because they answer different questions:

| Activity | What it establishes | Needs an API key |
|---|---|---|
| **Review** | What was done, and which claims the evidence supports | No |
| **Recompute** | The published tables follow from the retained calls. Audits the *analysis* | No |
| **Rerun** | The result survives fresh nondeterministic draws. Tests *procedural robustness* | Yes |

Recomputation is not a weaker form of rerunning. It is a different claim, and
neither alone proves the other. Both have a shared blind spot, stated below.

## Status of the evidence

| Result | Status | How to verify it |
|---|---|---|
| K = 5 human pilot (METHODS §8.1) | **Complete** | `packing reproduce`, from `pilot` |
| K = 20 sweep (METHODS §8.3) | **Complete.** The observed contrast stands; its reading as a load-only effect is not identified, because position moved with K | `packing reproduce`, from `sweep` |
| K = 20 position decomposition (METHODS §8.3.2) | **Complete, exploratory.** Not pre-specified, not corrected for multiplicity | `packing reproduce`, from `sweep` |
| Scale anchoring (METHODS §3.3) and calibration gate (§8.2) | **Complete** | `packing calibration-report --run calib` |
| K = 40 fixed-position ladder, rungs K = 1, 30, 40 (METHODS §8.5) | **Complete.** Pre-registered before execution. Primary contrast significant; the trend across rungs (§8.5.1) and the per-operation effects are exploratory, and the per-operation effects are underpowered | `packing reproduce`, then `packing ladder-analyze --run ladder-k40` |

The K = 20 sweep is qualified, not retired. Position is nested inside the call,
so `E[position | K] = (K-1)/2` exactly and support is triangular: the recovery
figures are real, and attributing the movement to load alone is what the design
cannot support. The K = 40 ladder exists to remove that confound by construction,
and its own estimand is deliberately narrower still: appending 39 criteria behind
a target fixed at position 0, which does not separate load from recency or from
distance to the end of context.

## Recompute, in one command

No API key, no spend.

```bash
uv sync --frozen
uv run packing reproduce
```

Against a downloaded bundle, verifying checksums first:

```bash
uv run packing reproduce --sums SHA256SUMS --root .
```

It prints **PASS**, **MISMATCH** with the disagreeing values named, or
**UNAVAILABLE** with the absent artifacts named. Counts and identifiers are
compared exactly; only floats carry a tolerance, and only for last-decimal
library drift. Expected values live in `replication/expected.json` and are
regenerated only when an analysis *definition* changes, never to make a failing
check pass.

## Claim-to-artifact map

| Claim | Retained input | Command |
|---|---|---|
| §8.1 pilot accuracy, tie rate, paired contrast, MDE | `runs/pilot/` | `packing reproduce` |
| §3.3 scale anchoring, §8.2 calibration verdicts | `runs/calib/`, `runs/calib2/`, `runs/ladder/` | `packing calibration-report --run calib` |
| §8.3 sweep recovery, tie rates, discrimination | `runs/sweep/` | `packing reproduce` |
| §8.3.2 position decomposition, attenuation | `runs/sweep/` | `packing reproduce` |
| §8.5 to §8.5.4 K = 40 ladder: primary contrast, trend slope, numbers sentinel | `runs/ladder-k40/` | `packing reproduce`, and `packing ladder-analyze --run ladder-k40` for the diagnostics |

**A naming collision to be aware of.** `runs/ladder/` is the *severity* ladder
from calibration. The K = 40 fixed-position ladder is `runs/ladder-k40/`. Two
unrelated things called "ladder"; the on-disk names are unchanged so existing
commands keep working.

**A definition to be aware of.** The published judge tie rate (0.4800 → 0.4796)
is over human-**scoreable** pairs, so the denominator excludes human ties.
Computing it over all outcomes gives 0.5123 → 0.5033. §8.1 says "judge tie rate"
without naming the denominator, and `packing reproduce` pins the scoreable one.

## The pinned frames

The completed runs are reproducible only against the exact frame, not against the
fetch recipe, because a hosted dataset need not return the same records forever.

There are two. The pilot, the sweep and the calibration runs were drawn from a
200-item frame; the K = 40 ladder needed a larger eligible pool and was drawn
from a 1,600-item one. Both ship in the bundle, and each run's `meta.json`
records the digest of the frame it was actually built against.

| Field | Pilot, sweep, calibration | K = 40 ladder |
|---|---|---|
| Path | `data/helpsteer2.jsonl` | `data/helpsteer2-ladder.jsonl` |
| Items | 200 | 1,600 |
| `dataset_sha256` | `a4b0101e1dbbd06f27e266a8047cbb88871d0b18b2a3ff775b8233889c19f5b6` | `d7df63ab1e61de5608ff6a5deee92ef9dabb628319cd1665f4988dfda44df9ae` |
| `ordered_item_id_sha256` | `756c1d1a432f09266e495dd7c3b125cfc7ec1d242aff514e52ff7addbfa6b563` | `234f0ecb8363787a8e8a1d6516344ba3de0e5d36c5c1e63e910b2fa80fb46c12` |
| Source | `nvidia/HelpSteer2`, CC-BY-4.0 | `nvidia/HelpSteer2`, CC-BY-4.0 |
| Fetch recipe | `packing fetch --limit 200` | `packing fetch --limit 1600` |

The ordered-id digest is recorded separately because two files can hold identical
examples in a different order. Seeded assignment here consumes a sorted
permutation, so order does not change this sample, but a replicator cannot tell
the two files apart without it.

Redistribution is permitted under CC-BY-4.0 with attribution, so both frames ship
inside the bundle rather than being available on request.

The bundle also carries the sampling plans under `plans/`. A frame says which
prompts were available and a run's `items.jsonl` says what was built from them;
only the plan says which units were drawn, with which operation, side and seed.
With all three a replicator can check the sample against the design rather than
only recomputing the published numbers:

```bash
python - <<'EOF'
from pathlib import Path
from packing.sampling_plan import read_plan
from packing.live import rebuild_items
plan = read_plan(Path("plans/ladder-plan-k30.json"))
items, specs = rebuild_items(plan, Path("data/helpsteer2-ladder.jsonl"))
print(len(items), "units", len(specs), "calls")
EOF
```

`rebuild_items` verifies the frame and manifest digests first and refuses on a
mismatch, so a wrong frame fails loudly instead of rebuilding a different
sample. The K = 40 ladder was collected in two passes and both plans are
included; the union of their call identities is exactly the 7,776 planned
calls.

## The replication bundle

```bash
uv run packing bundle --out dist --version v5
```

Builds `dist/packing-replication-v5.tar.gz`, 16.9 MB, containing both pinned frames, the sampling plans,
the retained call records for every run that backs a published claim, the frozen
expected values, `uv.lock`, a `MANIFEST.json` recording the code commit and
request settings, and `SHA256SUMS` over every file. Building uploads nothing;
publishing is a separate deliberate step.

`replication/RELEASE.json` is the committed pointer: release identifier, archive
SHA-256 `64857357ce85726c`, a digest for all 27 files, the code commit, model and
request settings, the per-section commands, and the approximate call date range.
`dist/` itself is gitignored, so 40 MB of append-only call records never enter
clone history; the archive is a release asset.

## Unpacking the bundle

The CLI resolves a run name to `runs/<name>/`, so the archive has to land there:

```bash
tar xzf packing-replication-v5.tar.gz
cp -R packing-replication-v5/runs/. runs/
cp packing-replication-v5/data/*.jsonl data/
uv run packing reproduce
```

`runs/*` is gitignored, which is why the artefacts travel as a release asset
rather than in the tree. A run directory copied anywhere else will not be found:
`packing ladder-analyze --run ladder-k40` reads `runs/ladder-k40/`, not an
arbitrary path.

## Cost figures are lower bounds, not billing

Every dollar figure in `METHODS.md` is computed from the token counts in retained
call records. A provider-level retry is billed in full and never written, since
only the successful attempt reaches the record, so the true spend is higher by
however many retries occurred. On the K = 40 ladder the gap was a factor of 3.4:
10,216,217 input tokens billed against 3,042,025 recorded.

Use the reported costs to compare conditions with each other, which is what they
are for. Do not reconcile them against a provider invoice.

## Fresh reruns are not expected to reproduce exact numbers

The published values are reproducible by **recomputing** them from the retained
call records. They are not reproducible by fresh API execution, and that is not a
defect to apologise for:

- the judge requests specify no sampling seed, deliberately, because lowering
  temperature to stabilise output can raise minimum variance and inject bias
  (METHODS §3.1);
- `gpt-5-mini` is an **alias**, not a pinned snapshot, so hosted behaviour may
  differ from the behaviour that produced these records.

A fresh rerun produces a new realisation of the experiment. Use it to ask whether
the effect and the analysis survive new draws, not to expect row-for-row or
decimal-for-decimal identity.

### What is not recorded

Honest gaps in the retained records, which limit what a replicator can
reconstruct:

- provider response identifiers;
- per-call wall-clock timestamps;
- a pinned model snapshot.

Recorded per call: identity, item and side, model alias, nominal group size,
replicate, partition hash, group membership and positions, input, output,
reasoning and cached token counts, latency, near-cap flag, scores and rationales.
The K = 40 sampling plan additionally freezes the dataset and criterion-manifest
digests, the seed, and the request settings, so it is better instrumented than
the K = 20 runs. Neither carries response ids.

## What neither recompute nor rerun can audit

**Construction.** Call records that are internally consistent with a *wrong*
experiment recompute perfectly and rerun perfectly.

This is not hypothetical. During the K = 40 build, the corruption builder paired
the corrupted version of response A against the untouched response B, two
different answers, so the contrast would have measured which answer the judge
preferred rather than whether it detected the seeded edit. It also left the
ground-truth labels empty. Both bugs were caught by reading how the existing
synthetic stratum encodes gold, not by any numerical check, and a test that used
identical text for both sides had hidden the first one.

So the reviewable surface is not only the numbers. It is `corrupt.py`,
`corrupt_extra.py`, `prompt.py`, the operation catalogue in METHODS §3.4, and
the criterion definitions with their anchors.

## Where the design record lives

- `METHODS.md` states the protocol as it now stands, with §10 recording
  pre-specification status: which claims are confirmatory, which exploratory.
- `PREREGISTRATION.md` is the pre-registration for the **confirmatory** run, frozen 2026-09-03. It governs neither the pilot nor the K = 20 sweep, and the K = 40 ladder has its own frozen protocol in `docs/superpowers/specs/2026-09-08-load-ladder-design.md`.
- `docs/superpowers/specs/` holds designs; `docs/superpowers/plans/` holds
  implementation plans.

Protocol documents describe the current design. Evidence status lives here.
