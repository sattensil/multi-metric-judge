# 30-minute packing talk: design

**Date:** 2026-09-03  
**Status:** approved; **narrative revised 2026-09-07** (see Revision below)  
**Audience:** technical AI/ML practitioners

## Goal

A 30-minute claim-first research talk, in the same dual-view HTML pattern as the APJ AgentControl deck: projector (`index.html`) plus synchronized presenter notes (`presenter.html`). The talk presents the original research question, this repo’s protocol, and the completed GPT-5-mini sweep under the **operation-recovery** estimand.

## Narrative

Claim → experiment → evidence → limits → decision. About 16 full-screen slides, ~26 minutes plus ~4 minutes Q&A.

1. Title / question  
2. You pay to read the input every time  
3. Two hypotheses (cost save vs divided attention)  
4. Why not judge-vs-judge  
5. HelpSteer2’s five-criterion ceiling and synthetic K=20  
6. Estimand: directional recovery of known corruption operations  
7. Controls: same criteria, random partitions/order/sides, per-criterion output budget  
8. Sweep scale: 10,880 calls, `gpt-5-mini`, K=1–20  
9. Cost curve (~$0.61 → ~$0.22 per 1k criteria, uncached)  
10. Recovery curves (semantic 70.5%→72.0%; surface 78.8%→68.3%)  
11. Inference: neither K=20 contrast survives Holm  
12. Rival: surface drop is largely positional  
13. Leakage as construct-overlap context, not exclusion  
14. Power / MDE ~13–14 pts; exploratory  
15. Defensible claim  
16. Production takeaway + Q&A  

Human-stratum K=5 pilot (flat accuracy, ~half cost) is a supporting beat, not a separate act.

## Visual system

Original dark research aesthetic inspired by APJ’s clarity (Sora / Inter / Roboto Mono, lime/cyan accents, large type). No LaunchDarkly logos, mascots, or brand background JPEGs. CSS-only charts (no Chart.js). Reduced-motion supported.

## Interaction

- Regular view: scroll-snap slides, dots, keyboard (arrows / space / home / end).  
- Presenter view: current/next, goal tag, clock window, 30-minute timer, say/dir notes.  
- Sync: `BroadcastChannel("deck-sync-packing-v1")`, same ping/nav protocol as APJ.  
- Serve over HTTP for dual-window sync.

## Files

- `presentation/index.html`  
- `presentation/presenter.html`  
- `presentation/README.md`  

No new provider calls. Numbers come from the completed `runs/sweep` reanalysis and METHODS/README.

## Claims on stage

- Conclusions attach to **operations**, not abstract criteria.  
- Sweep is **exploratory** for any stronger isolation claim.  
- Ties and wrong directions are misses.  
- Clustering unit is source prompt.  
- Do not claim Holm-significant packing degradation.

## Validation

- Test pins: equal slide/note counts, shared channel name, no network fetches in the deck.  
- Manual: open both views, keyboard nav, desktop and narrow layout.  


---

## Revision, 2026-09-07: claim-first reordering

The narrative above led with the K = 20 synthetic sweep and demoted the K = 5
human-stratum pilot to "a supporting beat". That inverted the evidence: the
pilot carries a tight clustered interval (+0.005, 95% CI −0.034 to +0.043, n =
144 items) while the sweep's MDE is 13–14 points and nothing in it survives
Holm. Presenting the weaker result first meant building to a reveal that the
next three slides had to withdraw.

The deck was reordered so the claim matches the evidence:

1. Title / question
2. You pay to read the input every time
3. Two hypotheses, both plausible
4. Why not judge versus judge
5. **The answer: at five criteria, packing is free**, the claim, up front
6. What is held constant
7. The tie rate is co-primary, and it is flat
8. Estimators tested against a known answer
9. **Anchor every scale point, or the judge invents a shorter one**
10. **Your difficulty classification will be wrong**
11. **Detection rate hides two different failures**
12. Past the labelled ceiling: twenty criteria by corruption
13. How far it pushes: the K = 20 sweep
14. What this study could not have seen
15. What I will actually claim
16. If you run judges in production

Slides 9–11 are new, promoting measured methodology findings (§3.3, §8.2) that
the original narrative omitted entirely. Slides 12–13 compress the old 5–10 into
the extension they actually are. The old slide 12 asserted the surface decline
was "largely positional"; nothing in the repo supports that, so it is now
presented as an unadjudicated rival and folded into slide 14.

Unchanged: the visual system, the dual-view architecture, the sync channel, and
every claim-discipline constraint under "Claims on stage".
