# 30-minute packing talk — design

**Date:** 2026-09-03  
**Status:** approved  
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
