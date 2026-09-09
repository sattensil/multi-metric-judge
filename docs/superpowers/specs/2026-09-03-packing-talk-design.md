# 30-minute packing talk: design

**Date:** 2026-09-03  
**Status:** approved  
**Audience:** technical AI/ML practitioners

## Goal

A 30-minute claim-first research talk, in the same dual-view HTML pattern as the APJ AgentControl deck: projector (`index.html`) plus synchronized presenter notes (`presenter.html`). The talk presents the original research question, this repo’s protocol, and the completed GPT-5-mini sweep under the **operation-recovery** estimand.

## Narrative

Claim, then experiment, evidence, limits, decision. 19 full-screen slides, about 28 minutes plus Q and A.

The claim leads, because the evidence supports it: the K = 5 human-stratum pilot
carries a tight clustered interval (+0.005, 95% CI −0.034 to +0.043, n = 144),
while the K = 20 sweep's MDE is 13 to 14 points and nothing in it survives Holm.
Opening on the weaker result would mean building to a reveal the next three
slides have to withdraw.

1. Title / question  
2. You pay to read the input every time  
3. Two hypotheses, both plausible  
4. Why not judge versus judge  
5. **The answer: at five criteria, packing is free**  
6. What is held constant  
7. The tie rate is co-primary, and it is flat  
8. Estimators tested against a known answer  
9. Anchor every scale point, or the judge invents a shorter one  
10. Your difficulty classification will be wrong  
11. Detection rate hides two different failures  
12. Past the labelled ceiling: twenty criteria by corruption  
13. How far it pushes: the K = 20 sweep  
14. **Position, not count**, the K = 20 rival adjudicated  
15. **At forty, it does break**, the K = 40 contrast  
16. But three operations carry it  
17. What this still cannot say  
18. What I will actually claim  
19. If you run judges in production  

Slides 9 to 11 carry measured methodology findings (§3.3, §8.2). Slides 12 and 13
present the synthetic sweep as the extension it is. The K = 20 sweep is shown and
discounted in the same breath, since its movements sit inside the band it can
resolve.

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
