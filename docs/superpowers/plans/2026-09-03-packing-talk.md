# Packing Talk Implementation Plan

> **For agentic workers:** Implement inline in this session. User already approved the design.

**Goal:** Ship a 16-slide 30-minute talk with regular and presenter views, synced like the APJ deck.

**Architecture:** Two self-contained HTML files plus a README. Keyboard nav and `BroadcastChannel("deck-sync-packing-v1")` copy the APJ protocol. Slide copy is frozen to the approved narrative and sweep numbers.

**Tech Stack:** HTML, CSS, vanilla JS, Google Fonts (Sora/Inter/Roboto Mono). No Chart.js.

**Spec:** `docs/superpowers/specs/2026-09-03-packing-talk-design.md`

## Global Constraints

- 19 slides in `index.html` matching 19 `TITLES` / `NOTES` / `GOALS` / `WINDOWS` / `ENDS` in `presenter.html`
- Channel name exactly `deck-sync-packing-v1`
- Exploratory operation-recovery framing; no Holm-significant packing-degradation claim
- Numbers from completed sweep reanalysis; no API calls
- No LaunchDarkly brand assets

## File map

- Create: `presentation/index.html`
- Create: `presentation/presenter.html`
- Create: `presentation/README.md`
- Create: `tests/test_presentation.py`

### Task 1: Consistency test then both views

**Files:**
- Create: `tests/test_presentation.py`
- Create: `presentation/index.html`
- Create: `presentation/presenter.html`
- Create: `presentation/README.md`

- [x] Pin slide/channel invariants
- [x] Build regular + presenter views
- [x] Document how to present

Do not commit unless the user asks.
