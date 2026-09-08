# Packing talk: 30 minutes

Two self-contained HTML files. No build step, no bundler, no charting library,
no data fetched at runtime: every number on stage is baked into the markup, so
the deck presents fine on a hostile conference network or none at all. Webfonts
are the one external reference; they load non-blocking, with a system fallback
stack behind each, so a dead network costs you the typeface and nothing else.

| File | Role |
|---|---|
| `index.html` | Projector view, 16 full-screen slides |
| `presenter.html` | Presenter view: say/direction notes, next slide, 30-minute timer, pace |

## Presenting

The two views talk over `BroadcastChannel("deck-sync-packing-v1")`, which is
scoped to a browser origin. `file://` URLs do **not** share an origin, so serve
the directory over HTTP or the views will not sync:

```bash
python3 -m http.server 8000 --directory presentation
```

Then open `http://localhost:8000/index.html` on the projector and
`http://localhost:8000/presenter.html` on the laptop screen. Navigating either
one moves the other; scrolling the projector view counts as navigating. The
presenter view's indicator reads `synced` once it has heard from the projector.

Either view works alone if sync is unavailable.

## Keys

| Key | Effect |
|---|---|
| `→` `↓` `space` `enter` `page down` | Next slide |
| `←` `↑` `page up` | Previous slide |
| `home` / `end` | First / last slide |
| `T` | Start or pause the timer *(presenter view)* |
| `R` | Reset the timer *(presenter view)* |

The pace readout compares elapsed time against the slide's target window and
reports ahead / on pace / behind, with a ±45s tolerance. Windows total ~26
minutes, leaving ~4 for questions.

## Editing

Slides carry the claim and the number, nothing an audience has to read while
you talk. Every supporting sentence, caveat and source figure belongs in the
presenter notes instead, so adding prose to a slide is almost always the wrong
move.

Slide copy lives in `index.html`; the notes for slide *n* live at index *n-1* of
`TITLES`, `GOALS`, `WINDOWS`, `NOTES` and `ENDS` in `presenter.html`. Those
arrays are positional, so **adding a slide to one file without the other
silently misaligns every note after it**. `tests/test_presentation.py` pins the
alignment. It checks that each slide's `data-title` matches `TITLES` in order,
that all five arrays are the same length, that both views name the same channel,
and that neither view makes a network call at runtime.

```bash
uv run pytest tests/test_presentation.py
```

Webfonts (Sora / Inter / Roboto Mono) are the one external reference. Every
stack has a system fallback, so the deck degrades rather than breaking offline.

## Where the numbers come from

**The headline claim on slide 5 is `METHODS.md` §8.1**, the K = 5 human-stratum
pilot: accuracy, tie rate, cost, tokens per criterion, and the paired contrast
with its clustered interval. This is the best-evidenced result in the study and
the deck now leads with it. Scale-anchoring figures (slide 9) are §3.3;
calibration findings (slides 10-11) are §8.2. Controls, randomisation and the output-budget invariant are
§4.1–4.3. The operation catalogue and detection classes are §3.4; the
calibration results that reduced 20 operations to 17 measured are §8.2. The
operation-recovery estimand and the exploratory label are `PREREGISTRATION.md`
§1a and `METHODS.md` §10.0c. Sweep-level figures (10,880 calls, the design
table, the cost endpoints, the recovery endpoints, Holm outcomes and the MDE)
are `METHODS.md` §8.3, with §8.3.1 recording exactly which of them were
regenerated (the design, reproducible any time via `packing sweep --dry-run`)
and which are transcribed from the original analysis because `runs/` was not
retained. Read §8.3.1 before defending a number on stage.
