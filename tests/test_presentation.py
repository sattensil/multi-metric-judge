"""Consistency pins for the packing talk deck.

The two views are separate HTML files that have to stay in lockstep. The
presenter view indexes its notes by slide number, so a slide added to one file
and not the other silently misaligns every note after it -- a failure that only
shows up on stage. These tests pin that alignment, the sync channel name, and
the offline constraint: the deck must present with no network, so every number
is baked into the markup rather than fetched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DECK = Path(__file__).resolve().parents[1] / "presentation"
SLIDES = 16
CHANNEL = "deck-sync-packing-v1"
TITLE = "More Signal Per Call: How Many Metrics Can One LLM Judge Score Before It Lies?"


def _read(name: str) -> str:
    path = DECK / name
    if name != "index.html" and not path.exists():
        pytest.skip(f"{name} is local-only and absent from this checkout")
    assert path.exists(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def _js_array(src: str, name: str) -> list[str]:
    """Top-level entries of `const <name> = [...]`, as raw source strings.

    Counting commas naively would split inside nested arrays, inside string
    literals, and inside comments, so this tracks bracket depth, string state
    and comment state. Comments matter: an apostrophe in a `// like this one`
    would otherwise open a string that swallows the rest of the array, and this
    file's own style puts prose comments next to these arrays.
    """
    match = re.search(rf"const\s+{name}\s*=\s*\[", src)
    assert match is not None, f"const {name} = [ ... ] not found"

    entries: list[str] = []
    depth, start, quote, escaped = 1, match.end(), "", False
    comment = ""  # "" | "line" | "block"
    i = match.end()
    while i < len(src):
        char = src[i]
        pair = src[i:i + 2]
        if comment == "line":
            if char == "\n":
                comment = ""
        elif comment == "block":
            if pair == "*/":
                comment = ""
                i += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif pair == "//":
            comment = "line"
            i += 1
        elif pair == "/*":
            comment = "block"
            i += 1
        elif char in "\"'`":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
            if depth == 0:
                entries.append(src[start:i])
                break
        elif char == "," and depth == 1:
            entries.append(src[start:i])
            start = i + 1
        i += 1
    else:
        pytest.fail(f"unterminated array literal for {name}")

    return [entry.strip() for entry in entries if entry.strip()]


def _unquote(literal: str) -> str:
    literal = literal.strip()
    assert literal[0] == literal[-1] == '"', f"expected a double-quoted string: {literal[:40]}"
    return literal[1:-1].replace('\\"', '"').replace("\\\\", "\\")


@pytest.fixture(scope="module")
def index() -> str:
    return _read("index.html")


@pytest.fixture(scope="module")
def presenter() -> str:
    return _read("presenter.html")


def test_projector_view_has_the_pinned_slide_count(index: str) -> None:
    assert len(re.findall(r'<section class="slide"', index)) == SLIDES
    for removed in (
        "Why not judge versus judge",
        "What is held constant",
        "Estimators tested against a known answer",
        "Your difficulty classification will be wrong",
        "Detection rate hides two different failures",
        "How far it pushes: the K equals 20 sweep",
        "Position, not count",
    ):
        assert f'data-title="{removed}"' not in index


def test_tie_slide_states_the_plain_language_result(index: str) -> None:
    assert 'data-title="Packing did not increase ties"' in index
    assert "48.00%" in index
    assert "47.96%" in index
    for jargon in ("co-primary", "The trap", "Report it next to accuracy. Always."):
        assert jargon not in index


def test_dataset_and_experiment_design_are_visible_on_slides(index: str) -> None:
    for title in (
        "The data: HelpSteer2",
        "Experiment one: one metric versus five",
        "We built a synthetic corruption dataset",
        "How synthetic corruption supports forty metrics",
    ):
        assert f'data-title="{title}"' in index
    for detail in (
        "10,681 prompts",
        "two responses",
        "five human-scored metrics",
        "No new human labels",
        "operation + target + direction + seed",
        "1 measured target",
        "39 load-only metrics",
    ):
        assert detail in index


def test_deck_does_not_use_the_semantic_surface_taxonomy(
    index: str, presenter: str
) -> None:
    for src in (index, presenter):
        assert not re.search(r"\b(?:semantic|surface)\b", src, re.I)


def test_projector_is_anchored_on_the_title_question(index: str) -> None:
    assert f"<title>{TITLE}</title>" in index
    assert f'data-title="{TITLE}"' in index


def test_presenter_arrays_are_all_slide_length(presenter: str) -> None:
    for name in ("TITLES", "NOTES", "GOALS", "WINDOWS", "ENDS"):
        assert len(_js_array(presenter, name)) == SLIDES, f"{name} is not {SLIDES} long"


def test_presenter_titles_match_slide_titles(index: str, presenter: str) -> None:
    """The alignment that actually matters: note *n* must describe slide *n*."""
    on_slides = re.findall(r'<section class="slide"[^>]*data-title="([^"]*)"', index)
    in_notes = [_unquote(entry) for entry in _js_array(presenter, "TITLES")]
    assert on_slides == in_notes


def test_title_question_anchors_both_views(index: str, presenter: str) -> None:
    assert f"<title>{TITLE}</title>" in index
    assert f'data-title="{TITLE}"' in index
    assert _unquote(_js_array(presenter, "TITLES")[0]) == TITLE


def test_slide_timings_are_contiguous_and_land_at_thirty_minutes(
    presenter: str,
) -> None:
    windows = [_unquote(entry) for entry in _js_array(presenter, "WINDOWS")]
    parsed = [
        tuple(
            int(minutes) * 60 + int(seconds)
            for minutes, seconds in re.findall(r"(\d+):(\d{2})", window)
        )
        for window in windows
    ]
    assert all(len(window) == 2 for window in parsed)
    assert parsed[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(parsed, parsed[1:]))
    assert parsed[-1][1] == 30 * 60

    ends = [int(entry) for entry in _js_array(presenter, "ENDS")]
    assert ends == [window[1] for window in parsed]


def test_presenter_talk_track_stays_in_short_beats(presenter: str) -> None:
    for i, entry in enumerate(_js_array(presenter, "NOTES")):
        say = re.search(r"say:\s*\[(.*?)\]\s*,\s*dir:", entry, re.S)
        assert say is not None
        beats = re.findall(r'"(?:\\.|[^"\\])*"', say.group(1))
        assert len(beats) <= 5, f"slide {i + 1} has {len(beats)} speaking beats"


def test_presenter_notes_are_a_full_speakable_script(presenter: str) -> None:
    words = 0
    for entry in _js_array(presenter, "NOTES"):
        say = re.search(r"say:\s*\[(.*?)\]\s*,\s*dir:", entry, re.S)
        assert say is not None
        spoken = " ".join(re.findall(r'"((?:\\.|[^"\\])*)"', say.group(1)))
        spoken = re.sub(r"<[^>]+>", " ", spoken)
        words += len(re.findall(r"\b[\w%]+\b", spoken))
    assert 2_100 <= words <= 2_900, f"presenter script is {words} words"


def test_every_note_carries_say_and_dir(presenter: str) -> None:
    for i, entry in enumerate(_js_array(presenter, "NOTES")):
        assert "say:" in entry, f"note {i} has no say bullets"
        assert "dir:" in entry, f"note {i} has no direction cue"


def test_both_views_share_one_channel(index: str, presenter: str) -> None:
    for src in (index, presenter):
        found = set(re.findall(r'BroadcastChannel\(\s*"([^"]+)"', src))
        assert found == {CHANNEL}, f"expected only {CHANNEL!r}, found {found!r}"


@pytest.mark.parametrize("view", ["index.html", "presenter.html"])
def test_deck_makes_no_network_calls(view: str) -> None:
    """A conference room is a hostile network. Nothing may be fetched at runtime."""
    src = _read(view)
    for pattern in (r"\bfetch\s*\(", r"XMLHttpRequest", r"WebSocket", r"sendBeacon", r"\bimport\s*\("):
        assert not re.search(pattern, src), f"{view} makes a runtime network call: {pattern}"
    assert "<script src=" not in src, f"{view} loads an external script"


@pytest.mark.parametrize("view", ["index.html", "presenter.html"])
def test_only_external_reference_is_fonts_and_it_degrades(view: str) -> None:
    """Webfonts are the one allowed external reference, so every stack needs a fallback."""
    src = _read(view)
    hosts = set(re.findall(r'href="https?://([^/"]+)', src))
    assert hosts <= {"fonts.googleapis.com", "fonts.gstatic.com"}, f"unexpected hosts: {hosts}"
    for family in ("Sora", "Inter", "Roboto Mono"):
        stack = re.search(rf'--font-[a-z]+:\s*"{re.escape(family)}"\s*,\s*([^;]+);', src)
        assert stack is not None, f"{family} has no fallback stack in {view}"
        assert stack.group(1).strip(), f"{family} fallback stack is empty in {view}"


def test_readme_documents_the_channel_and_both_views() -> None:
    readme = _read("README.md")
    for needle in (CHANNEL, "index.html", "presenter.html"):
        assert needle in readme


def _strip_js_comments_and_scan(src: str) -> list[int]:
    """Offsets where a string literal is immediately followed by another one.

    Two adjacent string literals with no comma between them is a syntax error
    inside an array literal, and it is the exact failure a hand-edited notes
    array invites: replace a block of bullets, forget the comma on the last
    surviving line. `_js_array` cannot see it, because it only splits at depth
    1 and the notes live nested inside a `say:` array, so the whole presenter
    script silently fails to parse and the operator gets no notes and no
    controls. Nothing else in this suite executes the JavaScript.
    """
    offenders: list[int] = []
    i, quote, escaped, comment = 0, "", False, ""
    end_of_string = -1
    while i < len(src):
        char, pair = src[i], src[i:i + 2]
        if comment == "line":
            if char == "\n":
                comment = ""
        elif comment == "block":
            if pair == "*/":
                comment = ""
                i += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
                end_of_string = i
        elif pair == "//":
            comment = "line"
            i += 1
        elif pair == "/*":
            comment = "block"
            i += 1
        elif char in "\"'`":
            if end_of_string >= 0 and not src[end_of_string + 1:i].strip():
                offenders.append(i)
            quote = char
        elif not char.isspace():
            end_of_string = -1
        i += 1
    return offenders


@pytest.mark.parametrize("view", ["index.html", "presenter.html"])
def test_inline_script_has_no_adjacent_string_literals(view: str) -> None:
    src = _read(view)
    for block in re.findall(r"<script[^>]*>(.*?)</script>", src, re.S):
        offenders = _strip_js_comments_and_scan(block)
        assert not offenders, (
            f"{view}: string literal followed by another with no comma at "
            f"offsets {offenders}; the script will not parse"
        )


def test_the_deck_reports_the_published_ladder_numbers() -> None:
    """The claim slide is the one people photograph, so it may not drift.

    Pinned against `replication/expected.json` rather than a literal, so the
    deck and the published claims cannot disagree: a re-pin that moves theta
    fails here until the slide and the notes are moved with it.
    """
    expected = json.loads(
        (Path(__file__).resolve().parents[1] / "replication" / "expected.json").read_text()
    )
    ladder = expected["claims"]["fixed_position_ladder"]
    index = _read("index.html")

    for value in (ladder["theta"], ladder["ci_low"], ladder["ci_high"]):
        rendered = f"{abs(value) * 100:.1f}"
        assert rendered in index, f"{rendered} (from {value}) is not on a slide"

    calls = f"{ladder['successful_calls']:,}"
    assert calls in index, f"successful call count {calls} is not on a slide"
    presenter_path = DECK / "presenter.html"
    if presenter_path.exists():
        presenter = presenter_path.read_text(encoding="utf-8")
        assert calls in presenter, f"successful call count {calls} is not in the notes"

    superseded = ("&minus;7.7", "6,289", "p = 0.0016")
    for stale in superseded:
        assert stale not in index, f"superseded value {stale!r} is still on a slide"
