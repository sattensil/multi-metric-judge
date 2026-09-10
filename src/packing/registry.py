"""The corruption catalogue, in one neutral place.

Eligibility screening and dataset construction both need the catalogue. Reaching
it through the command-line module would make data code depend on the CLI, so it
lives here and `cli.py` imports it like everything else.
"""

from __future__ import annotations

from .corrupt import CORRUPTIONS, Corruption
from .corrupt_extra import EXTRA_CORRUPTIONS

ALL_CORRUPTIONS: tuple[Corruption, ...] = CORRUPTIONS + EXTRA_CORRUPTIONS
CORRUPTION_KEYS: frozenset[str] = frozenset(c.key for c in ALL_CORRUPTIONS)
_BY_KEY: dict[str, Corruption] = {c.key: c for c in ALL_CORRUPTIONS}


def corruption_for(key: str) -> Corruption:
    if key not in _BY_KEY:
        raise KeyError(f"unknown corruption {key!r}; known: {sorted(_BY_KEY)}")
    return _BY_KEY[key]
