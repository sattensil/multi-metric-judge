"""The corruption registry, importable without pulling in the CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from packing.registry import ALL_CORRUPTIONS, CORRUPTION_KEYS, corruption_for


def test_registry_holds_the_eighteen_surviving_operations():
    # METHODS 3.4 catalogues twenty. Two were removed after calibration
    # (METHODS 10, `unsupport` on discrimination and `vague` for inverting), so
    # the live catalogue is eighteen. `jargon` stays in the pool as packing load
    # and is excluded from measurement by MUTE_OPERATIONS rather than removed.
    assert len(ALL_CORRUPTIONS) == 18
    assert len(CORRUPTION_KEYS) == 18
    assert "unsupport" not in CORRUPTION_KEYS
    assert "vague" not in CORRUPTION_KEYS
    assert "jargon" in CORRUPTION_KEYS


def test_lookup_by_key():
    assert corruption_for("shuffle").targeted_criterion == "coherence"
    assert corruption_for("numbers").targeted_criterion == "correctness"


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="nope"):
        corruption_for("nope")


def test_importing_the_registry_does_not_import_the_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import packing.registry, sys; "
            "sys.exit(1 if 'packing.cli' in sys.modules else 0)",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, "registry must not pull in the CLI layer"
