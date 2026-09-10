from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "make_release", Path(__file__).parents[1] / "scripts" / "make_release.py"
)
assert _SPEC is not None and _SPEC.loader is not None
make_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_release)


def test_release_metadata_refuses_a_dirty_worktree(monkeypatch):
    monkeypatch.setattr(
        make_release,
        "git",
        lambda *args: " M src/packing/runner.py\n?? tests/test_runner.py",
    )

    with pytest.raises(SystemExit, match="dirty working tree"):
        make_release.require_clean_tree()


def test_release_metadata_accepts_a_clean_worktree(monkeypatch):
    monkeypatch.setattr(make_release, "git", lambda *args: "")
    make_release.require_clean_tree()


def test_release_metadata_rejects_a_bundle_from_another_commit():
    with pytest.raises(SystemExit, match="bundle was built from commit"):
        make_release.require_bundle_commit(
            {"code_commit": "a" * 40},
            "b" * 40,
        )
