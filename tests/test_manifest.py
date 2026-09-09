"""Manifest invariants: the criterion pool is part of the treatment, so it is
validated and hashed rather than assumed."""

from __future__ import annotations

import pytest

from packing.criteria import Criterion
from packing.manifest import CriterionManifest, manifest_for


def _criterion(key: str) -> Criterion:
    return Criterion(
        key=key,
        label=key.title(),
        description=f"Test criterion {key}.",
        anchors={0: "none", 1: "poor", 2: "fair", 3: "good", 4: "full"},
    )


def test_hash_is_stable_and_order_sensitive():
    a = manifest_for((_criterion("x"), _criterion("y")))
    b = manifest_for((_criterion("x"), _criterion("y")))
    c = manifest_for((_criterion("y"), _criterion("x")))
    assert a.hash == b.hash
    assert a.hash != c.hash, "ordering is part of the manifest identity"
    assert len(a.hash) == 16, "display prefix"
    assert len(a.full_hash) == 64, "persist the full digest"
    assert a.full_hash.startswith(a.hash)


def test_hash_changes_when_anchor_prose_changes():
    a = manifest_for((_criterion("x"),))
    edited = Criterion(
        key="x",
        label="X",
        description="Test criterion x.",
        anchors={0: "none", 1: "poor", 2: "fair", 3: "good", 4: "COMPLETE"},
    )
    assert manifest_for((edited,)).full_hash != a.full_hash


def test_validate_rejects_wrong_count():
    m = manifest_for((_criterion("x"),))
    with pytest.raises(ValueError, match="expected 40 criteria, found 1"):
        m.validate(expect=40)


def test_validate_rejects_duplicate_keys():
    m = CriterionManifest(criteria=(_criterion("x"), _criterion("x")))
    with pytest.raises(ValueError, match="duplicate criterion keys"):
        m.validate(expect=2)


def test_validate_accepts_a_well_formed_pool():
    m = manifest_for(tuple(_criterion(f"c{i}") for i in range(40)))
    m.validate(expect=40)


def test_length_stats_reports_the_distribution():
    m = manifest_for((_criterion("x"), _criterion("yy")))
    stats = m.length_stats
    assert set(stats) == {"min", "max", "mean", "total"}
    assert stats["max"] >= stats["min"]
