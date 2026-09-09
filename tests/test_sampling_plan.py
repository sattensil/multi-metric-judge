"""The sampling plan is the frozen record the live run consumes."""

from __future__ import annotations

import pytest

from packing.sampling_plan import (
    SamplingPlan,
    Unit,
    file_sha256,
    read_plan,
    write_plan,
)


def _plan(units=None):
    return SamplingPlan(
        dataset_id="helpsteer2",
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
        seed=20260908,
        model="gpt-5-mini",
        model_settings={"reasoning_effort": "low", "temperature": None},
        levels=(1, 5, 10, 20, 40),
        units=units
        or (
            Unit(
                base_item_id="hs2-000001",
                lane="core",
                side="b",
                eligible=True,
                operation="shuffle",
                realisation_seed=123,
                companion_seed=456,
                status="active",
            ),
        ),
        call_identities=("hs2-000001|b|gpt-5-mini|0|abc|0",),
    )


def test_round_trips_through_disk(tmp_path):
    path = tmp_path / "plan.json"
    write_plan(path, _plan())
    assert read_plan(path) == _plan()


def test_hashes_are_full_sha256(tmp_path):
    path = tmp_path / "plan.json"
    write_plan(path, _plan())
    loaded = read_plan(path)
    assert len(loaded.dataset_sha256) == 64
    assert len(loaded.manifest_sha256) == 64


def test_a_truncated_digest_is_rejected():
    with pytest.raises(ValueError, match="must be a full SHA-256"):
        SamplingPlan(
            dataset_id="x",
            dataset_sha256="short",
            manifest_sha256="b" * 64,
            seed=1,
            model="mock",
            model_settings={},
            levels=(1,),
            units=(),
        )


def test_file_sha256_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("x\n", encoding="utf-8")
    b.write_text("y\n", encoding="utf-8")
    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)
    assert len(file_sha256(a)) == 64


def test_active_and_reserve_units_are_distinguished():
    plan = _plan(
        units=(
            Unit("p1", "core", "a", True, "shuffle", 1, 2, "active"),
            Unit("p2", "core", "b", True, "filler", 3, 4, "reserve"),
        )
    )
    assert [u.status for u in plan.units] == ["active", "reserve"]
    assert plan.active_count == 1


def test_duplicate_base_items_are_rejected():
    with pytest.raises(ValueError, match="duplicate base item"):
        _plan(
            units=(
                Unit("p1", "core", "a", True, "shuffle", 1, 2, "active"),
                Unit("p1", "numbers", "b", True, "numbers", 3, 4, "active"),
            )
        )


def test_a_unit_may_belong_to_only_one_arm():
    # Enforced by the duplicate check, stated separately because it is the design
    # constraint rather than an implementation detail.
    with pytest.raises(ValueError, match="duplicate base item"):
        _plan(
            units=(
                Unit("p9", "core", "a", True, "flatten", 1, 2, "active"),
                Unit("p9", "numbers", "a", True, "numbers", 1, 2, "active"),
            )
        )


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="status must be"):
        _plan(units=(Unit("p1", "core", "a", True, "shuffle", 1, 2, "maybe"),))


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError, match="side must be"):
        _plan(units=(Unit("p1", "core", "c", True, "shuffle", 1, 2, "active"),))
