"""The replication layer: checksum verification, comparison semantics, and the
frame-provenance digests that pin the completed runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packing.replication import (
    EXACT_KEYS,
    compare,
    file_sha256,
    ordered_item_id_sha256,
    verify_checksums,
)


def _frame(path: Path, ids: list[str]) -> None:
    path.write_text(
        "\n".join(json.dumps({"item_id": i, "prompt": "p"}) for i in ids) + "\n",
        encoding="utf-8",
    )


def test_float_drift_within_tolerance_passes():
    results = compare({"s": {"x": 0.80513}}, {"s": {"x": 0.8051}})
    assert all(r.ok for r in results)


def test_float_drift_beyond_tolerance_fails():
    results = compare({"s": {"x": 0.8060}}, {"s": {"x": 0.8051}})
    assert not results[0].ok


def test_counts_are_compared_exactly_with_no_tolerance():
    # A count off by one is a different dataset, not library drift.
    assert "n_items" in EXACT_KEYS
    results = compare({"s": {"n_items": 145}}, {"s": {"n_items": 144}})
    assert not results[0].ok


def test_a_missing_value_fails_rather_than_being_skipped():
    results = compare({"s": {}}, {"s": {"x": 0.5}})
    assert not results[0].ok
    assert results[0].actual is None


def test_comparison_reports_the_offending_key():
    results = compare({"s": {"x": 1.0}}, {"s": {"x": 0.0}})
    assert "s.x" in results[0].line
    assert "FAIL" in results[0].line


def test_ordered_item_id_digest_is_order_sensitive(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _frame(a, ["p1", "p2", "p3"])
    _frame(b, ["p3", "p2", "p1"])
    # Same examples, different order. The content digests differ too, but the
    # ordered-id digest is what names the reason.
    assert ordered_item_id_sha256(a) != ordered_item_id_sha256(b)
    assert len(ordered_item_id_sha256(a)) == 64


def test_ordered_item_id_digest_is_stable(tmp_path):
    path = tmp_path / "a.jsonl"
    _frame(path, ["p1", "p2"])
    assert ordered_item_id_sha256(path) == ordered_item_id_sha256(path)


def test_checksums_pass_on_untouched_files(tmp_path):
    payload = tmp_path / "calls.jsonl"
    payload.write_text('{"identity":"x"}\n', encoding="utf-8")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{file_sha256(payload)}  calls.jsonl\n", encoding="utf-8")
    assert verify_checksums(sums, tmp_path) == []


def test_checksums_detect_a_modified_file(tmp_path):
    payload = tmp_path / "calls.jsonl"
    payload.write_text('{"identity":"x"}\n', encoding="utf-8")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{file_sha256(payload)}  calls.jsonl\n", encoding="utf-8")
    payload.write_text('{"identity":"tampered"}\n', encoding="utf-8")
    problems = verify_checksums(sums, tmp_path)
    assert len(problems) == 1
    assert "digest mismatch" in problems[0]


def test_checksums_detect_an_absent_file(tmp_path):
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'0' * 64}  gone.jsonl\n", encoding="utf-8")
    problems = verify_checksums(sums, tmp_path)
    assert problems == ["missing: gone.jsonl"]


def test_the_frozen_expectations_file_is_well_formed():
    payload = json.loads(Path("replication/expected.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    claims = payload["claims"]
    assert {"methods_8_1", "methods_8_3", "methods_8_3_2"} <= set(claims)

    # The published pilot figures, pinned. If these drift, either the analysis
    # definition changed or something is wrong; either way it is not a silent edit.
    assert claims["methods_8_1"]["accuracy_k1"] == 0.8051
    assert claims["methods_8_1"]["accuracy_k5"] == 0.8039
    assert claims["methods_8_1"]["judge_tie_rate_scoreable_k1"] == 0.48
    assert claims["methods_8_1"]["n_items"] == 144
    assert claims["methods_8_3"]["calls"] == 10880
    assert claims["methods_8_3"]["surface_recovery_k20"] == 0.6833
    assert claims["methods_8_3_2"]["surface_attenuation_fraction"] == 0.7467


def test_the_tie_rate_denominator_is_the_scoreable_one():
    # Guard against a future refactor silently switching denominators: the
    # published 0.4800 is over human-scoreable pairs, and computing it over all
    # outcomes gives 0.5123.
    payload = json.loads(Path("replication/expected.json").read_text(encoding="utf-8"))
    key = "judge_tie_rate_scoreable_k1"
    assert key in payload["claims"]["methods_8_1"]
    assert payload["claims"]["methods_8_1"][key] != 0.5123


def test_every_computed_claim_is_either_pinned_or_declared_pending():
    """A computed claim must not silently stop being checked.

    Dropping a claim from `expected.json` used to be invisible: `reproduce`
    would pass on whatever remained. Anything the code computes now has to be
    either pinned or listed under `pending` with a reason, so removing a pin is
    a deliberate act rather than an omission.
    """
    from packing import cli
    from packing import replication as rep

    computed = set(rep.compute_claims(cli._load_run))
    document = json.loads(
        Path("replication/expected.json").read_text(encoding="utf-8")
    )
    accounted = set(document["claims"]) | set(document.get("pending", {}))
    unaccounted = computed - accounted
    assert not unaccounted, (
        f"computed but neither pinned nor declared pending: {sorted(unaccounted)}"
    )
