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


def test_a_computed_value_nobody_pinned_is_reported_as_a_gap():
    """`compare` is driven by the expected file, so it cannot see this.

    An unpinned claim is not merely unverified, it is invisibly unverified: the
    run still prints PASS. That happened once here already, when the ladder
    claims were unpinned during collection and the suite stayed green.
    """
    from packing.replication import coverage_gaps

    actual = {
        "methods_8_5": {"theta": -0.0629, "se": 0.0225, "trend_slope": -0.0096},
        "methods_9_9": {"whatever": 1.0},
    }
    expected = {"methods_8_5": {"theta": -0.0629, "se": 0.0225}}

    gaps = coverage_gaps(actual, expected)
    assert any("trend_slope" in gap for gap in gaps), gaps
    assert any("methods_9_9" in gap for gap in gaps), gaps
    assert not any("theta" in gap for gap in gaps), "pinned values are not gaps"


def test_full_coverage_reports_no_gaps():
    from packing.replication import coverage_gaps

    values = {"theta": -0.0629, "se": 0.0225}
    assert coverage_gaps({"methods_8_5": values}, {"methods_8_5": values}) == []


def test_the_shipped_expected_file_pins_every_computed_claim():
    """The repository's own coverage, not a fixture's.

    This is the assertion that would have caught the unpinned ladder claims,
    and it is why `reproduce --strict` exists.
    """
    import json
    from pathlib import Path

    from packing.cli import _load_run
    from packing.replication import compute_claims, coverage_gaps

    root = Path(__file__).resolve().parents[1]
    if not (root / "runs" / "ladder-k40" / "calls.jsonl").exists():
        pytest.skip("run artifacts absent; see REPLICATION.md")

    expected = json.loads(
        (root / "replication" / "expected.json").read_text(encoding="utf-8")
    )["claims"]
    assert coverage_gaps(compute_claims(_load_run), expected) == []


def test_the_bundle_carries_every_frame_its_runs_were_drawn_from():
    """A run's own frame going missing is silent until someone rebuilds.

    `reproduce` reads the already-built `items.jsonl`, so it passes with the
    frame absent. What breaks is the chain from source frame to sample: the
    digest gate in `verify_plan` rejects a frame that is not the one the plan
    was built against, so nobody can check that the units were drawn as
    claimed. That is how the K = 40 ladder's 1,600-item frame was left out of a
    published bundle while the 200-item pilot frame stood in for it.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    release_path = root / "replication" / "RELEASE.json"
    if not release_path.exists():
        pytest.skip("no release pointer committed")
    release = json.loads(release_path.read_text(encoding="utf-8"))

    bundled = {name for name in release["files"] if name.startswith("data/")}
    frames = release.get("frames")
    assert frames, "the release pointer records no frame provenance"

    digests = {frame["dataset_sha256"] for frame in frames}
    for frame in frames:
        assert frame["path"] in bundled, (
            f"{frame['path']} is described in the release but not in the archive"
        )

    # Every run that names a frame must have that frame in the bundle.
    for run in release["runs_included"]:
        meta = root / "runs" / run / "meta.json"
        if not meta.exists():
            continue
        digest = json.loads(meta.read_text(encoding="utf-8")).get("dataset_sha256")
        if not digest:
            continue
        assert digest in digests, (
            f"run {run!r} was built against frame {digest[:16]}, which the "
            "bundle does not carry"
        )
