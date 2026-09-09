"""Concurrent execution must be indistinguishable from serial, and never corrupt
the resume file.

The resume file is the one artifact whose corruption would be expensive and
silent: a torn line in the middle of a 38,000-call run means either redoing paid
work or analysing a truncated dataset.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from packing.budget import Budget
from packing.criteria import get_criteria
from packing.providers.base import JudgeResult, ProviderError
from packing.unevaluable import FatalProtocolError
from packing.runner import (
    completed_identities,
    execute,
    execute_concurrent,
    plan,
    read_records,
)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_recovery import _items, _truth_fn  # noqa: E402

CRITERIA = get_criteria("helpsteer2")


class SlowProvider:
    """Deterministic, with a little latency so threads genuinely interleave."""

    name = "slow"
    price_in_per_m = 0.0
    price_out_per_m = 0.0

    def __init__(
        self,
        truth_fn,
        delay: float = 0.002,
        fail_every: int | None = None,
        failure_message: str = "truncated at max_completion_tokens=772",
    ):
        self.model = "slow"
        self.truth_fn = truth_fn
        self.delay = delay
        self.fail_every = fail_every
        # The message decides the failure class, so it is realistic rather than
        # generic: an unrecognised message classifies as fatal and aborts the run
        # by design, which is a different behaviour from a judge failing.
        self.failure_message = failure_message
        self.calls = 0
        self._lock = threading.Lock()

    def judge(self, prompt, group, budget: Budget) -> JudgeResult:
        import time

        with self._lock:
            self.calls += 1
            n = self.calls
        time.sleep(self.delay)
        if self.fail_every and n % self.fail_every == 0:
            raise ProviderError(f"{self.failure_message} (call {n})")
        return JudgeResult(
            evaluations={
                c.key: {
                    "score": self.truth_fn(prompt, c),
                    "reasoning": "r",
                    "evidence": "e",
                }
                for c in group
            },
            input_tokens=100,
            output_tokens=50 * len(group),
        )


def _specs(items):
    return plan(items, CRITERIA, model="slow", replicates=1)


def test_concurrent_matches_serial_exactly(tmp_path: Path) -> None:
    items = _items(12, seed=21)
    truth = _truth_fn(items)
    specs = _specs(items)

    serial_path = tmp_path / "serial.jsonl"
    execute(_specs(items), SlowProvider(truth), serial_path)

    concurrent_path = tmp_path / "concurrent.jsonl"
    execute_concurrent(specs, SlowProvider(truth), concurrent_path, workers=8)

    def keyed(path):
        return {r["identity"]: r for r in read_records(path)}

    serial, concurrent = keyed(serial_path), keyed(concurrent_path)
    assert set(serial) == set(concurrent), "concurrency changed which calls were made"
    for identity, record in serial.items():
        other = concurrent[identity]
        # Ignore fields that legitimately vary run to run.
        for field in ("scores", "group", "group_size", "positions", "item_id", "side"):
            assert record[field] == other[field], f"{identity}: {field} differs"


def test_no_torn_lines_under_concurrency(tmp_path: Path) -> None:
    """Every line must be complete, parseable JSON with a unique identity."""
    items = _items(20, seed=22)
    path = tmp_path / "calls.jsonl"
    execute_concurrent(
        _specs(items), SlowProvider(_truth_fn(items)), path, workers=16
    )

    identities = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            assert line, "blank line in resume file"
            record = json.loads(line)  # raises on a torn line
            identities.append(record["identity"])
    assert len(identities) == len(set(identities)), "duplicate identity written"


def test_concurrent_resume_is_idempotent(tmp_path: Path) -> None:
    items = _items(15, seed=23)
    truth = _truth_fn(items)
    specs = _specs(items)
    path = tmp_path / "calls.jsonl"

    first = execute_concurrent(specs, SlowProvider(truth), path, workers=8)
    assert first.written == len(specs)
    assert first.skipped == 0

    provider = SlowProvider(truth)
    second = execute_concurrent(specs, provider, path, workers=8)
    assert second.written == 0
    assert second.skipped == len(specs)
    assert provider.calls == 0, "resume re-called the provider"
    assert len(completed_identities(path)) == len(specs)


def test_treatment_failures_are_counted_not_fatal(tmp_path: Path) -> None:
    """A judge that truncates is data, not a crash: the run continues."""
    items = _items(10, seed=24)
    path = tmp_path / "calls.jsonl"
    stats = execute_concurrent(
        _specs(items),
        SlowProvider(_truth_fn(items), fail_every=7),
        path,
        workers=8,
        max_failures=1000,
    )
    assert stats.failed > 0
    assert stats.written > 0
    assert stats.written + stats.failed == stats.attempted


def test_max_failures_aborts_and_preserves_work(tmp_path: Path) -> None:
    """An abort must still leave completed calls on disk -- paid work is never discarded."""
    items = _items(40, seed=25)
    path = tmp_path / "calls.jsonl"
    with pytest.raises(RuntimeError, match="exceeded max_failures"):
        execute_concurrent(
            _specs(items),
            SlowProvider(_truth_fn(items), fail_every=3),
            path,
            workers=4,
            max_failures=5,
        )
    assert path.exists()
    # Whatever succeeded before the abort is intact and resumable.
    for record in read_records(path):
        assert "identity" in record


def test_an_unrecognised_failure_aborts_rather_than_being_scored(
    tmp_path: Path,
) -> None:
    """Unknown failures are fatal by design.

    A two-class scheme would score a configuration or SDK bug as a judge miss,
    which is the one error mode no sample size fixes. So an unrecognised message
    stops the run instead of becoming an observation.
    """
    items = _items(10, seed=26)
    path = tmp_path / "calls.jsonl"
    with pytest.raises(FatalProtocolError, match="unsupported parameter"):
        execute_concurrent(
            _specs(items),
            SlowProvider(
                _truth_fn(items),
                fail_every=7,
                failure_message="invalid_request_error: unsupported parameter",
            ),
            path,
            workers=4,
            max_failures=1000,
        )


def test_a_transient_failure_is_not_a_completed_identity(tmp_path: Path) -> None:
    """Resume must retry a network blip rather than treat it as done."""
    items = _items(10, seed=27)
    path = tmp_path / "calls.jsonl"
    stats = execute_concurrent(
        _specs(items),
        SlowProvider(
            _truth_fn(items),
            fail_every=5,
            failure_message="Connection reset by peer",
        ),
        path,
        workers=4,
        max_failures=1000,
    )
    assert stats.failed > 0
    done = completed_identities(path)
    failures = [r for r in read_records(path) if r.get("failure")]
    assert failures, "the failure rows are persisted"
    for record in failures:
        assert record["identity"] not in done


def test_workers_one_is_valid(tmp_path: Path) -> None:
    items = _items(6, seed=26)
    path = tmp_path / "calls.jsonl"
    stats = execute_concurrent(
        _specs(items), SlowProvider(_truth_fn(items)), path, workers=1
    )
    assert stats.written == len(_specs(items))
