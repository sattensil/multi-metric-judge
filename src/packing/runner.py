"""The sweep: item x nominal_N x replicate x model, resumable.

Resumability is not a convenience. A confirmatory run is tens of thousands of
calls; without resume, one interruption doubles or triples the spend. Progress is
an append-only JSONL keyed by a call identity, and a restart skips any identity
already present.

Call identity is ``(item_id, side, model, replicate, partition_hash, group_index)``.
It deliberately excludes ``nominal_n``, because two different nominal levels can
produce the same partition (e.g. N=5 and N=6 for K=5) and the work is genuinely
identical -- so it should be done once and reused for both.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .budget import budget_for, near_cap
from .criteria import Criterion
from .dataset import PairedItem
from .partition import (
    Partition,
    Placement,
    make_partition,
    nominal_levels,
    partition_seed,
)
from .prompt import build, token_estimate
from .providers.base import JudgeProvider, ProviderError
from .schema import schema_token_estimate


@dataclass(frozen=True)
class CallSpec:
    """One judge API call."""

    item: PairedItem
    side: str  # "a" or "b"
    group: tuple[Criterion, ...]
    group_index: int
    partition: Partition
    nominal_n: int
    replicate: int
    model: str

    @property
    def identity(self) -> str:
        return "|".join(
            [
                self.item.item_id,
                self.side,
                self.model,
                str(self.replicate),
                self.partition.hash,
                str(self.group_index),
            ]
        )

    @property
    def response_text(self) -> str:
        return self.item.response_a if self.side == "a" else self.item.response_b


def plan(
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    model: str,
    replicates: int = 1,
    base_seed: int = 20260826,
    levels: tuple[int, ...] | None = None,
) -> list[CallSpec]:
    """Enumerate every call in the sweep, deduplicated by identity.

    Deduplication matters: at K=5, nominal levels 1..5 produce 5+3+2+2+1 groups
    naively, but some partitions coincide across levels. Paying twice for
    identical work would also double-count those observations in the analysis.
    """
    levels = levels or nominal_levels(len(criteria))
    seen: set[str] = set()
    specs: list[CallSpec] = []

    for item in items:
        for nominal_n in levels:
            for replicate in range(replicates):
                seed = partition_seed(item.item_id, nominal_n, replicate, base_seed)
                partition = make_partition(criteria, nominal_n, seed=seed)
                for group_index, group in enumerate(partition.groups):
                    for side in ("a", "b"):
                        spec = CallSpec(
                            item=item,
                            side=side,
                            group=group,
                            group_index=group_index,
                            partition=partition,
                            nominal_n=nominal_n,
                            replicate=replicate,
                            model=model,
                        )
                        if spec.identity in seen:
                            continue
                        seen.add(spec.identity)
                        specs.append(spec)
    return specs


def estimate(specs: list[CallSpec]) -> dict[str, Any]:
    """Dry-run token and cost projection. No network access."""
    input_tokens = 0
    output_ceiling = 0
    for spec in specs:
        prompt = build(
            group=spec.group,
            prompt_text=spec.item.prompt,
            response_text=spec.response_text,
        )
        input_tokens += token_estimate(prompt) + schema_token_estimate(spec.group)
        output_ceiling += budget_for(len(spec.group)).content_budget

    return {
        "calls": len(specs),
        "est_input_tokens": input_tokens,
        "est_output_tokens_ceiling": output_ceiling,
    }


def project_cost(
    estimate_dict: dict[str, Any], price_in_per_m: float, price_out_per_m: float
) -> dict[str, float]:
    """Upper bound: assumes every call fills its output budget, which it will not.

    Deliberately pessimistic -- a dry run that under-promises is useless for
    deciding whether to spend.
    """
    cost_in = estimate_dict["est_input_tokens"] / 1e6 * price_in_per_m
    cost_out = estimate_dict["est_output_tokens_ceiling"] / 1e6 * price_out_per_m
    return {
        "usd_input": round(cost_in, 4),
        "usd_output_ceiling": round(cost_out, 4),
        "usd_total_ceiling": round(cost_in + cost_out, 4),
    }


# ------------------------------------------------------------------- resume ---


def completed_identities(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line from a hard kill. Skip it; the call will
                # simply be redone.
                continue
            identity = record.get("identity")
            if identity:
                done.add(identity)
    return done


@dataclass
class RunStats:
    attempted: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _build_record(spec: CallSpec, result: Any) -> dict[str, Any]:
    return {
        "identity": spec.identity,
        "item_id": spec.item.item_id,
        "side": spec.side,
        "model": spec.model,
        "nominal_n": spec.nominal_n,
        "replicate": spec.replicate,
        "partition_hash": spec.partition.hash,
        "group_index": spec.group_index,
        "group_size": len(spec.group),
        "group": [c.key for c in spec.group],
        "schema_tokens": schema_token_estimate(spec.group),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "latency_ms": round(result.latency_ms, 2),
        "near_cap": near_cap(
            len(spec.group), result.output_tokens, result.reasoning_tokens
        ),
        "scores": {k: v["score"] for k, v in result.evaluations.items()},
        "reasoning": {k: v.get("reasoning", "") for k, v in result.evaluations.items()},
        "positions": {c.key: i for i, c in enumerate(spec.group)},
    }


def execute_concurrent(
    specs: list[CallSpec],
    provider: JudgeProvider,
    out_path: Path,
    *,
    workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
    max_failures: int | None = None,
) -> RunStats:
    """Run the sweep with a thread pool.

    Judge calls are IO-bound (measured ~5.5s each against gpt-5-mini), so serial
    execution makes a 13,680-call confirmatory run take about nine hours. Threads
    are the right tool: the provider SDKs release the GIL on network IO.

    Concurrency does not affect the statistics. Each call is independent, results
    carry their own identity, and the analysis keys on that identity rather than on
    file order. What it does require is a write lock -- interleaved partial lines
    would corrupt the resume file, which is the one thing that must never happen.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_identities(out_path)
    pending = [spec for spec in specs if spec.identity not in done]

    stats = RunStats(skipped=len(specs) - len(pending))
    if not pending:
        return stats

    write_lock = threading.Lock()
    state_lock = threading.Lock()
    abort = threading.Event()
    total = len(pending)
    completed = 0

    def work(spec: CallSpec):
        if abort.is_set():
            return None
        budget = budget_for(len(spec.group))
        prompt = build(
            group=spec.group,
            prompt_text=spec.item.prompt,
            response_text=spec.response_text,
        )
        return spec, provider.judge(prompt, spec.group, budget)

    with out_path.open("a") as fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, spec): spec for spec in pending}
            for future in as_completed(futures):
                spec = futures[future]
                with state_lock:
                    completed += 1
                    stats.attempted += 1
                    index = completed
                if on_progress:
                    on_progress(index, total)

                try:
                    outcome = future.result()
                except ProviderError as exc:
                    with state_lock:
                        stats.failed += 1
                        stats.errors.append(f"{spec.identity}: {exc}")
                        too_many = (
                            max_failures is not None and stats.failed > max_failures
                        )
                    if too_many:
                        # Stop issuing new work, but let in-flight calls land so
                        # their results are not paid for and thrown away.
                        abort.set()
                    continue
                except Exception as exc:  # noqa: BLE001 - unexpected, still must not lose the run
                    with state_lock:
                        stats.failed += 1
                        stats.errors.append(f"{spec.identity}: unexpected {exc!r}")
                    continue

                if outcome is None:
                    continue
                spec, result = outcome
                record = _build_record(spec, result)
                with write_lock:
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                    fh.flush()
                with state_lock:
                    stats.written += 1

    if abort.is_set() and max_failures is not None and stats.failed > max_failures:
        raise RuntimeError(
            f"aborted: {stats.failed} failures exceeded max_failures={max_failures}. "
            f"Completed work is saved; rerun to resume. First errors: "
            f"{stats.errors[:3]}"
        )
    return stats


def execute(
    specs: list[CallSpec],
    provider: JudgeProvider,
    out_path: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    max_failures: int | None = None,
) -> RunStats:
    """Run the sweep, appending one record per successful call.

    Failures are recorded and counted but do not abort the run unless
    ``max_failures`` is exceeded -- a systematic failure (e.g. truncation at large
    group sizes) must stop the run rather than silently producing a dataset whose
    missingness correlates with the treatment.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_identities(out_path)
    stats = RunStats()
    total = len(specs)

    with out_path.open("a") as fh:
        for index, spec in enumerate(specs, start=1):
            if on_progress:
                on_progress(index, total)
            if spec.identity in done:
                stats.skipped += 1
                continue

            stats.attempted += 1
            budget = budget_for(len(spec.group))
            prompt = build(
                group=spec.group,
                prompt_text=spec.item.prompt,
                response_text=spec.response_text,
            )
            try:
                result = provider.judge(prompt, spec.group, budget)
            except ProviderError as exc:
                stats.failed += 1
                stats.errors.append(f"{spec.identity}: {exc}")
                if max_failures is not None and stats.failed > max_failures:
                    raise RuntimeError(
                        f"aborting: {stats.failed} failures exceeds max_failures="
                        f"{max_failures}. Last: {exc}"
                    ) from exc
                continue

            record = {
                "identity": spec.identity,
                "item_id": spec.item.item_id,
                "side": spec.side,
                "model": spec.model,
                "nominal_n": spec.nominal_n,
                "replicate": spec.replicate,
                "partition_hash": spec.partition.hash,
                "group_index": spec.group_index,
                "group_size": len(spec.group),
                "group": [c.key for c in spec.group],
                "schema_tokens": schema_token_estimate(spec.group),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "cached_input_tokens": result.cached_input_tokens,
                "latency_ms": round(result.latency_ms, 2),
                "near_cap": near_cap(
            len(spec.group), result.output_tokens, result.reasoning_tokens
        ),
                "scores": {k: v["score"] for k, v in result.evaluations.items()},
                "reasoning": {k: v.get("reasoning", "") for k, v in result.evaluations.items()},
                "positions": {
                    c.key: i for i, c in enumerate(spec.group)
                },
            }
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            fh.flush()
            stats.written += 1

    return stats


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def plan_balanced(
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    model: str,
    replicates: dict[int, int],
    base_seed: int = 20260826,
) -> list[CallSpec]:
    """Plan a sweep with a per-nominal-level replicate allocation.

    A flat replicate count leaves the large-group cells underpowered, because
    ragged partitions produce small groups as a by-product of several levels while
    the largest group size comes from exactly one. See ``design.suggest_replicates``.
    """
    seen: set[str] = set()
    specs: list[CallSpec] = []
    for level in sorted(replicates):
        count = replicates[level]
        for spec in plan(
            items,
            criteria,
            model=model,
            replicates=count,
            base_seed=base_seed,
            levels=(level,),
        ):
            if spec.identity in seen:
                continue
            seen.add(spec.identity)
            specs.append(spec)
    return specs


def plan_targeted_blocks(
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    model: str,
    levels: tuple[int, ...],
    replicates: int = 1,
    base_seed: int = 20260827,
) -> list[CallSpec]:
    """Plan a packing sweep, judging ONLY the block containing the targeted criterion.

    The treatment is "how many criteria share the call with the one under test", so
    the other blocks of the partition carry no information about it. Judging them
    would multiply cost by ``ceil(K/g)`` for nothing: at K = 20 this is the
    difference between 2 calls per item per level and up to 40.

    The full criterion pool still participates -- it determines the partition, so
    the block's companions are drawn from all K -- which is what makes the load
    real rather than simulated.

    Companions in the judged block yield specificity observations at no extra cost:
    they were not corrupted, so their scores should not move.
    """
    seen: set[str] = set()
    specs: list[CallSpec] = []

    for item in items:
        if not item.targeted_criterion:
            raise ValueError(f"{item.item_id} has no targeted_criterion")
        for nominal_n in levels:
            for replicate in range(replicates):
                seed = partition_seed(
                    item.item_id, nominal_n, replicate, base_seed
                )
                partition = make_partition(criteria, nominal_n, seed=seed)
                placement = partition.placement_for(item.targeted_criterion)
                group = partition.groups[placement.group_index]
                for side in ("a", "b"):
                    spec = CallSpec(
                        item=item,
                        side=side,
                        group=group,
                        group_index=placement.group_index,
                        partition=partition,
                        nominal_n=nominal_n,
                        replicate=replicate,
                        model=model,
                    )
                    if spec.identity in seen:
                        continue
                    seen.add(spec.identity)
                    specs.append(spec)
    return specs


def plan_fixed_position(
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    model: str,
    levels: tuple[int, ...],
    base_seed: int = 20260908,
) -> list[CallSpec]:
    """Plan a ladder with the targeted criterion pinned at position 0.

    The K = 20 sweep drew a random partition, which coupled load to position:
    position is nested inside the call, so E[position | K] = (K-1)/2 exactly and
    support is triangular. Here the target sits first at every rung and companions
    pad the call, which removes the ordinal-position confound.

    Companions are a nested prefix of one seeded permutation of the non-target
    criteria, shared by both sides. Without nesting, K would change both the
    amount of trailing load and which particular criteria are present, putting
    composition noise into the endpoint contrast.

    This does not make load the only thing that varies. The target stays first, so
    K also varies the amount of trailing material, the target's recency, and its
    distance from the end of the context. Those stay coupled by design and the
    estimand is written to match.
    """
    pool_size = len(criteria)
    for level in levels:
        if level < 1:
            raise ValueError(f"level {level} must be >= 1")
        if level > pool_size:
            raise ValueError(f"level {level} exceeds pool size {pool_size}")

    specs: list[CallSpec] = []
    for item in items:
        if not item.targeted_criterion:
            raise ValueError(f"{item.item_id} has no targeted_criterion")
        target = next(
            (c for c in criteria if c.key == item.targeted_criterion), None
        )
        if target is None:
            raise ValueError(
                f"{item.targeted_criterion!r} not in the criterion pool"
            )

        companions = [c for c in criteria if c.key != target.key]
        random.Random(
            partition_seed(f"{item.item_id}:companions", 0, 0, base_seed)
        ).shuffle(companions)

        for nominal_n in levels:
            group = (target, *companions[: nominal_n - 1])
            placements = tuple(
                Placement(
                    criterion=c,
                    group_index=0,
                    position=index,
                    group_size=len(group),
                )
                for index, c in enumerate(group)
            )
            partition = Partition(
                nominal_n=nominal_n,
                groups=(group,),
                seed=base_seed,
                placements=placements,
            )
            for side in ("a", "b"):
                specs.append(
                    CallSpec(
                        item=item,
                        side=side,
                        group=group,
                        group_index=0,
                        partition=partition,
                        nominal_n=nominal_n,
                        replicate=0,
                        model=model,
                    )
                )

    # Interleave across rungs so provider drift cannot align with load. Emitted
    # prompt-major, a prompt's K = 1 and K = 40 calls are adjacent and any
    # mid-run change in provider behaviour lands disproportionately on whichever
    # rungs were in flight. Resume keys on identity rather than order, so
    # reordering is safe, and the seed keeps the run reproducible.
    random.Random(partition_seed("ladder-order", 0, 0, base_seed)).shuffle(specs)
    return specs
