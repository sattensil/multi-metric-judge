"""Command line interface.

Everything except ``run``/``pilot``/``select-model`` works with no API key, so the
design, the estimators, and the cost projection can all be reviewed before any
money is spent.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import analyze, calibrate as calib, cost, design, power
from .criteria import HELPSTEER2_CRITERIA, get_criteria
from .synthetic_criteria import SYNTHETIC_ONLY_CRITERIA
from .dataset import (
    build_corrupted_items,
    fetch as fetch_dataset,
    filter_min_scoreable,
    read_jsonl,
    scoreable_report,
    write_jsonl,
)
from .partition import nominal_levels
from .registry import ALL_CORRUPTIONS
from .providers.base import CATALOG, get_provider
from .runner import (
    estimate,
    execute,
    execute_concurrent,
    plan_balanced,
    plan_targeted_blocks,
    project_cost,
    read_records,
)
from .score import tie_rates

console = Console()
DEFAULT_TARGET = 7


def _run_dir(name: str) -> Path:
    return Path("runs") / name


def _load_run(name: str):
    directory = _run_dir(name)
    items_path = directory / "items.jsonl"
    calls_path = directory / "calls.jsonl"
    meta_path = directory / "meta.json"
    if not items_path.exists() or not calls_path.exists():
        raise click.ClickException(
            f"run {name!r} is incomplete; expected {items_path} and {calls_path}"
        )
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    items = read_jsonl(items_path)
    dataset = meta.get("dataset", "helpsteer2")
    if dataset == "synthetic":
        criteria = ALL_CRITERIA
    elif dataset == "ladder":
        from .ladder_criteria import LADDER_CRITERIA

        criteria = LADDER_CRITERIA
    else:
        criteria = get_criteria(dataset)
    records = list(read_records(calls_path))
    return items, criteria, records, meta


def _show(frame: pd.DataFrame, title: str) -> None:
    if frame is None or len(frame) == 0:
        console.print(f"[yellow]{title}: no rows[/yellow]")
        return
    console.print(f"\n[bold]{title}[/bold]")
    console.print(frame.to_string())


# Loads a .env from the project root into the environment if present. Values are
# never read back out or printed anywhere in this CLI -- `doctor` reports only
# whether a variable is set, never any part of its contents.
load_dotenv()


@click.group()
def cli() -> None:
    """Does judge accuracy degrade as you pack more criteria into one call?"""


# ---------------------------------------------------------------- doctor -----

# Which env var each provider reads, so `doctor` can tell you what is missing
# for the model you actually intend to use.
PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "mock": (),
}


@cli.command()
def doctor() -> None:
    """Pre-flight check: credentials, versions, and data, before spending anything.

    Reports only whether a credential is SET. It never prints any part of a key,
    and nothing here writes to the network.
    """
    import os
    import sys
    from importlib.metadata import PackageNotFoundError, version

    console.print(f"python           {sys.version.split()[0]}")
    console.print(f"env file         {'.env found' if Path('.env').exists() else 'no .env (using shell environment)'}")

    console.print("\n[bold]credentials[/bold] (presence only; values are never read or shown)")
    for provider, variables in PROVIDER_ENV.items():
        if not variables:
            continue
        found = [v for v in variables if os.environ.get(v)]
        if found:
            console.print(f"  {provider:<10} [green]SET[/green]   via {', '.join(found)}")
        else:
            console.print(
                f"  {provider:<10} [red]unset[/red] expected one of: {', '.join(variables)}"
            )

    console.print("\n[bold]sdk versions[/bold]")
    for package in ("openai", "google-genai", "anthropic", "datasets", "statsmodels", "scipy", "pandas"):
        try:
            console.print(f"  {package:<14} {version(package)}")
        except PackageNotFoundError:
            optional = package in {"google-genai", "anthropic"}
            note = "not installed (optional extra)" if optional else "[red]MISSING[/red]"
            console.print(f"  {package:<14} {note}")

    console.print("\n[bold]models available[/bold]")
    for model, entry in CATALOG.items():
        provider = entry["provider"]
        variables = PROVIDER_ENV.get(provider, ())
        ready = provider == "mock" or any(os.environ.get(v) for v in variables)
        mark = "[green]ready[/green]" if ready else "[yellow]needs credential[/yellow]"
        console.print(
            f"  {model:<24} {provider:<10} ${entry['in']:.2f}/${entry['out']:.2f} per 1M  {mark}"
        )

    console.print("\n[bold]data[/bold]")
    data_dir = Path("data")
    files = sorted(data_dir.glob("*.jsonl")) if data_dir.exists() else []
    if not files:
        console.print("  [yellow]none; run `packing fetch --dataset helpsteer2 --min-scoreable 4`[/yellow]")
    for path in files:
        count = sum(1 for _ in path.open())
        console.print(f"  {path}  {count} items")

    console.print(
        "\n[dim]Nothing above sent a request. Next: `packing run --dry-run` to cost a run, "
        "then `packing select-model` for Gate 0.[/dim]"
    )


# ----------------------------------------------------------------- fetch -----


@cli.command()
@click.option("--dataset", default="helpsteer2", show_default=True)
@click.option("--limit", type=int, default=None, help="Max items to keep.")
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option(
    "--min-scoreable",
    type=int,
    default=0,
    show_default=True,
    help="Keep only items with at least this many non-tied criteria (0 = keep all).",
)
def fetch(dataset: str, limit: int | None, out: Path | None, min_scoreable: int) -> None:
    """Download a substrate and normalise it to paired items."""
    out = out or Path("data") / f"{dataset}.jsonl"
    criteria = get_criteria(dataset)
    with console.status(f"fetching {dataset}..."):
        items = fetch_dataset(dataset, limit=limit)

    before = len(items)
    if min_scoreable:
        items = filter_min_scoreable(items, criteria, min_scoreable)
        console.print(
            f"filter >= {min_scoreable} scoreable criteria: kept "
            f"[bold]{len(items)}[/bold] of {before} ({len(items)/max(1,before):.1%})"
        )

    write_jsonl(items, out)
    console.print(f"[green]wrote {len(items)} paired items to {out}[/green]")


# ---------------------------------------------------------------- design -----


@cli.command()
@click.option("--dataset", default="helpsteer2", show_default=True)
@click.option("--target", type=int, default=DEFAULT_TARGET, show_default=True)
def design_cmd(dataset: str, target: int) -> None:
    """Show design balance across group sizes, and the replicate allocation."""
    criteria = get_criteria(dataset)
    k = len(criteria)

    baseline = design.observations_per_replicate(k)
    replicates = design.suggest_replicates(k, target)
    projected = design.projected_observations(k, replicates)

    table = Table(title=f"K={k} design balance (observations per item, per side)")
    table.add_column("group size", justify="right")
    table.add_column("naive (1 rep/level)", justify="right")
    table.add_column(f"balanced (target {target})", justify="right")
    for size in sorted(projected):
        table.add_row(str(size), str(baseline.get(size, 0)), str(projected[size]))
    console.print(table)
    console.print(
        f"naive imbalance max/min = [bold]{design.balance_ratio(baseline):.2f}[/bold]  ->  "
        f"balanced = [bold]{design.balance_ratio(projected):.2f}[/bold]"
    )
    console.print(f"replicate allocation per nominal level: {replicates}")


cli.add_command(design_cmd, name="design")


# ------------------------------------------------------------------- run -----


def _prepare(dataset: str, items_path: Path, limit: int | None):
    if not items_path.exists():
        raise click.ClickException(
            f"{items_path} not found; run `packing fetch --dataset {dataset}` first"
        )
    items = read_jsonl(items_path)
    if limit:
        items = items[:limit]
    if not items:
        raise click.ClickException("no items to run")
    return items, get_criteria(dataset)


@cli.command()
@click.option("--dataset", default="helpsteer2", show_default=True)
@click.option("--items", "items_path", type=click.Path(path_type=Path), default=None)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--name", default="run", show_default=True, help="Run directory name.")
@click.option("--limit", type=int, default=None, help="Max items.")
@click.option("--target", type=int, default=DEFAULT_TARGET, show_default=True)
@click.option("--replicates", "replicate_spec", default="", help='e.g. "3:4,5:2"')
@click.option("--levels", default="", help='Restrict nominal levels, e.g. "1,5"')
@click.option("--dry-run", is_flag=True, help="Plan and cost only; no API calls.")
@click.option("--max-failures", type=int, default=25, show_default=True)
@click.option(
    "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent judge calls. Calls are IO-bound; 1 forces serial.",
)
def run(
    dataset: str,
    items_path: Path | None,
    model: str,
    name: str,
    limit: int | None,
    target: int,
    replicate_spec: str,
    levels: str,
    dry_run: bool,
    max_failures: int,
    workers: int,
) -> None:
    """Execute (or cost out) the sweep."""
    items_path = items_path or Path("data") / f"{dataset}.jsonl"
    items, criteria = _prepare(dataset, items_path, limit)
    k = len(criteria)

    swept = (
        tuple(int(x) for x in levels.split(",") if x.strip())
        if levels.strip()
        else nominal_levels(k)
    )
    if replicate_spec.strip():
        allocation = design.parse_replicate_spec(replicate_spec, swept)
    else:
        suggested = design.suggest_replicates(k, target, levels=swept)
        allocation = {n: suggested[n] for n in swept}

    specs = plan_balanced(items, criteria, model=model, replicates=allocation)
    est = estimate(specs)
    entry = CATALOG.get(model, {"in": 0.0, "out": 0.0})
    projection = project_cost(est, entry["in"], entry["out"])

    console.print(f"[bold]{len(items)}[/bold] items, K={k}, levels={swept}")
    console.print(f"replicates per level: {allocation}")
    console.print(
        f"observations per group size: "
        f"{design.projected_observations(k, allocation, levels=swept)}"
    )
    console.print(f"calls: [bold]{est['calls']}[/bold]  ({est['calls']/len(items):.1f}/item)")
    console.print(
        f"est input tokens {est['est_input_tokens']:,}, "
        f"output ceiling {est['est_output_tokens_ceiling']:,}"
    )
    console.print(
        f"[bold]cost ceiling ${projection['usd_total_ceiling']:.2f}[/bold] "
        f"(input ${projection['usd_input']:.2f} + output ceiling "
        f"${projection['usd_output_ceiling']:.2f}); actual output will be lower"
    )

    if dry_run:
        console.print("[yellow]dry run: nothing sent[/yellow]")
        return

    directory = _run_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(items, directory / "items.jsonl")
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "model": model,
                "levels": list(swept),
                "replicates": allocation,
                "n_items": len(items),
                "estimate": est,
                "cost_ceiling": projection,
            },
            indent=2,
        )
    )

    provider = get_provider(model)
    with console.status("judging...") as status:

        def progress(index: int, total: int) -> None:
            if index % 25 == 0 or index == total:
                status.update(f"judging... {index}/{total}")

        stats = execute_concurrent(
            specs,
            provider,
            directory / "calls.jsonl",
            workers=max(1, workers),
            on_progress=progress,
            max_failures=max_failures,
        )

    console.print(
        f"[green]written {stats.written}, skipped {stats.skipped}, "
        f"failed {stats.failed}[/green]"
    )
    for error in stats.errors[:10]:
        console.print(f"  [red]{error}[/red]")
    if stats.failed:
        console.print(
            "[yellow]Check whether failures correlate with group size -- "
            "missingness that tracks the treatment biases the estimand.[/yellow]"
        )


@cli.command()
@click.option("--dataset", default="helpsteer2", show_default=True)
@click.option("--items", "items_path", type=click.Path(path_type=Path), default=None)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--limit", type=int, default=150, show_default=True)
@click.option("--name", default="pilot", show_default=True)
@click.option("--workers", type=int, default=8, show_default=True)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def pilot(ctx, dataset, items_path, model, limit, name, workers, dry_run) -> None:
    """Gate 3: the cheap two-level pilot (group_size 1 vs K) that sizes the real run.

    Only the extreme levels are run. The pilot's job is to measure the per-item
    difference SD for the power calculation, and the 1-vs-K contrast is where that
    signal is largest per dollar.
    """
    criteria = get_criteria(dataset)
    ctx.invoke(
        run,
        dataset=dataset,
        items_path=items_path,
        model=model,
        name=name,
        limit=limit,
        target=DEFAULT_TARGET,
        levels=f"1,{len(criteria)}",
        replicate_spec="",
        dry_run=dry_run,
        max_failures=25,
        workers=workers,
    )


# ---------------------------------------------------------- select-model -----


@cli.command("select-model")
@click.option("--dataset", default="helpsteer2", show_default=True)
@click.option("--items", "items_path", type=click.Path(path_type=Path), default=None)
@click.option("--limit", type=int, default=100, show_default=True)
@click.option(
    "--models",
    default="gpt-5-nano,gpt-5-mini,gemini-3.1-flash-lite",
    show_default=True,
)
@click.option("--floor", type=float, default=0.62, show_default=True)
@click.option("--workers", type=int, default=8, show_default=True)
def select_model(
    dataset: str,
    items_path: Path | None,
    limit: int,
    models: str,
    floor: float,
    workers: int,
) -> None:
    """Gate 0: cheapest model with headroom to degrade.

    Runs each candidate at group_size 1 only. A model already near the 0.5
    pairwise baseline cannot show degradation, so picking on price alone would
    produce a null result for the wrong reason.
    """
    items_path = items_path or Path("data") / f"{dataset}.jsonl"
    items, criteria = _prepare(dataset, items_path, limit)
    candidates = [m.strip() for m in models.split(",") if m.strip()]

    if len(items) < 50:
        console.print(
            f"[yellow]{len(items)} items is too few to judge against the headroom "
            f"floor; treat this run as an API request-shape check only.[/yellow]"
        )

    table = Table(title=f"Gate 0: group_size=1 headroom (floor {floor})")
    table.add_column("model")
    table.add_column("$/1M in", justify="right")
    table.add_column("$/1M out", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("n", justify="right")
    table.add_column("verdict")

    passing: list[tuple[float, str]] = []
    for model in candidates:
        entry = CATALOG.get(model)
        if entry is None:
            table.add_row(model, "-", "-", "-", "-", "[red]unknown model[/red]")
            continue
        directory = _run_dir(f"select-{model}")
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(items, directory / "items.jsonl")
        (directory / "meta.json").write_text(
            json.dumps({"dataset": dataset, "model": model, "levels": [1]}, indent=2)
        )
        specs = plan_balanced(items, criteria, model=model, replicates={1: 1})
        try:
            provider = get_provider(model)
            with console.status(f"{model}: judging {len(specs)} calls..."):
                execute_concurrent(
                    specs,
                    provider,
                    directory / "calls.jsonl",
                    workers=max(1, workers),
                    max_failures=20,
                )
            records = list(read_records(directory / "calls.jsonl"))
            frame = analyze.to_frame(
                analyze.build_outcomes(records, items, criteria)
            )
            accuracy = analyze.resolved(frame)["correct_num"].mean()
            n = int(analyze.resolved(frame).shape[0])
        except Exception as exc:
            table.add_row(model, f"{entry['in']:.2f}", f"{entry['out']:.2f}", "-", "-", f"[red]{str(exc)[:40]}[/red]")
            continue

        ok = accuracy >= floor
        if ok:
            passing.append((entry["in"] + entry["out"], model))
        table.add_row(
            model,
            f"{entry['in']:.2f}",
            f"{entry['out']:.2f}",
            f"{accuracy:.3f}",
            str(n),
            "[green]headroom[/green]" if ok else "[red]near the 0.5 baseline[/red]",
        )

    console.print(table)
    if passing:
        _, chosen = min(passing)
        console.print(f"[bold green]choose: {chosen}[/bold green] (cheapest clearing the floor)")
    else:
        console.print(
            "[red]no candidate cleared the floor. Either raise the model tier or "
            "revisit the rubric anchors -- a judge at chance cannot show degradation."
            "[/red]"
        )


@cli.command("ties")
@click.option("--dataset", default="helpsteer2", show_default=True)
@click.option("--items", "items_path", type=click.Path(path_type=Path), default=None)
def ties(dataset: str, items_path: Path | None) -> None:
    """Gate 2, label side: tie structure of the ground truth, before any judging.

    Run this before spending anything. A criterion whose human labels tie more
    often than they order cannot support a pairwise outcome no matter how good
    the judge is.
    """
    items_path = items_path or Path("data") / f"{dataset}.jsonl"
    items, criteria = _prepare(dataset, items_path, None)
    report = scoreable_report(items, criteria)

    table = Table(title=f"Gate 2: human label tie structure ({report['n_items']} items)")
    table.add_column("criterion")
    table.add_column("tie rate", justify="right")
    table.add_column("usable n", justify="right")
    table.add_column("primary outcome")
    for key, stats in report["per_criterion"].items():
        rate = stats["human_tie_rate"]
        table.add_row(
            key,
            f"{rate:.1%}",
            str(stats["usable_n"]),
            "pairwise" if rate < 0.50 else "[yellow]QWK (pairwise unusable)[/yellow]",
        )
    console.print(table)

    dist = report["scoreable_per_item_distribution"]
    k = len(criteria)
    console.print(f"\nscoreable criteria per item (of {k}):")
    cumulative = 0
    for count in range(k, -1, -1):
        cumulative += dist.get(count, 0)
        console.print(
            f"  {count}: {dist.get(count, 0):>6}   cumulative >= {count}: "
            f"{cumulative:>6} ({cumulative/report['n_items']:.1%})"
        )
    console.print(
        f"\nmean scoreable per item: [bold]{report['mean_scoreable_per_item']:.2f}[/bold] of {k}"
    )
    console.print(
        f"items contributing nothing to the pairwise outcome: "
        f"[bold]{report['items_with_zero_scoreable']}[/bold]"
    )
    mean_scoreable = report["mean_scoreable_per_item"]
    if report["items_with_zero_scoreable"] or mean_scoreable < k - 1:
        console.print(
            "\n[yellow]Per-item accuracy is the unit the paired estimator uses. "
            "Items carrying only one or two usable criteria make it noisy -- "
            "consider `fetch --min-scoreable 4`.[/yellow]"
        )
    else:
        console.print(
            "\n[green]Tie structure supports a pairwise primary outcome.[/green] "
            "Note that a filtered population has mechanically low tie rates: this "
            "is a property of the selection, not of the criteria. Report the filter "
            "alongside any accuracy figure, and run the looser-filter sensitivity "
            "analysis."
        )


ALL_CRITERIA = HELPSTEER2_CRITERIA + SYNTHETIC_ONLY_CRITERIA
# METHODS 3.4: a hypothesis, replaced by measurement here.
SEMANTIC_KEYS = {
    "shuffle", "numbers", "drop_answer", "contradict", "truncate", "rename",
    "unsupport", "vague",
}


@cli.command("calibrate")
@click.option("--items", "items_path", type=click.Path(path_type=Path),
              default=Path("data/helpsteer2.jsonl"), show_default=True)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--per-corruption", type=int, default=40, show_default=True,
              help="Synthetic items per corruption.")
@click.option("--name", default="calib", show_default=True)
@click.option("--workers", type=int, default=16, show_default=True)
@click.option("--only", default="", help="Comma-separated corruption keys.")
@click.option("--severities", default="",
              help='Severity ladder, e.g. "1,2,4,8". Empty = each op\'s default.')
@click.option("--dry-run", is_flag=True)
def calibrate_cmd(items_path, model, per_corruption, name, workers, only,
                  severities, dry_run):
    """Gate: detection rate per corruption at group_size=1, before any sweep.

    Assigns each operation to test / positive-control / excluded by MEASURED
    detection rate rather than by the hypothesised detection class.
    """
    if not items_path.exists():
        raise click.ClickException(f"{items_path} not found; run `packing fetch` first")
    bases = read_jsonl(items_path)
    wanted = {k.strip() for k in only.split(",") if k.strip()}
    corruptions = [c for c in ALL_CORRUPTIONS if not wanted or c.key in wanted]

    directory = _run_dir(name)
    directory.mkdir(parents=True, exist_ok=True)

    ladder = [int(x) for x in severities.split(",") if x.strip()] or [None]
    all_items = []
    for corruption in corruptions:
        for sev in ladder:
            built = build_corrupted_items(
                bases, corruption, ALL_CRITERIA, limit=per_corruption, severity=sev
            )
            if len(built) < per_corruption:
                console.print(
                    f"[yellow]{corruption.key} sev={sev}: only {len(built)} of "
                    f"{per_corruption} items (low applicability)[/yellow]"
                )
            all_items.extend(built)

    # group_size = 1: each item is judged on its targeted criterion alone.
    specs = []
    for item in all_items:
        criterion = next(c for c in ALL_CRITERIA if c.key == item.targeted_criterion)
        specs.extend(
            plan_balanced([item], (criterion,), model=model, replicates={1: 1})
        )

    est = estimate(specs)
    entry = CATALOG.get(model, {"in": 0.0, "out": 0.0})
    projection = project_cost(est, entry["in"], entry["out"])
    console.print(
        f"{len(corruptions)} corruptions, {len(all_items)} items, "
        f"[bold]{est['calls']}[/bold] calls, cost ceiling "
        f"[bold]${projection['usd_total_ceiling']:.2f}[/bold]"
    )
    if dry_run:
        console.print("[yellow]dry run: nothing sent[/yellow]")
        return

    write_jsonl(all_items, directory / "items.jsonl")
    (directory / "meta.json").write_text(json.dumps({
        "dataset": "synthetic", "model": model, "levels": [1],
        "per_corruption": per_corruption,
        "corruptions": [c.key for c in corruptions],
        "severities": ladder,
    }, indent=2))

    provider = get_provider(model)
    with console.status("judging...") as status:
        def progress(i, total):
            if i % 50 == 0 or i == total:
                status.update(f"judging... {i}/{total}")
        stats = execute_concurrent(
            specs, provider, directory / "calls.jsonl",
            workers=max(1, workers), on_progress=progress, max_failures=60,
        )
    console.print(
        f"[green]written {stats.written}, skipped {stats.skipped}, "
        f"failed {stats.failed}[/green]"
    )
    for error in stats.errors[:5]:
        console.print(f"  [red]{error}[/red]")
    _report_calibration(name)


@cli.command("calibration-report")
@click.option("--run", "name", default="calib", show_default=True)
def calibration_report(name):
    """Re-print the calibration verdicts from an existing run."""
    _report_calibration(name)


def _report_calibration(name: str) -> None:
    directory = _run_dir(name)
    items = read_jsonl(directory / "items.jsonl")
    records = list(read_records(directory / "calls.jsonl"))
    by_id = {i.item_id: i for i in items}

    detected: dict[str, dict[str, int]] = {}
    for item_id, group in _group_by_item(records):
        item = by_id.get(item_id)
        if item is None or "a" not in group or "b" not in group:
            continue
        key = item.targeted_criterion
        score_a = group["a"]["scores"].get(key)
        score_b = group["b"]["scores"].get(key)
        if score_a is None or score_b is None:
            continue
        expected = item.human_a[key] - item.human_b[key]
        observed = score_a - score_b
        bucket = detected.setdefault(
            (item.meta["corruption"], item.meta["severity"]),
            {"hit": 0, "wrong": 0, "tie": 0},
        )
        if observed == 0:
            bucket["tie"] += 1
        elif (observed > 0) == (expected > 0):
            bucket["hit"] += 1
        else:
            bucket["wrong"] += 1

    verdicts = []
    for (key, sev), results in detected.items():
        corruption = next((c for c in ALL_CORRUPTIONS if c.key == key), None)
        if corruption is None:
            continue
        v = calib.verdict_for(
            key, corruption.targeted_criterion,
            "semantic" if key in SEMANTIC_KEYS else "surface",
            results["hit"], results["wrong"], results["tie"],
        )
        verdicts.append((sev, v))
    ladder_mode = len({s for s, _ in verdicts}) > 1

    table = Table(title=f"Calibration at group_size=1 (band {calib.BAND_LOW}-{calib.BAND_HIGH})")
    for col in ("corruption", "sev", "n", "detect", "tie", "discrim", "role"):
        table.add_column(col, justify="right" if col not in ("corruption", "role") else "left")
    colour = {"test": "green", "ceiling": "yellow", "insensitive": "cyan",
              "unreliable": "red", "insufficient": "dim"}
    order = sorted(verdicts, key=lambda sv: (sv[1].corruption, sv[0] or 0)) \
        if ladder_mode else sorted(verdicts, key=lambda sv: -sv[1].rate)
    for sev, v in order:
        c = colour.get(v.role, "white")
        disc = f"{v.discrimination:.2f}" if v.detected + v.wrong else "-"
        table.add_row(v.corruption, str(sev), str(v.n), f"{v.rate:.2f}",
                      f"{v.tie_rate:.2f}", disc, f"[{c}]{v.role}[/{c}]")
    console.print(table)

    if ladder_mode:
        console.print(calib.window_report(verdicts))
        return
    summary = calib.agreement_with_hypothesis([v for _s, v in verdicts])
    console.print("\n[bold]Hypothesis check (METHODS 3.4)[/bold]")
    console.print(f"  surface operations at ceiling : {summary.get('surface_at_ceiling')}")
    console.print(f"  semantic operations in band   : {summary.get('semantic_in_test_band')}")
    if summary.get("insensitive_needing_severity"):
        console.print(
            f"\n[cyan]{len(summary['insensitive_needing_severity'])} operations "
            f"insensitive at current severity (high ties, correct when they "
            f"commit) -- raise severity rather than exclude:[/cyan]"
        )
        for row in summary["insensitive_needing_severity"]:
            console.print(f"    {row['corruption']:<14} tie={row['tie_rate']:.2f} "
                          f"discrim={row['discrimination']}")
    if summary.get("unreliable"):
        console.print(f"\n[red]{len(summary['unreliable'])} operations UNRELIABLE "
                      f"(judges backwards when committing):[/red]")
        for row in summary["unreliable"]:
            console.print(f"    {row['corruption']:<14} discrim={row['discrimination']}")
    if summary.get("discrepancies"):
        console.print(f"  [yellow]{len(summary['discrepancies'])} discrepancies; "
                      f"measurement governs[/yellow]")
        for row in summary["discrepancies"]:
            console.print(f"    {row['corruption']:<14} {row['hypothesised']:<9} "
                          f"-> {row['role']} ({row['detection_rate']})")


def _group_by_item(records):
    grouped: dict[tuple, dict] = {}
    for r in records:
        grouped.setdefault(
            (r["item_id"], r["partition_hash"], r["group_index"]), {}
        )[r["side"]] = r
    for (item_id, _ph, _gi), sides in grouped.items():
        yield item_id, sides


# Excluded from measurement by the calibration gate. `jargon` is MUTE (87-97% ties
# at every severity 1-30), so `complexity` has no valid operation; it remains in the
# criterion pool as packing load.
MUTE_OPERATIONS = {"jargon"}
SWEEP_LEVELS = (1, 2, 3, 5, 8, 12, 16, 20)


@cli.command("sweep")
@click.option("--items", "items_path", type=click.Path(path_type=Path),
              default=Path("data/helpsteer2.jsonl"), show_default=True)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--per-corruption", type=int, default=40, show_default=True)
@click.option("--levels", default=",".join(str(x) for x in SWEEP_LEVELS), show_default=True)
@click.option("--replicates", type=int, default=1, show_default=True)
@click.option("--name", default="sweep", show_default=True)
@click.option("--workers", type=int, default=16, show_default=True)
@click.option("--dry-run", is_flag=True)
def sweep_cmd(items_path, model, per_corruption, levels, replicates, name,
              workers, dry_run):
    """The packing sweep: does detection degrade as criteria per call rises?

    K = 20 criteria form the packing load. Only the block containing each item's
    targeted criterion is judged, which is what makes K = 20 affordable.
    """
    if not items_path.exists():
        raise click.ClickException(f"{items_path} not found; run `packing fetch` first")
    if model != "gpt-5-mini":
        raise click.ClickException(
            "synthetic operation roles are calibrated only for gpt-5-mini; "
            "run a model-specific calibration before sweeping another model"
        )
    bases = read_jsonl(items_path)
    swept = tuple(int(x) for x in levels.split(",") if x.strip())
    measured = [c for c in ALL_CORRUPTIONS if c.key not in MUTE_OPERATIONS]

    all_items = []
    for corruption in measured:
        built = build_corrupted_items(
            bases, corruption, ALL_CRITERIA, limit=per_corruption
        )
        if len(built) < per_corruption:
            console.print(
                f"[yellow]{corruption.key}: {len(built)}/{per_corruption} items"
                f"[/yellow]"
            )
        all_items.extend(built)

    specs = plan_targeted_blocks(
        all_items, ALL_CRITERIA, model=model, levels=swept, replicates=replicates
    )
    est = estimate(specs)
    entry = CATALOG.get(model, {"in": 0.0, "out": 0.0})
    projection = project_cost(est, entry["in"], entry["out"])

    from collections import Counter
    sizes = Counter(len(sp.group) for sp in specs)
    console.print(
        f"K={len(ALL_CRITERIA)} pool, {len(measured)} measured operations, "
        f"{len(all_items)} items, levels={swept}"
    )
    console.print(f"observations per actual group size: {dict(sorted(sizes.items()))}")
    console.print(f"calls: [bold]{est['calls']}[/bold]")
    console.print(
        f"est input {est['est_input_tokens']:,}, output ceiling "
        f"{est['est_output_tokens_ceiling']:,}"
    )
    console.print(
        f"[bold]cost ceiling ${projection['usd_total_ceiling']:.2f}[/bold] "
        f"(actual output will be lower)"
    )
    if dry_run:
        console.print("[yellow]dry run: nothing sent[/yellow]")
        return

    directory = _run_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_items, directory / "items.jsonl")
    (directory / "meta.json").write_text(json.dumps({
        "dataset": "synthetic", "model": model, "levels": list(swept),
        "per_corruption": per_corruption, "replicates": replicates,
        "measured": [c.key for c in measured],
        "mute_excluded": sorted(MUTE_OPERATIONS),
        "k_pool": len(ALL_CRITERIA),
        "operation_roles": {
            **{key: "test" for key in analyze.SYNTHETIC_TEST_OPERATIONS},
            **{
                key: "positive_control"
                for key in analyze.SYNTHETIC_POSITIVE_CONTROLS
            },
        },
        "detection_classes": {
            key: (
                "semantic"
                if key in analyze.SYNTHETIC_SEMANTIC_OPERATIONS
                else "surface"
            )
            for key in (
                analyze.SYNTHETIC_TEST_OPERATIONS
                | analyze.SYNTHETIC_POSITIVE_CONTROLS
            )
        },
        "calibration_provenance": {
            "model": "gpt-5-mini",
            "runs": ["calib", "ladder"],
            "rule": "measured role at group_size=1; ladder-selected severity",
        },
    }, indent=2))

    provider = get_provider(model)
    with console.status("judging...") as status:
        def progress(i, total):
            if i % 100 == 0 or i == total:
                status.update(f"judging... {i}/{total}")
        stats = execute_concurrent(
            specs, provider, directory / "calls.jsonl",
            workers=max(1, workers), on_progress=progress, max_failures=150,
        )
    console.print(
        f"[green]written {stats.written}, skipped {stats.skipped}, "
        f"failed {stats.failed}[/green]"
    )
    for error in stats.errors[:8]:
        console.print(f"  [red]{error}[/red]")


# ----------------------------------------------------------------- score -----


@cli.command()
@click.option("--run", "name", default="pilot", show_default=True)
def score(name: str) -> None:
    """Gate 2: tie rates and descriptive accuracy."""
    items, criteria, records, _ = _load_run(name)
    outcomes = analyze.build_outcomes(records, items, criteria)
    if not outcomes:
        raise click.ClickException("no complete pairs; both sides of a pair are required")
    frame = analyze.to_frame(outcomes)

    rates = pd.DataFrame(tie_rates(outcomes)).T.sort_index()
    _show(rates.round(4), "Gate 2: tie rates per criterion")
    crippled = rates.index[rates["resolved_fraction"] < 0.5].tolist()
    if crippled:
        console.print(
            f"[yellow]criteria with <50% resolvable observations: {crippled}. "
            f"Use QWK as the primary outcome for these.[/yellow]"
        )

    desc = analyze.descriptives(frame)
    _show(desc["by_group_size"].round(4), "Accuracy by group size (descriptive only)")
    _show(
        desc["judge_tie_by_group_size"].round(4),
        "Judge tie rate by group size (co-primary outcome)",
    )
    _show(analyze.qwk_by_group_size(frame, criteria).round(4), "QWK by criterion and group size")


# --------------------------------------------------------------- analyze -----


@cli.command("analyze")
@click.option("--run", "name", default="pilot", show_default=True)
@click.option("--reference", type=int, default=1, show_default=True)
@click.option("--mixed/--no-mixed", default=True, show_default=True)
def analyze_cmd(name: str, reference: int, mixed: bool) -> None:
    """Primary and secondary estimators."""
    items, criteria, records, meta = _load_run(name)
    frame = analyze.to_frame(analyze.build_outcomes(records, items, criteria))
    if frame.empty:
        raise click.ClickException("no outcomes to analyse")

    if meta.get("dataset") == "synthetic":
        frame = analyze.annotate_synthetic(
            frame,
            items,
            operation_roles=meta.get("operation_roles"),
            detection_classes=meta.get("detection_classes"),
        )
        _show(
            analyze.synthetic_recovery(frame).round(4),
            "Synthetic targeted recovery (stratified; hits, ties, and wrong directions)",
        )
        _show(
            analyze.synthetic_specificity(frame).round(4),
            "Untouched-companion specificity (construct-overlap diagnostic)",
        )
        _show(
            analyze.synthetic_leakage_summary(frame).round(4),
            "Leakage diagnostic (worst systematic companion shift vs target)",
        )
        _show(
            analyze.synthetic_leakage_matrix(frame).round(4),
            "Leakage matrix: mean corruption-aligned score shift",
        )

        # Calibration-defined test operations carry the primary estimand.
        # Positive controls remain in the descriptive table above: their purpose
        # is to detect wholesale disengagement, not estimate a gradual slope.
        # Leakage is reported above as context; it does not drop operations.
        primary = analyze.synthetic_recovery_frame(
            frame[frame["calibration_role"] == "test"]
        )
        for detection_class, chunk in primary.groupby("detection_class"):
            label = f"{detection_class} test operations"
            _show(
                analyze.paired_ladder(chunk, reference_size=reference),
                f"PRIMARY ({label}): paired recovery contrast vs group_size={reference}",
            )
            try:
                form = analyze.functional_form(chunk)
                console.print(f"\n[bold]Functional form ({label})[/bold]")
                console.print(json.dumps(form, indent=2, default=str))
            except Exception as exc:
                console.print(
                    f"[yellow]functional form unavailable for {label}: {exc}[/yellow]"
                )
            try:
                competition = analyze.position_competition(chunk)
                console.print(
                    f"\n[bold]Rival hypothesis for {label}: position, not count[/bold]"
                )
                console.print(json.dumps(competition, indent=2, default=str))
            except Exception as exc:
                console.print(
                    f"[yellow]position competition unavailable for {label}: {exc}[/yellow]"
                )
            _show(
                analyze.by_criterion_slopes(chunk).round(4),
                f"Per-operation slopes ({label}; Holm-corrected)",
            )
            if mixed:
                try:
                    fit = analyze.fit_mixed(chunk)
                    console.print(
                        f"\n[bold]Crossed mixed model ({label}; variational)[/bold]"
                    )
                    console.print(str(fit.summary()))
                except Exception as exc:
                    console.print(
                        f"[yellow]crossed mixed model unavailable for {label}: "
                        f"{str(exc)[:120]}[/yellow]"
                    )

        usage = cost.usage_frame(records)
        entry = CATALOG.get(meta.get("model", ""), {"in": 0.0, "out": 0.0})
        cost_table = cost.cost_per_criterion(usage, entry["in"], entry["out"])
        _show(
            cost_table.round(6),
            "Cost per criterion by group size",
        )
        _show(
            analyze.synthetic_frontier(frame, cost_table).round(6),
            "Synthetic recovery/cost frontier (descriptive Pareto set)",
        )
        console.print("\n[bold]Cache check[/bold]")
        console.print(json.dumps(cost.cache_report(usage), indent=2))
        return

    _show(
        analyze.paired_ladder(frame, reference_size=reference),
        f"PRIMARY: within-item paired contrast vs group_size={reference} (Holm-corrected)",
    )

    try:
        form = analyze.functional_form(frame)
        console.print("\n[bold]Functional form[/bold]")
        console.print(json.dumps(form, indent=2, default=str))
    except Exception as exc:
        console.print(f"[yellow]functional form unavailable: {exc}[/yellow]")

    try:
        competition = analyze.position_competition(frame)
        console.print("\n[bold]Rival hypothesis: position, not count[/bold]")
        console.print(json.dumps(competition, indent=2, default=str))
    except Exception as exc:
        console.print(f"[yellow]position competition unavailable: {exc}[/yellow]")

    _show(
        analyze.by_criterion_slopes(frame).round(4),
        "ACTIONABLE: per-criterion group-size slope (Holm-corrected)",
    )

    if mixed:
        try:
            fit = analyze.fit_mixed(frame)
            console.print("\n[bold]Crossed mixed model (variational; approximate SDs)[/bold]")
            console.print(str(fit.summary()))
        except Exception as exc:
            console.print(
                f"[yellow]crossed mixed model unavailable ({str(exc)[:120]}); "
                f"the paired contrast and cluster-robust logit above stand alone[/yellow]"
            )

    usage = cost.usage_frame(records)
    entry = CATALOG.get(meta.get("model", ""), {"in": 0.0, "out": 0.0})
    cost_table = cost.cost_per_criterion(usage, entry["in"], entry["out"])
    _show(cost_table.round(6), "Cost per criterion by group size")
    console.print("\n[bold]Cache check[/bold]")
    console.print(json.dumps(cost.cache_report(usage), indent=2))

    accuracy_table = analyze.descriptives(frame)["by_group_size"]
    _show(
        cost.frontier(accuracy_table, cost_table).round(6),
        "Accuracy vs cost frontier (knee = cheapest packing indistinguishable from best)",
    )


# ----------------------------------------------------------------- power -----


@cli.command("power")
@click.option("--run", "name", default="pilot", show_default=True)
@click.option("--reference", type=int, default=1, show_default=True)
@click.option("--comparison", type=int, default=None)
def power_cmd(name: str, reference: int, comparison: int | None) -> None:
    """Gate 3: size the confirmatory run from pilot variance."""
    items, criteria, records, meta = _load_run(name)
    frame = analyze.to_frame(analyze.build_outcomes(records, items, criteria))
    comparison = comparison or len(criteria)

    if meta.get("dataset") == "synthetic":
        frame = analyze.annotate_synthetic(
            frame,
            items,
            operation_roles=meta.get("operation_roles"),
            detection_classes=meta.get("detection_classes"),
        )
        frame = analyze.synthetic_recovery_frame(
            frame[frame["calibration_role"] == "test"]
        )
        found = False
        for detection_class, chunk in frame.groupby("detection_class"):
            result = power.plan_from_pilot(chunk, reference, comparison)
            if result is None:
                continue
            found = True
            console.print(
                f"\n[bold]Power planning ({detection_class} test operations): "
                f"group_size {reference} vs {comparison}[/bold]"
            )
            console.print(json.dumps(result.as_dict(), indent=2))
        if not found:
            raise click.ClickException(
                f"not enough paired synthetic base items at group sizes "
                f"{reference} and {comparison}"
            )
        console.print(
            "\n[yellow]Size the confirmatory run from the MDE grid, not from the "
            "observed effect.[/yellow]"
        )
        return

    result = power.plan_from_pilot(frame, reference, comparison)
    if result is None:
        raise click.ClickException(
            f"not enough paired items at group sizes {reference} and {comparison}"
        )
    console.print(f"\n[bold]Power planning: group_size {reference} vs {comparison}[/bold]")
    console.print(json.dumps(result.as_dict(), indent=2))
    console.print(
        "\n[yellow]Size the confirmatory run from the MDE grid, not from the "
        "observed effect: a pilot that reached significance is more likely to have "
        "overshot the true effect than undershot it.[/yellow]"
    )


if __name__ == "__main__":
    cli()


@cli.command("ladder")
@click.option(
    "--items", "items_path", type=click.Path(),
    default="data/helpsteer2.jsonl", show_default=True,
)
@click.option("--model", default="gpt-5-mini", show_default=True)
@click.option("--core-n", type=int, default=464, show_default=True)
@click.option("--numbers-n", type=int, default=184, show_default=True)
@click.option("--core-candidates", type=int, default=508, show_default=True)
@click.option("--per-op-max", type=int, default=96, show_default=True)
@click.option("--levels", default="1,5,10,20,40", show_default=True)
@click.option("--seed", type=int, default=20260908, show_default=True)
@click.option(
    "--plan-out", type=click.Path(),
    default="runs/ladder-plan.json", show_default=True,
)
@click.option(
    "--allow-short", is_flag=True, default=False,
    help="Development only. Never use for the pre-registered budget.",
)
@click.option("--dry-run/--no-dry-run", default=True, show_default=True)
def ladder_cmd(
    items_path: str,
    model: str,
    core_n: int,
    numbers_n: int,
    core_candidates: int,
    per_op_max: int,
    levels: str,
    seed: int,
    plan_out: str,
    allow_short: bool,
    dry_run: bool,
) -> None:
    """The fixed-position load ladder: does trailing load change recovery?"""
    from collections import Counter

    from .eligibility import CORE_OPERATIONS, NUMBERS_OPERATION, select_response
    from .ladder_criteria import LADDER_CRITERIA, LADDER_MANIFEST
    from .ladder_dataset import activate, build_ladder_item, operation_queues
    from .lanes import fill_lane, split_lanes
    from .runner import plan_fixed_position
    from .sampling_plan import SamplingPlan, Unit, file_sha256, write_plan

    LADDER_MANIFEST.validate(expect=len(LADDER_CRITERIA))

    # Pricing must not silently become zero. A missing catalogue entry is a
    # configuration error, not a free model.
    if model not in CATALOG:
        raise click.ClickException(
            f"no pricing for {model!r}; add it to CATALOG before costing a run"
        )
    entry = CATALOG[model]

    swept = tuple(int(x) for x in levels.split(",") if x.strip())
    if len(set(swept)) != len(swept):
        raise click.ClickException(f"levels must be unique, got {swept}")
    if not swept or swept[0] != 1 or swept[-1] != len(LADDER_CRITERIA):
        raise click.ClickException(
            f"levels must run from 1 to {len(LADDER_CRITERIA)}; got {swept}"
        )
    if core_n % len(CORE_OPERATIONS) != 0:
        raise click.ClickException(
            f"core-n {core_n} is not divisible by {len(CORE_OPERATIONS)}; equal "
            "quotas are part of the primary estimand"
        )

    path = Path(items_path)
    if not path.exists():
        raise click.ClickException(f"{path} not found; run `packing fetch` first")
    bases = read_jsonl(path)
    ids = [b.item_id for b in bases]
    if len(set(ids)) != len(ids):
        raise click.ClickException("duplicate base-prompt ids in the input file")
    by_id = {b.item_id: b for b in bases}

    try:
        core_stream, numbers_stream = split_lanes(
            ids, seed=seed, core_candidates=core_candidates
        )
    except ValueError as exc:
        # A frame too small for the requested lane split is a user error with an
        # obvious remedy, so it reads as a message rather than a traceback.
        raise click.ClickException(
            f"{exc}. Fetch a larger frame with `packing fetch --limit N`, or "
            "lower --core-candidates."
        ) from exc

    # Screen once, and KEEP the side. The side that satisfied all eight core
    # operations is the one that put this prompt in A_core = 1. Re-deriving it
    # later from the assigned operation alone can pick the other response, which
    # is eligible for that operation but not for all eight.
    core_sides = {
        item_id: select_response(by_id[item_id], CORE_OPERATIONS, seed=seed)
        for item_id in core_stream
    }
    numbers_sides = {
        item_id: select_response(by_id[item_id], (NUMBERS_OPERATION,), seed=seed)
        for item_id in numbers_stream
    }

    core_ids, core_rest = fill_lane(
        core_stream, lambda i: core_sides[i] is not None, core_n
    )
    numbers_ids, numbers_rest = fill_lane(
        numbers_stream, lambda i: numbers_sides[i] is not None, numbers_n
    )

    console.print(
        f"K={len(LADDER_CRITERIA)} pool, manifest {LADDER_MANIFEST.hash}, "
        f"lengths {LADDER_MANIFEST.length_stats}"
    )
    console.print(
        f"core lane: {len(core_stream)} candidates, "
        f"{len(core_ids)}/{core_n} filled"
    )
    console.print(
        f"numbers lane: {len(numbers_stream)} candidates, "
        f"{len(numbers_ids)}/{numbers_n} filled"
    )
    short = len(core_ids) < core_n or len(numbers_ids) < numbers_n
    if short and not allow_short:
        raise click.ClickException(
            "lane short of quota. A partial sample costs and plans a different "
            "experiment, so this is an error rather than a warning. Fetch a "
            "larger frame, or pass --allow-short for development only."
        )
    if short:
        console.print(
            "[yellow]SHORT SAMPLE: development only, not the budget[/yellow]"
        )

    per_op = min(core_n // len(CORE_OPERATIONS), per_op_max)
    queue_depth = min(per_op_max, len(core_ids) // len(CORE_OPERATIONS))
    if queue_depth < 1:
        raise click.ClickException(
            f"only {len(core_ids)} core units for {len(CORE_OPERATIONS)} "
            "operations; cannot form queues"
        )
    queues = operation_queues(
        core_ids, CORE_OPERATIONS, seed=seed, per_op_max=queue_depth
    )
    assigned = activate(queues, per_op=min(per_op, queue_depth))

    items: list = []
    units: list = []
    for item_id, op in assigned.items():
        side = core_sides[item_id]
        if side is None:
            raise click.ClickException(
                f"{item_id} passed screening with no side; invariant violated"
            )
        built = build_ladder_item(
            by_id[item_id], op, side, LADDER_CRITERIA, seed=seed
        )
        if built is None:
            # Screening already proved the operation applies to this side, so a
            # None here means the two paths disagree. That is a bug, not a
            # droppable item: skipping it would break equal quotas.
            raise click.ClickException(
                f"{item_id}/{op}/{side} screened eligible but did not build"
            )
        items.append(built)
        units.append(
            Unit(
                base_item_id=item_id,
                lane="core",
                side=side,
                eligible=True,
                operation=op,
                realisation_seed=built.meta["realisation_seed"],
                companion_seed=seed,
                status="active",
            )
        )
    for item_id in numbers_ids:
        side = numbers_sides[item_id]
        if side is None:
            raise click.ClickException(
                f"{item_id} passed screening with no side; invariant violated"
            )
        built = build_ladder_item(
            by_id[item_id], NUMBERS_OPERATION, side, LADDER_CRITERIA, seed=seed
        )
        if built is None:
            raise click.ClickException(
                f"{item_id}/numbers/{side} screened eligible but did not build"
            )
        items.append(built)
        units.append(
            Unit(
                base_item_id=item_id,
                lane="numbers",
                side=side,
                eligible=True,
                operation=NUMBERS_OPERATION,
                realisation_seed=built.meta["realisation_seed"],
                companion_seed=seed,
                status="active",
            )
        )

    # Reserves: the screened, eligible remainder of each lane, in frozen order.
    # A persistent transient failure is replaced from here, so the replacement is
    # drawn from the same population and the draw order was fixed before any
    # outcome existed. Without them there is nothing legitimate to replace with.
    for lane, rest, sides in (
        ("core", core_rest, core_sides),
        ("numbers", numbers_rest, numbers_sides),
    ):
        for item_id in rest:
            side = sides.get(item_id)
            if side is None:
                continue
            units.append(
                Unit(
                    base_item_id=item_id,
                    lane=lane,
                    side=side,
                    eligible=True,
                    operation="",
                    realisation_seed=0,
                    companion_seed=seed,
                    status="reserve",
                )
            )

    # A silent shortfall would destroy the equal weighting the primary estimand
    # assumes, so the post-build check is mandatory rather than advisory.
    built_counts = Counter(
        u.operation
        for u in units
        if u.lane == "core" and u.status == "active"
    )
    if not allow_short and len(set(built_counts.values())) > 1:
        raise click.ClickException(
            f"unequal core quotas after build: {dict(built_counts)}"
        )

    specs = plan_fixed_position(
        items, LADDER_CRITERIA, model=model, levels=swept, base_seed=seed
    )
    est = estimate(specs)
    projection = project_cost(est, entry["in"], entry["out"])
    console.print(f"items: {len(items)}  levels={swept}")
    console.print(f"calls: {len(specs)}")
    console.print(
        f"est input {est['est_input_tokens']:,}, "
        f"output ceiling {est['est_output_tokens_ceiling']:,}"
    )
    console.print(
        f"[bold]cost ceiling ${projection['usd_total_ceiling']:.2f}[/bold] "
        "(actual output will be lower)"
    )

    plan = SamplingPlan(
        dataset_id=path.stem,
        dataset_sha256=file_sha256(path),
        manifest_sha256=LADDER_MANIFEST.full_hash,
        seed=seed,
        model=model,
        model_settings={"reasoning_effort": "low", "temperature": None},
        levels=swept,
        units=tuple(units),
        call_identities=tuple(s.identity for s in specs),
    )
    Path(plan_out).parent.mkdir(parents=True, exist_ok=True)
    write_plan(Path(plan_out), plan)
    console.print(
        f"sampling plan written to {plan_out} "
        f"({plan.active_count} active units)"
    )

    if dry_run:
        console.print("dry run: nothing sent")
        return
    raise click.ClickException(
        "live execution consumes the frozen sampling plan and is not implemented "
        "here; see the live-execution prerequisites in the plan"
    )


@cli.command("ladder-run")
@click.option("--plan", "plan_path", type=click.Path(), required=True)
@click.option("--items", "items_path", type=click.Path(), required=True)
@click.option("--name", default="ladder", show_default=True)
@click.option("--workers", type=int, default=16, show_default=True)
@click.option("--max-failures", type=int, default=40, show_default=True)
@click.option(
    "--timeout", type=float, default=90.0, show_default=True,
    help="Per-request timeout in seconds, passed to the provider client.",
)
@click.option(
    "--sdk-retries", type=int, default=1, show_default=True,
    help="Provider-level retries. Each retry is a fully billed call whose "
         "tokens are never recorded, so this bounds invisible spend.",
)
@click.option(
    "--stall-seconds", type=float, default=300.0, show_default=True,
    help="Abort if no call completes for this long. Written work is resumable, "
         "so aborting a hung run loses nothing.",
)
@click.option(
    "--archive-to", type=click.Path(), default="artifacts", show_default=True,
    help="Where the run is copied after execution. Must not be under runs/.",
)
@click.option(
    "--confirm-spend", is_flag=True, default=False,
    help="Required to send anything. Without it this is a plan check only.",
)
def ladder_run_cmd(
    plan_path: str,
    items_path: str,
    name: str,
    workers: int,
    max_failures: int,
    timeout: float,
    sdk_retries: int,
    stall_seconds: float,
    archive_to: str,
    confirm_spend: bool,
) -> None:
    """Execute a frozen ladder sampling plan.

    Consumes the artifact rather than re-screening, so the sample cannot drift
    between planning and execution.
    """
    from .live import archive_run, rebuild_items, verify_run_dir
    from .runner import execute_concurrent
    from .sampling_plan import read_plan

    plan = read_plan(Path(plan_path))
    items, specs = rebuild_items(plan, Path(items_path))

    if plan.model not in CATALOG:
        raise click.ClickException(f"no pricing for {plan.model!r}")
    entry = CATALOG[plan.model]
    est = estimate(specs)
    projection = project_cost(est, entry["in"], entry["out"])

    console.print(f"plan {plan_path}: manifest {plan.manifest_sha256[:16]}, seed {plan.seed}")
    console.print(f"integrity verified: dataset and manifest digests match")
    console.print(f"units {plan.active_count}, calls {len(specs)}, model {plan.model}")
    console.print(
        f"[bold]cost ceiling ${projection['usd_total_ceiling']:.2f}[/bold] "
        "for one attempt per call; provider retries bill on top and are not "
        "counted here, so this is not a bound on the invoice"
    )

    if not confirm_spend:
        console.print(
            "[yellow]no --confirm-spend: plan verified, nothing sent[/yellow]"
        )
        return

    if Path(archive_to).resolve().is_relative_to(Path("runs").resolve()):
        raise click.ClickException(
            f"--archive-to {archive_to} is under runs/, which .gitignore covers. "
            "The point of archival is to escape that rule."
        )

    run_dir = _run_dir(name)
    # Before anything is overwritten: an existing run directory may already hold
    # paid calls, and only some of this metadata may legitimately change.
    verify_run_dir(run_dir, plan)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(items, run_dir / "items.jsonl")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "ladder",
                "model": plan.model,
                "levels": list(plan.levels),
                "seed": plan.seed,
                "manifest_sha256": plan.manifest_sha256,
                "dataset_sha256": plan.dataset_sha256,
                "plan_path": str(plan_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Three guards, each for a distinct way the first attempt at this run went
    # wrong. A shorter per-request timeout so a stuck read surfaces as a failure
    # the retry and replacement path handles. Fewer provider-level retries,
    # because each retry is a fully billed call whose tokens never reach the
    # record: the first attempt billed 10.2M input tokens against 3.0M recorded.
    # And a watchdog, because a hung SSL read never errors at all, so the pool
    # never drains and nothing detects it.
    provider = get_provider(plan.model, timeout=timeout, max_retries=sdk_retries)
    calls_path = run_dir / "calls.jsonl"
    stop = threading.Event()

    def _watchdog() -> None:
        last_size = -1
        last_change = time.monotonic()
        while not stop.wait(15.0):
            size = calls_path.stat().st_size if calls_path.exists() else 0
            if size != last_size:
                last_size, last_change = size, time.monotonic()
                continue
            idle = time.monotonic() - last_change
            if idle > stall_seconds:
                console.print(
                    f"[red]stalled: no call completed for {idle:.0f}s. Aborting. "
                    "Rerun the same command to resume from what landed.[/red]"
                )
                # os._exit, not sys.exit: the worker threads are blocked in a
                # socket read that will not return, so a clean shutdown would
                # itself hang and keep billing.
                os._exit(75)

    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        stats = execute_concurrent(
            specs,
            provider,
            calls_path,
            workers=workers,
            max_failures=max_failures,
        )
    finally:
        stop.set()
    console.print(
        f"attempted {stats.attempted}, written {stats.written}, "
        f"skipped {stats.skipped}, failed {stats.failed}"
    )
    if stats.errors:
        for line in stats.errors[:10]:
            console.print(f"[yellow]{line}[/yellow]")

    destination = archive_run(run_dir, Path(archive_to))
    console.print(f"[bold]archived to {destination}[/bold]")


# Runs whose retained call records back a published claim. `ladder` here is the
# SEVERITY ladder from calibration, not the K = 40 fixed-position ladder, which
# is `ladder-k40`. Two unrelated things called "ladder" is a trap; the names are
# kept as they are on disk so existing commands keep working, and the collision
# is called out here and in REPLICATION.md.
# Every run that backs a published claim. `ladder` is the severity ladder from
# calibration; `ladder-k40` is the fixed-position K = 40 experiment. The names are
# confusingly close and mean different things.
CLAIM_RUNS: tuple[str, ...] = (
    "pilot",
    "sweep",
    "calib",
    "calib2",
    "ladder",
    "ladder-k40",
)


@cli.command("reproduce")
@click.option(
    "--expected", "expected_path", type=click.Path(),
    default="replication/expected.json", show_default=True,
)
@click.option(
    "--sums", "sums_path", type=click.Path(), default=None,
    help="Optional SHA256SUMS to verify artifacts against before recomputing.",
)
@click.option("--root", type=click.Path(), default=".", show_default=True)
@click.option(
    "--tolerance", type=float, default=None,
    help="Float comparison tolerance. Counts and ids are always exact.",
)
@click.option(
    "--strict", is_flag=True,
    help="Fail if any recomputed claim value is not pinned in the expected file.",
)
def reproduce_cmd(
    expected_path: str,
    sums_path: str | None,
    root: str,
    tolerance: float | None,
    strict: bool,
) -> None:
    """Recompute every published claim from retained artifacts. No API key.

    Audits the analysis path. It does not audit construction: call records that
    are internally consistent with a wrong experiment still recompute cleanly.
    """
    from .replication import (
        FLOAT_TOLERANCE,
        compare,
        compute_claims,
        coverage_gaps,
        verify_checksums,
    )

    if sums_path:
        problems = verify_checksums(Path(sums_path), Path(root))
        if problems:
            console.print("[red]artifact verification failed[/red]")
            for problem in problems[:20]:
                console.print(f"  {problem}")
            raise click.ClickException(
                f"{len(problems)} artifact problem(s); recomputation not attempted"
            )
        console.print("[green]artifact checksums verified[/green]")

    missing = [name for name in CLAIM_RUNS if not (_run_dir(name) / "calls.jsonl").exists()]
    payload = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    expected = payload["claims"]

    try:
        claims = compute_claims(_load_run)
    except Exception as exc:  # noqa: BLE001 - report which artifact is absent
        console.print(f"[yellow]UNAVAILABLE: {exc}[/yellow]")
        if missing:
            console.print(
                f"missing run artifacts: {', '.join(missing)}. Download the "
                "replication bundle, or see REPLICATION.md."
            )
        raise click.ClickException("recomputation could not run") from exc

    results = compare(
        claims, expected,
        tolerance=FLOAT_TOLERANCE if tolerance is None else tolerance,
    )
    failures = [r for r in results if not r.ok]
    by_section: dict[str, list] = {}
    for result in results:
        by_section.setdefault(result.section, []).append(result)
    for section, rows in by_section.items():
        bad = sum(1 for r in rows if not r.ok)
        state = "[green]PASS[/green]" if not bad else "[red]MISMATCH[/red]"
        console.print(f"\n{section}: {state} ({len(rows) - bad}/{len(rows)})")
        for result in rows:
            if not result.ok:
                console.print(f"[red]{result.line}[/red]")

    gaps = coverage_gaps(claims, expected)
    if gaps:
        console.print("\n[yellow]unpinned, so recomputed but not verified:[/yellow]")
        for gap in gaps:
            console.print(f"[yellow]  {gap}[/yellow]")

    console.print()
    if failures:
        raise click.ClickException(
            f"MISMATCH: {len(failures)} of {len(results)} values disagree"
        )
    if gaps and strict:
        raise click.ClickException(
            f"{len(gaps)} computed value(s) are not pinned; --strict treats an "
            "unverified claim as a failure"
        )
    console.print(
        f"[bold green]PASS[/bold green]: {len(results)} published values "
        "recomputed from retained artifacts"
    )
    # State the boundary in the same breath as the result. A PASS that reads as
    # "the whole document is verified" is worse than no check at all.
    console.print(
        "[dim]covers METHODS 8.1, 8.3, 8.3.2 and 8.5. Not covered here: the "
        "calibration gate (3.3, 8.2), which needs `packing calibration-report`, "
        "and the ladder diagnostics (8.5.2, 8.5.3), which need "
        "`packing ladder-analyze`.[/dim]"
    )
    if missing:
        console.print(
            f"[yellow]note: {', '.join(missing)} absent, so claims backed only "
            "by those runs were not checked[/yellow]"
        )


@cli.command("bundle")
@click.option("--out", "out_dir", type=click.Path(), default="dist", show_default=True)
@click.option("--version", default="v1", show_default=True)
@click.option(
    "--items", "items_path", type=click.Path(),
    default="data/helpsteer2.jsonl", show_default=True,
)
def bundle_cmd(out_dir: str, version: str, items_path: str) -> None:
    """Build the checksum-pinned replication bundle. No API key, no upload.

    Publishing the archive is a separate, deliberate step.
    """
    import subprocess
    import tarfile

    from .replication import file_sha256, ordered_item_id_sha256

    frame = Path(items_path)
    if not frame.exists():
        raise click.ClickException(f"{frame} not found")

    name = f"packing-replication-{version}"
    staging = Path(out_dir) / name
    if staging.exists():
        raise click.ClickException(
            f"{staging} already exists; bundles are never overwritten"
        )
    (staging / "data").mkdir(parents=True)
    shutil.copy2(frame, staging / "data" / frame.name)

    present, absent = [], []
    for run in CLAIM_RUNS:
        source = _run_dir(run)
        if not (source / "calls.jsonl").exists():
            absent.append(run)
            continue
        destination = staging / "runs" / run
        destination.mkdir(parents=True)
        for filename in ("calls.jsonl", "items.jsonl", "meta.json"):
            candidate = source / filename
            if candidate.exists():
                shutil.copy2(candidate, destination / filename)
        present.append(run)

    (staging / "environment").mkdir()
    for filename in ("uv.lock", "pyproject.toml"):
        if Path(filename).exists():
            shutil.copy2(filename, staging / "environment" / filename)
    shutil.copy2("replication/expected.json", staging / "expected.json")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    manifest = {
        "bundle": name,
        "code_commit": commit,
        "runs_included": present,
        "runs_absent": absent,
        "model": "gpt-5-mini",
        "model_is_an_alias": True,
        "request_settings": {
            "temperature": None,
            "reasoning_effort": "low",
            "structured_outputs": "strict",
        },
        "frame": {
            "path": f"data/{frame.name}",
            "dataset_sha256": file_sha256(frame),
            "ordered_item_id_sha256": ordered_item_id_sha256(frame),
            "dataset_source": "nvidia/HelpSteer2",
            "dataset_licence": "CC-BY-4.0",
            "fetch_command": f"packing fetch --limit {sum(1 for _ in frame.open())}",
            "n_items": sum(1 for _ in frame.open()),
        },
        "not_recorded": [
            "provider response ids",
            "per-call wall-clock timestamps",
            "pinned model snapshot (gpt-5-mini is an alias)",
        ],
        "recompute_command": "packing reproduce --sums SHA256SUMS",
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{file_sha256(path)}  {path.relative_to(staging)}")
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive = Path(out_dir) / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname=name)

    size_mb = archive.stat().st_size / 1e6
    console.print(f"runs included: {', '.join(present)}")
    if absent:
        console.print(f"[yellow]runs absent: {', '.join(absent)}[/yellow]")
    console.print(f"files: {len(lines)}")
    console.print(f"[bold]{archive}[/bold]  {size_mb:.1f} MB")
    console.print(f"archive sha256: {file_sha256(archive)}")
    console.print("\nNothing was uploaded. Publishing the archive is a separate step.")


@cli.command("ladder-analyze")
@click.option("--run", "name", default="ladder-k40", show_default=True)
@click.option("--reference", type=int, default=1, show_default=True)
@click.option("--comparison", type=int, default=40, show_default=True)
def ladder_analyze_cmd(name: str, reference: int, comparison: int) -> None:
    """The ladder's pre-registered estimand, plus its diagnostics.

    The generic `analyze` command reports the sweep's stratified recovery tables,
    which is not this design's primary quantity. The primary here is one
    equal-weighted contrast across the eight core operations.
    """
    from .eligibility import CORE_OPERATIONS, NUMBERS_OPERATION
    from .ladder_criteria import LADDER_CRITERIA

    items, criteria, records, meta = _load_run(name)
    frame = analyze.to_frame(analyze.build_outcomes(records, items, LADDER_CRITERIA))
    if frame.empty:
        raise click.ClickException("no outcomes to analyse")
    frame = analyze.annotate_synthetic(frame, items)

    # Recovery is defined on the targeted criterion only. Companions are neutral
    # by construction and contribute ties, not outcomes.
    targeted = analyze.synthetic_recovery_frame(frame[frame["is_target"]])
    core = targeted[targeted["operation"].isin(CORE_OPERATIONS)]
    numbers = targeted[targeted["operation"] == NUMBERS_OPERATION]

    console.print(
        f"[bold]run {name}[/bold]  manifest {str(meta.get('manifest_sha256'))[:16]}  "
        f"seed {meta.get('seed')}  levels {meta.get('levels')}"
    )
    console.print(
        f"outcomes: {len(targeted)} targeted "
        f"({len(core)} core, {len(numbers)} numbers)"
    )

    _show(
        targeted.groupby(["operation", "group_size"])["correct_num"]
        .agg(["size", "mean"])
        .round(4),
        "Directional recovery by operation and rung (descriptive)",
    )

    console.print("\n[bold]PRIMARY: equal-weighted core contrast[/bold]")
    primary = analyze.stratified_endpoint(
        core, reference, comparison, operations=tuple(CORE_OPERATIONS)
    )
    if primary is None:
        console.print(
            "[yellow]not estimable yet: an operation is missing or thin. "
            "The estimand is an average over all eight, so a subset is a "
            "different quantity.[/yellow]"
        )
    else:
        console.print(json.dumps(primary.as_row(), indent=2))
        console.print(f"per operation: {json.dumps(primary.per_operation)}")

    console.print("\n[bold]Sensitivity to the unevaluable rule[/bold]")
    for rule in ("primary", "lower_bound", "upper_bound", "complete_case"):
        alt = analyze.to_frame(
            analyze.build_outcomes(
                records,
                items,
                LADDER_CRITERIA,
                unevaluable_rule=rule,
                reference_size=reference,
            )
        )
        if alt.empty:
            console.print(f"  {rule:<14} no outcomes")
            continue
        alt = analyze.annotate_synthetic(alt, items)
        alt_core = analyze.synthetic_recovery_frame(alt[alt["is_target"]])
        alt_core = alt_core[alt_core["operation"].isin(CORE_OPERATIONS)]
        result = analyze.stratified_endpoint(
            alt_core, reference, comparison, operations=tuple(CORE_OPERATIONS)
        )
        if result is None:
            console.print(f"  {rule:<14} not estimable")
        else:
            console.print(
                f"  {rule:<14} theta {result.theta:+.4f}  "
                f"CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]  "
                f"p {result.p_value:.4g}"
            )

    console.print("\n[bold]KEY SECONDARY: numbers correctness sentinel[/bold]")
    paired = analyze.paired_contrast(numbers, reference, comparison)
    if paired is None:
        console.print("[yellow]not estimable yet[/yellow]")
    else:
        console.print(json.dumps(paired.as_row(), indent=2))

    _show(
        analyze.synthetic_recovery_frame(frame[frame["is_target"]])
        .groupby("group_size")
        .agg(
            n=("correct_num", "size"),
            recovery=("correct_num", "mean"),
            unevaluable=("unevaluable", "mean"),
        )
        .round(4),
        "Diagnostics by rung: recovery and unevaluable frequency",
    )
    _show(
        frame.groupby("group_size")["judge_tie"].agg(["size", "mean"]).round(4),
        "Judge tie rate by rung (diagnostic secondary)",
    )
