"""Cost accounting and the accuracy-versus-cost frontier.

The second outcome of the study. At group_size=1 the item's input tokens are paid
K times; at group_size=K, once. So cost per criterion falls with packing while
accuracy (possibly) falls too, and the deliverable is not "packing is bad" but
"here is the frontier, pick your point on it".

Caching caveat, checked rather than assumed: the major providers require roughly
1024 tokens of stable prefix before a cache engages. Short items may never reach
it, in which case the high-N cost advantage is real and uncached. ``cache_report``
reads the provider's own cached-token counter instead of trusting an assumption.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .budget import near_cap


def usage_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "group_size": r["group_size"],
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "reasoning_tokens": r.get("reasoning_tokens", 0),
            "cached_input_tokens": r.get("cached_input_tokens", 0),
            "schema_tokens": r.get("schema_tokens", 0),
            "latency_ms": r.get("latency_ms", 0.0),
            # Derived from retained token counts so one definition is applied to
            # every record.
            "near_cap": near_cap(
                r["group_size"], r.get("output_tokens", 0), r.get("reasoning_tokens", 0)
            ),
            "model": r.get("model", ""),
        }
        for r in records
        if not r.get("failure")
    ]
    return pd.DataFrame(rows)


def cost_per_criterion(
    usage: pd.DataFrame, price_in_per_m: float, price_out_per_m: float
) -> pd.DataFrame:
    """Spend per criterion scored, by group size.

    Cached input is billed at roughly a tenth of the uncached rate on the major
    providers; both figures are reported so the frontier can be read either way.
    """
    if usage.empty:
        return usage

    frame = usage.copy()
    uncached_in = (frame["input_tokens"] - frame["cached_input_tokens"]).clip(lower=0)
    frame["usd_uncached"] = (
        frame["input_tokens"] / 1e6 * price_in_per_m
        + frame["output_tokens"] / 1e6 * price_out_per_m
    )
    frame["usd_with_cache"] = (
        uncached_in / 1e6 * price_in_per_m
        + frame["cached_input_tokens"] / 1e6 * price_in_per_m * 0.10
        + frame["output_tokens"] / 1e6 * price_out_per_m
    )

    grouped = frame.groupby("group_size").agg(
        calls=("input_tokens", "size"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        reasoning_tokens=("reasoning_tokens", "sum"),
        cached_input_tokens=("cached_input_tokens", "sum"),
        usd_uncached=("usd_uncached", "sum"),
        usd_with_cache=("usd_with_cache", "sum"),
        near_cap_rate=("near_cap", "mean"),
        median_latency_ms=("latency_ms", "median"),
    )
    # Each call at group size g scores g criteria.
    grouped["criteria_scored"] = grouped.index.to_series() * grouped["calls"]
    grouped["usd_per_1k_criteria"] = (
        grouped["usd_uncached"] / grouped["criteria_scored"] * 1000
    )
    grouped["usd_per_1k_criteria_cached"] = (
        grouped["usd_with_cache"] / grouped["criteria_scored"] * 1000
    )
    grouped["input_tokens_per_criterion"] = (
        grouped["input_tokens"] / grouped["criteria_scored"]
    )
    grouped["reasoning_tokens_per_criterion"] = (
        grouped["reasoning_tokens"] / grouped["criteria_scored"]
    )
    return grouped


def cache_report(usage: pd.DataFrame) -> dict[str, Any]:
    """Did caching actually engage? Verified, not assumed."""
    if usage.empty:
        return {"engaged": False, "reason": "no usage records"}
    total_in = int(usage["input_tokens"].sum())
    total_cached = int(usage["cached_input_tokens"].sum())
    return {
        "engaged": total_cached > 0,
        "cached_input_tokens": total_cached,
        "total_input_tokens": total_in,
        "cache_hit_fraction": (total_cached / total_in) if total_in else 0.0,
        "note": (
            "Cache did not engage. Expected when items are shorter than the "
            "provider's minimum cacheable prefix (~1024 tokens); the cost "
            "advantage of larger group sizes is then real and uncached."
            if total_cached == 0
            else "Cache engaged; report both cached and uncached frontiers."
        ),
    }


def frontier(
    accuracy_by_size: pd.DataFrame, cost_by_size: pd.DataFrame
) -> pd.DataFrame:
    """Join accuracy and cost per criterion, and mark the knee.

    The knee is the largest group size whose accuracy is still within one
    standard error of the best observed accuracy -- i.e. the cheapest packing you
    cannot yet distinguish from the best one.
    """
    joined = accuracy_by_size.join(cost_by_size, how="inner")
    if joined.empty or "accuracy" not in joined:
        return joined

    best = joined["accuracy"].max()
    # Binomial SE at each size, for the "indistinguishable from best" rule.
    se = (joined["accuracy"] * (1 - joined["accuracy"]) / joined["n"]).pow(0.5)
    joined["within_1se_of_best"] = joined["accuracy"] >= (best - se)
    candidates = joined.index[joined["within_1se_of_best"]]
    joined["knee"] = False
    if len(candidates):
        joined.loc[max(candidates), "knee"] = True
    return joined
