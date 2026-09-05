"""Sample-size planning, following Miller, *Adding Error Bars to Evals*.

Two forms are provided because they answer different questions:

- ``miller_sample_size`` is the paper's general two-condition formula, useful for
  planning from assumed variances before any data exists.
- ``paired_sample_size`` works from the *observed* standard deviation of per-item
  differences, which is what the pilot actually measures and is strictly better
  once you have it. It is also the form that matches the primary estimator, since
  the paired contrast is what will be reported.

The pilot exists to turn assumptions into measurements: run it, call
``plan_from_pilot``, then decide whether the confirmatory run is affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .analyze import independence_unit, resolved


def _z(alpha: float, power: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


# ------------------------------------------------- Miller's general formula ---


def miller_sample_size(
    *,
    omega_sq: float,
    sigma_sq_a: float,
    sigma_sq_b: float,
    k_a: int = 1,
    k_b: int = 1,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """n = (z_{a/2} + z_b)^2 (w^2 + s_A^2/K_A + s_B^2/K_B) / d^2."""
    if mde <= 0:
        raise ValueError("mde must be > 0")
    variance = omega_sq + sigma_sq_a / max(1, k_a) + sigma_sq_b / max(1, k_b)
    return int(np.ceil(_z(alpha, power) ** 2 * variance / mde**2))


def miller_mde(
    *,
    omega_sq: float,
    sigma_sq_a: float,
    sigma_sq_b: float,
    n: int,
    k_a: int = 1,
    k_b: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """d = (z_{a/2} + z_b) sqrt((w^2 + s_A^2/K_A + s_B^2/K_B) / n)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    variance = omega_sq + sigma_sq_a / max(1, k_a) + sigma_sq_b / max(1, k_b)
    return float(_z(alpha, power) * np.sqrt(variance / n))


# --------------------------------------------------------- paired variants ---


def paired_sample_size(
    *, diff_sd: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Items needed to detect ``mde`` in the within-item paired difference."""
    if mde <= 0:
        raise ValueError("mde must be > 0")
    if diff_sd <= 0:
        raise ValueError("diff_sd must be > 0")
    return int(np.ceil((_z(alpha, power) * diff_sd / mde) ** 2))


def paired_mde(
    *, diff_sd: float, n_items: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    return float(_z(alpha, power) * diff_sd / np.sqrt(n_items))


# ------------------------------------------------------------ from a pilot ---


@dataclass
class PilotEstimate:
    reference_size: int
    comparison_size: int
    n_items: int
    observed_difference: float
    diff_sd: float
    mde_at_pilot_n: float
    required_n_for_observed: int | None
    required_n_grid: dict[float, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_size": self.reference_size,
            "comparison_size": self.comparison_size,
            "pilot_n_items": self.n_items,
            "observed_difference": round(self.observed_difference, 4),
            "per_item_difference_sd": round(self.diff_sd, 4),
            "mde_at_pilot_n": round(self.mde_at_pilot_n, 4),
            "required_n_for_observed_effect": self.required_n_for_observed,
            "required_n_by_mde": self.required_n_grid,
        }


def per_item_differences(
    frame: pd.DataFrame, reference_size: int, comparison_size: int
) -> np.ndarray:
    """Per-item accuracy differences, the unit the paired estimator operates on."""
    res = resolved(frame)
    cluster = independence_unit(res)
    per_item = (
        res[res["group_size"].isin([reference_size, comparison_size])]
        .groupby([cluster, "group_size"])["correct_num"]
        .mean()
        .unstack("group_size")
    )
    if reference_size not in per_item or comparison_size not in per_item:
        return np.array([])
    paired = per_item[[reference_size, comparison_size]].dropna()
    return (paired[comparison_size] - paired[reference_size]).to_numpy()


def plan_from_pilot(
    frame: pd.DataFrame,
    reference_size: int,
    comparison_size: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    mde_grid: tuple[float, ...] = (0.10, 0.05, 0.03, 0.02),
) -> PilotEstimate | None:
    """Turn pilot data into a confirmatory sample size.

    ``required_n_for_observed`` answers "how many items to reliably detect an
    effect this size", and the grid answers "what if the true effect is smaller
    than the pilot happened to show" -- which it usually is, since a pilot that
    reached significance is more likely to have overshot than undershot.
    """
    diffs = per_item_differences(frame, reference_size, comparison_size)
    if diffs.size < 3:
        return None
    sd = float(diffs.std(ddof=1))
    if sd <= 0:
        return None
    observed = float(diffs.mean())

    required = None
    if abs(observed) > 1e-9:
        required = paired_sample_size(
            diff_sd=sd, mde=abs(observed), alpha=alpha, power=power
        )

    return PilotEstimate(
        reference_size=reference_size,
        comparison_size=comparison_size,
        n_items=int(diffs.size),
        observed_difference=observed,
        diff_sd=sd,
        mde_at_pilot_n=paired_mde(
            diff_sd=sd, n_items=int(diffs.size), alpha=alpha, power=power
        ),
        required_n_for_observed=required,
        required_n_grid={
            mde: paired_sample_size(diff_sd=sd, mde=mde, alpha=alpha, power=power)
            for mde in mde_grid
        },
    )
