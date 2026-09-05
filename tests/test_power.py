"""Power formulas, validated against the published worked example.

Miller, *Adding Error Bars to Evals* (arXiv 2411.00640), reports that detecting
delta = 0.03 at 80% power with alpha = 0.05 "will need to contain at least ...
approximately 969 independent questions". Reproducing that number from the
implementation is a check against a primary source, not a self-consistency check.

Back-solving the paper's figure: (z_0.025 + z_0.80)^2 = 2.80158^2 = 7.8488, and
969 * 0.03^2 / 7.8488 = 0.1111, so the total variance term under his stated
assumptions is 1/9. That value is used below.
"""

from __future__ import annotations

import numpy as np
import pytest

from packing.power import (
    miller_mde,
    miller_sample_size,
    paired_mde,
    paired_sample_size,
)

MILLER_TOTAL_VARIANCE = 1 / 9


def test_reproduces_millers_969_questions() -> None:
    n = miller_sample_size(
        omega_sq=MILLER_TOTAL_VARIANCE,
        sigma_sq_a=0.0,
        sigma_sq_b=0.0,
        mde=0.03,
        alpha=0.05,
        power=0.80,
    )
    assert n == pytest.approx(969, abs=1)


def test_mde_and_sample_size_are_inverses() -> None:
    n = miller_sample_size(
        omega_sq=MILLER_TOTAL_VARIANCE, sigma_sq_a=0.0, sigma_sq_b=0.0, mde=0.03
    )
    recovered = miller_mde(
        omega_sq=MILLER_TOTAL_VARIANCE, sigma_sq_a=0.0, sigma_sq_b=0.0, n=n
    )
    assert recovered == pytest.approx(0.03, rel=1e-3)


def test_resampling_reduces_required_n() -> None:
    """Miller's variance-reduction claim: more samples per question buys power."""
    kwargs = dict(omega_sq=0.05, sigma_sq_a=0.25, sigma_sq_b=0.25, mde=0.05)
    n1 = miller_sample_size(**kwargs, k_a=1, k_b=1)
    n4 = miller_sample_size(**kwargs, k_a=4, k_b=4)
    n10 = miller_sample_size(**kwargs, k_a=10, k_b=10)
    assert n1 > n4 > n10


def test_resampling_mde_direction_matches_paper() -> None:
    """K from 1 to 10 must shrink the MDE substantially, as reported."""
    kwargs = dict(omega_sq=0.05, sigma_sq_a=0.25, sigma_sq_b=0.25, n=100)
    mde1 = miller_mde(**kwargs, k_a=1, k_b=1)
    mde10 = miller_mde(**kwargs, k_a=10, k_b=10)
    assert mde10 < mde1
    assert mde10 / mde1 < 0.75


def test_conditional_variance_cannot_be_resampled_away() -> None:
    """omega^2 is irreducible: infinite resampling still leaves the between-condition
    variance, so n has a floor."""
    kwargs = dict(omega_sq=0.05, sigma_sq_a=0.25, sigma_sq_b=0.25, mde=0.05)
    huge = miller_sample_size(**kwargs, k_a=10_000, k_b=10_000)
    floor = miller_sample_size(
        omega_sq=0.05, sigma_sq_a=0.0, sigma_sq_b=0.0, mde=0.05
    )
    assert huge == pytest.approx(floor, rel=0.01)


# ------------------------------------------------------------------ paired ---


def test_paired_formulas_are_inverses() -> None:
    n = paired_sample_size(diff_sd=0.33, mde=0.05)
    assert paired_mde(diff_sd=0.33, n_items=n) == pytest.approx(0.05, rel=1e-2)


def test_paired_sample_size_scales_as_inverse_square_of_mde() -> None:
    """Halving the detectable effect must quadruple the item count."""
    n_big = paired_sample_size(diff_sd=0.33, mde=0.10)
    n_small = paired_sample_size(diff_sd=0.33, mde=0.05)
    assert n_small / n_big == pytest.approx(4.0, rel=0.05)


def test_higher_power_needs_more_items() -> None:
    assert paired_sample_size(diff_sd=0.3, mde=0.05, power=0.95) > paired_sample_size(
        diff_sd=0.3, mde=0.05, power=0.80
    )


def test_power_functions_reject_nonsense() -> None:
    with pytest.raises(ValueError):
        paired_sample_size(diff_sd=0.3, mde=0.0)
    with pytest.raises(ValueError):
        paired_sample_size(diff_sd=0.0, mde=0.05)
    with pytest.raises(ValueError):
        paired_mde(diff_sd=0.3, n_items=0)
    with pytest.raises(ValueError):
        miller_sample_size(omega_sq=0.1, sigma_sq_a=0, sigma_sq_b=0, mde=0)
    with pytest.raises(ValueError):
        miller_mde(omega_sq=0.1, sigma_sq_a=0, sigma_sq_b=0, n=0)
