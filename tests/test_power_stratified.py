"""Planning variance for the equal-quota estimator. Under fixed quotas the
relevant variance is the average WITHIN-operation variance, not the variance
pooled across all differences, which also carries between-operation mean spread."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packing.power import sigma_eff, stratified_mde, stratified_sample_size

OPS = ("op1", "op2")


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "base_unit": u,
                "operation": o,
                "group_size": g,
                "correct_num": v,
                "resolved": True,
                "judge_tie": False,
            }
            for u, o, g, v in rows
        ]
    )


def _paired(prefix, operation, diffs):
    rows = []
    for i, d in enumerate(diffs):
        rows += [
            (f"{prefix}{i}", operation, 1, 0.0),
            (f"{prefix}{i}", operation, 40, d),
        ]
    return rows


def test_sigma_eff_ignores_between_operation_mean_spread():
    # Both strata have zero within variance but opposite means. A pooled SD would
    # be large; the equal-quota planning variance is zero.
    rows = _paired("a", "op1", [1.0] * 5) + _paired("b", "op2", [-1.0] * 5)
    assert sigma_eff(_frame(rows), 1, 40, operations=OPS) == 0.0


def test_sigma_eff_averages_variances_not_standard_deviations():
    rng = np.random.default_rng(0)
    rows = _paired("a", "op1", rng.normal(0, 0.2, 400))
    rows += _paired("b", "op2", rng.normal(0, 0.6, 400))
    expected = float(np.sqrt((0.2**2 + 0.6**2) / 2))
    got = sigma_eff(_frame(rows), 1, 40, operations=OPS)
    assert abs(got - expected) < 0.05, (got, expected)


def test_sigma_eff_returns_none_when_a_stratum_is_missing():
    rows = _paired("a", "op1", [0.1, 0.2, 0.3])
    assert sigma_eff(_frame(rows), 1, 40, operations=OPS) is None


def test_sigma_eff_returns_none_when_a_stratum_is_a_single_unit():
    rows = _paired("a", "op1", [0.1, 0.2, 0.3]) + _paired("b", "op2", [0.1])
    assert sigma_eff(_frame(rows), 1, 40, operations=OPS) is None


def test_sample_size_matches_the_preregistered_figures():
    # Spec section 8: sigma_eff 0.3345 needs 354 for a 0.05 effect, and the
    # CI-high 0.3763 needs 447. A planning function that disagrees with the
    # protocol it plans is worse than useless.
    assert stratified_sample_size(0.3345, 0.05, strata=8) == 354
    assert stratified_sample_size(0.3763, 0.05, strata=8) == 447


def test_sample_size_is_the_exact_form_not_the_normal_approximation():
    # The normal approximation gives 352 and 445 for these inputs.
    assert stratified_sample_size(0.3345, 0.05, strata=8) != 352
    assert stratified_sample_size(0.3763, 0.05, strata=8) != 445


def test_mde_and_sample_size_round_trip_within_the_approximation_gap():
    # stratified_sample_size is exact and stratified_mde is the normal
    # approximation, so they agree closely but not identically.
    n = stratified_sample_size(0.3345, 0.05, strata=8)
    assert abs(stratified_mde(0.3345, n, strata=8) - 0.05) < 0.001


def test_sample_size_rejects_a_nonpositive_effect():
    with pytest.raises(ValueError, match="mde must be > 0"):
        stratified_sample_size(0.3, 0.0, strata=8)


def test_sample_size_rejects_a_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma must be > 0"):
        stratified_sample_size(0.0, 0.05, strata=8)
