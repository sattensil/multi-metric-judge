"""The primary estimator: equal weight per operation, heteroskedastic SE, and
Welch-Satterthwaite degrees of freedom. Equal quotas define the weights, so the
estimator has to apply them rather than average all rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packing.analyze import (
    cluster_wald_nonlinearity,
    operation_heterogeneity,
    paired_contrast,
    stratified_endpoint,
)

OPS = ("op1", "op2")


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "base_unit": unit,
                "operation": operation,
                "group_size": size,
                "correct_num": value,
                "judge_tie": False,
                # `resolved` is what analyze.resolved() filters on (analyze.py:406).
                "resolved": True,
            }
            for unit, operation, size, value in rows
        ]
    )


def test_returns_none_without_both_endpoints():
    frame = _frame([("u1", "op1", 1, 1.0)])
    assert stratified_endpoint(frame, 1, 40, operations=OPS) is None


def _paired(prefix, operation, diffs):
    """Rows whose per-unit difference is exactly each value in `diffs`."""
    rows = []
    for i, d in enumerate(diffs):
        rows += [
            (f"{prefix}{i}", operation, 1, 0.0),
            (f"{prefix}{i}", operation, 40, d),
        ]
    return rows


def test_equal_weighting_ignores_unequal_operation_counts():
    # op1 contributes 6 units averaging +1.0, op2 contributes 3 averaging 0.0.
    # A flat mean over units would give 6/9 = +0.667; equal weighting gives +0.5.
    rows = _paired("a", "op1", [1.0, 1.0, 1.0, 1.0, 1.2, 0.8])
    rows += _paired("b", "op2", [0.1, -0.1, 0.0])
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result is not None
    assert result.theta == 0.5


def test_per_operation_means_and_counts_are_reported():
    rows = _paired("a", "op1", [0.8, 1.0, 1.2, 1.0])
    rows += _paired("b", "op2", [-0.8, -1.0, -1.2, -1.0])
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result.per_operation["op1"] == (1.0, 4)
    assert result.per_operation["op2"] == (-1.0, 4)
    assert result.theta == 0.0


def test_zero_variance_in_every_stratum_returns_none():
    # The SE is genuinely zero and a t statistic is undefined, so there is no
    # result to report rather than an infinitely significant one.
    rows = _paired("a", "op1", [1.0, 1.0, 1.0])
    rows += _paired("b", "op2", [1.0, 1.0, 1.0])
    assert stratified_endpoint(_frame(rows), 1, 40, operations=OPS) is None


def test_satterthwaite_df_lies_between_the_smallest_stratum_and_the_total():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, rng.normal(0.1, 0.3))]
    for i in range(10):
        rows += [(f"b{i}", "op2", 1, 0.0), (f"b{i}", "op2", 40, rng.normal(0.1, 0.9))]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert 9 <= result.df <= 40, result.df


def test_a_real_shift_is_detected():
    rng = np.random.default_rng(1)
    rows = []
    for op in OPS:
        for i in range(60):
            rows += [
                (f"{op}{i}", op, 1, 0.0),
                (f"{op}{i}", op, 40, rng.normal(-0.30, 0.20)),
            ]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result.theta < -0.2
    assert result.p_value < 0.01
    assert result.ci_high < 0


def test_a_null_is_not_detected():
    rng = np.random.default_rng(2)
    rows = []
    for op in OPS:
        for i in range(60):
            rows += [
                (f"{op}{i}", op, 1, 0.0),
                (f"{op}{i}", op, 40, rng.normal(0.0, 0.20)),
            ]
    result = stratified_endpoint(_frame(rows), 1, 40, operations=OPS)
    assert result.p_value > 0.05
    assert result.ci_low < 0 < result.ci_high


def test_a_stratum_with_too_few_units_is_rejected():
    rows = [("a0", "op1", 1, 0.0), ("a0", "op1", 40, 1.0)]
    for i in range(4):
        rows += [(f"b{i}", "op2", 1, 0.0), (f"b{i}", "op2", 40, 1.0)]
    assert stratified_endpoint(_frame(rows), 1, 40, operations=OPS) is None


def test_a_missing_operation_is_rejected():
    rows = []
    for i in range(5):
        rows += [(f"a{i}", "op1", 1, 0.0), (f"a{i}", "op1", 40, 1.0)]
    assert stratified_endpoint(_frame(rows), 1, 40, operations=OPS) is None


def test_a_frame_without_an_operation_column_raises():
    import pytest

    frame = pd.DataFrame(
        [
            {
                "base_unit": "u",
                "group_size": 1,
                "correct_num": 1.0,
                "judge_tie": False,
                "resolved": True,
            }
        ]
    )
    with pytest.raises(ValueError, match="no `operation` column"):
        stratified_endpoint(frame, 1, 40, operations=OPS)


def test_operation_heterogeneity_reports_holm_and_omnibus_tests():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(40):
        rows += [
            (f"a{i}", "op1", 1, 1.0),
            (f"a{i}", "op1", 40, float(rng.random() < 0.3)),
            (f"b{i}", "op2", 1, float(rng.random() < 0.7)),
            (f"b{i}", "op2", 40, float(rng.random() < 0.7)),
        ]
    het = operation_heterogeneity(_frame(rows), 1, 40, operations=OPS)
    assert "p_holm" in het["table"].columns
    assert list(het["table"].index) == ["op1", "op2"]
    assert het["table"].loc["op1", "p_holm"] < 0.01
    assert het["anova_p"] < 0.01
    assert het["kruskal_p"] < 0.01


def test_cluster_wald_nonlinearity_runs_across_multiple_rungs():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(80):
        for k in (1, 10, 20, 40):
            prob = 0.80 - 0.003 * (k - 1)
            rows.append((f"u{i}", "op1", k, float(rng.random() < prob)))
    res = cluster_wald_nonlinearity(_frame(rows))
    assert res["df"] == 2
    assert 0.0 <= res["p"] <= 1.0


def test_paired_contrast_supports_tie_tolerant_credit_and_tie_rate():
    frame = pd.DataFrame(
        [
            {"item_id": "i1", "group_size": 1, "correct_num": np.nan, "credit": 0.5, "judge_tie": True, "human_tie": False, "resolved": False},
            {"item_id": "i1", "group_size": 5, "correct_num": np.nan, "credit": 0.5, "judge_tie": True, "human_tie": False, "resolved": False},
            {"item_id": "i2", "group_size": 1, "correct_num": 1.0, "credit": 1.0, "judge_tie": False, "human_tie": False, "resolved": True},
            {"item_id": "i2", "group_size": 5, "correct_num": 0.0, "credit": 0.0, "judge_tie": False, "human_tie": False, "resolved": True},
            {"item_id": "i3", "group_size": 1, "correct_num": 1.0, "credit": 1.0, "judge_tie": False, "human_tie": False, "resolved": True},
            {"item_id": "i3", "group_size": 5, "correct_num": 1.0, "credit": 1.0, "judge_tie": False, "human_tie": False, "resolved": True},
            {"item_id": "i4", "group_size": 1, "correct_num": 0.0, "credit": 0.0, "judge_tie": False, "human_tie": False, "resolved": True},
            {"item_id": "i4", "group_size": 5, "correct_num": np.nan, "credit": 0.5, "judge_tie": True, "human_tie": False, "resolved": False},
        ]
    )
    credit_res = paired_contrast(frame, 1, 5, column="credit", resolved_only=False)
    assert credit_res is not None
    assert credit_res.n_items == 4
    tie_res = paired_contrast(frame, 1, 5, column="judge_tie", resolved_only=False)
    assert tie_res is not None
    assert tie_res.n_items == 4

