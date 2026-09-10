"""Estimators for the group-size effect.

Three estimators, in the order the plan specifies, all reported together because
they fail in different ways:

1. ``paired_contrast`` -- within-item paired differences with clustered SEs.
   Highest power, fewest assumptions, and it removes item difficulty exactly
   rather than modelling it. This is the primary.
2. ``fit_logit`` -- logistic regression with cluster-robust (by item) standard
   errors. Gives coefficients, the position covariate, and likelihoods for the
   functional-form test. Always runs.
3. ``fit_mixed`` -- crossed random intercepts for item and criterion. Richest,
   but variational and the least likely to be available, so never the only
   estimator reported.

Naive standard errors are never produced by this module. Observations cluster by
item (K criteria x several group sizes each) and Miller measured clustered SEs
running over 3x larger than naive ones on comparable data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .criteria import Criterion
from .dataset import PairedItem
from .budget import near_cap
from .score import Outcome, make_outcome, quadratic_weighted_kappa
from .unevaluable import FATAL, TRANSIENT, classify_failure, unevaluable_value

SYNTHETIC_POSITIVE_CONTROLS = frozenset(
    {"contradict", "hedge", "lowercase", "offtopic", "slang", "typos"}
)
SYNTHETIC_TEST_OPERATIONS = frozenset(
    {
        "agreement",
        "depunct",
        "drop_answer",
        "duplicate",
        "filler",
        "flatten",
        "numbers",
        "rename",
        "scramble_list",
        "shuffle",
        "truncate",
    }
)
SYNTHETIC_SEMANTIC_OPERATIONS = frozenset(
    {"contradict", "drop_answer", "numbers", "rename", "shuffle", "truncate"}
)

# --------------------------------------------------------------- assembly ----


def _near_cap_of(record: dict[str, Any]) -> bool:
    return near_cap(
        record["group_size"],
        record.get("output_tokens", 0),
        record.get("reasoning_tokens", 0),
    )


def _unevaluable_outcome(
    *,
    item_id: str,
    criterion: str,
    group_size: int,
    position: int,
    nominal_n: int,
    replicate: int,
    model: str,
    partition_hash: str,
    value: float,
) -> Outcome:
    """One outcome for a paired unit the judge did not return readable output for.

    Marked resolved: the composite outcome is defined for this unit, and leaving
    it unresolved would have `resolved()` drop exactly the observations the
    definition exists to retain.
    """
    return Outcome(
        item_id=item_id,
        criterion=criterion,
        group_size=group_size,
        position=position,
        nominal_n=nominal_n,
        replicate=replicate,
        model=model,
        partition_hash=partition_hash,
        human_a=0,
        human_b=0,
        judge_a=0,
        judge_b=0,
        human_tie=False,
        judge_tie=False,
        correct=bool(value),
        credit=value,
        abs_error=float("nan"),
        unevaluable=True,
    )


def build_outcomes(
    records: Iterable[dict[str, Any]],
    items: list[PairedItem],
    criteria: tuple[Criterion, ...],
    *,
    unevaluable_rule: str = "primary",
    reference_size: int = 1,
) -> list[Outcome]:
    """Join per-side judge records into per-(item, criterion) outcomes.

    A criterion contributes an outcome only when *both* sides were judged in the
    same partition -- a pairwise comparison needs both halves. Half-complete
    pairs are dropped and counted rather than imputed.

    Unevaluable output is scored here rather than at the call, because recovery
    is a comparison between the two sides. A call-level rule would emit two
    observations for a prompt whose sides both failed. Exactly one outcome is
    emitted per (item, criterion, rung) regardless of how many sides failed.

    Transient failures are dropped: they are replaced upstream and carry no
    information about the judge.
    """
    by_key: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    for record in records:
        key = (
            record["item_id"],
            record["model"],
            record["replicate"],
            record["partition_hash"],
            record["group_index"],
        )
        by_key.setdefault(key, {})[record["side"]] = record

    items_by_id = {item.item_id: item for item in items}
    criteria_by_key = {c.key: c for c in criteria}

    outcomes: list[Outcome] = []
    for (item_id, model, replicate, partition_hash, _gi), sides in by_key.items():
        if "a" not in sides or "b" not in sides:
            continue
        item = items_by_id.get(item_id)
        if item is None:
            continue
        rec_a, rec_b = sides["a"], sides["b"]

        # Classification is derived from the retained message so analysis and
        # resume apply one definition to every record.
        classes = tuple(
            classify_failure(rec.get("failure") or "")
            for rec in (rec_a, rec_b)
            if rec.get("failure")
        )
        if TRANSIENT in classes:
            continue
        # A terminal failure is a harness or billing fault, never a judge
        # behaviour, so it cannot become a scored miss.
        if FATAL in classes:
            continue

        if rec_a.get("failure") or rec_b.get("failure"):
            value = unevaluable_value(
                unevaluable_rule, rec_a["group_size"], reference_size
            )
            if value is None:
                continue
            for criterion_key, position in rec_a["positions"].items():
                if criterion_key not in criteria_by_key:
                    continue
                if criterion_key != item.targeted_criterion:
                    # Companions carry no gold, so an unevaluable call says
                    # nothing about them.
                    continue
                outcomes.append(
                    _unevaluable_outcome(
                        item_id=item_id,
                        criterion=criterion_key,
                        group_size=rec_a["group_size"],
                        position=position,
                        nominal_n=rec_a["nominal_n"],
                        replicate=replicate,
                        model=model,
                        partition_hash=partition_hash,
                        value=value,
                    )
                )
            continue

        for criterion_key, position in rec_a["positions"].items():
            if criterion_key not in criteria_by_key:
                continue
            if criterion_key not in rec_b["scores"]:
                continue
            outcomes.append(
                make_outcome(
                    item_id=item_id,
                    criterion=criterion_key,
                    group_size=rec_a["group_size"],
                    position=position,
                    nominal_n=rec_a["nominal_n"],
                    replicate=replicate,
                    model=model,
                    partition_hash=partition_hash,
                    human_a=item.human_a[criterion_key],
                    human_b=item.human_b[criterion_key],
                    judge_a=rec_a["scores"][criterion_key],
                    judge_b=rec_b["scores"][criterion_key],
                    output_tokens=rec_a.get("output_tokens", 0)
                    + rec_b.get("output_tokens", 0),
                    schema_tokens=rec_a.get("schema_tokens", 0),
                    # Recomputed rather than trusted: a stored flag can carry a
                    # different near-cap definition, and both inputs are on the
                    # record, so every run is analysable on one definition.
                    near_cap=_near_cap_of(rec_a) or _near_cap_of(rec_b),
                )
            )
    return outcomes


def to_frame(outcomes: list[Outcome]) -> pd.DataFrame:
    frame = pd.DataFrame([o.as_row() for o in outcomes])
    if frame.empty:
        return frame
    frame["resolved"] = frame["correct"].notna()
    frame["correct_num"] = frame["correct"].astype("float")
    return frame


def annotate_synthetic(
    frame: pd.DataFrame,
    items: list[PairedItem],
    *,
    operation_roles: dict[str, str] | None = None,
    detection_classes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach synthetic-gold provenance and corruption-aligned score movement."""
    metadata = {}
    for item in items:
        if item.source != "synthetic" and item.dataset != "synthetic":
            continue
        # `corruption` is the sweep builder's key, `operation` the ladder's.
        operation = item.meta.get("corruption") or item.meta.get("operation")
        direction = item.meta.get("direction")
        corrupted_side = item.meta.get("corrupted_side")
        if not operation or direction not in (-1, 1) or corrupted_side not in ("a", "b"):
            raise ValueError(f"{item.item_id}: incomplete synthetic corruption metadata")
        metadata[item.item_id] = {
            "operation": operation,
            "targeted_criterion": item.targeted_criterion,
            "corruption_direction": direction,
            "corrupted_side": corrupted_side,
            "base_unit": (
                str(item.meta["base_item"])
                if item.meta.get("base_item")
                else item.item_id
            ),
        }

    if frame.empty:
        return frame.copy()
    out = frame.merge(
        pd.DataFrame.from_dict(metadata, orient="index").rename_axis("item_id").reset_index(),
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    if out["operation"].isna().any():
        missing = sorted(out.loc[out["operation"].isna(), "item_id"].unique())
        raise ValueError(f"missing synthetic metadata for items: {missing[:5]}")

    out["is_target"] = out["criterion"] == out["targeted_criterion"]
    if operation_roles is None:
        operation_roles = {
            **{key: "test" for key in SYNTHETIC_TEST_OPERATIONS},
            **{key: "positive_control" for key in SYNTHETIC_POSITIVE_CONTROLS},
        }
    if detection_classes is None:
        detection_classes = {
            key: ("semantic" if key in SYNTHETIC_SEMANTIC_OPERATIONS else "surface")
            for key in SYNTHETIC_TEST_OPERATIONS | SYNTHETIC_POSITIVE_CONTROLS
        }
    out["calibration_role"] = out["operation"].map(operation_roles).fillna("unclassified")
    out["detection_class"] = out["operation"].map(detection_classes).fillna("unclassified")
    out["corrupted_delta"] = np.where(
        out["corrupted_side"] == "a",
        out["judge_a"] - out["judge_b"],
        out["judge_b"] - out["judge_a"],
    )
    out["aligned_delta"] = out["corrupted_delta"] * out["corruption_direction"]
    out["absolute_delta"] = out["corrupted_delta"].abs()
    return out


def synthetic_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    """Targeted corruption recovery: hit, tie, and wrong-direction rates by size.

    Gold is directional recovery of a known operation, not criterion-isolated
    human-quality accuracy. Hits require the judge to move the targeted score in
    the seeded direction; ties and wrong directions are misses.
    """
    target = frame[frame["is_target"]].copy()
    group_columns = [
        column
        for column in ("calibration_role", "detection_class", "group_size")
        if column in target
    ]
    rows = []
    for keys, chunk in target.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        n = len(chunk)
        evaluable = (
            chunk[~chunk["unevaluable"]]
            if "unevaluable" in chunk
            else chunk
        )
        n_evaluable = len(evaluable)
        hits = int((evaluable["correct"] == True).sum())  # noqa: E712
        wrong = int((evaluable["correct"] == False).sum())  # noqa: E712
        ties = int(evaluable["judge_tie"].sum())
        decided = hits + wrong
        row.update(
            {
                "n": n,
                "n_evaluable": n_evaluable,
                "hits": hits,
                "wrong": wrong,
                "ties": ties,
                "recovery_rate": hits / n if n else np.nan,
                "tie_rate": ties / n_evaluable if n_evaluable else np.nan,
                "unevaluable_rate": (
                    (n - n_evaluable) / n if n else np.nan
                ),
                "discrimination": hits / decided if decided else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index(group_columns).sort_index()


def judge_tie_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Judge tie rate by rung among outcomes the judge actually evaluated."""
    evaluable = frame[~frame["unevaluable"]] if "unevaluable" in frame else frame
    if evaluable.empty:
        return pd.DataFrame(columns=["n_evaluable", "judge_tie_rate"])
    return (
        evaluable.groupby("group_size")["judge_tie"]
        .agg(["size", "mean"])
        .rename(
            columns={
                "size": "n_evaluable",
                "mean": "judge_tie_rate",
            }
        )
        .round(4)
    )


def synthetic_recovery_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Inference frame where target ties and wrong directions are misses.

    Synthetic gold asserts that the corruption moved the targeted score. A judge
    tie means the operation was not recovered; dropping it would condition on
    the judge having recovered the treatment.
    """
    target = frame[frame["is_target"]].copy()
    target["recovered"] = target["correct"] == True  # noqa: E712
    target["correct_num"] = target["recovered"].astype(float)
    target["resolved"] = True
    return target


def synthetic_specificity(frame: pd.DataFrame) -> pd.DataFrame:
    """Rate at which untouched companion criteria remain unchanged."""
    companions = frame[~frame["is_target"]].copy()
    group_columns = ["group_size"]
    columns = [
        "n",
        "specificity",
        "movement_rate",
        "mean_absolute_delta",
        "mean_aligned_delta",
    ]
    if companions.empty:
        return pd.DataFrame(columns=columns).rename_axis("group_size")
    rows = []
    for keys, chunk in companions.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(chunk)
        ties = int(chunk["judge_tie"].sum())
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n": n,
                "specificity": ties / n if n else np.nan,
                "movement_rate": (n - ties) / n if n else np.nan,
                "mean_absolute_delta": (chunk["judge_a"] - chunk["judge_b"]).abs().mean(),
                "mean_aligned_delta": chunk["aligned_delta"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index(group_columns).sort_index()


def synthetic_leakage_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Operation × criterion matrix of systematic corruption-aligned movement.

    Averaging signed, aligned deltas lets independent judge noise cancel instead
    of misclassifying every score fluctuation as construct leakage.
    """
    return (
        frame.groupby(["operation", "criterion"])["aligned_delta"]
        .mean()
        .unstack("criterion")
        .sort_index()
    )


def synthetic_leakage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each operation's strongest companion movement with its target."""
    rows = []
    for operation, chunk in frame.groupby("operation"):
        targets = chunk[chunk["is_target"]]
        companions = chunk[~chunk["is_target"]]
        call_keys = [
            key
            for key in ("item_id", "model", "replicate", "partition_hash")
            if key in chunk.columns
        ]
        if "partition_hash" not in call_keys:
            call_keys.append("group_size")

        comparisons = []
        for criterion, companion_rows in companions.groupby("criterion"):
            matched = companion_rows[call_keys + ["aligned_delta"]].merge(
                targets[call_keys + ["aligned_delta"]],
                on=call_keys,
                how="inner",
                suffixes=("_companion", "_target"),
                validate="many_to_one",
            )
            if matched.empty:
                continue
            companion_mean = float(matched["aligned_delta_companion"].mean())
            target_mean = float(matched["aligned_delta_target"].mean())
            ratio = (
                abs(companion_mean) / abs(target_mean)
                if target_mean != 0
                else np.nan
            )
            comparisons.append((criterion, companion_mean, target_mean, ratio))

        if comparisons:
            worst_criterion, worst_mean, target_mean, ratio = max(
                comparisons,
                key=lambda result: (
                    result[3] if np.isfinite(result[3]) else float("-inf")
                ),
            )
        else:
            worst_criterion = None
            worst_mean = ratio = np.nan
            target_mean = (
                float(targets["aligned_delta"].mean()) if len(targets) else np.nan
            )
        rows.append(
            {
                "operation": operation,
                "target_mean_aligned_delta": target_mean,
                "worst_companion": worst_criterion,
                "worst_companion_mean_aligned_delta": worst_mean,
                "worst_to_target_ratio": ratio,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("operation").sort_index()


def synthetic_frontier(
    frame: pd.DataFrame, cost_by_size: pd.DataFrame
) -> pd.DataFrame:
    """Descriptive Pareto frontier for calibrated test operations.

    No naive standard error or ad-hoc "knee" is used. Inferential uncertainty is
    supplied separately by the paired contrasts; this table only marks points
    that are not dominated on both observed recovery and cost.
    """
    recovery = synthetic_recovery(frame).reset_index()
    recovery = recovery[recovery["calibration_role"] == "test"]
    if recovery.empty or cost_by_size.empty:
        return pd.DataFrame()
    joined = recovery.merge(
        cost_by_size.reset_index(), on="group_size", how="inner", validate="many_to_one"
    )
    joined["pareto_efficient"] = True
    cost_column = "usd_per_1k_criteria"
    for detection_class, indexes in joined.groupby("detection_class").groups.items():
        del detection_class
        for index in indexes:
            row = joined.loc[index]
            competitors = joined.loc[indexes]
            dominates = (
                (competitors[cost_column] <= row[cost_column])
                & (competitors["recovery_rate"] >= row["recovery_rate"])
                & (
                    (competitors[cost_column] < row[cost_column])
                    | (competitors["recovery_rate"] > row["recovery_rate"])
                )
            )
            if dominates.any():
                joined.loc[index, "pareto_efficient"] = False
    return joined.set_index(["detection_class", "group_size"]).sort_index()


def resolved(frame: pd.DataFrame) -> pd.DataFrame:
    """Observations with a defined pairwise outcome (no human tie, no judge tie)."""
    return frame[frame["resolved"]].copy()


def independence_unit(frame: pd.DataFrame) -> str:
    """Column identifying statistically independent source responses."""
    return "base_unit" if "base_unit" in frame.columns else "item_id"


# ----------------------------------------------------------- descriptives ----


def descriptives(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Marginal means. Description only -- never inference; the SEs here ignore clustering."""
    res = resolved(frame)
    evaluable = (
        ~frame["unevaluable"]
        if "unevaluable" in frame
        else pd.Series(True, index=frame.index)
    )
    out = {
        "by_group_size": res.groupby("group_size")["correct_num"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"}),
        "by_position": res.groupby("position")["correct_num"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"}),
        "by_criterion_group_size": res.groupby(["criterion", "group_size"])["correct_num"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"}),
        # Co-primary: judge indecision may itself rise with group size, and
        # excluding it from the accuracy model is only defensible if it is
        # reported separately.
        "judge_tie_by_group_size": frame[~frame["human_tie"] & evaluable]
        .groupby("group_size")["judge_tie"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "judge_tie_rate", "count": "n"}),
    }
    return out


def qwk_by_group_size(
    frame: pd.DataFrame, criteria: tuple[Criterion, ...]
) -> pd.DataFrame:
    """QWK on absolute scores, the fallback outcome for tie-crippled criteria.

    Pools both sides of each pair, since QWK is a pointwise agreement statistic.
    """
    by_key = {c.key: c for c in criteria}
    rows = []
    for (criterion_key, group_size), chunk in frame.groupby(["criterion", "group_size"]):
        criterion = by_key.get(criterion_key)
        if criterion is None:
            continue
        judge = list(chunk["judge_a"]) + list(chunk["judge_b"])
        human = list(chunk["human_a"]) + list(chunk["human_b"])
        try:
            value = quadratic_weighted_kappa(
                judge, human, criterion.scale_min, criterion.scale_max
            )
        except ValueError:
            continue
        rows.append(
            {
                "criterion": criterion_key,
                "group_size": group_size,
                "qwk": value,
                "n": len(chunk),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------- primary: paired contrast --------


@dataclass
class PairedResult:
    reference_size: int
    comparison_size: int
    n_items: int
    mean_difference: float
    clustered_se: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float

    def as_row(self) -> dict[str, Any]:
        return {
            "reference_size": self.reference_size,
            "comparison_size": self.comparison_size,
            "n_items": self.n_items,
            "mean_difference": round(self.mean_difference, 4),
            "clustered_se": round(self.clustered_se, 4),
            "t": round(self.t_stat, 3),
            "p": self.p_value,
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
        }


def paired_contrast(
    frame: pd.DataFrame, reference_size: int, comparison_size: int
) -> PairedResult | None:
    """Within-item accuracy difference between two group sizes.

    The item is the cluster and the unit of analysis: per-item accuracy is
    averaged across that item's criteria and replicates *first*, then differences
    are taken across items. Pooling raw observations would violate independence
    (Miller: multiple answers to the same question are not independent draws), and
    it is also the step that buys the variance reduction, since item difficulty
    cancels within the difference.
    """
    res = resolved(frame)
    cluster = independence_unit(res)
    per_item = (
        res[res["group_size"].isin([reference_size, comparison_size])]
        .groupby([cluster, "group_size"])["correct_num"]
        .mean()
        .unstack("group_size")
    )
    if reference_size not in per_item or comparison_size not in per_item:
        return None
    paired = per_item[[reference_size, comparison_size]].dropna()
    if len(paired) < 3:
        return None

    diff = (paired[comparison_size] - paired[reference_size]).to_numpy()
    n = len(diff)
    mean = float(diff.mean())
    # Items are independent of one another, so the clustered SE of the mean
    # difference is the usual SE of the per-item differences -- clustering is
    # already absorbed by collapsing to one value per item.
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    if not se or np.isnan(se):
        return None
    t_stat = mean / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    crit = float(stats.t.ppf(0.975, df=n - 1))
    return PairedResult(
        reference_size=reference_size,
        comparison_size=comparison_size,
        n_items=n,
        mean_difference=mean,
        clustered_se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
    )


def paired_ladder(frame: pd.DataFrame, reference_size: int = 1) -> pd.DataFrame:
    """Every group size contrasted against the reference."""
    res = resolved(frame)
    sizes = sorted(int(s) for s in res["group_size"].unique() if s != reference_size)
    rows = []
    for size in sizes:
        result = paired_contrast(frame, reference_size, size)
        if result is not None:
            rows.append(result.as_row())
    frame_out = pd.DataFrame(rows)
    if not frame_out.empty:
        frame_out["p_holm"] = holm(list(frame_out["p"]))
    return frame_out


# --------------------------------------------------- logit, cluster-robust ---


def fit_logit(
    frame: pd.DataFrame, *, formula: str = "correct_num ~ C(group_size) + position"
) -> Any:
    """Logistic regression with cluster-robust SEs, clustered on item."""
    import statsmodels.formula.api as smf

    res = resolved(frame)
    if res.empty:
        raise ValueError("no resolved observations to fit")
    codes = pd.Categorical(res[independence_unit(res)]).codes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.logit(formula, data=res)
        return model.fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": codes}, maxiter=200
        )


def lrt(restricted: Any, full: Any) -> dict[str, Any]:
    """Unclustered likelihood comparison for descriptive shape diagnostics.

    The statistic uses ordinary likelihoods and therefore ignores repeated
    observations within item. It must not select the inferential model or be
    presented as a cluster-valid hypothesis test.
    """
    stat = float(2 * (full.llf - restricted.llf))
    df = int(full.df_model - restricted.df_model)
    interpretation = "descriptive only; ignores item clustering"
    if df <= 0:
        return {
            "lr_stat": stat,
            "df": df,
            "p": float("nan"),
            "interpretation": interpretation,
        }
    return {
        "lr_stat": stat,
        "df": df,
        "p": float(stats.chi2.sf(stat, df)),
        "interpretation": interpretation,
    }


def functional_form(frame: pd.DataFrame) -> dict[str, Any]:
    """Is the group-size effect a cliff or a slope?

    The factor model makes no shape assumption and answers "where does it break";
    the linear model gives one interpretable slope and more power. Report the one
    the data supports.
    """
    linear = fit_logit(frame, formula="correct_num ~ group_size + position")
    factor = fit_logit(frame, formula="correct_num ~ C(group_size) + position")
    test = lrt(linear, factor)
    return {
        "linear_slope": float(linear.params.get("group_size", float("nan"))),
        "linear_slope_se": float(linear.bse.get("group_size", float("nan"))),
        "linear_p": float(linear.pvalues.get("group_size", float("nan"))),
        "lrt_factor_vs_linear": test,
        "prefer": "not selected",
        "linear_aic": float(linear.aic),
        "factor_aic": float(factor.aic),
    }


def position_competition(frame: pd.DataFrame) -> dict[str, Any]:
    """Does `position` absorb the group-size effect?

    The rival hypothesis: accuracy degrades with where a criterion sits in the
    prompt, not with how many share the call. If the group_size slope collapses
    once position enters, the finding changes character entirely -- and becomes
    more actionable, since ordering is free to control.
    """
    without = fit_logit(frame, formula="correct_num ~ group_size")
    with_pos = fit_logit(frame, formula="correct_num ~ group_size + position")
    interaction = fit_logit(
        frame, formula="correct_num ~ group_size * position"
    )
    slope_before = float(without.params.get("group_size", float("nan")))
    slope_after = float(with_pos.params.get("group_size", float("nan")))
    p_before = float(without.pvalues.get("group_size", float("nan")))

    # Attenuation is a ratio to the base slope, so it is meaningless when that
    # slope is itself indistinguishable from zero -- dividing by noise produces
    # impressive-looking percentages out of nothing. Only report it when there is
    # a base effect to attenuate.
    if not np.isfinite(p_before) or p_before > 0.05 or abs(slope_before) < 1e-6:
        attenuation = float("nan")
    else:
        attenuation = (slope_before - slope_after) / slope_before

    return {
        "group_size_p_without_position": p_before,
        "group_size_slope_without_position": slope_before,
        "group_size_slope_with_position": slope_after,
        "attenuation_fraction": attenuation,
        "position_slope": float(with_pos.params.get("position", float("nan"))),
        "position_p": float(with_pos.pvalues.get("position", float("nan"))),
        "interaction_p": float(
            interaction.pvalues.get("group_size:position", float("nan"))
        ),
    }


# ------------------------------------------------ per-criterion interaction --


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * p_values[index]
        running = max(running, min(1.0, value))
        adjusted[index] = running
    return adjusted


def by_criterion_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-criterion group-size slope, Holm-corrected.

    This is the actionable output: if some criteria degrade under packing and
    others do not, the recommendation becomes "bundle the robust ones, isolate the
    fragile ones" rather than a single global rule.
    """
    rows = []
    for criterion, chunk in resolved(frame).groupby("criterion"):
        if chunk["group_size"].nunique() < 2 or chunk["correct_num"].nunique() < 2:
            continue
        try:
            fit = fit_logit(chunk, formula="correct_num ~ group_size + position")
        except Exception as exc:  # separation, singularity, too little data
            rows.append({"criterion": criterion, "n": len(chunk), "error": str(exc)[:80]})
            continue
        rows.append(
            {
                "criterion": criterion,
                "n": len(chunk),
                "slope": float(fit.params.get("group_size", float("nan"))),
                "se": float(fit.bse.get("group_size", float("nan"))),
                "p": float(fit.pvalues.get("group_size", float("nan"))),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and "p" in out and out["p"].notna().any():
        mask = out["p"].notna()
        adjusted = holm(list(out.loc[mask, "p"]))
        out.loc[mask, "p_holm"] = adjusted
    return out


# ------------------------------------------------------- crossed mixed model --


def fit_mixed(frame: pd.DataFrame) -> Any:
    """Crossed random intercepts for item and criterion.

    ``correct ~ group_size + position + (1|item) + (1|criterion)``

    statsmodels' Bayesian mixed GLM is variational, so the point estimates are
    trustworthy but the posterior SDs are approximate. Reported alongside the
    paired contrast and the cluster-robust logit, never instead of them.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    res = resolved(frame)
    if res.empty:
        raise ValueError("no resolved observations to fit")
    cluster = independence_unit(res)
    if res[cluster].nunique() < 2 or res["criterion"].nunique() < 2:
        raise ValueError("crossed model needs >= 2 items and >= 2 criteria")

    vc = {"item": f"0 + C({cluster})", "criterion": "0 + C(criterion)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula(
            "correct_num ~ group_size + position", vc, res
        )
        return model.fit_vb(verbose=False)


# ------------------------------------------- stratified endpoint estimator ---


@dataclass(frozen=True)
class StratifiedResult:
    """Equal-weight average across operation-specific endpoint differences."""

    theta: float
    se: float
    df: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    per_operation: dict[str, tuple[float, int]]

    def as_row(self) -> dict[str, Any]:
        return {
            "theta": round(self.theta, 4),
            "se": round(self.se, 4),
            "df": round(self.df, 1),
            "t": round(self.t_stat, 3),
            "p": self.p_value,
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
        }


def stratified_endpoint(
    frame: pd.DataFrame,
    reference_size: int,
    comparison_size: int,
    *,
    operations: tuple[str, ...],
    min_units: int = 3,
) -> StratifiedResult | None:
    """Equal weight per operation, not per observation.

    Equal quotas define the weights, so the estimator must apply them. An
    unqualified mean over all per-unit differences would re-weight operations by
    however many units each happened to contribute, which is exactly what quotas
    exist to prevent.

        theta = (1/O) * sum_o mean(d_o)
        Var   = (1/O^2) * sum_o s_o^2 / n_o
        df    = Welch-Satterthwaite over the same strata

    Returns None rather than a partial answer when any operation is missing or
    thin: the estimand is an average over all of them, so a subset is a different
    quantity and should not be reported under the same name.
    """
    res = resolved(frame)
    cluster = independence_unit(res)
    if "operation" not in res.columns:
        raise ValueError("frame has no `operation` column")

    per_unit = (
        res[res["group_size"].isin([reference_size, comparison_size])]
        .groupby([cluster, "operation", "group_size"])["correct_num"]
        .mean()
        .unstack("group_size")
    )
    if reference_size not in per_unit or comparison_size not in per_unit:
        return None
    paired = per_unit[[reference_size, comparison_size]].dropna()
    if paired.empty:
        return None
    diffs = (paired[comparison_size] - paired[reference_size]).rename("d")

    means: list[float] = []
    variance_terms: list[float] = []
    df_terms: list[float] = []
    per_operation: dict[str, tuple[float, int]] = {}
    for operation in operations:
        try:
            values = diffs.xs(operation, level="operation").to_numpy()
        except KeyError:
            return None
        n = len(values)
        if n < min_units:
            return None
        mean = float(values.mean())
        per_operation[operation] = (round(mean, 4), n)
        means.append(mean)
        term = float(values.var(ddof=1)) / n if n > 1 else 0.0
        variance_terms.append(term)
        df_terms.append(term**2 / (n - 1) if n > 1 else 0.0)

    count = len(operations)
    theta = float(np.mean(means))
    variance = sum(variance_terms) / count**2
    se = float(np.sqrt(variance))
    if not se or np.isnan(se):
        return None
    denominator = sum(df_terms)
    df = (
        float(sum(variance_terms) ** 2 / denominator)
        if denominator > 0
        else float(count - 1)
    )
    t_stat = theta / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
    crit = float(stats.t.ppf(0.975, df=df))
    return StratifiedResult(
        theta=theta,
        se=se,
        df=df,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=theta - crit * se,
        ci_high=theta + crit * se,
        per_operation=per_operation,
    )
