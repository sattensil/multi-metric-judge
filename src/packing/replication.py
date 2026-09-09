"""Recompute the published claims from retained artifacts, and verify them.

Three activities the repository has to support separately, because they answer
different questions:

- **Review**: read what was done and which claims it supports.
- **Recompute**: regenerate the published numbers from the exact retained calls.
  Audits the analysis path. No API key, no spend.
- **Rerun**: make fresh judge calls under the same protocol. Tests procedural
  robustness. The judge is nondeterministic and the model name is an alias, so
  exact agreement is not expected and is not the goal.

Recomputation is not a weaker form of rerunning; it is a different claim. But
neither audits *construction*: a corruption builder that pairs the wrong two
texts produces call records perfectly consistent with a wrong experiment, and no
amount of recomputation surfaces that. Construction is audited by reading
`corrupt.py`, `prompt.py` and the operation catalogue, which is why those are
part of the reviewable surface rather than only the numbers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Floats are stored to 4 decimal places, so anything at or below this is a
# rounding artefact rather than a disagreement. Counts, ids and hashes are
# compared exactly and never fall under this tolerance.
FLOAT_TOLERANCE = 1e-4

# Keys whose value must match exactly regardless of type.
EXACT_KEYS = frozenset({"calls", "n_items", "successful_calls", "numbers_n_items"})


class ReplicationError(RuntimeError):
    """The artifacts and the expected values disagree, or artifacts are absent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_item_id_sha256(path: Path) -> str:
    """Digest of the item ids in file order.

    Two frames can hold identical examples in different orders. Seeded
    assignment consumes a sorted permutation, so order does not change the
    sample here, but it does change any pipeline that walks the file directly,
    and a replicator has no way to tell the two files apart without this.
    """
    ids = [json.loads(line)["item_id"] for line in Path(path).read_text().splitlines() if line.strip()]
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# --------------------------------------------------------------- the claims ---


def compute_claims(load_run) -> dict[str, dict[str, Any]]:
    """Every published claim value, recomputed from retained call records.

    `load_run` is injected so this is testable against fixtures without a
    `runs/` tree.
    """
    from . import analyze
    from .power import plan_from_pilot

    claims: dict[str, dict[str, Any]] = {}

    # ---- METHODS 8.1: human-stratum pilot --------------------------------
    items, criteria, records, _meta = load_run("pilot")
    frame = analyze.to_frame(analyze.build_outcomes(records, items, criteria))
    resolved = analyze.resolved(frame)
    accuracy = resolved.groupby("group_size")["correct_num"].mean()

    # The published judge tie rate is over human-SCOREABLE pairs, not over all
    # outcomes. METHODS 8.1 says "judge tie rate" without naming the
    # denominator, and the two differ materially: 0.4800 against 0.5123 at
    # K = 1. Pinned here so a replicator computing the obvious thing does not
    # conclude the table is wrong.
    scoreable = frame[~frame["human_tie"]]
    tie = scoreable.groupby("group_size")["judge_tie"].mean()

    contrast = analyze.paired_contrast(frame, 1, 5)
    plan = plan_from_pilot(frame, 1, 5)
    claims["methods_8_1"] = {
        "accuracy_k1": float(accuracy.loc[1]),
        "accuracy_k5": float(accuracy.loc[5]),
        "judge_tie_rate_scoreable_k1": float(tie.loc[1]),
        "judge_tie_rate_scoreable_k5": float(tie.loc[5]),
        "paired_mean_difference": float(contrast.mean_difference),
        "paired_clustered_se": float(contrast.clustered_se),
        "n_items": int(contrast.n_items),
        "per_item_difference_sd": float(plan.diff_sd),
        "mde_at_pilot_n": float(plan.mde_at_pilot_n),
    }

    # ---- METHODS 8.3 and 8.3.2: synthetic sweep --------------------------
    items, criteria, records, meta = load_run("sweep")
    frame = analyze.to_frame(analyze.build_outcomes(records, items, criteria))
    frame = analyze.annotate_synthetic(
        frame,
        items,
        operation_roles=meta.get("operation_roles"),
        detection_classes=meta.get("detection_classes"),
    )
    recovery = analyze.synthetic_recovery(frame).xs("test", level="calibration_role")
    claims["methods_8_3"] = {
        "calls": len(records),
        "semantic_recovery_k1": float(recovery.loc[("semantic", 1), "recovery_rate"]),
        "semantic_recovery_k20": float(recovery.loc[("semantic", 20), "recovery_rate"]),
        "surface_recovery_k1": float(recovery.loc[("surface", 1), "recovery_rate"]),
        "surface_recovery_k20": float(recovery.loc[("surface", 20), "recovery_rate"]),
        "surface_tie_rate_k1": float(recovery.loc[("surface", 1), "tie_rate"]),
        "surface_tie_rate_k20": float(recovery.loc[("surface", 20), "tie_rate"]),
        "surface_discrimination_k20": float(
            recovery.loc[("surface", 20), "discrimination"]
        ),
    }

    primary = analyze.synthetic_recovery_frame(
        frame[frame["calibration_role"] == "test"]
    )
    decomposition: dict[str, Any] = {}
    for detection_class, chunk in primary.groupby("detection_class"):
        competition = analyze.position_competition(chunk)
        for key, value in competition.items():
            value = float(value)
            if value != value:  # NaN: suppressed attenuation, not a number
                continue
            decomposition[f"{detection_class}_{key}"] = value
    claims["methods_8_3_2"] = decomposition

    # ---- METHODS 8.5: fixed-position K = 40 ladder ------------------------
    # Optional, because the ladder artefacts are a separate bundle from the
    # K = 20 release. A reproducer who has only the K = 20 archive should get a
    # pass on what they hold rather than a failure on what they do not.
    try:
        items, criteria, records, meta = load_run("ladder-k40")
    except Exception:
        return claims

    from .eligibility import CORE_OPERATIONS, NUMBERS_OPERATION
    from .ladder_criteria import LADDER_CRITERIA

    frame = analyze.to_frame(analyze.build_outcomes(records, items, LADDER_CRITERIA))
    frame = analyze.annotate_synthetic(frame, items)
    targeted = analyze.synthetic_recovery_frame(frame[frame["is_target"]])
    core = targeted[targeted["operation"].isin(CORE_OPERATIONS)]

    primary_40 = analyze.stratified_endpoint(
        core, 1, 40, operations=tuple(CORE_OPERATIONS)
    )
    ladder: dict[str, Any] = {
        "successful_calls": sum(1 for r in records if not r.get("failure")),
    }
    if primary_40 is not None:
        ladder.update(
            {
                "theta": float(primary_40.theta),
                "se": float(primary_40.se),
                "ci_low": float(primary_40.ci_low),
                "ci_high": float(primary_40.ci_high),
            }
        )
    sentinel = analyze.paired_contrast(
        targeted[targeted["operation"] == NUMBERS_OPERATION], 1, 40
    )
    if sentinel is not None:
        ladder["numbers_mean_difference"] = float(sentinel.mean_difference)
        ladder["numbers_n_items"] = int(sentinel.n_items)
    claims["methods_8_5"] = ladder
    return claims


# ------------------------------------------------------------- comparison ---


@dataclass(frozen=True)
class Comparison:
    section: str
    key: str
    expected: Any
    actual: Any
    ok: bool

    @property
    def line(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        return f"  {mark} {self.section}.{self.key}: expected {self.expected}, got {self.actual}"


def compare(
    actual: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> list[Comparison]:
    """Compare recomputed claims against frozen values.

    Tolerance applies only to floats, and only because numerical libraries move
    the last decimal place between versions. Counts and ids are exact.
    """
    results: list[Comparison] = []
    for section, values in expected.items():
        got_section = actual.get(section, {})
        for key, want in values.items():
            got = got_section.get(key)
            if got is None:
                results.append(Comparison(section, key, want, None, False))
                continue
            if key in EXACT_KEYS or isinstance(want, (bool, int, str)):
                # Counts, ids, flags and hashes are exact. Only floats get a
                # tolerance, and only for last-decimal library drift.
                ok = got == want
            else:
                ok = abs(float(got) - float(want)) <= tolerance
            results.append(Comparison(section, key, want, got, ok))
    return results


def verify_checksums(sums_path: Path, root: Path) -> list[str]:
    """Check every file listed in a SHA256SUMS against `root`.

    Returns a list of human-readable problems, empty when everything matches.
    """
    problems: list[str] = []
    for line in Path(sums_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, _, relative = line.partition("  ")
        target = Path(root) / relative.strip()
        if not target.exists():
            problems.append(f"missing: {relative.strip()}")
            continue
        actual = file_sha256(target)
        if actual != expected:
            problems.append(
                f"digest mismatch: {relative.strip()} "
                f"(expected {expected[:16]}, got {actual[:16]})"
            )
    return problems
