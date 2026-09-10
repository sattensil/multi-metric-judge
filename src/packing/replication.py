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
import subprocess
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


def clean_git_commit(root: Path) -> str:
    """Return HEAD only when it exactly describes the bundled working tree."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise ReplicationError(
            "refusing to bundle a dirty working tree; commit or remove changes first"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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

    # ---- Human-label pilot -------------------------------------------------
    items, criteria, records, _meta = load_run("pilot")
    frame = analyze.to_frame(analyze.build_outcomes(records, items, criteria))
    resolved = analyze.resolved(frame)
    accuracy = resolved.groupby("group_size")["correct_num"].mean()

    # The published judge tie rate is over human-SCOREABLE pairs, not over all
    # outcomes. The denominator is human-scoreable pairs; including human-label
    # ties would change the K = 1 value from 0.4800 to 0.5123.
    scoreable = frame[~frame["human_tie"]]
    tie = scoreable.groupby("group_size")["judge_tie"].mean()

    contrast = analyze.paired_contrast(frame, 1, 5)
    plan = plan_from_pilot(frame, 1, 5)
    claims["human_pilot"] = {
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

    # ---- Synthetic sweep ---------------------------------------------------
    items, criteria, records, meta = load_run("sweep")
    frame = analyze.to_frame(analyze.build_outcomes(records, items, criteria))
    frame = analyze.annotate_synthetic(
        frame,
        items,
        operation_roles=meta.get("operation_roles"),
        detection_classes=meta.get("detection_classes"),
    )
    recovery = analyze.synthetic_recovery(frame).xs("test", level="calibration_role")
    claims["synthetic_sweep"] = {
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
    claims["position_analysis"] = decomposition

    # ---- Fixed-position K = 40 ladder --------------------------------------
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

    # The trend, which is the analysis that actually answers where degradation
    # begins. `position` is deliberately constant in this design, so the shared
    # `functional_form` helper cannot be used: it carries position as a covariate
    # and the fit is singular.
    trend = analyze.fit_logit(core, formula="correct_num ~ group_size")
    ladder["trend_slope"] = float(trend.params["group_size"])
    ladder["trend_se"] = float(trend.bse["group_size"])
    factor = analyze.fit_logit(core, formula="correct_num ~ C(group_size)")
    ladder["trend_lrt_factor_vs_linear_p"] = float(
        analyze.lrt(trend, factor)["p"]
    )
    claims["fixed_position_ladder"] = ladder
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


def coverage_gaps(
    actual: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> list[str]:
    """Claim values that were recomputed but are not pinned to anything.

    `compare` is driven by the expected file. A computed value with no pinned
    counterpart is therefore reported explicitly instead of being silently
    omitted from verification.
    """
    gaps: list[str] = []
    for section, values in sorted(actual.items()):
        pinned = expected.get(section)
        if pinned is None:
            gaps.append(f"{section}: computed but no pinned section ({len(values)} values)")
            continue
        for key in sorted(values):
            if key not in pinned:
                gaps.append(f"{section}.{key}: computed but not pinned")
    return gaps


def verify_release_archive(archive: Path, release_path: Path) -> list[str]:
    """Check an archive against the independently committed release pointer."""
    archive = Path(archive)
    release = json.loads(Path(release_path).read_text(encoding="utf-8"))
    problems: list[str] = []
    if archive.name != release.get("archive"):
        problems.append(
            f"archive name mismatch: expected {release.get('archive')}, "
            f"got {archive.name}"
        )
    if not archive.exists():
        problems.append(f"missing archive: {archive}")
        return problems
    expected_size = release.get("archive_size_bytes")
    if expected_size is not None and archive.stat().st_size != expected_size:
        problems.append(
            f"size mismatch: expected {expected_size}, "
            f"got {archive.stat().st_size}"
        )
    expected_digest = release.get("archive_sha256")
    actual_digest = file_sha256(archive)
    if actual_digest != expected_digest:
        problems.append(
            f"digest mismatch: expected {expected_digest}, got {actual_digest}"
        )
    return problems


def verify_run_artifacts(run_dir: Path) -> list[str]:
    """Return missing files required to recompute a retained claim run."""
    run_dir = Path(run_dir)
    return [
        f"missing {name}"
        for name in ("calls.jsonl", "items.jsonl", "meta.json")
        if not (run_dir / name).is_file()
    ]


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
